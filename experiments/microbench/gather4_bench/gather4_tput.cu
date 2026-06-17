// gather4_tput.cu — Faithful on-board microbenchmark of the FlashMLA DSA sparse-prefill
// KV-latent GATHER stage, on B200 (sm100).
//
// WHY: the whole-kernel latency model is gather-bound. The real kernel
// (csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/phase1.cuh:438) gathers the
// selected KV-latent rows with the TMA `tile::gather4` instruction
// (cp.async.bulk.tensor.2d...tile::gather4), NOT a contiguous tma2d tile load and NOT
// cp.async/LDGSTS. Per k-iter it fetches B_TOPK=64 scattered token rows x D_QK=512 bf16
// = 64 KiB, using int4 row indices and the descriptor built by ku::make_tensor_map with
// box {64,1}, bf16, SWIZZLE_128B, L2_256B (phase1.cuh:1011).
//
// This microbench reuses FlashMLA's OWN gather primitive (ku::tma_gather4) and OWN
// descriptor builder (ku::make_tensor_map), issuing the kernel's exact scattered pattern
// over a software pipeline (PIPE deep, matching the kernel's NUM_K_BUFS=4), and times it
// with CUDA events. It measures the genuine scattered-gather4 effective bandwidth as a
// function of working-set size s_kv (L2->HBM regime) and occupancy (CTAs/SM).
//
// Fidelity note: the real kernel issues the 2-SM-cooperative variant
// (tma_gather4_cta_group_2, cta_group::2) splitting the column range across a CTA pair.
// Here we issue the single-CTA cta_group::1 variant (ku::tma_gather4): same gather4
// instruction, same box, same scattered descriptor, same bytes moved through L2/HBM, but
// no 2-SM cluster/peer-mbarrier machinery. The bytes/s through the memory system are the
// faithful quantity the latency model consumes.
//
// Build (see build_and_run.sh). Output: a RESULT,... CSV line per (s_kv, ctas_per_sm).

#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>

#include <cute/tensor.hpp>
#include "kerutils/host/host.h"               // ku::make_tensor_map
#include "kerutils/device/sm100/intrinsics.cuh" // ku::tma_gather4, transac_bar_t

namespace ku = kerutils;
using bf16 = __nv_bfloat16;

#ifndef PIPE
#define PIPE 4                 // matches kernel NUM_K_BUFS=4 (quad-buffered)
#endif
#ifndef B_TOPK
#define B_TOPK 64              // KV block per k-iter (config.h B_TOPK=64)
#endif
#ifndef D_QK
#define D_QK 512               // full latent column dim of the KV tensor (config.h D_QK=512)
#endif
#ifndef D_GATHER
#define D_GATHER 256           // cols gathered PER BLOCK = one CTA's half (kernel: local_col<(D_K/64)/2=4).
#endif                         //   Two CTAs cover the full 512; per-SM stream is one CTA's 256-col half.
#ifndef KV_STRIDE
#define KV_STRIDE 576          // per-token storage stride in the latent tensor (TMA_K_STRIDE)
#endif
#ifndef ITERS
#define ITERS 4096             // k-iters (KV tiles) gathered per block; enough for steady state
#endif
#ifndef INDEX_TILES
#define INDEX_TILES 512        // distinct random index-tiles, reused cyclically
#endif

constexpr int COL_TILE   = 64;                 // gather4 box columns
constexpr int N_COLTILE  = D_GATHER / COL_TILE; // 4 column tiles per block (one CTA half)
constexpr int N_ROWGRP   = B_TOPK / 4;         // 16 row-groups (gather4 fetches 4 rows)
constexpr int TILE_ELEMS = B_TOPK * D_GATHER;  // 64*256 = 16384 bf16 (one CTA's K-buffer)
constexpr int TILE_BYTES = TILE_ELEMS * 2;     // 32768 bytes gathered per k-iter PER BLOCK

// ---- raw mbarrier ops on a uint64 cell in shared memory (version-stable PTX) ----
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint64_t* b) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(smem_addr(b)));
}
__device__ __forceinline__ void mbar_expect(uint64_t* b, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(smem_addr(b)), "r"(bytes));
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_%=:\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "  @p bra DONE_%=;\n"
        "  bra WAIT_%=;\n"
        "DONE_%=:\n"
        "}\n"
        :: "r"(smem_addr(b)), "r"(phase));
}

extern "C" __global__ void gather4_kernel(const __grid_constant__ CUtensorMap desc,
                                          const int* __restrict__ g_indices,
                                          int n_index_tiles) {
    extern __shared__ char smem_raw[];
    bf16*     smem = reinterpret_cast<bf16*>(smem_raw);          // PIPE * TILE_ELEMS bf16
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem + (size_t)PIPE * TILE_ELEMS);

    const bool elect = (threadIdx.x == 0);
    if (elect) {
        for (int i = 0; i < PIPE; ++i) mbar_init(&bars[i]);
    }
    __syncthreads();

    const int64_t cache_hint = 0;  // default L2 policy descriptor

    auto issue = [&](int tile) {
        const int slot = tile % PIPE;
        bf16* dst = smem + (size_t)slot * TILE_ELEMS;
        const int* idxbase = g_indices + (size_t)(tile % n_index_tiles) * B_TOPK;
        mbar_expect(&bars[slot], TILE_BYTES);
        #pragma unroll
        for (int rg = 0; rg < N_ROWGRP; ++rg) {
            int4 rows = *reinterpret_cast<const int4*>(idxbase + rg * 4);
            #pragma unroll
            for (int ct = 0; ct < N_COLTILE; ++ct) {
                bf16* sp = dst + (size_t)ct * (COL_TILE * B_TOPK) + (size_t)(rg * 4) * COL_TILE;
                ku::tma_gather4(&desc,
                                *reinterpret_cast<ku::transac_bar_t*>(&bars[slot]),
                                sp, ct * COL_TILE, rows, cache_hint);
            }
        }
    };

    if (elect) {
        const int prologue = PIPE < ITERS ? PIPE : ITERS;
        for (int t = 0; t < prologue; ++t) issue(t);
        for (int t = 0; t < ITERS; ++t) {
            const int slot = t % PIPE;
            const uint32_t phase = (uint32_t)((t / PIPE) & 1);
            mbar_wait(&bars[slot], phase);
            const int nxt = t + PIPE;
            if (nxt < ITERS) issue(nxt);
        }
    }
    __syncthreads();
}

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1);} } while(0)

int main(int argc, char** argv) {
    // sweep set: working-set s_kv (tokens) and occupancy (CTAs/SM)
    std::vector<int> s_kv_list   = {4096, 16384, 65536, 262144, 1048576};
    std::vector<int> ctas_list   = {1, 2, 4, 8};
    if (argc >= 2) s_kv_list = { atoi(argv[1]) };
    if (argc >= 3) ctas_list = { atoi(argv[2]) };

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int num_sms = prop.multiProcessorCount;

    const size_t smem_bytes = (size_t)PIPE * TILE_ELEMS * sizeof(bf16) + (size_t)PIPE * sizeof(uint64_t);
    CUDA_CHECK(cudaFuncSetAttribute(gather4_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

    // host RNG indices (scattered token rows), max s_kv
    const int max_s_kv = s_kv_list.back();
    std::mt19937 rng(12345);
    std::vector<int> h_idx((size_t)INDEX_TILES * B_TOPK);

    printf("# gather4 faithful microbench (B200 sm100): box{%d,1} bf16 SWIZZLE_128B, "
           "PIPE=%d B_TOPK=%d D_QK=%d KV_STRIDE=%d ITERS=%d  num_sms=%d  smem/block=%zuKiB\n",
           COL_TILE, PIPE, B_TOPK, D_QK, KV_STRIDE, ITERS, num_sms, smem_bytes >> 10);
    printf("RESULT_HEADER,s_kv,ctas_per_sm,blocks,kv_MiB,gather_tbps,ms,bytes_per_block_MiB\n");

    for (int s_kv : s_kv_list) {
        // allocate KV latent [s_kv, KV_STRIDE] bf16
        const size_t kv_elems = (size_t)s_kv * KV_STRIDE;
        bf16* d_kv = nullptr;
        CUDA_CHECK(cudaMalloc(&d_kv, kv_elems * sizeof(bf16)));
        CUDA_CHECK(cudaMemset(d_kv, 0, kv_elems * sizeof(bf16)));

        // descriptor: EXACTLY the kernel's prefill tensor_map_kv (phase1.cuh:1011)
        // make_tensor_map({D_QK, s_kv}, {stride_bytes}, {64,1}, ptr, BF16, SWIZZLE_128B, L2_256B)
        CUtensorMap desc = ku::make_tensor_map(
            { (uint64_t)D_QK, (uint64_t)s_kv },
            { (uint64_t)KV_STRIDE * sizeof(bf16) },
            { (uint32_t)COL_TILE, 1u },
            d_kv,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        );

        // random scattered indices in [0, s_kv)
        std::uniform_int_distribution<int> dist(0, s_kv - 1);
        for (auto& v : h_idx) v = dist(rng);
        int* d_idx = nullptr;
        CUDA_CHECK(cudaMalloc(&d_idx, h_idx.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_idx, h_idx.data(), h_idx.size() * sizeof(int), cudaMemcpyHostToDevice));

        for (int ctas : ctas_list) {
            const int blocks = num_sms * ctas;
            // warmup
            gather4_kernel<<<blocks, 128, smem_bytes>>>(desc, d_idx, INDEX_TILES);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            cudaEvent_t a, b;
            CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
            const int REP = 3;
            CUDA_CHECK(cudaEventRecord(a));
            for (int r = 0; r < REP; ++r)
                gather4_kernel<<<blocks, 128, smem_bytes>>>(desc, d_idx, INDEX_TILES);
            CUDA_CHECK(cudaEventRecord(b));
            CUDA_CHECK(cudaEventSynchronize(b));
            float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
            ms /= REP;

            const double bytes_per_block = (double)ITERS * TILE_BYTES;
            const double total_bytes = bytes_per_block * blocks;
            const double tbps = total_bytes / (ms * 1e-3) / 1e12;
            printf("RESULT,%d,%d,%d,%.1f,%.4f,%.4f,%.1f\n",
                   s_kv, ctas, blocks, (double)kv_elems * 2 / (1<<20),
                   tbps, ms, bytes_per_block / (1<<20));
            fflush(stdout);
            CUDA_CHECK(cudaEventDestroy(a)); CUDA_CHECK(cudaEventDestroy(b));
        }
        CUDA_CHECK(cudaFree(d_idx));
        CUDA_CHECK(cudaFree(d_kv));
    }
    return 0;
}
