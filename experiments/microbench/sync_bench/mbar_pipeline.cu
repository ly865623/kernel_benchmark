// Multi-stage warp-specialized producer/consumer mbarrier-pipeline microbenchmark (B200, sm_100a).
//
// PURPOSE (constraint-compliant): measure the per-iteration *synchronization* cost of a
// warp-specialized producer->consumer pipeline that has the SAME multi-level structure as the
// FlashMLA DSA sparse-prefill k-loop, but with NO matmul / gather / softmax data payload.
// We never run the fused operator. We only exercise the cross-warp mbarrier handshakes
// (mbarrier.arrive / mbarrier.try_wait.parity) that couple the pipeline stages, under the
// kernel's real buffer depth (NUM_K_BUFS=4) and real chain length (multi-level: gather ->
// coord -> KV-transform -> QK -> softmax -> SV).
//
// Model mapping: the DSA per-k-iter "pipeline synchronization residual" is the part of the
// steady-state iteration period NOT explained by single-engine operator throughput. This bench
// isolates the cross-warp handshake-chain contribution to that period. Combined offline with the
// already-measured isolated MMA single-op latency (178/210 cyc) it reconstructs t_sync per iter.
//
// Each pipeline stage = one warp. Stage s waits on full[s] (produced by stage s-1), optionally
// spins a fixed compute-stand-in of LAT_s cycles (to expose per-stage instruction latency without
// doing real work), then arrives full[s+1]. Buffer depth B lets stage s run up to B iterations
// ahead of stage s+1 (ring of B mbarriers per boundary, phase-toggled).
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a mbar_pipeline.cu -o mbar_pipeline.out -lcuda
// Run:   ./mbar_pipeline.out <S_stages> <B_bufs> <ITERS> [lat0 lat1 ...]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define MAX_STAGES 12
#define MAX_BUFS   8

__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(bar)));
}
// spin until the mbarrier flips to `phase`
__device__ __forceinline__ void mbar_wait(uint64_t* bar, int phase) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(bar);
    uint32_t done = 0;
    while (!done) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n\t"
            "selp.u32 %0, 1, 0, p;\n\t"
            "}\n"
            : "=r"(done) : "r"(addr), "r"(phase));
    }
}
__device__ __forceinline__ void spin_cycles(uint32_t n) {
    // expose a fixed instruction latency without memory/tensor work
    uint32_t start = clock();
    while ((clock() - start) < n) { __threadfence_block(); }
}

// full[boundary][buf] : producer of `boundary` (stage `boundary`) signals consumer (stage boundary+1)
// boundaries = S (stage s produces boundary s, consumed by stage s+1; last stage produces boundary
// S-1 back to a sink). empty[boundary][buf] : consumer signals producer the buffer is free.
__global__ void pipeline_kernel(int S, int B, int ITERS, const uint32_t* __restrict__ lat,
                                unsigned long long* __restrict__ out_cycles) {
    extern __shared__ uint64_t smem[];
    // S stages -> Nb = S-1 boundaries. boundary i couples stage i (producer) -> stage i+1 (consumer).
    const int Nb = S - 1;
    uint64_t* full  = smem;            // full[Nb*B]
    uint64_t* empty = smem + Nb * B;   // empty[Nb*B]
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    if (threadIdx.x == 0) {
        for (int i = 0; i < Nb * B; ++i) { mbar_init(&full[i], 1);  }
        for (int i = 0; i < Nb * B; ++i) { mbar_init(&empty[i], 1); }
    }
    __syncthreads();
    // prime: buffers initially free -> arrive each empty once (completes empty phase-0).
    if (threadIdx.x == 0) {
        for (int i = 0; i < Nb * B; ++i) mbar_arrive(&empty[i]);
    }
    __syncthreads();

    if (warp >= S) return;                 // only S warps participate as stages
    const bool is_first = (warp == 0);     // source: only produces boundary `warp`
    const bool is_last  = (warp == S - 1); // sink: only consumes boundary `warp-1`
    const uint32_t my_lat = lat[warp];

    unsigned long long t0 = 0;
    if (lane == 0 && is_last) t0 = clock64();

    for (int it = 0; it < ITERS; ++it) {
        const int buf = it % B;
        const int ph  = (it / B) & 1;      // both full and empty toggle per ring wrap
        if (lane == 0) {
            // consume input from boundary (warp-1)
            if (!is_first) mbar_wait(&full[(warp - 1) * B + buf], ph);
            // produce output on boundary (warp): wait buffer free, work, mark full
            if (!is_last) {
                mbar_wait(&empty[warp * B + buf], ph);
                if (my_lat) spin_cycles(my_lat);
                mbar_arrive(&full[warp * B + buf]);
            } else if (my_lat) {
                spin_cycles(my_lat);       // sink still pays its own stage latency
            }
            // free the input buffer for the previous stage
            if (!is_first) mbar_arrive(&empty[(warp - 1) * B + buf]);
        }
    }
    if (lane == 0 && is_last) {
        unsigned long long t1 = clock64();
        out_cycles[0] = t1 - t0;
    }
}

int main(int argc, char** argv) {
    int S = argc > 1 ? atoi(argv[1]) : 6;
    int B = argc > 2 ? atoi(argv[2]) : 4;
    int ITERS = argc > 3 ? atoi(argv[3]) : 100000;
    std::vector<uint32_t> lat(S, 0);
    for (int i = 0; i < S && (4 + i) < argc; ++i) lat[i] = (uint32_t)atoi(argv[4 + i]);
    if (S > MAX_STAGES || B > MAX_BUFS) { printf("S/B too big\n"); return 1; }

    uint32_t* d_lat; cudaMalloc(&d_lat, S * sizeof(uint32_t));
    cudaMemcpy(d_lat, lat.data(), S * sizeof(uint32_t), cudaMemcpyHostToDevice);
    unsigned long long* d_out; cudaMalloc(&d_out, sizeof(unsigned long long));

    size_t shmem = 2 * (S - 1) * B * sizeof(uint64_t);
    cudaFuncSetAttribute(pipeline_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);

    // warmup
    pipeline_kernel<<<1, S * 32, shmem>>>(S, B, 1000, d_lat, d_out);
    cudaDeviceSynchronize();

    // measure (kernel self-timed via clock64 on last stage)
    pipeline_kernel<<<1, S * 32, shmem>>>(S, B, ITERS, d_lat, d_out);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("CUDA err: %s\n", cudaGetErrorString(err)); return 2; }

    unsigned long long cyc = 0;
    cudaMemcpy(&cyc, d_out, sizeof(cyc), cudaMemcpyDeviceToHost);

    // get clock rate to convert (cudaDeviceProp.clockRate removed in CUDA 13)
    int clk_khz = 0; cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);
    double ghz = clk_khz / 1e6; // attribute is kHz
    double per_iter_cyc = (double)cyc / ITERS;
    double per_iter_ns  = per_iter_cyc / ghz;
    printf("S=%d B=%d ITERS=%d  total_cyc=%llu  per_iter=%.2f cyc = %.2f ns  (clk %.3f GHz)\n",
           S, B, ITERS, cyc, per_iter_cyc, per_iter_ns, ghz);
    printf("CSV,%d,%d,%llu,%.3f,%.3f\n", S, B, cyc, per_iter_cyc, per_iter_ns);
    return 0;
}
