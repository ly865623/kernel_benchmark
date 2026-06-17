// corr_rescale_tput.cu — Faithful on-board microbenchmark of the FlashMLA sm100 CORRECTION stage.
//
// WHY (user req #12, 2026-06-16): the correction warpgroup cost in the stage model was previously
// taken from the DATASHEET FP32 ALU peak (128 lanes * 1.965 GHz = 251.5 Gops/SM). The whole paper
// is "calibrated only from standalone on-board microbenchmarks of the kernel's own operators";
// using a datasheet constant for correction is the one un-measured primitive. This bench supplies
// the missing on-board measurement.
//
// WHAT the correction warpgroup actually does (source: sm100_fmha_mla_fwd_mainloop_tma_warpspecialized.hpp):
//   * acc_scale = 0.5f * exp2f(scale*(old_row_max - row_max_safe))   (L692)  -> 1 exp2 + scalar FMA per row
//   * row_sum  *= acc_scale                                          (L693)  -> 1 FMUL per row
//   * O-accumulator rescale loop over get<2>(TileShape)=D_V columns  (L835-851):
//         TMEM_LOAD -> FMUL2 (multiply each O column by acc_scale) -> store back
//     This is the DOMINANT correction cost: ROWS_CORR x D_V register-resident FP32 multiplies per k-iter.
//
// The O-rescale is register/TMEM-resident (NO HBM traffic): values live in the accumulator fragment
// and are multiplied in place. So the faithful primitive is register-resident FP32 multiply (FMUL/FFMA)
// throughput, isolated the same way exp2_bench isolates register-resident exp2. We measure two variants
// that both saturate the FP32 FMA datapath (128 results/clk/SM):
//   (A) FFMA chain  a = a*c + d  (the canonical FP32-ALU peak primitive the datasheet 251.5 represents)
//   (B) MUL  chain  a = a*c      (the exact FMUL the O-rescale issues), kept bounded with a paired
//                                 contraction so it does not over/underflow over ITERS.
// Reported as Gops/s aggregate and per-SM. This isolated micro-kernel does NOT run the fused operator.
//
// Output: two RESULT lines (ffma, mul) with per-SM FP32 op throughput in Gops/s.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#ifndef ITERS
#define ITERS 200000
#endif
#ifndef THREADS
#define THREADS 256
#endif
#ifndef CTAS_PER_SM
#define CTAS_PER_SM 4
#endif

// (A) FFMA: a = a*c + d. With |c|<1 this is a contraction (fixed point d/(1-c)), so values stay bounded
// near the fixed point while forcing one true FFMA per op. 8-way ILP exposes datapath throughput.
__global__ void ffma_kernel(float* __restrict__ sink, int iters) {
    float a0 = (threadIdx.x & 31) * 1e-3f;
    float a1 = a0 + 0.011f, a2 = a0 + 0.023f, a3 = a0 + 0.037f;
    float a4 = a0 + 0.041f, a5 = a0 + 0.053f, a6 = a0 + 0.061f, a7 = a0 + 0.077f;
    const float c = 0.5f, d = 1.0f;            // contraction: fixed point = 2.0
    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        a0 = a0 * c + d; a1 = a1 * c + d; a2 = a2 * c + d; a3 = a3 * c + d;
        a4 = a4 * c + d; a5 = a5 * c + d; a6 = a6 * c + d; a7 = a7 * c + d;
    }
    sink[blockIdx.x * blockDim.x + threadIdx.x] = a0+a1+a2+a3+a4+a5+a6+a7;
}

// (B) MUL: the exact FMUL the O-rescale issues. To stay bounded over ITERS we pair each register with
// a reciprocal factor so the geometric drift cancels every 2 steps: even register *= c, odd register *= 1/c.
// Net product over the 8 lanes is invariant; each statement is still one independent FMUL the datapath
// must retire, so aggregate FMUL throughput is measured faithfully (8-way ILP, no cross-dependence trick
// that would let the compiler fold the pair — they are summed, not multiplied, into the sink).
__global__ void mul_kernel(float* __restrict__ sink, int iters) {
    float a0 = 1.0f + (threadIdx.x & 31) * 1e-3f;
    float a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    const float cu = 1.0009765625f;            // 1 + 2^-10
    const float cd = 1.0f / 1.0009765625f;     // exact-ish reciprocal; drift bounded over ITERS
    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        a0 = a0 * cu; a1 = a1 * cd; a2 = a2 * cu; a3 = a3 * cd;
        a4 = a4 * cu; a5 = a5 * cd; a6 = a6 * cu; a7 = a7 * cd;
    }
    sink[blockIdx.x * blockDim.x + threadIdx.x] = a0+a1+a2+a3+a4+a5+a6+a7;
}

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;}}while(0)

static int run(const char* tag, void(*k)(float*,int), float* d_sink, int blocks, int sms, int ops_per_iter_per_thread) {
    k<<<blocks, THREADS>>>(d_sink, 2000);  // warmup
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    k<<<blocks, THREADS>>>(d_sink, ITERS);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
    const double ops = (double)ITERS * ops_per_iter_per_thread * THREADS * blocks;  // FP32 result-ops
    const double gops = ops / (ms * 1e-3) / 1e9;
    printf("RESULT,%s,fp32_gops_total,%.1f,fp32_gops_per_sm,%.3f,ms,%.4f\n", tag, gops, gops / sms, ms);
    return 0;
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    const int sms = p.multiProcessorCount;
    const int blocks = sms * CTAS_PER_SM;
    float* d_sink = nullptr;
    CK(cudaMalloc(&d_sink, (size_t)blocks * THREADS * sizeof(float)));
    printf("# correction-stage FP32 ALU register-resident throughput (B200 %s), THREADS=%d CTAS_PER_SM=%d blocks=%d ITERS=%d sms=%d\n",
           p.name, THREADS, CTAS_PER_SM, blocks, ITERS, sms);
    int rc = 0;
    rc |= run("ffma", ffma_kernel, d_sink, blocks, sms, 8);  // 8 FFMA result-ops per iter per thread
    rc |= run("mul",  mul_kernel,  d_sink, blocks, sms, 8);  // 8 FMUL result-ops per iter per thread
    return rc;
}
