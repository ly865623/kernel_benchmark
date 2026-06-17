// exp2_tput.cu — Faithful on-board microbenchmark of the FlashMLA softmax core op:
// register-resident exp2f (MUFU.EX2 SFU), B200 (sm100).
//
// WHY: the kernel's softmax (phase1.cuh:848,897) is online-softmax using exp2f on the score
// block held in TMEM/registers — there is NO HBM traffic. The off-the-shelf elementwise_throughput
// bench streams 40 GB through HBM and is therefore MEMORY-bound (~240 GOps @ 1.9 TB/s), which does
// NOT measure the SFU EX2 compute rate the kernel actually pays. This bench keeps every value in
// registers (4-way ILP, self-contracting recurrence acc=exp2f(acc)-1 stays bounded near 0) so the
// measured rate is the true register-resident exp2 throughput.
//
// Output: RESULT line with exp2 GOps/s (aggregate) and per-SM Gops/s.

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

__global__ void ex2_kernel(float* __restrict__ sink, int iters) {
    // 4-way ILP to expose SFU throughput (not latency). Self-contracting recurrence
    // acc = exp2f(acc) - 1  keeps values bounded near 0 while forcing a true EX2 each step.
    float a0 = (threadIdx.x & 31) * 1e-3f;
    float a1 = a0 + 0.011f, a2 = a0 + 0.023f, a3 = a0 + 0.037f;
    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        a0 = exp2f(a0) - 1.0f;
        a1 = exp2f(a1) - 1.0f;
        a2 = exp2f(a2) - 1.0f;
        a3 = exp2f(a3) - 1.0f;
    }
    // unconditional write of the (data-dependent) result prevents dead-code elimination
    sink[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3;
}

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;}}while(0)

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    const int sms = p.multiProcessorCount;
    const int blocks = sms * CTAS_PER_SM;

    float* d_sink = nullptr;
    CK(cudaMalloc(&d_sink, (size_t)blocks * THREADS * sizeof(float)));

    ex2_kernel<<<blocks, THREADS>>>(d_sink, 2000);  // warmup
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    ex2_kernel<<<blocks, THREADS>>>(d_sink, ITERS);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));

    const double ops = (double)ITERS * 4 * THREADS * blocks;   // exp2 count
    const double gops = ops / (ms * 1e-3) / 1e9;
    printf("# exp2 SFU register-resident throughput (B200), THREADS=%d CTAS_PER_SM=%d blocks=%d ITERS=%d sms=%d\n",
           THREADS, CTAS_PER_SM, blocks, ITERS, sms);
    printf("RESULT,exp2_gops_total,%.1f,exp2_gops_per_sm,%.3f,ms,%.4f\n",
           gops, gops / sms, ms);
    return 0;
}
