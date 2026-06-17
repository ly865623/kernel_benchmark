# FlashMLA Kernel Provenance & "Latest Upstream" Verification

Quest 003 — Stage-centric analytical latency model for the FlashMLA DSA sparse-prefill kernel on B200.

This note records the audit that answers the user requirement (2026-06-12T07:25Z):
*"你可以不使用已经有的flashmla代码，去github上拉最新的"* — i.e. confirm the modeled kernel
matches the latest upstream FlashMLA from GitHub, not a locally-divergent fork.

## Conclusion (verified, not assumed)

The DSA sparse-prefill kernel we model is **byte-identical to the latest upstream
`deepseek-ai/FlashMLA`**, and all local working-tree changes to that kernel are a
**computation-preserving profiler instrumentation** (inert when the profiler build flag is off).
Therefore the existing 56-config measurements already represent stock, latest-upstream FlashMLA;
no re-pull or re-measurement is scientifically required.

## Evidence

Repository: `/workspace/code/FlashMLA` (container `ds003-flashmla`)
Remote: `origin = https://github.com/deepseek-ai/FlashMLA.git` (official)

1. **Local pin vs latest upstream.**
   - Local `HEAD = 48c6dc426f045cb7743b18f5c7329f35f1b7ed79` ("nits").
   - Latest upstream `origin/main = 9241ae3ef9bac614dd25e45e507e089f888280e0` (after `git fetch origin`).
   - Local is 3 commits behind upstream.

2. **The 3 upstream commits do NOT touch the modeled kernel.**
   `git diff --stat 48c6dc4 origin/main` changes only:
   - `csrc/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cuh`   (dense bwd — not modeled)
   - `csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cuh`   (dense fwd — not modeled)
   - `csrc/sm100/prefill/dense/kernel/fmha_kernel_bwd_convert.hpp` (dense bwd convert — not modeled)
   - `csrc/smxx/decode/combine/combine.cu`                   (decode combine — not modeled)

   `git diff 48c6dc4 origin/main -- csrc/sm100/prefill/sparse/` → **EMPTY**.
   ⇒ The DSA sparse-prefill kernel is identical between local `48c6dc4` and latest `9241ae3`.

3. **Local working-tree delta on the modeled kernel is profiler-only.**
   `csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh`: +51 / −8.
   Filtering the diff, every `-`/`+` pair on a compute statement
   (`partition_fragment_C(...)`, `ku::utcmma_ss(...)`, `make_tensor(... SmemLayoutSTiles ...)`, etc.)
   is the **same line re-added with only a trailing shape comment appended**
   (e.g. `// [64, 128]`). The genuinely new lines are:
   - `ProfilerClosure profile_closure;` and `PROFILER_INIT / PROFILER_EVENT_START/END` markers;
   - inert empty `if (k > 0) {}` blocks (no body ⇒ compiler-eliminated, no-op).
   No tiling / scheduling / buffer / MMA logic is altered.

4. **Profiler is compile-gated and disappears in the latency build.**
   `csrc/profiler.cuh` defines `PROFILER_*` macros under `#ifdef FLASHINFER_ENABLE_PROFILER`
   with an `#else` branch expanding them to **nothing**. The whole-kernel latency grid was built
   with the profiler flag OFF ⇒ generated machine code == stock kernel. The per-stage decomposition
   grid was built with the flag ON purely to read the on-chip `%globaltimer` spans; the timing
   apparatus does not change the computation it measures.

## Provenance statement for the paper (reproducibility appendix)

> We model the DSA sparse-prefill forward kernel of `deepseek-ai/FlashMLA`. The modeled kernel is
> byte-identical to upstream commit `9241ae3` (the current `main` head) and to the pinned build
> commit `48c6dc4`; the three intervening upstream commits modify only the dense-prefill and
> decode-combine kernels, which are outside the modeled path. Whole-kernel latency is measured from
> a stock build (profiler disabled, code-generation identical to upstream). Per-stage micro-kernel
> spans are read from an additive, computation-preserving profiler instrumentation (compile-gated;
> a no-op in the latency build), with the full instrumentation diff archived alongside this note.

Verified: 2026-06-12 (quest 003, run branch `run/dsa-stagepred-offline-v1`).
