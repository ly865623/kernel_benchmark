# Microbench-driven stage costs (on-board, B200) — measurement log

All numbers from standalone microbenchmarks in /home/liuy/code/microbench-blackwell, run on B200,
ncu (no sudo) + kernel self-timing. NOT from FLASHINFER_ENABLE_PROFILER. SM clock 1965 MHz max.

## MMA (tcgen05, bf16->fp32, TS, 2-SM) — umma_throughput / umma_latency
- QK (M128 N128 K16): 37.106 cyc/MMA; 178 cyc single-op latency.
- SV (M128 N256 K16): 64.648 cyc/MMA; 210 cyc single-op latency.
  Per k-iter the kernel does QK over K=576 (36 K-steps) and SV N=512=2xN256 over K=128 (8 K-steps).

### PIPELINE-DEPTH PROVENANCE (corrects an earlier "steady-state" mislabel)  [AUDIT 2026-06-17]
A user re-ran umma_throughput and got QK ~32 cyc/MMA, not 37.106. Both numbers are real; the
difference is PIPELINE DEPTH (back-to-back MMAs in the timed loop). From the suite's own depth
sweep (tput_results_full.csv, BF16/TS/cta2/M128 N128 K16):
    depth 16 -> 41.90 | depth 32 -> 36.97 | depth 64 -> 34.48 | depth 128 -> 33.24 | depth 256 -> 32.65
- 32.65 cyc is the ASYMPTOTIC deep-pipeline peak (depth 256; = tput_results_max.csv). This is what
  `make && ./umma_tput.out` reports and what the user measured.
- 37.106 is the value used by the model; it sits at depth ~32, NOT the asymptote. Calling it
  "steady-state" (original wording) was WRONG and is corrected here.
- WHICH is faithful to the DSA kernel? The kernel runs QK in SHORT bursts: ~18 back-to-back
  M128N128K16 MMAs per k-iter per 2-CTA stream (line below: 18 atoms, K=D_Q/2=288 = 18xK16), which
  then drain against the online-softmax recurrence (QK->softmax->SV serial within a k-iter). At the
  kernel's true burst depth (~16-18) measured throughput is ~41 cyc -> so 37.106 is already at/below
  the kernel's realistic operating point, and the asymptotic 32.65 is UNREACHABLE for this access
  pattern (it would over-credit the tensor core).
- EMPIRICAL CONFIRMATION (dsa_stage_model_piecewise.py, 104-cfg grid): substituting the asymptote
  QK=32.654 makes the small kernel under-predict MORE (signed -4.9% -> -7.9%) and worsens MAPE
  (small 4.92% -> 7.90%; both 3.88% -> 5.49%). The model favors the non-peak rate, consistent with
  the kernel not sustaining asymptotic tensor throughput. Headline stays 3.88% at the current 37.106.
- KNOWN CALIBRATION CAVEAT (honest): the two MMA constants were read at INCONSISTENT depths — QK at
  depth ~32 (37.106) but SV at the depth-256 asymptote (64.648). SV's per-k-iter burst is only ~8
  K-steps, so by the same burst argument SV is also calibrated optimistically; SV is not the binder
  (QK is) so this has small effect, but a single first-principles depth-selection rule applied to
  BOTH stages is not yet derived (future work). Reproduce the sweep:
    cd ~/code/microbench-blackwell/umma_throughput
    make clean && make CTA_GROUP=2 MMA_M=128 MMA_N=128 MMA_K=16 MMA_DEPTH=256 && ./umma_tput.out  # -> 32.65 (peak)
    make clean && make CTA_GROUP=2 MMA_M=128 MMA_N=128 MMA_K=16 MMA_DEPTH=32  && ./umma_tput.out  # -> ~37 (kernel-depth)

## Gather / HBM (tma2d_throughput, ncu dram__bytes_read.sum.per_second)
- ~3.75 TB/s, ~flat vs occupancy (CTAS_PER_SM 1..16), SMEM_WIDTH=128 H=8 stages=4.
  CAVEAT: contiguous-tile proxy; kernel gathers SCATTERED 64x576 bf16 (1152B) rows -> needs a
  faithful scattered-gather microbench (BW is very tile-shape sensitive: 1.10 TB/s narrow tiles).

## TODO (next): faithful scattered gather; softmax exp(128x64) compute cost; q_load/epilogue TMA;
## then compose dsa_stage_model_microbench.py and report MAPE vs grid_v2.json.

## Faithful scattered GATHER (TMA tile::gather4) — on-board, B200  [LANDED]
Source: experiments/microbench/gather4_bench/gather4_tput.cu (reuses FlashMLA's OWN
ku::tma_gather4 + ku::make_tensor_map). The real kernel gathers KV-latent rows with
cp.async.bulk.tensor.2d...tile::gather4 (phase1.cuh:438), box{64,1} bf16 SWIZZLE_128B,
descriptor {D_QK=512, s_kv} stride 576*2B (phase1.cuh:1011) — NOT contiguous tma2d, NOT
cp.async/LDGSTS. Per-CTA k-iter = 64 scattered token rows x 256 bf16 (one CTA half) = 32 KB;
two CTAs (cta_group::2) cover the full 512.

RESULT (PIPE=4 = kernel NUM_K_BUFS=4; compute-sanitizer memcheck: 0 errors):
- Effective scattered-gather4 BW = ~2.0 TB/s (1.96-2.04), and FLAT vs:
    * working set s_kv 4096(4.5MiB,L2-resident) -> 1048576(1.15GiB,HBM) : no change
    * occupancy ctas/SM 1->8 (148->1184 blocks)                          : no change (global ceiling)
- => per 64-token KV block gather time = 32768 B / 2.02e12 B/s = 16.2 ns (one CTA half; both
     CTAs run concurrently, so a full 2-CTA tile's 64 KB also completes in ~16.2 ns).
- PIPE=6 -> 1.55 TB/s (lower; larger smem hurts) => PIPE=4 is both faithful and peak.
Files: experiments/microbench/results/gather4_scatter_bw.csv

KEY CORRECTIONS vs prior proxies:
1. Contiguous tma2d proxy = 3.75 TB/s OVER-estimates gather BW by ~1.9x. The true scattered
   gather4 ceiling is ~2.0 TB/s (576-elem row stride -> partial DRAM burst efficiency).
2. Gather BW is ~independent of occupancy AND of L2-residency -> it is instruction/path
   saturated at ~2.0 TB/s, not classic-HBM-bandwidth bound. This kills v2's profiler-fitted
   "contention grows with occupancy" term.
FIDELITY CAVEAT: measured with cta_group::1 (single-CTA) tma_gather4; real kernel uses
cta_group::2 (2-SM cooperative). Same instruction/box/descriptor/bytes; per-CTA 256-col half
is the per-SM stream, so the 2.0 TB/s per-CTA figure is the model-relevant quantity.

## Softmax core op: register-resident exp2f (MUFU.EX2 SFU) — on-board, B200  [LANDED]
Source: experiments/microbench/exp2_bench/exp2_tput.cu. Kernel softmax (phase1.cuh:848,897) is
online-softmax exp2f on the score block in TMEM/registers — NO HBM traffic.
WHY a new bench: the suite's elementwise_throughput streams 40 GB through HBM -> MEMORY-bound
(240 GOps @ 1.92 TB/s), which does NOT measure the SFU EX2 compute rate the kernel pays. This
bench keeps all values in registers (4-way ILP, self-contracting acc=exp2f(acc)-1).
RESULT (occupancy sweep): exp2 SFU saturates at ~3.67 TOps/s aggregate = ~24.8 GOps/s/SM
(~12.6 exp2/cycle/SM @ 1965 MHz), reached by 4 CTAs/SM. Files: results/exp2_sfu.csv.
MODEL USE: per-k-iter softmax exp2 count = (H_Q/2)*B_TOPK = 64*64 = 4096 exp2 per CTA over the
[128xB_TOPK] score block. Whether softmax or gather BINDS the overlapped pipeline is decided in
the recomposition vs grid_v2.json — note FlashInfer's MLA doc independently calls the MLA kernel
"softmax-bottlenecked", so a softmax-bound finding would be consistent & defensible.

## STILL TODO before composition: q_load (TMA Q 128x512 prologue) + epilogue O-store (128x512),
## both amortized over k_tiles per tile. Then dsa_stage_model_microbench.py -> MAPE vs grid_v2.json.

## COMPOSITION (dsa_stage_model_microbench.py) — on-board microbench model vs grid_v2  [LANDED]
Per-k-iter op costs (B200 @1965MHz, from the 4 on-board microbenches above; atom counts from
config.h/phase1.cuh, 2x1SM per-CTA stream):
- QK tensor = 18 atoms x 37.106 cyc = 339.9 ns  (TiledMMA_P: M128 N=B_TOPK*2=128, K=D_Q/2=288)
- SV tensor =  4 atoms x 64.648 cyc = 131.6 ns  (TiledMMA_O: per-CTA N=D_V/2=256, K=B_TOPK=64)
- tensor(QK+SV, same cores -> serial) = 471.5 ns ; gather = 16.2 ns ; softmax exp2 = 165.2 ns
- overlapped t_step = max = 471.5 ns -> BINDER = tensor (QK MMA), NOT gather/softmax.
- T_pro(q_load 128x576 bf16 @3.75TB/s)=39.3 ns ; T_epi(O 128x512)=35.0 ns (once/tile, amortized).
- k_tiles = ceil(topk/B_TOPK), B_TOPK=64 -> 16 for topk=1024 (grid's k_tiles field uses /128).

KEY FINDING: the 4 isolated core ops account for only **38.9%** of the real per-iter time
(measured bs=1 wave 19.6 us vs microbench core wave 7.6 us). Even fully serialized (652.9 ns/iter)
they are ~53%. The remaining ~749 ns/k-iter is the warp-specialized pipeline's sync/dependency
overhead (named-barrier waits, TMEM<->reg, rescale, pipeline fill) that isolated microbenchmarks
fundamentally cannot measure. This REFINES the profiler-era "gather+softmax bound" claim: the
overlapped binder is tensor (QK), and op-sum alone is necessary-but-insufficient.

MAPE vs grid_v2.json (56 configs, topk=1024 fixed):
- micro_overlap (zero-fit, ops only, overlapped) : 62.44%  (worse than roofline -> ops insufficient)
- micro_serial  (zero-fit, ops only, serialized) : 48.13%
- anchored (microbench op-decomp + ONE on-board bs=1 sync-residual anchor + wave-quant): 3.75% (worst 9.16%)
- roofline reference                              : 50.28%
The anchored microbench model matches the profiler-based v2 (4.92%) while being FULLY profiler-free.
CAVEAT: this grid fixes topk=1024, so it validates wave-quantization + s_kv-independence + absolute
level (1 anchor) but does NOT independently exercise the microbench op-cost SCALING across topk
(topk=2048 dispatches a DIFFERENT kernel, topk<=1280 gate). Files: dsa_stage_model_microbench.py,
results/stage_model_microbench_results.json.
