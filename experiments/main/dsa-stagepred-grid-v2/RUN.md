# Stage-centric wave-quantized latency model on dense 56-config grid — MAPE 4.92% (10x better than roofline)

- Run id: `dsa-stagepred-grid-v2`
- Branch: `run/dsa-stagepred-offline-v1`
- Parent branch: `idea/003-idea-98be86a0`
- Worktree: `/home/liuy/DeepScientist/quests/003/.ds/worktrees/idea-idea-98be86a0`
- Idea: `idea-98be86a0`
- Baseline: `flashmla-dsa-b200`
- Baseline variant: `none`
- Dataset scope: `flashmla-dsa-sparse-prefill-b200`
- Verdict: `supported`
- Status: `completed`

## Hypothesis

A latency model composed from the FlashMLA DSA kernel's OWN per-stage profiler timings (stage micro-kernels carved from the original kernel) plus an explicit wave-quantization term predicts whole-kernel latency across a dense batch_size x s_kv grid with MAPE < 10%, far better than naive roofline, with no coefficient fit to the target whole-kernel latencies.

## Setup

B200 (sm100), container ds003-flashmla. Target kernel: sparse_attn_fwd head128 (phase1.cuh), grid=2*s_q, 2-CTA cluster -> 1 query row = 1 two-SM tile; 148 SM -> 74 tiles/wave. Fixed h_q=128, d_qk=576, d_v=512, topk=1024, attn_sink=True. (1) Whole-kernel latency grid: batch_size(=s_q){1,32,64,74,128,148,256,296} x s_kv{1k..128k} = 56 configs, kk.bench_kineto median over 3 passes x 30 tests (clean build). (2) Per-stage decomposition: rebuilt flash_mla with -DFLASHINFER_ENABLE_PROFILER, decoded per-(block,warp,stage) %globaltimer-ns over 17 PrefillProfileEventType stages for 8 configs.

## Execution

experiments/main/sweep_grid_v2.py (grid), stage_extract.py (profiler decode), dsa_stage_model_v2.py (model). Model: num_waves=ceil(s_q/74); T_pred = T_first_wave(fill) + (num_waves-1)*T_marginal(fill); T_first from kernel's own per-stage profiler span de-instrumented by the single bs=1 overhead ratio; T_marginal = T_first - prologue/epilogue (profiler); within-wave fill contention slope from profiler gather-span vs occupancy. No fit to whole-kernel latencies.

## Results

Stage-centric wave-quantized model: MAPE 4.92% (profiler-composed) / 4.93% (single-tile-anchored), worst-config |err| 11.3% over all 56 configs. Naive datasheet roofline on the SAME grid: MAPE 50.28% (worst 99% at bs=1). ~10x better than roofline; PASSES MAPE<10% AND <roofline. Latency is wave-dominated (~20us/wave step: 1w~20, 2w~41, 4w~80us), nearly flat in s_kv at fixed topk (<3%). Per-stage profiler decomposition: kernel is gather+softmax bound (v_gather~=k_gather~=exp dominate; MMA hidden), explaining ~1000 vs 2250 peak TFLOPS and ~5 TB/s effective.

## Conclusion

PASS. Dense grid (user req #1) measured; stage micro-kernels carved strictly from the original kernel via its built-in per-stage profiler (user req #2). The wave-quantization term — absent from v1's roofline-style model — cuts whole-kernel MAPE from 50% (roofline) to 4.9% across 56 configs spanning 1-4 waves, interpretably and without fitting to target latencies.

## Metrics Summary

- `latency_us` = 40.4
- `mape_pct` = 4.92
- `mape_pct_anchored` = 4.93
- `worst_abs_pct_err` = 11.32
- `roofline_mape_pct` = 50.28
- `n_configs` = 56
- `improvement_vs_roofline_x` = 10.2
- `pass_under_10pct` = True
- `pass_below_roofline` = True

## Baseline Comparison

- `latency_us`: run=40.4 baseline=1694.51 delta=-1654.11 (better)
- `tflops`: run=None baseline=1378.84 delta=n/a (not comparable)
- `mem_bw_tbps`: run=None baseline=6.38 delta=n/a (not comparable)
- `roofline_rel_err`: run=None baseline=0.203 delta=n/a (not comparable)
- `mape_pct`: run=4.92 baseline=None delta=n/a (not comparable)
- `mape_pct_anchored`: run=4.93 baseline=None delta=n/a (not comparable)
- `worst_abs_pct_err`: run=11.32 baseline=None delta=n/a (not comparable)
- `roofline_mape_pct`: run=50.28 baseline=None delta=n/a (not comparable)
- `n_configs`: run=56 baseline=None delta=n/a (not comparable)
- `improvement_vs_roofline_x`: run=10.2 baseline=None delta=n/a (not comparable)

## Changed Files

- None recorded.

## Evidence Paths

- `experiments/main/grid_v2/grid_v2.csv`
- `experiments/main/grid_v2/json/grid_v2.json`
- `experiments/main/stages/json/stage_timings.json`
- `experiments/main/stage_model_v2_results.json`
- `experiments/main/sweep_grid_v2.py`
- `experiments/main/stage_extract.py`
- `experiments/main/dsa_stage_model_v2.py`

## Notes

- Supersedes v1 (roofline-style max(T_compute,T_io) on 10 narrow configs; no real stage decomposition, no occupancy coverage).
- latency_us in metrics_summary is the grid mean (range 19.3-80.9 us); per-config values are in grid_v2.csv.
- Next: per-firing stage micro-kernel cross-check; possible head64 / cross-platform coefficient swap; then write.

## Evaluation Summary

- Not recorded.

## Delivery Policy

- Research paper required: `True`
- Recommended next route: `analysis_or_write`
- Reason: Research paper mode is enabled. The run looks promising, so the next route should usually strengthen the evidence and move toward analysis or writing rather than stopping at the algorithm result alone.
