# Stage-centric analytical predictor of FlashMLA DSA sparse-prefill whole-kernel latency on B200 (offline MAPE)

- Run id: `dsa-stagepred-offline-v1`
- Branch: `run/dsa-stagepred-offline-v1`
- Parent branch: `idea/003-idea-98be86a0`
- Worktree: `/home/liuy/DeepScientist/quests/003/.ds/worktrees/idea-idea-98be86a0`
- Idea: `idea-98be86a0`
- Baseline: `flashmla-dsa-b200`
- Baseline variant: `none`
- Dataset scope: `flashmla-dsa-sparse-prefill-b200`
- Verdict: `PASS_WITH_MINOR_RESIDUAL`
- Status: `completed`

## Hypothesis

A closed-form, microbenchmark-calibrated stage-centric latency model (paper Eq.1-8 + a data-dependent DSA sparse-tile extension, K_tiles=ceil(topk/bK)) can predict FlashMLA DSA sparse_attn_fwd whole-kernel latency on B200 with whole-kernel MAPE < 10%, strictly below the naive datasheet roofline error, using only coefficients measured from microbenchmarks or read from datasheets (no free coefficient fitted to the ground-truth latencies).

## Setup

B200 (148 SM, 1.965 GHz, bf16 TC peak 2371 TFLOPS, HBM contiguous 7.1 TB/s, KV-gather effective 6.69 TB/s). Hardware coefficients sourced from microbench-blackwell measurements / datasheet (Table II). Predictor: experiments/main/dsa_predictor.py implementing T_kernel = T_launch + K_tiles*(max(T_compute, T_eff_io) + ...) + T_writeback with DSA top-k-driven tile count and a measured non-contiguous KV-gather bandwidth de-rating. Evaluation: offline closed-form prediction over the 10 baseline ground-truth configs (no new DSA GPU sweep) from baselines/local/flashmla-dsa-b200/json/ground_truth_prefill.json.

## Execution

Read FlashMLA sm100 sparse-prefill source to fix bK / tile shapes / pipeline params; assembled the B200 coefficient table; implemented the closed-form predictor; ran offline prediction over the 10 baseline configs (3 shape families v32 / model1_cfg2 / model1_cfg1, s_kv 8k-64k). All 10 configs classified memory-bound. Output written to experiments/main/predictions.json.

## Results

Whole-kernel MAPE = 6.85% over 10 configs (PASS < 10% target). Worst single-config absolute error = 10.83% (model1_cfg2, marginally above 10%). Naive datasheet-peak roofline MAPE = 19.52%, so the interpretable model beats the roofline bar by ~2.8x. Per-config absolute errors: v32 family 5.0-6.0%; model1_cfg2 family 10.8%; model1_cfg1 family 4.3-4.9%. Every config is memory-bound with predicted T_io dominating T_compute; the per-stage breakdown (T_compute, T_io_kv, T_io_qo, K_tiles) is reported per config, satisfying the interpretability constraint.

## Conclusion

PASS on the headline target (MAPE 6.85% < 10%, strictly below roofline). The model predicts the required baseline metric latency_us per config (predicted-vs-measured table in metric_rows); the aggregate accuracy metrics are reported in metrics_summary. One residual weakness: the model1_cfg2 family sits at 10.8% (slightly over 10%) with a consistent same-sign under-prediction, suggesting a structured residual at topk=1024 that the next analysis step should tighten (candidate: producer-consumer overlap term C2 or KV-gather de-rating refinement). Interpretability constraint held: no free coefficient was fit to the ground-truth latencies.

## Metrics Summary

- `latency_us` = 1609.18
- `whole_kernel_mape_pct` = 6.851
- `worst_config_abs_pct_err` = 10.832
- `roofline_mape_pct` = 19.516
- `n_configs` = 10
- `pass_under_10pct` = True
- `pass_below_roofline` = True

## Baseline Comparison

- `latency_us`: run=1609.18 baseline=1694.51 delta=-85.33 (better)
- `tflops`: run=None baseline=1378.84 delta=n/a (not comparable)
- `mem_bw_tbps`: run=None baseline=6.38 delta=n/a (not comparable)
- `roofline_rel_err`: run=None baseline=0.203 delta=n/a (not comparable)
- `whole_kernel_mape_pct`: run=6.851 baseline=None delta=n/a (not comparable)
- `worst_config_abs_pct_err`: run=10.832 baseline=None delta=n/a (not comparable)
- `roofline_mape_pct`: run=19.516 baseline=None delta=n/a (not comparable)
- `n_configs`: run=10 baseline=None delta=n/a (not comparable)

## Changed Files

- `experiments/main/dsa_predictor.py`
- `experiments/main/predictions.json`

## Evidence Paths

- `experiments/main/predictions.json`
- `experiments/main/dsa_predictor.py`
- `baselines/local/flashmla-dsa-b200/json/ground_truth_prefill.json`

## Notes

- Result completed ~2026-06-11T10:57Z under prior agent instance run-baf4c344 but was left unrecorded for ~15h; this record durably captures it.
- model1_cfg2 family (topk=1024) is the only family above 10% (10.8%) with consistent under-prediction = structured residual to address next.
- All coefficients from microbench/datasheet; no free fit to ground-truth latencies (interpretability constraint upheld).

## Evaluation Summary

- Not recorded.

## Delivery Policy

- Research paper required: `True`
- Recommended next route: `analysis_or_write`
- Reason: Research paper mode is enabled. The run looks promising, so the next route should usually strengthen the evidence and move toward analysis or writing rather than stopping at the algorithm result alone.
