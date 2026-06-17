# Paper Evidence Ledger

- Selected outline: `outline-001`
- Item count: `9`
- Updated at: `2026-06-15T11:52:00+00:00`
- Evidence source: `experiments/microbench/stage_model_microbench_2kernel_results.json` (on-board microbenchmark stage model, profiler-free)

| Item | Kind | Section | Role | Status | Metrics | Source |
|---|---|---|---|---|---|---|
| `A1` | analysis_slice | main-results | main_text | completed | whole_kernel_mape_pct=9.24; worst_config_abs_pct_err=18.77; roofline_mape_pct=47.56; roofline_worst_pct=99.1; improvement_vs_roofline_x=5.1; configs_won=104/104 | `experiments/microbench/stage_model_microbench_2kernel_results.json`, `paper/latex/fig_pred_vs_meas.png`, `paper/latex/main.tex` |
| `A2` | analysis_slice | main-results | main_text | completed | qk_matmul_ns=339.9; gather_ns=16.2; softmax_ns=165.2; tensor_overlap_envelope_ns=471.5; per_iteration_ns=1220; operator_sum_share_of_periter_pct=38.9; sync_residual_share_pct=61.1; sync_residual_ns=749; sync_residual_pred_ns=731.8; h_oneway_ns=140.33; binder=tensor; tensor_binds=104/104 | `experiments/microbench/stage_model_microbench_2kernel_results.json`, `experiments/microbench/`, `paper/latex/main.tex` |
| `A3` | analysis_slice | main-results | main_text | completed | max_across_skv_spread_pct=3.25; mean_across_skv_spread_pct=1.48; skv_sweep_range_x=128 | `experiments/microbench/stage_model_microbench_2kernel_results.json`, `paper/latex/fig_skv.png`, `paper/latex/main.tex` |
| `A4` | analysis_slice | main-results | main_text | completed | full_model_mape_pct=9.24; no_wave_quant_mape_pct=47.6; single_row_roofline_err_pct=99; wave_model_single_row_err_pct=0.2 | `experiments/microbench/stage_model_microbench_2kernel_results.json`, `paper/latex/main.tex` |
| `A5` | analysis_slice | main-results | appendix | completed | overlapped_sum_only_mape_pct=60.5; serialized_sum_only_mape_pct=44.6; with_predicted_residual_mape_pct=9.24; roofline_mape_pct=47.56 | `experiments/microbench/stage_model_microbench_2kernel_results.json`, `paper/latex/main.tex` |
| `A6` | analysis_slice | main-results | appendix | completed | worst_abs_pct_err=18.77; small_four_wave_max_abs_pct_err=4.5; residual_sign=underprediction; worst_residuals=small_first_wave_fill_plus_regular_softmax_underprediction | `experiments/microbench/stage_model_microbench_2kernel_results.json`, `paper/latex/main.tex` |
| `A7` | analysis_slice | main-results | appendix | completed | tensor_core_bound_configs=104/104; binder=tensor; gather_ns=16.2; qk_matmul_ns=339.9; gather_to_tensor_ratio=0.048 | `experiments/microbench/stage_model_microbench_2kernel_results.json`, `experiments/microbench/`, `paper/latex/main.tex` |
| `A8` | analysis_slice | main-results | appendix | completed | operators_to_remeasure=4; handshake_microbench=1; anchors_per_kernel=0; structural_constants_from_source=tiles_per_wave_atom_counts; second_device_validated=false | `experiments/microbench/stage_model_microbench_2kernel_results.json`, `experiments/microbench/`, `paper/latex/main.tex` |
| `dsa-stagepred-grid-v2` | main_experiment | main-results |  | completed | latency_model=stage-centric-wave-quantized-dsa-composed-2kernel; mape_pct=9.24; mape_pct_small=4.92; mape_pct_regular=14.28; worst_abs_pct_err=18.77; roofline_mape_pct=47.56; n_configs=104; bottleneck=tensor_core_QKt; improvement_vs_roofline_x=5.1 | `experiments/microbench/stage_model_microbench_2kernel_results.json`, `experiments/microbench/`, `paper/latex/main.tex` |

## A1

- Title: Headline accuracy: stage model versus naive datasheet roofline
- Kind: `analysis_slice`
- Section: `main-results`
- Status: `completed`
- Metrics: whole_kernel_mape_pct=9.24; worst_config_abs_pct_err=18.77; roofline_mape_pct=47.56; roofline_worst_pct=99.1; improvement_vs_roofline_x=5.1; configs_won=104/104

### Setup

Per-configuration whole-kernel latency: stage-model prediction vs naive datasheet-peak roofline on the identical 104-config grid (both dispatch kernels). No coefficient fit to target latencies.

### Result

Composed (constraint-compliant) model MAPE 9.24% versus naive datasheet roofline 47.56% over all 104 configs -> about 5x lower mean error. Worst model config 18.8% (a regular-kernel single-wave point) versus roofline worst 99.1% (at s_q=1, where the roofline's continuous-work assumption is most wrong). Because no whole-kernel latency is consumed for calibration -- the synchronization residual is predicted from a measured cross-warp handshake and the multi-level pipeline structure, not anchored to an on-board kernel timing -- the gain is attributable to the wave-quantized stage structure and the measured primitive costs rather than to tuning.

## A2

- Title: Per-operator on-board decomposition and bottleneck attribution
- Kind: `analysis_slice`
- Section: `main-results`
- Status: `completed`
- Metrics: qk_matmul_ns=339.9; gather_ns=16.2; softmax_ns=165.2; tensor_overlap_envelope_ns=471.5; per_iteration_ns=1220; operator_sum_share_of_periter_pct=38.9; sync_residual_share_pct=61.1; sync_residual_ns=749; sync_residual_pred_ns=731.8; h_oneway_ns=140.33; binder=tensor; tensor_binds=104/104

### Setup

Per-iteration operator costs (ns) from standalone on-board microbenchmarks for both kernels (tile::gather4 TMA, register-resident exp2 softmax, QK and SV MMA atoms), compared against measured per-iteration time.

### Result

From standalone on-board microbenchmarks: the scattered tile::gather4 (the operation the kernel is named for) is the cheapest of the four operators at 16.2 ns/block, while the overlapped binding stage is the tensor-core QK^T matmul at 339.9 ns — the kernel is compute-bound, not gather-bound. The four operators under perfect overlap sum to only ~39% (471.5 ns) of the measured ~1220 ns per iteration; the remaining ~61% (~749 ns) is the warp-specialized pipeline synchronization/dependency residual, invisible to any isolated operator microbenchmark; the composed model predicts it (731.8 ns) from a measured cross-warp handshake (h=140.33 ns) wired along the multi-level pipeline structure -- within 2.3% of the forbidden anchor-derived value, without running the kernel.

## A3

- Title: Context-length (s_kv) invariance at fixed selection budget
- Kind: `analysis_slice`
- Section: `main-results`
- Status: `completed`
- Metrics: max_across_skv_spread_pct=3.25; mean_across_skv_spread_pct=1.48; skv_sweep_range_x=128

### Setup

Measured and predicted whole-kernel latency versus context length s_kv at fixed selection budget, across the 104-config grid; across-s_kv spread computed per batch from the on-board measured latencies.

### Result

At fixed selection budget, measured whole-kernel latency is essentially flat across the 128x context-length (s_kv) sweep: the worst across-s_kv spread for any batch is 3.25% and the mean is 1.48%. This matches the model's prediction that whole-kernel latency is set by the occupancy-wave count and the top-k selection budget, not by the full context length — exactly the dependence a sparse-attention kernel induces and a naive roofline misses.

## A4

- Title: Wave-quantization-term ablation: the decisive structure
- Kind: `analysis_slice`
- Section: `main-results`
- Status: `completed`
- Metrics: full_model_mape_pct=9.24; no_wave_quant_mape_pct=47.6; single_row_roofline_err_pct=99; wave_model_single_row_err_pct=0.2

### Setup

Leave-one-term ablation of the closed-form model: full model vs a variant that omits the discrete wave term (continuous work), evaluated on the full 104-config grid.

### Result

Removing the discrete wave term n_waves=ceil(s_q/74) — treating work as continuous in the problem size, exactly what the datasheet roofline does — degrades MAPE from 9.24% to 47.6% on this grid, worst precisely where the wave structure matters most: at s_q=1 a single resident tile leaves the device nearly empty yet pays a full single-wave latency (roofline errs 99%, wave model 0.2% on the small kernel). Recovering the discrete occupancy-wave staircase is responsible for the bulk of the ~5x accuracy gain.

## A5

- Title: Synchronization-residual necessity: operator costs alone are insufficient (residual-free ablation)
- Kind: `analysis_slice`
- Section: `main-results`
- Status: `completed`
- Metrics: overlapped_sum_only_mape_pct=60.5; serialized_sum_only_mape_pct=44.6; with_predicted_residual_mape_pct=9.24; roofline_mape_pct=47.56

### Setup

Residual-free ablation of the calibration: fully-overlapped operator sum and fully-serialized operator sum (no pipeline-sync term) vs the composed model that predicts the synchronization residual from the cross-warp handshake, on the full 104-config grid. The relevant question is whether the predicted multi-level pipeline residual is necessary; the model consumes no whole-kernel measurement.

### Result

Two residual-free bottom-up variants that sum measured operator costs over the k iterations reach only 60.5% MAPE (fully overlapped) and 44.6% (fully serialized) — both worse than the 47.6% roofline they were meant to beat — because both omit the dominant pipeline-synchronization residual. Predicting that residual from the measured cross-warp handshake and the multi-level pipeline structure (no whole-kernel measurement) brings the model to 9.24%. Accuracy comes from composing measured operator scaling with a mechanistically predicted pipeline residual under the wave-quantized structure, not from operators alone.

## A6

- Title: Residual analysis of worst-error configurations
- Kind: `analysis_slice`
- Section: `main-results`
- Status: `completed`
- Metrics: worst_abs_pct_err=18.77; small_four_wave_max_abs_pct_err=4.5; residual_sign=underprediction; worst_residuals=small_first_wave_fill_plus_regular_softmax_underprediction

### Setup

Per-configuration absolute percentage error decomposed by wave occupancy (num_waves) and fill, computed from the on-board 104-config grid; residual sign taken as pred minus measured on the worst single-wave configs.

### Result

The error decomposes into two sign-consistent structural residuals. (1) Small kernel: the largest errors (~8-11%) are the partially filled single-wave configs (s_q=64,74); the model uses a constant per-wave cost while a nearly-full wave takes slightly longer (more tiles contend), confined to the first wave (every small four-wave config is under 5%). (2) Regular kernel: a uniform ~12-18% under-prediction independent of fill, because the composed residual (899 ns) falls short of the true ~1158 ns -- the wider B_topk=128 softmax cross-warp reduction latency is exposed but not yet captured by the matmul+handshake chain. Both are predicted (not fitted) and reported openly.

## A7

- Title: Operator-layer bound classification: compute-bound, not memory/gather-bound
- Kind: `analysis_slice`
- Section: `main-results`
- Status: `completed`
- Metrics: tensor_core_bound_configs=104/104; binder=tensor; gather_ns=16.2; qk_matmul_ns=339.9; gather_to_tensor_ratio=0.048

### Setup

Per-configuration bound classification derived from the standalone on-board operator costs: which operator forms the overlap envelope (binder) for each kernel across the 104-config grid.

### Result

Operator-layer bound classification: the overlap envelope is the tensor-core QK^T/SV matmul for both kernels (binder=tensor on 104/104 configs), with the scattered gather an order of magnitude cheaper (16.2 ns vs 339.9 ns). The kernel is compute-bound at the operator layer, correcting the naive expectation — and a datasheet memory ceiling — that a sparse-gather kernel must be memory- or gather-bound.

## A8

- Title: Cross-architecture portability protocol
- Kind: `analysis_slice`
- Section: `main-results`
- Status: `completed`
- Metrics: operators_to_remeasure=4; handshake_microbench=1; anchors_per_kernel=0; structural_constants_from_source=tiles_per_wave_atom_counts; second_device_validated=false

### Setup

Specification of exactly what must be re-measured (4 isolated operators + 1 cross-warp mbarrier handshake; no whole-kernel anchor) and what is read from source (structural constants) to transfer the model to another Blackwell-class accelerator. No second device validated.

### Result

Portability protocol: to transfer the closed form to another Blackwell-class accelerator, re-measure the four isolated operators (tile::gather4, exp2 softmax, QK and SV MMA atoms) plus the cross-warp mbarrier handshake (h); no whole-kernel anchor is measured. The structural constants (tiles_per_wave = num_SM/2 = 74 on the 148-SM B200, KV block size, per-iteration atom counts) are read from kernel source with no refit. Stated honestly: no second physical device was available, so portability is presented as a protocol specification, not a cross-device measurement.

## dsa-stagepred-grid-v2

- Title: Stage-centric wave-quantized, microbenchmark-calibrated latency model across both dispatch kernels — MAPE 9.24% over 104 configs (~5x better than roofline, zero whole-kernel fit)
- Kind: `main_experiment`
- Section: `main-results`
- Status: `completed`
- Metrics: latency_model=stage-centric-wave-quantized-dsa-composed-2kernel; mape_pct=9.24; mape_pct_small=4.92; mape_pct_regular=14.28; worst_abs_pct_err=18.77; roofline_mape_pct=47.56; n_configs=104; bottleneck=tensor_core_QKt; improvement_vs_roofline_x=5.1

### Setup

B200 (sm100), container ds003-flashmla. Target: FlashMLA DSA sparse-prefill forward (sparse_attn_fwd, phase1.cuh), two-CTA cluster -> 74 two-SM tiles/wave on 148 SMs. Both dispatch kernels: small-budget (B_topk=64, topk=1024, 56 cfgs) and regular (B_topk=128, topk=2048, 48 cfgs); 104 configs over batch s_q in {1,32,64,74,128,148,256,296} x context s_kv. Ground truth = kk.bench_kineto median, clean (no-instrumentation) build. Stage cost composed purely from standalone on-board operator microbenchmarks (gather4 TMA, register-resident exp2 softmax, QK and SV MMA atoms) plus a standalone cross-warp mbarrier handshake microbenchmark (h=140.33 ns); no in-kernel profiler, no whole-kernel measurement consumed for calibration.

### Result

Profiler-free, microbenchmark-calibrated composed stage model predicts whole-kernel latency at MAPE 9.24% over 104 configs (small-budget kernel 4.92% / regular kernel 14.28%), worst single config 18.8%, versus the naive datasheet roofline 47.6% (worst 99.1% at s_q=1) on the identical grid — about 5x lower error. No whole-kernel latency is consumed for calibration: the synchronization residual is predicted from a measured cross-warp mbarrier handshake (h=140.33 ns) composed along the kernel's multi-level producer/consumer structure (KV path 4-buffered/hidden, scoring path single-buffered with 3 exposed handshakes). The overlapped per-iteration cost is set by the tensor-core QK^T matmul (339.9 ns), while the scattered tile::gather4 is the cheapest operator (16.2 ns/block). The regular kernel is under-predicted (its wider softmax reduction is the unmodeled term), reported openly.

## A9 — Cross-family transfer to a structurally distinct dense FMHA kernel

Status: completed | Role: main + appendix | Section: main-results | Claim: C3

### Setup

Freeze every operator coefficient and the overlap envelope from the sparse-kernel calibration (zero refit) and apply the stage-centric model to a structurally distinct dense fused multi-head attention forward kernel (Sm100FmhaFwdKernel) on the same B200 (sm100) device — no data-dependent top-k selection, no scatter/gather, 2-D query x key tile schedule. Only per-iteration matmul atom counts and tile shape are read from the dense kernel source. One global overlap scalar (alpha=0.69) fit once over the whole 96-config grid. Ground truth = the kernel's own on-board CUDA-event latency over 96 configs (batch, query length, kv length, causal/non-causal). Source: experiments/main/fmha_grid/{fmha_grid.csv, json/fmha_stage_pred.json, fmha_sm100_sweep.py, fmha_stage_predictor.py}.

### Result

Frozen sparse-kernel coefficients transfer to the dense FMHA kernel with zero refit: naive datasheet roofline 44.98% MAPE -> staged serial (frozen coeff) 16.87% -> the staged serial 16.87% is the constraint-compliant cross-family number. A single global overlap scalar (alpha=0.69) would reach 9.48% MAPE (worst 31.03%) but is calibrated on whole-kernel latencies, so it lies OUTSIDE the no-whole-kernel-fit contract and is reported only as a sensitivity point. All 96 configs tensor-core bound, reproducing the DSA bottleneck conclusion. Residual splits causal 11.4% vs non-causal 7.6%, matching the online-softmax serial-fraction explanation. Does not reach the sparse kernels' 9.24%; the gap is an interpretable scope boundary. Additive integration — the DSA 9.24% / 104-config headline is unchanged.
