# Paper Experiment Matrix

- Selected outline: `outline-001`
- Rows: `9`
- Updated at: `2026-06-15T11:52:00+00:00`

| Item | Section | Required | Status | Metrics |
|---|---|---|---|---|
| `A1` | main-results | True | completed | whole_kernel_mape_pct=9.24; worst_config_abs_pct_err=18.77; roofline_mape_pct=47.56; roofline_worst_pct=99.1; improvement_vs_roofline_x=5.1; configs_won=104/104 |
| `A2` | main-results | True | completed | qk_matmul_ns=339.9; gather_ns=16.2; softmax_ns=165.2; tensor_overlap_envelope_ns=471.5; per_iteration_ns=1220; operator_sum_share_of_periter_pct=38.9; sync_residual_share_pct=61.1; sync_residual_ns=749; sync_residual_pred_ns=731.8; h_oneway_ns=140.33; binder=tensor; tensor_binds=104/104 |
| `A3` | main-results | True | completed | max_across_skv_spread_pct=3.25; mean_across_skv_spread_pct=1.48; skv_sweep_range_x=128 |
| `A4` | main-results | True | completed | full_model_mape_pct=9.24; no_wave_quant_mape_pct=47.6; single_row_roofline_err_pct=99; wave_model_single_row_err_pct=0.2 |
| `dsa-stagepred-grid-v2` | main-results | True | completed | latency_model=stage-centric-wave-quantized-dsa-composed-2kernel; mape_pct=9.24; mape_pct_small=4.92; mape_pct_regular=14.28; worst_abs_pct_err=18.77; roofline_mape_pct=47.56; n_configs=104; bottleneck=tensor_core_QKt; improvement_vs_roofline_x=5.1 |
| `A5` | main-results | False | completed | overlapped_sum_only_mape_pct=60.5; serialized_sum_only_mape_pct=44.6; with_predicted_residual_mape_pct=9.24; roofline_mape_pct=47.56 |
| `A6` | main-results | False | completed | worst_abs_pct_err=18.77; small_four_wave_max_abs_pct_err=4.5; residual_sign=underprediction; worst_residuals=small_first_wave_fill_plus_regular_softmax_underprediction |
| `A7` | main-results | False | completed | tensor_core_bound_configs=104/104; binder=tensor; gather_ns=16.2; qk_matmul_ns=339.9; gather_to_tensor_ratio=0.048 |
| `A8` | main-results | False | completed | operators_to_remeasure=4; handshake_microbench=1; anchors_per_kernel=0; structural_constants_from_source=tiles_per_wave_atom_counts; second_device_validated=false |
