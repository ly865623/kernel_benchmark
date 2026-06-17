# Impact · Main Results

## Claim Links

- `C1`
- `C3`
- `C2`

## Impact Notes

- `Headline accuracy: stage model versus naive datasheet roofline`: Establishes C1: the stage-centric model meets the sub-10% aggregate pre-sweep target and is ~5x more accurate than the datasheet roofline, with zero whole-kernel calibration.
- `Per-operator on-board decomposition and bottleneck attribution`: Supports C2/C3: the on-board operator measurement overturns the gather-bound expectation (tensor-core binds) and quantifies the pipeline-sync residual, which the composed model predicts from the measured handshake and multi-level structure.
- `Context-length (s_kv) invariance at fixed selection budget`: Supports C1: validates the model's structural claim that cost is wave- and selection-budget-driven, not s_kv-driven.
- `Wave-quantization-term ablation: the decisive structure`: Supports C1: isolates wave quantization as the single structural ingredient separating the model from the roofline.
- `Synchronization-residual necessity: operator costs alone are insufficient (residual-free ablation)`: Supports C2: quantifies why a bottom-up operator sum is insufficient and why the predicted multi-level synchronization residual is required.
- `Residual analysis of worst-error configurations`: Strengthens transparency: the residual is structured, interpretable, and reported openly rather than absorbed into a fitted constant.
- `Operator-layer bound classification: compute-bound, not memory/gather-bound`: Supports C3: corrects the bottleneck attribution from memory/gather-bound (profiler-era) to tensor-core compute-bound (on-board).
- `Cross-architecture portability protocol`: Defines the reproducible recipe to re-target the model; scopes the portability claim honestly (protocol, single device).
- `Stage-centric wave-quantized, microbenchmark-calibrated latency model across both dispatch kernels — MAPE 9.24% over 104 configs (~5x better than roofline, zero whole-kernel fit)`: Headline result: an interpretable, zero-whole-kernel-fit, on-board-microbenchmark-calibrated composed model is accurate (9.24% MAPE) and portable across the family's two dispatch kernels.
