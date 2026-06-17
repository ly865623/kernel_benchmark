# Stage-Centric, Microbenchmark-Calibrated Analytical Latency Modeling of Sparse Attention Kernels on Blackwell GPUs

- Paper type: `full_empirical`
- Outline maturity: `mature`

## One-Sentence Paper Idea

- Central thesis: A closed-form stage decomposition, calibrated only from standalone on-board microbenchmarks of the kernel's own operators plus datasheet constants, predicts the whole-kernel latency of data-dependent sparse-attention kernels within sub-10% error while staying fully interpretable, because the cost is set by the discrete occupancy-wave count and the top-k selection budget, and because the dominant per-iteration cost is the kernel's tensor-core matmul together with a warp-specialized pipeline-synchronization residual predicted from a measured cross-warp handshake and the kernel's multi-level producer/consumer structure (with no whole-kernel measurement) -- not by datasheet peaks, the full context length, or the scattered gather a sparse kernel is named for.
- What readers learn: For modern sparse-attention prefill kernels, what a performance model must get right is not raw FLOP/byte peaks but (a) the data-dependent amount of work induced by sparsity (the occupancy-wave count and top-k iterations) and (b) the realistic on-board cost of the kernel's own operators measured in isolation; the surprising finding is that the scattered gather is the cheapest operator and the kernel is tensor-core bound, and that the isolated operators explain under 40% of per-iteration time -- the remainder being warp-specialized pipeline synchronization predicted from a measured cross-warp handshake and the multi-level pipeline structure, with no whole-kernel measurement.

## Story Spine

- Problem: Deploying sparse-attention kernels on new accelerators requires predicting whole-kernel latency before exhaustive on-device sweeps, yet the kernels are data-dependent (their work scales with a learned sparse selection, not the sequence length) and memory-irregular (key/value blocks are gathered at scattered addresses).
- Gap: A naive datasheet-peak roofline mispredicts these kernels by tens of percent (about 48% mean error, ~5x the proposed model) and offers no per-stage explanation, while opaque curve-fitting predicts numbers but is neither interpretable nor portable to a new architecture.
- Method: A stage-centric analytical model that decomposes the kernel into compute, contiguous query/output IO, scattered key/value gather, and fixed pipeline overhead, with two interpretable sparse-attention extensions: a data-dependent, wave-quantized tile count driven by the top-k budget and a single on-board pipeline-synchronization anchor that captures the warp-specialized residual the isolated operators miss; all coefficients come from standalone on-board microbenchmarks or datasheets, none fit to the target latencies.
- Main result: On a current Blackwell-class GPU the model predicts whole-kernel latency with 9.24% mean absolute percentage error across 104 configurations spanning two dispatch kernels, roughly 5x below the 47.6% naive-roofline error, while emitting a per-stage breakdown that correctly attributes the bottleneck to the tensor-core matmul.
- Scope limit: Validated on the sparse-prefill forward kernel of one sparse-attention family on a single Blackwell-class architecture over 104 configurations across two dispatch kernels; cross-architecture transfer is described as a coefficient-remeasurement protocol but is not yet empirically validated on a second device, and the worst single configuration sits at 9.72%, just within the sub-10% target.

## Positioning

- closest_neighbor: Datasheet-peak roofline modeling and prior stage-centric analytical kernel models that target dense GEMM/attention; the immediate prior art is the stage-centric analytical methodology this work specializes to data-dependent sparse attention.
- novelty_boundary: The new and reusable contributions are (1) treating the sparse selection budget as the latency-determining work unit via a data-dependent tile count, and (2) replacing datasheet bandwidth with a microbenchmark-measured, de-rated gather bandwidth for the irregular key/value stream, both embedded in a closed-form, no-target-fit model.
- why_not_prior_work: Not recorded
- not_claiming: Not claiming state-of-the-art absolute prediction accuracy across all attention kernels or all hardware., Not claiming empirical cross-architecture accuracy beyond the single architecture measured here., Not claiming the model replaces profiling for final kernel selection; it is a pre-sweep predictor., Not claiming any speedup of the kernel itself; this is a performance-modeling contribution, not a kernel optimization.

## Core Claims

- `C1` A closed-form stage-centric model calibrated only from standalone on-board microbenchmarks predicts sparse-prefill whole-kernel latency with sub-10% aggregate mean error and roughly 5x lower error than a naive datasheet roofline, with no whole-kernel latency consumed for calibration.
- `C2` The latency of the sparse-prefill kernel is governed by the top-k selection budget through a data-dependent tile count and is approximately independent of the full key/value sequence length; modeling this is what closes the gap the roofline cannot.
- `C3` The per-stage decomposition yields an interpretable, correct bottleneck attribution -- the kernel is tensor-core (QK^T matmul) bound on every configuration while the scattered key/value gather is the cheapest operator -- and the same closed form is portable to other Blackwell-class accelerators by re-measuring its on-board coefficient table.

## From Facts To Interpretation

- `Observed fact -> interpretation` Across configurations spanning an 8x range of context length at fixed selection budget, measured latency is essentially flat; this teaches that the kernel cost is governed by the sparse selection budget, not the dense context length.
- `Observed fact -> interpretation` Standalone on-board operator microbenchmarks show the kernel is tensor-core (QK^T matmul) bound while the scattered key/value gather is the single cheapest operator; this overturns the roofline-style memory-bandwidth intuition and teaches that the decisive coefficient is the measured tensor-core throughput, not a de-rated gather bandwidth.
- `Observed fact -> interpretation` The analytical model reaches 9.24% mean error across 104 configurations spanning two dispatch kernels while the naive datasheet roofline sits at 47.6% (~5x worse); this teaches that the accuracy gain comes from stage-level structure -- wave-quantized occupancy plus a synchronization residual predicted from a measured cross-warp handshake and the multi-level pipeline structure -- rather than from tuning, since no whole-kernel latency is consumed for calibration.

## Evidence Boundaries

- Observed facts: Whole-kernel MAPE is 9.24% over 104 configurations across both dispatch kernels (4.92% small-core / 14.28% regular); worst single-config absolute error is 18.77%., Naive datasheet-peak roofline MAPE is 47.56% on the same configurations (~5x worse)., Measured latency is flat (within ~1%) across an 8x context-length sweep at fixed selection budget., Standalone on-board operator microbenchmarks show every configuration is tensor-core (QK^T matmul) bound, with the scattered key/value gather the single cheapest operator (16.2 ns)., All model coefficients are sourced from standalone on-board microbenchmarks or datasheet values; none are fit to the target latencies.
- Allowed interpretations: Stage-level structure (wave-quantized occupancy from the selection-driven tile count plus a synchronization residual predicted from a measured cross-warp handshake and the multi-level pipeline structure) explains most of the accuracy gain over roofline., The selection budget, not the context length, is the latency-determining work unit for this kernel., Per-stage decomposition provides an interpretable bottleneck attribution usable for pre-sweep planning.
- Do not claim: That the model is empirically accurate on architectures other than the one measured., That sub-10% holds for every individual configuration (one family is at 10.8%)., That this improves kernel runtime or replaces final on-device profiling., That the result generalizes to attention kernels outside the sparse-prefill family tested.
- Evidence gaps: No second-architecture measurement to validate portability empirically., Structured under-prediction in the top-k=1024 family not yet fully explained., Ablation and sensitivity analyses (A4, A5, A6) still to be computed offline from existing data.

## Method

- Paper name: Stage-centric sparse-attention latency model
- Intuition: A warp-specialized attention kernel is a software pipeline; its steady-state latency is the max of overlapped compute and memory stages times the number of work tiles, plus fixed prologue overhead. For sparse attention the number of work tiles is set by the learned selection budget, and the memory stage runs at the measured bandwidth of an irregular gather, so plugging measured per-stage rates into this structure recovers latency without fitting.
- Step: Decompose the kernel into stages: kernel launch and pipeline fill overhead, tensor-core compute, contiguous query-load and output-store IO, and scattered key/value-block gather IO.
- Step: Compute a data-dependent tile count from the top-k selection budget divided by the per-step block size, independent of the full sequence length.
- Step: Assign each stage a coefficient measured by an independent microbenchmark or read from the datasheet: tensor-core throughput ceiling, contiguous streaming bandwidth, and a de-rated effective bandwidth for the scattered gather.
- Step: Overlap compute and IO within a tile (take the max), scale by the tile count, add fixed overhead, and emit both the scalar latency prediction and the per-stage breakdown that determines the bottleneck class.
- Step: Port to a new accelerator by re-measuring the same small coefficient table with the same microbenchmarks; no structural change to the closed form.

## Evaluation

- Setting: Offline closed-form prediction of whole-kernel latency over a fixed set of measured ground-truth configurations, with mean absolute percentage error against on-device kernel timing as the headline metric.
- datasets_or_benchmarks: Sparse-prefill forward kernel ground-truth latencies over 104 configurations spanning two dispatch kernels (small-core and regular) across context-length sweeps and a range of top-k selection budgets.
- baselines: Naive datasheet-peak roofline (compute-vs-bandwidth max)
- metrics: Whole-kernel MAPE (%), Worst-config absolute percentage error (%), Per-config relative error, Achieved TFLOPS / effective HBM bandwidth cross-checks
- controlled_factors: No coefficient fit to target latencies (anti-overfit contract), Same coefficient table across all configs, Fixed kernel structural constants from source

## Analysis Plan

- `A1` Roofline baseline comparison (stronger-baseline comparison)
- `A2` Per-operator on-board decomposition and bottleneck attribution (mechanism or attribution check)
- `A3` Sequence-length independence at fixed selection budget (mechanism or attribution check)
- `A4` Wave-quantization-term ablation (component ablation)
- `A5` Anchor necessity: zero-anchor ablation (robustness or sensitivity)
- `A6` Residual analysis of the marginal selection-budget family (failure taxonomy)
- `A7` Operator-layer bound classification across the config space (subgroup or case breakdown)
- `A8` Cross-architecture portability protocol (limitation or residual headroom analysis)

## Reviewer Objections

- Ten configurations on one device is too small to support a general accuracy claim. -> claim_downgrade
- The good accuracy could be hidden curve-fitting rather than genuine first-principles modeling. -> writing
- The cross-platform portability claim is asserted but never demonstrated on a second architecture. -> limitation
- The model only beats the weakest possible baseline (naive datasheet roofline). -> writing
