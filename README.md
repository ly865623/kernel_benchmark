# Stage-Centric Latency Model for the FlashMLA DSA Sparse-Prefill Kernel (B200)

End-to-end reproduction guide for the experimental results behind the paper
*"Stage-Centric Microbenchmark-Calibrated Latency Modeling of Sparse Attention Kernels on Blackwell GPUs"*.

**Headline result.** An interpretable, wave-quantized analytical model predicts the whole-kernel
latency of DeepSeek's DSA sparse-prefill kernel (`sparse_attn_fwd`) across **104 configurations**
spanning both selection budgets the kernel family dispatches (`topk ∈ {1024, 2048}`) with
**MAPE = 3.88%** overall (small `B_TOPK=64` kernel 4.92%, regular `B_TOPK=128` kernel 2.67%),
versus **47.6%** for a naive datasheet roofline — with **no coefficient fit to the target
whole-kernel latencies**. The headline model is produced (no GPU) by
`experiments/microbench/dsa_stage_model_piecewise.py` (§6). An earlier two-kernel variant *without*
the microbenchmarked correction stage lands at 6.01%; see §6.

> **Calibration provenance — the scientific point of the work.** Every model coefficient is a
> *standalone on-board microbenchmark of the kernel's own primitive* (scattered `tile::gather4`
> KV gather, register-resident softmax `exp2`, the `QK^T`/`SV` tensor-core matmul atoms, the FP32
> online-softmax correction, and a bare `mbarrier` cross-warp handshake), plus per-iteration atom
> counts read directly from the kernel source. **The fused kernel is never run to calibrate the
> model: no whole-kernel latency and no in-kernel profiler (`FLASHINFER_ENABLE_PROFILER`) timing is
> consumed as a calibration input.** On-device whole-kernel latencies appear *only* to score MAPE,
> and are themselves measured from a clean profiler-off (production) build.
>
> ⚠️ **Do not confuse this with the earlier profiler-based model.** A *superseded* earlier stage
> model (`experiments/main/dsa_stage_model_v2.py`, §5) instead composed the per-stage spans read
> from the kernel's `FLASHINFER_ENABLE_PROFILER` instrumentation, and *coincidentally* also scores
> 4.92% on the dense 56-config (`topk=1024`) grid. **That model is an internal cross-check only — it
> is NOT the manuscript's model or its calibration path.** Its 4.92% is a numeric coincidence with
> the microbench small-kernel 4.92% above.

---

## 1. What gets reproduced

| # | Artifact | Produced by | Output |
|---|----------|-------------|--------|
| 1 | Ground-truth baseline latency (10 configs) + naive roofline bar | `experiments/baseline/measure_dsa_prefill.py` | `baselines/local/flashmla-dsa-b200/` |
| 2 | Dense whole-kernel latency grid (56 configs) | `experiments/main/sweep_grid_v2.py` | `experiments/main/grid_v2/` |
| 3 | Per-stage profiler decomposition (17 pipeline stages) — _used by the §5 cross-check only_ | `experiments/main/stage_extract.py` | `experiments/main/stages/` |
| 4 | _Earlier profiler-based_ stage model → MAPE 4.92% (**internal cross-check, NOT the paper headline**) | `experiments/main/dsa_stage_model_v2.py` | `experiments/main/stage_model_v2_results.json`, `predictions.json` |
| 5 | top-k scan grid (top-k∈{1024,2048}, 104 configs) | top-k sweep (see §6) | `experiments/analysis-results/topk_scan_v2/json/grid_topk_v2.json` |
| 6 | **Manuscript headline** — microbench-calibrated, profiler-free piecewise model → MAPE **3.88%** | `experiments/microbench/dsa_stage_model_piecewise.py` | `experiments/microbench/piecewise_results.json` |
| 6b | Earlier no-correction / amortization variants (reference, not the headline) | `eval_*.py` in `topk_scan_v2/` | `experiments/analysis-results/topk_scan_v2/json/model_eval_*.json` |
| 7 | Paper-facing top-k figure | `experiments/main/plot_topk_v2.py` | `experiments/analysis-results/topk_scan_v2/topk_scan_latency_skv8192.png` |

**Step 6 (`dsa_stage_model_piecewise.py`) is the manuscript headline** — microbench-calibrated and
profiler-free. Steps 1–2 + 5 produce the measured ground-truth latency grids it is *scored* against.
**Steps 3–4 are the earlier profiler-based cross-check, not the paper's calibration path** (see the
provenance note above).

---

## 2. Hardware & environment

The measurement steps (1, 2, 3, 5) **require a real NVIDIA B200 (sm100) GPU**. The model and
evaluation steps (4, 6, 7) are pure post-processing of the recorded JSON/CSV and run anywhere with
Python 3 + NumPy/Matplotlib.

| Item | Value |
|------|-------|
| GPU | NVIDIA B200 (sm100), 148 SMs, SM clock 1.965 GHz |
| Container | `nvcr.io/nvidia/pytorch:26.01-py3-v0` (referred to as `ds003-flashmla`) |
| Framework | PyTorch 2.10.0a0 (NGC 26.01), CUDA 13.1 |
| Kernel source | `deepseek-ai/FlashMLA`, commit `48c6dc4` (byte-identical to upstream `main` for the modeled sparse-prefill path — see `experiments/main/PROVENANCE.md`) |

The measurement scripts locate the FlashMLA checkout through the `FLASHMLA_ROOT` environment
variable (default `/workspace/code/FlashMLA`) and reuse its test harness
(`tests/lib.py`, `tests/ref.py`, `kernelkit`) unmodified.

---

## 3. Step 0 — Build the kernel (measurement steps only)

Two builds of FlashMLA are needed, differing only in one compile flag:

- **Clean / production build** (no profiler) — used for the whole-kernel latency grids (steps 1, 2, 5).
  Generated machine code is identical to stock upstream.
- **Profiler build** (`-DFLASHINFER_ENABLE_PROFILER`) — used **only** for the per-stage
  decomposition (step 3). The profiler macros are compile-gated and expand to nothing in the clean
  build, so they do not alter the measured computation (full audit in
  `experiments/main/PROVENANCE.md`).

```bash
# inside the container, with the GPU visible
export FLASHMLA_ROOT=/workspace/code/FlashMLA
cd $FLASHMLA_ROOT

# (a) clean build for latency grids
git checkout 48c6dc4
python setup.py install        # or the repo's documented build entrypoint

# (b) profiler build for stage decomposition (rebuild when you reach step 3)
FLASHINFER_ENABLE_PROFILER=1 python setup.py install
#   equivalently: pass -DFLASHINFER_ENABLE_PROFILER to the kernel nvcc flags
```

> The exact upstream build invocation follows FlashMLA's own README; only the profiler flag is
> project-specific. Rebuild clean again before re-running any latency grid.

---

## 4. Step 1 — Ground-truth baseline

Measures the median `sparse_attn_fwd` kernel time (kineto / CUPTI, L2-flushed) over 10
representative configs, plus achieved TFLOPS, effective HBM bandwidth, and the naive-roofline
relative error that the analytical model must beat.

```bash
export FLASHMLA_ROOT=/workspace/code/FlashMLA   # clean build
cd <repo-root>
python experiments/baseline/measure_dsa_prefill.py \
    --out baselines/local/flashmla-dsa-b200 \
    --num-tests 30 --passes 3
```

- **Code:** `experiments/baseline/measure_dsa_prefill.py`
- **Outputs:** `baselines/local/flashmla-dsa-b200/json/ground_truth_prefill.json`, `ground_truth_prefill.csv`, `sweep.log`
- **Comparison contract (canonical):** `baselines/local/flashmla-dsa-b200/json/metric_contract.json`
  (primary metric `latency_us`; supporting `tflops`, `mem_bw_tbps`, `roofline_rel_err`).

---

## 5. Steps 2–4 — Measured dense grid + the earlier profiler-based cross-check (NOT the paper headline)

> The model in Step 4 (`dsa_stage_model_v2.py`) is the project's **earlier, profiler-based** stage
> model: it composes the kernel's per-stage `FLASHINFER_ENABLE_PROFILER` spans (Step 3). It is kept
> as a transparent internal cross-check and is **not** the manuscript's model — the paper headline
> is the microbenchmark-calibrated, profiler-free piecewise model of §6. The dense 56-config grid of
> Step 2 is also the ground truth the §6 microbench model is scored against on the `topk=1024` subset.

### Step 2 — Dense whole-kernel latency grid (56 configs)

`batch_size (= s_q) ∈ {1,32,64,74,128,148,256,296} × s_kv ∈ {1k,4k,8k,16k,32k,64k,128k}`,
fixed `h_q=128, d_qk=576, d_v=512, topk=1024, attn_sink=True`. The batch points are chosen on the
wave boundary (74 two-SM tiles = 1 wave on 148 SMs): `{74,148,296}` = 1/2/4 full waves.

```bash
export FLASHMLA_ROOT=/workspace/code/FlashMLA   # clean build
python experiments/main/sweep_grid_v2.py \
    --out experiments/main/grid_v2 \
    --num-tests 30 --passes 3 --topk 1024 --d-qk 576
# quick check first: add --smoke for a single tiny config
```

- **Code:** `experiments/main/sweep_grid_v2.py`
- **Outputs:** `experiments/main/grid_v2/json/grid_v2.json`, `experiments/main/grid_v2/grid_v2.csv`

> Note: the script's `--out` default points at the original run worktree. Pass
> `--out experiments/main/grid_v2` so the model in step 4 finds the data where it expects it.

### Step 3 — Per-stage profiler decomposition

Runs the real kernel with a profiler buffer and decodes the on-chip `%globaltimer` (ns) spans of
the kernel's own 17 pipeline stages — these *are* the "stage micro-kernels", carved from the
original kernel, not external synthetic benchmarks.

```bash
export FLASHMLA_ROOT=/workspace/code/FlashMLA   # PROFILER build (-DFLASHINFER_ENABLE_PROFILER)
python experiments/main/stage_extract.py \
    --out experiments/main/stages \
    --full-grid                     # 56-config grid; omit for the 8 representative configs
# quick check first: --smoke (single config 74×8192)
```

- **Code:** `experiments/main/stage_extract.py`
- **Outputs:** `experiments/main/stages/json/stage_timings.json` (+ `stage_grid.csv`, `stage_timings_full.json`)
- **Profiler-era reading (superseded by the manuscript):** the raw profiler spans suggest a
  gather/softmax-heavy loop. **The microbench analysis (§6) refines this:** the scattered gather is
  actually cheap (16–32 ns/block) and the overlapped per-iteration cost is set by the **tensor-core
  `QK^T` matmul** — this QK-bound conclusion is the manuscript's, and it overturns the naive
  gather-bound expectation. See `experiments/microbench/results/MEASUREMENT_NOTES.md`.

### Step 4 — Earlier profiler-based stage model (cross-check, no GPU needed)

> **NOT the paper headline.** This model consumes the kernel's per-stage `FLASHINFER_ENABLE_PROFILER`
> spans (`stage_timings.json` from Step 3) as its coefficient source. The manuscript model instead
> draws every coefficient from standalone microbenchmarks and consumes **no** profiler timing — see
> §6. The profiler-free microbench equivalent of this dense-grid model is
> `experiments/microbench/dsa_stage_model_microbench.py` (anchored MAPE ≈ 3.75% on the same 56-config
> `topk=1024` grid), which is the version aligned with the paper's calibration contract.

```bash
python experiments/main/dsa_stage_model_v2.py
```

- **Code:** `experiments/main/dsa_stage_model_v2.py`
- **Reads:** `experiments/main/grid_v2/json/grid_v2.json`, `experiments/main/stages/json/stage_timings.json`
  (the second input is the profiler decomposition — this is exactly what makes Step 4 profiler-based)
- **Outputs:** `experiments/main/stage_model_v2_results.json`, `experiments/main/predictions.json`
- **Model (interpretable, closed form):**
  - `num_tiles = s_q`; `num_waves = ceil(num_tiles / 74)`; `fill = num_tiles / (num_waves·74)`
  - `T_pred = T_launch + num_waves · T_wave(fill)`, with the per-iteration step time set by the
    profiler-measured bottleneck (gather/softmax) stage and a mild HBM-contention slope read from
    the profiler — **not fit to whole-kernel latency**.
  - Two transparent coefficient sources reported: **(A)** profiler-only, **(B)** single-tile (bs=1)
    anchored. Both land at MAPE ≈ 4.9%.
- **Expected metrics:** `mape_pct = 4.92`, `mape_pct_anchored = 4.93`, `worst_abs_pct_err = 11.32`,
  `roofline_mape_pct = 50.28`, `n_configs = 56`, `improvement_vs_roofline_x = 10.2`.

> `experiments/main/dsa_predictor.py` is a standalone reference implementation of the same
> stage-centric forward model wired directly to the baseline ground-truth JSON; useful for an
> independent cross-check of the analytical formulation.

The recorded run record is `experiments/main/dsa-stagepred-grid-v2/RUN.md` / `RESULT.json`.

---

## 6. Steps 5–7 — Manuscript headline: microbench-calibrated piecewise model (104 configs)

A grid sweeps `topk ∈ {1024, 2048}` (with the kernel's `B_TOPK` block constraint:
multiple of 128, `topk ≤ s_kv`), giving 104 effective configs. **This 104-config evaluation is the
paper's headline result (MAPE 3.88%); the piecewise model below is the manuscript's model.**
top-k=2048 dispatches a *different* compiled kernel
(`topk > 1280` → regular B_TOPK=128 kernel; `topk ≤ 1280` → small B_TOPK=64 kernel — see
`csrc/api/sparse_fwd.h`).

The top-k grid was measured with a clean profiler-off build (provenance:
`experiments/analysis-results/topk_scan_v2/PROVENANCE.md`); the resulting
`grid_topk_v2.json` is committed, so the evaluation below runs without a GPU.

```bash
# (PRIMARY) manuscript generalization headline — piecewise correction model -> 3.88%
#   small B_TOPK=64 kernel (topk=1024): 4.92% ; regular B_TOPK=128 kernel (topk=2048): 2.67% ; both (104 cfg): 3.88%
#   the correction stage (t_sm_reduce + t_correction) is added to t_scoring ONLY for B_TOPK=128;
#   its cost is microbenchmarked (correction_bench), not fit to whole-kernel latency.
#   runs from the repo root, reads the committed grid_topk_v2.json, no GPU.
python experiments/microbench/dsa_stage_model_piecewise.py   # -> experiments/microbench/piecewise_results.json

# --- earlier no-correction / amortization variants (reference only, NOT the manuscript headline) ---
cd experiments/analysis-results/topk_scan_v2

# (5/6a) published v2 model, no refit, split by top-k (regular kernel un-corrected)
python eval_model_on_topk.py            # -> json/model_eval_on_topk.json

# (6b) two-kernel extension WITHOUT correction: small 4.83% / regular 7.39% / combined 6.01%
python eval_two_kernel.py               # -> json/model_eval_two_kernel.json

# (6c) fixed-cost (prologue/epilogue) amortization correction from the profiler
python eval_amortized_variant.py        # -> json/model_eval_amortized_variant.json

# (7) paper-facing figure
python ../../main/plot_topk_v2.py       # -> topk_scan_latency_skv8192.png
```

- **Code:** `eval_model_on_topk.py`, `eval_two_kernel.py`, `eval_amortized_variant.py`, `plot_topk_v2.py`
- **Key finding:** `latency(2048)/latency(1024) = 1.61×` (sub-2×, fixed-overhead amortization).
  The manuscript headline two-kernel result is the **piecewise correction model: small 4.92% /
  regular 2.67% / combined 3.88%** (`experiments/microbench/dsa_stage_model_piecewise.py`). The
  no-correction two-kernel variant (`eval_two_kernel.py`) lands at small 4.83% / regular 7.39% /
  combined 6.01%; adding the microbenchmarked correction stage (`t_sm_reduce + t_correction`) to the
  regular B_TOPK=128 kernel — exposed there but pipeline-covered in the small kernel — is what closes
  the gap to 2.67% / 3.88%.

> These eval scripts import the published model module verbatim (so every `topk=1024` prediction is
> byte-identical to the headline) and currently use **absolute paths** into this worktree
> (`.../paper-dsa-stagepred-grid-v2/...`). If you relocate the repo, update the `MAIN`/`HERE`/`TOPK`
> path constants at the top of each script, or run them from this worktree root.

---

## 6.5 Microbenchmark calibration data (how the headline model's coefficients are measured)

The §6 headline model consumes **no** whole-kernel or profiler timing as a calibration input — every
coefficient comes from one of **five standalone microbenchmarks** of the kernel's own primitives. The
sources, build/run commands, and committed CSV outputs all live under `experiments/microbench/`.
**Re-measuring requires a real B200 (sm_100) + `nvcc` (CUDA 13.1); the output CSVs are committed, so
the numbers can be inspected without a GPU.** The per-iteration atom *counts* (`qk_atoms=18`,
`sv_atoms=4/8`) are **read from the kernel source** (`config.h`, `phase1.cuh`), not measured.

| Microbench | Measures → model constant (in `dsa_stage_model_piecewise.py`) | Source + run | Committed output |
|---|---|---|---|
| MMA atoms | QK/SV tensor-core throughput + single-op latency → `MMA_N128_CYC=37.106`/`LAT=178`, `MMA_N256_CYC=64.648`/`LAT=210` (cyc) | external suite `/home/liuy/code/microbench-blackwell/umma_throughput` (+ `umma_latency`): `make && ./umma_tput.out` | `experiments/microbench/results/mma_costs.csv` |
| Scattered KV gather | `tile::gather4` BW → `GATHER_NS_64=16.2` ns/64-token block (~2.0 TB/s) | `experiments/microbench/gather4_bench/build_and_run.sh` (reuses FlashMLA's own `ku::tma_gather4`; needs `FM=/home/liuy/code/FlashMLA`, `arch=sm_100f`) | `results/gather4_scatter_bw.csv` |
| Softmax `exp2` | register-resident `MUFU.EX2` SFU throughput → `EXP2_OPS_PER_S_PER_SM=24.8e9` | `experiments/microbench/exp2_bench/exp2_tput.cu` (`nvcc -O3 -arch=sm_100f exp2_tput.cu -o exp2_tput.out && ./exp2_tput.out`) | `results/exp2_sfu.csv` |
| FP32-ALU correction | FMUL/FFMA rate for the online-softmax rescale → `FP32_MUL_OPS=176.567e9`, `FP32_FFMA_OPS=174.600e9` | `experiments/microbench/correction_bench/build_and_run.sh` | `correction_bench/corr_fp32_alu.csv` |
| Pipeline handshake | bare `mbarrier` cross-warp one-way signal latency → `H_ONEWAY_NS=140.33` (= 280.66/2) | `experiments/microbench/sync_bench/mbar_pipeline.cu` (`nvcc -O3 -arch=sm_100a mbar_pipeline.cu -o mbar_pipeline.out && ./mbar_pipeline.out`) | `sync_bench/sweep*.csv` + `sync_bench/NOTES.md` |

The full measurement log, per-op atom-count derivation, and the key findings (gather is cheap; the
overlapped binder is the `QK^T` matmul; the four isolated ops account for <40% of per-iter time, the
rest being the predicted synchronization residual) are in
`experiments/microbench/results/MEASUREMENT_NOTES.md` and `sync_bench/NOTES.md`.

> The `gather4` and MMA benches deliberately reuse the kernel's *own* primitives (FlashMLA / CUTLASS
> headers) at the kernel's exact instruction shapes, so they measure the same hardware path the fused
> kernel pays — not a generic datasheet proxy. The `exp2`, correction, and handshake benches are
> self-contained CUDA.

---

## 7. Repository map

```
README.md                         <- this file
brief.md / plan.md / PLAN.md      <- quest framing, research plan, node contract
SUMMARY.md / status.md            <- durable quest summary & status

baselines/
  local/flashmla-dsa-b200/        <- measured ground truth + metric_contract.json (canonical comparison)
  imported/flashmla-dsa-b200/     <- attachment record (attachment.yaml)

experiments/
  baseline/measure_dsa_prefill.py <- step 1: ground-truth measurement harness
  main/
    sweep_grid_v2.py              <- step 2: 56-config whole-kernel latency sweep
    stage_extract.py              <- step 3: per-stage profiler decomposition (feeds the §5 cross-check only)
    dsa_stage_model_v2.py         <- step 4: EARLIER profiler-based stage model (cross-check, NOT the paper headline)
    dsa_predictor.py              <- standalone reference predictor (cross-check)
    plot_topk_v2.py               <- top-k figure renderer
    grid_v2/                      <- step 2 outputs (json/grid_v2.json, grid_v2.csv)
    stages/                       <- step 3 outputs (json/stage_timings.json, ...)
    stage_model_v2_results.json   <- step 4 result; predictions.json
    PROVENANCE.md                 <- upstream-equivalence + profiler audit (read this)
    dsa-stagepred-grid-v2/        <- recorded run (RUN.md, RESULT.json)
  microbench/                     <- the MANUSCRIPT model + its standalone microbenchmarks (profiler-free)
    dsa_stage_model_piecewise.py  <- *** PAPER HEADLINE *** microbench-calibrated piecewise model -> 3.88%
    dsa_stage_model_microbench.py <- profiler-free dense 56-config model (anchored ~3.75%)
    gather4_bench/ exp2_bench/ correction_bench/ sync_bench/  <- standalone on-board op microbenchmarks
    results/MEASUREMENT_NOTES.md  <- microbench measurement log + per-op atom costs (read this)
  analysis-results/topk_scan_v2/  <- steps 5-7: top-k grid (ground truth), eval_*.py, PROVENANCE.md, figure

paper/                            <- outline, evidence ledger, manuscript bundle (LaTeX under paper/latex/)
```

---

## 8. Minimal reproduction (no GPU)

If you only need to regenerate the model results and figures from the committed measurement data:

```bash
cd <repo-root>
# *** PAPER HEADLINE *** microbench-calibrated, profiler-free piecewise model -> 3.88% (104 cfg)
python experiments/microbench/dsa_stage_model_piecewise.py          # small 4.92% / regular 2.67% / both 3.88%
python experiments/microbench/dsa_stage_model_microbench.py         # profiler-free dense 56-cfg model (~3.75%)
# earlier profiler-based stage model (internal cross-check, NOT the paper headline):
python experiments/main/dsa_stage_model_v2.py                       # profiler-based, coincidentally 4.92%
# reference variants + figure:
cd experiments/analysis-results/topk_scan_v2
python eval_model_on_topk.py && python eval_two_kernel.py && python eval_amortized_variant.py
python ../../main/plot_topk_v2.py
```

To reproduce the *measurements* from scratch you need the B200 + container of §2 and must rerun
steps 1→2→3 (rebuilding the kernel per §3) before step 4.

---

## 9. Honesty notes

- **Manuscript model (§6) calibration:** every coefficient is a standalone on-board microbenchmark of
  the kernel's own primitive plus a source-read atom count; **no whole-kernel latency and no in-kernel
  `FLASHINFER_ENABLE_PROFILER` timing is consumed as a calibration input.** Whole-kernel latencies are
  used only to score MAPE. (See `experiments/microbench/results/MEASUREMENT_NOTES.md`.)
- **Earlier profiler-based model (§5, `dsa_stage_model_v2.py`):** this *does* read the kernel's
  per-stage profiler spans as its coefficient source. It is an internal cross-check, not the paper's
  model; it coincidentally also scores 4.92% on the dense `topk=1024` grid. No coefficient in *either*
  model is fit to the target whole-kernel latencies.
- The modeled sparse-prefill kernel is byte-identical to upstream FlashMLA `main`; the local diff is
  computation-preserving profiler instrumentation, inert in the latency build
  (`experiments/main/PROVENANCE.md`). All scored whole-kernel grids were measured with the profiler
  **off** (clean production build).
- `latency_us = 40.4` in the run record is the **grid mean** (per-config range 19.3–80.9 µs);
  per-config values live in `grid_v2.csv`.
