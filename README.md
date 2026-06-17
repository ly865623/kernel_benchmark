# Stage-Centric Latency Model for the FlashMLA DSA Sparse-Prefill Kernel (B200)

End-to-end reproduction guide for the experimental results behind the paper
*"Stage-Centric Microbenchmark-Calibrated Latency Modeling of Sparse Attention Kernels on Blackwell GPUs"*.

**Headline result.** An interpretable, wave-quantized analytical model predicts the whole-kernel
latency of DeepSeek's DSA sparse-prefill kernel (`sparse_attn_fwd`) across **104 configurations**
spanning both selection budgets the kernel family dispatches (`topk ∈ {1024, 2048}`) with
**MAPE = 3.88%** overall (small `B_TOPK=64` kernel 4.92%, regular `B_TOPK=128` kernel 2.67%),
versus **47.6%** for a naive datasheet roofline — with **no coefficient fit to the target
whole-kernel latencies**. The headline model is produced (no GPU) by
`experiments/microbench/dsa_stage_model_piecewise.py` (§5–§7).

> **The scientific point of this repo is the calibration path.** Every model coefficient is a
> *standalone on-board microbenchmark of the kernel's own primitive* — the scattered `tile::gather4`
> KV gather, the register-resident softmax `exp2`, the `QK^T`/`SV` tensor-core matmul atoms, the FP32
> online-softmax correction, and a bare `mbarrier` cross-warp handshake — plus per-iteration atom
> *counts* read directly from the kernel source. **The fused kernel is never run to calibrate the
> model: no whole-kernel latency and no in-kernel profiler (`FLASHINFER_ENABLE_PROFILER`) timing is
> consumed as a calibration input.** On-device whole-kernel latencies appear *only* to score MAPE,
> and are measured from a clean profiler-off (production) build.
>
> ⚠️ **This is NOT the older profiler-based model.** An earlier, now *superseded*, stage model
> (`experiments/main/dsa_stage_model_v2.py`) instead composed per-stage spans read from the kernel's
> `FLASHINFER_ENABLE_PROFILER` instrumentation. It is retained **only as an internal cross-check** in
> Appendix A — it is **not** the manuscript's model or its calibration path. (It coincidentally also
> scores 4.92% on the dense `topk=1024` grid, a numeric coincidence with the microbench small-kernel
> 4.92%.)

---

## 0. TL;DR — what to run

```bash
cd <repo-root>

# (A) Reproduce the paper headline on CPU — reads committed microbench CSVs, NO GPU needed.
python experiments/microbench/dsa_stage_model_piecewise.py   # -> small 4.92% / regular 2.67% / both 3.88%

# (B) Regenerate the microbench calibration data from scratch — needs a real B200 (sm_100) + nvcc.
bash experiments/microbench/run_all_microbench.sh            # rebuilds + reruns all 5 primitive benches
```

The model in (A) is the manuscript headline. The five primitive costs it consumes are produced by
the microbenchmarks in (B); their reference outputs are committed, which is why (A) runs without a
GPU. **The rest of this README explains exactly how the microbench data in (B) is produced.**

---

## 1. What gets reproduced

The **primary path** is the microbenchmark-calibrated model (rows ★). Everything else is either the
ground truth it is *scored against*, or the legacy profiler-based cross-check in Appendix A.

| Path | Artifact | Produced by | Output |
|------|----------|-------------|--------|
| ★ **calib** | **5 primitive microbenchmark costs** (MMA, gather4, exp2, correction, handshake) | `experiments/microbench/run_all_microbench.sh` (§5) | `experiments/microbench/results/*.csv`, `*_bench/*.csv` |
| ★ **model** | **Manuscript headline** — microbench-calibrated, profiler-free piecewise model → MAPE **3.88%** | `experiments/microbench/dsa_stage_model_piecewise.py` (§7) | `experiments/microbench/piecewise_results.json` |
| score | Ground-truth baseline latency (10 configs) + naive roofline bar | `experiments/baseline/measure_dsa_prefill.py` (§6) | `baselines/local/flashmla-dsa-b200/` |
| score | top-k scan grid (top-k∈{1024,2048}, 104 configs) — the latencies the headline model is scored on | top-k sweep (§6) | `experiments/analysis-results/topk_scan_v2/json/grid_topk_v2.json` |
| score | Dense whole-kernel latency grid (56 configs, topk=1024) | `experiments/main/sweep_grid_v2.py` (§6) | `experiments/main/grid_v2/` |
| fig | Paper-facing top-k figure | `experiments/main/plot_topk_v2.py` (§6) | `experiments/analysis-results/topk_scan_v2/topk_scan_latency_skv8192.png` |
| A (legacy) | Per-stage profiler decomposition (17 pipeline stages) | `experiments/main/stage_extract.py` (App. A) | `experiments/main/stages/` |
| A (legacy) | _Earlier profiler-based_ stage model → 4.92% (**internal cross-check, NOT the headline**) | `experiments/main/dsa_stage_model_v2.py` (App. A) | `experiments/main/stage_model_v2_results.json` |

---

## 2. Hardware & environment

| Item | Value |
|------|-------|
| GPU | NVIDIA B200 (sm100), 148 SMs, SM clock 1.965 GHz |
| Container | `nvcr.io/nvidia/pytorch:26.01-py3-v0` (referred to as `ds003-flashmla`) |
| Framework | PyTorch 2.10.0a0 (NGC 26.01), CUDA 13.1 |
| Compiler | `nvcc` (CUDA 13.x Blackwell toolkit); microbench arches `sm_100f` / `sm_100a` |
| Kernel source | `deepseek-ai/FlashMLA`, commit `48c6dc4` (byte-identical to upstream `main` for the modeled sparse-prefill path — see `experiments/main/PROVENANCE.md`) |

**What needs a GPU:** the microbenchmark *measurements* (§5) and the ground-truth latency grids (§6)
require a real B200 (sm100). **What does not:** the headline model and all evaluation/figure scripts
(§7–§8) are pure post-processing of committed JSON/CSV and run anywhere with Python 3 + NumPy/Matplotlib.

The on-GPU scripts locate the FlashMLA checkout through the `FLASHMLA_ROOT` (measurement harness) and
`FM` (gather4 microbench) environment variables (default `/workspace/code/FlashMLA` and
`/home/liuy/code/FlashMLA` respectively) and reuse its test harness / primitives unmodified.

---

## 3. Step 0 — Build the kernel (GPU paths only)

A single **clean / production build** of FlashMLA (no profiler) is all the primary path needs — it is
used for the ground-truth latency grids of §6, and the `gather4` microbench of §5 reuses its TMA
headers. The profiler build is only required for the legacy Appendix A cross-check.

```bash
# inside the container, with the GPU visible
export FLASHMLA_ROOT=/workspace/code/FlashMLA
cd $FLASHMLA_ROOT
git checkout 48c6dc4
python setup.py install        # or the repo's documented build entrypoint
```

> The exact upstream build invocation follows FlashMLA's own README. The generated machine code for
> the modeled sparse-prefill path is identical to stock upstream; full audit in
> `experiments/main/PROVENANCE.md`. (Appendix A additionally needs a
> `FLASHINFER_ENABLE_PROFILER=1` rebuild; the profiler macros are compile-gated and expand to nothing
> in the clean build, so they do not alter the measured computation.)

---

## 4. Reproduction at a glance

```
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  PRIMARY (microbenchmark-calibrated) path                                 │
   │                                                                           │
   │  §5  5 primitive microbenchmarks   ── measure ──▶  5 reference CSVs        │
   │      (run_all_microbench.sh)                       (the model's coeffs)    │
   │                                                          │                 │
   │  §6  ground-truth latency grids    ── measure ──▶  grid JSON (for scoring) │
   │      (baseline + top-k sweep)                            │                 │
   │                                                          ▼                 │
   │  §7  piecewise model  ── reads CSVs + grid JSON ──▶  MAPE 3.88%  (no GPU)  │
   └─────────────────────────────────────────────────────────────────────────┘
   Appendix A: legacy profiler-based stage model (cross-check only, NOT the paper).
```

---

## 5. ★ Microbenchmark calibration data — how the model's coefficients are generated

**This is the core of the repo.** The §7 headline model consumes **no** whole-kernel or profiler
timing; every coefficient is one of **five standalone microbenchmarks** of the kernel's own
primitives. The per-iteration atom *counts* (`qk_atoms=18`, `sv_atoms=4/8`) are **read from the
kernel source** (`config.h`, `phase1.cuh`), not measured.

### 5.1 One-key driver

```bash
# from the repo root; needs B200 (sm_100) + nvcc. Non-destructive: raw output goes to
# experiments/microbench/regen/, the committed reference CSVs are NEVER overwritten.
bash experiments/microbench/run_all_microbench.sh

# override external dependency locations if needed:
FM=/path/to/FlashMLA MB_SUITE=/path/to/microbench-blackwell \
    bash experiments/microbench/run_all_microbench.sh
```

The driver builds and runs all five microbenchmarks, tees each to `regen/`, and prints a
pass/fail/skip summary. To confirm a match on your hardware, diff `regen/*.regen.csv` (and the raw
`regen/*.raw.txt`) against the committed reference CSVs. A live-vs-committed verification table from a
real B200 run is in `experiments/microbench/REGEN_VERIFICATION.md`.

> If `nvcc`/B200 is unavailable the driver exits early with a clear message — the committed reference
> CSVs already hold the measured numbers, so the paper model still reproduces on CPU (§8).

### 5.2 The five microbenchmarks (source → command → committed CSV → model constant)

Each bench can also be built and run individually; the driver just wraps these.

| # | Primitive → model constant (used in `dsa_stage_model_piecewise.py`) | Source + standalone command | Committed output |
|---|---|---|---|
| 1 | **MMA tensor-core** QK/SV throughput + single-op latency → `MMA_N128_CYC=37.106`/`LAT=178`, `MMA_N256_CYC=64.648`/`LAT=210` (cyc) | external UMMA suite `${MB_SUITE}/umma_throughput` (+ `umma_latency`): `make && ./umma_tput.out` | `experiments/microbench/results/mma_costs.csv` |
| 2 | **Scattered KV gather** (`tile::gather4`) BW → `GATHER_NS_64≈16.2` ns / 64-token block (~2.0 TB/s) | `experiments/microbench/gather4_bench/build_and_run.sh` (reuses FlashMLA's own `ku::tma_gather4`; needs `FM=…/FlashMLA`, arch `sm_100f`) | `experiments/microbench/results/gather4_scatter_bw.csv` |
| 3 | **Softmax `exp2`** register-resident `MUFU.EX2` SFU throughput → `EXP2_OPS_PER_S_PER_SM=24.8e9` | `nvcc -O3 -arch=sm_100f experiments/microbench/exp2_bench/exp2_tput.cu -o exp2_tput.out && ./exp2_tput.out` | `experiments/microbench/results/exp2_sfu.csv` |
| 4 | **FP32-ALU correction** FMUL/FFMA rate for the online-softmax rescale → `FP32_MUL_OPS=176.567e9`, `FP32_FFMA_OPS=174.600e9` | `experiments/microbench/correction_bench/build_and_run.sh` (pure CUDA, arch `sm_100f`) | `experiments/microbench/correction_bench/corr_fp32_alu.csv` |
| 5 | **Pipeline handshake** bare `mbarrier` cross-warp one-way signal latency → `H_ONEWAY_NS=140.33` (= 280.66 / 2) | `nvcc -O3 -std=c++17 -arch=sm_100a experiments/microbench/sync_bench/mbar_pipeline.cu -o mbar_pipeline.out && ./mbar_pipeline.out <stages> 4 100000` | `experiments/microbench/sync_bench/sweep*.csv` |

Notes on the two "reuse-the-kernel's-own-primitive" benches:

- **MMA (#1)** lives in a *separate* UMMA throughput/latency suite (`microbench-blackwell`), hence the
  `MB_SUITE` env var; only its measured per-op cost is consumed (`mma_costs.csv`). It measures the
  tensor-core matmul atoms at the kernel's exact instruction shapes (QK: M128·N128·K16 BF16, 36
  K-steps; SV: M128·N256·K16 BF16, 8 K-steps).
- **gather4 (#2)** compiles FlashMLA's *own* `ku::tma_gather4` / CUTLASS headers at the kernel's exact
  gather shape, so it measures the same hardware path the fused kernel pays — not a datasheet proxy.
- **exp2 (#3)**, **correction (#4)**, and **handshake (#5)** are self-contained CUDA.

### 5.3 What the numbers mean

The full measurement log, the per-op atom-count derivation, and the key findings — the scattered
gather is *cheap* (16–32 ns/block); the overlapped per-iteration binder is the **`QK^T` tensor-core
matmul** (not the gather, overturning the naive gather-bound expectation); the four isolated ops
account for <40% of per-iter time, the rest being the predicted synchronization residual set by the
handshake floor — are written up in `experiments/microbench/results/MEASUREMENT_NOTES.md` and
`experiments/microbench/sync_bench/NOTES.md`.

---

## 6. Ground-truth latency grids (measured only to *score* the model)

These supply the whole-kernel latencies the §7 model is graded against. They are **never** fed back as
calibration inputs. All were measured with a clean profiler-off (production) build.

### 6.1 Baseline ground truth + roofline (10 configs)

```bash
export FLASHMLA_ROOT=/workspace/code/FlashMLA   # clean build
python experiments/baseline/measure_dsa_prefill.py \
    --out baselines/local/flashmla-dsa-b200 --num-tests 30 --passes 3
```
- **Outputs:** `baselines/local/flashmla-dsa-b200/json/ground_truth_prefill.json`, `…csv`, `sweep.log`
- **Comparison contract (canonical):** `baselines/local/flashmla-dsa-b200/json/metric_contract.json`
  (primary `latency_us`; supporting `tflops`, `mem_bw_tbps`, `roofline_rel_err`).

### 6.2 Top-k scan grid (104 configs) — the headline scoring set

`topk ∈ {1024, 2048}` under the kernel's `B_TOPK` block constraint (multiple of 128, `topk ≤ s_kv`),
giving 104 effective configs. `topk=2048` dispatches a *different* compiled kernel
(`topk > 1280` → regular `B_TOPK=128`; `topk ≤ 1280` → small `B_TOPK=64`; see `csrc/api/sparse_fwd.h`).
The measured grid is committed (`grid_topk_v2.json`), so §7 scoring runs without a GPU.

- **Committed ground truth:** `experiments/analysis-results/topk_scan_v2/json/grid_topk_v2.json`
- **Provenance:** `experiments/analysis-results/topk_scan_v2/PROVENANCE.md`

### 6.3 Dense 56-config grid (topk=1024)

```bash
export FLASHMLA_ROOT=/workspace/code/FlashMLA   # clean build
python experiments/main/sweep_grid_v2.py --out experiments/main/grid_v2 \
    --num-tests 30 --passes 3 --topk 1024 --d-qk 576
# quick check first: add --smoke for a single tiny config
```
`batch_size (= s_q) ∈ {1,32,64,74,128,148,256,296} × s_kv ∈ {1k…128k}`, fixed
`h_q=128, d_qk=576, d_v=512, topk=1024, attn_sink=True`. Batch points sit on the wave boundary
(74 two-SM tiles = 1 wave on 148 SMs): `{74,148,296}` = 1/2/4 full waves.
- **Outputs:** `experiments/main/grid_v2/json/grid_v2.json`, `grid_v2.csv`
- Used to score the dense profiler-free model `dsa_stage_model_microbench.py` (≈3.75%) and the legacy
  Appendix-A model.

### 6.4 Paper figure

```bash
python experiments/main/plot_topk_v2.py   # -> experiments/analysis-results/topk_scan_v2/topk_scan_latency_skv8192.png
```

---

## 7. ★ Headline model — microbench-calibrated piecewise model (104 configs → 3.88%)

```bash
# reads the committed microbench CSVs (§5) + grid_topk_v2.json (§6.2); NO GPU.
python experiments/microbench/dsa_stage_model_piecewise.py   # -> experiments/microbench/piecewise_results.json
```
- **Result:** small `B_TOPK=64` kernel (topk=1024) **4.92%** · regular `B_TOPK=128` kernel (topk=2048)
  **2.67%** · both (104 cfg) **3.88%**.
- **Model (interpretable, closed form):** `num_tiles = s_q`; `num_waves = ceil(num_tiles/74)`;
  `T_pred = T_launch + num_waves · T_wave(fill)`, with the per-iteration step time built from the §5
  primitive costs (QK-matmul-bound binder + gather/exp2/correction overlapped + handshake residual).
- **The correction stage** (`t_sm_reduce + t_correction`, microbenchmarked in §5 #4) is added to
  `t_scoring` **only for the regular `B_TOPK=128` kernel**, where it is pipeline-exposed; in the small
  `B_TOPK=64` kernel it is pipeline-covered. Adding it is what closes the regular-kernel gap from a
  no-correction 7.39% down to 2.67%.

Reference variants (not the headline; for ablation/context only):
```bash
python experiments/microbench/dsa_stage_model_microbench.py            # profiler-free dense 56-cfg model (~3.75%)
cd experiments/analysis-results/topk_scan_v2
python eval_model_on_topk.py        # published v2 model, split by top-k, regular kernel un-corrected
python eval_two_kernel.py           # two-kernel WITHOUT correction: small 4.83% / regular 7.39% / both 6.01%
python eval_amortized_variant.py    # fixed-cost amortization correction variant
```
- **Key finding:** `latency(2048)/latency(1024) = 1.61×` (sub-2×; fixed-overhead amortization).

> These eval scripts import the published model module verbatim and currently use **absolute paths**
> into the original worktree. If you relocate the repo, update the `MAIN`/`HERE`/`TOPK` path constants
> at the top of each script, or run them from this directory.

---

## 8. Minimal reproduction (no GPU)

Regenerate every model result and figure from the committed measurement data:

```bash
cd <repo-root>
# *** PAPER HEADLINE *** microbench-calibrated, profiler-free piecewise model -> 3.88% (104 cfg)
python experiments/microbench/dsa_stage_model_piecewise.py     # small 4.92% / regular 2.67% / both 3.88%
python experiments/microbench/dsa_stage_model_microbench.py    # profiler-free dense 56-cfg model (~3.75%)
# reference variants + figure:
cd experiments/analysis-results/topk_scan_v2
python eval_model_on_topk.py && python eval_two_kernel.py && python eval_amortized_variant.py
python ../../main/plot_topk_v2.py
```

To reproduce the *measurements* from scratch you need the B200 + container of §2: run the five
microbenchmarks (§5) and the ground-truth grids (§6), rebuilding the kernel per §3.

---

## 9. Repository map

```
README.md                         <- this file
brief.md / plan.md / PLAN.md      <- quest framing, research plan, node contract
SUMMARY.md / status.md            <- durable quest summary & status

baselines/
  local/flashmla-dsa-b200/        <- measured ground truth + metric_contract.json (canonical comparison)
  imported/flashmla-dsa-b200/     <- attachment record (attachment.yaml)

experiments/
  microbench/                     <- ★ PRIMARY: the manuscript model + its standalone microbenchmarks (profiler-free)
    run_all_microbench.sh         <- ★ one-key driver: rebuild+rerun all 5 calibration microbenchmarks
    REGEN_VERIFICATION.md         <- live-B200 vs committed verification table (read this)
    dsa_stage_model_piecewise.py  <- *** PAPER HEADLINE *** microbench-calibrated piecewise model -> 3.88%
    dsa_stage_model_microbench.py <- profiler-free dense 56-config model (~3.75%)
    gather4_bench/ exp2_bench/ correction_bench/ sync_bench/  <- the 5 standalone on-board op microbenchmarks
    results/                      <- mma_costs.csv, gather4_scatter_bw.csv, exp2_sfu.csv, MEASUREMENT_NOTES.md
  baseline/measure_dsa_prefill.py <- §6.1: ground-truth measurement harness
  analysis-results/topk_scan_v2/  <- §6.2: top-k grid (committed ground truth), eval_*.py, PROVENANCE.md, figure
  main/
    sweep_grid_v2.py              <- §6.3: 56-config whole-kernel latency sweep
    plot_topk_v2.py               <- §6.4: top-k figure renderer
    grid_v2/                      <- §6.3 outputs (json/grid_v2.json, grid_v2.csv)
    PROVENANCE.md                 <- upstream-equivalence + profiler audit (read this)
    stage_extract.py              <- App. A: per-stage profiler decomposition (legacy)
    dsa_stage_model_v2.py         <- App. A: EARLIER profiler-based stage model (cross-check, NOT the headline)
    dsa_predictor.py              <- App. A: standalone reference predictor (cross-check)
    stages/                       <- App. A profiler outputs

paper/                            <- outline, evidence ledger, manuscript bundle (LaTeX under paper/latex/)
```

---

## Appendix A. Legacy profiler-based stage model (internal cross-check — NOT the paper)

> Retained for transparency only. This path reads the kernel's per-stage
> `FLASHINFER_ENABLE_PROFILER` spans as its coefficient source — the manuscript model (§7) instead
> draws every coefficient from the standalone microbenchmarks of §5 and consumes **no** profiler
> timing. The profiler-free equivalent of this dense-grid model is
> `experiments/microbench/dsa_stage_model_microbench.py` (§7), which is the version aligned with the
> paper's calibration contract.

**A.1 Per-stage profiler decomposition** — needs a `FLASHINFER_ENABLE_PROFILER=1` rebuild (§3):
```bash
export FLASHMLA_ROOT=/workspace/code/FlashMLA   # PROFILER build
python experiments/main/stage_extract.py --out experiments/main/stages --full-grid   # --smoke for 1 config
```
Decodes the on-chip `%globaltimer` (ns) spans of the kernel's 17 pipeline stages →
`experiments/main/stages/json/stage_timings.json`.

**A.2 Earlier profiler-based stage model** (no GPU; reads A.1 + §6.3 grid):
```bash
python experiments/main/dsa_stage_model_v2.py
```
- **Reads:** `experiments/main/grid_v2/json/grid_v2.json`, `experiments/main/stages/json/stage_timings.json`
- **Outputs:** `experiments/main/stage_model_v2_results.json`, `predictions.json`
- **Expected metrics:** `mape_pct = 4.92`, `mape_pct_anchored = 4.93`, `worst_abs_pct_err = 11.32`,
  `roofline_mape_pct = 50.28`, `n_configs = 56`, `improvement_vs_roofline_x = 10.2`.
- The recorded run record is `experiments/main/dsa-stagepred-grid-v2/RUN.md` / `RESULT.json`.

The profiler-era reading (gather/softmax-heavy loop) is **superseded** by the §5 microbench analysis,
which shows the gather is cheap and the per-iteration cost is `QK^T`-matmul-bound. No coefficient in
*either* model is fit to the target whole-kernel latencies.

---

## Honesty notes

- **Manuscript model (§5/§7) calibration:** every coefficient is a standalone on-board microbenchmark
  of the kernel's own primitive plus a source-read atom count; **no whole-kernel latency and no
  in-kernel `FLASHINFER_ENABLE_PROFILER` timing is consumed as a calibration input.** Whole-kernel
  latencies are used only to score MAPE. (See `experiments/microbench/results/MEASUREMENT_NOTES.md`
  and `REGEN_VERIFICATION.md`.)
- **Legacy profiler-based model (Appendix A, `dsa_stage_model_v2.py`):** this *does* read the kernel's
  per-stage profiler spans as its coefficient source. It is an internal cross-check, not the paper's
  model; it coincidentally also scores 4.92% on the dense `topk=1024` grid.
- The modeled sparse-prefill kernel is byte-identical to upstream FlashMLA `main`; the local diff is
  computation-preserving profiler instrumentation, inert in the latency build
  (`experiments/main/PROVENANCE.md`). All scored whole-kernel grids were measured with the profiler
  **off** (clean production build).
- `latency_us = 40.4` in the run record is the **grid mean** (per-config range 19.3–80.9 µs);
  per-config values live in `grid_v2.csv`.
