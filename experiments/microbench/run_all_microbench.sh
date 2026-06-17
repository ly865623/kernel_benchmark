#!/bin/bash
# =============================================================================
# run_all_microbench.sh — one-key driver for the 5 calibration microbenchmarks
# that produce every constant consumed by dsa_stage_model_piecewise.py
# (the paper's head-line microbench-calibrated model, 3.88% MAPE).
#
# Each microbench measures ONE primitive cost on real B200 (sm_100) hardware:
#   1. MMA tensor-core  -> QK/SV throughput + single-op latency  (mma_costs.csv)
#   2. KV gather (TMA tile::gather4) -> scattered-gather BW       (gather4_scatter_bw.csv)
#   3. softmax exp2 (SFU MUFU.EX2) -> EX2 throughput              (exp2_sfu.csv)
#   4. correction FP32-ALU -> FMUL/FFMA rate                      (corr_fp32_alu.csv)
#   5. pipeline handshake (mbarrier) -> one-way signal latency    (sweep*.csv)
#
# IMPORTANT — provenance, do not misread:
#   * The atom *counts* (qk=18, sv=4/8 MMAs per tile, etc.) are NOT measured here;
#     they are read from the FlashMLA kernel source (config.h / phase1.cuh) and are
#     documented in results/MEASUREMENT_NOTES.md and sync_bench/NOTES.md.
#     These microbenches measure only the per-atom throughput / latency.
#   * Re-measuring requires real B200 (sm_100) + nvcc (CUDA 13.x Blackwell toolkit).
#     The committed reference CSVs already contain these numbers, so the paper's
#     3.88% model can be reproduced on CPU with NO GPU:
#         python dsa_stage_model_piecewise.py
#   * This script is NON-DESTRUCTIVE: raw stdout is written under ./regen/, it does
#     NOT overwrite the committed reference CSVs. Diff regen/ against the references
#     to confirm a match on your hardware.
#
# Dependencies are vendored in-repo under thirdparty/ (no external paths needed):
#   * thirdparty/FlashMLA            — pinned git submodule (gather4 reuses its TMA primitives)
#                                      populate with: git submodule update --init --recursive
#   * thirdparty/microbench-blackwell — vendored UMMA throughput/latency suite (MMA atoms)
#
# Usage:
#   bash run_all_microbench.sh              # build + run all 5 (needs B200)
#   FM=/path/to/FlashMLA bash run_all_microbench.sh                 # override FlashMLA location
#   MB_SUITE=/path/to/microbench-blackwell bash run_all_microbench.sh  # override MMA suite location
# =============================================================================
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
REGEN="$HERE/regen"
mkdir -p "$REGEN"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

# Dependency locations default to the in-repo thirdparty/ copies (override via env).
FM="${FM:-$REPO_ROOT/thirdparty/FlashMLA}"                       # FlashMLA submodule (gather4 reuses its TMA primitives)
MB_SUITE="${MB_SUITE:-$REPO_ROOT/thirdparty/microbench-blackwell}" # vendored UMMA throughput/latency suite (MMA atoms)
ARCH_F="${ARCH_F:-sm_100f}"   # gather4/exp2/correction
ARCH_A="${ARCH_A:-sm_100a}"   # mbar_pipeline

pass=0; fail=0; skip=0
ok()   { echo "  [OK]   $1"; pass=$((pass+1)); }
bad()  { echo "  [FAIL] $1"; fail=$((fail+1)); }
note() { echo "  [SKIP] $1"; skip=$((skip+1)); }

echo "=============================================================="
echo " DSA microbench calibration — regenerate all 5 primitive costs"
echo " regen dir : $REGEN"
echo " FlashMLA  : $FM"
echo " MMA suite : $MB_SUITE"
echo "=============================================================="

# ---- preflight -------------------------------------------------------------
if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found. The microbenches need the CUDA 13.x Blackwell toolkit."
  echo "       (The committed reference CSVs already hold the measured numbers; the"
  echo "        paper model runs on CPU via:  python dsa_stage_model_piecewise.py)"
  exit 2
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  echo "GPU detected: ${GPU:-unknown}"
  case "$GPU" in
    *B200*|*GB200*) : ;;
    *) echo "WARNING: GPU does not look like a B200/sm_100 part. Results may not match the paper." ;;
  esac
else
  echo "WARNING: nvidia-smi not found; cannot confirm B200. Continuing anyway."
fi
echo

# ---- 2. KV gather (TMA tile::gather4) -------------------------------------
echo "[2/5] gather4 — KV-latent scattered gather bandwidth"
if [ -d "$FM/csrc" ]; then
  if FM="$FM" bash "$HERE/gather4_bench/build_and_run.sh" 2>&1 | tee "$REGEN/gather4.raw.txt"; then
    grep '^RESULT' "$REGEN/gather4.raw.txt" > "$REGEN/gather4_scatter_bw.regen.csv" 2>/dev/null || true
    ok "gather4 -> regen/gather4_scatter_bw.regen.csv (ref: results/gather4_scatter_bw.csv)"
  else
    bad "gather4 build/run failed (see regen/gather4.raw.txt)"
  fi
else
  note "gather4 — FlashMLA source not found at FM=$FM (needed for ku::tma_gather4); set FM=..."
fi
echo

# ---- 3. softmax exp2 (SFU MUFU.EX2) ---------------------------------------
echo "[3/5] exp2 — register-resident SFU EX2 throughput"
if nvcc -O3 -arch="$ARCH_F" "$HERE/exp2_bench/exp2_tput.cu" -o "$HERE/exp2_bench/exp2_tput.out" 2> "$REGEN/exp2.build.log"; then
  if "$HERE/exp2_bench/exp2_tput.out" 2>&1 | tee "$REGEN/exp2.raw.txt"; then
    ok "exp2 -> regen/exp2.raw.txt (ref: results/exp2_sfu.csv)"
  else
    bad "exp2 run failed (see regen/exp2.raw.txt)"
  fi
else
  bad "exp2 build failed (see regen/exp2.build.log)"
fi
echo

# ---- 4. correction FP32-ALU -----------------------------------------------
echo "[4/5] correction — FP32 FMUL/FFMA rescale rate"
if bash "$HERE/correction_bench/build_and_run.sh" 2>&1 | tee "$REGEN/correction.raw.txt"; then
  ok "correction -> regen/correction.raw.txt (ref: correction_bench/corr_fp32_alu.csv)"
else
  bad "correction build/run failed (see regen/correction.raw.txt)"
fi
echo

# ---- 5. pipeline handshake (mbarrier) -------------------------------------
echo "[5/5] sync — mbarrier handshake latency sweep (S=2..7, B=4)"
if nvcc -O3 -std=c++17 -arch="$ARCH_A" "$HERE/sync_bench/mbar_pipeline.cu" -o "$HERE/sync_bench/mbar_pipeline.out" 2> "$REGEN/sync.build.log"; then
  echo "stages,bufs,total_cyc,per_iter_cyc,per_iter_ns" > "$REGEN/sweep.regen.csv"
  sync_ok=1
  for S in 2 3 4 5 6 7; do
    line="$("$HERE/sync_bench/mbar_pipeline.out" "$S" 4 100000 2>&1 | tee -a "$REGEN/sync.raw.txt" | grep '^CSV,')"
    if [ -n "$line" ]; then
      echo "${line#CSV,}" >> "$REGEN/sweep.regen.csv"
    else
      sync_ok=0
    fi
  done
  # S=2 ping-pong -> one-way mbarrier signal latency h = per_iter_ns/2 (paper: 140.33 ns)
  echo "  (S=2 per-iter / 2 = one-way handshake latency h; paper uses h=140.33 ns)"
  if [ "$sync_ok" = "1" ]; then
    ok "sync -> regen/sweep.regen.csv (ref: sync_bench/sweep.csv, sweep_S.csv, sweep_B.csv)"
  else
    bad "sync — some S configs produced no CSV line (see regen/sync.raw.txt)"
  fi
else
  bad "sync build failed (see regen/sync.build.log)"
fi
echo

# ---- 1. MMA tensor-core atoms (vendored UMMA suite) -----------------------
# Listed last only for output ordering; the suite is vendored in-repo under
# thirdparty/microbench-blackwell (self-contained CUDA, no CUTLASS needed).
echo "[1/5] MMA — UMMA QK/SV throughput + single-op latency  (thirdparty/microbench-blackwell)"
if [ -d "$MB_SUITE/umma_throughput" ]; then
  ( cd "$MB_SUITE/umma_throughput" && make >/dev/null 2>&1 && ./umma_tput.out ) 2>&1 | tee "$REGEN/umma_throughput.raw.txt" && \
    ok "umma_throughput -> regen/umma_throughput.raw.txt (ref: results/mma_costs.csv)" || \
    bad "umma_throughput build/run failed"
  if [ -d "$MB_SUITE/umma_latency" ]; then
    ( cd "$MB_SUITE/umma_latency" && make >/dev/null 2>&1 && ./umma_lat.out ) 2>&1 | tee "$REGEN/umma_latency.raw.txt" && \
      ok "umma_latency -> regen/umma_latency.raw.txt (ref: results/mma_costs.csv)" || \
      bad "umma_latency build/run failed"
  else
    note "umma_latency — not found under MB_SUITE=$MB_SUITE"
  fi
else
  note "MMA — UMMA suite not found at MB_SUITE=$MB_SUITE/umma_throughput; expected in-repo at thirdparty/microbench-blackwell (or set MB_SUITE=...)"
  echo "       (mma_costs.csv: QK=37.106 cyc/op SV=64.648 cyc/op; QK_lat=178 cyc SV_lat=210 cyc)"
fi
echo

echo "=============================================================="
echo " DONE.  passed=$pass  failed=$fail  skipped=$skip"
echo " Raw outputs + *.regen.csv are under: $REGEN"
echo " Compare against committed references, then verify the model:"
echo "     python dsa_stage_model_piecewise.py     # head-line 3.88% MAPE"
echo "=============================================================="
[ "$fail" -eq 0 ]
