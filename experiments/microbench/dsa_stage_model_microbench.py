#!/usr/bin/env python3
"""
Microbench-driven, stage-centric wave-quantized latency model for the FlashMLA DSA
sparse-prefill kernel (head128) on B200.

EVERY per-stage coefficient comes from a STANDALONE ON-BOARD microbenchmark of the kernel's
own primitives (NOT from FLASHINFER_ENABLE_PROFILER in-kernel markers, which the user rejected,
and NOT fit to the whole-kernel target latencies). Validation target is the real on-board
56-config whole-kernel grid grid_v2.json (CUDA-event timed, the same ground truth the v2 model used).

On-board microbench coefficients (all measured on B200 @ 1965 MHz; see results/MEASUREMENT_NOTES.md):
  - QK UMMA atom (M128 N128 K16, bf16, 2x1SM/TS) : 37.106 cyc/atom   [results/mma_costs.csv]
  - SV UMMA atom (M128 N256 K16, bf16, 2x1SM/SS) : 64.648 cyc/atom   [results/mma_costs.csv]
  - scattered KV gather4 (kernel's own ku::tma_gather4): 16.2 ns per 64-token KV block
                                                                     [results/gather4_scatter_bw.csv]
  - softmax exp2f SFU (register-resident): 24.8 GOps/s/SM            [results/exp2_sfu.csv]
  - contiguous TMA (q_load / epilogue O-store) BW : 3.75 TB/s        [results/gather_bw_occupancy.csv]

Per-k-iteration MMA atom counts are taken from the kernel source (config.h + phase1.cuh), 2x1SM
cooperative, per-CTA stream:
  - QK: TiledMMA_P = SM100_MMA_2x1SM<M=H_Q=128, N=B_TOPK*2=128>, contraction D_Q/2=288 per CTA
        -> 288/16 = 18 atoms / k-iter
  - SV: TiledMMA_O = SM100_MMA_2x1SM<M=H_Q=128, N=256>, per-CTA N=D_V/2=256, K=B_TOPK=64
        -> (256/256)*(64/16) = 4 atoms / k-iter
  - k_tiles = ceil(topk / B_TOPK), B_TOPK = 64 (the kernel's real KV block; the grid's k_tiles
    field uses /128 and is NOT the kernel's per-iter count).

Wave quantization (from grid_v2 'fixed'/structure): one 2-SM tile per query row; 74 two-SM tiles
per wave on a 148-SM B200; num_waves = ceil(num_tiles / 74).
"""
import json, math, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(os.path.dirname(HERE), "main", "grid_v2", "json", "grid_v2.json")

# ---------------- on-board microbench coefficients ----------------
CLK_GHZ        = 1.965                  # B200 max SM clock, measured
NS_PER_CYC     = 1.0 / CLK_GHZ          # 0.5089 ns/cyc
QK_CYC_ATOM    = 37.106                 # cyc per M128N128K16 atom (on-board)
SV_CYC_ATOM    = 64.648                 # cyc per M128N256K16 atom (on-board)
GATHER_NS_BLK  = 16.2                   # ns per 64-token KV block (on-board scattered gather4)
EXP2_OPS_PER_S_PER_SM = 24.8e9          # exp2 SFU rate per SM (on-board)
TMA_BW_TBPS    = 3.75                   # contiguous TMA BW (on-board), for q_load/epilogue

# ---------------- kernel structural constants (from source) ----------------
H_Q, D_Q, D_V, B_TOPK = 128, 576, 512, 64
QK_ATOMS = (D_Q // 2) // 16             # 18 atoms / k-iter (2x1SM, K split per CTA)
SV_ATOMS = (D_V // 2 // 256) * (B_TOPK // 16)  # 4 atoms / k-iter (per-CTA N256, K=B_TOPK)
TILES_PER_WAVE = 74

# ---------------- per-k-iteration on-board op costs (ns) ----------------
t_qk      = QK_ATOMS * QK_CYC_ATOM * NS_PER_CYC          # tensor-core QK time / k-iter
t_sv      = SV_ATOMS * SV_CYC_ATOM * NS_PER_CYC          # tensor-core SV time / k-iter
t_tensor  = t_qk + t_sv                                   # QK+SV share the SAME tensor cores -> serial
t_gather  = GATHER_NS_BLK                                 # TMA gather of one 64-token KV block / k-iter
t_softmax = (H_Q // 2) * B_TOPK / EXP2_OPS_PER_S_PER_SM * 1e9   # 4096 exp2 / k-iter / CTA

# once-per-tile contiguous TMA (amortized over the tile, not per k-iter)
T_pro = H_Q * D_Q * 2 / (TMA_BW_TBPS * 1e12) * 1e9        # Q[128x576] bf16 load (ns)
T_epi = H_Q * D_V * 2 / (TMA_BW_TBPS * 1e12) * 1e9        # O[128x512] bf16 store (ns)

# overlapped steady-state step time: the 3 engines (tensor / TMA-gather / SFU) run concurrently
t_step_overlap = max(t_tensor, t_gather, t_softmax)       # ns / k-iter
t_step_serial  = t_tensor + t_gather + t_softmax          # ns / k-iter (no overlap, upper bound)
binder = max((("tensor", t_tensor), ("gather", t_gather), ("softmax", t_softmax)),
             key=lambda kv: kv[1])[0]

print("=== on-board per-k-iter op costs (ns) ===")
print(f"  QK tensor   = {t_qk:7.2f}  ({QK_ATOMS} atoms x {QK_CYC_ATOM} cyc)")
print(f"  SV tensor   = {t_sv:7.2f}  ({SV_ATOMS} atoms x {SV_CYC_ATOM} cyc)")
print(f"  tensor(QK+SV)= {t_tensor:7.2f}")
print(f"  gather      = {t_gather:7.2f}")
print(f"  softmax exp2= {t_softmax:7.2f}")
print(f"  -> overlapped t_step = max = {t_step_overlap:7.2f} ns  (binder: {binder})")
print(f"  -> serial    t_step = sum = {t_step_serial:7.2f} ns")
print(f"  T_pro = {T_pro:.2f} ns ; T_epi = {T_epi:.2f} ns")

# ---------------- load validation target ----------------
g = json.load(open(GRID))
rows = g["rows"]
meas = {(r["batch_size"], r["s_kv"], r["topk"]): r["latency_us"] for r in rows}

def num_waves(batch):
    return max(1, math.ceil(batch / TILES_PER_WAVE))

def k_tiles(topk):
    return max(1, math.ceil(topk / B_TOPK))

# clean single-tile (bs=1) wave time, averaged over s_kv -> the ONE honest anchor for the
# warp-specialized pipeline's sync/dependency residual (everything the 4 core ops don't capture).
bs1 = [r["latency_us"] for r in rows if r["batch_size"] == 1]
T_wave_anchor = statistics.mean(bs1)           # us, = measured 1-wave time at topk=1024
KT_REF = k_tiles(1024)                          # 16

# the residual sync/overhead per k-iter implied by the anchor (microbench-only cannot supply it)
core_wave_us = (T_pro + KT_REF * t_step_overlap + T_epi) / 1e3
overhead_per_kiter_ns = (T_wave_anchor * 1e3 - (T_pro + T_epi)) / KT_REF - t_step_overlap
print(f"\n=== anchor ===")
print(f"  measured bs=1 wave   = {T_wave_anchor:.3f} us")
print(f"  microbench core wave = {core_wave_us:.3f} us  (T_pro + 16*t_step_overlap + T_epi)")
print(f"  core-ops fraction of real per-iter time = {core_wave_us/T_wave_anchor*100:.1f}%")
print(f"  => sync/overlap residual per k-iter = {overhead_per_kiter_ns:.1f} ns")

def predict(batch, s_kv, topk, mode):
    nw = num_waves(batch)
    kt = k_tiles(topk)
    if mode == "micro_overlap":      # zero-fit: only microbench op costs, overlapped
        t_wave = (T_pro + kt * t_step_overlap + T_epi) / 1e3
    elif mode == "micro_serial":     # zero-fit: only microbench op costs, fully serialized
        t_wave = (T_pro + kt * t_step_serial + T_epi) / 1e3
    elif mode == "anchored":         # microbench op-scaling + ONE bs=1 anchor for sync residual
        t_wave = (T_pro + kt * (t_step_overlap + overhead_per_kiter_ns) + T_epi) / 1e3
    return nw * t_wave

def evaluate(mode):
    errs, out = [], []
    for r in rows:
        m = r["latency_us"]
        p = predict(r["batch_size"], r["s_kv"], r["topk"], mode)
        e = abs(p - m) / m
        errs.append(e)
        out.append({"batch_size": r["batch_size"], "s_kv": r["s_kv"], "num_waves": r["num_waves"],
                    "wave_fill": round(r["wave_fill"], 3), "measured_us": round(m, 3),
                    "pred_us": round(p, 3), "abs_pct_err": round(e * 100, 2)})
    return statistics.mean(errs) * 100, max(errs) * 100, out

rfl_mape = statistics.mean(abs(r["roofline_rel_err"]) for r in rows) * 100

print("\n=== MODEL EVALUATION (56 configs; topk=1024 fixed) ===")
results = {}
for mode in ["micro_overlap", "micro_serial", "anchored"]:
    mape, worst, out = evaluate(mode)
    results[mode] = {"mape_pct": mape, "worst_abs_pct_err": worst, "per_config": out}
    print(f"[{mode:14s}] MAPE = {mape:6.2f}%  worst = {worst:6.2f}%")
print(f"[{'roofline':14s}] MAPE = {rfl_mape:6.2f}%  (reference baseline)")

print("\nbatch s_kv   waves fill   meas_us  pred_us(anchored)  err%")
for o in results["anchored"]["per_config"]:
    print(f"{o['batch_size']:>5} {o['s_kv']:>7} {o['num_waves']:>5} {o['wave_fill']:<5} "
          f"{o['measured_us']:>8} {o['pred_us']:>10}        {o['abs_pct_err']:>5}")

summary = {
    "model": "stage-centric-wave-quantized-dsa-microbench",
    "method": ("T_pred = num_waves * (T_pro + k_tiles*t_step + T_epi); ALL per-stage costs from "
               "standalone on-board microbenchmarks of the kernel's own primitives (no profiler, "
               "no fit to whole-kernel latencies). Wave quantization on 74 two-SM tiles/wave."),
    "clock_ghz": CLK_GHZ, "tiles_per_wave": TILES_PER_WAVE,
    "kernel_const": {"H_Q": H_Q, "D_Q": D_Q, "D_V": D_V, "B_TOPK": B_TOPK,
                     "qk_atoms_per_kiter": QK_ATOMS, "sv_atoms_per_kiter": SV_ATOMS},
    "per_kiter_ns": {"qk": t_qk, "sv": t_sv, "tensor": t_tensor, "gather": t_gather,
                     "softmax": t_softmax, "t_step_overlap": t_step_overlap,
                     "t_step_serial": t_step_serial, "binder": binder},
    "tile_amortized_ns": {"T_pro": T_pro, "T_epi": T_epi},
    "anchor": {"bs1_wave_us": T_wave_anchor, "k_tiles_ref": KT_REF,
               "core_ops_fraction_pct": core_wave_us / T_wave_anchor * 100,
               "sync_overhead_per_kiter_ns": overhead_per_kiter_ns},
    "roofline_mape_pct": rfl_mape,
    "results": {k: {"mape_pct": v["mape_pct"], "worst_abs_pct_err": v["worst_abs_pct_err"]}
                for k, v in results.items()},
    "per_config_anchored": results["anchored"]["per_config"],
    "per_config_micro_overlap": results["micro_overlap"]["per_config"],
}
outp = os.path.join(HERE, "stage_model_microbench_results.json")
json.dump(summary, open(outp, "w"), indent=2)
print("\nWROTE", outp)
