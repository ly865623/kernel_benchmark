#!/usr/bin/env python3
"""
CORRECTION-stage extension of the constraint-compliant composed model.

Motivation (user req #11, 2026-06-16): the composed model omits the online-softmax
`correction` warpgroup stage. The FlashMLA sm100 mainloop is a multi-stage consumer
pipeline  mma -> softmax0/softmax1 -> correction:
  * softmax_step  : rowmax fmax-reduce, FMA-scale (L652), exp2, rowsum add-reduce  -> O(B_TOPK)
  * correction    : acc_scale=0.5*exp2(...) (L692); O-accumulator FMUL2 rescale loop over
                    get<2>(TileShape)=D_V (L835-851)                               -> O(D_V)
The composed model counts ONLY exp2 in softmax and NOTHING for correction. Per-k-iter
sync-residual vs the (forbidden) oracle: small -17ns (ok); regular -258ns (-22%).

This script adds the missing consumer compute FROM PRIMITIVES/DATASHEET + source structure
(NO whole-kernel fit, NO grid fitting) and tests several principled pipeline formulations,
printing the full per-stage decomposition + signed errors so the mechanism is inspectable.

Strong constraint preserved: forbidden single-row whole-kernel anchor stays REFERENCE only.
"""
import json, math, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.normpath(os.path.join(os.path.dirname(HERE), "analysis-results",
                                     "topk_scan_v2", "json", "grid_topk_v2.json"))

# ---------------- on-board / datasheet primitives (B200 @ 1.965 GHz) ----------------
CLK_GHZ      = 1.965
NS_PER_CYC   = 1.0 / CLK_GHZ
MMA_N128_CYC = 37.106; MMA_N256_CYC = 64.648
MMA_N128_LAT = 178.0;  MMA_N256_LAT = 210.0
GATHER_NS_64 = 16.2
EXP2_OPS_PER_S_PER_SM = 24.8e9                 # measured exp2 SFU rate (exp2_sfu.csv)
TMA_BW_TBPS  = 3.75
H_ONEWAY_NS  = 280.66 / 2.0                    # 140.33 ns one-way cross-warp mbarrier (sync_bench)

# CUDA-core FP32 ALU throughput per SM. MEASURED ON-BOARD (user req #12): register-resident
# FP32 multiply/FFMA microbench (correction_bench/corr_rescale_tput.cu, corr_fp32_alu.csv, B200):
#   mul  = 176.567 Gops/SM   (the exact FMUL the O-rescale loop issues)
#   ffma = 174.600 Gops/SM   (softmax rowmax/rowsum reduce + scale FMA)
# The achieved register-resident rate is ~70% of the 251.5 Gops/SM datasheet peak (128 lanes x clk),
# so the datasheet constant UNDER-costed correction. Use the measured FMUL rate for the O-rescale and
# the measured FFMA rate for the softmax reductions -- both standalone on-board, no whole-kernel fit.
FP32_MUL_OPS_PER_S_PER_SM  = 176.567e9            # on-board, corr_fp32_alu.csv (mul)
FP32_FFMA_OPS_PER_S_PER_SM = 174.600e9            # on-board, corr_fp32_alu.csv (ffma)
DATASHEET_FP32_PEAK_PER_SM = 128 * CLK_GHZ * 1e9  # 251.5e9 -- reference only, not used in predict

H_Q, D_Q, D_V = 128, 576, 512
TILES_PER_WAVE = 74
# 2x1SM split: scoring/output path covers D_V/2 columns per CTA; softmax handles H_Q/2 rows/warp.
ROWS_SOFTMAX = H_Q // 2          # 64 rows per softmax warp (matches composed model's exp2 count)
ROWS_CORR    = H_Q              # correction partition_fragment_C(PV) = 128 rows
COLS_CORR    = D_V // 2         # 256 cols per CTA (2-CTA split on D_V)

KERNELS = {
    1024: dict(B_TOPK=64,  qk_atom_cyc=MMA_N128_CYC, qk_atom_lat=MMA_N128_LAT, qk_atoms=18, sv_atoms=64 // 16),
    2048: dict(B_TOPK=128, qk_atom_cyc=MMA_N256_CYC, qk_atom_lat=MMA_N256_LAT, qk_atoms=18, sv_atoms=128 // 16),
}

T_pro = H_Q * D_Q * 2 / (TMA_BW_TBPS * 1e12) * 1e9
T_epi = H_Q * D_V * 2 / (TMA_BW_TBPS * 1e12) * 1e9


def num_waves(b): return max(1, math.ceil(b / TILES_PER_WAVE))
def k_tiles(tk):  return max(1, math.ceil(tk / KERNELS[tk]["B_TOPK"]))
def chain_lat(atoms, tput, lat): return ((atoms - 1) * tput + lat) * NS_PER_CYC


def stages(topk):
    k = KERNELS[topk]; b = k["B_TOPK"]
    # producer (tensor cores): QK + PV matmul throughput, serial on the tensor pipe
    t_qk = k["qk_atoms"] * k["qk_atom_cyc"] * NS_PER_CYC
    t_sv = k["sv_atoms"] * MMA_N256_CYC      * NS_PER_CYC
    t_tensor = t_qk + t_sv
    t_gather = GATHER_NS_64 * (b / 64)
    t_overlap = max(t_tensor, t_gather)                       # producer-side throughput floor

    # consumer compute, from primitives + source op-counts
    t_exp2     = ROWS_SOFTMAX * b / EXP2_OPS_PER_S_PER_SM * 1e9                # exp2 over rows x B_TOPK
    # 3 FP32 ALU passes the composed model omits: rowmax fmax, scale FMA (L652), rowsum add
    t_softmax_reduce = 3 * ROWS_SOFTMAX * b / FP32_FFMA_OPS_PER_S_PER_SM * 1e9
    t_softmax_full   = t_exp2 + t_softmax_reduce
    # correction O-rescale: FMUL2 over rows x D_V (per k-iter), source L835-851
    t_correction = ROWS_CORR * COLS_CORR / FP32_MUL_OPS_PER_S_PER_SM * 1e9

    t_consumer = t_softmax_full + t_correction                # softmax + correction warpgroup compute

    # latency-bound single-buffered scoring recurrence (the existing composed t_scoring)
    t_qk_chain = chain_lat(k["qk_atoms"], k["qk_atom_cyc"], k["qk_atom_lat"])
    t_sv_chain = chain_lat(k["sv_atoms"], MMA_N256_CYC, MMA_N256_LAT)
    t_scoring  = t_qk_chain + H_ONEWAY_NS + t_exp2 + H_ONEWAY_NS + t_sv_chain + H_ONEWAY_NS

    return dict(t_tensor=t_tensor, t_gather=t_gather, t_overlap=t_overlap,
                t_exp2=t_exp2, t_softmax_reduce=t_softmax_reduce, t_softmax_full=t_softmax_full,
                t_correction=t_correction, t_consumer=t_consumer,
                t_qk_chain=t_qk_chain, t_sv_chain=t_sv_chain, t_scoring=t_scoring)


# ---------- forbidden oracle residual (reference only; never used in predict) ----------
g = json.load(open(GRID)); rows = g["rows"]
topks = sorted({r["topk"] for r in rows})
oracle = {}
for tk in topks:
    bs1 = statistics.mean(r["latency_us"] for r in rows if r["topk"] == tk and r["batch_size"] == 1)
    oracle[tk] = (bs1 * 1e3 - (T_pro + T_epi)) / k_tiles(tk) - stages(tk)["t_overlap"]


def t_step(topk, mode):
    s = stages(topk)
    if mode == "composed":          # current model: serial scoring recurrence
        return max(s["t_overlap"], s["t_scoring"])
    if mode == "corr_serial":       # add full softmax reductions + correction to the serial recurrence
        extra = s["t_softmax_reduce"] + s["t_correction"]
        return max(s["t_overlap"], s["t_scoring"] + extra)
    if mode == "corr_exposed":      # pipeline-imbalance: expose only the consumer EXCESS over the matmul producer
        excess = max(0.0, s["t_consumer"] - s["t_tensor"])
        return max(s["t_overlap"], s["t_scoring"] + excess)
    if mode == "corr_consumer_max": # period = max(serial scoring path, full consumer stage incl. handshakes)
        t_cons_path = H_ONEWAY_NS + s["t_softmax_full"] + H_ONEWAY_NS + s["t_correction"] + H_ONEWAY_NS
        return max(s["t_overlap"], s["t_scoring"], s["t_qk_chain"] + t_cons_path)
    raise ValueError(mode)


def predict(batch, topk, mode):
    return num_waves(batch) * (T_pro + k_tiles(topk) * t_step(topk, mode) + T_epi) / 1e3


def evaluate(sub, mode):
    es = [(predict(r["batch_size"], r["topk"], mode) - r["latency_us"]) / r["latency_us"] for r in sub]
    mape = statistics.mean(abs(e) for e in es) * 100
    signed = statistics.mean(es) * 100
    worst = max(abs(e) for e in es) * 100
    return mape, signed, worst


print(f"GRID={GRID}  (n={len(rows)})")
print("=== per-kernel stage decomposition (ns/k-iter), primitives + source op-counts ===")
print(f"   FP32 on-board: mul={FP32_MUL_OPS_PER_S_PER_SM/1e9:.1f} ffma={FP32_FFMA_OPS_PER_S_PER_SM/1e9:.1f} Gops/SM "
      f"(datasheet peak {DATASHEET_FP32_PEAK_PER_SM/1e9:.1f}, ref only)   correction rows x cols = {ROWS_CORR}x{COLS_CORR}")
for tk in topks:
    s = stages(tk)
    print(f"  topk={tk:5d} B_TOPK={KERNELS[tk]['B_TOPK']:3d}: "
          f"t_tensor={s['t_tensor']:6.1f} t_exp2={s['t_exp2']:6.1f} "
          f"t_sm_reduce={s['t_softmax_reduce']:6.1f} t_correction={s['t_correction']:6.1f} "
          f"-> t_consumer={s['t_consumer']:6.1f}")
    print(f"            scoring(serial)={s['t_scoring']:7.1f}  consumer-excess over tensor="
          f"{max(0.0, s['t_consumer']-s['t_tensor']):6.1f}  [forbidden oracle resid={oracle[tk]:7.1f}]")

print("\n=== MAPE on 104-cfg grid (signed = mean signed err; negative = underestimate) ===")
subs = {1024: [r for r in rows if r['topk'] == 1024],
        2048: [r for r in rows if r['topk'] == 2048], "both": rows}
out = {}
for mode in ["composed", "corr_serial", "corr_exposed", "corr_consumer_max"]:
    line = {}
    for key, sub in subs.items():
        mape, signed, worst = evaluate(sub, mode)
        line[str(key)] = dict(mape=round(mape, 2), signed=round(signed, 2), worst=round(worst, 2), n=len(sub))
    out[mode] = line
    print(f"  [{mode:18s}] small={line['1024']['mape']:5.2f}% (signed {line['1024']['signed']:+5.2f}) "
          f"regular={line['2048']['mape']:5.2f}% (signed {line['2048']['signed']:+6.2f}) "
          f"both={line['both']['mape']:5.2f}%")

json.dump(out, open(os.path.join(HERE, "stage_model_correction_results.json"), "w"), indent=2)
print("\nWROTE stage_model_correction_results.json")
