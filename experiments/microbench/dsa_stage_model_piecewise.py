#!/usr/bin/env python3
"""
Piecewise correction model:
- topk=1024 (B_TOPK=64): composed (no explicit correction; small-kernel correction is
  pipeline-covered by the longer QK chain path)
- topk=2048 (B_TOPK=128): corr_serial (add t_softmax_reduce + t_correction to t_scoring;
  correction is NOT covered because the extra ops add to the critical serial path)

Physical reason for the regime split: the correction warpgroup execute O-accumulator
FMUL2 rescaling (O(D_V) ops) AFTER SV accumulation. In the small kernel the QK chain
is short enough that the recurrence period is still set by pipeline overlap; in the
regular kernel the extended consumer path (softmax_reduce + correction) adds to the
bottleneck serial chain, not to the overlap floor.
"""
import json, math, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.normpath(os.path.join(os.path.dirname(HERE), "analysis-results",
                                     "topk_scan_v2", "json", "grid_topk_v2.json"))

# ---- microbench constants ----
CLK_GHZ      = 1.965
NS_PER_CYC   = 1.0 / CLK_GHZ
# QK throughput at the kernel's pipeline depth (~depth-32), NOT the deep-pipeline asymptote.
# umma_throughput depth sweep (BF16/TS/cta2/M128N128K16): d16=41.9 d32=37.0 d64=34.5 d256=32.65.
# The kernel runs QK in ~18-MMA bursts that drain on the online-softmax recurrence, so the
# asymptotic peak (32.65) is unreachable; using it worsens MAPE (small 4.92%->7.9%). See
# results/MEASUREMENT_NOTES.md "PIPELINE-DEPTH PROVENANCE".
MMA_N128_CYC = 37.106; MMA_N128_LAT = 178.0
MMA_N256_CYC = 64.648; MMA_N256_LAT = 210.0
GATHER_NS_64 = 16.2
EXP2_OPS_PER_S_PER_SM = 24.8e9
TMA_BW_TBPS  = 3.75
H_ONEWAY_NS  = 280.66 / 2.0   # = 140.33 ns
# On-board FP32-ALU (correction_bench/corr_fp32_alu.csv)
FP32_MUL_OPS  = 176.567e9    # FMUL rate for O-accumulator rescale
FP32_FFMA_OPS = 174.600e9    # FFMA rate for softmax reductions

H_Q, D_Q, D_V = 128, 576, 512
TILES_PER_WAVE = 74
ROWS_SOFTMAX = H_Q // 2  # 64
ROWS_CORR    = H_Q       # 128
COLS_CORR    = D_V // 2  # 256

KERNELS = {
    1024: dict(B_TOPK=64,  qk_atom_cyc=MMA_N128_CYC, qk_atom_lat=MMA_N128_LAT, qk_atoms=18, sv_atoms=4),
    2048: dict(B_TOPK=128, qk_atom_cyc=MMA_N256_CYC, qk_atom_lat=MMA_N256_LAT, qk_atoms=18, sv_atoms=8),
}

T_pro = H_Q * D_Q * 2 / (TMA_BW_TBPS * 1e12) * 1e9
T_epi = H_Q * D_V * 2 / (TMA_BW_TBPS * 1e12) * 1e9

def num_waves(b): return max(1, math.ceil(b / TILES_PER_WAVE))
def k_tiles(tk):  return max(1, math.ceil(tk / KERNELS[tk]["B_TOPK"]))
def chain_lat(atoms, tput, lat): return ((atoms - 1) * tput + lat) * NS_PER_CYC

def t_step_piecewise(topk):
    k = KERNELS[topk]; b = k["B_TOPK"]
    t_qk  = k["qk_atoms"] * k["qk_atom_cyc"] * NS_PER_CYC
    t_sv  = k["sv_atoms"] * MMA_N256_CYC * NS_PER_CYC
    t_tensor = t_qk + t_sv
    t_gather = GATHER_NS_64 * (b / 64)
    t_exp2   = ROWS_SOFTMAX * b / EXP2_OPS_PER_S_PER_SM * 1e9
    t_overlap = max(t_tensor, t_gather, t_exp2)

    t_qk_chain = chain_lat(k["qk_atoms"], k["qk_atom_cyc"], k["qk_atom_lat"])
    t_sv_chain = chain_lat(k["sv_atoms"], MMA_N256_CYC, MMA_N256_LAT)
    t_scoring  = t_qk_chain + H_ONEWAY_NS + t_exp2 + H_ONEWAY_NS + t_sv_chain + H_ONEWAY_NS

    if topk == 1024:
        # small kernel: correction is covered; keep composed t_scoring
        return max(t_overlap, t_scoring), t_scoring, 0.0, 0.0
    else:
        # regular kernel: correction extends the serial chain
        t_sm_reduce  = 3 * ROWS_SOFTMAX * b / FP32_FFMA_OPS * 1e9
        t_correction = ROWS_CORR * COLS_CORR / FP32_MUL_OPS * 1e9
        t_scoring_new = t_scoring + t_sm_reduce + t_correction
        return max(t_overlap, t_scoring_new), t_scoring_new, t_sm_reduce, t_correction

def predict(batch, topk):
    ts, _, _, _ = t_step_piecewise(topk)
    return num_waves(batch) * (T_pro + k_tiles(topk) * ts + T_epi) / 1e3

# ---- run on grid ----
g = json.load(open(GRID))
rows = g["rows"]
topks = sorted({r["topk"] for r in rows})

print("=== Stage decomposition for each kernel ===")
for tk in topks:
    ts, t_sc, t_smr, t_corr = t_step_piecewise(tk)
    b = KERNELS[tk]["B_TOPK"]
    print(f"  topk={tk}  B_TOPK={b}  t_step={ts:.1f} ns  t_scoring={t_sc:.1f} ns  "
          f"t_sm_reduce={t_smr:.1f} ns  t_correction={t_corr:.1f} ns")

print("\n=== Per-config predictions for paper table (sample) ===")
print(f"{'topk':>5} {'batch':>5} {'nwav':>4} {'fill':>5} {'meas_us':>9} {'pred_us':>9} {'err%':>7} {'rfl%':>7}")
# Table rows shown in paper
table_rows = [(1024, [1,32,64,74,128,148,256,296]), (2048, [1,32,64,74,128,148,256,296])]
table_out = {}
for tk, batches in table_rows:
    for b in batches:
        match = [r for r in rows if r["topk"]==tk and r["batch_size"]==b]
        if not match: continue
        r = match[0]
        p = predict(b, tk)
        err = (p - r["latency_us"]) / r["latency_us"] * 100
        rfl = r.get("roofline_rel_err", 0) * 100
        nw = num_waves(b)
        fill = b / (nw * TILES_PER_WAVE)
        print(f"  {tk:5d} {b:5d} {nw:4d} {fill:5.2f} {r['latency_us']:9.2f} {p:9.2f} {err:7.1f} {rfl:7.1f}")
        table_out[f"{tk}_{b}"] = dict(topk=tk, batch=b, nwaves=nw, fill=round(fill,2),
                                       meas=round(r["latency_us"],2), pred=round(p,2),
                                       err_pct=round(err,1), rfl_pct=round(rfl,1))

print("\n=== Aggregate MAPE ===")
subsets = {1024: [r for r in rows if r["topk"]==1024],
           2048: [r for r in rows if r["topk"]==2048], "both": rows}
mape_out = {}
for key, sub in subsets.items():
    errs = [(predict(r["batch_size"],r["topk"]) - r["latency_us"])/r["latency_us"] for r in sub]
    mape = statistics.mean(abs(e) for e in errs) * 100
    signed = statistics.mean(errs) * 100
    worst = max(abs(e) for e in errs) * 100
    mape_out[str(key)] = dict(mape=round(mape,2), signed=round(signed,2), worst=round(worst,1), n=len(sub))
    print(f"  topk={key:6}  n={len(sub):3d}  MAPE={mape:.2f}%  signed={signed:+.2f}%  worst={worst:.1f}%")

out = {"table": table_out, "mape": mape_out,
       "model": "piecewise_correction(small=composed, regular=corr_serial)"}
json.dump(out, open(os.path.join(HERE, "piecewise_results.json"), "w"), indent=2)
print("\nWROTE piecewise_results.json")
