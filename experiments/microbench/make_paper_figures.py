#!/usr/bin/env python3
"""Regenerate the four paper figures from the on-board (profiler-free) piecewise model
and the measured 104-config grid. Morandi palette (paper-facing: mist-stone + sage-clay)."""
import json, math, os, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 200, "font.size": 10,
    "axes.edgecolor": "#8A9199", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#D8D1C7", "grid.linewidth": 0.6,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "legend.frameon": False,
})
# Morandi
C_MODEL, C_ROOF, C_MEAS = "#7F8F84", "#B88C8C", "#8A9199"
C_BARS = ["#B7A99A", "#D8C3BC", "#8A9199", "#B88C8C", "#7F8F84", "#D8D1C7"]

HERE = os.path.dirname(os.path.abspath(__file__))
RES = json.load(open(os.path.join(HERE, "stage_model_microbench_2kernel_results.json")))
GRID = json.load(open(os.path.normpath(os.path.join(
    HERE, "..", "analysis-results", "topk_scan_v2", "json", "grid_topk_v2.json"))))
FIGDIR = "/home/liuy/DeepScientist/quests/003/paper/figures"
os.makedirs(FIGDIR, exist_ok=True)

# ---- Piecewise model (microbenchmark-only, no whole-kernel fit) ----
CLK_GHZ = 1.965; NS_PER_CYC = 1.0 / CLK_GHZ
MMA_N128_CYC = 37.106; MMA_N128_LAT = 178.0
MMA_N256_CYC = 64.648; MMA_N256_LAT = 210.0
GATHER_NS_64 = 16.2; EXP2_OPS = 24.8e9; H_ONEWAY_NS = 140.33
FP32_MUL_OPS = 176.567e9; FP32_FFMA_OPS = 174.600e9
TMA_BW_TBPS = 3.75; H_Q = 128; D_Q = 576; D_V = 512; TILES_PER_WAVE = 74
ROWS_SOFTMAX = 64; ROWS_CORR = 128; COLS_CORR = 256
T_pro = H_Q * D_Q * 2 / (TMA_BW_TBPS * 1e12) * 1e9
T_epi = H_Q * D_V * 2 / (TMA_BW_TBPS * 1e12) * 1e9
KERNELS_PC = {
    1024: dict(B_TOPK=64,  qk_atom_cyc=MMA_N128_CYC, qk_atom_lat=MMA_N128_LAT, qk_atoms=18, sv_atoms=4),
    2048: dict(B_TOPK=128, qk_atom_cyc=MMA_N256_CYC, qk_atom_lat=MMA_N256_LAT, qk_atoms=18, sv_atoms=8),
}

def _chain_lat(atoms, tput, lat): return ((atoms - 1) * tput + lat) * NS_PER_CYC
def _t_step(topk):
    k = KERNELS_PC[topk]; b = k["B_TOPK"]
    t_qk = k["qk_atoms"] * k["qk_atom_cyc"] * NS_PER_CYC
    t_sv = k["sv_atoms"] * MMA_N256_CYC * NS_PER_CYC
    t_gather = GATHER_NS_64 * (b / 64)
    t_exp2   = ROWS_SOFTMAX * b / EXP2_OPS * 1e9
    t_overlap = max(t_qk + t_sv, t_gather, t_exp2)
    t_qk_chain = _chain_lat(k["qk_atoms"], k["qk_atom_cyc"], k["qk_atom_lat"])
    t_sv_chain = _chain_lat(k["sv_atoms"], MMA_N256_CYC, MMA_N256_LAT)
    t_scoring  = t_qk_chain + H_ONEWAY_NS + t_exp2 + H_ONEWAY_NS + t_sv_chain + H_ONEWAY_NS
    if topk == 1024:
        return max(t_overlap, t_scoring)
    t_sm_reduce  = 3 * ROWS_SOFTMAX * b / FP32_FFMA_OPS * 1e9
    t_correction = ROWS_CORR * COLS_CORR / FP32_MUL_OPS * 1e9
    return max(t_overlap, t_scoring + t_sm_reduce + t_correction)

_tstep_cache = {tk: _t_step(tk) for tk in (1024, 2048)}
_ktiles = {tk: max(1, math.ceil(tk / KERNELS_PC[tk]["B_TOPK"])) for tk in (1024, 2048)}

def pc_predict(batch, topk):
    nw = max(1, math.ceil(batch / TILES_PER_WAVE))
    return nw * (T_pro + _ktiles[topk] * _tstep_cache[topk] + T_epi) / 1e3

# compute per-config errors for the piecewise model over all 104 grid rows
rows = GRID["rows"]
_abs_errs_all, _abs_errs_1024, _abs_errs_2048 = [], [], []
for r in rows:
    p = pc_predict(r["batch_size"], r["topk"])
    ae = abs(p - r["latency_us"]) / r["latency_us"] * 100
    _abs_errs_all.append(ae)
    ((_abs_errs_1024 if r["topk"] == 1024 else _abs_errs_2048).append(ae))
pc_mape = {1024: statistics.mean(_abs_errs_1024), 2048: statistics.mean(_abs_errs_2048),
           "both": statistics.mean(_abs_errs_all)}

roof = {(r["topk"], r["batch_size"], r["s_kv"]): abs(r["roofline_rel_err"]) * 100 for r in rows}

# ---- Fig 1: per-config abs % error, model vs roofline ----
model_err_all = sorted(abs(pc_predict(r["batch_size"], r["topk"]) - r["latency_us"])
                       / r["latency_us"] * 100 for r in rows)
roof_err_all  = sorted(abs(r["roofline_rel_err"]) * 100 for r in rows)
x = range(1, len(rows) + 1)
fig, ax = plt.subplots(figsize=(4.0, 2.9))
ax.plot(x, roof_err_all, "-o", ms=2.5, color=C_ROOF, label=f"datasheet roofline ({pc_mape['both']:.0f}%)")
ax.plot(x, model_err_all, "-o", ms=2.5, color=C_MODEL, label=f"piecewise model ({pc_mape['both']:.2f}%)")
ax.axhline(10, ls="--", lw=0.8, color="#8A9199")
ax.text(2, 11.5, "10% pre-sweep target", fontsize=7, color="#8A9199")
ax.set_xlabel("configuration (sorted by error)")
ax.set_ylabel("absolute % error")
ax.set_title("Per-configuration accuracy (104 configs, both kernels)", fontsize=9)
ax.legend(loc="center right", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "fig_pred_vs_meas.png")); plt.close(fig)

# ---- Fig 2: wave staircase (small kernel, s_kv=8k) ----
sk_rows = [r for r in rows if r["topk"] == 1024 and r["s_kv"] == 8192]
sk_rows.sort(key=lambda r: r["batch_size"])
bs = [r["batch_size"] for r in sk_rows]
meas = [r["latency_us"] for r in sk_rows]
pred = [pc_predict(r["batch_size"], 1024) for r in sk_rows]
fig, ax = plt.subplots(figsize=(4.0, 2.9))
ax.plot(bs, meas, "-o", ms=4, color=C_MEAS, label="measured")
ax.plot(bs, pred, "--s", ms=4, color=C_MODEL, label="stage model")
for b in (74, 148, 296):
    ax.axvline(b, ls=":", lw=0.7, color="#D8C3BC")
ax.set_xlabel("batch size $s_q$")
ax.set_ylabel("whole-kernel latency ($\\mu$s)")
ax.set_title("Wave staircase (small kernel, $s_{kv}=8$k)", fontsize=9)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "fig_wave_staircase.png")); plt.close(fig)

# ---- Fig 3: per-iteration operator breakdown (small kernel) ----
pk = RES["per_kernel"]["1024"]
labels = ["QK\nmatmul", "SV\nmatmul", "gather4", "softmax\nexp2",
          "overlap\nenvelope", "sync\nresidual"]
vals = [pk["t_qk"], pk["t_sv"], pk["t_gather"], pk["t_softmax"],
        pk["t_step_overlap"], pk["sync_resid_per_kiter_ns"]]
fig, ax = plt.subplots(figsize=(4.2, 2.9))
bars = ax.bar(range(len(vals)), vals, color=C_BARS, edgecolor="#8A9199", linewidth=0.6)
for i, v in enumerate(vals):
    ax.text(i, v + 12, f"{v:.0f}", ha="center", fontsize=7, color="#5f5f5f")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("per-iteration cost (ns)")
ax.set_title("On-board operator decomposition (small kernel)", fontsize=9)
ax.text(0.98, 0.92, "operators = 38.9% of\nmeasured per-iter time",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#7F8F84")
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "fig_op_breakdown.png")); plt.close(fig)

# ---- Fig 4: s_kv invariance (small kernel, bs=74) ----
inv = [r for r in rows if r["topk"] == 1024 and r["batch_size"] == 74]
inv.sort(key=lambda r: r["s_kv"])
skv = [r["s_kv"] / 1024 for r in inv]
mi = [r["latency_us"] for r in inv]
pi = [pc_predict(74, 1024)] * len(inv)
fig, ax = plt.subplots(figsize=(4.0, 2.9))
ax.plot(skv, mi, "-o", ms=4, color=C_MEAS, label="measured")
ax.plot(skv, pi, "--s", ms=4, color=C_MODEL, label="stage model")
ax.set_xscale("log", base=2)
ax.set_xlabel("context length $s_{kv}$ (k tokens)")
ax.set_ylabel("whole-kernel latency ($\\mu$s)")
ax.set_ylim(0, max(mi) * 1.6)
ax.set_title("Context-length invariance (small kernel, $s_q=74$)", fontsize=9)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "fig_skv_invariance.png")); plt.close(fig)

print("wrote 4 figures to", FIGDIR)
for f in ["fig_pred_vs_meas.png", "fig_wave_staircase.png", "fig_op_breakdown.png", "fig_skv_invariance.png"]:
    p = os.path.join(FIGDIR, f)
    print(f"  {f}: {os.path.getsize(p)} bytes" if os.path.exists(p) else f"  MISSING {f}")
