#!/usr/bin/env python3
"""
Cross-kernel extension of the profiler-free, on-board microbench stage-cost model
to BOTH forward kernels the FlashMLA DSA family dispatches by selection budget:

  - small-budget kernel : topk<=1280, B_TOPK=64   (already validated at topk=1024)
  - regular kernel      : topk>1280 , B_TOPK=128   (topk=2048)

EVERY per-stage coefficient is from a STANDALONE ON-BOARD microbenchmark of the kernel's
own primitives (NOT FLASHINFER_ENABLE_PROFILER, which the user rejected; NOT fit to the
whole-kernel target latencies). The regular kernel uses the SAME measured atom costs:
its QK matmul widens to N=256 (= the measured SV atom 64.648 cyc) and its SV matmul
doubles its K=B_TOPK contraction to 8 atoms; gather doubles to a 128-token block and
softmax doubles to 8192 exp2/k-iter. Only one quantity is re-measured per kernel: the
single bs=1 wave anchor for the warp-specialized pipeline sync residual.

Validation target: the on-board 104-config whole-kernel grid topk_scan_v2/grid_topk_v2.json
(CUDA-event timed, clean build, topk in {1024,2048}).
"""
import json, math, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(os.path.dirname(HERE), "main", "..", "analysis-results",
                    "topk_scan_v2", "json", "grid_topk_v2.json")

# ---------------- on-board microbench coefficients (B200 @ 1965 MHz) ----------------
CLK_GHZ        = 1.965
NS_PER_CYC     = 1.0 / CLK_GHZ                 # 0.5089 ns/cyc
MMA_N128_CYC   = 37.106                        # M128 N128 K16 atom (on-board) [mma_costs.csv]
MMA_N256_CYC   = 64.648                        # M128 N256 K16 atom (on-board) [mma_costs.csv]
GATHER_NS_64   = 16.2                          # ns per 64-token KV block (on-board gather4)
EXP2_OPS_PER_S_PER_SM = 24.8e9                 # exp2 SFU rate per SM (on-board) [exp2_sfu.csv]
TMA_BW_TBPS    = 3.75                          # contiguous TMA BW (on-board), q_load/epilogue

# ---------------- kernel structural constants (from source config.h/phase1.cuh) ----------------
H_Q, D_Q, D_V = 128, 576, 512
TILES_PER_WAVE = 74

# Per-kernel selection-block + derived per-k-iter atom structure (2x1SM, per-CTA stream).
#   QK: N = B_TOPK*2 ; atoms = (D_Q/2)/16 = 18 (contraction fixed at 288)
#   SV: N = D_V/2 = 256 ; atoms = (D_V/2/256)*(B_TOPK/16) = B_TOPK/16
KERNELS = {
    1024: dict(B_TOPK=64,  qk_atom_cyc=MMA_N128_CYC, qk_atoms=18, sv_atoms=64 // 16),   # 4
    2048: dict(B_TOPK=128, qk_atom_cyc=MMA_N256_CYC, qk_atoms=18, sv_atoms=128 // 16),  # 8
}

# once-per-tile contiguous TMA (topk-independent)
T_pro = H_Q * D_Q * 2 / (TMA_BW_TBPS * 1e12) * 1e9     # Q[128x576] bf16 load (ns)
T_epi = H_Q * D_V * 2 / (TMA_BW_TBPS * 1e12) * 1e9     # O[128x512] bf16 store (ns)


def op_costs(topk):
    """Per-k-iter on-board op costs (ns) for the kernel that handles this topk."""
    k = KERNELS[topk]
    b_topk = k["B_TOPK"]
    t_qk = k["qk_atoms"] * k["qk_atom_cyc"] * NS_PER_CYC
    t_sv = k["sv_atoms"] * MMA_N256_CYC      * NS_PER_CYC
    t_tensor = t_qk + t_sv                                  # QK,SV share tensor cores -> serial
    t_gather = GATHER_NS_64 * (b_topk / 64)                 # one B_TOPK-token block / k-iter
    t_softmax = (H_Q // 2) * b_topk / EXP2_OPS_PER_S_PER_SM * 1e9
    t_step_overlap = max(t_tensor, t_gather, t_softmax)
    binder = max((("tensor", t_tensor), ("gather", t_gather), ("softmax", t_softmax)),
                 key=lambda kv: kv[1])[0]
    return dict(t_qk=t_qk, t_sv=t_sv, t_tensor=t_tensor, t_gather=t_gather,
                t_softmax=t_softmax, t_step_overlap=t_step_overlap, binder=binder)


def k_tiles(topk):
    return max(1, math.ceil(topk / KERNELS[topk]["B_TOPK"]))


def num_waves(batch):
    return max(1, math.ceil(batch / TILES_PER_WAVE))


# ---------------- load on-board validation target ----------------
g = json.load(open(os.path.normpath(GRID)))
rows = g["rows"]
topks = sorted({r["topk"] for r in rows})

# per-kernel single-row anchor (mean bs=1 wave over s_kv) -> the ONE honest re-anchor that
# captures the warp-specialized pipeline sync/dependency residual (microbench cannot supply it)
anchor = {}
overhead = {}
for tk in topks:
    bs1 = [r["latency_us"] for r in rows if r["topk"] == tk and r["batch_size"] == 1]
    anchor[tk] = statistics.mean(bs1)
    oc = op_costs(tk)
    kt = k_tiles(tk)
    overhead[tk] = (anchor[tk] * 1e3 - (T_pro + T_epi)) / kt - oc["t_step_overlap"]


def predict(batch, topk, mode):
    nw = num_waves(batch)
    kt = k_tiles(topk)
    oc = op_costs(topk)
    if mode == "micro_overlap":
        t_wave = (T_pro + kt * oc["t_step_overlap"] + T_epi) / 1e3
    elif mode == "micro_serial":
        t_wave = (T_pro + kt * (oc["t_tensor"] + oc["t_gather"] + oc["t_softmax"]) + T_epi) / 1e3
    elif mode == "anchored":
        t_wave = (T_pro + kt * (oc["t_step_overlap"] + overhead[topk]) + T_epi) / 1e3
    return nw * t_wave


def evaluate(subset, mode):
    errs, out = [], []
    for r in subset:
        m = r["latency_us"]
        p = predict(r["batch_size"], r["topk"], mode)
        e = abs(p - m) / m
        errs.append(e)
        out.append(dict(topk=r["topk"], batch_size=r["batch_size"], s_kv=r["s_kv"],
                        num_waves=r["num_waves"], wave_fill=round(r["wave_fill"], 3),
                        measured_us=round(m, 3), pred_us=round(p, 3),
                        abs_pct_err=round(e * 100, 2)))
    return statistics.mean(errs) * 100, max(errs) * 100, out


print("=== on-board per-k-iter op costs (ns) per kernel ===")
for tk in topks:
    oc = op_costs(tk)
    print(f"  topk={tk:5d} (B_TOPK={KERNELS[tk]['B_TOPK']:3d}): "
          f"QK={oc['t_qk']:7.2f} SV={oc['t_sv']:7.2f} tensor={oc['t_tensor']:7.2f} "
          f"gather={oc['t_gather']:6.2f} softmax={oc['t_softmax']:7.2f} "
          f"-> t_step(max)={oc['t_step_overlap']:7.2f} binder={oc['binder']} "
          f"| k_tiles={k_tiles(tk)} anchor={anchor[tk]:.3f}us "
          f"resid/kiter={overhead[tk]:.1f}ns")
print(f"  T_pro={T_pro:.2f}ns T_epi={T_epi:.2f}ns")

rfl_mape = statistics.mean(abs(r["roofline_rel_err"]) for r in rows) * 100

print("\n=== MODEL EVALUATION (on-board grid, anchored) ===")
subsets = {1024: [r for r in rows if r["topk"] == 1024],
           2048: [r for r in rows if r["topk"] == 2048],
           "both": rows}
report = {}
for key, sub in subsets.items():
    mape, worst, out = evaluate(sub, "anchored")
    rf = statistics.mean(abs(r["roofline_rel_err"]) for r in sub) * 100
    report[str(key)] = dict(n_cfg=len(sub), mape_pct=round(mape, 2),
                            worst_abs_pct_err=round(worst, 2), roofline_mape_pct=round(rf, 2),
                            per_config=out)
    label = f"topk={key}" if key != "both" else "BOTH kernels"
    print(f"  [{label:13s}] n={len(sub):3d}  anchored MAPE = {mape:5.2f}%  "
          f"worst = {worst:5.2f}%   (roofline {rf:5.2f}%)")

# zero-fit references on the full grid (show op-costs alone are insufficient -> motivate anchor)
for mode in ["micro_overlap", "micro_serial"]:
    mape, worst, _ = evaluate(rows, mode)
    print(f"  [{mode:13s}] n={len(rows):3d}  zero-fit MAPE = {mape:5.2f}%  worst = {worst:5.2f}%")

summary = {
    "model": "stage-centric-wave-quantized-dsa-microbench-2kernel",
    "method": ("T_pred = num_waves * (T_pro + k_tiles*(t_step_overlap + sync_resid) + T_epi); "
               "all per-stage costs from on-board microbenchmarks of the kernel's own primitives "
               "(no profiler, no fit to whole-kernel latencies); one bs=1 anchor re-measured per "
               "kernel for the warp-specialized pipeline sync residual."),
    "clock_ghz": CLK_GHZ, "tiles_per_wave": TILES_PER_WAVE,
    "tile_amortized_ns": {"T_pro": T_pro, "T_epi": T_epi},
    "per_kernel": {str(tk): dict(B_TOPK=KERNELS[tk]["B_TOPK"], k_tiles=k_tiles(tk),
                                 anchor_us=round(anchor[tk], 3),
                                 sync_resid_per_kiter_ns=round(overhead[tk], 1),
                                 **{k: round(v, 3) for k, v in op_costs(tk).items() if k != "binder"},
                                 binder=op_costs(tk)["binder"]) for tk in topks},
    "roofline_mape_pct": round(rfl_mape, 2),
    "results": {k: {kk: vv for kk, vv in v.items() if kk != "per_config"} for k, v in report.items()},
    "per_config": {k: v["per_config"] for k, v in report.items()},
}
outp = os.path.join(HERE, "stage_model_microbench_2kernel_results.json")
json.dump(summary, open(outp, "w"), indent=2)
print("\nWROTE", outp)
