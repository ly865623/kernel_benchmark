#!/usr/bin/env python3
"""
CONSTRAINT-COMPLIANT pipeline-synchronization model for the FlashMLA DSA sparse-prefill
kernel family (B200, sm_100a).

STRONG CONSTRAINT (user, 2026-06-16): the whole fused operator may NEVER be run on-board
to obtain its performance. The previous "anchored" model (3.27% MAPE) is therefore
INVALID: its overhead[topk] = anchor/k_tiles - t_step_overlap back-derives the
per-iteration sync residual from the measured bs=1 *whole-kernel* latency (clean_bs1).
That single-row anchor is a forbidden whole-operator measurement.

This model predicts the per-k-iteration cost t_step ENTIRELY from isolated primitive
microbenchmarks + the source-read pipeline structure, with ZERO fitting to any
whole-kernel latency:

  Primitives (all standalone on-board, no fused operator):
    - MMA atom throughput + single-op latency : mma_costs.csv (37.106/64.648 cyc tput;
      178/210 cyc latency)
    - exp2 SFU throughput                      : exp2_sfu.csv (24.8 GOps/SM)
    - gather block cost                        : gather4 (16.2 ns / 64-token block)
    - contiguous TMA BW                        : 3.75 TB/s (q load / o store)
    - cross-warp mbarrier one-way signal latency: sync_bench (S=2,B=1 ping-pong period
      280.66 ns -> one-way = 140.33 ns; buffer-depth-insensitive, payload-free)

  Multi-level producer/consumer structure (user emphasis; source phase1.cuh):
    The k-loop is a MULTI-LEVEL warp-specialized pipeline:
      gather -> coord -> KV-transform -> QK(mma) -> softmax -> SV(mma)
    The KV-production sub-pipeline is NUM_K_BUFS=4 deep  => fully overlapped =>
      its latency is HIDDEN behind compute (only its throughput matters; it never
      gates the steady-state period for the configs in the grid).
    The SCORING/OUTPUT path (P, S) is SINGLE-buffered (get<1>) => its three cross-warp
      boundaries are EXPOSED and force a serial recurrence per k-iteration:
        QK -> [h] -> softmax -> [h] -> SV -> [h] -> (next-iter QK, buffer reuse)
      where each [h] = one one-way cross-warp mbarrier signal, and each compute stage
      pays its dependent-chain LATENCY (not just throughput), because the single buffer
      blocks cross-iteration overlap on this path.

  Composition (per k-iter):
      t_chain(stage)  = (atoms-1)*atom_tput_cyc + atom_latency_cyc   [dependent MMA chain]
      t_scoring_path  = t_chain(QK) + h + t_softmax + h + t_chain(SV) + h
      t_step          = max(t_step_overlap, t_scoring_path)
      (t_step_overlap = max over engines of per-iter throughput, the bottom-up floor.)

  The sync residual reported for comparison is t_step - t_step_overlap, but it is a
  PREDICTION, never consumed from the kernel.

Validation target: on-board 104-config whole-kernel grid grid_topk_v2.json (used ONLY to
score the prediction; never to calibrate any coefficient).
"""
import json, math, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(os.path.dirname(HERE), "main", "..", "analysis-results",
                    "topk_scan_v2", "json", "grid_topk_v2.json")

# ---------------- on-board microbench coefficients (B200 @ 1965 MHz) ----------------
CLK_GHZ        = 1.965
NS_PER_CYC     = 1.0 / CLK_GHZ                 # 0.5089 ns/cyc
MMA_N128_CYC   = 37.106                        # M128 N128 K16 atom tput  [mma_costs.csv]
MMA_N256_CYC   = 64.648                        # M128 N256 K16 atom tput  [mma_costs.csv]
MMA_N128_LAT   = 178.0                         # M128 N128 K16 single-op latency [mma_costs.csv]
MMA_N256_LAT   = 210.0                         # M128 N256 K16 single-op latency [mma_costs.csv]
GATHER_NS_64   = 16.2                          # ns per 64-token KV block (on-board gather4)
EXP2_OPS_PER_S_PER_SM = 24.8e9                 # exp2 SFU rate per SM (on-board) [exp2_sfu.csv]
TMA_BW_TBPS    = 3.75                          # contiguous TMA BW (on-board), q_load/epilogue

# cross-warp mbarrier ONE-WAY signal latency (ns), measured payload-free in sync_bench:
#   S=2,B=1 ping-pong steady-state period = 280.66 ns = 2 one-way signals (fwd + free)
#   -> one-way = 140.33 ns. Buffer-insensitive (B=1==B=2==B=4), so this is the exposed
#   per-boundary signalling cost on the single-buffered scoring path.
H_ONEWAY_NS    = 280.66 / 2.0                  # = 140.33 ns  [sync_bench/sweep_B.csv]

# ---------------- kernel structural constants (from source config.h/phase1.cuh) ----------------
H_Q, D_Q, D_V = 128, 576, 512
TILES_PER_WAVE = 74

# Per-kernel selection-block + derived per-k-iter atom structure (2x1SM, per-CTA stream).
#   QK: N = B_TOPK*2 ; atoms = (D_Q/2)/16 = 18 (contraction fixed at 288)
#   SV: N = D_V/2 = 256 ; atoms = (D_V/2/256)*(B_TOPK/16) = B_TOPK/16
KERNELS = {
    1024: dict(B_TOPK=64,  qk_atom_cyc=MMA_N128_CYC, qk_atom_lat=MMA_N128_LAT, qk_atoms=18, sv_atoms=64 // 16),   # 4
    2048: dict(B_TOPK=128, qk_atom_cyc=MMA_N256_CYC, qk_atom_lat=MMA_N256_LAT, qk_atoms=18, sv_atoms=128 // 16),  # 8
}

# once-per-tile contiguous TMA (topk-independent)
T_pro = H_Q * D_Q * 2 / (TMA_BW_TBPS * 1e12) * 1e9     # Q[128x576] bf16 load (ns)
T_epi = H_Q * D_V * 2 / (TMA_BW_TBPS * 1e12) * 1e9     # O[128x512] bf16 store (ns)


def num_waves(batch):
    return max(1, math.ceil(batch / TILES_PER_WAVE))


def op_costs(topk):
    """Per-k-iter on-board op costs (ns) for the kernel that handles this topk."""
    k = KERNELS[topk]
    b_topk = k["B_TOPK"]
    t_qk = k["qk_atoms"] * k["qk_atom_cyc"] * NS_PER_CYC
    t_sv = k["sv_atoms"] * MMA_N256_CYC      * NS_PER_CYC
    t_tensor = t_qk + t_sv                                  # QK,SV share tensor cores -> serial
    t_gather = GATHER_NS_64 * (b_topk / 64)                 # one B_TOPK-token block / k-iter
    t_softmax = (H_Q // 2) * b_topk / EXP2_OPS_PER_S_PER_SM * 1e9
    t_step_overlap = max(t_tensor, t_gather, t_softmax)     # bottom-up throughput floor
    binder = max((("tensor", t_tensor), ("gather", t_gather), ("softmax", t_softmax)),
                 key=lambda kv: kv[1])[0]
    return dict(t_qk=t_qk, t_sv=t_sv, t_tensor=t_tensor, t_gather=t_gather,
                t_softmax=t_softmax, t_step_overlap=t_step_overlap, binder=binder)


def chain_latency_ns(atoms, atom_tput_cyc, atom_lat_cyc):
    """Latency of a dependent accumulation chain of `atoms` MMA atoms (pipelined issue;
    last atom drains the full single-op latency)."""
    return ((atoms - 1) * atom_tput_cyc + atom_lat_cyc) * NS_PER_CYC


def composed_costs(topk):
    """Constraint-compliant per-k-iter cost from primitives + source structure only."""
    k = KERNELS[topk]
    oc = op_costs(topk)
    t_qk_chain = chain_latency_ns(k["qk_atoms"], k["qk_atom_cyc"], k["qk_atom_lat"])
    t_sv_chain = chain_latency_ns(k["sv_atoms"], MMA_N256_CYC, MMA_N256_LAT)
    # single-buffered scoring recurrence: QK -> h -> softmax -> h -> SV -> h(buffer reuse)
    t_scoring = t_qk_chain + H_ONEWAY_NS + oc["t_softmax"] + H_ONEWAY_NS + t_sv_chain + H_ONEWAY_NS
    t_step = max(oc["t_step_overlap"], t_scoring)
    return dict(t_qk_chain=t_qk_chain, t_sv_chain=t_sv_chain, t_scoring=t_scoring,
                t_step=t_step, t_sync_pred=t_step - oc["t_step_overlap"])


def k_tiles(topk):
    return max(1, math.ceil(topk / KERNELS[topk]["B_TOPK"]))


# ---------------- load on-board validation target (scoring only; never calibrated) ----------------
g = json.load(open(os.path.normpath(GRID)))
rows = g["rows"]
topks = sorted({r["topk"] for r in rows})

# REFERENCE ONLY (forbidden): the old anchored residual, kept to show how close the
# primitive-only composition gets to the (invalid) anchor-informed value. NOT used in predict().
anchor = {}
overhead_oracle = {}
for tk in topks:
    bs1 = [r["latency_us"] for r in rows if r["topk"] == tk and r["batch_size"] == 1]
    anchor[tk] = statistics.mean(bs1)
    oc = op_costs(tk)
    overhead_oracle[tk] = (anchor[tk] * 1e3 - (T_pro + T_epi)) / k_tiles(tk) - oc["t_step_overlap"]


def predict(batch, topk, mode):
    nw = num_waves(batch)
    kt = k_tiles(topk)
    oc = op_costs(topk)
    if mode == "micro_overlap":          # zero-fit bottom-up throughput floor
        t_wave = (T_pro + kt * oc["t_step_overlap"] + T_epi) / 1e3
    elif mode == "micro_serial":         # zero-fit naive serial-engine sum
        t_wave = (T_pro + kt * (oc["t_tensor"] + oc["t_gather"] + oc["t_softmax"]) + T_epi) / 1e3
    elif mode == "composed":             # CONSTRAINT-COMPLIANT primitive composition
        t_wave = (T_pro + kt * composed_costs(topk)["t_step"] + T_epi) / 1e3
    elif mode == "anchored_oracle":      # FORBIDDEN reference (consumes whole-kernel anchor)
        t_wave = (T_pro + kt * (oc["t_step_overlap"] + overhead_oracle[topk]) + T_epi) / 1e3
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


print("=== per-k-iter composition (ns), primitives + source structure only ===")
print(f"  H_ONEWAY (cross-warp mbarrier signal) = {H_ONEWAY_NS:.2f} ns   "
      f"T_pro={T_pro:.2f} T_epi={T_epi:.2f}")
for tk in topks:
    oc = op_costs(tk)
    cc = composed_costs(tk)
    print(f"  topk={tk:5d} (B_TOPK={KERNELS[tk]['B_TOPK']:3d}): "
          f"QK_chain={cc['t_qk_chain']:7.2f} softmax={oc['t_softmax']:7.2f} "
          f"SV_chain={cc['t_sv_chain']:7.2f} +3h={3*H_ONEWAY_NS:6.1f}")
    print(f"               t_overlap(floor)={oc['t_step_overlap']:7.2f}  "
          f"t_scoring(serial)={cc['t_scoring']:7.2f}  -> t_step={cc['t_step']:7.2f}  "
          f"t_sync_pred={cc['t_sync_pred']:7.1f}ns  "
          f"[forbidden oracle resid={overhead_oracle[tk]:7.1f}ns, "
          f"err={abs(cc['t_sync_pred']-overhead_oracle[tk])/overhead_oracle[tk]*100:5.1f}%]")

print("\n=== MODEL EVALUATION on on-board grid (composed = constraint-compliant) ===")
subsets = {1024: [r for r in rows if r["topk"] == 1024],
           2048: [r for r in rows if r["topk"] == 2048],
           "both": rows}
report = {}
for key, sub in subsets.items():
    rf = statistics.mean(abs(r["roofline_rel_err"]) for r in sub) * 100
    line = {"n_cfg": len(sub), "roofline_mape_pct": round(rf, 2)}
    for mode in ["composed", "micro_overlap", "micro_serial", "anchored_oracle"]:
        mape, worst, out = evaluate(sub, mode)
        line[mode] = dict(mape_pct=round(mape, 2), worst_abs_pct_err=round(worst, 2))
        if mode == "composed":
            line["composed_per_config"] = out
    report[str(key)] = line
    label = f"topk={key}" if key != "both" else "BOTH kernels"
    print(f"  [{label:13s}] n={len(sub):3d}  "
          f"COMPOSED={line['composed']['mape_pct']:5.2f}% (worst {line['composed']['worst_abs_pct_err']:5.1f}%)  "
          f"| floor(overlap)={line['micro_overlap']['mape_pct']:5.2f}%  "
          f"serial={line['micro_serial']['mape_pct']:5.2f}%  "
          f"roofline={rf:5.2f}%  "
          f"[oracle(forbidden)={line['anchored_oracle']['mape_pct']:4.2f}%]")

summary = {
    "model": "stage-centric-wave-quantized-dsa-COMPOSED-constraint-compliant",
    "constraint": ("whole fused operator never run on-board; per-k-iter t_step composed from "
                   "isolated primitive microbenchmarks + source pipeline structure; ZERO fit to "
                   "whole-kernel latency. anchored_oracle column is the FORBIDDEN reference only."),
    "method": ("T_pred = num_waves*(T_pro + k_tiles*t_step + T_epi); "
               "t_step = max(t_step_overlap, t_scoring); "
               "t_scoring = chain_lat(QK) + h + t_softmax + h + chain_lat(SV) + h; "
               "h = one-way cross-warp mbarrier signal latency (sync_bench)."),
    "primitives": {
        "clk_ghz": CLK_GHZ, "mma_n128_cyc": MMA_N128_CYC, "mma_n256_cyc": MMA_N256_CYC,
        "mma_n128_lat_cyc": MMA_N128_LAT, "mma_n256_lat_cyc": MMA_N256_LAT,
        "gather_ns_64": GATHER_NS_64, "exp2_gops_per_sm": EXP2_OPS_PER_S_PER_SM / 1e9,
        "tma_bw_tbps": TMA_BW_TBPS, "h_oneway_ns": round(H_ONEWAY_NS, 3),
        "tiles_per_wave": TILES_PER_WAVE,
    },
    "tile_amortized_ns": {"T_pro": T_pro, "T_epi": T_epi},
    "per_kernel": {str(tk): dict(B_TOPK=KERNELS[tk]["B_TOPK"], k_tiles=k_tiles(tk),
                                 **{k: round(v, 3) for k, v in composed_costs(tk).items()},
                                 t_step_overlap=round(op_costs(tk)["t_step_overlap"], 3),
                                 forbidden_oracle_resid_ns=round(overhead_oracle[tk], 1))
                   for tk in topks},
    "results": {k: {kk: vv for kk, vv in v.items() if kk != "composed_per_config"}
                for k, v in report.items()},
    "composed_per_config": {k: v.get("composed_per_config") for k, v in report.items()},
}
outp = os.path.join(HERE, "stage_model_composed_results.json")
json.dump(summary, open(outp, "w"), indent=2)
print("\nWROTE", outp)
