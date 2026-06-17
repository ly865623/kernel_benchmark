#!/usr/bin/env python3
"""NON-DESTRUCTIVE strengthening slice for the topk-generalization question.

The published model dsa_stage_model_v2 scales the WHOLE per-tile pipeline span linearly by
K_tiles=ceil(topk/128). That implicitly scales the per-tile prologue/epilogue (Q load, Q
smem->tmem, O epilogue) -- which are FIXED per-tile costs -- by K_tiles too, so doubling topk
predicts a 2.0x per-tile span. The kernel's own profiler says prologue/epilogue are ~27.6% of
the critical-warp span and do NOT depend on K_tiles, so only the ~72.4% loop portion should
scale. This is a fixed-cost amortization correction read straight from the profiler decomposition
(NOT fit to the topk=2048 whole-kernel latencies).

Correction (single multiplicative factor on the existing k_tiles scaling):
    f_fixed = PE_span / critical_span   (profiler-measured, = 0.2759 at bs=1,skv=8192)
    kfac(k) = f_fixed + (1 - f_fixed) * (k_tiles / K_TILES_REF)
By construction kfac(K_TILES_REF) == 1, so EVERY topk=1024 prediction is byte-identical to the
published model and the 4.92% headline is untouched. Only topk!=1024 moves.

We reuse the published module's already-calibrated constants (a, b, overhead, clean_bs1, PE_us,
K_TILES_REF) verbatim -- no refitting of anything. Evaluate on the existing 104-config grid.
"""
import json, os, math, statistics, importlib.util

MAIN = "/home/liuy/DeepScientist/quests/003/.ds/worktrees/paper-dsa-stagepred-grid-v2/experiments/main"
HERE = "/home/liuy/DeepScientist/quests/003/.ds/worktrees/paper-dsa-stagepred-grid-v2/experiments/analysis-results/topk_scan_v2"
TOPK = os.path.join(HERE, "json", "grid_topk_v2.json")

# import the published model (runs its own calibration on import) and reuse its constants verbatim
spec = importlib.util.spec_from_file_location("dsa_stage_model_v2", os.path.join(MAIN, "dsa_stage_model_v2.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
a, b = mod.a, mod.b
overhead = mod.overhead
clean_bs1 = mod.clean_bs1
PE_us = mod.PE_us
K_TILES_REF = mod.K_TILES_REF
fill_of = mod.fill_of
TPW = mod.TILES_PER_WAVE

# profiler-measured fixed (prologue+epilogue) fraction of the critical-warp span at bs=1,skv=8192
stages = json.load(open(os.path.join(MAIN, "stages", "json", "stage_timings.json")))
sb = {(c["s_q"], c["s_kv"]): c for c in stages["configs"]}
ss = sb[(1, 8192)]["stage_summary_per_block_ns"]
PE_span = sum(ss[s]["per_block_total_ns_mean"] for s in ("q_load", "q_smem2tmem", "epilogue_o") if s in ss) / 1e3
CRIT_span = max(v["per_block_total_ns_mean"] for v in ss.values()) / 1e3
F_FIXED = PE_span / CRIT_span

def kfac(k_tiles):
    return F_FIXED + (1.0 - F_FIXED) * (k_tiles / K_TILES_REF)

def predict_amortized(batch, s_kv, topk, mode):
    nt = batch
    nw = max(1, math.ceil(nt / TPW))
    fill = nt / (nw * TPW)
    k_tiles = max(1, (topk + 127) // 128)
    kf = kfac(k_tiles)                      # <-- replaces linear (k_tiles/K_TILES_REF)
    if mode == "profiler":
        span_prof_us = (a + b * fill) * kf
        t_first = span_prof_us / overhead
    else:  # single-tile-anchored (paper primary)
        shape = (a + b * fill) / (a + b * fill_of(1))
        t_first = clean_bs1 * shape * kf
    pe = min(PE_us, 0.6 * t_first)          # prologue/epilogue is FIXED, not k_tiles-scaled
    t_marginal = t_first - pe
    return t_first + (nw - 1) * t_marginal

rows = json.load(open(TOPK))["rows"]

def stats(subset, predfn, mode):
    errs = [abs(predfn(r["batch_size"], r["s_kv"], r["topk"], mode) - r["latency_us"]) / r["latency_us"] for r in subset]
    abss = [abs(predfn(r["batch_size"], r["s_kv"], r["topk"], mode) - r["latency_us"]) for r in subset]
    return {"n": len(subset), "mape_pct": round(statistics.mean(errs) * 100, 3),
            "mae_us": round(statistics.mean(abss), 4), "worst_pct": round(max(errs) * 100, 2)}

def split(predfn, mode):
    return {"all": stats(rows, predfn, mode),
            "topk_1024": stats([r for r in rows if r["topk"] == 1024], predfn, mode),
            "topk_2048": stats([r for r in rows if r["topk"] == 2048], predfn, mode)}

print(f"[profiler-derived] fixed(prologue+epilogue) fraction F_FIXED = {F_FIXED:.4f}  "
      f"(PE_span={PE_span:.3f}us / crit_span={CRIT_span:.3f}us)")
print(f"[check] kfac(K_TILES_REF=8) = {kfac(8):.6f}  (must be 1.0 -> topk=1024 unchanged)")
print(f"[check] kfac(16, topk=2048) = {kfac(16):.4f}  (published linear factor = 2.0)\n")

for mode in ["anchored", "profiler"]:
    base = split(mod.predict, mode)
    amo = split(predict_amortized, mode)
    print(f"=== mode={mode} ===")
    for k in ["topk_1024", "topk_2048", "all"]:
        print(f"  {k:10s}  baseline MAPE={base[k]['mape_pct']:6.2f}% MAE={base[k]['mae_us']:7.3f}us "
              f"worst={base[k]['worst_pct']:6.2f}%   ->  amortized MAPE={amo[k]['mape_pct']:6.2f}% "
              f"MAE={amo[k]['mae_us']:7.3f}us worst={amo[k]['worst_pct']:6.2f}%")
    print()

out = {
    "slice": "topk-generalization-amortized-variant",
    "purpose": "Non-destructive: does a profiler-derived fixed-cost amortization on K_tiles restore "
               "cross-topk accuracy WITHOUT perturbing the published topk=1024 headline?",
    "correction": "kfac(k)=F_FIXED+(1-F_FIXED)*(k/K_TILES_REF); F_FIXED from profiler prologue/epilogue fraction",
    "f_fixed_profiler": round(F_FIXED, 4), "k_tiles_ref": K_TILES_REF,
    "kfac_topk2048": round(kfac(16), 4), "published_linear_factor_topk2048": 2.0,
    "measured_ratio_2048_over_1024_mean": 1.614,
    "no_refit": True, "published_model_file_untouched": True,
    "anchored": {"baseline": split(mod.predict, "anchored"), "amortized": split(predict_amortized, "anchored")},
    "profiler": {"baseline": split(mod.predict, "profiler"), "amortized": split(predict_amortized, "profiler")},
}
outp = os.path.join(HERE, "json", "model_eval_amortized_variant.json")
json.dump(out, open(outp, "w"), indent=2)
print("WROTE", outp)
