#!/usr/bin/env python3
"""Evaluate the EXISTING calibrated stage model (dsa_stage_model_v2) on the topk scan v2
grid (topk in {1024,2048}, 104 configs). No refitting: reuse the model's own calibrated
constants and predict() exactly as published. Report MAPE/MAE overall and split by topk."""
import json, os, statistics, importlib.util

MAIN = "/home/liuy/DeepScientist/quests/003/.ds/worktrees/paper-dsa-stagepred-grid-v2/experiments/main"
TOPK = "/home/liuy/DeepScientist/quests/003/.ds/worktrees/paper-dsa-stagepred-grid-v2/experiments/analysis-results/topk_scan_v2/json/grid_topk_v2.json"

# load the published model module (runs its calibration on import)
spec = importlib.util.spec_from_file_location("dsa_stage_model_v2", os.path.join(MAIN, "dsa_stage_model_v2.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

rows = json.load(open(TOPK))["rows"]

def stats(subset, mode):
    errs, abss, rfl = [], [], []
    for r in subset:
        m = r["latency_us"]
        p = mod.predict(r["batch_size"], r["s_kv"], r["topk"], mode)
        errs.append(abs(p - m) / m)
        abss.append(abs(p - m))
        rfl.append(r["roofline_rel_err"])
    return {
        "n": len(subset),
        "mape_pct": round(statistics.mean(errs) * 100, 3),
        "mae_us": round(statistics.mean(abss), 4),
        "worst_pct": round(max(errs) * 100, 2),
        "roofline_mape_pct": round(statistics.mean(rfl) * 100, 2),
    }

print("=== Existing stage model (v2, anchored) evaluated on topk scan v2 grid — NO refit ===")
for mode in ["anchored", "profiler"]:
    print(f"\n--- mode={mode} ---")
    allr = stats(rows, mode)
    k1 = stats([r for r in rows if r["topk"] == 1024], mode)
    k2 = stats([r for r in rows if r["topk"] == 2048], mode)
    print(f"ALL  (n={allr['n']:3d}): MAPE={allr['mape_pct']:6.2f}%  MAE={allr['mae_us']:7.3f}us  worst={allr['worst_pct']:6.2f}%  roofline={allr['roofline_mape_pct']:.2f}%")
    print(f"k1024(n={k1['n']:3d}): MAPE={k1['mape_pct']:6.2f}%  MAE={k1['mae_us']:7.3f}us  worst={k1['worst_pct']:6.2f}%")
    print(f"k2048(n={k2['n']:3d}): MAPE={k2['mape_pct']:6.2f}%  MAE={k2['mae_us']:7.3f}us  worst={k2['worst_pct']:6.2f}%")

# durable output (anchored = paper primary)
out = {
    "model": "stage-centric-wave-quantized-dsa-v2 (existing calibration, no refit)",
    "eval_grid": "grid_topk_v2 (topk in {1024,2048}, 104 configs)",
    "anchored": {
        "all": stats(rows, "anchored"),
        "topk_1024": stats([r for r in rows if r["topk"] == 1024], "anchored"),
        "topk_2048": stats([r for r in rows if r["topk"] == 2048], "anchored"),
    },
    "profiler": {
        "all": stats(rows, "profiler"),
        "topk_1024": stats([r for r in rows if r["topk"] == 1024], "profiler"),
        "topk_2048": stats([r for r in rows if r["topk"] == 2048], "profiler"),
    },
    "note": "K_tiles=ceil(topk/128) so model scales topk=2048 work x2 vs topk=1024 (linear). "
            "Measured latency(2048)/latency(1024)=1.61x (sub-2x, fixed-overhead amortization), "
            "so the linear-K_tiles model is expected to over-predict topk=2048.",
}
outp = "/home/liuy/DeepScientist/quests/003/.ds/worktrees/paper-dsa-stagepred-grid-v2/experiments/analysis-results/topk_scan_v2/json/model_eval_on_topk.json"
json.dump(out, open(outp, "w"), indent=2)
print("\nWROTE", outp)
