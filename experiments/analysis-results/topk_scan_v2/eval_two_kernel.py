#!/usr/bin/env python3
"""Two-kernel extension of the stage-centric latency model (option C).

FlashMLA's sparse-prefill kernel has TWO compiled variants dispatched by the top-k budget
(csrc/api/sparse_fwd.h, commit 48c6dc4):
    topk <= 1280  -> fwd_for_small_topk/head128 , B_TOPK=64  (the "small" kernel)
    topk  > 1280  -> fwd/head128                , B_TOPK=128 (the "regular" kernel)
The published model dsa_stage_model_v2 is anchored on the small kernel's measured single-tile
(bs=1) latency and predicts the rest of the grid from a shared wave-quantization + occupancy
structure. topk=2048 runs the REGULAR kernel, so a small-kernel anchor cannot predict it -- that
is the 26% cross-kernel error reported earlier (and correctly diagnosed as out-of-scope, not a bug).

This slice extends the SAME structural model to both kernels by giving each kernel its own
single-tile (bs=1) anchor, exactly as the published headline does for the small kernel:
    pred(batch, s_kv) = anchor_bs1[kernel] * shape(fill)   [+ wave PE amortization]
where shape(fill) = (a + b*fill)/(a + b*fill(1)) is the profiler-measured occupancy/HBM-contention
shape (a hardware property of the 148-SM B200, reused unchanged), and the within-kernel
k_tiles/K_TILES_REF ratio is exactly 1 (each subset has a single top-k value), so the kernel
switch is absorbed entirely into the per-kernel bs=1 anchor -- NO refit of any shape constant.

Honesty boundary (identical methodology to the published headline):
  - small kernel  : anchor = mean measured bs=1 latency over s_kv (7 of 56 configs inform it);
                    predictions for batch>1 (49 configs) are genuine out-of-anchor.
                    We reuse the PUBLISHED model verbatim here, so the 4.92% headline is byte-identical.
  - regular kernel: anchor = mean measured bs=1 latency over s_kv (6 of 48 configs inform it);
                    predictions for batch>1 (42 configs) are genuine out-of-anchor.
The published model file dsa_stage_model_v2.py is NOT modified.
"""
import json, os, math, statistics, importlib.util

MAIN = "/home/liuy/DeepScientist/quests/003/.ds/worktrees/paper-dsa-stagepred-grid-v2/experiments/main"
HERE = "/home/liuy/DeepScientist/quests/003/.ds/worktrees/paper-dsa-stagepred-grid-v2/experiments/analysis-results/topk_scan_v2"
TOPK = os.path.join(HERE, "json", "grid_topk_v2.json")

# import the published model (runs its own calibration on import) and reuse its constants verbatim
spec = importlib.util.spec_from_file_location("dsa_stage_model_v2", os.path.join(MAIN, "dsa_stage_model_v2.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
a, b = mod.a, mod.b
PE_us = mod.PE_us
fill_of = mod.fill_of
TPW = mod.TILES_PER_WAVE

rows = json.load(open(TOPK))["rows"]
SMALL = [r for r in rows if r["topk"] <= 1280]   # topk=1024 -> small kernel (B_TOPK=64)
REG   = [r for r in rows if r["topk"]  > 1280]   # topk=2048 -> regular kernel (B_TOPK=128)

def bs1_anchor(subset):
    bs1 = [r["latency_us"] for r in subset if r["batch_size"] == 1]
    return statistics.mean(bs1), len(bs1)

def predict_kernel(batch, anchor_bs1):
    """Shared structural predictor; within-kernel k_tiles factor == 1 (single top-k per subset)."""
    nt = batch
    nw = max(1, math.ceil(nt / TPW))
    fill = nt / (nw * TPW)
    shape = (a + b * fill) / (a + b * fill_of(1))
    t_first = anchor_bs1 * shape
    pe = min(PE_us, 0.6 * t_first)              # prologue/epilogue fixed per tile (same head dims)
    t_marginal = t_first - pe
    return t_first + (nw - 1) * t_marginal

def stats(subset, predfn):
    errs, abss, out = [], [], []
    for r in subset:
        p = predfn(r)
        m = r["latency_us"]
        e = abs(p - m) / m
        errs.append(e); abss.append(abs(p - m))
        out.append({"batch_size": r["batch_size"], "s_kv": r["s_kv"], "topk": r["topk"],
                    "kernel": r["kernel"], "num_waves": r["num_waves"],
                    "wave_fill": round(r["wave_fill"], 3), "measured_us": round(m, 3),
                    "pred_us": round(p, 3), "abs_pct_err": round(e * 100, 2),
                    "roofline_pct_err": round(r["roofline_rel_err"] * 100, 2)})
    return {"n": len(subset), "mape_pct": round(statistics.mean(errs) * 100, 3),
            "mae_us": round(statistics.mean(abss), 4), "worst_pct": round(max(errs) * 100, 2),
            "roofline_mape_pct": round(statistics.mean([r["roofline_rel_err"] for r in subset]) * 100, 3)}, out

# ----- small kernel: reuse PUBLISHED model verbatim (headline byte-identical) -----
small_pred = lambda r: mod.predict(r["batch_size"], r["s_kv"], r["topk"], "anchored")
small_stat, small_out = stats(SMALL, small_pred)

# ----- regular kernel: same structure, own bs=1 anchor -----
anchor_reg, n_reg_bs1 = bs1_anchor(REG)
anchor_small, n_small_bs1 = bs1_anchor(SMALL)
reg_pred = lambda r: predict_kernel(r["batch_size"], anchor_reg)
reg_stat, reg_out = stats(REG, reg_pred)

# ----- combined two-kernel -----
def combined(predfn_small, predfn_reg):
    errs, abss = [], []
    for r in rows:
        p = predfn_small(r) if r["topk"] <= 1280 else predfn_reg(r)
        m = r["latency_us"]; errs.append(abs(p - m) / m); abss.append(abs(p - m))
    return {"n": len(rows), "mape_pct": round(statistics.mean(errs) * 100, 3),
            "mae_us": round(statistics.mean(abss), 4), "worst_pct": round(max(errs) * 100, 2),
            "roofline_mape_pct": round(statistics.mean([r["roofline_rel_err"] for r in rows]) * 100, 3)}
comb = combined(small_pred, reg_pred)

# sanity: confirm the published topk=1024 predictions are byte-identical to the published model
#         (we did not touch mod.predict, so this is true by construction; assert anyway)
identical = all(abs(mod.predict(r["batch_size"], r["s_kv"], r["topk"], "anchored")
                    - small_pred(r)) < 1e-12 for r in SMALL)

print("=== TWO-KERNEL STAGE MODEL (option C) ===")
print(f"small kernel  (topk<=1280, B_TOPK=64) : anchor bs=1 = {anchor_small:.3f} us "
      f"({n_small_bs1} anchor configs, {small_stat['n']-n_small_bs1} genuine predictions)")
print(f"regular kernel(topk> 1280, B_TOPK=128): anchor bs=1 = {anchor_reg:.3f} us "
      f"({n_reg_bs1} anchor configs, {reg_stat['n']-n_reg_bs1} genuine predictions)")
print()
hdr = f"{'subset':14s} {'n':>3} {'MAPE%':>7} {'MAE_us':>8} {'worst%':>7} {'roofline%':>9}"
print(hdr); print("-" * len(hdr))
for name, s in [("small(topk1024)", small_stat), ("regular(topk2048)", reg_stat), ("two-kernel ALL", comb)]:
    print(f"{name:14s} {s['n']:>3} {s['mape_pct']:>7.2f} {s['mae_us']:>8.3f} {s['worst_pct']:>7.2f} {s['roofline_mape_pct']:>9.2f}")
print(f"\nheadline topk=1024 byte-identical to published model: {identical}")

out = {
    "slice": "two-kernel-stage-model-extension",
    "purpose": "Extend the stage-centric model to BOTH FlashMLA sparse kernels (small B_TOPK=64 and "
               "regular B_TOPK=128) by giving each kernel its own measured bs=1 anchor and reusing the "
               "shared profiler-measured wave/occupancy structure unchanged.",
    "dispatch": {"small_kernel": "topk<=1280, fwd_for_small_topk/head128, B_TOPK=64",
                 "regular_kernel": "topk>1280, fwd/head128, B_TOPK=128", "source_commit": "48c6dc4"},
    "shared_structure": "pred=anchor_bs1*shape(fill)[+wave PE amortization]; shape & PE from small-kernel "
                        "profiler (B200 HW property), reused unchanged; within-kernel k_tiles ratio==1.",
    "no_refit_of_shape": True, "published_model_file_untouched": True,
    "headline_topk1024_byte_identical": bool(identical),
    "anchor_small_us": round(anchor_small, 4), "anchor_small_config_count": n_small_bs1,
    "anchor_regular_us": round(anchor_reg, 4), "anchor_regular_config_count": n_reg_bs1,
    "results": {"small_topk1024": small_stat, "regular_topk2048": reg_stat, "two_kernel_all": comb},
    "per_config_small": small_out, "per_config_regular": reg_out,
}
outp = os.path.join(HERE, "json", "model_eval_two_kernel.json")
json.dump(out, open(outp, "w"), indent=2)
print("\nWROTE", outp)
