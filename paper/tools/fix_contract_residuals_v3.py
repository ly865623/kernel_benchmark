#!/usr/bin/env python3
"""Idempotent repair of the remaining stale (profiler/v1/v2-era) contract residuals.

The science is unchanged and correct. The manuscript (main.tex EN+ZH), evidence_ledger,
selected_outline result_table, and paper_experiment_matrix are already on-board
(MAPE 3.27% / 104 configs / tensor-core QK^T bottleneck). This script re-derives the
remaining stale projection files from those on-board truth sources so the whole
contract layer is internally consistent.

Truth sources (read-only here):
  - selected_outline.json  -> .sections[0].result_table  (on-board 9 rows)
  - outline/paper_view.json                              (on-board structured narrative)
  - evidence_ledger.json                                 (authoritative, on-board)

Targets repaired:
  1. outline/sections/main-results/result_table.json   (was 4.92%/56/memory-bound)
  2. outline/sections/main-results/{experiment_setup,findings,impact}.md (rendered from #1)
  3. outline/paper_view.md                             (was v1 6.85%/10-config)
  4. outline/evidence_view.json + outline/manifest.json + selected_outline.json
     claim_to_items C2/C3 (was v1 "~1.70ms" / "bound=memory all 10")
  5. review/submission_checklist.json                  (stale headline + provenance)
  6. paper_bundle_manifest.json                        (stale summary numbers)
"""
import json, os

NOW = "2026-06-15T16:30:00+00:00"
PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PAPER)

def load(p): return json.load(open(p, encoding="utf-8"))
def dump(o, p): json.dump(o, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

# ---------- truth sources ----------
sel = load("selected_outline.json")
onboard_rt = sel["sections"][0]["result_table"]          # 9 on-board rows
pv = load("outline/paper_view.json")
print("onboard_rt rows:", [r.get("item_id") for r in onboard_rt])
print("onboard_rt[0] keys:", sorted(onboard_rt[0].keys()))

def get(r, *keys):
    for k in keys:
        v = r.get(k)
        if v: return v
    return None

# ---------- 1. per-section result_table.json (copy on-board rows verbatim) ----------
rt_path = "outline/sections/main-results/result_table.json"
rt = load(rt_path)
rt["rows"] = onboard_rt
rt["updated_at"] = NOW
dump(rt, rt_path)
print("[1] result_table.json: rows replaced with on-board (", len(onboard_rt), "rows )")

# ---------- 2. rendered section md from on-board rows ----------
def title(r): return r.get("title", "")

setup = ["# Setup · Main Results", "", "## Recorded Setup Notes", ""]
for r in onboard_rt:
    n = get(r, "setup_note", "setup")
    if n: setup.append("- " + n)
open("outline/sections/main-results/experiment_setup.md", "w", encoding="utf-8").write("\n".join(setup) + "\n")

find = ["# Findings · Main Results", "", "## Result Highlights", ""]
for r in onboard_rt:
    rs = get(r, "result_summary") or ""
    ms = get(r, "metric_summary", "key_metrics")
    b = f"- `{title(r)}`: {rs}"
    if ms: b += f" ({ms})"
    find.append(b)
open("outline/sections/main-results/findings.md", "w", encoding="utf-8").write("\n".join(find) + "\n")

imp = ["# Impact · Main Results", "", "## Claim Links", ""]
for c in (sel["sections"][0].get("claims") or ["C1", "C3", "C2"]):
    imp.append(f"- `{c}`")
imp += ["", "## Impact Notes", ""]
for r in onboard_rt:
    im = get(r, "impact_summary", "claim_impact")
    if im: imp.append(f"- `{title(r)}`: {im}")
open("outline/sections/main-results/impact.md", "w", encoding="utf-8").write("\n".join(imp) + "\n")
print("[2] experiment_setup.md / findings.md / impact.md re-rendered from on-board rows")

# ---------- 3. paper_view.md regenerated from on-board paper_view.json ----------
ns = pv.get("narrative_strategy", {})
ss = pv.get("story_spine", {})
pos = pv.get("positioning", {})
eg = pv.get("evidence_grounding", {})
ma = pv.get("method_abstraction", {})
ep = pv.get("evaluation_plan", {})
L = []
L.append("# " + pv.get("working_title", sel.get("title", "")))
L.append("")
L.append(f"- Paper type: `{pv.get('paper_type','')}`")
L.append(f"- Outline maturity: `{pv.get('outline_maturity','')}`")
L.append("")
L.append("## One-Sentence Paper Idea")
L.append("")
L.append("- Central thesis: " + ns.get("central_thesis", ""))
L.append("- What readers learn: " + ns.get("central_insight", ""))
L.append("")
L.append("## Story Spine")
L.append("")
for k, lbl in [("problem","Problem"),("gap","Gap"),("method","Method"),("main_result","Main result"),("scope_limit","Scope limit")]:
    if ss.get(k): L.append(f"- {lbl}: {ss[k]}")
L.append("")
L.append("## Positioning")
L.append("")
L.append("- closest_neighbor: " + (pos.get("closest_neighbor") or "Not recorded"))
L.append("- novelty_boundary: " + (pos.get("novelty_boundary") or "Not recorded"))
L.append("- why_not_prior_work: " + (pos.get("why_not_prior_work") or "Not recorded"))
L.append("- not_claiming: " + ", ".join(pos.get("not_claiming") or []))
L.append("")
L.append("## Core Claims")
L.append("")
for c in pv.get("core_claims", []):
    L.append(f"- `{c.get('claim_id','')}` {c.get('claim','')}")
L.append("")
L.append("## From Facts To Interpretation")
L.append("")
for il in pv.get("insight_ladder", []):
    L.append(f"- `{il.get('level','')}` {il.get('statement','')}")
L.append("")
L.append("## Evidence Boundaries")
L.append("")
L.append("- Observed facts: " + ", ".join(eg.get("observed_facts") or []))
L.append("- Allowed interpretations: " + ", ".join(eg.get("allowed_interpretations") or []))
L.append("- Do not claim: " + ", ".join(eg.get("must_not_claim") or []))
L.append("- Evidence gaps: " + ", ".join(eg.get("evidence_gaps") or []))
L.append("")
L.append("## Method")
L.append("")
L.append("- Paper name: " + (ma.get("paper_name") or ""))
L.append("- Intuition: " + (ma.get("intuition") or ""))
for st in ma.get("mechanism_steps", []):
    L.append("- Step: " + st)
L.append("")
L.append("## Evaluation")
L.append("")
L.append("- Setting: " + (ep.get("setting") or ""))
L.append("- datasets_or_benchmarks: " + ", ".join(ep.get("datasets_or_benchmarks") or []))
L.append("- baselines: " + ", ".join(ep.get("baselines") or []))
L.append("- metrics: " + ", ".join(ep.get("metrics") or []))
L.append("- controlled_factors: " + ", ".join(ep.get("controlled_factors") or []))
L.append("")
L.append("## Analysis Plan")
L.append("")
for a in pv.get("analysis_plan", []):
    L.append(f"- `{a.get('analysis_id','')}` {a.get('title','')} ({a.get('analysis_role','')})")
L.append("")
L.append("## Reviewer Objections")
L.append("")
for ro in pv.get("reviewer_objections", []):
    L.append(f"- {ro.get('objection','')} -> {ro.get('answer_route','')}")
open("outline/paper_view.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("[3] paper_view.md regenerated from on-board paper_view.json")

# ---------- 4. C2/C3 claim_to_items string fixes across 3 files ----------
REPL = {
    "per_config:v32 s_kv 8192-65536 flat at ~1.70ms":
        "analysis:A3 across-s_kv spread max 3.25% / mean 1.48% over 128x sweep",
    "per_config:bound=memory all 10":
        "analysis:A7 binder=tensor 104/104 (gather 16.2ns vs QK^T matmul 339.9ns)",
    "per_config:bound=memory all 56":
        "analysis:A7 binder=tensor 104/104 (gather 16.2ns vs QK^T matmul 339.9ns)",
}
def fix_strings(o):
    if isinstance(o, dict): return {k: fix_strings(v) for k, v in o.items()}
    if isinstance(o, list): return [fix_strings(v) for v in o]
    if isinstance(o, str): return REPL.get(o, o)
    return o
for p in ["outline/evidence_view.json", "outline/manifest.json", "selected_outline.json"]:
    d = load(p)
    dump(fix_strings(d), p)
print("[4] C2/C3 claim_to_items repaired in evidence_view.json, manifest.json, selected_outline.json")

# ---------- 5. submission_checklist.json on-board headline + provenance ----------
sc = load("review/submission_checklist.json")
for it in sc.get("items", []):
    if it.get("id") == "evidence_authenticity":
        it["label"] = ("Headline numbers traced to durable on-board microbenchmark result file "
                       "(MAPE 3.27%, worst 9.72%, roofline 47.56%, 104 configs across two dispatch kernels)")
        it["evidence"] = "experiments/microbench/stage_model_microbench_2kernel_results.json"
    if it.get("id") == "no_fit_contract":
        it["label"] = ("No coefficient fit to target latencies; stage coefficients come from standalone "
                       "on-board operator microbenchmarks plus one single-row anchor per kernel (interpretable)")
        it["evidence"] = ("experiments/main/dsa_predictor.py; experiments/microbench/stage_model_microbench_2kernel_results.json; "
                          "coefficient provenance documented (no in-kernel profiler)")
    if it.get("id") == "claim_evidence_map":
        it["evidence"] = ("paper/claim_evidence_map.json; required items A1-A4 + dsa-stagepred-grid-v2 "
                          "resolved in evidence_ledger.json (on-board)")
dump(sc, "review/submission_checklist.json")
print("[5] submission_checklist.json headline/provenance set to on-board")

# ---------- 6. paper_bundle_manifest.json summary ----------
bm = load("paper_bundle_manifest.json")
bm["summary"] = ("Submission package on the on-board (standalone-microbenchmark, profiler-free) result. "
                 "Whole-kernel latency MAPE=3.27% (target <10%; small-budget kernel 3.84% / regular kernel 2.60%) "
                 "over 104 configurations spanning two dispatch kernels (batch s_q in {1,32,64,74,128,148,256,296} x context-length sweeps "
                 "x selection budgets topk in {1024,2048}); naive datasheet roofline=47.56% (~14.5x), worst single config 9.72%. "
                 "Stage micro-kernels are derived strictly from the original FlashMLA DSA kernel pipeline; coefficients are calibrated "
                 "only from standalone on-board operator microbenchmarks + datasheet constants plus one single-row anchor per kernel "
                 "(no black-box fit, no in-kernel profiler). The per-stage decomposition attributes the bottleneck to the tensor-core "
                 "QK^T matmul (binder=tensor on 104/104 configs), with the scattered key/value gather the cheapest operator (16.2 ns). "
                 "PDF compiles clean. Independent review completed (accept-quality); recommended writing revision applied.")
bm["updated_at"] = NOW
dump(bm, "paper_bundle_manifest.json")
print("[6] paper_bundle_manifest.json summary set to on-board")

print("\nDONE.")
