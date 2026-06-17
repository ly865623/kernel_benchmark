#!/usr/bin/env python3
"""Faithfully rewrite the paper_view narrative layer (central_thesis, insight_ladder,
core_claims, evidence_grounding, analysis_plan, positioning) from the stale v1
memory-bound / gather-dominant / 10-config story to the on-board microbenchmark story
(3.27% MAPE over 104 configs, tensor-core QK^T bound, gather cheapest), matching the
already-repaired evidence_ledger.json. Auditable: asserts the stale anchor text is
present before replacing, prints a hit report, and writes only on full success.

Targets all three outline files that carry an identical paper_view:
  - paper/selected_outline.json        (paper_view under /paper_view)
  - paper/outline/manifest.json        (paper_view under /paper_view)
  - paper/outline/paper_view.json      (paper_view at root)
"""
import json, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # paper/

# ---- On-board canonical text (grounded in evidence_ledger.json + on-board central_insight) ----
CENTRAL_THESIS = (
    "A closed-form stage decomposition, calibrated only from standalone on-board "
    "microbenchmarks of the kernel's own operators plus datasheet constants, predicts "
    "the whole-kernel latency of data-dependent sparse-attention kernels within sub-10% "
    "error while staying fully interpretable, because the cost is set by the discrete "
    "occupancy-wave count and the top-k selection budget, and because the dominant "
    "per-iteration cost is the kernel's tensor-core matmul together with a warp-specialized "
    "pipeline-synchronization residual supplied by a single on-board anchor -- not by "
    "datasheet peaks, the full context length, or the scattered gather a sparse kernel is "
    "named for."
)
CENTRAL_INSIGHT = (
    "For modern sparse-attention prefill kernels, what a performance model must get right "
    "is not raw FLOP/byte peaks but (a) the data-dependent amount of work induced by "
    "sparsity (the occupancy-wave count and top-k iterations) and (b) the realistic on-board "
    "cost of the kernel's own operators measured in isolation; the surprising finding is "
    "that the scattered gather is the cheapest operator and the kernel is tensor-core bound, "
    "and that the isolated operators explain under 40% of per-iteration time -- the remainder "
    "being warp-specialized pipeline synchronization captured by one anchor."
)

# insight_ladder[1] (was: 'Every configuration is classified memory-bound ... gather term dominating')
IL1_STATEMENT = (
    "Standalone on-board operator microbenchmarks show the kernel is tensor-core (QK^T matmul) "
    "bound while the scattered key/value gather is the single cheapest operator; this overturns "
    "the roofline-style memory-bandwidth intuition and teaches that the decisive coefficient is "
    "the measured tensor-core throughput, not a de-rated gather bandwidth."
)
IL1_RISK = (
    "The compute/memory boundary is anchored in standalone on-board operator measurements rather "
    "than a model-internal classification; a zero-anchor ablation shows the isolated operator "
    "costs alone explain under 40% of per-iteration time, motivating the pipeline-synchronization anchor."
)
# insight_ladder[2] (was: '6.85% mean error while the naive datasheet roofline sits at 19.5%')
IL2_STATEMENT = (
    "The analytical model reaches 3.27% mean error across 104 configurations spanning two dispatch "
    "kernels while the naive datasheet roofline sits at 47.6% (~14x worse); this teaches that the "
    "accuracy gain comes from stage-level structure -- wave-quantized occupancy plus a single on-board "
    "pipeline-synchronization anchor -- rather than from tuning, since no coefficient is fit to the target latencies."
)

# core_claims
C1_CLAIM = (
    "A closed-form stage-centric model calibrated only from standalone on-board microbenchmarks predicts "
    "sparse-prefill whole-kernel latency with sub-10% mean error and roughly an order of magnitude lower "
    "error than a naive datasheet roofline, with no coefficient fit to the target latencies."
)
C1_SCOPE = (
    "Sparse-attention sparse-prefill forward kernel on one Blackwell-class GPU, 104 configurations across "
    "two dispatch kernels (small-core and regular) covering selection budgets and context-length sweeps."
)
C3_CLAIM = (
    "The per-stage decomposition yields an interpretable, correct bottleneck attribution -- the kernel is "
    "tensor-core (QK^T matmul) bound on every configuration while the scattered key/value gather is the "
    "cheapest operator -- and the same closed form is portable to other Blackwell-class accelerators by "
    "re-measuring its on-board coefficient table."
)

OBS_FACTS = [
    "Whole-kernel MAPE is 3.27% over 104 configurations across both dispatch kernels (3.84% small-core / 2.60% regular); worst single-config absolute error is 9.72%.",
    "Naive datasheet-peak roofline MAPE is 47.56% on the same configurations (~14x worse).",
    "Measured latency is flat (within ~1%) across an 8x context-length sweep at fixed selection budget.",
    "Standalone on-board operator microbenchmarks show every configuration is tensor-core (QK^T matmul) bound, with the scattered key/value gather the single cheapest operator (16.2 ns).",
    "All model coefficients are sourced from standalone on-board microbenchmarks or datasheet values; none are fit to the target latencies.",
]
ALLOWED_INTERP0 = (
    "Stage-level structure (wave-quantized occupancy from the selection-driven tile count plus a single "
    "on-board pipeline-synchronization anchor) explains most of the accuracy gain over roofline."
)

# analysis_plan: realign titles/reviewer_questions to the on-board A1-A8 family (matches evidence_ledger)
AP_FIXES = {
    1: {"title": "Per-operator on-board decomposition and bottleneck attribution",
        "reviewer_question": "Does the model's bottleneck attribution (tensor-core bound, gather cheapest) reflect standalone on-board operator measurements rather than a modeling artifact?"},
    3: {"title": "Wave-quantization-term ablation",
        "reviewer_question": "How much of the accuracy gain comes from modeling the discrete occupancy-wave count rather than a continuous occupancy proxy?",
        "target_display": "Main-text ablation table: full model vs (continuous-occupancy proxy) vs (no wave term)"},
    4: {"title": "Anchor necessity: zero-anchor ablation",
        "reviewer_question": "Are the isolated on-board operator costs alone sufficient, or is the single warp-specialized pipeline-synchronization anchor required?",
        "target_display": "Ablation: predicted vs measured per-iteration time with and without the pipeline-synchronization anchor"},
    6: {"title": "Operator-layer bound classification across the config space",
        "reviewer_question": "Is the kernel compute (tensor-core) bound across shape families and budgets, or does the classification flip by regime?",
        "failure_interpretation": "If some shape families flip to memory-bound, scope the tensor-core-bound attribution to the measured regime and report the boundary explicitly."},
}

NOVELTY_BOUNDARY_OLD_FRAG = "measured gather bandwidth"
# We rewrite the (2) clause of novelty_boundary.

def patch_paper_view(pv, report):
    ns = pv["narrative_strategy"]
    # central_thesis (file copy may be old; force on-board)
    ns["central_thesis"] = CENTRAL_THESIS
    report.append("set narrative_strategy.central_thesis -> on-board")
    ns["central_insight"] = CENTRAL_INSIGHT
    report.append("set narrative_strategy.central_insight -> on-board")

    il = pv["insight_ladder"]
    assert "memory-bound" in il[1]["statement"], "insight_ladder[1] anchor missing"
    il[1]["statement"] = IL1_STATEMENT
    il[1]["risk"] = IL1_RISK
    il[1]["claim_links"] = ["C3"]
    report.append("rewrote insight_ladder[1] (memory-bound -> tensor-core bound)")
    assert "6.85%" in il[2]["statement"], "insight_ladder[2] anchor (6.85%) missing"
    il[2]["statement"] = IL2_STATEMENT
    report.append("rewrote insight_ladder[2] (6.85%/19.5% -> 3.27%/47.6%)")

    cc = pv["core_claims"]
    assert "ten configurations" in cc[0]["scope"], "core_claims[0] scope anchor missing"
    cc[0]["claim"] = C1_CLAIM
    cc[0]["scope"] = C1_SCOPE
    report.append("rewrote core_claims C1 (ten configs -> 104 configs / two kernels)")
    assert "memory-bound" in cc[2]["claim"], "core_claims[2] anchor missing"
    cc[2]["claim"] = C3_CLAIM
    report.append("rewrote core_claims C3 (memory-bound/gather -> tensor-core bound)")

    eg = pv["evidence_grounding"]
    of = eg["observed_facts"]
    assert any("6.85%" in x for x in of), "observed_facts 6.85% anchor missing"
    eg["observed_facts"] = OBS_FACTS
    report.append("replaced evidence_grounding.observed_facts (5 facts -> on-board)")
    ai = eg["allowed_interpretations"]
    assert "measured gather bandwidth" in ai[0], "allowed_interpretations[0] anchor missing"
    ai[0] = ALLOWED_INTERP0
    report.append("rewrote allowed_interpretations[0] (gather bandwidth -> wave/anchor)")

    ap = pv["analysis_plan"]
    for idx, fix in AP_FIXES.items():
        for k, v in fix.items():
            ap[idx][k] = v
        report.append(f"realigned analysis_plan[{idx}] -> {fix['title']}")

    pos = pv["positioning"]
    nb = pos.get("novelty_boundary", "")
    if NOVELTY_BOUNDARY_OLD_FRAG in nb:
        pos["novelty_boundary"] = (
            "The new and reusable contributions are (1) treating the sparse selection budget as the "
            "latency-determining work unit via a data-dependent, wave-quantized tile count, and (2) a "
            "single on-board pipeline-synchronization anchor that, together with standalone operator "
            "microbenchmarks, captures the per-iteration cost the isolated operators miss -- yielding a "
            "tensor-core-bound attribution rather than the gather-bound one a roofline would assume."
        )
        report.append("rewrote positioning.novelty_boundary ((2) gather bandwidth -> on-board anchor)")
    return pv

def locate_pv(doc):
    if "paper_view" in doc:
        return doc["paper_view"], lambda new: doc.__setitem__("paper_view", new)
    # root-level paper_view (paper_view.json)
    if "narrative_strategy" in doc and "insight_ladder" in doc:
        return doc, None  # patched in place
    raise SystemExit("could not locate paper_view")

FILES = ["selected_outline.json", "outline/manifest.json", "outline/paper_view.json"]

def main():
    all_reports = {}
    parsed = {}
    for rel in FILES:
        p = os.path.join(ROOT, rel)
        with open(p) as f:
            doc = json.load(f)
        pv, setter = locate_pv(doc)
        report = []
        patch_paper_view(pv, report)
        parsed[p] = doc
        all_reports[rel] = report
    # all asserts passed for all files -> write
    for p, doc in parsed.items():
        with open(p, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
    for rel, rep in all_reports.items():
        print(f"=== {rel} ({len(rep)} edits) ===")
        for r in rep:
            print("  -", r)
    print("OK: all three files rewritten to on-board paper_view.")

if __name__ == "__main__":
    main()
