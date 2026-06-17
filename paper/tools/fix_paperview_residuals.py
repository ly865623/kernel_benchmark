#!/usr/bin/env python3
"""Second pass: clear the v1 residuals the first pass did not target -- story_spine,
evaluation_plan benchmark description, and the load-bearing evidence_view metric values
(whole_kernel_mape 6.851->3.27, roofline_mape 19.516->47.56, worst 10.832->9.72).
Auditable: asserts each anchor present before replacing; writes only on full success."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # paper/

GAP = ("A naive datasheet-peak roofline mispredicts these kernels by roughly an order of "
       "magnitude (about 48% mean error, ~14x the proposed model) and offers no per-stage "
       "explanation, while opaque curve-fitting predicts numbers but is neither interpretable "
       "nor portable to a new architecture.")
METHOD = ("A stage-centric analytical model that decomposes the kernel into compute, contiguous "
          "query/output IO, scattered key/value gather, and fixed pipeline overhead, with two "
          "interpretable sparse-attention extensions: a data-dependent, wave-quantized tile count "
          "driven by the top-k budget and a single on-board pipeline-synchronization anchor that "
          "captures the warp-specialized residual the isolated operators miss; all coefficients "
          "come from standalone on-board microbenchmarks or datasheets, none fit to the target latencies.")
MAIN_RESULT = ("On a current Blackwell-class GPU the model predicts whole-kernel latency with 3.27% "
               "mean absolute percentage error across 104 configurations spanning two dispatch "
               "kernels, roughly 14x below the 47.6% naive-roofline error, while emitting a per-stage "
               "breakdown that correctly attributes the bottleneck to the tensor-core matmul.")
SCOPE_LIMIT = ("Validated on the sparse-prefill forward kernel of one sparse-attention family on a "
               "single Blackwell-class architecture over 104 configurations across two dispatch "
               "kernels; cross-architecture transfer is described as a coefficient-remeasurement "
               "protocol but is not yet empirically validated on a second device, and the worst single "
               "configuration sits at 9.72%, just within the sub-10% target.")
BENCH0 = ("Sparse-prefill forward kernel ground-truth latencies over 104 configurations spanning two "
          "dispatch kernels (small-core and regular) across context-length sweeps and a range of "
          "top-k selection budgets.")

METRIC_REPL = [
    ("metric:whole_kernel_mape_pct=6.851", "metric:whole_kernel_mape_pct=3.27"),
    ("metric:roofline_mape_pct=19.516", "metric:roofline_mape_pct=47.56"),
    ("metric:worst_config_abs_pct_err=10.832", "metric:worst_config_abs_pct_err=9.72"),
]

def get_pv(doc):
    if "paper_view" in doc:
        return doc["paper_view"]
    if "narrative_strategy" in doc and "story_spine" in doc:
        return doc
    return None

def patch_story(doc, report):
    pv = get_pv(doc)
    if not pv or "story_spine" not in pv:
        return
    ss = pv["story_spine"]
    assert "12-25%" in ss["gap"]; ss["gap"] = GAP; report.append("story_spine.gap")
    assert "effective gather bandwidth" in ss["method"]; ss["method"] = METHOD; report.append("story_spine.method")
    assert "6.85%" in ss["main_result"]; ss["main_result"] = MAIN_RESULT; report.append("story_spine.main_result")
    assert "ten configurations" in ss["scope_limit"]; ss["scope_limit"] = SCOPE_LIMIT; report.append("story_spine.scope_limit")
    ep = pv.get("evaluation_plan", {})
    dob = ep.get("datasets_or_benchmarks")
    if dob and "ten configurations" in dob[0]:
        dob[0] = BENCH0; report.append("evaluation_plan.datasets_or_benchmarks[0]")

def patch_metrics(doc, report):
    """Replace metric strings anywhere in the document (covers evidence_view in any layout)."""
    n = [0]
    def walk(o):
        if isinstance(o, dict):
            for k in list(o.keys()):
                o[k] = walk(o[k])
            return o
        if isinstance(o, list):
            return [walk(x) for x in o]
        if isinstance(o, str):
            for old, new in METRIC_REPL:
                if o == old:
                    n[0] += 1
                    return new
        return o
    walk(doc)
    if n[0]:
        report.append(f"evidence_view metric replacements: {n[0]}")

FILES = ["selected_outline.json", "outline/manifest.json", "outline/paper_view.json", "outline/evidence_view.json"]

def main():
    parsed, reports = {}, {}
    for rel in FILES:
        p = os.path.join(ROOT, rel)
        with open(p) as f:
            doc = json.load(f)
        rep = []
        patch_story(doc, rep)
        patch_metrics(doc, rep)
        parsed[p] = doc
        reports[rel] = rep
    for p, doc in parsed.items():
        with open(p, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
    for rel, rep in reports.items():
        print(f"=== {rel} ({len(rep)} edits) ===")
        for r in rep:
            print("  -", r)
    print("OK: residuals cleared.")

if __name__ == "__main__":
    main()
