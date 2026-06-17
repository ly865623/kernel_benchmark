# Independent Skeptical Review — On-Board (Profiler-Free) Manuscript

- paper_line: `paper-dsa-stagepred-grid-v2-outline-001-dsa-stagepred-grid-v2`
- outline_ref: `outline-001`
- review_date: 2026-06-15
- reviewer_stance: independent skeptical audit of the **on-board (standalone-microbenchmark, profiler-free)** manuscript before finalize
- supersedes: the 2026-06-12 review (`review.md` profiler-era, MAPE 4.92% / 56 configs) — that audit reviewed the retired v2 profiler-composed result and is no longer valid.
- truth source for every number below: `paper/evidence_ledger.json` (on-board), cross-checked against `experiments/microbench/stage_model_microbench_2kernel_results.json` and `paper/latex/main.tex`.

## Verdict

**Accept-quality as a focused, honest performance-modeling contribution, scoped to a single Blackwell-class device.** The headline claim is well supported and the methodology is unusually transparent (no coefficient fit to target latencies). The main reviewer-facing risks are (i) breadth — one device, one kernel family, baseline is only the naive datasheet roofline; and (ii) a modeling-honesty point that must stay visible: the isolated operators explain under 40% of per-iteration time and a single measured anchor supplies the rest. None of these require new experiments; they require honest scoping, which the current draft already does. No new experiments required.

## What was audited

- Manuscript `paper/latex/main.tex` (EN+ZH), compiled PDF `paper/latex/main.pdf` (compile_report: `compile_ok=true`, recompiled 2026-06-15 after the on-board rewrite; 0 profiler references).
- Evidence ledger, selected outline, per-section result table, experiment matrix.
- Verified the manuscript is on-board: 12× "3.27", 12× "104", 0 stale live claims. The single "56 configs" occurrence is the legitimate small-budget-kernel subset count (56 of the 104), with its own MAPE 3.84% — not the retired 56-config profiler grid.

## Claim-by-claim audit

### C1 — sub-10% mean error, ~order-of-magnitude better than roofline, no target fit
- Support: MAPE **3.27%** over **104 configs** (small-budget kernel 3.84% / regular 2.60%); worst single config **9.72%**; naive datasheet roofline **47.56%** on the identical grid (~14.5×); model wins 104/104 configs. No constant is fit to target latencies; the only on-device whole-kernel datum is one single-row anchor per kernel (19.61 µs small / 32.28 µs regular).
- Skeptical pushback: the baseline is the *weakest* obvious one (naive datasheet roofline). The paper already answers this with the zero-anchor and no-wave ablations (A4/A5), which are the right move — but a reviewer may still ask for a tuned/curve-fit baseline as an upper-reference. **Recommendation: writing-only** — keep the "no-fit contract" framing prominent; optionally note in limitations that stronger learned baselines are out of scope by design (interpretability-first).
- Verdict: **supported.**

### C2 — latency governed by top-k selection budget, ~independent of context length
- Support: A3 — at fixed selection budget, measured whole-kernel latency is flat across a 128× context-length (s_kv) sweep: worst across-s_kv spread 3.25%, mean 1.48%. Matches the wave-/budget-driven tile count.
- Skeptical pushback: flatness could partly reflect a measurement floor; the draft acknowledges this and corroborates with the modeled tile-count flatness. Adequate.
- Verdict: **supported.**

### C3 — interpretable, correct bottleneck attribution (tensor-core bound; gather cheapest); portable by re-measurement
- Support: A2/A7 — standalone on-board operator microbenchmarks give binder = tensor-core QK^T matmul on **104/104** configs (339.9 ns), with the scattered gather the **cheapest** operator (16.2 ns), i.e. the kernel is compute-bound at the operator layer, overturning the roofline/"sparse-gather ⇒ memory-bound" intuition. A8 specifies the coefficient-remeasurement portability protocol.
- **Most important honesty point:** the four isolated operators under perfect overlap sum to only ~39% (471.5 ns) of the measured ~1220 ns per iteration; the remaining ~61% (749 ns) is a warp-specialized pipeline-synchronization residual supplied by **one measured anchor**. This is disclosed (A5 zero-anchor ablation: operator-sum-only variants reach 60.5%/44.6% MAPE, worse than roofline). A reviewer could read "compute-bound from microbenchmarks" as overstated when most of the time is an unmodeled sync residual captured by a calibrated anchor. **Recommendation: writing-only** — the attribution claim must stay phrased as "operator-layer bound classification + a measured pipeline-sync residual," exactly as the ledger states; do not let any summary sentence imply the isolated operators alone explain the latency.
- Portability is a protocol, **not** an empirical transfer result (no second device). Correctly scoped.
- Verdict: **supported, with the residual disclosure as a non-negotiable framing constraint.**

## Cross-cutting weaknesses (all already scoped honestly in the draft)
1. Single device (one Blackwell-class GPU); cross-architecture is protocol-only. Stated as a limitation.
2. One sparse-attention kernel family (FlashMLA DSA sparse-prefill forward). Stated.
3. Worst single config 9.72% — just inside the sub-10% target; A6 shows it is a structured, sign-consistent first-wave under-prediction, reported openly rather than smoothed.
4. Baseline breadth: only the naive datasheet roofline; mitigated by ablations rather than additional external baselines.

## Findings this round (contract-integrity audit, not science)

The science is unchanged and correct. The audit found and repaired a **contract-layer consistency failure** that prior turns had repeatedly reported as fully resolved when it was not:

- 9 derived/contract files on the active paper branch still carried retired v1 (10-config) or v2 (56-config, memory-bound) numbers presented as current truth: `outline/sections/main-results/result_table.json` + the three rendered section `.md`, `outline/paper_view.md`, the C2/C3 claim items in `evidence_view.json` / `manifest.json` / `selected_outline.json`, `review/submission_checklist.json`, and `paper_bundle_manifest.json`. All re-derived to on-board diction from `evidence_ledger.json` via the idempotent, auditable script `paper/tools/fix_contract_residuals_v3.py`. Post-repair full-tree scan: **0 stale-as-current lines.** (commit 036e9fe)

### Open residuals routed to finalize/merge (none block the science or the PDF)
1. **Two-tree state drift.** The authoritative on-board contract files live on the active paper branch (worktree); the repo-root `main` tree is many commits behind and still carries old-era `evidence_ledger.{json,md}`, `review/review.md`, `submission_checklist.json`, and outline C2/C3. The on-board, compiling `main.tex`/PDF live only at repo-root. These two trees were never consolidated, which is why each prior pass "found one more" stale file (it was inspecting one tree at a time). **Action: consolidate the on-board paper-branch contract files into `main` so a single authoritative on-board state exists.** Performed this round (see consolidation commit).
2. **Outline-registry `central_insight` cache.** `validate_academic_outline` reads a runtime registry field that still holds the pre-on-board memory-bandwidth wording. Every on-disk file (manifest paper_view, ledger, manuscript) is on-board; the central *thesis* is on-board. This field does not enter the PDF, bundle, evidence ledger, or submission gate (`submission_ready=true`). Clearing it requires one full-payload `submit_paper_outline(revise)`; that operation re-materialized stale snapshots over good files **twice** earlier in this quest, so it is deliberately **not** attempted blind. Recovery path if desired: re-register with the complete on-board `detailed_outline` (paper_view + evidence_view + sections) in a single call, then re-verify and `git`-restore on any regression.
3. **Stale handoff snapshot.** `handoffs/final-deliverable-v2/` is a 2026-06-12 profiler-era export (8 stale markers, 0 on-board). Marked SUPERSEDED pointing to the current on-board `paper/latex`.
4. **Coverage tool `compile_ok=false` vs report `compile_ok=true`.** `validate_manuscript_coverage` reports `compile_ok=false` while `paper/latex/compile_report.json` records a clean recompile (PDF newer than tex). Treated as a tool-reading discrepancy; a fresh `pdflatex` pass before actual submission is advisable to confirm.

## Route recommendation
- The active paper line is on-board, internally consistent, and review-passed as a scoped single-device performance-modeling paper.
- Remaining work is **consolidation + finalize**, not new science: (a) one authoritative on-board tree (done this round), (b) optionally clear the registry `central_insight` cache via a single complete-payload re-registration, (c) confirm a clean LaTeX recompile.
- No new experiments are required for the current claims.
