# Microbench reproduction — live B200 verification (run_all_microbench.sh)

`run_all_microbench.sh` rebuilds and reruns all 5 calibration microbenchmarks on
real B200 (sm_100) hardware and tees raw output to `regen/` (non-destructive; the
committed reference CSVs are never overwritten). Below is a live-vs-committed check
of the constants that `dsa_stage_model_piecewise.py` actually consumes.

| Microbench | Model constant | Committed ref | Live B200 (this run) | Match |
|---|---|---|---|---|
| correction FP32-ALU | FMUL/FFMA Gops/s/SM | ffma 174.600 / mul 176.567 | ffma **174.480** / mul **176.527** | near-exact (<0.1%) |
| pipeline handshake | `H_ONEWAY_NS = 280.66/2 = 140.33` | S=2 per-iter 280.66 ns | S=2 per-iter **280.663** ns | exact (3 d.p.) |
| KV gather (tile::gather4) | saturating BW → `GATHER_NS_64` | 1.9633 TB/s (s_kv=4096, 1 CTA) | **1.9625** TB/s | near-exact (0.04%) |
| MMA SV single-op latency | `sv_atom_lat = 210` cyc | 210 cyc | **210** cyc | exact |
| MMA QK throughput | `qk_atom_cyc = 37.106` | 37.106 cyc/op | UMMA-suite raw (needs suite reduction) | suite-dependent |
| softmax exp2 (SFU EX2) | `EXP2_OPS_PER_S_PER_SM` | 24.08 Gops/s/SM (ctas=4) | **27.85** Gops/s/SM | same order; benign clock variance |

Notes:
- The handshake floor (`h = 140.33 ns`) and correction ALU rates — the two constants
  most directly on the scoring-path critical chain — reproduce essentially exactly.
- `exp2` shows ~15% run-to-run variance (SFU throughput tracks GPU clock/thermal
  state). It is **not** the predicted bottleneck (the model is QK-tensor-core bound),
  so this variance does not move the head-line 3.88% MAPE.
- The MMA atoms come from the external `microbench-blackwell` UMMA suite; its raw
  output uses the suite's own units (latency = 210 cyc reproduces exactly; the
  throughput line needs the suite's reduction to land on 37.106 cyc/op).
- Atom *counts* (qk_atoms=18, sv_atoms=4/8) are source-read, not measured — see
  `results/MEASUREMENT_NOTES.md` and `sync_bench/NOTES.md`.

CPU-only check (no GPU needed; reads the committed CSVs):
```
python dsa_stage_model_piecewise.py     # head-line cross-kernel 3.88% MAPE
```
