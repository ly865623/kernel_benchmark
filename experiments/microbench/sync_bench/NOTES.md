# Constraint-compliant pipeline-synchronization microbench — measurement notes (B200, sm_100a)

Route A (decision-487c400c): predict the DSA kernel's per-k-iteration "pipeline synchronization
residual" (749 ns small / 1157.6 ns regular) **without running the fused operator**, using only
isolated primitive microbenchmarks + source-read pipeline structure.

## What this bench measures
`mbar_pipeline.cu`: a warp-specialized multi-stage producer->consumer pipeline using raw PTX
`mbarrier.arrive` / `mbarrier.try_wait.parity`, with NO matmul/gather/softmax payload. S stages
(= S warps), S-1 mbarrier-coupled boundaries, B-deep buffering. Self-timed by the sink warp's
clock64() over ITERS iterations. This isolates the cross-warp mbarrier handshake-chain cost — the
synchronization coupling that bottom-up *operator* microbenchmarks cannot see.

## Findings (clk 1.965 GHz)

### Handshake floor vs #stages (B=4, payload-free)
| S (stages) | boundaries | per-iter ns |
|---|---|---|
| 2 | 1 | 280.7 |
| 3 | 2 | 359.5 |
| 4 | 3 | 353.2 |
| 5 | 4 | 353.2 |
| 6 | 5 | 366.5 |
| 7 | 6 | 368.3 |

KEY: the handshake cost **saturates at ~350-370 ns/iter** beyond 2 stages. The multi-level
producer/consumer chain (user's correction — it IS multi-level) does NOT add latency linearly:
with buffering the stages pipeline, so steady-state period = bottleneck warp's per-iter mbarrier
loop, not the sum over levels. S=2 is cheaper because its sink/source warps do only one handshake.

### Handshake floor vs buffer depth (S=6, payload-free)
B=2:364.5  B=3:365.0  B=4:365.5  B=6:366.5  B=8:367.5 ns  -> **flat in B**. Buffer depth does not
change the steady-state period (it only changes startup decoupling), consistent with the kernel's
NUM_K_BUFS=4 being a latency-hiding device, not a throughput knob.

=> CLEAN PRIMITIVE: t_handshake_floor ~= 365 ns/iter for this warp-specialized pipeline pattern.

## Interpretation toward the model
- DSA per-iter (small) = 1220 ns. Operator overlap envelope (throughput) = 471 ns (39%).
- Pure handshake floor (this bench) ~= 365 ns. 471 + 365 = 836 ns; still < 1220.
- Remaining ~384 ns must be **exposed instruction latency** along the intra-iteration recurrence
  P=QK^T -> S=softmax(P) -> O=S·V (serial within an iteration), which throughput-overlap omits.
  These latencies are ALREADY measured as isolated primitives:
    umma single-op latency: QK(M128N128K16)=178 cyc=90.6ns, SV(M128N256K16)=210 cyc=106.9ns
    (mma_costs.csv; from microbench-blackwell/umma_latency, NOT the fused kernel)
  plus TMA gather completion latency (tma2d_latency bench) and exp2 SFU latency.

## CAVEAT (fidelity, must address before claiming)
- The synthetic per-stage latency injection (`spin_cycles`) gave noisy, non-monotonic results
  (L=100->399, L=200->375, L=400->399 ns) — it is NOT a reliable way to add exposed latency.
  DECISION: do NOT inject synthetic latency. Instead compose t_sync analytically:
    t_sync_pred = t_handshake_floor(measured ~365ns) + exposed_recurrence_latency(measured umma/tma/sfu
                  latencies, structured by the source dependency chain), ZERO fitting to whole-kernel.
- Open question: whether 365ns handshake + measured exposed latencies reconstruct 749/1157 ns
  closely enough to beat the 44.6% pure-operator floor and approach sub-10%. This is the decisive
  test (next step). If it cannot, Route B (honest reframe).

### Buffer-depth sweep (sweep_B.csv) — DECISIVE primitive
Per-iter period is **buffer-INSENSITIVE** in the payload-free bench: B=1==B=2==B=4 to 3 d.p.
| S | per-iter ns (any B) |
|---|---|
| 2 (1 boundary, ping-pong) | 280.66 |
| 3 (2 boundaries) | 352.67 |
| 4 | 352.68 |
| 6 | 364.38 |
=> the S=2 ping-pong = one full cross-warp round-trip (fwd signal + buffer-free signal) =
280.66 ns => **one-way cross-warp mbarrier signal latency h = 140.33 ns**. This is the
exposed per-boundary cost on the kernel's SINGLE-buffered scoring path (get<1>), where
stages cannot overlap — unlike the 4-buffered KV path whose latency is hidden.

## DECISIVE RESULT — composed model (dsa_stage_model_composed.py, 104-cfg grid)
t_step = max(t_overlap, t_scoring); t_scoring = chain_lat(QK) + h + t_softmax + h +
chain_lat(SV) + h. Multi-level structure respected: 3 EXPOSED handshakes on the
single-buffered QK->softmax->SV recurrence; KV 4-buffered path hidden under it.
ZERO fit to any whole-kernel latency.
| model | BOTH | small | regular |
|---|---|---|---|
| **COMPOSED (constraint-compliant)** | **9.24%** | 4.92% | 14.28% |
| bottom-up overlap floor | 60.47% | 62.50% | 58.10% |
| naive serial sum | 44.62% | 48.22% | 40.42% |
| roofline | 47.56% | — | 44.27% |
| anchored oracle (FORBIDDEN ref) | 3.27% | 3.84% | 2.60% |
Small sync_resid_pred=731.8ns vs forbidden oracle 749.2ns (2.3% off) WITHOUT using the
oracle. Regular underpredicts (899 vs 1157, 22%) -> NOT covert fitting (same h/formula).
=> Route A SUCCEEDS: sub-10% aggregate, crushes 44.6% bottom-up floor, fully compliant.

## Raw data: sweep_S.csv, sweep_B.csv
