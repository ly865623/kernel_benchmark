# top-k scan v2 — provenance

- dataset: grid_topk_v2.json (104 rows), grid_topk_v2.csv
- comparator: flashmla-dsa-sparse-prefill-b200-topk-scan-v2
- device: NVIDIA B200 (sm100), torch 2.10.0a0+...nv26.01, CUDA 13.1
- kernel: FlashMLA commit 48c6dc4, **production build (NO -DFLASHINFER_ENABLE_PROFILER)** — rebuilt 2026-06-12T13:19 in container ds003-flashmla
- equivalence check: topk=1024 @ bs=74,s_kv=8192 = 21.65 us vs published grid_v2 21.53 us (+0.54%, within noise)
- grid: topk in {1024,2048} x batch in {1,32,64,74,128,148,256,296} x s_kv in {1k,4k,8k,16k,32k,64k,128k}
  - kernel constraint: topk multiple of B_TOPK=128 and topk<=s_kv; clamped+deduped -> 104 effective configs
  - s_kv=1024 admits only topk=1024 (topk=2048 clamps to 1024)
- key finding: latency(topk=2048)/latency(topk=1024) = 1.614 mean (min 1.584, max 1.674) across 48 matched pairs -> sub-2x (fixed-overhead amortization)
- measurement: kk.bench_kineto sparse_attn_fwd kernel time, num_tests=30, passes=3, median; spreads <0.5%
- scope: SEPARATE from published grid_v2.json; does NOT modify paper headline numbers (MAPE 4.92%)
