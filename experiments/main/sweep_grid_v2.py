#!/usr/bin/env python3
"""
Dense whole-kernel latency sweep for the FlashMLA DSA sparse-prefill kernel on B200.

Grid (per user requirement 2026-06-12):
    batch_size (= s_q, number of query rows) in {1, 32, 64, 74, 128, 148, 256, 296}
    s_kv                                      in {1k, 4k, 8k, 16k, 32k, 64k, 128k}
  => 8 x 7 = 56 configs.

Why these batch sizes: the kernel launches grid = dim3(2*s_q) with a 2-CTA cluster, so each
query row occupies one 2-SM tile. B200 has 148 SMs => 74 two-SM tiles per wave. Thus
{74,148,296} = {1,2,4} full waves and {1,32,64,128,256} probe sub-wave / tail occupancy.
This is a deliberate wave-quantization sweep.

Fixed structural config (DSA "v32" head128 family):
    h_q=128, h_kv=1, d_qk=576, d_v=512, topk=1024 (<= smallest s_kv=1024, multiple of B_TOPK=128),
    have_attn_sink=True. topk is fixed so compute is constant across s_kv and s_kv isolates the
    scatter-gather address-spread effect on effective KV bandwidth.

Measures the `sparse_attn_fwd` kernel time via kk.bench_kineto (L2-flushed), median over `passes`
independent measurement passes, plus FLOPs / mem-vol from the repo's own accounting and the
naive datasheet roofline for context. Reuses the unmodified FlashMLA test harness.

Run inside container ds003-flashmla (nvcr.io/nvidia/pytorch:26.01-py3-v0).
"""
import argparse, json, os, sys, statistics, csv, time

FM = os.environ.get("FLASHMLA_ROOT", "/workspace/code/FlashMLA")
sys.path.insert(0, os.path.join(FM, "tests"))
sys.path.insert(0, FM)

import torch
import kernelkit as kk
import lib
from lib import TestParam

B200_TENSOR_PEAK_BF16 = 2250.0   # TFLOPS datasheet peak (bf16 inputs)
B200_HBM_PEAK_TBPS = 8.0         # TB/s datasheet peak

BATCH_SIZES = [1, 32, 64, 74, 128, 148, 256, 296]      # = s_q
S_KV_LIST   = [1024, 4096, 8192, 16384, 32768, 65536, 131072]
NUM_SM = 148
TILES_PER_WAVE = NUM_SM // 2   # 2-SM cluster -> 74 tiles/wave


def measure_one(p: TestParam, num_tests: int, passes: int):
    if p.seed == -1:
        p.seed = 0
    torch.cuda.empty_cache()
    t = lib.generate_testcase(p)
    torch.cuda.synchronize()

    def run():
        return lib.run_flash_mla_sparse_fwd(p, t, False)

    run(); torch.cuda.synchronize()   # warm-up / functional

    fm = lib.count_flop_and_mem_vol(p, t)

    lat = []
    for _ in range(passes):
        res = kk.bench_kineto(run, num_tests=num_tests)
        lat.append(res.get_kernel_time("sparse_attn_fwd"))  # seconds
        torch.cuda.synchronize()
    lat_med = statistics.median(lat)
    lat_min, lat_max = min(lat), max(lat)
    spread = (lat_max - lat_min) / lat_med if lat_med > 0 else float("nan")

    tflops = fm.fwd_flop / lat_med / 1e12
    mem_bw = fm.fwd_mem_vol / lat_med / 1e12

    t_compute = fm.fwd_flop / (B200_TENSOR_PEAK_BF16 * 1e12)
    t_mem = fm.fwd_mem_vol / (B200_HBM_PEAK_TBPS * 1e12)
    roofline_s = max(t_compute, t_mem)
    roofline_rel_err = abs(roofline_s - lat_med) / lat_med

    num_tiles = p.s_q                       # one 2-SM tile per query row
    num_waves = (num_tiles + TILES_PER_WAVE - 1) // TILES_PER_WAVE
    wave_fill = num_tiles / (num_waves * TILES_PER_WAVE)   # occupancy efficiency of the last wave set
    k_tiles = (p.topk + 127) // 128         # KV-block iterations per row (B_TOPK=128)

    return {
        "kernel": "sparse_attn_fwd", "stage": "prefill",
        "d_qk": p.d_qk, "d_v": p.d_v, "h_q": p.h_q, "h_kv": p.h_kv,
        "batch_size": p.s_q, "s_q": p.s_q, "s_kv": p.s_kv, "topk": p.topk,
        "num_tiles": num_tiles, "num_waves": num_waves, "wave_fill": wave_fill, "k_tiles": k_tiles,
        "fwd_flop": fm.fwd_flop, "fwd_mem_vol_bytes": fm.fwd_mem_vol,
        "latency_us": lat_med * 1e6, "latency_us_min": lat_min * 1e6, "latency_us_max": lat_max * 1e6,
        "latency_spread_frac": spread, "latency_passes_us": [x * 1e6 for x in lat],
        "tflops": tflops, "mem_bw_tbps": mem_bw,
        "roofline_pred_us": roofline_s * 1e6, "roofline_rel_err": roofline_rel_err,
        "roofline_bound": "compute" if t_compute >= t_mem else "memory",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/quest/.ds/worktrees/idea-idea-98be86a0/experiments/main/grid_v2")
    ap.add_argument("--num-tests", type=int, default=30)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("--d-qk", type=int, default=576)
    ap.add_argument("--smoke", action="store_true", help="single tiny config only")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.float16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    os.makedirs(os.path.join(args.out, "json"), exist_ok=True)
    print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)

    if args.smoke:
        grid = [(1, 8192), (74, 8192), (296, 131072)]
    else:
        grid = [(b, s) for b in BATCH_SIZES for s in S_KV_LIST]

    rows = []
    t0 = time.time()
    for i, (b, s_kv) in enumerate(grid):
        topk = min(args.topk, s_kv)
        # topk must be a multiple of B_TOPK=128 and <= s_kv
        topk = max(128, (topk // 128) * 128)
        p = TestParam(s_q=b, s_kv=s_kv, topk=topk, h_q=128, d_qk=args.d_qk, d_v=512,
                      have_attn_sink=True, check_correctness=False, num_runs=0)
        try:
            r = measure_one(p, args.num_tests, args.passes)
            rows.append(r)
            print(f"[{i+1}/{len(grid)}] bs={b:>4} s_kv={s_kv:>7} topk={topk:>5} "
                  f"waves={r['num_waves']} fill={r['wave_fill']:.3f} -> "
                  f"{r['latency_us']:8.2f} us  {r['tflops']:7.1f} TFLOPS  {r['mem_bw_tbps']:5.2f} TB/s  "
                  f"spread={r['latency_spread_frac']*100:.2f}%  rfl_err={r['roofline_rel_err']*100:.1f}%",
                  flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(grid)}] bs={b} s_kv={s_kv} topk={topk} FAILED: {e}", flush=True)

    payload = {
        "comparator": "flashmla-dsa-sparse-prefill-b200-grid-v2",
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "eval_path": "kk.bench_kineto(flash_mla.flash_mla_sparse_fwd).get_kernel_time('sparse_attn_fwd')",
        "num_tests": args.num_tests, "passes": args.passes,
        "fixed": {"h_q": 128, "h_kv": 1, "d_qk": args.d_qk, "d_v": 512, "topk": args.topk, "have_attn_sink": True},
        "num_sm": NUM_SM, "tiles_per_wave": TILES_PER_WAVE,
        "batch_sizes": BATCH_SIZES, "s_kv_list": S_KV_LIST,
        "metric_directions": {"latency_us": "lower_better", "tflops": "higher_better", "mem_bw_tbps": "higher_better"},
        "elapsed_s": time.time() - t0,
        "rows": rows,
    }
    jpath = os.path.join(args.out, "json", "grid_v2.json")
    with open(jpath, "w") as f:
        json.dump(payload, f, indent=2)
    print("WROTE", jpath, "rows=", len(rows), flush=True)

    if rows:
        cpath = os.path.join(args.out, "grid_v2.csv")
        keys = ["batch_size","s_q","s_kv","topk","num_tiles","num_waves","wave_fill","k_tiles",
                "d_qk","d_v","h_q","fwd_flop","fwd_mem_vol_bytes","latency_us","latency_us_min",
                "latency_us_max","latency_spread_frac","tflops","mem_bw_tbps","roofline_pred_us",
                "roofline_rel_err","roofline_bound"]
        with open(cpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("WROTE", cpath, flush=True)


if __name__ == "__main__":
    main()
