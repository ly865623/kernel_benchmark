#!/usr/bin/env python3
"""
Ground-truth measurement harness for FlashMLA DSA sparse-prefill kernel on B200.

Produces the comparison-ready baseline comparator for quest 003: the measured
execution time (kineto / CUPTI, L2-flushed, mean over num_tests) of the
`sparse_attn_fwd` CUDA kernel, plus FLOPs / memory volume from the repo's own
accounting, and a naive-roofline prediction for context.

Reuses the FlashMLA test harness (tests/lib.py, ref.py, kernelkit) unmodified.
Run inside the NGC container `nvcr.io/nvidia/pytorch:26.01-py3-v0`.
"""
import argparse, json, os, sys, statistics, csv

# FlashMLA test harness lives here (mounted at /workspace/code inside container)
FM = os.environ.get("FLASHMLA_ROOT", "/workspace/code/FlashMLA")
sys.path.insert(0, os.path.join(FM, "tests"))
sys.path.insert(0, FM)  # so `import flash_mla` (repo package, not pip-installed) resolves

import torch
import kernelkit as kk
import lib
import ref
from lib import TestParam

# --- B200 datasheet peaks for naive-roofline context (paper Table II) ---
B200_TENSOR_PEAK = {           # TFLOPS, datasheet peak by input dtype
    "bf16": 2250.0, "fp16": 2250.0, "fp8": 4500.0,
}
B200_HBM_PEAK_TBPS = 8.0       # TB/s datasheet peak


def measure_one(p: TestParam, num_tests: int, passes: int):
    """Measure one prefill config. Returns dict of results."""
    if p.seed == -1:
        p.seed = 0  # fixed deterministic seed for reproducibility
    torch.cuda.empty_cache()
    t = lib.generate_testcase(p)
    torch.cuda.synchronize()

    def run():
        return lib.run_flash_mla_sparse_fwd(p, t, False)

    # warm-up / functional run
    run(); torch.cuda.synchronize()

    fm = lib.count_flop_and_mem_vol(p, t)

    # repeated independent measurement passes for cross-pass repeatability
    lat = []
    for _ in range(passes):
        res = kk.bench_kineto(run, num_tests=num_tests)
        lat.append(res.get_kernel_time("sparse_attn_fwd"))  # seconds
        torch.cuda.synchronize()
    lat_med = statistics.median(lat)
    lat_min, lat_max = min(lat), max(lat)
    spread = (lat_max - lat_min) / lat_med if lat_med > 0 else float("nan")

    tflops = fm.fwd_flop / lat_med / 1e12
    mem_bw = fm.fwd_mem_vol / lat_med / 1e12  # TB/s (mem_vol in bytes)

    # naive roofline context (datasheet peaks) -- dtype is bf16 inputs
    t_compute = fm.fwd_flop / (B200_TENSOR_PEAK["bf16"] * 1e12)
    t_mem = fm.fwd_mem_vol / (B200_HBM_PEAK_TBPS * 1e12)
    roofline_s = max(t_compute, t_mem)
    roofline_rel_err = abs(roofline_s - lat_med) / lat_med

    return {
        "kernel": "sparse_attn_fwd",
        "stage": "prefill",
        "d_qk": p.d_qk, "d_v": p.d_v, "h_q": p.h_q, "h_kv": p.h_kv,
        "s_q": p.s_q, "s_kv": p.s_kv, "topk": p.topk,
        "have_attn_sink": p.have_attn_sink,
        "fwd_flop": fm.fwd_flop, "fwd_mem_vol_bytes": fm.fwd_mem_vol,
        "latency_us": lat_med * 1e6,
        "latency_us_min": lat_min * 1e6, "latency_us_max": lat_max * 1e6,
        "latency_spread_frac": spread,
        "latency_passes_us": [x * 1e6 for x in lat],
        "tflops": tflops, "mem_bw_tbps": mem_bw,
        "roofline_pred_us": roofline_s * 1e6,
        "roofline_rel_err": roofline_rel_err,
        "roofline_bound": "compute" if t_compute >= t_mem else "memory",
    }


def correctness_check(p: TestParam):
    """Run one config through the reference and report allclose verdict."""
    if p.seed == -1:
        p.seed = 0
    t = lib.generate_testcase(p)
    torch.cuda.synchronize()
    out, max_logits, lse = lib.run_flash_mla_sparse_fwd(p, t, False)
    torch.cuda.synchronize()
    ref_out, ref_out_fp32, ref_max_logits, ref_lse = ref.ref_sparse_attn_fwd(p, t)
    ref_lse[ref_lse == float("-inf")] = float("+inf")
    torch.cuda.synchronize()
    ok = True
    ok &= kk.check_is_allclose("out", out.float(), ref_out_fp32, abs_tol=8e-4, rel_tol=3.01/128, cos_diff_tol=7e-6)
    ok &= kk.check_is_allclose("max_logits", max_logits, ref_max_logits, abs_tol=1e-6, rel_tol=2.01/65536)
    ok &= kk.check_is_allclose("lse", lse, ref_lse, abs_tol=1e-6, rel_tol=2.01/65536)
    return bool(ok)


# Representative DSA prefill configs (from tests' performance_case_templates).
# (label, d_qk, h_q, topk, [s_kv ...]); s_q fixed at 4096.
CONFIG_TEMPLATES = [
    ("v32",        576, 128, 2048, [8192, 16384, 32768, 65536]),
    ("model1_cfg2",512, 128, 1024, [8192, 32768, 65536]),
    ("model1_cfg1",512,  64,  512, [8192, 32768, 65536]),
]


def build_perf_cases():
    cases = []
    for (label, d_qk, h_q, topk, s_kv_list) in CONFIG_TEMPLATES:
        for s_kv in s_kv_list:
            cases.append((label, TestParam(s_q=4096, s_kv=s_kv, topk=topk, h_q=h_q,
                                           d_qk=d_qk, d_v=512, have_attn_sink=True,
                                           check_correctness=False, num_runs=0)))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/quest/baselines/local/flashmla-dsa-b200")
    ap.add_argument("--num-tests", type=int, default=30)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--smoke", action="store_true", help="single tiny config only")
    ap.add_argument("--no-correctness", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.float16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    os.makedirs(os.path.join(args.out, "json"), exist_ok=True)
    print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))

    if args.smoke:
        cases = [("smoke", TestParam(s_q=4096, s_kv=8192, topk=2048, h_q=128,
                                     d_qk=576, d_v=512, have_attn_sink=True,
                                     check_correctness=False, num_runs=0))]
    else:
        cases = build_perf_cases()

    # correctness on one small config
    correctness = None
    if not args.no_correctness:
        cp = TestParam(s_q=62, s_kv=512, topk=512, h_q=128, d_qk=576, d_v=512,
                       check_correctness=True, num_runs=0)
        try:
            correctness = correctness_check(cp)
            print(f"[correctness] small config (s_q=62,s_kv=512,topk=512,h_q=128,d_qk=576): {'PASS' if correctness else 'FAIL'}")
        except Exception as e:
            correctness = f"error: {e}"
            print("[correctness] error:", e)

    rows = []
    for (label, p) in cases:
        try:
            r = measure_one(p, args.num_tests, args.passes)
            r["label"] = label
            rows.append(r)
            print(f"[{label}] s_kv={p.s_kv:>6} topk={p.topk} h_q={p.h_q} d_qk={p.d_qk} "
                  f"-> {r['latency_us']:8.1f} us  {r['tflops']:7.1f} TFLOPS  "
                  f"{r['mem_bw_tbps']:5.2f} TB/s  spread={r['latency_spread_frac']*100:.2f}%  "
                  f"roofline_err={r['roofline_rel_err']*100:.1f}%")
        except Exception as e:
            print(f"[{label}] s_kv={p.s_kv} FAILED: {e}")

    payload = {
        "comparator": "flashmla-dsa-sparse-prefill-b200",
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "eval_path": "kk.bench_kineto(flash_mla.flash_mla_sparse_fwd).get_kernel_time('sparse_attn_fwd')",
        "num_tests": args.num_tests, "passes": args.passes,
        "correctness_small_config": correctness,
        "metric_directions": {"latency_us": "lower_better", "tflops": "higher_better", "mem_bw_tbps": "higher_better"},
        "rows": rows,
    }
    jpath = os.path.join(args.out, "json", "ground_truth_prefill.json")
    with open(jpath, "w") as f:
        json.dump(payload, f, indent=2)
    print("WROTE", jpath, "rows=", len(rows))

    if rows:
        cpath = os.path.join(args.out, "ground_truth_prefill.csv")
        keys = ["label","stage","kernel","d_qk","d_v","h_q","h_kv","s_q","s_kv","topk",
                "fwd_flop","fwd_mem_vol_bytes","latency_us","latency_spread_frac",
                "tflops","mem_bw_tbps","roofline_pred_us","roofline_rel_err","roofline_bound"]
        with open(cpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("WROTE", cpath)


if __name__ == "__main__":
    main()
