#!/usr/bin/env python3
"""
Per-stage timing extraction strictly from the ORIGINAL FlashMLA DSA sparse-prefill kernel.

Requires flash_mla built with -DFLASHINFER_ENABLE_PROFILER. The kernel wraps every pipeline
stage in PROFILER_EVENT_START/END with a PrefillProfileEventType; this script runs the real
`sparse_attn_fwd` kernel with a profiler buffer, decodes the buffer (csrc/profiler.cuh layout),
pairs Begin/End per (block, warp, event) using the on-chip %globaltimer (ns), and reports the
measured per-stage durations. These ARE the stage micro-kernels carved from the real kernel:
the stage boundaries are the kernel's own.

Buffer layout (csrc/profiler.cuh):
  uint64 [0] = {nblocks (u32), ngroups (u32)}
  uint64 [i] = {tag (u32), timestamp_ns (u32)}
  tag bits: 0-1 event_type(0=begin,1=end,2=instant); 2-6 event_idx; 7-23 block*ngroups+group; 24-31 sm
PROFILER_INIT was called with group_idx = canonical warp_idx, num_groups = 16.
"""
import argparse, json, os, sys, csv
from collections import defaultdict

FM = os.environ.get("FLASHMLA_ROOT", "/workspace/code/FlashMLA")
sys.path.insert(0, os.path.join(FM, "tests"))
sys.path.insert(0, FM)

import torch
import lib
from lib import TestParam
from flash_mla.profiler import PREFILL_EVENT_NAMES as EVENT_NAMES

EVENT_BEGIN, EVENT_END, EVENT_INSTANT = 0, 1, 2

# Map raw events to coarse pipeline stages for the analytical model.
STAGE_OF = {
    "launch-tma-cp-p": "q_load", "finish-tma-cp-p": "q_load", "smem2Rmem-p": "q_smem2tmem",
    "launch-tma-gather-k0": "k_gather", "launch-tma-gather-k1": "k_gather",
    "launch-tma-gather-v0": "v_gather", "launch-tma-gather-v1": "v_gather",
    "launch-gemm-p0": "qk_mma", "launch-gemm-p1": "qk_mma",
    "launch-gemm-o0": "sv_mma", "launch-gemm-o1": "sv_mma",
    "check-valid-indices": "valid_check", "tmem2reg-cp-p": "p_load",
    "calic-pi-max": "rowmax", "calic-exp": "exp", "rescale-o": "rescale_o", "update-output": "epilogue_o",
}


def decode_buffer(buf: torch.Tensor):
    """Return (nblocks, ngroups, recs) where recs is a list of
    (block, group, warp, event_idx, event_type, sm, ts_ns). Vectorized with numpy."""
    import numpy as np
    u32 = buf.cpu().view(dtype=torch.uint32).numpy()  # len 2*n
    nblocks = int(u32[0]); ngroups = int(u32[1])
    tag = u32[2::2].astype(np.uint32)      # one per entry i>=1
    ts = u32[3::2].astype(np.int64)
    nz = (tag != 0) | (u32[3::2].astype(np.uint32) != 0)
    tag = tag[nz]; ts = ts[nz]
    sm_id = (tag >> 24) & 0xFF
    bg = (tag >> 7) & 0x1FFFF
    event_idx = (tag >> 2) & 0x1F
    etype = tag & 0x3
    block_idx = (bg // ngroups).astype(np.int64)
    group = (bg % ngroups).astype(np.int64)
    recs = list(zip(block_idx.tolist(), group.tolist(), group.tolist(),
                    event_idx.tolist(), etype.tolist(), sm_id.tolist(), ts.tolist()))
    return nblocks, ngroups, recs


def pair_durations(recs):
    """Pair Begin/End per (block, group, event_idx) in sequence -> list of durations (ns) per event name."""
    streams = defaultdict(list)
    for (block, group, warp, eidx, etype, sm, ts) in recs:
        streams[(block, group, eidx)].append((etype, ts))
    per_event = defaultdict(list)        # event_name -> [durations ns]
    per_block_event = defaultdict(lambda: defaultdict(float))  # block -> event_name -> total ns
    per_event_count = defaultdict(int)
    for (block, group, eidx), seq in streams.items():
        if eidx >= len(EVENT_NAMES):
            continue
        name = EVENT_NAMES[eidx]
        open_ts = None
        for (etype, ts) in seq:
            if etype == EVENT_BEGIN:
                open_ts = ts
            elif etype == EVENT_END and open_ts is not None:
                d = (ts - open_ts) & 0xFFFFFFFF  # wrap-safe
                per_event[name].append(d)
                per_block_event[block][name] += d
                per_event_count[name] += 1
                open_ts = None
    return per_event, per_block_event, per_event_count


def summarize(per_event, per_block_event):
    import statistics
    rows = []
    # per-stage: aggregate raw events into coarse stages, per block then mean over blocks
    stage_block = defaultdict(lambda: defaultdict(float))
    for block, evmap in per_block_event.items():
        for name, tot in evmap.items():
            stage_block[block][STAGE_OF.get(name, name)] += tot
    stages = sorted({s for b in stage_block.values() for s in b})
    nblocks = max(len(per_block_event), 1)
    stage_summary = {}
    for s in stages:
        vals = [stage_block[b].get(s, 0.0) for b in stage_block]
        stage_summary[s] = {
            "per_block_total_ns_mean": statistics.mean(vals) if vals else 0.0,
            "per_block_total_ns_median": statistics.median(vals) if vals else 0.0,
            "n_blocks": len(vals),
        }
    raw_summary = {}
    for name, ds in per_event.items():
        raw_summary[name] = {
            "count": len(ds), "mean_ns": statistics.mean(ds) if ds else 0.0,
            "median_ns": statistics.median(ds) if ds else 0.0,
            "stage": STAGE_OF.get(name, name),
        }
    return stage_summary, raw_summary


def run_one(s_q, s_kv, topk, d_qk, buf_size, warmup):
    p = TestParam(s_q=s_q, s_kv=s_kv, topk=topk, h_q=128, d_qk=d_qk, d_v=512,
                  seed=0, have_attn_sink=True, check_correctness=False, num_runs=0,
                  profiler_buffer_size=buf_size)
    t = lib.generate_testcase(p)
    torch.cuda.synchronize()
    for _ in range(warmup):
        lib.run_flash_mla_sparse_fwd(p, t, False)
    torch.cuda.synchronize()
    t.profile_buf.zero_()
    torch.cuda.synchronize()
    lib.run_flash_mla_sparse_fwd(p, t, False)
    torch.cuda.synchronize()
    nblocks, ngroups, recs = decode_buffer(t.profile_buf)
    per_event, per_block_event, cnt = pair_durations(recs)
    stage_summary, raw_summary = summarize(per_event, per_block_event)
    k_tiles = (topk + 127) // 128
    return {
        "s_q": s_q, "s_kv": s_kv, "topk": topk, "d_qk": d_qk,
        "nblocks": nblocks, "ngroups": ngroups, "k_tiles": k_tiles,
        "n_entries": len(recs),
        "stage_summary_per_block_ns": stage_summary,
        "raw_event_summary_ns": raw_summary,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/quest/.ds/worktrees/idea-idea-98be86a0/experiments/main/stages")
    ap.add_argument("--buf-size", type=int, default=1 << 24)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--d-qk", type=int, default=576)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full-grid", action="store_true",
                    help="Profile the full 56-config batch x s_kv grid (mirrors the whole-kernel sweep).")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.float16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    os.makedirs(os.path.join(args.out, "json"), exist_ok=True)
    print("device:", torch.cuda.get_device_name(0), flush=True)

    # Representative configs: small batch (1 full wave = 74) to characterize the per-tile pipeline,
    # across s_kv to expose any gather-spread effect, plus a few batch sizes for completeness.
    if args.smoke:
        configs = [(74, 8192, 1024)]
    elif args.full_grid:
        # Full 56-config grid mirroring the whole-kernel latency sweep (user req 2026-06-12):
        # batch_size(=s_q) in {1,32,64,74,128,148,256,296} x s_kv in {1k..128k}, fixed topk=1024.
        # Makes the per-stage micro-kernel decomposition as dense as the whole-kernel grid.
        configs = []
        for bs in [1, 32, 64, 74, 128, 148, 256, 296]:
            for s_kv in [1024, 4096, 8192, 16384, 32768, 65536, 131072]:
                configs.append((bs, s_kv, 1024))
    else:
        configs = []
        for s_kv in [1024, 8192, 32768, 131072]:
            configs.append((74, s_kv, 1024))     # 1 full wave, vary gather spread
        for bs in [1, 32, 148, 296]:
            configs.append((bs, 8192, 1024))     # vary occupancy at fixed s_kv

    results = []
    for (s_q, s_kv, topk) in configs:
        try:
            r = run_one(s_q, s_kv, topk, args.d_qk, args.buf_size, args.warmup)
            results.append(r)
            ss = r["stage_summary_per_block_ns"]
            top = sorted(ss.items(), key=lambda kv: -kv[1]["per_block_total_ns_mean"])[:6]
            print(f"[bs={s_q} s_kv={s_kv} k_tiles={r['k_tiles']}] nblocks={r['nblocks']} "
                  f"entries={r['n_entries']}", flush=True)
            for name, st in top:
                print(f"    {name:14s} {st['per_block_total_ns_mean']:9.1f} ns/block "
                      f"(median {st['per_block_total_ns_median']:.1f})", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[bs={s_q} s_kv={s_kv}] FAILED: {e}", flush=True)

    payload = {"device": torch.cuda.get_device_name(0), "event_names": EVENT_NAMES,
               "stage_of": STAGE_OF, "fixed": {"h_q": 128, "d_qk": args.d_qk, "d_v": 512},
               "configs": results}
    jname = "stage_timings_full.json" if args.full_grid else "stage_timings.json"
    jpath = os.path.join(args.out, "json", jname)
    with open(jpath, "w") as f:
        json.dump(payload, f, indent=2)
    print("WROTE", jpath, "configs=", len(results), flush=True)

    # Flat per-stage CSV: one row per config, one column per pipeline stage (per-block mean ns).
    # This is the human-visible "stage micro-kernel" decomposition table.
    import math
    stage_names = sorted({s for r in results for s in r["stage_summary_per_block_ns"]})
    cpath = os.path.join(args.out, "json", "stage_grid.csv")
    with open(cpath, "w", newline="") as f:
        w = csv.writer(f)
        header = ["batch_size", "s_kv", "topk", "nblocks", "num_waves", "n_entries",
                  "total_pipeline_ns"] + [f"stage_{s}_ns" for s in stage_names]
        w.writerow(header)
        for r in results:
            ss = r["stage_summary_per_block_ns"]
            num_waves = max(1, math.ceil(r["s_q"] / 74))
            total = sum(v["per_block_total_ns_mean"] for v in ss.values())
            row = [r["s_q"], r["s_kv"], r["topk"], r["nblocks"], num_waves,
                   r["n_entries"], round(total, 2)]
            row += [round(ss.get(s, {}).get("per_block_total_ns_mean", 0.0), 2) for s in stage_names]
            w.writerow(row)
    print("WROTE", cpath, "rows=", len(results), "stages=", len(stage_names), flush=True)


if __name__ == "__main__":
    main()
