#!/usr/bin/env python3
"""Summarize the raw Nsight Compute exports of scripts/profile_reduction.sh.

Reads results/ncu_reduction/<version>.csv and writes results/ncu_reduction.md:
  1. key metrics per version, chosen to show the effects discussed in PMPP chapter 10:
     control divergence (active threads per warp instruction), memory divergence
     (global sectors per request: 4 = fully coalesced 4-byte loads), global memory traffic,
     shared memory, barriers, occupancy;
  2. the top warp-stall reasons per version.
A metric missing from this ncu version is shown as "n/a" (`ncu --query-metrics` lists the
available names).

Usage: python3 scripts/summarize_ncu_reduction.py
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NCU = ROOT / "results" / "ncu_reduction"
CONFIGS = ["simple", "convergent", "shared", "coarsened", "cub"]

# (label, candidate metric names (first available wins), formatter)
KEY_METRICS = [
    ("Duration (ms)", ["gpu__time_duration.sum"], "time"),
    ("Memory SOL (%)", ["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"], "pct"),
    ("DRAM throughput (%)", ["dram__throughput.avg.pct_of_peak_sustained_elapsed",
                             "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"], "pct"),
    ("Active threads per warp instruction (of 32)",
     ["smsp__thread_inst_executed_per_inst_executed.ratio"], "num"),
    ("Global load requests (M)", ["l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"], "mega"),
    ("Global load sectors (M)", ["l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"], "mega"),
    ("Global store sectors (M)", ["l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"], "mega"),
    ("Shared-memory wavefronts (M)", ["l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"], "mega"),
    ("Instructions executed (M)", ["smsp__inst_executed.sum"], "mega"),
    ("Achieved occupancy (%)", ["sm__warps_active.avg.pct_of_peak_sustained_active"], "pct"),
    ("Barrier stalls (cycles per instruction)",
     ["smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"], "num"),
]
STALL_RE = re.compile(r"^smsp__average_warps_issue_stalled_(.+)_per_issue_active\.ratio$")


def to_float(s):
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return None


def load(name):
    """Return ({metric: (value, unit)}, kernel name) for the profiled kernel."""
    rows = list(csv.reader((NCU / f"{name}.csv").open()))
    header, units, values = rows[0], rows[1], rows[2]  # header, units row, first kernel
    kname = values[header.index("Kernel Name")] if "Kernel Name" in header else "?"
    return {h: (v, u) for h, u, v in zip(header, units, values)}, kname


def get(data, names):
    for m in names:
        if m in data:
            return data[m]
    return None


def fmt(vu, kind):
    if vu is None:
        return "n/a"
    v, unit = vu
    x = to_float(v)
    if x is None:
        return v
    unit = unit.lower()
    if kind == "time":
        x = x / 1e6 if unit == "ns" else x / 1e3 if unit in ("us", "usecond") else x * 1e3 if unit == "s" else x
        return f"{x:.3f}"
    if kind == "mega":
        return f"{x / 1e6:,.1f}"
    if kind == "num":
        return f"{x:.2f}"
    return f"{x:.1f}"


data, names = {}, {}
for c in CONFIGS:
    if (NCU / f"{c}.csv").exists():
        data[c], names[c] = load(c)
cfgs = list(data)

out = ["### Nsight Compute, sum reduction kernels (one launch, real clock)\n"]
out.append("| Metric | " + " | ".join(f"`{c}`" for c in cfgs) + " |")
out.append("|---|" + "---|" * len(cfgs))
for label, metrics, kind in KEY_METRICS:
    out.append(f"| {label} | " + " | ".join(fmt(get(data[c], metrics), kind) for c in cfgs) + " |")

# Derived: global sectors per request (memory coalescing, PMPP 6.1 / 10.5)
cells = []
for c in cfgs:
    req = get(data[c], ["l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"])
    sec = get(data[c], ["l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"])
    r, s = (to_float(req[0]) if req else None), (to_float(sec[0]) if sec else None)
    cells.append(f"{s / r:.2f}" if r and s is not None else "n/a")
out.append("| Global load sectors per request (4 = fully coalesced) | " + " | ".join(cells) + " |")

out.append("\n### Top warp-stall reasons (cycles per issued instruction)\n")
out.append("| Version | 1st | 2nd | 3rd | 4th |")
out.append("|---|---|---|---|---|")
for c in cfgs:
    stalls = []
    for metric, (v, _) in data[c].items():
        m = STALL_RE.match(metric)
        x = to_float(v)
        if m and x is not None and not m.group(1).endswith("_not_issued"):
            stalls.append((x, m.group(1)))
    stalls.sort(reverse=True)
    cells = [f"{name} ({x:.1f})" for x, name in stalls[:4]] or ["n/a"]
    out.append(f"| `{c}` | " + " | ".join(cells + [""] * (4 - len(cells))) + " |")

out.append("\nKernels: " + "; ".join(f"`{c}` = {names[c]}" for c in cfgs))
text = "\n".join(out) + "\n"
(ROOT / "results" / "ncu_reduction.md").write_text(text)
print(text)
