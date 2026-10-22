#!/usr/bin/env python3
"""Summarize the raw Nsight Compute exports of scripts/profile_matmul.sh.

Reads results/ncu/<config>.csv and writes results/ncu_matmul.md:
  1. a table of key metrics per kernel (time, clock, speed-of-light, memory, occupancy,
     shared memory, FMA pipe);
  2. the top warp-stall reasons per kernel (cycles stalled per issued instruction).
A metric missing from this ncu version is shown as "n/a" (names can change between
versions; `ncu --query-metrics` lists the available ones).

Usage: python3 scripts/summarize_ncu.py
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NCU = ROOT / "results" / "ncu"
CONFIGS = ["naive", "tiled16", "tiled32", "tiled16_c4", "cublas"]

# (label, metric name, formatter)
KEY_METRICS = [
    ("Duration (ms)", "gpu__time_duration.sum", "time"),
    ("SM clock (GHz)", "sm__cycles_elapsed.avg.per_second", "clock"),
    ("Compute SOL (%)", "sm__throughput.avg.pct_of_peak_sustained_elapsed", "pct"),
    ("Memory SOL (%)", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "pct"),
    ("DRAM throughput (%)", "dram__throughput.avg.pct_of_peak_sustained_elapsed", "pct"),
    ("DRAM read (MB)", "dram__bytes_read.sum", "mb"),
    ("L2 hit rate (%)", "lts__t_sector_hit_rate.pct", "pct"),
    ("L1 hit rate (%)", "l1tex__t_sector_hit_rate.pct", "pct"),
    ("Theoretical occupancy (%)", "sm__maximum_warps_per_active_cycle_pct", "pct"),
    ("Achieved occupancy (%)", "sm__warps_active.avg.pct_of_peak_sustained_active", "pct"),
    ("FMA pipe active (%)", "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active", "pct"),
    ("LSU pipe active (%)", "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active", "pct"),
    ("Shared-mem load wavefronts (M)", "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", "mega"),
    ("Shared-mem load bank conflicts (M)", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", "mega"),
    ("Instructions executed (M)", "smsp__inst_executed.sum", "mega"),
]
STALL_RE = re.compile(r"^smsp__average_warps_issue_stalled_(.+)_per_issue_active\.ratio$")


def to_float(s):
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return None


def load(name):
    """Return {metric: (value_str, unit)} for the single profiled kernel."""
    path = NCU / f"{name}.csv"
    rows = list(csv.reader(path.open()))
    # ncu raw CSV: header row, then a units row, then one row per kernel
    header, units, values = rows[0], rows[1], rows[2]
    return {h: (v, u) for h, u, v in zip(header, units, values)}, values[header.index("Kernel Name")] \
        if "Kernel Name" in header else "?"


def fmt(value_unit, kind):
    if value_unit is None:
        return "n/a"
    v, unit = value_unit
    x = to_float(v)
    if x is None:
        return v
    unit = unit.lower()
    if kind == "time":  # normalize to ms
        x = x / 1e6 if unit == "ns" else x / 1e3 if unit in ("us", "usecond") else x * 1e3 if unit == "s" else x
        return f"{x:.2f}"
    if kind == "clock":  # normalize to GHz
        x = x / 1e9 if unit in ("hz", "cycle/second", "cycle/s") else x / 1e3 if unit == "mhz" else x
        return f"{x:.2f}"
    if kind == "mb":
        scale = {"byte": 1e-6, "kbyte": 1e-3, "mbyte": 1, "gbyte": 1e3}.get(unit, 1)
        return f"{x * scale:,.0f}"
    if kind == "mega":
        return f"{x / 1e6:,.1f}"
    return f"{x:.1f}"


data, names = {}, {}
for c in CONFIGS:
    if (NCU / f"{c}.csv").exists():
        data[c], names[c] = load(c)
cfgs = list(data)

out = ["### Nsight Compute, SGEMM kernels (one launch, real clock)\n"]
out.append("| Metric | " + " | ".join(f"`{c}`" for c in cfgs) + " |")
out.append("|---|" + "---|" * len(cfgs))
for label, metric, kind in KEY_METRICS:
    out.append(f"| {label} | " + " | ".join(fmt(data[c].get(metric), kind) for c in cfgs) + " |")

out.append("\n### Top warp-stall reasons (cycles per issued instruction)\n")
out.append("| Kernel | 1st | 2nd | 3rd | 4th |")
out.append("|---|---|---|---|---|")
for c in cfgs:
    stalls = []
    for metric, (v, _) in data[c].items():
        m = STALL_RE.match(metric)
        x = to_float(v)
        if m and x is not None and not m.group(1).endswith("_not_issued"):
            stalls.append((x, m.group(1)))
    stalls.sort(reverse=True)
    cells = [f"{name} ({x:.1f})" for x, name in stalls[:4]] or ["n/a"] * 4
    out.append(f"| `{c}` | " + " | ".join(cells + [""] * (4 - len(cells))) + " |")

out.append("\nKernel names: " + "; ".join(f"`{c}` = {names[c]}" for c in cfgs))
text = "\n".join(out) + "\n"
(ROOT / "results" / "ncu_matmul.md").write_text(text)
print(text)
