#!/usr/bin/env python3
"""Turn results/sweep_reduction.csv into the README table and two figures.

Outputs:
  results/reduction_table.md      markdown tables (summary per version + coarsening grid)
  results/reduction_gbps.png      effective bandwidth vs input size, one line per version
  results/reduction_coarsening.png  coarsened kernel: bandwidth vs COARSE_FACTOR per BLOCK_DIM

Usage: python3 scripts/plot_reduction.py
Requires: pandas, matplotlib
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (WSL / headless)
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PEAK_GBPS = 384.0  # RTX 5060 Laptop: 128-bit bus x 24 Gbps GDDR7
L2_BYTES = 32 * 1024 * 1024

# Categorical palette in fixed order (validated for colour-vision deficiency); every
# line also has its own marker and a legend entry, so identity never relies on colour.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
MARKERS = ["o", "s", "D", "^", "P", "v"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

df = pd.read_csv(RES / "sweep_reduction.csv")
df["config"] = df.apply(
    lambda r: r["kernel"] if r["block_dim"] == 0
    else f'{r["kernel"]} b{r["block_dim"]} c{r["coarse"]}', axis=1)
df["mib"] = df["n"] * 4 / 2**20
df["pct_peak"] = 100 * df["gbps"] / PEAK_GBPS
nmax = df["n"].max()
families = list(dict.fromkeys(df["kernel"]))  # order of the sweep: simple, ..., cub


def best_of(kernel):
    sub = df[(df.kernel == kernel) & (df.n == nmax)]
    return sub.sort_values("gbps").iloc[-1]["config"]


best = [best_of(k) for k in families]
ref_name = "cub" if "cub" in families else None
ref = df[df.kernel == ref_name].set_index("n")["gbps"] if ref_name else None
if ref is not None:
    df["pct_ref"] = 100 * df["gbps"] / df["n"].map(ref)

# ---------------------------------------------------------------- tables
sizes = sorted(df.n.unique())
lines = [f"### Sum reduction, FP32 (median of {int(df.reps.iloc[0])} runs, effective bandwidth)\n"]
lines.append("| Version (best configuration) | " +
             " | ".join(f"n = 2^{int(n).bit_length() - 1} ({n * 4 / 2**20:.0f} MiB)" for n in sizes) + " |")
lines.append("|---|" + "---|" * len(sizes))
for cfg in best:
    sub = df[df.config == cfg].set_index("n").sort_index()
    cells = []
    for r in sub.itertuples():
        extra = f", {r.pct_ref:.0f} % of CUB" if ref is not None and r.kernel != ref_name else ""
        cells.append(f"{r.gbps:,.1f} GB/s ({r.pct_peak:.0f} % of peak{extra})")
    lines.append(f"| `{cfg}` | " + " | ".join(cells) + " |")
lines.append(f"\n_Peak = {PEAK_GBPS:.0f} GB/s (theoretical DRAM bandwidth). "
             f"Sizes up to {L2_BYTES // 2**20} MiB fit in the L2 cache._\n")

co = df[(df.kernel == "coarsened") & (df.n == nmax)] if "coarsened" in families else df.iloc[0:0]
if len(co):
    coarses = sorted(co.coarse.unique())
    lines.append(f"### Coarsened kernel, n = 2^{int(nmax).bit_length() - 1} "
                 "(GB/s, % of peak) by BLOCK_DIM x COARSE_FACTOR\n")
    lines.append("| BLOCK_DIM \\ COARSE_FACTOR | " + " | ".join(str(c) for c in coarses) + " |")
    lines.append("|---|" + "---|" * len(coarses))
    for b in sorted(co.block_dim.unique()):
        row = co[co.block_dim == b].set_index("coarse")
        lines.append(f"| {b} | " + " | ".join(
            f'{row.loc[c, "gbps"]:,.1f} ({row.loc[c, "pct_peak"]:.0f} %)' if c in row.index else "-"
            for c in coarses) + " |")

noisy = df[100 * df.stddev_ms / df.median_ms > 2]
if len(noisy):
    lines.append("\n_Runs with stddev > 2 % of the median:_ " + ", ".join(
        f'`{r.config}` n=2^{int(r.n).bit_length() - 1} ({100 * r.stddev_ms / r.median_ms:.1f} %)'
        for r in noisy.itertuples()))
(RES / "reduction_table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ---------------------------------------------------------------- styling
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 150,
})

# ---------------------------------------------------------------- figure 1
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for i, cfg in enumerate(best):
    sub = df[df.config == cfg].sort_values("n")
    ax.plot(sub.mib, sub.gbps, color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)],
            markersize=6, linewidth=2, label=cfg)
    last = sub.iloc[-1]
    ax.annotate(f"{last.gbps:,.0f}", (last.mib, last.gbps), xytext=(8, 0),
                textcoords="offset points", va="center", color=INK, fontsize=9)
ax.axhline(PEAK_GBPS, color=INK2, linewidth=1, linestyle="--")
ax.annotate(f"theoretical DRAM peak {PEAK_GBPS:.0f} GB/s", (df.mib.min(), PEAK_GBPS),
            xytext=(0, 4), textcoords="offset points", color=INK2, fontsize=8)
ax.axvline(L2_BYTES / 2**20, color=GRID, linewidth=1.5)
ax.annotate("L2 size", (L2_BYTES / 2**20, 0), xytext=(4, 4), textcoords="offset points",
            color=INK2, fontsize=8)
ax.set_xscale("log", base=2)
ax.set_xticks(sorted(df.mib.unique()), [f"{m:.0f}" for m in sorted(df.mib.unique())])
ax.set_xlabel("Input size (MiB)")
ax.set_ylabel("Effective bandwidth (GB/s, median)")
ax.set_ylim(bottom=0)
ax.set_xlim(right=df.mib.max() * 1.6)
ax.set_title("Sum reduction, RTX 5060 Laptop", color=INK, loc="left")
ax.legend(frameon=False, fontsize=9, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.38))
fig.tight_layout()
fig.savefig(RES / "reduction_gbps.png", bbox_inches="tight")

# ---------------------------------------------------------------- figure 2
if len(co):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for i, b in enumerate(sorted(co.block_dim.unique())):
        row = co[co.block_dim == b].sort_values("coarse")
        ax.plot(row.coarse, row.gbps, color=COLORS[i], marker=MARKERS[i], markersize=6,
                linewidth=2, label=f"BLOCK_DIM {b}")
    ax.axhline(PEAK_GBPS, color=INK2, linewidth=1, linestyle="--")
    if ref is not None:
        ax.axhline(ref.loc[nmax], color=COLORS[5], linewidth=1, linestyle=":")
        ax.annotate(f"CUB {ref.loc[nmax]:,.0f}", (co.coarse.min(), ref.loc[nmax]), xytext=(0, -12),
                    textcoords="offset points", color=INK2, fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(coarses, [str(c) for c in coarses])
    ax.set_xlabel("COARSE_FACTOR (elements per thread = 2 x COARSE_FACTOR)")
    ax.set_ylabel("GB/s (median)")
    ax.set_title(f"Coarsened reduction, n = 2^{int(nmax).bit_length() - 1}", color=INK, loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(RES / "reduction_coarsening.png", bbox_inches="tight")

print(f"\nWrote {RES / 'reduction_table.md'} and the figures in {RES}")
