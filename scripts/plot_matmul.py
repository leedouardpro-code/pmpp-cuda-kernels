#!/usr/bin/env python3
"""Turn results/sweep_matmul.csv into the README table and two figures.

Outputs:
  results/matmul_table.md        markdown tables (summary + full tile x coarse grid)
  results/matmul_gflops.png      GFLOP/s vs matrix size, one line per kernel
  results/matmul_configs.png     tile x coarsening grid at one size (default 2048)

Usage: python3 scripts/plot_matmul.py [grid_size]
Requires: pandas, matplotlib  (pip install --break-system-packages pandas matplotlib)
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (WSL / headless)
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
GRID_N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048

# Categorical palette in fixed order (validated for colour-vision deficiency).
# Two slots are low-contrast on white, so every line also has its own marker
# and a legend entry: identity never relies on colour alone.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
MARKERS = ["o", "s", "D", "^", "v"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

df = pd.read_csv(RES / "sweep_matmul.csv")
df["n"] = df["M"]
df["config"] = df.apply(
    lambda r: r["kernel"] if r["tile"] == 0 else f'{r["kernel"]} t{r["tile"]} c{r["coarse"]}', axis=1)
cublas = df[df.kernel == "cublas"].set_index("n")["gflops"]
naive = df[df.kernel == "naive"].set_index("n")["gflops"]
df["pct_cublas"] = 100 * df["gflops"] / df["n"].map(cublas)
df["speedup_naive"] = df["gflops"] / df["n"].map(naive)
df["cv_pct"] = 100 * df["stddev_ms"] / df["median_ms"]

# Best hand-written configuration = highest median GFLOP/s at the largest size.
nmax = df["n"].max()
hand = df[df.kernel != "cublas"]
best_cfg = hand[hand.n == nmax].sort_values("gflops").iloc[-1]["config"]

# ---------------------------------------------------------------- tables
lines = [f"### SGEMM, square M = N = K, FP32 (median of {int(df.reps.iloc[0])} runs)\n"]
summary_cfgs = ["naive", "tiled t16 c1", "tiled t32 c1", best_cfg, "cublas"]
lines.append("| Kernel | " + " | ".join(f"n = {n}" for n in sorted(df.n.unique())) + " |")
lines.append("|---|" + "---|" * df.n.nunique())
for cfg in dict.fromkeys(summary_cfgs):
    sub = df[df.config == cfg].set_index("n").sort_index()
    cells = [f'{r.gflops:,.0f} GFLOP/s ({r.pct_cublas:.0f} %)' for r in sub.itertuples()]
    lines.append(f"| `{cfg}` | " + " | ".join(cells) + " |")
lines.append("\n_Percentages are relative to cuBLAS at the same size._\n")

lines.append(f"### Tile x coarsening grid, n = {GRID_N} (GFLOP/s, % of cuBLAS)\n")
g = df[(df.kernel == "tiled coarsened") & (df.n == GRID_N)]
coarses = sorted(g.coarse.unique())
lines.append("| Tile \\ coarse | " + " | ".join(str(c) for c in coarses) + " |")
lines.append("|---|" + "---|" * len(coarses))
for t in sorted(g.tile.unique()):
    row = g[g.tile == t].set_index("coarse")
    lines.append(f"| {t} | " + " | ".join(
        f'{row.loc[c, "gflops"]:,.0f} ({row.loc[c, "pct_cublas"]:.1f} %)' for c in coarses) + " |")

noisy = df[df.cv_pct > 2]
if len(noisy):
    lines.append("\n_Runs with stddev > 2 % of the median:_ " + ", ".join(
        f'`{r.config}` n={r.n} ({r.cv_pct:.1f} %)' for r in noisy.itertuples()))
(RES / "matmul_table.md").write_text("\n".join(lines) + "\n")
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
for i, cfg in enumerate(dict.fromkeys(summary_cfgs)):
    sub = df[df.config == cfg].sort_values("n")
    ax.plot(sub.n, sub.gflops, color=COLORS[i], marker=MARKERS[i], markersize=6,
            linewidth=2, label=cfg)
    # Direct label only on lines that are visually separated (the three tiled
    # variants overlap; they are identified by the legend and their markers).
    if cfg in ("naive", "cublas"):
        last = sub.iloc[-1]
        ax.annotate(f"{last.gflops:,.0f}", (last.n, last.gflops), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(sorted(df.n.unique()), [str(n) for n in sorted(df.n.unique())])
ax.set_xlabel("Matrix size n (M = N = K)")
ax.set_ylabel("GFLOP/s (median, log scale)")
ax.set_title("SGEMM FP32 throughput, RTX 5060 Laptop", color=INK, loc="left")
ax.legend(frameon=False, fontsize=9, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.38))
ax.set_xlim(right=nmax * 1.6)  # room for the direct labels
fig.tight_layout()
fig.savefig(RES / "matmul_gflops.png", bbox_inches="tight")

# ---------------------------------------------------------------- figure 2
fig, ax = plt.subplots(figsize=(6, 3.8))
tiles = sorted(g.tile.unique())
width = 0.38
for i, t in enumerate(tiles):
    row = g[g.tile == t].set_index("coarse").loc[coarses]
    x = [j + (i - (len(tiles) - 1) / 2) * (width + 0.02) for j in range(len(coarses))]
    bars = ax.bar(x, row.gflops, width=width, color=COLORS[i], label=f"tile {t}",
                  edgecolor="white", linewidth=2)
    for b, v in zip(bars, row.gflops):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom",
                fontsize=8, color=INK)
ax.set_xticks(range(len(coarses)), [f"coarse {c}" for c in coarses])
ax.set_ylabel("GFLOP/s (median)")
ax.set_title(f"Tile size x coarsening factor, n = {GRID_N}", color=INK, loc="left")
ax.legend(frameon=False, loc="lower right")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(RES / "matmul_configs.png", bbox_inches="tight")
print(f"\nWrote {RES/'matmul_table.md'}, {RES/'matmul_gflops.png'}, {RES/'matmul_configs.png'}")
