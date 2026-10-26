#!/usr/bin/env bash
# Nsight Compute profile of the sum-reduction kernels (PMPP chapter 10), n = 2^26.
# One kernel launch per version, after warmup, full metric set.
#
# Outputs:
#   results/ncu_reduction/<version>.ncu-rep   full reports (Nsight Compute GUI; not committed)
#   results/ncu_reduction/<version>.csv       raw metrics exported from each report
#   then run: python3 scripts/summarize_ncu_reduction.py  ->  results/ncu_reduction.md
#
# Usage: scripts/profile_reduction.sh [log2_n]     (default 26)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG2N="${1:-26}"
OUT="$ROOT/results/ncu_reduction"
mkdir -p "$OUT"

# profile <name> <folder> <extra nvcc flags> <kernel name regex>
profile() {
  local name="$1" dir="$ROOT/$2" extra="$3" kernel="$4"
  if [ ! -d "$dir" ]; then echo ">> $2: folder not found, skipped"; return 0; fi
  echo ">> $name"
  make -C "$dir" clean >/dev/null
  make -C "$dir" bench_reduction EXTRA="$extra" >/dev/null 2>&1
  # --clock-control none : real clock, same conditions as the benchmarks
  # -k regex:...         : profile only the reduction kernel (skip the cudaMemset / CUB helpers)
  # --launch-skip 3      : skip the first warmup launches
  ncu --clock-control none --set full -k "regex:$kernel" --launch-skip 3 --launch-count 1 \
      -f -o "$OUT/$name" "$dir/bench_reduction" "$LOG2N" 1 >/dev/null
  ncu --import "$OUT/$name.ncu-rep" --page raw --csv > "$OUT/$name.csv"
  make -C "$dir" clean >/dev/null
}

# ---- adapt the folder names to your repository ----
profile simple     ch10/reduction_simple     "-DBLOCK_DIM=512 -DCOARSE_FACTOR=1" "^Reduction"
profile convergent ch10/reduction_convergent "-DBLOCK_DIM=512 -DCOARSE_FACTOR=1" "^Reduction"
profile shared     ch10/reduction_shared     "-DBLOCK_DIM=512 -DCOARSE_FACTOR=1" "^Reduction"
profile coarsened  ch10/reduction_coarsened  "-DBLOCK_DIM=512 -DCOARSE_FACTOR=4" "^Reduction"
# CUB launches two kernels per call; profile the main one (one partial sum per block)
profile cub        ch10/cub_reduction        ""                                  "DeviceReduceKernel"

echo "Reports in results/ncu_reduction/. Next: python3 scripts/summarize_ncu_reduction.py"
