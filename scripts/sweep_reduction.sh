#!/usr/bin/env bash
# Sweep of every sum-reduction variant (PMPP chapter 10) over input sizes.
#   simple (fig. 10.6) | convergent (10.9) | shared memory (10.11) |
#   coarsened (10.15) x BLOCK_DIM x COARSE_FACTOR | CUB DeviceReduce::Sum
# Each configuration is rebuilt with its -D flags, CHECKED (test_reduction must pass),
# then benchmarked. A configuration that fails its test stops the sweep.
# A folder that does not exist is skipped with a warning (edit VARIANTS below).
#
# Outputs (committed with the repo: they are the evidence behind the README):
#   results/sweep_reduction.csv   one line per (configuration, size)
#   results/env_reduction.txt     date, commit, toolkit, driver, measured SM clock before/after
#   results/build_reduction.log   compiler output (ptxas registers / shared memory per variant)
#
# Usage: scripts/sweep_reduction.sh [reps]     (default 30; run from anywhere)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPS="${1:-30}"
# log2 of the input size: 2^20 floats = 4 MiB (fits in the 32 MiB L2 cache),
# 2^22 = 16 MiB (fits), 2^24 = 64 MiB and 2^26 = 256 MiB (DRAM-bound).
LOG2_SIZES=(20 22 24 26)
OUT="$ROOT/results"
CSV="$OUT/sweep_reduction.csv"
ENV="$OUT/env_reduction.txt"
LOG="$OUT/build_reduction.log"
mkdir -p "$OUT"
: > "$LOG"

# ---- Context: what was measured, on what, with which clock ----
make -C "$ROOT/utils" clock_probe >>"$LOG" 2>&1
{
  echo "date    : $(date -Iseconds)"
  echo "commit  : $(git -C "$ROOT" rev-parse --short HEAD)$(git -C "$ROOT" diff --quiet || echo ' (uncommitted changes)')"
  echo "nvcc    : $(nvcc --version | grep release)"
  echo "gpu     : $(nvidia-smi --query-gpu=name,driver_version,power.limit --format=csv,noheader)"
  echo "reps    : $REPS"
  echo "--- clock_probe BEFORE sweep ---"
  "$ROOT/utils/clock_probe"
} > "$ENV"

echo "kernel,block_dim,coarse,n,reps,median_ms,min_ms,max_ms,stddev_ms,gbps" > "$CSV"

# run <folder> <extra nvcc flags>
run() {
  local rel="$1" dir="$ROOT/$1" extra="$2"
  if [ ! -d "$dir" ]; then
    echo ">> $rel: folder not found, skipped"; return 0
  fi
  echo ">> $rel ${extra:-(default flags)}"
  make -C "$dir" clean >>"$LOG" 2>&1
  echo "=== $rel $extra ===" >>"$LOG"
  if ! make -C "$dir" test_reduction bench_reduction EXTRA="$extra" >>"$LOG" 2>&1; then
    echo "   BUILD FAILED, see results/build_reduction.log"; exit 1
  fi
  if ! "$dir/test_reduction" >/dev/null; then
    echo "   TEST FAILED: run '$dir/test_reduction' to see which case"; exit 1
  fi
  for l in "${LOG2_SIZES[@]}"; do
    "$dir/bench_reduction" "$l" "$REPS" | grep '^CSV,' | cut -d, -f2- >> "$CSV"
    echo "   n=2^$l done"
  done
  make -C "$dir" clean >>"$LOG" 2>&1   # leave default builds behind, not sweep variants
}

start=$SECONDS
# ---- Variants: adapt the folder names to your repository ----
run ch10/reduction_simple     "-DBLOCK_DIM=512 -DCOARSE_FACTOR=1"
run ch10/reduction_convergent "-DBLOCK_DIM=512 -DCOARSE_FACTOR=1"
run ch10/reduction_shared     "-DBLOCK_DIM=512 -DCOARSE_FACTOR=1"
for b in 256 512 1024; do
  for c in 1 2 4 8 16; do
    run ch10/reduction_coarsened "-DBLOCK_DIM=$b -DCOARSE_FACTOR=$c"
  done
done
run ch10/cub_reduction ""

{
  echo "--- clock_probe AFTER sweep ---"
  "$ROOT/utils/clock_probe"
} >> "$ENV"

echo "Done in $((SECONDS - start)) s: $(($(wc -l < "$CSV") - 1)) measurements in results/sweep_reduction.csv"
