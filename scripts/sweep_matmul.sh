#!/usr/bin/env bash
# Sweep of every matmul variant over square sizes, for the README results.
#   naive (ch3) | tiled 16/32 (ch5) | tiled+coarsened 16/32 x 1/2/4 (ch6) | cuBLAS
# Each configuration is rebuilt with its -D flags, CHECKED (test_matmul must pass),
# then benchmarked. A configuration that fails its test stops the sweep.
#
# Outputs (committed with the repo: they are the evidence behind the README):
#   results/sweep_matmul.csv   one line per (configuration, size)
#   results/env_matmul.txt     date, commit, toolkit, driver, measured SM clock before/after
#   results/build.log          compiler output (ptxas registers / shared memory per variant)
#
# Usage: scripts/sweep_matmul.sh [reps]     (default 30; run from anywhere)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPS="${1:-30}"
SIZES=(512 1024 2048 4096)
OUT="$ROOT/results"
CSV="$OUT/sweep_matmul.csv"
ENV="$OUT/env_matmul.txt"
LOG="$OUT/build.log"
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

echo "kernel,tile,coarse,M,N,K,reps,median_ms,min_ms,max_ms,stddev_ms,gflops" > "$CSV"

# run <folder> <extra nvcc flags>
run() {
  local dir="$ROOT/$1" extra="$2"
  echo ">> $1 ${extra:-(default flags)}"
  make -C "$dir" clean >>"$LOG" 2>&1
  echo "=== $1 $extra ===" >>"$LOG"
  if ! make -C "$dir" test_matmul bench_matmul EXTRA="$extra" >>"$LOG" 2>&1; then
    echo "   BUILD FAILED, see results/build.log"; exit 1
  fi
  if ! "$dir/test_matmul" >/dev/null; then
    echo "   TEST FAILED: run '$dir/test_matmul' to see which case"; exit 1
  fi
  for n in "${SIZES[@]}"; do
    "$dir/bench_matmul" "$n" "$REPS" | grep '^CSV,' | cut -d, -f2- >> "$CSV"
    echo "   n=$n done"
  done
  make -C "$dir" clean >>"$LOG" 2>&1   # leave default builds behind, not sweep variants
}

start=$SECONDS
run ch3/matmul ""
for t in 16 32; do
  run ch5/matmul_tiling "-DTILE_WIDTH=$t"
done
for t in 16 32; do
  for c in 1 2 4; do
    run ch6/matmul_coarsening "-DTILE_WIDTH=$t -DCOARSE_FACTOR=$c"
  done
done
run cublas ""

{
  echo "--- clock_probe AFTER sweep ---"
  "$ROOT/utils/clock_probe"
} >> "$ENV"

echo "Done in $((SECONDS - start)) s: $(($(wc -l < "$CSV") - 1)) measurements in results/sweep_matmul.csv"
