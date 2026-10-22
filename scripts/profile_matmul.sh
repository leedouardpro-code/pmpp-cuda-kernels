#!/usr/bin/env bash
# Nsight Compute profile of the SGEMM kernels discussed in the README:
#   naive (ch3) | tiled 16 and 32 (ch5) | tiled 16 + coarsening 4 (ch6) | cuBLAS
# One kernel launch per configuration, after warmup, full metric set.
#
# Outputs:
#   results/ncu/<config>.ncu-rep   full reports (open with the Nsight Compute GUI; not committed)
#   results/ncu/<config>.csv       raw metrics exported from each report
#   then run: python3 scripts/summarize_ncu.py  ->  results/ncu_matmul.md
#
# Usage: scripts/profile_matmul.sh [n]     (default n = 4096)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
N="${1:-4096}"
OUT="$ROOT/results/ncu"
mkdir -p "$OUT"

# profile <name> <folder> <extra nvcc flags>
profile() {
  local name="$1" dir="$ROOT/$2" extra="$3"
  echo ">> $name"
  make -C "$dir" clean >/dev/null
  make -C "$dir" bench_matmul EXTRA="$extra" >/dev/null 2>&1
  # --clock-control none : keep the real clock (ncu locks the base clock by default),
  #                        same conditions as the benchmarks
  # --launch-skip 3      : skip the first warmup launches (GPU leaves its idle state)
  # --set full           : every section (SOL, memory, occupancy, warp stalls, source)
  ncu --clock-control none --set full --launch-skip 3 --launch-count 1 \
      -f -o "$OUT/$name" "$dir/bench_matmul" "$N" 1 >/dev/null
  ncu --import "$OUT/$name.ncu-rep" --page raw --csv > "$OUT/$name.csv"
  make -C "$dir" clean >/dev/null
}

profile naive          ch3/matmul            ""
profile tiled16        ch5/matmul_tiling     "-DTILE_WIDTH=16"
profile tiled32        ch5/matmul_tiling     "-DTILE_WIDTH=32"
profile tiled16_c4     ch6/matmul_coarsening "-DTILE_WIDTH=16 -DCOARSE_FACTOR=4"
profile cublas         cublas                ""

echo "Reports in results/ncu/. Next: python3 scripts/summarize_ncu.py"
