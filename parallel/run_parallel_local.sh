#!/usr/bin/env bash
# Local parallel dispatch (CPU — for testing + the timed scaling experiment).
# Runs all enumerated cells with K workers, threads pinned to 1 each, K bounded by
# cores AND a RAM budget so we never OOM. K auto = min(cores, 80%RAM / PER_CELL_GB).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CORES=$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu )
if command -v free >/dev/null; then RAM_GB=$(free -g | awk '/Mem:/{print $2}'); else RAM_GB=$(( $(sysctl -n hw.memsize)/1073741824 )); fi
PER_CELL_GB="${PER_CELL_GB:-3}"
RAM_BUDGET_GB="${RAM_BUDGET_GB:-$(( RAM_GB * 8 / 10 ))}"
K_RAM=$(( RAM_BUDGET_GB / PER_CELL_GB )); [ "$K_RAM" -lt 1 ] && K_RAM=1
K="${K:-$(( CORES < K_RAM ? CORES : K_RAM ))}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "[parallel-local] cores=$CORES ram=${RAM_GB}G per_cell~${PER_CELL_GB}G budget=${RAM_BUDGET_GB}G -> K=$K workers (threads pinned to 1)"
bash "$HERE/enumerate_cells.sh" | xargs -P "$K" -L1 bash "$HERE/run_cell.sh"
