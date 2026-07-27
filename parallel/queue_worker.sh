#!/usr/bin/env bash
# queue_worker.sh <ncores> <cells_file> <claims_dir> <done_dir> <parallel_dir>
# Runs INSIDE one holder node (launched by queue_dispatch.sh via srun --overlap).
# Spawns <ncores> worker loops; each scans the SHARED cell list top-to-bottom and
# atomically claims each cell with `mkdir` (atomic on NFS — the first worker on ANY
# node to mkdir a cell's dir owns it; everyone else skips). One fresh run_cell.sh
# process per claimed cell, so no state leaks between cells. Dynamic load balance:
# a fast worker keeps claiming; no core idles while cells remain.
set -uo pipefail
NCORES="$1"; CELLS="$2"; CLAIMS="$3"; DONE="$4"; PDIR="$5"
# Re-source the env so PROJECT/OUT_ROOT/FINAL_LOGS/CELL_TIMING_LOG/FORCE_CPU are set
# regardless of how srun propagated the environment.
# shellcheck disable=SC1091
source "$PDIR/final_v1_env.sh"

worker() {
    while IFS=$'\t' read -r ds variant bb fold tier; do
        [ -z "${ds:-}" ] && continue
        key="${ds}__${variant}__${bb}__f${fold}__${tier}"
        # Atomic claim; skip if already claimed (this run or a previous resume).
        mkdir "$CLAIMS/$key" 2>/dev/null || continue
        [ -f "$DONE/$key" ] && continue    # completed on a prior resume
        bash "$PDIR/run_cell.sh" "$ds" "$variant" "$bb" "$fold" "$tier" \
            > "$FINAL_LOGS/cell_${key}.log" 2>&1
        touch "$DONE/$key"
    done < "$CELLS"
}

pids=()
for _ in $(seq "$NCORES"); do worker & pids+=($!); done
wait "${pids[@]}"
echo "[worker $(hostname)] pool of $NCORES drained"
