#!/usr/bin/env bash
# launch.sh — submit the GSAT replication array (current MotifSAT package).
#   CHECK=1   -> one 2-epoch run per cell (first row of each cell) into runs_check/
#   IDX=spec  -> explicit array spec (e.g. re-run failures)
#   default   -> all rows, at most MAXPAR (64) concurrent GPUs
set -euo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
S="$REPO/gsat_replication_v2/_scripts"
mkdir -p "$REPO/gsat_replication_v2/_dispatch/logs"
N=$(cd "$S" && python3 cells.py --n)
if [ "${CHECK:-0}" = 1 ]; then
    SPEC=$(cd "$S" && python3 cells.py --list | awk '!seen[$2]++ {print $1}' | paste -sd, -)
elif [ -n "${IDX:-}" ]; then
    SPEC="$IDX"
else
    SPEC="0-$((N-1))%${MAXPAR:-64}"
fi
echo "submitting array=$SPEC (of $N runs) CHECK=${CHECK:-0}"
CHECK="${CHECK:-0}" sbatch --export=ALL --array="$SPEC" "$S/run_array.sh"
