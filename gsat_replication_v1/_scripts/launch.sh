#!/usr/bin/env bash
# launch.sh — submit the replication array.
#   SMOKE=1      -> only the GIN/mutag/seed0 row (one full 100-epoch run)
#   IDX=a-b,c    -> explicit array spec (e.g. re-run failures)
#   default      -> all rows of cells.py --list, at most MAXPAR concurrent (default 64 GPUs)
set -euo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
S="$REPO/gsat_replication_v1/_scripts"
mkdir -p "$REPO/gsat_replication_v1/_dispatch/logs"
N=$(python3 "$S/cells.py" --n)
if [ "${SMOKE:-0}" = 1 ]; then
    SPEC=$(python3 "$S/cells.py" --list | awk '$2=="GIN" && $3=="mutag" && $4==0 {print $1}')
elif [ -n "${IDX:-}" ]; then
    SPEC="$IDX"
else
    SPEC="0-$((N-1))%${MAXPAR:-64}"
fi
echo "submitting array=$SPEC (of $N runs)"
sbatch --array="$SPEC" "$S/repl_array.sh"
