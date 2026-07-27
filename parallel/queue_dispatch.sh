#!/usr/bin/env bash
# queue_dispatch.sh — run every enumerated cell across the acquired holders, 1 core
# = 1 cell, dynamically load-balanced by a SHARED NFS work-queue (no static shards,
# so no node finishes early while another is still busy — all cores stay fed until
# the queue drains). Resumable: re-running skips already-claimed/done cells.
#
# Usage (after sourcing final_v1_env.sh):
#   bash parallel/queue_dispatch.sh            # run all enumerated cells
#   MAX_CELLS=1 bash parallel/queue_dispatch.sh # smoke test: only the first cell
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/final_v1_env.sh"

CELLS="$QUEUE_DIR/cells.tsv"
CLAIMS="$QUEUE_DIR/claims"
DONE="$QUEUE_DIR/done"
mkdir -p "$CLAIMS" "$DONE"

# 1. Build the cell list once (reuse if present → resume-safe).
if [ ! -s "$CELLS" ]; then
    bash "$HERE/enumerate_cells.sh" > "$CELLS.full"
    if [ -n "${MAX_CELLS:-}" ]; then head -n "$MAX_CELLS" "$CELLS.full" > "$CELLS"
    else mv "$CELLS.full" "$CELLS"; fi
fi
N=$(wc -l < "$CELLS")
[ "$N" -eq 0 ] && { echo "[dispatch] no cells enumerated — did stage0 build the vocab?"; exit 1; }
echo "[dispatch] $N cells ($(ls "$DONE" 2>/dev/null | wc -l) already done) → $CELLS"

# 2. Launch a worker pool inside each holder via srun --overlap (backgrounded).
pids=()
for spec in $HOLDERS; do
    jobid="${spec%%:*}"; cores="${spec##*:}"
    st=$(squeue -j "$jobid" -h -o "%T" 2>/dev/null || echo "?")
    if [ "$st" != "RUNNING" ]; then echo "[dispatch] SKIP holder $jobid (state=$st)"; continue; fi
    echo "[dispatch] holder $jobid → $cores workers"
    srun --overlap --jobid="$jobid" --ntasks=1 \
        bash "$HERE/queue_worker.sh" "$cores" "$CELLS" "$CLAIMS" "$DONE" "$HERE" &
    pids+=($!)
done
[ "${#pids[@]}" -eq 0 ] && { echo "[dispatch] no RUNNING holders"; exit 1; }

# 3. Wait for all holders to drain the queue.
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[dispatch] workers exited. done=$(ls "$DONE" | wc -l)/$N  (rc=$rc)"
exit "$rc"
