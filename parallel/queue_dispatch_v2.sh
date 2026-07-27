#!/usr/bin/env bash
# queue_dispatch_v2.sh — full-coverage mixed CPU<->GPU dispatch (config-driven).
# Enumerates cells from config/experiment_matrix.yaml (BBBP by default), splits them into
# a GPU file (vtrain-first, then ante-hoc) and a CPU file (post-hoc), and launches a mixed
# worker pool inside each holder via srun --overlap:
#   GPU workers -> vtrain + ante-hoc (GPU-pinned)
#   CPU workers -> post-hoc first (ckpt-gated), then ante-hoc overflow (CUDA hidden)
# Resumable (mkdir-claim + done markers survive restarts/VPN drops).
#   ONLY_DATASET=BBBP bash parallel/queue_dispatch_v2.sh
#   MAX_CELLS=6 ONLY_DATASET=BBBP bash parallel/queue_dispatch_v2.sh   # smoke test
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/final_v2_env.sh"
ONLY="${ONLY_DATASET:-BBBP}"

CELLS="$QUEUE_DIR/cells.tsv"; CELLS_GPU="$QUEUE_DIR/cells_gpu.tsv"; CELLS_CPU="$QUEUE_DIR/cells_cpu.tsv"
CLAIMS="$QUEUE_DIR/claims"; DONE="$QUEUE_DIR/done"; mkdir -p "$CLAIMS" "$DONE"

if [ ! -s "$CELLS" ]; then
    python3 "$HERE/enumerate_from_config.py" "$PROJECT/config/experiment_matrix.yaml" \
        --only-dataset "$ONLY" > "$CELLS.full"
    if [ -n "${MAX_CELLS:-}" ]; then head -n "$MAX_CELLS" "$CELLS.full" > "$CELLS"; else mv "$CELLS.full" "$CELLS"; fi
    # GPU file: vtrain FIRST (checkpoints appear early → unblocks post-hoc), then ante-hoc
    awk -F'\t' '$7=="gpu" && $6=="vtrain"' "$CELLS" >  "$CELLS_GPU"
    awk -F'\t' '$7=="gpu" && $6!="vtrain"' "$CELLS" >> "$CELLS_GPU"
    awk -F'\t' '$7=="cpu"'                 "$CELLS" >  "$CELLS_CPU"
fi
TOTAL=$(wc -l < "$CELLS"); [ "$TOTAL" -eq 0 ] && { echo "[dispatch-v2] no cells"; exit 1; }
echo "[dispatch-v2] $TOTAL cells ($(wc -l < "$CELLS_GPU") gpu, $(wc -l < "$CELLS_CPU") cpu); done=$(ls "$DONE" 2>/dev/null | wc -l)"

pids=()
for spec in $HOLDERS; do
    jobid="${spec%%:*}"; rest="${spec#*:}"; cores="${rest%%:*}"; gpus="${rest##*:}"
    st=$(squeue -j "$jobid" -h -o "%T" 2>/dev/null || echo "?")
    [ "$st" = RUNNING ] || { echo "[dispatch-v2] SKIP holder $jobid (state=$st)"; continue; }
    echo "[dispatch-v2] holder $jobid -> $gpus gpu + $((cores-2*gpus)) cpu workers"
    srun --overlap --jobid="$jobid" --ntasks=1 \
        bash "$HERE/queue_node.sh" "$gpus" "$cores" "$TOTAL" "$CLAIMS" "$DONE" "$HERE" "$CELLS_GPU" "$CELLS_CPU" &
    pids+=($!)
done
[ "${#pids[@]}" -eq 0 ] && { echo "[dispatch-v2] no RUNNING holders"; exit 1; }
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[dispatch-v2] workers exited. done=$(ls "$DONE" | wc -l)/$TOTAL (rc=$rc)"
exit "$rc"
