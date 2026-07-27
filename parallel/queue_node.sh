#!/usr/bin/env bash
# queue_node.sh <gpus> <cores> <total> <claims> <done> <pdir> <cells_gpu> <cells_cpu>
# Runs INSIDE one holder (launched by queue_dispatch_v2.sh via srun --overlap).
# Spawns <gpus> GPU workers (1 per GPU, pinned) + (cores - 2*gpus) CPU workers, then waits.
#   GPU workers drain cells_gpu (vtrain-first, so checkpoints appear early).
#   CPU workers drain cells_cpu (post-hoc, ckpt-gated) FIRST, then cells_gpu (ante-hoc overflow).
set -uo pipefail
GPUS=$1; CORES=$2; TOTAL=$3; CLAIMS=$4; DONE=$5; PDIR=$6; CELLS_GPU=$7; CELLS_CPU=$8
pids=()
for i in $(seq 0 $((GPUS-1))); do
    bash "$PDIR/queue_worker_v2.sh" "$i" "$TOTAL" "$CLAIMS" "$DONE" "$PDIR" "$CELLS_GPU" & pids+=($!)
done
NCPU=$(( CORES - 2*GPUS )); [ "$NCPU" -lt 1 ] && NCPU=1
for _ in $(seq "$NCPU"); do
    bash "$PDIR/queue_worker_v2.sh" "" "$TOTAL" "$CLAIMS" "$DONE" "$PDIR" "$CELLS_CPU" "$CELLS_GPU" & pids+=($!)
done
echo "[node $(hostname)] $GPUS gpu + $NCPU cpu workers"
wait "${pids[@]}"
echo "[node $(hostname)] drained"
