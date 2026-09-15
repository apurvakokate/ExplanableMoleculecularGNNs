#!/usr/bin/env bash
#SBATCH -J fp_cell
#SBATCH -p preempt
#SBATCH --gres=gpu:1 -c 2 --mem=24G -t 12:00:00 --requeue --exclude=cn-gpu3
#SBATCH -o /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/fragnet/planted/_logs/cell_%A_%a.out
# One SLURM array over a flat manifest; each task runs ONE self-contained cell end-to-end.
# MANIFEST (path) and DEV (gpu|cpu) are passed at submit time via --export (fragnet_planted_submit.sh),
# which also overrides --gres / -t for CPU mode. --requeue: a preempted task re-runs from the top;
# the cell's skip-if-done makes that idempotent. 2 cores per GPU per policy.
set -uo pipefail
FB=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/fragnet_baseline
: "${MANIFEST:?MANIFEST not exported — submit via fragnet_planted_submit.sh}"
DEV=${DEV:-gpu}
mkdir -p /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/fragnet/planted/_logs
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MANIFEST")
[ -z "$LINE" ] && { echo "no manifest line for task $SLURM_ARRAY_TASK_ID"; exit 0; }
echo "TASK $SLURM_ARRAY_TASK_ID -> [$LINE] DEV=$DEV manifest=$MANIFEST"
bash "$FB/fragnet_planted_cell.sh" $LINE "$DEV"
