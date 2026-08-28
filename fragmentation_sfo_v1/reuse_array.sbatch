#!/usr/bin/env bash
#SBATCH -J sfo_reuse
#SBATCH -p preempt
#SBATCH --gres=gpu:0 -c 2 --mem=20G -t 4:00:00 --requeue --exclude=cn-gpu3
#SBATCH -a 0-399%150
#SBATCH -o /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/fragmentation_v2/_deploy/logs/reuse_%a.out
set -uo pipefail
DEP=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/fragmentation_v2/_deploy
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $DEP/reuse_manifest.txt)
echo "TASK $SLURM_ARRAY_TASK_ID -> $LINE"
bash $DEP/sfo_run_cell.sh $LINE
