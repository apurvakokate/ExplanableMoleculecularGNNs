#!/usr/bin/env bash
#SBATCH -J gtroc_bg
#SBATCH -p preempt,dgxh,dgx2,eecs,gpu,share
#SBATCH --gres=gpu:0 -c 2 --mem=20G -t 6:00:00 --requeue --exclude=cn-gpu3
#SBATCH -a 0-299%150
#SBATCH -o /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/mose_replication_v2/gtroc_bygroup/_logs/bg_%a.out
# 300 tasks = 150 planted rules x {native, posthoc}; the manifest is the one the planted
# GT-ROC tables were run from. CPU only, 2 cores per task.
set -uo pipefail
P=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $P/fragmentation_v2/_deploy/gtroc_planted_manifest.txt)
echo "TASK $SLURM_ARRAY_TASK_ID -> $LINE"
bash $P/analysis/planted_bygroup/bygroup_cell.sh $LINE
