#!/usr/bin/env bash
#SBATCH -J reuse_agg
#SBATCH -p preempt
#SBATCH --gres=gpu:0 -c 2 --mem=8G -t 20:00 --exclude=cn-gpu3
#SBATCH -o /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/fragmentation_v2/_deploy/logs/reuse_aggregate.out
bash /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/fragmentation_v2/_deploy/reuse_aggregate.sh
