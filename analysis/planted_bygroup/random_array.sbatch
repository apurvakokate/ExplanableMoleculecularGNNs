#!/usr/bin/env bash
#SBATCH -J gtroc_rand
#SBATCH -p preempt,dgxh,dgx2,eecs,gpu,share
#SBATCH --gres=gpu:0 -c 2 --mem=20G -t 6:00:00 --requeue --exclude=cn-gpu3
#SBATCH -a 0-180
#SBATCH -o /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/mose_replication_v2/gtroc_bygroup/_logs/rand_%a.out
# Random-explainer baseline for the planted measures. Tasks 0-149: the 150 rules of planted_v2
# (MoSE-GNN and baselines). Tasks 150-180: the 31 rules of the MotifSAT campaign (planted_v3).
# CPU only, 2 cores per task. Reads gt_cache only; writes one .jsonl per rule under gtroc_bygroup/random.
set -uo pipefail
P=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
export PYTHONPATH=$P WANDB_MODE=disabled OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd $P
I=$SLURM_ARRAY_TASK_ID
if [ "$I" -lt 150 ]; then
  TREE=v2
  LINE=$(grep " native$" $P/fragmentation_v2/_deploy/gtroc_planted_manifest.txt | sed -n "$((I+1))p")
  DS=$(echo "$LINE" | cut -d" " -f1); RULE=$(echo "$LINE" | cut -d" " -f2)
else
  TREE=v3
  LINE=$(sed -n "$((I-149))p" $P/spur_msat_v1/worklist.tsv)
  DS=$(echo "$LINE" | cut -f1); RULE=$(echo "$LINE" | cut -f2)
fi
OUT=$P/mose_replication_v2/gtroc_bygroup/random; mkdir -p $OUT
echo "TASK $I -> $TREE $DS $RULE"
[ -n "$DS" ] && [ -n "$RULE" ] || { echo "FATAL: empty task line"; exit 2; }
if [ -f "$OUT/${TREE}__${DS}__${RULE}.jsonl.done" ]; then echo "SKIP (done)"; exit 0; fi
python3 -u $P/analysis/planted_bygroup/random_baseline.py "$TREE" "$DS" "$RULE" "$OUT/${TREE}__${DS}__${RULE}.jsonl"
