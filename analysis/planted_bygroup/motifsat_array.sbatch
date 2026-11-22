#!/usr/bin/env bash
#SBATCH -J gtroc_bgm
#SBATCH -p preempt,dgxh,dgx2,eecs,gpu,share
#SBATCH --gres=gpu:0 -c 2 --mem=20G -t 6:00:00 --requeue --exclude=cn-gpu3
#SBATCH -a 0-30
#SBATCH -o /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/mose_replication_v2/gtroc_bygroup/_logs/bgm_%a.out
# 31 tasks = the 31 planted rules of the MotifSAT campaign (work list of spur_msat_v1).
# CPU only, 2 cores per task. Reads saved scores; writes one .jsonl per rule under gtroc_bygroup/motifsat.
set -uo pipefail
P=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $P/spur_msat_v1/worklist.tsv)
DS=$(echo "$LINE" | cut -f1); RULE=$(echo "$LINE" | cut -f2)
OUT=$P/mose_replication_v2/gtroc_bygroup/motifsat; mkdir -p $OUT
echo "TASK $SLURM_ARRAY_TASK_ID -> $DS $RULE"
if [ -f "$OUT/${DS}__${RULE}.jsonl.done" ]; then echo "SKIP (done)"; exit 0; fi
python3 -u $P/analysis/planted_bygroup/motifsat_bygroup.py "$DS" "$RULE" "$OUT/${DS}__${RULE}.jsonl"
