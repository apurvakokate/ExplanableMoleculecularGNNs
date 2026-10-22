#!/usr/bin/env bash
#SBATCH -J gsat_repl2
#SBATCH -p preempt
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH -o /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/gsat_replication_v2/_dispatch/logs/run_%A_%a.out
# run_array.sh — one array task = one row of `cells.py --list` (cell, fold, seed),
# trained with MotifSAT/run.py. CHECK=1 -> 2-epoch check run into runs_check/.
# Idempotent: skips when <run dir>/summary.json exists. Failures -> failures.tsv.
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
R="$REPO/gsat_replication_v2"
IDX="${SLURM_ARRAY_TASK_ID:?array task id required}"
CHK=""; ROOT="$R/runs"
[ "${CHECK:-0}" = 1 ] && { CHK="--check"; ROOT="$R/runs_check"; }
read -r NAME FOLD SEED ARGS < <(cd "$R/_scripts" && python3 cells.py --args "$IDX" $CHK)
OUT="$ROOT/$NAME/fold$FOLD/seed$SEED"
[ -s "$OUT/summary.json" ] && { echo "DONE already: $OUT"; exit 0; }
case "$NAME" in
  *mutag*) DATA="$REPO/data" ;;                                   # mutag raw + artifacts
  *)       DATA=/nfs/hpc/share/kokatea/ChemIntuit/GSAT_upstream_6812f5c/data ;;  # all 6 OGB sets
esac
source ~/hpc-share/anaconda3/etc/profile.d/conda.sh
conda activate l2xgnn
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled TRACK_EXPL_EVERY=0
echo "[$(date)] idx=$IDX $NAME fold=$FOLD seed=$SEED host=$(hostname)"
if python3 MotifSAT/run.py $ARGS --data_root "$DATA" --out_dir "$OUT" --final_out_dir; then
    echo "[$(date)] OK $OUT"
else
    rc=$?
    mkdir -p "$R/_dispatch"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%F_%T)" "$IDX" "$NAME" "$FOLD" "$SEED" \
        "rc=$rc job=${SLURM_JOB_ID} check=${CHECK:-0}" >> "$R/_dispatch/failures.tsv"
    exit $rc
fi
