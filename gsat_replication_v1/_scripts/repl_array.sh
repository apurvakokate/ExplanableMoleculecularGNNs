#!/usr/bin/env bash
#SBATCH -J gsat_repl
#SBATCH -p preempt
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH -o /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/gsat_replication_v1/_dispatch/logs/run_%A_%a.out
# repl_array.sh — one array task = one (backbone, dataset, seed) row of `cells.py --list`.
# Idempotent: skips if results/<bb>/<ds>/seed<k>.json exists (preempt requeue / relaunch safe).
# Failure -> appends to _dispatch/failures.tsv; no auto-retry.
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
R="$REPO/gsat_replication_v1"
UP=/nfs/hpc/share/kokatea/ChemIntuit/GSAT_upstream_6812f5c  # pristine upstream clone, outside the synced repo
IDX="${SLURM_ARRAY_TASK_ID:?array task id required}"
read -r _ BB DS SEED < <(python3 "$R/_scripts/cells.py" --list | awk -v i="$IDX" '$1==i')
[ -n "${BB:-}" ] || { echo "no run for index $IDX"; exit 1; }
OUT="$R/results/$BB/$DS/seed$SEED.json"
[ -s "$OUT" ] && { echo "DONE already: $OUT"; exit 0; }

source ~/hpc-share/anaconda3/etc/profile.d/conda.sh
conda activate l2xgnn
cd "$UP/src"
echo "[$(date)] idx=$IDX $BB $DS seed=$SEED host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none}"
if python "$R/_scripts/run_one_seed.py" --dataset "$DS" --backbone "$BB" --seed "$SEED" --cuda 0 --out "$OUT"; then
    echo "[$(date)] OK $OUT"
else
    rc=$?
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%F_%T)" "$IDX" "$BB" "$DS" "$SEED" "rc=$rc job=${SLURM_JOB_ID}" >> "$R/_dispatch/failures.tsv"
    exit $rc
fi
