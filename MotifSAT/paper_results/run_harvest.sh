#!/usr/bin/env bash
# run_harvest.sh — parallel table build. Submits one CPU job per (dataset, fold)
# = 40 tasks (each loads its data ONCE for T3), then a build job (afterok) that
# reduces the partials into the 3 tables. Idempotent: re-run to refill missing
# partials, then re-run build (or the printed manual command).
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
PR="$REPO/MotifSAT/paper_results"
DELIV="$REPO/motifsat_paper_deliverables_v1"
RUNS="$DELIV/base_runs"
VOCAB="$REPO/vocab_final_v2"
FOLDS_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS
PROC="$REPO/processed_final_v2"
OUT="$DELIV/paper_tables"; PART="$OUT/partials"; LOG="$OUT/logs"
PART_MODE="${PART_MODE:-preempt}"
mkdir -p "$PART" "$LOG"

DATASETS="${DATASETS:-BBBP hERG Mutagenicity Benzene_Verified_GT Alkane_Carbonyl_Verified_GT Fluoride_Carbonyl_Verified_GT esol Lipophilicity}"
FOLDS="${FOLDS:-0 1 2 3 4}"
ENV="source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn; export PYTHONPATH=$REPO"

jids=()
for ds in $DATASETS; do for f in $FOLDS; do
  jid=$(sbatch --parsable --requeue -p "$PART_MODE" --gres=gpu:0 -c 2 --mem 8G -t 2:00:00 \
    -J "harv_${ds}_f${f}" -o "$LOG/harv_${ds}_f${f}_%j.out" \
    --wrap "$ENV; cd $PR && python3 harvest.py --dataset $ds --fold $f --runs $RUNS \
            --vocab_root $VOCAB --data_root $FOLDS_ROOT --processed_root $PROC --out $PART") \
    && { jids+=("$jid"); echo "harvest $ds fold$f -> $jid"; } \
    || echo "  [FAIL submit] $ds fold$f"
done; done

BUILD_CMD="cd $PR && python3 build_tables.py --partials $PART --out $OUT"
echo "manual build (if the dep job does not fire): $BUILD_CMD"
if [ ${#jids[@]} -gt 0 ]; then
  dep=$(IFS=:; echo "${jids[*]}")
  bjid=$(sbatch --parsable --requeue -p "$PART_MODE" --gres=gpu:0 -c 2 --mem 8G -t 0:30:00 \
    --dependency=afterok:"$dep" -J harv_build -o "$LOG/build_%j.out" \
    --wrap "$ENV; $BUILD_CMD") && echo "build (afterok ${#jids[@]} jobs) -> $bjid ; tables -> $OUT"
fi
