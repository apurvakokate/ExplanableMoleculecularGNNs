#!/usr/bin/env bash
# run_harvest.sh — parallel table build. One CPU job per (dataset, fold) = 40 tasks
# (each loads its data ONCE for T3), then a build job (afterok) that reduces the
# partials into the 3 tables. sbatch-a-script pattern (this SLURM build rejects
# --wrap): HDS/HFOLD passed via env + --export=ALL. Idempotent — re-run to refill.
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
PR="$REPO/MotifSAT/paper_results"
DELIV="$REPO/motifsat_paper_deliverables_v1"
OUT="$DELIV/paper_tables"; PART="$OUT/partials"; LOG="$OUT/logs"
PART_MODE="${PART_MODE:-preempt}"
mkdir -p "$PART" "$LOG"

DATASETS="${DATASETS:-BBBP hERG Mutagenicity Benzene_Verified_GT Alkane_Carbonyl_Verified_GT Fluoride_Carbonyl_Verified_GT esol Lipophilicity}"
FOLDS="${FOLDS:-0 1 2 3 4}"

jids=()
for ds in $DATASETS; do for f in $FOLDS; do
  jid=$(HDS="$ds" HFOLD="$f" sbatch --parsable --requeue -p "$PART_MODE" \
        --gres=gpu:0 -c 2 --mem 8G -t 2:00:00 \
        -J "harv_${ds}_f${f}" -o "$LOG/harv_${ds}_f${f}_%j.out" \
        --export=ALL "$PR/harvest_job.sh") \
    && { jids+=("$jid"); echo "harvest $ds fold$f -> $jid"; } \
    || echo "  [FAIL submit] $ds fold$f"
done; done

echo "manual build: sbatch --export=ALL $PR/build_job.sh"
if [ ${#jids[@]} -gt 0 ]; then
  dep=$(IFS=:; echo "${jids[*]}")
  bjid=$(sbatch --parsable --requeue -p "$PART_MODE" --gres=gpu:0 -c 2 --mem 8G -t 0:30:00 \
    --dependency=afterok:"$dep" -J harv_build -o "$LOG/build_%j.out" \
    --export=ALL "$PR/build_job.sh") \
    && echo "build (afterok ${#jids[@]} jobs) -> $bjid ; tables -> $OUT"
fi
