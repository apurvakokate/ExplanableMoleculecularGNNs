#!/bin/bash
# Build the task list and submit the element-matched probe as ONE SLURM array.
#
#   8 datasets x 5 backbones x 5 folds = 200 cells, one per array task.
#   -c 2 per task; $CONC caps concurrent tasks, so the campaign holds 2*CONC
#   cores. Default CONC=200 = full spread = 400 cores, chosen for speed. That is
#   above the 300-core standing figure; preempt carries it (that partition has
#   ~3000 cores plus idle, vs share's 1350 group cap), but if CPU-only work
#   elsewhere starts sitting in QOSGrpCpuLimit, lower CONC -- we would be
#   competing with ourselves.
#
# Idempotent: cells with a .done marker are left out of the task list, so a
# preempted or partially failed array is resumed by re-running this script.
#
# Usage:   bash analysis/probe_elem_launch.sh            # submit
#          DRY=1 bash analysis/probe_elem_launch.sh      # print, submit nothing
#          DATASETS="BBBP hERG" bash analysis/probe_elem_launch.sh
set -euo pipefail

P=${P:-/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor}
OUT=${OUT:-$P/probe_elem_v1}
PART=${PART:-preempt}
ACCT=${ACCT:-eecs}            # now mandatory on this cluster
CONC=${CONC:-200}             # concurrent tasks => 2*CONC cores (200 => 400)
MEM=${MEM:-16G}               # tune from the single-cell measurement
TIME=${TIME:-6:00:00}
SPLITS=${SPLITS:-all}
MIN_N=${MIN_N:-30}
DRY=${DRY:-0}

DATASETS=${DATASETS:-"Alkane_Carbonyl_Verified_GT BBBP Benzene_Verified_GT \
Fluoride_Carbonyl_Verified_GT hERG Mutagenicity esol Lipophilicity"}
BACKBONES=${BACKBONES:-"GCN GIN GAT SAGE PNA"}
FOLDS=${FOLDS:-"fold0 fold1 fold2 fold3 fold4"}

mkdir -p "$OUT/_logs" "$OUT/_tasklists"
TL=$OUT/_tasklists/tasklist_$(date +%Y%m%d_%H%M%S).tsv
: > "$TL"

total=0; skipped=0
for ds in $DATASETS; do
  for bb in $BACKBONES; do
    for fd in $FOLDS; do
      total=$((total+1))
      if [ -f "$OUT/$ds/${bb}_${fd}.done" ]; then skipped=$((skipped+1)); continue; fi
      printf '%s\t%s\t%s\n' "$ds" "$bb" "$fd" >> "$TL"
    done
  done
done
N=$(wc -l < "$TL" | tr -d ' ')

echo "cells total=$total  already done=$skipped  to run=$N"
echo "tasklist   $TL"
echo "out        $OUT"
echo "array      1-$N%$CONC   (-c 2, --gres=gpu:0  =>  up to $((2*CONC)) cores)"
echo "resources  -A $ACCT -p $PART --mem=$MEM -t $TIME   splits=$SPLITS min_n=$MIN_N"

[ "$N" -eq 0 ] && { echo "nothing to run"; exit 0; }
if [ "$DRY" != "0" ]; then echo "DRY=1 -> not submitting"; head -5 "$TL"; exit 0; fi

JID=$(sbatch --parsable \
  -A "$ACCT" -p "$PART" \
  --array="1-$N%$CONC" \
  --mem="$MEM" -t "$TIME" \
  -J probe_elem \
  -o "$OUT/_logs/task_%A_%a.log" \
  --export=ALL,P="$P",OUT="$OUT",TASKLIST="$TL",SPLITS="$SPLITS",MIN_N="$MIN_N" \
  analysis/probe_elem_task.sh)
echo "JOBID=$JID"
echo "monitor:  JOBID=$JID OUT=$OUT bash analysis/probe_elem_monitor.sh"
