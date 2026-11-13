#!/bin/bash
# Progress for the element-matched probe array. Read-only; submits nothing.
#   JOBID=<id> OUT=<campaign root> bash analysis/probe_elem_monitor.sh
# JOBID is optional -- without it the queue section is skipped and only the
# on-disk .done / failure counts are reported.
set -uo pipefail

P=${P:-/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor}
OUT=${OUT:-$P/probe_elem_v1}
JOBID=${JOBID:-}
DATASETS=${DATASETS:-"Alkane_Carbonyl_Verified_GT BBBP Benzene_Verified_GT \
Fluoride_Carbonyl_Verified_GT hERG Mutagenicity esol Lipophilicity"}
NEXP=${NEXP:-25}   # cells per dataset: 5 backbones x 5 folds

echo "=== $(date +%H:%M:%S)  out=$OUT ==="
tot=0
printf "%-32s %8s %8s\n" dataset done of
for ds in $DATASETS; do
  d=$(ls "$OUT/$ds"/*.done 2>/dev/null | wc -l | tr -d ' ')
  tot=$((tot+d))
  bar=""; [ "$d" -eq "$NEXP" ] && bar="  COMPLETE"
  printf "%-32s %8s %8s%s\n" "$ds" "$d" "$NEXP" "$bar"
done
echo "--------------------------------------------------"
printf "%-32s %8s %8s\n" TOTAL "$tot" "$(( $(echo $DATASETS | wc -w) * NEXP ))"

if [ -n "$JOBID" ]; then
  echo
  echo "=== queue (job $JOBID) ==="
  squeue -h -j "$JOBID" -o "%T" 2>/dev/null | sort | uniq -c | sed 's/^/  /'
  R=$(squeue -h -j "$JOBID" -t RUNNING -o "%i" 2>/dev/null | wc -l | tr -d ' ')
  echo "  running tasks=$R  cores held=$((2*R))"
  echo
  echo "=== terminal states (sacct) ==="
  sacct -j "$JOBID" --format=State -n 2>/dev/null \
    | grep -v '^\s*$' | awk '{print $1}' | sort | uniq -c | sed 's/^/  /'
  echo
  echo "=== peak memory / longest elapsed so far ==="
  sacct -j "$JOBID" --format=JobID,State,Elapsed,MaxRSS -n 2>/dev/null \
    | awk '$4 ~ /K$/ {gsub("K","",$4); if ($4+0>m) {m=$4+0}} 
           {if ($3 ~ /:/ && $3>e) e=$3} END {
             printf "  MaxRSS peak = %.2f GB\n  longest     = %s\n", m/1048576, e}'
fi

echo
echo "=== failures (non-zero exit in logs) ==="
grep -l "^FATAL\|Traceback\|^srun: error\|oom-kill" "$OUT"/_logs/*.log 2>/dev/null \
  | head -12 | while read -r f; do
      echo "  $(basename "$f"): $(grep -m1 "^FATAL\|Error\|oom-kill" "$f" | cut -c1-110)"
    done
nf=$(grep -l "^FATAL\|Traceback\|oom-kill" "$OUT"/_logs/*.log 2>/dev/null | wc -l | tr -d ' ')
echo "  failing logs: $nf"
