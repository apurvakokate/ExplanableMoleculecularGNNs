#!/usr/bin/env bash
#SBATCH -J paper_janitor
#SBATCH -p share,preempt
#SBATCH --gres=gpu:0
#SBATCH -c 2
#SBATCH --mem=2G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH -o /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/motifsat_paper_deliverables_v1/_dispatch_paper/janitor_%j.out
# paper_janitor.sh — keeps the paper-dataset campaign moving unattended (fork of variants_janitor.sh):
# every 30 min, a claim whose job is no longer in squeue and whose run dir lacks
# occurrence_summary.json (= orphaned by preemption / time limit) is MOVED (never deleted) to
# _dispatch_paper/claims_orphaned/<stamp>/, its partial run dir to paper_v1_interrupted/<stamp>/.
# If any were moved, workers are resubmitted for exactly the moved (vocab, dataset) pairs:
# GPU (mutag / tox21 / molhiv) 1 worker per 4 moved cells (max 8), CPU (clintox / bace / bbbp)
# likewise, PAR=2, worker = the repo's paper_worker.sh with r_table_paper.sh filled in.
# Submitted via sbatch stdin; this file is the record.
set -uo pipefail
R=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor; O=$R/motifsat_paper_deliverables_v1; S=$O/_scripts
D=$O/_dispatch_paper; LOG=$D/logs
for cycle in $(seq 1 23); do
  T=$(date +%Y%m%d%H%M%S); moved=0; : > "$D/.moved_$T"
  q=$(squeue -u kokatea -h -o %A) || { echo "$(date) squeue failed; skip cycle"; sleep 1800; continue; }
  [ -n "$q" ] || { echo "$(date) squeue empty; skip cycle"; sleep 1800; continue; }
  live=" $(echo "$q" | tr '\n' ' ') "
  for c in "$D"/claims/*; do
    [ -d "$c" ] || continue; [ -e "$c/.failed" ] && continue
    j=$(cut -f2 "$c/info" 2>/dev/null); [ -n "$j" ] || continue
    case "$live" in *" $j "*) continue;; esac
    IFS='|' read -r v p g ds rep bb <<< "$(basename "$c" | sed 's/__/|/g')"
    run="$O/paper_v1/$v/$p/$g/$ds/$rep/$bb"
    [ -s "$run/occurrence_summary.json" ] && continue
    mkdir -p "$D/claims_orphaned/$T"; mv "$c" "$D/claims_orphaned/$T/"
    if [ -d "$run" ]; then mkdir -p "$O/paper_v1_interrupted/$T/$v/$p/$g/$ds/$rep"; mv "$run" "$O/paper_v1_interrupted/$T/$v/$p/$g/$ds/$rep/"; fi
    echo "$v $ds" >> "$D/.moved_$T"; moved=$((moved+1))
  done
  echo "$(date) cycle $cycle: moved $moved orphaned cells (stamp $T)"
  if [ "$moved" -gt 0 ]; then
    W=$(mktemp); python3 - "$S/paper_worker.sh" "$S/r_table_paper.sh" "$W" <<'PY'
import sys
w, t, o = sys.argv[1:4]; s = open(w).read(); assert s.count('R_TABLE_FILLED_AT_LAUNCH') == 1
open(o, 'w').write(s.replace('R_TABLE_FILLED_AT_LAUNCH', open(t).read().strip()))
PY
    bash -n "$W" || { echo "worker syntax error"; rm -f "$W"; sleep 1800; continue; }
    for v in $(cut -d' ' -f1 "$D/.moved_$T" | sort -u); do
      for pool in gpu cpu; do
        if [ $pool = gpu ]; then dss=$(grep "^$v " "$D/.moved_$T" | cut -d' ' -f2 | sort -u | grep -E '^(mutag|ogbg-moltox21|ogbg-molhiv)$')
        else dss=$(grep "^$v " "$D/.moved_$T" | cut -d' ' -f2 | sort -u | grep -E '^(ogbg-molclintox|ogbg-molbace|ogbg-molbbbp)$'); fi
        [ -n "$dss" ] || continue
        n=$(grep -c -E "^$v ($(echo $dss | tr ' ' '|'))\$" "$D/.moved_$T"); n=$(( (n + 3) / 4 )); [ $n -gt 8 ] && n=8
        if [ $pool = gpu ]; then part=gpu,preempt; gres=gpu:1; mem=16G; dev=cuda; else part=share,preempt; gres=gpu:0; mem=12G; dev=cpu; fi
        sbatch --requeue --array=1-$n -p $part --gres=$gres -c 2 --mem=$mem -t 12:00:00 -J paper_$pool \
          -o "$LOG/${pool}_%A_%a.out" \
          --export=ALL,POOL_DATASETS=$(echo $dss | tr ' ' ':'),VOCABS=$v,DEVICE=$dev,CLIP=,SMOKE=0,PAR=2 "$W"
      done
    done
    rm -f "$W"
  fi
  rm -f "$D/.moved_$T"
  sleep 1800
done
