#!/usr/bin/env bash
#SBATCH -J planted_janitor
#SBATCH -p share,preempt
#SBATCH --gres=gpu:0
#SBATCH -c 2
#SBATCH --mem=2G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH -o /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/motifsat_paper_deliverables_v1/_dispatch_planted/janitor_%j.out
# planted_janitor.sh — fork of variants_janitor.sh for the planted-GT campaign. Every 30 min, a claim whose job
# is no longer in squeue and whose run dir lacks occurrence_summary.json (orphaned by preemption / time limit)
# is MOVED (never deleted) to _dispatch_planted/claims_orphaned/<stamp>/, its partial run dir to
# planted_v3/runs_interrupted/<stamp>/; if any were moved, 4 GPU workers (PAR=2, all three datasets) are
# submitted from the repo's planted_worker.sh with r_table.sh filled in. Submitted via sbatch stdin.
set -uo pipefail
R=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor; O=$R/motifsat_paper_deliverables_v1; S=$O/_scripts
D=$O/_dispatch_planted; LOG=$D/logs; P3=$O/planted_v3
for cycle in $(seq 1 23); do
  T=$(date +%Y%m%d%H%M%S); moved=0
  q=$(squeue -u kokatea -h -o %A) || { echo "$(date) squeue failed; skip cycle"; sleep 1800; continue; }
  [ -n "$q" ] || { echo "$(date) squeue empty; skip cycle"; sleep 1800; continue; }
  live=" $(echo "$q" | tr '\n' ' ') "
  for c in "$D"/claims/*; do
    [ -d "$c" ] || continue; [ -e "$c/.failed" ] && continue
    j=$(cut -f2 "$c/info" 2>/dev/null); [ -n "$j" ] || continue
    case "$live" in *" $j "*) continue;; esac
    IFS='|' read -r rule p g ds f bb <<< "$(basename "$c" | sed 's/__/|/g')"
    run="$P3/runs/$rule/$p/$g/$ds/fold${f#f}/$bb"
    [ -s "$run/occurrence_summary.json" ] && continue
    mkdir -p "$D/claims_orphaned/$T"; mv "$c" "$D/claims_orphaned/$T/"
    if [ -d "$run" ]; then mkdir -p "$P3/runs_interrupted/$T/$rule/$p/$g/$ds/fold${f#f}"; mv "$run" "$P3/runs_interrupted/$T/$rule/$p/$g/$ds/fold${f#f}/"; fi
    moved=$((moved+1))
  done
  echo "$(date) cycle $cycle: moved $moved orphaned cells (stamp $T)"
  if [ "$moved" -gt 0 ]; then
    W=$(mktemp); python3 - "$S/planted_worker.sh" "$S/r_table.sh" "$W" <<'PY'
import sys
w, t, o = sys.argv[1:4]; s = open(w).read(); assert s.count('R_TABLE_FILLED_AT_LAUNCH') == 1
open(o, 'w').write(s.replace('R_TABLE_FILLED_AT_LAUNCH', open(t).read().strip()))
PY
    bash -n "$W" && sbatch --requeue --array=1-4 -p gpu,dgx2,preempt --gres=gpu:1 -c 2 --mem=16G -t 12:00:00 \
      -J planted_gpu -o "$LOG/gpu_%A_%a.out" \
      --export=ALL,POOL_DATASETS=Mutagenicity:hERG:BBBP,DEVICE=cuda,CLIP=,SMOKE=0,PAR=2 "$W"
    rm -f "$W"
  fi
  sleep 1800
done
