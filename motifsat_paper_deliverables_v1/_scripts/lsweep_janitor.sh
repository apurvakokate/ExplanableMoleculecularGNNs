#!/usr/bin/env bash
#SBATCH -J lsweep_janitor
#SBATCH -p share,preempt
#SBATCH --gres=gpu:0
#SBATCH -c 2
#SBATCH --mem=2G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH -o /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/motifsat_paper_deliverables_v1/_dispatch_lsweep/janitor_%j.out
# lsweep_janitor.sh — keeps the coefficient sweep moving unattended (fork of variants_janitor.sh).
# Run:  ssh osuhpc sbatch < lsweep_janitor.sh
# Every 30 min, a claim whose job is no longer in squeue and whose run dir lacks occurrence_summary.json
# (= orphaned by preemption / time limit) is MOVED (never deleted) to _dispatch_lsweep/claims_orphaned/<stamp>/,
# its partial run dir to lsweep_v1_interrupted/<stamp>/. If any were moved, 4 GPU workers (PAR=2, all nine
# datasets, all sweep settings — the sweep's own cells only) are submitted, built from the repo's
# lsweep_worker.sh with r_table.sh / r_table_paper.sh filled in exactly as lsweep_launch.sh does.
# Caveat (as variants_janitor): a requeued array task keeps its job id, so a claim it held before
# preemption looks live; such cells are found by hand (claim job live but the worker log is past it).
set -uo pipefail
R=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor; O=$R/motifsat_paper_deliverables_v1; S=$O/_scripts
D=$O/_dispatch_lsweep; LOG=$D/logs
for cycle in $(seq 1 23); do
  T=$(date +%Y%m%d%H%M%S); moved=0
  q=$(squeue -u kokatea -h -o %A) || { echo "$(date) squeue failed; skip cycle"; sleep 1800; continue; }
  [ -n "$q" ] || { echo "$(date) squeue empty; skip cycle"; sleep 1800; continue; }
  live=" $(echo "$q" | tr '\n' ' ') "
  for c in "$D"/claims/*; do
    [ -d "$c" ] || continue; [ -e "$c/.failed" ] && continue
    j=$(cut -f2 "$c/info" 2>/dev/null); [ -n "$j" ] || continue
    case "$live" in *" $j "*) continue;; esac
    IFS='|' read -r v p g ds f bb <<< "$(basename "$c" | sed 's/__/|/g')"
    run="$O/lsweep_v1/$v/$p/$g/$ds/fold${f#f}/$bb"
    [ -s "$run/occurrence_summary.json" ] && continue
    mkdir -p "$D/claims_orphaned/$T"; mv "$c" "$D/claims_orphaned/$T/"
    if [ -d "$run" ]; then mkdir -p "$O/lsweep_v1_interrupted/$T/$v/$p/$g/$ds/fold${f#f}"; mv "$run" "$O/lsweep_v1_interrupted/$T/$v/$p/$g/$ds/fold${f#f}/"; fi
    moved=$((moved+1))
  done
  echo "$(date) cycle $cycle: moved $moved orphaned cells (stamp $T)"
  if [ "$moved" -gt 0 ]; then
    W=$(mktemp); python3 - "$S/lsweep_worker.sh" "$S/r_table.sh" "$S/r_table_paper.sh" "$W" <<'PY'
import sys
w, own, paper, out = sys.argv[1:5]
src = open(w).read(); assert src.count('R_TABLE_FILLED_AT_LAUNCH') == 1
tabs = []
for path, name in ((own, 'r_for_own'), (paper, 'r_for_paper')):
    t = open(path).read().strip(); assert t.count('r_for(){') == 1, path
    tabs.append(t.replace('r_for(){', f'{name}(){{', 1))
open(out, 'w').write(src.replace('R_TABLE_FILLED_AT_LAUNCH', '\n'.join(tabs)))
PY
    ALLDS=Benzene_Verified_GT:Alkane_Carbonyl_Verified_GT:Fluoride_Carbonyl_Verified_GT:mutag:hERG:Mutagenicity:BBBP:esol:Lipophilicity
    bash -n "$W" && sbatch --requeue --array=1-4 -p gpu,preempt --gres=gpu:1 -c 2 --mem=16G -t 12:00:00 \
      -J lsweep_gpu -o "$LOG/gpu_%A_%a.out" \
      --export=ALL,POOL_DATASETS=$ALLDS,DEVICE=cuda,SMOKE=0,PAR=2 "$W"
    rm -f "$W"
  fi
  sleep 1800
done
