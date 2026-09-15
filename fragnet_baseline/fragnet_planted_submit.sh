#!/usr/bin/env bash
# Build the FragNet PLANTED manifest (one "ds rule fold" per line) and submit ONE preempt array of
# self-contained cells (fragnet_planted_array.sbatch -> fragnet_planted_cell.sh). Everything is
# scheduled UPFRONT; SLURM keeps %MAXCC cells running and refills as they finish ("assign to next
# available resource"). --requeue + the cell's skip-if-done make the whole thing preemption-safe and
# resumable: RE-RUN THIS SCRIPT any time to mop up incomplete cells — finished cells (both unk eval
# CSVs present) are skipped, only missing/failed ones re-run. Tracking = artifacts + _status/*.status.
#
# Env knobs:  DEV=gpu|cpu (default gpu; GPU won the timing test ~15x)   MAXCC (default 75)
#             DATASETS="BBBP hERG Mutagenicity"                         TLIM (per-cell wall limit)
# Dry-run filters:  ONLY_DS=BBBP  ONLY_RULE=dnf_k1_r1  ONLY_FOLD=0
# Usage:  bash fragnet_planted_submit.sh            (submit)
#         DRYRUN=1 bash fragnet_planted_submit.sh   (build manifest + print sbatch, do NOT submit)
set -euo pipefail

PV=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/planted_v2
BASE=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/fragnet/planted
FB=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/fragnet_baseline
SBD=$BASE/_deploy; mkdir -p "$SBD" "$BASE/_logs"

DEV=${DEV:-gpu}
MAXCC=${MAXCC:-75}
DATASETS=${DATASETS:-"BBBP hERG Mutagenicity"}
folds=(0 1 2 3 4); [ -n "${ONLY_FOLD:-}" ] && folds=($ONLY_FOLD)

# ── build the manifest (unique name per invocation -> concurrent submits never clobber) ──
MAN=$SBD/manifest_$(date +%Y%m%d_%H%M%S)_$$.txt; : > "$MAN"
for ds in $DATASETS; do
  [ -n "${ONLY_DS:-}" ] && [ "$ds" != "$ONLY_DS" ] && continue
  for rid in $(ls -1 "$PV/$ds" 2>/dev/null | grep -E '^dnf_'); do
    [ -n "${ONLY_RULE:-}" ] && [ "$rid" != "$ONLY_RULE" ] && continue
    for f in "${folds[@]}"; do echo "$ds $rid $f" >> "$MAN"; done
  done
done
N=$(wc -l < "$MAN")
[ "$N" -eq 0 ] && { echo "ERROR: empty manifest ($MAN) — check PV/DATASETS/filters" >&2; exit 1; }
[ "$N" -gt 1000 ] && echo "WARN: $N > MaxArraySize 1001 — split with ONLY_DS." >&2

# ── device -> resources (GPU is the recommended mode; CPU is overflow) ──
if [ "$DEV" = cpu ]; then GRES="gpu:0"; TLIM=${TLIM:-48:00:00}; else GRES="gpu:1"; TLIM=${TLIM:-12:00:00}; fi
ARRAY="0-$((N-1))%$MAXCC"
echo "[submit] $N cells | DEV=$DEV | array $ARRAY | -p preempt --gres=$GRES -c 2 -t $TLIM --requeue"
echo "[submit] manifest=$MAN"

CMD=(sbatch --parsable --array="$ARRAY" --gres="$GRES" -t "$TLIM"
     --export=ALL,MANIFEST="$MAN",DEV="$DEV" "$FB/fragnet_planted_array.sbatch")
if [ -n "${DRYRUN:-}" ]; then
  echo "[DRYRUN] ${CMD[*]}"; echo "[DRYRUN] head of manifest:"; head -5 "$MAN"; exit 0
fi
jid=$("${CMD[@]}")
echo "submitted array job $jid ($N cells, %$MAXCC concurrent, DEV=$DEV)"
echo "watch:  squeue -u \$USER -o '%.11i %.9P %.9j %.2t %.10M %R'"
echo "status: cat $BASE/_status/*.status | sort | tail"
echo "rollup: python $FB/rollup_base_runs.py --base $BASE"
