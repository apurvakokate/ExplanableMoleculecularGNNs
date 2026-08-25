#!/usr/bin/env bash
# One mose_replication_v2 worker: drain the shared worklist via a flock'd cursor, and for
# each shard set every path explicitly, skip if already complete, run evaluate.py (or the
# node-mask probe), then post-flight the output. CPU-only; no rescoring (post-hoc READS atts).
#
# GAT double-count guard is structural: final_v2 / ablated shards carry --backbone "GCN GIN
# PNA SAGE" (GAT skipped); GAT rows come only from final_v2_gath1 and land in *__GAT_* files.
set -uo pipefail

P=${P:?set P (Claude+Cursor base)}
V2=${V2:-$P/mose_replication_v2}
DATA_ROOT=${DATA_ROOT:-/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS}
Q=${QUEUE:-$V2/_deploy/worklist.tsv}
CUR=${CURSOR:-$V2/_deploy/cursor}
LK=${QLOCK:-$V2/_deploy/lock}
WALL=${WALL_SECONDS:-43000}
START=$(date +%s)

source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh
conda activate l2xgnn
python3 -c "import torch, torch_geometric" 2>/dev/null || { echo "FATAL: l2xgnn env"; exit 3; }
cd "$P"; export PYTHONPATH=$P WANDB_MODE=disabled
mkdir -p "$V2/_deploy/logs" "$V2/_deploy/manifest"
echo "worker up: node=$(hostname) wall=${WALL}s $(date)"

# Provenance manifest: one row per shard this worker touches — source folder(s) -> destination.
# Per-worker file to avoid NFS append races; build_manifest.py merges them into manifest.csv.
MANI=$V2/_deploy/manifest/$(hostname)_$$.tsv
[ -s "$MANI" ] || printf 'ts\ttier\tdataset\trule\tmethod\tunk_mode\tbbgroup\tstatus\trows\tsource_ckpt\tsource_artifacts\tdest_rollup\tdest_artifacts\n' > "$MANI"
manifest() {  # $1=status $2=rows $3=src_ckpt $4=src_art $5=dest_rollup $6=dest_artifacts
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Iseconds)" "$tier" "$ds" "${rule:--}" "$meth" "${unk_mode:--}" "${bbgroup:--}" \
    "$1" "$2" "$3" "${4:--}" "$5" "${6:--}" >> "$MANI"
}

# rows_ok: CSV has data + (for gt tiers) a GT-ROC column. shard_done: the .done sentinel,
# written ONLY after evaluate.py exits 0 (a full sweep). A preempted/SIGXCPU'd partial CSV has
# no .done, so it is NOT treated as complete and gets re-run (never silently half-skipped).
rows_ok() {  # $1=csv  $2=tier
  [ -s "$1" ] || return 1
  local n; n=$(( $(wc -l < "$1") - 1 )); [ "$n" -gt 0 ] || return 1
  case "$2" in
    source|planted) head -1 "$1" | grep -qE "gtroc_instance|gtroc_global" || return 1 ;;
  esac
  return 0
}
shard_done() { [ -f "$1.done" ]; }

ok=0; fail=0; skip=0
while :; do
  left=$(( WALL - ( $(date +%s) - START ) )); [ "$left" -lt 900 ] && { echo "stopping: ${left}s left"; break; }

  exec 9>"$LK"; flock 9
  cur=$(cat "$CUR" 2>/dev/null || echo 1)
  row=$(sed -n "${cur}p" "$Q")
  [ -n "$row" ] && echo $((cur+1)) > "$CUR"
  flock -u 9; exec 9>&-
  [ -z "$row" ] && { echo "queue empty at row $cur"; break; }
  row=${row%$'\r'}; [ "${row:0:1}" = "#" ] && continue

  IFS=$'\t' read -r tier ds rule meth unk_mode bbs bbgroup ckpt_rel art_rel vocab <<< "$row"
  # restore '-' placeholders to empty (see make_worklist.py: IFS=tab collapses real empties)
  [ "$rule" = "-" ] && rule=""
  [ "$unk_mode" = "-" ] && unk_mode=""
  [ "$bbs" = "-" ] && bbs=""
  [ "$art_rel" = "-" ] && art_rel=""

  # ---- node-mask probe shard -------------------------------------------------
  if [ "$tier" = "probe" ]; then
    OUT=$V2/probe/probe_vsvanilla_${ds}.csv; mkdir -p "$(dirname "$OUT")"
    if shard_done "$OUT"; then echo "SKIP probe $ds (done)"; skip=$((skip+1)); continue; fi
    rm -f "$OUT.done"
    echo "== probe $ds  ${left}s left  $(date +%H:%M:%S) =="
    python3 -u analysis/probe_vs_vanilla.py --dataset "$ds" \
      --data_root "$DATA_ROOT" --vocab_root "$P/vocab_final_v2" --save "$OUT" \
      > "$V2/_deploy/logs/probe_${ds}.log" 2>&1
    prc=$?
    # per-backbone sources (not a fallback): GAT everything from gath1; nonGAT fixed+vanilla
    # from final_v2 and the learnable arm from ablated_completely_v1.
    P_SRC="nonGAT[fixed=final_v2/mose,learn=ablated_completely_v1/mose,vanilla=final_v2/vanilla] GAT[all=final_v2_gath1]"
    if [ "$prc" -eq 0 ] && [ -s "$OUT" ]; then r=$(( $(wc -l < "$OUT") - 1 )); touch "$OUT.done"
      echo "RESULT probe $ds OK rows=$r"; ok=$((ok+1)); manifest OK "$r" "$P_SRC" - "$OUT" -
    else echo "RESULT probe $ds FAIL rc=$prc"; fail=$((fail+1)); manifest FAIL 0 "$P_SRC" - "$OUT" -; fi
    continue
  fi

  # ---- evaluate.py shard: set EVERY path explicitly --------------------------
  CKPT=$P/$ckpt_rel
  if [ "$tier" = "planted" ]; then
    VROOT=$P/planted_v2/$ds/_shared/vocab
    PROOT=$P/planted_v2/$ds/_shared/processed
    subdir="$ds/$rule"
  else
    VROOT=$P/vocab_final_v2
    PROOT=$P/processed_final_v2
    subdir="$ds"
  fi
  # bbgroup is a SUBDIRECTORY (isolates GAT cleanly); the filename carries only method[+unk_mode]
  DESTDIR=$V2/rollups/$tier/$subdir/$bbgroup; mkdir -p "$DESTDIR"
  tag="${meth}${unk_mode:+__$unk_mode}"                     # e.g. mose__fixed  or  mage
  lab="$tier/$subdir/$bbgroup/$tag"                         # human label for logs/RESULT
  DEST=$DESTDIR/metrics_${tag}_unk-exclude.csv
  DROOT=$V2/artifacts/$tier/$subdir/$meth/$bbgroup

  if shard_done "$DEST"; then echo "SKIP $lab (done)"; skip=$((skip+1)); continue; fi

  ART="";  SRC_ART="-"; [ -n "$art_rel"  ] && { ART="--artifacts_root $P/$art_rel"; SRC_ART="$P/$art_rel"; }
  UNKM=""; [ -n "$unk_mode" ] && UNKM="--unk_mode $unk_mode"
  BB="";   [ -n "$bbs"      ] && BB="--backbone $bbs"
  echo "== $lab  ${left}s left  $(date +%H:%M:%S) =="
  rm -f "$DEST.done"

  python3 -u analysis/evaluate.py --method "$meth" --dataset "$ds" \
    --vocab "$vocab" --gt_tier "$tier" --unk exclude $UNKM $BB $ART \
    --ckpt_root "$CKPT" --dest_root "$DROOT" --dest "$DEST" \
    --data_root "$DATA_ROOT" --vocab_root "$VROOT" --processed_root "$PROOT" \
    --device cpu > "$V2/_deploy/logs/${tier}_${ds}_${rule}_${bbgroup}_${tag}.log" 2>&1
  rc=$?

  # OK only on a clean full sweep (rc==0) with real rows; else re-runnable (no .done written)
  if [ "$rc" -eq 0 ] && rows_ok "$DEST" "$tier"; then r=$(( $(wc -l < "$DEST") - 1 )); touch "$DEST.done"
    echo "RESULT $lab OK rc=$rc rows=$r"; ok=$((ok+1)); manifest OK "$r" "$CKPT" "$SRC_ART" "$DEST" "$DROOT"
  else echo "RESULT $lab FAIL rc=$rc"; fail=$((fail+1)); manifest FAIL 0 "$CKPT" "$SRC_ART" "$DEST" "$DROOT"; fi
done
echo "worker done: ok=$ok fail=$fail skip=$skip $(date)"
