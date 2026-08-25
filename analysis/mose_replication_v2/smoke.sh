#!/usr/bin/env bash
# Pre-flight smoke: run 5 representative shards INLINE (no SLURM) to validate the new code +
# path wiring before the full drain. Exercises: --backbone GAT-skip, --unk_mode fixed vs
# learnable_shared, GAT-from-gath1, post-hoc source GT-ROC (no rescore), planted path, and the
# node-mask probe's did_base/did_learn. Then runs the double-count audit on the smoke output.
set -euo pipefail

P=${P:-/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor}
V2=${V2:-$P/mose_replication_v2}
D=$V2/_deploy; mkdir -p "$D/logs"

RULE=$(for d in "$P"/planted_v2/BBBP/*/; do b=$(basename "$d"); [ "${b#_}" = "$b" ] && { echo "$b"; break; }; done)
[ -n "${RULE:-}" ] || { echo "FATAL: no planted BBBP rule found"; exit 1; }
echo "smoke planted rule = $RULE"

W=$D/worklist_smoke.tsv
{
  printf '# smoke worklist\n'
  # tier<TAB>ds<TAB>rule<TAB>method<TAB>unk_mode<TAB>backbones<TAB>bbgroup<TAB>ckpt_rel<TAB>art_rel<TAB>vocab
  printf 'none\tBBBP\t-\tmose\tfixed\tGCN GIN PNA SAGE\tnonGAT\tfinal_v2\t\trbrics_filter\n'
  printf 'none\tBBBP\t-\tmose\tlearnable_shared\tGAT\tGAT\tfinal_v2_gath1\t\trbrics_filter\n'
  printf 'source\tBenzene_Verified_GT\t-\tmage\t\tGCN GIN PNA SAGE\tnonGAT\tfinal_v2\tposthoc_v1\trbrics\n'
  printf 'planted\tBBBP\t%s\tmose\tfixed\t\tall\tplanted_v2/BBBP/%s\t\trbrics_filter\n' "$RULE" "$RULE"
  printf 'probe\tBBBP\t-\tprobe\t\t\t-\t\t\t\n'
} > "$W"

echo 2 > "$D/cursor_smoke"; : > "$D/lock_smoke"
QUEUE="$W" CURSOR="$D/cursor_smoke" QLOCK="$D/lock_smoke" WALL_SECONDS=7200 \
  bash "$P/analysis/mose_replication_v2/eval_worker.sh"

echo; echo "===== SMOKE CHECKS ====="
f_fix=$V2/rollups/none/BBBP/metrics_mose__fixed__nonGAT_unk-exclude.csv
f_lrn=$V2/rollups/none/BBBP/metrics_mose__learnable_shared__GAT_unk-exclude.csv
echo "-- nonGAT fixed rollup: backbones present (must NOT contain GAT) --"
[ -s "$f_fix" ] && cut -d, -f3 "$f_fix" | tail -n +2 | sort -u | tr '\n' ' '; echo
echo "-- GAT learnable rollup: backbone + unk_mode --"
[ -s "$f_lrn" ] && { echo "cols:"; head -1 "$f_lrn" | tr ',' '\n' | grep -nE 'unk_mode|backbone'; \
   echo "rows:"; cut -d, -f3,8 "$f_lrn" 2>/dev/null | head; }
echo "-- probe did_base/did_learn columns --"
head -1 "$V2/probe/probe_vsvanilla_BBBP.csv" 2>/dev/null | tr ',' '\n' | grep -nE 'did_base|did_learn' || echo "  (probe csv missing)"
echo; echo "===== AUDIT ====="
python3 "$P/analysis/mose_replication_v2/audit_rollups.py" --v2 "$V2" || true
echo; echo "===== MANIFEST (source -> dest provenance) ====="
python3 "$P/analysis/mose_replication_v2/build_manifest.py" --v2 "$V2" || true
column -t -s, "$V2/_deploy/manifest.csv" 2>/dev/null | cut -c1-200 || true
