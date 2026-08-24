#!/usr/bin/env bash
# =============================================================================
# make_paper_deliverables.sh
#   Regenerate ALL paper tables + the importance-impact plot from the sealed
#   mose_replication bundle (+ the source-GT GT-ROC recompute for table 3).
#
#   Deliverables written to $OUT (default: <ROOT>/paper_deliverables/):
#     1. classification_auc.tex, regression_rmse_orig.tex, regression_rmse_std.tex
#          Model performance: Vanilla / MoSE / MoSE_U / GSAT, mean±std over folds.
#     2. grouped_pearson_u.tex
#          Grouped (unweighted) Pearson, real labels. Filt = exclude-unk (kept
#          motifs only) / Full = include-unk (full vocab). Axis-A kept-set filter.
#     3. gtroc_instance_tvt.tex
#          Instance GT-ROC pooled over train+valid+test (graph-weighted). Filt =
#          exclude (kept-motif nodes) / Full = include (all-node GT recovery).
#     4. importance_vs_impact_mose_real.{png,pdf}
#          MoSE-fixed / MoSE+Learn-Unknown / Vanilla motif impact vs MoSE-fixed
#          importance, kept motifs, train+val+test.
#
#   PREREQ for table 3: the per-split GT-ROC recompute must exist at $SRCGT.
#   Produce it once with:   sbatch analysis/srcgt_gtroc_array.sbatch
#   (reads models/atts from the bundle; writes $SRCGT; ~1h on the array).
#
#   Everything is EVALUATION on existing checkpoints — no retraining.
#
#   Significance highlighting (analysis/table_sig.py): per (dataset,backbone) row
#   and within each unk condition, the best cell is \textbf{bold} and every cell
#   not significantly worse than it (Welch's two-sided t-test on the per-fold
#   mean/std, Holm-corrected) is \underline{underlined}. Toggle via env vars:
#     SIG=0            disable highlighting
#     SIG_ALPHA=0.10   significance level (default 0.05)
#     SIG_STYLE=bold_all   bold best+ties instead of bold-best/underline-ties
#   NOTE: Welch is a SUMMARY-ONLY (unpaired) test — it ignores the fold pairing;
#   a paired t-test on the raw folds would be stronger. State the test in captions.
#
#   Override any path via env vars, e.g.:
#     ROOT=/path/to/Claude+Cursor OUT=~/tables bash analysis/make_paper_deliverables.sh
# =============================================================================
set -uo pipefail

ROOT=${ROOT:-/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor}
BUNDLE=${BUNDLE:-$ROOT/mose_replication}
SRCGT=${SRCGT:-$ROOT/eval_srcgt_gtroc_v1}
FOLDS=${FOLDS:-/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS}
OUT=${OUT:-$ROOT/paper_deliverables}

HERE="$(cd "$(dirname "$0")" && pwd)"     # .../analysis
REPO="$(dirname "$HERE")"
mkdir -p "$OUT"
export PYTHONPATH="$REPO" MPLBACKEND=Agg BUNDLE ROOT SRCGT FOLDS

echo "bundle=$BUNDLE"; echo "out=$OUT"; echo

echo "== (1) Model-performance tables =="
OUTDIR="$OUT" python3 "$HERE/gen_model_perf_tables.py" | sed 's/^/   /'

echo "== (2) Grouped-Pearson (Filt=kept / Full=full-vocab; unweighted) =="
python3 "$HERE/gen_grouped_pearson_table.py" --build --metric grouped_pearson_u \
  --gat_rollups   "$BUNDLE/eval_gath1/gath1_rollup_all.csv" "$BUNDLE/eval_gath1/gath1_rollup_include.csv" \
  --other_rollups "$BUNDLE/eval_real/eval_real_rollup_all.csv" "$BUNDLE/eval_real/eval_real_mose_learn.csv" \
  --out "$OUT/grouped_pearson_u.tex" && echo "   wrote grouped_pearson_u.tex"

echo "== (3) Instance GT-ROC, train+valid+test (Filt=kept-node / Full=all-node GT) =="
if [ -d "$SRCGT" ]; then
  OUTDIR="$OUT" python3 "$HERE/gen_gtroc_tvt_table.py" | sed 's/^/   /'
else
  echo "   !! SRCGT not found: $SRCGT"
  echo "   !! run first:  sbatch $HERE/srcgt_gtroc_array.sbatch   (then rerun this script)"
fi

echo "== (4) Importance-vs-Impact plot =="
python3 "$HERE/plot_importance_vs_impact_bars.py" --plots 1 --tier real \
  --antehoc_root "$BUNDLE/eval_real" "$BUNDLE/eval_gath1" --mose_unk unk-fixed \
  --mose_lu_root "$BUNDLE/eval_real" "$BUNDLE/eval_gath1" \
  --posthoc_root "$ROOT/posthoc_v1" \
  --save_dir "$OUT" --formats png pdf 2>&1 | sed 's/^/   /'

echo
echo "== DONE -> $OUT =="
ls -1 "$OUT"
