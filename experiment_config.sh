#!/usr/bin/env bash
# =============================================================================
# experiment_config.sh
# Edit this file once, then source it before any phase:
#   source experiment_config.sh          # from project root
#   source ../experiment_config.sh       # from a subdirectory like MotifBreakdown/
#   bash run_experiments.sh phase1
#
# All paths are anchored to this file's own directory so the script works
# regardless of which directory you source it from.
# =============================================================================

# ── Resolve the directory this file lives in (project root) ──────────────────
# Works whether sourced as:  source ./experiment_config.sh
#                            source /abs/path/to/experiment_config.sh
#                            source ../experiment_config.sh
_CFG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Paths (all absolute, anchored to project root) ────────────────────────────
export PROJECT="$_CFG_DIR"           # repo root: MOSE-GNN/, MotifSAT/, etc.
export DATA_ROOT="${DATA_ROOT:-/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/}"
export VOCAB_ROOT="${VOCAB_ROOT:-$_CFG_DIR/vocab_output}"    # phase1 writes here
export OUT_ROOT="${OUT_ROOT:-$_CFG_DIR/results}"   # override before run: export OUT_ROOT=$PROJECT/results_motifsat_ib
export PROCESSED_ROOT="${PROCESSED_ROOT:-$_CFG_DIR/processed}"   # PyG processed .pt cache
# mutag TUDataset exports (mutag_<fold>.csv under MUTAG_DATA_ROOT; TUDataset in mutag/)
export MUTAG_DATA_ROOT="${MUTAG_DATA_ROOT:-$_CFG_DIR/data}"
# OGB molecule datasets (PyG auto-download cache)
export OGB_DATA_ROOT="${OGB_DATA_ROOT:-$_CFG_DIR/data/ogb}"

# ── Datasets ──────────────────────────────────────────────────────────────────
# CSV benchmarks (FOLDS/{dataset}_{fold}.csv under DATA_ROOT):
export DATASETS_CSV="${DATASETS_CSV:-Mutagenicity BBBP hERG Benzene Alkane_Carbonyl Fluoride_Carbonyl Lipophilicity esol}"
# Special datasets needing phase0 export (mutag TUDataset, OGB PyG):
#   mutag          — source GT; no --use_gt; fold 0 only
#   ogbg-molhiv    — atom_encoder forced at train; fold 0 only
#   ogbg-molbace   — same as molhiv
export DATASETS_SPECIAL="${DATASETS_SPECIAL:-mutag ogbg-molbace ogbg-molhiv}"
# Union used by all phases. Override entirely: export DATASETS="BBBP Benzene"
export DATASETS="${DATASETS:-$DATASETS_CSV $DATASETS_SPECIAL}"
# Lipophilicity / esol (regression) skip rule mining and phase-4 GT only;
# phases 1–3 still build vocabs and coverage plots; phase 5 trains on real labels.

# ── Cross-validation folds ────────────────────────────────────────────────────
export FOLDS="${FOLDS:-0 1 2 3 4}"

# ── Model architecture ────────────────────────────────────────────────────────
# All backbone architectures to train.  Add or remove as needed.
# Each backbone produces its own subdirectory in results/.
export BACKBONES="${BACKBONES:-GIN GCN SAGE GAT PNA}"

# Legacy single-backbone variable — unused (all loops use BACKBONES).
# export BACKBONE="${BACKBONE:-GIN}"
export NODE_ENCODER="${NODE_ENCODER:-onehot}"  # onehot | linear | atom_encoder (OGB only)
export ENCODER_NORM="${ENCODER_NORM:-off}"    # off | on (LayerNorm after encoder; vanilla only)
# Phase 5 training budget (max epochs; early stopping may stop sooner).
export EPOCHS="${EPOCHS:-500}"
# Per-conv normalization for vanilla / GSAT / MotifSAT / baselines (l2 | layernorm | none)
export CONV_NORMALIZE="${CONV_NORMALIZE:-none}"
# MOSE uses the same default (none) — motif weights carry magnitude information.
export MOSE_CONV_NORMALIZE="${MOSE_CONV_NORMALIZE:-none}"
# MOSE multi-explanation (H0/H1/H2) runs post-hoc — see phase multi_explanation.
# Set to 1 only to run inline during MOSE training (not recommended at scale).
export MOSE_RUN_MULTI_EXPLANATION="${MOSE_RUN_MULTI_EXPLANATION:-0}"
# GSAT: learn_edge_att=False (default) uses node scores × for edge message scaling.
# Set GSAT_LEARN_EDGE_ATT=1 for the legacy separate edge-attention MLP path.
export GSAT_LEARN_EDGE_ATT="${GSAT_LEARN_EDGE_ATT:-0}"
# Default injection CLI flags (override for ablation sweeps):
export MOSE_INJ="${MOSE_INJ:---w_feat --w_readout}"          # 101
export MOTIFSAT_INJ="${MOTIFSAT_INJ:---w_feat --w_message --w_readout}"  # 111
export GSAT_INJ="${GSAT_INJ:---w_message}"                    # 010

# ── Phase 3: thresholds ───────────────────────────────────────────────────────
# Thresholds are set per-dataset in a dict — no shell variable needed.
# After reviewing phase 2 coverage plots, edit CHOSEN_THRESHOLD in:
#   SharedModules/data/threshold_config.py
# Key: CHOSEN_THRESHOLD[variant_filter_name][dataset] = threshold_pct
# e.g. CHOSEN_THRESHOLD["all_fallback_bpe_filter"]["Mutagenicity"] = 0.002

# ── Phase 4: rule index (set after reviewing available rules) ─────────────────
# Leave empty until phase 3 is done.
# Only set if not already exported by the caller.
# This lets you do:  export RULE_INDEX=0 && bash run_experiments.sh phase4
export RULE_INDEX="${RULE_INDEX:-}"          # e.g. 0

# ── Phase 4/5: limit to selected fragmentation algorithms ───────────────────
# Comma-separated short names. Unset = all four base variants.
#   rbrics, rbrics_old, rbrics_with_struct_fallback, all_fallback_bpe
# Aliases: old  struct|struct_fallback  all|v4
# Example: export VOCAB_FOCUS=rbrics,all_fallback_bpe
export VOCAB_FOCUS="${VOCAB_FOCUS:-}"
# MOSE phase5_mose runs filtered variants (*_filter) before base variants.
# Set MOSE_BASE=1 to also train MOSE on unfiltered base vocabs (opt-in ablation).
export MOSE_BASE="${MOSE_BASE:-0}"

# ── Phase 0–5 resume ──────────────────────────────────────────────────────────
# Skip work when expected artifacts already exist (default on).
# Phases 0–4: vocabs, coverage plots, filtered vocabs, gt_cache.
# Phase 5: summary.json + best_model.pt per run dir.
#
#   export SKIP_EXISTING=1          # default — skip completed work
#   export FORCE_RERUN=1            # redo everything (overrides SKIP_EXISTING)
#   export FORCE_PHASE1=1           # redo phase1 vocabs only
#
# Run phases 0–4 for all DATASETS × four base variants:
#   bash run_experiments.sh phase0_4
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
# Set WANDB_FLAGS to enable logging for phase5 training runs.
# Example: export WANDB_FLAGS="--use_wandb --wandb_project ChemIntuit"
export WANDB_FLAGS="${WANDB_FLAGS:-}"

# ── Post-hoc explainers (phase5_baselines) ────────────────────────────────────
# GNNExplainer: ``GNNEX_EPOCHS`` optimization steps per test graph (default 200).
# Optional cap for quick sweeps only — unset = all test graphs.
# export GNNEX_MAX_GRAPHS=200
export GNNEX_EPOCHS="${GNNEX_EPOCHS:-200}"
# export PGEX_MAX_GRAPHS=   # unset = all test graphs
# Reuse mode: augment ALREADY-COMPLETED baseline runs with the newer metrics
# (top_bottom / gt_vs_outside) by LOADING their saved explainer scores instead of
# re-optimizing the masks. Skips the expensive GNNExplainer/PGExplainer step and
# processes completed runs WITHOUT FORCE_RERUN (incomplete runs are skipped).
#   REUSE_EXPLAINER_SCORES=1 bash run_experiments.sh phase5_baselines
#   REUSE_EXPLAINER_SCORES=1 bash run_experiments.sh phase5_baselines_gt
export REUSE_EXPLAINER_SCORES="${REUSE_EXPLAINER_SCORES:-0}"

# ── Confirmation ──────────────────────────────────────────────────────────────
echo "experiment_config.sh loaded"
echo "  PROJECT    = $PROJECT"
echo "  DATA_ROOT  = $DATA_ROOT"
echo "  VOCAB_ROOT = $VOCAB_ROOT"
echo "  OUT_ROOT   = $OUT_ROOT"
echo "  DATASETS   = $DATASETS"
echo "  CSV        = $DATASETS_CSV"
echo "  SPECIAL    = $DATASETS_SPECIAL"
echo "  CONV_NORM  = $CONV_NORMALIZE  (MOSE: $MOSE_CONV_NORMALIZE)"
echo "  BACKBONES  = $BACKBONES"
echo "  EPOCHS     = $EPOCHS   (phase 5 max; early stopping may finish earlier)"
echo "  GNNEX      = max_graphs=${GNNEX_MAX_GRAPHS:-all} epochs=$GNNEX_EPOCHS"
echo "  MUTAG_ROOT = $MUTAG_DATA_ROOT"
echo "  OGB_ROOT   = $OGB_DATA_ROOT"
if [ -n "$VOCAB_FOCUS" ]; then
    echo "  VOCAB_FOCUS= $VOCAB_FOCUS"
fi