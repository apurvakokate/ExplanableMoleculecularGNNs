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
export DATA_ROOT="/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/"
export VOCAB_ROOT="$_CFG_DIR/vocab_output"    # phase1 writes here
export OUT_ROOT="$_CFG_DIR/results"           # model training writes here
export PROCESSED_ROOT="$_CFG_DIR/processed"   # PyG processed .pt cache
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
export EPOCHS="${EPOCHS:-100}"
# Per-conv normalization passed to all phase5 trainers (l2 | layernorm | none)
export CONV_NORMALIZE="${CONV_NORMALIZE:-l2}"
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
#   MotifBreakdown/generate_vocab_rules.py
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

# ── Phase 5 resume ────────────────────────────────────────────────────────────
# Skip runs whose out_dir already has summary.json + best_model.pt (default on).
# phase5_* enforces exactly ONE dataset in DATASETS per invocation.
#
#   export DATASETS=mutag
#   export FOLDS=0
#   export VOCAB_FOCUS=rbrics,all_fallback_bpe
#   export SKIP_EXISTING=1          # default — skip completed runs
#   export FORCE_RERUN=1            # redo everything (overrides SKIP_EXISTING)
#   bash run_experiments.sh phase5_vanilla
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
# Set WANDB_FLAGS to enable logging for phase5 training runs.
# Example: export WANDB_FLAGS="--use_wandb --wandb_project ChemIntuit"
export WANDB_FLAGS="${WANDB_FLAGS:-}"

# ── Confirmation ──────────────────────────────────────────────────────────────
echo "experiment_config.sh loaded"
echo "  PROJECT    = $PROJECT"
echo "  DATA_ROOT  = $DATA_ROOT"
echo "  VOCAB_ROOT = $VOCAB_ROOT"
echo "  OUT_ROOT   = $OUT_ROOT"
echo "  DATASETS   = $DATASETS"
echo "  CSV        = $DATASETS_CSV"
echo "  SPECIAL    = $DATASETS_SPECIAL"
echo "  CONV_NORM  = $CONV_NORMALIZE"
echo "  BACKBONES  = $BACKBONES"
echo "  MUTAG_ROOT = $MUTAG_DATA_ROOT"
echo "  OGB_ROOT   = $OGB_DATA_ROOT"
if [ -n "$VOCAB_FOCUS" ]; then
    echo "  VOCAB_FOCUS= $VOCAB_FOCUS"
fi