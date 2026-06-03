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

# ── Datasets ──────────────────────────────────────────────────────────────────
# Choose from: Mutagenicity BBBP hERG Benzene Alkane_Carbonyl Fluoride_Carbonyl
#              esol Lipophilicity freesolv tox21
#              ogbg-molhiv ogbg-molbace ogbg-molbbbp (need node_encoder=atom_encoder)
export DATASETS="Mutagenicity BBBP Benzene"

# ── Cross-validation folds ────────────────────────────────────────────────────
export FOLDS="0 1 2 3 4"

# ── Model architecture ────────────────────────────────────────────────────────
# All backbone architectures to train.  Add or remove as needed.
# Each backbone produces its own subdirectory in results/.
export BACKBONES="${BACKBONES:-GIN GCN SAGE}"  # GIN | GCN | SAGE | GAT | PNA

# Legacy single-backbone variable kept for compatibility (used by _check_paths echo).
export BACKBONE="${BACKBONE:-GIN}"
export NODE_ENCODER="onehot"                  # onehot | linear | atom_encoder (OGB only)
export EPOCHS="100"

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

# ── W&B (optional) ────────────────────────────────────────────────────────────
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