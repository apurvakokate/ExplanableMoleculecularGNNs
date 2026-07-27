#!/usr/bin/env bash
# final_v2_env.sh — environment for the full-coverage run (config/experiment_matrix.yaml).
# SOURCE before any stage/dispatch/worker script. Overrides experiment_config.sh.
set -uo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT="${PROJECT:-$(cd "$_HERE/.." && pwd)}"
# conda env (rdkit / torch / PyG) — inherited by children. Idempotent.
if [ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV:-l2xgnn}" ]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "${CONDA_ENV:-l2xgnn}"
fi
# shellcheck disable=SC1091
source "$PROJECT/experiment_config.sh"

export OUT_ROOT="$PROJECT/final_v2"
export PROCESSED_ROOT="$PROJECT/processed_final_v2"
export VOCAB_ROOT="$PROJECT/vocab_final_v2"
export FINAL_TABLES="$OUT_ROOT/tables"
export FINAL_LOGS="$OUT_ROOT/logs"
export QUEUE_DIR="$OUT_ROOT/queue"
export CELL_TIMING_LOG="$FINAL_LOGS/cell_timing.tsv"

export CONV_NORMALIZE="none"
export EPOCHS="${EPOCHS:-500}"
# DNF GT planting (relabelled regime); tiers come from config (dnf_k2_r1 … dnf_k3_r2)
export RULE_ENGINE="dnf" RULE_TIERS="1" RULE_DNF_N="${RULE_DNF_N:-2}" RULE_DNF_ARMS="${RULE_DNF_ARMS:-balanced}"
# MOSE filtered-only (as audited); ante-hoc use their reg_config defaults
export MOSE_BASE=0
export WANDB_MODE=disabled
export SKIP_EXISTING="${SKIP_EXISTING:-1}"      # reuse final_v1 checkpoints/summaries where present

# holders as "jobid:cores:gpus"  (dgxh 64c/1g, dgx2 46c/1g, gpu1 20c/4g)
export HOLDERS="${HOLDERS:-20774533:64:1 20774534:46:1 20774535:20:4}"

mkdir -p "$OUT_ROOT" "$FINAL_TABLES" "$FINAL_LOGS" "$QUEUE_DIR"
