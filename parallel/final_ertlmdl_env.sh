#!/usr/bin/env bash
# final_ertlmdl_env.sh — environment for the conservative_ertl_ring {MDL,BPE} FCOL study.
# SOURCE before dispatch. Mirrors final_v2_env.sh with NEW roots + this study's config.
set -uo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT="${PROJECT:-$(cd "$_HERE/.." && pwd)}"
if [ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV:-l2xgnn}" ]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "${CONDA_ENV:-l2xgnn}"
fi
# shellcheck disable=SC1091
source "$PROJECT/experiment_config.sh"

# NEW roots — nothing in final_v2 is overwritten; vocab is built FROM SCRATCH into vocab_ertlmdl.
export OUT_ROOT="$PROJECT/final_ertlmdl"
export PROCESSED_ROOT="$PROJECT/processed_ertlmdl"
export VOCAB_ROOT="$PROJECT/vocab_ertlmdl"
export FINAL_TABLES="$OUT_ROOT/tables"
export FINAL_LOGS="$OUT_ROOT/logs"
export QUEUE_DIR="$OUT_ROOT/queue"
export CELL_TIMING_LOG="$FINAL_LOGS/cell_timing.tsv"

# This study's matrix (2 fragmentations, 8 datasets, planted DEFERRED, no MotifSAT).
export MATRIX="$PROJECT/config/experiment_matrix_ertl_mdl.yaml"
# Vanilla base GNNs are VOCAB-INDEPENDENT -> reuse from final_v2 (seed_vanilla_from_final_v2.sh).
export REUSE_VANILLA_FROM="${REUSE_VANILLA_FROM:-$PROJECT/final_v2}"

export CONV_NORMALIZE="none"
export EPOCHS="${EPOCHS:-500}"
# Planted DEFERRED this pass — no DNF relabel. (When run later: RULE_TIERS=3, easy/medium/hard.)
export RULE_ENGINE="none"
export MOSE_BASE=1                               # our variants are NOT *_filter -> MOSE must run on base vocab
export WANDB_MODE=disabled
export SKIP_EXISTING="${SKIP_EXISTING:-1}"       # reuse seeded vanilla + any completed cells

# Reserve >=300 CPU + >=64 GPU across preempt/dgxh/dgx2/gpu, then export HOLDERS ("jobid:cores:gpus ...").
export HOLDERS="${HOLDERS:-}"

mkdir -p "$OUT_ROOT" "$FINAL_TABLES" "$FINAL_LOGS" "$QUEUE_DIR"
[ -z "$HOLDERS" ] && echo "[final_ertlmdl_env] NOTE: HOLDERS empty — run parallel/reserve_holders.sh, export HOLDERS, then bash parallel/queue_dispatch_v2.sh" >&2
