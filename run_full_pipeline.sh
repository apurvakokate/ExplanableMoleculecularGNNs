#!/usr/bin/env bash
# =============================================================================
# run_full_pipeline.sh — end-to-end phase 0 -> 6 for ONE dataset.
#
# HPC — BBBP full sweep (after archival, clean start):
#   bash run_full_pipeline.sh full fresh
#
# HPC — mutag full (fold 0 only; needs phase0 export + TUDataset raw/):
#   bash run_full_pipeline.sh mutag full fresh
#   bash run_full_pipeline.sh submit mutag full fresh
#
# Submit to SLURM (edit SLURM_* / SLURM_SETUP below first):
#   bash run_full_pipeline.sh submit full fresh
#
# Other examples:
#   bash run_full_pipeline.sh                     # smoke: fold 0, GIN, MOSE filtered
#   bash run_full_pipeline.sh full                # full sweep, resume incomplete runs
#   bash run_full_pipeline.sh hERG full fresh
#
# Logs: logs/pipeline_<dataset>_<mode>_<timestamp>.log
# =============================================================================

set -u
_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$_REPO"

# ── Defaults ──────────────────────────────────────────────────────────────────
DATASET="${DATASET:-BBBP}"
MODE="${MODE:-smoke}"              # smoke | full
FRESH="${FRESH:-0}"
# Do not pre-init FAIL_FAST / SKIP_PHASE0 — MODE block sets defaults via ${VAR:-…}
SUBMIT=0

export VOCAB_FOCUS="${VOCAB_FOCUS:-rbrics,all_fallback_bpe}"
export RULE_INDEX="${RULE_INDEX:-0}"

# ── SLURM (for "submit" only) ───────────────────────────────────────────────
SLURM_JOB_NAME="${SLURM_JOB_NAME:-bbbp_full}"
SLURM_PARTITION="${SLURM_PARTITION:-gpu}"
SLURM_TIME="${SLURM_TIME:-72:00:00}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_MEM="${SLURM_MEM:-64G}"
SLURM_SETUP="${SLURM_SETUP:-}"    # e.g. 'module load cuda/12.1; conda activate chemintuit'

# ── W&B: offline on HPC ─────────────────────────────────────────────────────
export WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-ChemIntuit}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_FLAGS="--use_wandb --wandb_project $WANDB_PROJECT"
[ -n "$WANDB_ENTITY" ] && export WANDB_FLAGS="$WANDB_FLAGS --wandb_entity $WANDB_ENTITY"

RUN_ANALYZE="${RUN_ANALYZE:-1}"
# ANALYZE_ARGS set after CLI parses DATASET (below).

# ── CLI: [submit] [dataset] [full|smoke] [fresh] ─────────────────────────────
for _tok in "$@"; do
    case "$_tok" in
        submit) SUBMIT=1 ;;
        full|smoke) MODE="$_tok" ;;
        fresh) FRESH=1 ;;
        *) DATASET="$_tok" ;;
    esac
done

# ── MODE presets (applied after CLI) ──────────────────────────────────────────
if [ "$MODE" = "full" ]; then
    FOLDS="${FOLDS:-0 1 2 3 4}"
    BACKBONES="${BACKBONES:-GIN GCN SAGE GAT PNA}"
    EPOCHS="${EPOCHS:-500}"
    MOSE_BASE="${MOSE_BASE:-1}"
    FAIL_FAST="${FAIL_FAST:-1}"
    SKIP_PHASE0="${SKIP_PHASE0:-1}"
else
    FOLDS="${FOLDS:-0}"
    BACKBONES="${BACKBONES:-GIN}"
    EPOCHS="${EPOCHS:-200}"
    MOSE_BASE="${MOSE_BASE:-0}"
    FAIL_FAST="${FAIL_FAST:-0}"
    SKIP_PHASE0="${SKIP_PHASE0:-0}"
fi
export FOLDS BACKBONES EPOCHS MOSE_BASE FAIL_FAST SKIP_PHASE0

# ── Dataset-specific overrides (mutag / OGB) ─────────────────────────────────
# mutag: fold 0 only, phase0 export required, no synthetic GT (phase4).
_is_special_dataset() {
    case "$1" in
        mutag|ogbg-*) return 0 ;;
        *) return 1 ;;
    esac
}
if _is_special_dataset "$DATASET"; then
    FOLDS="${FOLDS:-0}"
    export FOLDS
    SKIP_PHASE0=0
    export SKIP_PHASE0
    if [ "$DATASET" = "mutag" ]; then
        SLURM_JOB_NAME="${SLURM_JOB_NAME:-mutag_full}"
        SLURM_TIME="${SLURM_TIME:-48:00:00}"
    fi
fi

# Analyze scoped to pipeline DATASET (avoids scanning unrelated results on disk).
_base_analyze="${ANALYZE_ARGS:---skip_regenerate}"
if [[ "$_base_analyze" == *"--dataset"* ]]; then
    export ANALYZE_ARGS="$_base_analyze"
else
    export ANALYZE_ARGS="$_base_analyze --dataset $DATASET"
fi

if [ "$FRESH" = "1" ]; then
    export FORCE_PHASE1=1 FORCE_RERUN=1 SKIP_EXISTING=0
else
    export SKIP_EXISTING="${SKIP_EXISTING:-1}"
fi

# ── SLURM submit ──────────────────────────────────────────────────────────────
if [ "$SUBMIT" = "1" ]; then
    mkdir -p logs
    _fresh_arg=""
    [ "$FRESH" = "1" ] && _fresh_arg="fresh"
    SBATCH_SCRIPT="logs/sbatch_${DATASET}_${MODE}_$(date +%Y%m%d_%H%M%S).sh"
    cat > "$SBATCH_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${SLURM_JOB_NAME}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --time=${SLURM_TIME}
#SBATCH --gres=gpu:${SLURM_GPUS}
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --output=logs/slurm_${DATASET}_${MODE}_%j.out
#SBATCH --error=logs/slurm_${DATASET}_${MODE}_%j.err

set -uo pipefail
cd "$_REPO"
EOF
    if [ -n "$SLURM_SETUP" ]; then
        printf '%s\n' "$SLURM_SETUP" >> "$SBATCH_SCRIPT"
    fi
    cat >> "$SBATCH_SCRIPT" <<EOF
exec bash run_full_pipeline.sh ${DATASET} ${MODE} ${_fresh_arg}
EOF
    chmod +x "$SBATCH_SCRIPT"
    echo "Submitting: $SBATCH_SCRIPT"
    sbatch "$SBATCH_SCRIPT"
    exit $?
fi

export DATASETS="$DATASET"
# CSV benchmarks only — mutag/OGB use source GT, not phase4 synthetic relabel.
if _is_special_dataset "$DATASET"; then
    export DATASETS_CSV="${DATASETS_CSV:-}"
else
    export DATASETS_CSV="${DATASETS_CSV:-$DATASET}"
fi

# shellcheck disable=SC1091
source ./experiment_config.sh

mkdir -p logs
LOG="logs/pipeline_${DATASET}_${MODE}_$(date +%Y%m%d_%H%M%S).log"

PHASES=()
[ "$SKIP_PHASE0" != "1" ] && PHASES+=( phase0 )
PHASES+=( phase1 phase2 phase3 )
# phase4 synthetic GT applies only to CSV benchmarks in GT_SUPPORTED_DATASETS.
if ! _is_special_dataset "$DATASET"; then
    PHASES+=( phase4 )
fi
PHASES+=(
  phase5_vanilla phase5_mose phase5_gsat phase5_motifsat phase5_baselines
)
# GT vanilla + post-hoc baselines (CSV benchmarks with phase-4 gt_cache only).
if ! _is_special_dataset "$DATASET"; then
    PHASES+=( phase5_vanilla_gt phase5_baselines_gt )
fi
PHASES+=( collect )
[ "$RUN_ANALYZE" = "1" ] && PHASES+=( analyze )

run_pipeline() {
  echo "############################################################"
  echo "# FULL PIPELINE — dataset=$DATASET   mode=$MODE   fresh=$FRESH"
  echo "#   VOCAB_FOCUS=$VOCAB_FOCUS  FOLDS='$FOLDS'  BACKBONES='$BACKBONES'"
  echo "#   EPOCHS=$EPOCHS  MOSE_BASE=$MOSE_BASE  RULE_INDEX=$RULE_INDEX"
  echo "#   SKIP_PHASE0=$SKIP_PHASE0  FAIL_FAST=$FAIL_FAST  DATASETS_CSV='${DATASETS_CSV:-}'"
  echo "#   SKIP_EXISTING=${SKIP_EXISTING:-?}  WANDB_MODE=$WANDB_MODE"
  echo "#   ANALYZE_ARGS=$ANALYZE_ARGS"
  echo "#   PROJECT=$PROJECT"
  echo "#   started $(date)"
  echo "############################################################"

  local phase rc pipeline_rc=0
  declare -a results=()

  for phase in "${PHASES[@]}"; do
    echo ""
    echo "==================== $phase ($(date +%H:%M:%S)) ===================="
    bash run_experiments.sh "$phase"
    rc=$?
    results+=("$phase: exit $rc")
    if [ "$rc" -ne 0 ]; then
      echo "  [error] $phase exited $rc"
      pipeline_rc=$rc
      if [ "$FAIL_FAST" = "1" ]; then
        echo "  FAIL_FAST=1 — stopping pipeline."
        break
      fi
      echo "  [warn] continuing with remaining phases."
    fi
  done

  echo ""
  echo "############################################################"
  echo "# PIPELINE SUMMARY — dataset=$DATASET   finished $(date)"
  printf '#   %s\n' "${results[@]}"
  echo "############################################################"
  return "$pipeline_rc"
}

run_pipeline 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo ""
echo "Log saved to: $LOG"
exit "$rc"
