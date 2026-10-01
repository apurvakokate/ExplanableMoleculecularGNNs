#!/usr/bin/env bash
# ablation_factorial_launch.sh — deploy GPU+CPU worker pools for the FULL 3x2x2 MoSE
# architecture factorial (rbrics_filter, REAL only). DRAFT — review before running.
# Claim-based pull (ablation_factorial_worker.sh): re-run anytime to add workers; they
# self-balance and never duplicate a cell.
#
# Grid = node_encoder{onehot,linear} x conv_normalize{none,l2,layernorm} x graph_pool{add,mean}
#        x unk_mode{fixed,learnable_shared}  (24 combos)
#        over 8 datasets x 5 folds x 5 backbones = 4,800 cells (existing per-split arms skipped).
# big datasets -> GPU, small -> CPU. preempt-first with fallback. mutag is NOT included.
#
# ENQUEUE-CHECK FIRST (prints commands, submits nothing that runs):
#   DRY_RUN=1 NGPU=1 NCPU=0 GPU_DATASETS=hERG bash ablation_factorial_launch.sh
#
# Overridable: NGPU NCPU CPU_CORES GPU_DATASETS CPU_DATASETS GPU_PART CPU_PART DONE_FILE OUT_ROOT DRY_RUN
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
W="$REPO/ablation_factorial_worker.sh"
LOG="$REPO/ablated_completely_v1/_dispatch_fac/logs"; mkdir -p "$LOG"

GPU_PART="${GPU_PART:-preempt,dgx2,gpu}"; CPU_PART="${CPU_PART:-preempt,share}"
NGPU="${NGPU:-16}"; NCPU="${NCPU:-64}"; CPU_CORES="${CPU_CORES:-2}"
DONE_FILE="${DONE_FILE:-summary_splits.json}"; DRY_RUN="${DRY_RUN:-0}"
# route by dataset size (union must cover all 8 REAL datasets, no overlap, NO mutag)
GPU_DATASETS="${GPU_DATASETS:-hERG Mutagenicity Alkane_Carbonyl_Verified_GT Benzene_Verified_GT}"
CPU_DATASETS="${CPU_DATASETS:-BBBP esol Lipophilicity Fluoride_Carbonyl_Verified_GT}"

submit_pool(){  # n device part gres cores mem datasets tag
  local n=$1 dev=$2 part=$3 gres=$4 cores=$5 mem=$6 dss=$7 tag=$8 i ok=0
  [ "$n" -gt 0 ] || { echo "skip $tag (n=0)"; return 0; }
  for i in $(seq 1 "$n"); do
    POOL_DATASETS="$dss" DEVICE="$dev" DONE_FILE="$DONE_FILE" DRY_RUN="$DRY_RUN" OUT_ROOT="${OUT_ROOT:-}" \
      sbatch --requeue -p "$part" --gres="$gres" -c "$cores" --mem="$mem" -t 12:00:00 \
        -J "mosefac_${tag}" -o "$LOG/${tag}_%j.out" --export=ALL "$W" >/dev/null \
      && ok=$((ok+1)) || echo "  [FAIL submit] $tag #$i"
  done
  echo "submitted $ok/$n $tag workers (device=$dev pool=[$dss])"
}

echo "=== mose-factorial fill: NGPU=$NGPU NCPU=$NCPU DRY_RUN=$DRY_RUN ==="
submit_pool "$NGPU" cuda "$GPU_PART" gpu:1 2            16G "$GPU_DATASETS" gpu   # 2 cores/GPU (hard rule)
submit_pool "$NCPU" cpu  "$CPU_PART" gpu:0 "$CPU_CORES" 8G  "$CPU_DATASETS" cpu
echo "=== deployed. Failures -> $REPO/ablated_completely_v1/_dispatch_fac/failures.tsv ==="
