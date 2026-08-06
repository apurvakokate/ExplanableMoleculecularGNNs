#!/usr/bin/env bash
# ablation_launch.sh — deploy the two worker pools (GPU + CPU) for the ablation.
# Workers are claim-based pull (ablation_worker.sh): RE-RUN this anytime to ADD workers;
# they self-balance and never duplicate a running/done cell. Datasets can be moved between
# pools freely (edit GPU_DATASETS/CPU_DATASETS and re-run) — safe mid-run.
#
# FIRST ATTEMPT = REAL only (TIER=real). Planted is staged; deploy it later with TIER=planted.
#
# Overridable: NGPU NCPU CPU_CORES TIER GPU_DATASETS CPU_DATASETS GPU_PART CPU_PART DONE_FILE
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
W=$REPO/ablation_worker.sh
LOG=$REPO/ablation_v2/_dispatch/logs; mkdir -p "$LOG"

TIER="${TIER:-real}"
# routing by dataset size (big -> GPU): Benzene 12k, hERG 9.9k, Alkane 8.7k, Mutagenicity 7.7k.
GPU_DATASETS="${GPU_DATASETS:-}"                                                # ablation_v2: CPU-only
CPU_DATASETS="${CPU_DATASETS:-Alkane_Carbonyl_Verified_GT Fluoride_Carbonyl_Verified_GT}"
NGPU="${NGPU:-0}"; NCPU="${NCPU:-125}"; CPU_CORES="${CPU_CORES:-2}"
GPU_PART="${GPU_PART:-preempt}"; CPU_PART="${CPU_PART:-preempt,share}"
DONE_FILE="${DONE_FILE:-summary_splits.json}"

submit_pool(){  # N  DEVICE  PART  GRES  CORES  MEM  DATASETS  tag
  local n=$1 dev=$2 part=$3 gres=$4 cores=$5 mem=$6 dss=$7 tag=$8 i ok=0
  for i in $(seq 1 "$n"); do
    POOL_DATASETS="$dss" DEVICE="$dev" TIER="$TIER" DONE_FILE="$DONE_FILE" \
      sbatch --requeue -p "$part" --gres="$gres" -c "$cores" --mem="$mem" -t 12:00:00 \
        -J "abl_${tag}" -o "$LOG/${tag}_%j.out" --export=ALL "$W" >/dev/null \
      && ok=$((ok+1)) || echo "  [FAIL submit] $tag #$i"
  done
  echo "submitted $ok/$n $tag workers (device=$dev, pool=[$dss])"
}

echo "=== ablation deploy: TIER=$TIER  DONE_FILE=$DONE_FILE ==="
submit_pool "$NGPU" cuda "$GPU_PART" gpu:1     2          16G "$GPU_DATASETS" gpu
submit_pool "$NCPU" cpu  "$CPU_PART" gpu:0     "$CPU_CORES" 8G  "$CPU_DATASETS" cpu
echo "=== deployed: $NGPU GPU + $NCPU CPU workers. Re-run to add more. Failures -> $REPO/ablation_v2/_dispatch/failures.tsv ==="
