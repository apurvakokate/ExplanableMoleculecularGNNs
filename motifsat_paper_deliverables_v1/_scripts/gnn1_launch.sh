#!/usr/bin/env bash
# gnn1_launch.sh — deploy GPU + CPU worker pools for the GNN1-design ablation.
# Fork of base_launch.sh. Claim-based pull (gnn1_worker.sh): RE-RUN anytime to ADD
# workers; they self-balance and never duplicate a running/done cell. Move datasets
# between pools freely (edit GPU_DATASETS/CPU_DATASETS) — safe mid-run.
#
# Grid = 32 configs x 8 datasets x 5 folds x 5 backbones = 6400 cells (stem gsat).
#
#   SMOKE=1  -> 1 GPU + 4 CPU workers; anchor + richest config on BBBP+Benzene /
#               fold0 / GIN (confirms plumbing, all 20 artifacts for motif_emb, and
#               GT-ROC emission). See gnn1_worker.sh SMOKE block.
#   default  -> FINAL: 64 GPU workers (2 cores each = 128) + 86 CPU workers (172) = 300 cores.
#
# HARD RULES honored: GPU jobs request exactly 2 cores/GPU; the ~300 CPU cores are on
# CPU-only (gpu:0) units, a separate pool from GPU-node cores; preempt queue.
# NO `sbatch --wrap` (this SLURM build rejects it) — we sbatch the worker script with
# --export=ALL so EPOCHS/BACKBONES/CONFIGS/BASE_PRESET propagate.
#
# Overridable: SMOKE NGPU NCPU CPU_CORES GPU_DATASETS CPU_DATASETS GPU_PART CPU_PART
#              EPOCHS BACKBONES CONFIGS BASE_PRESET.
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
W="$REPO/motifsat_paper_deliverables_v1/_scripts/gnn1_worker.sh"
LOG="$REPO/motifsat_paper_deliverables_v1/_dispatch_gnn1/logs"; mkdir -p "$LOG"

SMOKE="${SMOKE:-0}"
GPU_PART="${GPU_PART:-preempt}"; CPU_PART="${CPU_PART:-preempt,share}"

if [ "$SMOKE" = 1 ]; then
    NGPU="${NGPU:-1}"; NCPU="${NCPU:-4}"; CPU_CORES="${CPU_CORES:-2}"
    GPU_DATASETS="${GPU_DATASETS:-BBBP}"; CPU_DATASETS="${CPU_DATASETS:-Benzene_Verified_GT}"
else
    # FINAL: total 300 cores = 64 GPU workers (gpu:1 -c 2 = 128) + 86 CPU workers (-c 2 = 172).
    NGPU="${NGPU:-64}"; NCPU="${NCPU:-86}"; CPU_CORES="${CPU_CORES:-2}"
    # route by dataset size: big -> GPU, small -> CPU.
    GPU_DATASETS="${GPU_DATASETS:-Benzene_Verified_GT hERG Alkane_Carbonyl_Verified_GT Mutagenicity}"
    CPU_DATASETS="${CPU_DATASETS:-BBBP esol Lipophilicity Fluoride_Carbonyl_Verified_GT}"
fi

submit_pool(){  # n device part gres cores mem datasets tag
    local n=$1 dev=$2 part=$3 gres=$4 cores=$5 mem=$6 dss=$7 tag=$8 i ok=0
    [ "$n" -gt 0 ] || { echo "skip $tag (n=0)"; return 0; }
    for i in $(seq 1 "$n"); do
        POOL_DATASETS="$dss" DEVICE="$dev" SMOKE="$SMOKE" \
            sbatch --requeue -p "$part" --gres="$gres" -c "$cores" --mem="$mem" -t 12:00:00 \
                -J "gnn1_${tag}" -o "$LOG/${tag}_%j.out" --export=ALL "$W" >/dev/null \
            && ok=$((ok+1)) || echo "  [FAIL submit] $tag #$i"
    done
    echo "submitted $ok/$n $tag workers (device=$dev pool=[$dss])"
}

echo "=== gnn1-ablation deploy: SMOKE=$SMOKE  NGPU=$NGPU  NCPU=$NCPU ==="
submit_pool "$NGPU" cuda "$GPU_PART" gpu:1 2            16G "$GPU_DATASETS" gpu
submit_pool "$NCPU" cpu  "$CPU_PART" gpu:0 "$CPU_CORES" 8G  "$CPU_DATASETS" cpu
echo "=== deployed. Re-run to add workers. Status: gnn1_status.py --report ==="
echo "=== failures -> $REPO/motifsat_paper_deliverables_v1/_dispatch_gnn1/failures.tsv ==="
