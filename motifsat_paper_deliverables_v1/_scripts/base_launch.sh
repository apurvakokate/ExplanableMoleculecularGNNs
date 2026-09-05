#!/usr/bin/env bash
# base_launch.sh — deploy GPU + CPU worker pools for the MotifSAT base-runs.
# Claim-based pull (base_worker.sh): RE-RUN anytime to ADD workers; they self-balance
# and never duplicate a running/done cell. Move datasets between pools freely (edit
# GPU_DATASETS/CPU_DATASETS) — safe mid-run.
#
#   SMOKE=1  -> 1 GPU + 8 CPU workers, all 9 presets on BBBP / fold0 / GIN (9 cells).
#   default  -> FINAL: 64 GPU workers (2 cores each = 128) + 86 CPU workers (172) = 300 cores.
#
# Overridable: SMOKE NGPU NCPU CPU_CORES GPU_DATASETS CPU_DATASETS GPU_PART CPU_PART
#              EPOCHS BACKBONES  (EPOCHS/BACKBONES propagate via --export=ALL).
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
W="$REPO/motifsat_paper_deliverables_v1/_scripts/base_worker.sh"
LOG="$REPO/motifsat_paper_deliverables_v1/_dispatch/logs"; mkdir -p "$LOG"

SMOKE="${SMOKE:-0}"
GPU_PART="${GPU_PART:-preempt}"; CPU_PART="${CPU_PART:-preempt,share}"

if [ "$SMOKE" = 1 ]; then
    NGPU="${NGPU:-1}"; NCPU="${NCPU:-8}"; CPU_CORES="${CPU_CORES:-2}"
    GPU_DATASETS="${GPU_DATASETS:-BBBP}"; CPU_DATASETS="${CPU_DATASETS:-BBBP}"
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
                -J "msdel_${tag}" -o "$LOG/${tag}_%j.out" --export=ALL "$W" >/dev/null \
            && ok=$((ok+1)) || echo "  [FAIL submit] $tag #$i"
    done
    echo "submitted $ok/$n $tag workers (device=$dev pool=[$dss])"
}

echo "=== base-runs deploy: SMOKE=$SMOKE  NGPU=$NGPU  NCPU=$NCPU ==="
submit_pool "$NGPU" cuda "$GPU_PART" gpu:1 2            16G "$GPU_DATASETS" gpu
submit_pool "$NCPU" cpu  "$CPU_PART" gpu:0 "$CPU_CORES" 8G  "$CPU_DATASETS" cpu
echo "=== deployed. Re-run to add workers. Status: base_status.py --report ==="
echo "=== failures -> $REPO/motifsat_paper_deliverables_v1/_dispatch/failures.tsv ==="
