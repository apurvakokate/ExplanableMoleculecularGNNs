#!/usr/bin/env bash
# sfo_launch.sh — deploy GPU + CPU worker pools for MotifSAT on the SFO vocab.
# Fork of base_launch.sh -> drives sfo_worker.sh (writes sfo_runs/, per-fold roots).
# Claim-based pull: RE-RUN anytime to ADD workers; they self-balance.
#
#   SMOKE=1  -> 1 GPU + 8 CPU workers, all 9 presets on BBBP / fold0 / GIN (9 cells).
#   default  -> FINAL: 64 GPU + 86 CPU workers = 300 cores.
#
# Overridable: SMOKE NGPU NCPU CPU_CORES GPU_DATASETS CPU_DATASETS GPU_PART CPU_PART
#              EPOCHS PATIENCE BACKBONES  (propagate via --export=ALL).
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
W="$REPO/motifsat_paper_deliverables_v1/_scripts/sfo_worker.sh"
LOG="$REPO/motifsat_paper_deliverables_v1/_dispatch_sfo/logs"; mkdir -p "$LOG"

SMOKE="${SMOKE:-0}"
GPU_PART="${GPU_PART:-preempt}"; CPU_PART="${CPU_PART:-preempt}"

if [ "$SMOKE" = 1 ]; then
    NGPU="${NGPU:-1}"; NCPU="${NCPU:-8}"; CPU_CORES="${CPU_CORES:-2}"
    GPU_DATASETS="${GPU_DATASETS:-BBBP}"; CPU_DATASETS="${CPU_DATASETS:-BBBP}"
else
    NGPU="${NGPU:-64}"; NCPU="${NCPU:-86}"; CPU_CORES="${CPU_CORES:-2}"
    GPU_DATASETS="${GPU_DATASETS:-Benzene_Verified_GT hERG Alkane_Carbonyl_Verified_GT Mutagenicity}"
    CPU_DATASETS="${CPU_DATASETS:-BBBP esol Lipophilicity Fluoride_Carbonyl_Verified_GT}"
fi

submit_pool(){  # n device part gres cores mem datasets tag
    local n=$1 dev=$2 part=$3 gres=$4 cores=$5 mem=$6 dss=$7 tag=$8 i ok=0
    [ "$n" -gt 0 ] || { echo "skip $tag (n=0)"; return 0; }
    for i in $(seq 1 "$n"); do
        POOL_DATASETS="$dss" DEVICE="$dev" SMOKE="$SMOKE" \
            sbatch --requeue -p "$part" --gres="$gres" -c "$cores" --mem="$mem" -t 12:00:00 \
                -J "sfomd_${tag}" -o "$LOG/${tag}_%j.out" --export=ALL "$W" >/dev/null \
            && ok=$((ok+1)) || echo "  [FAIL submit] $tag #$i"
    done
    echo "submitted $ok/$n $tag workers (device=$dev pool=[$dss])"
}

echo "=== SFO-runs deploy: SMOKE=$SMOKE  NGPU=$NGPU  NCPU=$NCPU ==="
submit_pool "$NGPU" cuda "$GPU_PART" gpu:1 2            16G "$GPU_DATASETS" gpu
submit_pool "$NCPU" cpu  "$CPU_PART" gpu:0 "$CPU_CORES" 8G  "$CPU_DATASETS" cpu
echo "=== deployed. Status: base_status.py --runs sfo_runs --dispatch _dispatch_sfo --report ==="
echo "=== failures -> $REPO/motifsat_paper_deliverables_v1/_dispatch_sfo/failures.tsv ==="
