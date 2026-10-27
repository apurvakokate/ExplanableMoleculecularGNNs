#!/usr/bin/env bash
# planted_launch.sh — run LOCALLY. Submits planted_worker.sh through sbatch stdin (no file copy) as job arrays
# of claim-based workers; re-run to add workers. R_TABLE_FILLED_AT_LAUNCH is replaced by r_table.sh first.
#   GPU pool: hERG, Mutagenicity (NGPU workers, --gres=gpu:1 -c 2)   CPU pool: BBBP (NCPU workers, gpu:0 -c 2)
# SMOKE=1: one rule per dataset, fold 0, GIN, 2 epochs, separate tree (runs_smoke/), 3 GPU workers.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/motifsat_paper_deliverables_v1/_dispatch_planted/logs
SMOKE="${SMOKE:-0}"; CLIP="${CLIP:-}"; PAR="${PAR:-2}"
NGPU="${NGPU:-48}"; NCPU="${NCPU:-60}"
GPU_PART="${GPU_PART:-gpu,dgx2,preempt}"; CPU_PART="${CPU_PART:-share,preempt}"
GPU_DATASETS="${GPU_DATASETS:-Mutagenicity hERG}"; CPU_DATASETS="${CPU_DATASETS:-BBBP}"
[ "$SMOKE" = 1 ] && { NGPU=3; NCPU=0; GPU_DATASETS="BBBP hERG Mutagenicity"; PAR=1; }

W="$(mktemp)"; trap 'rm -f "$W"' EXIT
python3 - "$HERE/planted_worker.sh" "$HERE/r_table.sh" "$W" <<'PY'
import sys
w, t, out = sys.argv[1:4]
src = open(w).read(); tab = open(t).read()
assert src.count('R_TABLE_FILLED_AT_LAUNCH') == 1
open(out, 'w').write(src.replace('R_TABLE_FILLED_AT_LAUNCH', tab.strip()))
PY
bash -n "$W"
submit(){  # n device part gres mem datasets tag
    local n=$1 dev=$2 part=$3 gres=$4 mem=$5 dss=$6 tag=$7
    [ "$n" -gt 0 ] || { echo "skip $tag"; return 0; }
    ssh osuhpc "mkdir -p $LOG && sbatch --requeue --array=1-$n -p $part --gres=$gres -c 2 --mem=$mem \
        -t 12:00:00 -J planted_$tag -o $LOG/${tag}_%A_%a.out \
        --export=ALL,POOL_DATASETS=${dss// /:},DEVICE=$dev,CLIP=$CLIP,SMOKE=$SMOKE,PAR=$PAR${PRESETS:+,PRESETS=${PRESETS// /:}}${BACKBONES:+,BACKBONES=${BACKBONES// /:}}${FOLDS:+,FOLDS=${FOLDS// /:}}" < "$W"
}
submit "$NGPU" cuda "$GPU_PART" gpu:1 16G "$GPU_DATASETS" gpu
submit "$NCPU" cpu  "$CPU_PART" gpu:0 12G "$CPU_DATASETS" cpu
