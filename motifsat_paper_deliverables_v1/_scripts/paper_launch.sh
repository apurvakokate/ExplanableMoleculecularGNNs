#!/usr/bin/env bash
# paper_launch.sh — run LOCALLY. Submits paper_worker.sh to the HPC through sbatch stdin (no file
# copy), as job arrays of claim-based workers; re-run to add workers.
# The worker's R_TABLE_FILLED_AT_LAUNCH line is replaced by r_table_paper.sh (same dir) first.
#   GPU pool (big datasets) : NGPU workers, --gres=gpu:1 -c 2, partition GPU_PART
#   CPU pool (small datasets): NCPU workers, --gres=gpu:0 -c 2, partition CPU_PART
# Env passed to workers: POOL_DATASETS DEVICE CLIP SMOKE PAR (+ optional VOCABS PRESETS BACKBONES REPS);
# lists travel ':'-joined (no spaces through ssh + sbatch --export); the worker splits them.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/motifsat_paper_deliverables_v1/_dispatch_paper/logs
SMOKE="${SMOKE:-0}"; CLIP="${CLIP:-}"; PAR="${PAR:-2}"
NGPU="${NGPU:-48}"; NCPU="${NCPU:-100}"
GPU_PART="${GPU_PART:-gpu,preempt}"; CPU_PART="${CPU_PART:-share,preempt}"
# ground-truth set (mutag) first; by size: molhiv 41,127 > tox21 7,831 > mutag 2,937 on GPU,
# bbbp 2,039 / bace 1,513 / clintox 1,477 on CPU
GPU_DATASETS="${GPU_DATASETS:-mutag ogbg-moltox21 ogbg-molhiv}"
CPU_DATASETS="${CPU_DATASETS:-ogbg-molclintox ogbg-molbace ogbg-molbbbp}"
[ "$SMOKE" = 1 ] && { NGPU="${NGPU_SMOKE:-3}"; NCPU=0; GPU_DATASETS="${SMOKE_DATASETS:-mutag ogbg-molclintox ogbg-molhiv}"; }

W="$(mktemp)"; trap 'rm -f "$W"' EXIT
python3 - "$HERE/paper_worker.sh" "$HERE/r_table_paper.sh" "$W" <<'PY'
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
        -t 12:00:00 -J paper_$tag -o $LOG/${tag}_%A_%a.out \
        --export=ALL,POOL_DATASETS=${dss// /:},DEVICE=$dev,CLIP=$CLIP,SMOKE=$SMOKE,PAR=$PAR${VOCABS:+,VOCABS=${VOCABS// /:}}${PRESETS:+,PRESETS=${PRESETS// /:}}${BACKBONES:+,BACKBONES=${BACKBONES// /:}}${REPS:+,REPS=${REPS// /:}}" < "$W"
}
submit "$NGPU" cuda "$GPU_PART" gpu:1 16G "$GPU_DATASETS" gpu
submit "$NCPU" cpu  "$CPU_PART" gpu:0 12G "$CPU_DATASETS" cpu
