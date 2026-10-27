#!/usr/bin/env bash
# lsweep_launch.sh — run LOCALLY. Submits lsweep_worker.sh to the HPC through sbatch stdin (no file copy),
# as job arrays of claim-based workers; re-run to add workers.
# The worker's R_TABLE_FILLED_AT_LAUNCH line is replaced by r_table.sh (as r_for_own) and
# r_table_paper.sh (as r_for_paper), both from this directory.
#   GPU pool (big datasets + mutag): NGPU workers, --gres=gpu:1 -c 2, partition GPU_PART
#   CPU pool (small datasets)      : NCPU workers, --gres=gpu:0 -c 2, partition CPU_PART
# Defaults leave room for the planted-GT campaign running alongside (48 GPU + 60 CPU workers).
# SMOKE=1: 3 GPU workers, Benzene + mutag (both data paths), every setting, fold 0, GIN, 2 epochs.
# Env passed to workers: POOL_DATASETS DEVICE SMOKE PAR (+ optional SETTINGS BACKBONES FOLDS);
# lists travel ':'-joined (no spaces through ssh + sbatch --export); the worker splits them.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/motifsat_paper_deliverables_v1/_dispatch_lsweep/logs
SMOKE="${SMOKE:-0}"; PAR="${PAR:-2}"
NGPU="${NGPU:-16}"; NCPU="${NCPU:-40}"
GPU_PART="${GPU_PART:-gpu,preempt}"; CPU_PART="${CPU_PART:-share,preempt}"
# ground-truth sets first inside each pool (the worker loops dataset-outermost)
GPU_DATASETS="${GPU_DATASETS:-Benzene_Verified_GT Alkane_Carbonyl_Verified_GT mutag hERG Mutagenicity}"
CPU_DATASETS="${CPU_DATASETS:-Fluoride_Carbonyl_Verified_GT BBBP esol Lipophilicity}"
[ "$SMOKE" = 1 ] && { NGPU="${NGPU_SMOKE:-3}"; NCPU=0; GPU_DATASETS="Benzene_Verified_GT mutag"; }

W="$(mktemp)"; trap 'rm -f "$W"' EXIT
python3 - "$HERE/lsweep_worker.sh" "$HERE/r_table.sh" "$HERE/r_table_paper.sh" "$W" <<'PY'
import re, sys
w, own, paper, out = sys.argv[1:5]
src = open(w).read()
assert src.count('R_TABLE_FILLED_AT_LAUNCH') == 1
tabs = []
for path, name in ((own, 'r_for_own'), (paper, 'r_for_paper')):
    t = open(path).read().strip()
    assert t.count('r_for(){') == 1, path
    tabs.append(t.replace('r_for(){', f'{name}(){{', 1))
open(out, 'w').write(src.replace('R_TABLE_FILLED_AT_LAUNCH', '\n'.join(tabs)))
PY
bash -n "$W"

submit(){  # n device part gres mem datasets tag
    local n=$1 dev=$2 part=$3 gres=$4 mem=$5 dss=$6 tag=$7
    [ "$n" -gt 0 ] || { echo "skip $tag"; return 0; }
    ssh osuhpc "mkdir -p $LOG && sbatch --requeue --array=1-$n -p $part --gres=$gres -c 2 --mem=$mem \
        -t 12:00:00 -J lsweep_$tag -o $LOG/${tag}_%A_%a.out \
        --export=ALL,POOL_DATASETS=${dss// /:},DEVICE=$dev,SMOKE=$SMOKE,PAR=$PAR${SETTINGS:+,SETTINGS=${SETTINGS// /:}}${BACKBONES:+,BACKBONES=${BACKBONES// /:}}${FOLDS:+,FOLDS=${FOLDS// /:}}" < "$W"
}
submit "$NGPU" cuda "$GPU_PART" gpu:1 16G "$GPU_DATASETS" gpu
submit "$NCPU" cpu  "$CPU_PART" gpu:0 12G "$CPU_DATASETS" cpu
