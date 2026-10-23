#!/usr/bin/env bash
# variants_launch.sh — run LOCALLY. Submits variants_worker.sh to the HPC through sbatch stdin
# (no file copy), as job arrays of claim-based workers; re-run to add workers.
# The worker's R_TABLE_FILLED_AT_LAUNCH line is replaced by r_table.sh (same dir) first.
#   GPU pool (big datasets) : NGPU workers, --gres=gpu:1 -c 2, partition GPU_PART
#   CPU pool (small datasets): NCPU workers, --gres=gpu:0 -c 2, partition CPU_PART
# Env passed to workers: POOL_DATASETS DEVICE CLIP SMOKE (+ optional VOCABS PRESETS GROUPINGS).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/motifsat_paper_deliverables_v1/_dispatch_variants/logs
SMOKE="${SMOKE:-0}"; CLIP="${CLIP:-}"
NGPU="${NGPU:-48}"; NCPU="${NCPU:-86}"
GPU_PART="${GPU_PART:-preempt}"; CPU_PART="${CPU_PART:-share,preempt}"
# ground-truth sets first inside each pool (the worker loops dataset-outermost)
GPU_DATASETS="${GPU_DATASETS:-Benzene_Verified_GT Alkane_Carbonyl_Verified_GT hERG Mutagenicity}"
CPU_DATASETS="${CPU_DATASETS:-Fluoride_Carbonyl_Verified_GT BBBP esol Lipophilicity}"
[ "$SMOKE" = 1 ] && { NGPU=1; NCPU=0; }

W="$(mktemp)"; trap 'rm -f "$W"' EXIT
python3 - "$HERE/variants_worker.sh" "$HERE/r_table.sh" "$W" <<'PY'
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
        -t 12:00:00 -J variants_$tag -o $LOG/${tag}_%A_%a.out \
        --export=ALL,POOL_DATASETS='$dss',DEVICE=$dev,CLIP='$CLIP',SMOKE=$SMOKE${VOCABS:+,VOCABS='$VOCABS'}${PRESETS:+,PRESETS='$PRESETS'}${GROUPINGS:+,GROUPINGS='$GROUPINGS'}" < "$W"
}
submit "$NGPU" cuda "$GPU_PART" gpu:1 16G "$GPU_DATASETS" gpu
submit "$NCPU" cpu  "$CPU_PART" gpu:0 8G  "$CPU_DATASETS" cpu
