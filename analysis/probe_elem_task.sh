#!/bin/bash
# SLURM ARRAY TASK -- one cell (dataset x backbone x fold) of the element-matched
# node-identity probe. Submitted by probe_elem_launch.sh; not run directly.
#
# Resource shape is the project HARD rule: CPU-only evaluation work, so
# --gres=gpu:0 with exactly 2 cores. OMP/MKL are pinned to 2 so torch and
# sklearn do not oversubscribe the allocation.
#SBATCH --gres=gpu:0
#SBATCH -c 2
set -euo pipefail

P=${P:-/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor}
OUT=${OUT:-$P/probe_elem_v1}
DATA_ROOT=${DATA_ROOT:-/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS}
# vocab_final_v2 (2026-07-27) is the vocab these checkpoints were TRAINED with;
# vocab_final_v3 is 2026-10 and postdates them. summary.json does not record
# vocab_root, so this is pinned here deliberately. A mismatch changes the fold's
# kept_motif_ids -> motif_params shape -> load_state_dict fails loudly.
VOCAB=${VOCAB:-$P/vocab_final_v2}
SPLITS=${SPLITS:-all}
MIN_N=${MIN_N:-30}
TASKLIST=${TASKLIST:?TASKLIST not exported by the launcher}

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKLIST")
[ -n "$LINE" ] || { echo "FATAL: no line ${SLURM_ARRAY_TASK_ID} in $TASKLIST"; exit 2; }
read -r DS BB FOLD <<< "$LINE"

STEM=$OUT/$DS/${BB}_${FOLD}
if [ -f "$STEM.done" ]; then
  echo "SKIP $DS/$BB/$FOLD (already done)"; exit 0
fi

source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh
conda activate l2xgnn
cd "$P"
export PYTHONPATH="$P" WANDB_MODE=disabled
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2 TORCH_NUM_THREADS=2

echo "task=${SLURM_ARRAY_TASK_ID} cell=$DS/$BB/$FOLD host=$(hostname) start=$(date +%H:%M:%S)"
python3 -u analysis/probe_element_matched.py \
  --dataset "$DS" --backbone "$BB" --fold "$FOLD" \
  --data_root "$DATA_ROOT" --vocab_root "$VOCAB" \
  --splits "$SPLITS" --min_n "$MIN_N" \
  --out_dir "$OUT"
echo "end=$(date +%H:%M:%S)"
