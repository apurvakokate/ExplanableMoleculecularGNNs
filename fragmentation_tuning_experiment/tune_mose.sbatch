#!/bin/bash
#SBATCH -A eecs
#SBATCH -p share,gpu,dgx2
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 10:00:00
#SBATCH -J motune
#SBATCH -o fragmentation_tuning_experiment/logs/tune_%A_%a.out
#SBATCH --array=1-80%20
# ---------------------------------------------------------------------------
# MoSE hyperparameter tuning for the conservative_ertl_ring_mdl_filter vocab.
# Per-dataset backbone targeting (targets.tsv): only backbones where the deployed
# ERTL run loses to rbrics on pearson_motif. 16 HP cases (grid.tsv) each, fold 0.
# Task id -> (backbone, grid-case):  backbone = TARGETS[DATASET][(id-1)/16],
#                                     case     = (id-1) % 16 + 1.
# Always submit --array=1-80%20; tasks past this dataset's backbone count no-op.
# DATASET is passed via env:
#
#   source parallel/final_ertlmdl_env.sh
#   DATASET=esol sbatch fragmentation_tuning_experiment/tune_mose.sbatch
#
# Runs land in fragmentation_tuning_experiment/runs/ (NOT the production tree).
# Only HP knobs are overridden on the CLI; everything else replicates the
# deployed run_mose() invocation, so results are comparable to the deployed
# final_ertlmdl run for the same (backbone, dataset).
# ---------------------------------------------------------------------------
set -euo pipefail
: "${DATASET:?set DATASET=esol|Lipophilicity|hERG|Mutagenicity}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."                                  # -> PROJECT (repo root)
source parallel/final_ertlmdl_env.sh >/dev/null 2>&1   # roots + conda + NODE_ENCODER + DATA_ROOT

GRID="$HERE/grid.tsv"
OUT_TUNE="$HERE/runs"
mkdir -p "$HERE/logs" "$OUT_TUNE"

# Backbones to tune for THIS dataset (data-driven; see targets.tsv).
BBLINE=$(grep -vE '^[[:space:]]*#' "$HERE/targets.tsv" | awk -v d="$DATASET" '$1==d {print $2}')
[ -z "$BBLINE" ] && { echo "no target backbones for DATASET=$DATASET in targets.tsv"; exit 1; }
IFS=',' read -ra BACKBONES <<< "$BBLINE"
NBB=${#BACKBONES[@]}
NCASE=16
BB_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) / NCASE ))
CASE_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) % NCASE + 1 ))
if [ "$BB_IDX" -ge "$NBB" ]; then
    echo "task ${SLURM_ARRAY_TASK_ID}: $DATASET has only $NBB target backbone(s) (${BACKBONES[*]}) — no-op"; exit 0
fi
BACKBONE="${BACKBONES[$BB_IDX]}"

# CASE_IDX-th non-comment, non-blank row of the grid.
LINE=$(grep -vE '^[[:space:]]*#' "$GRID" | grep -vE '^[[:space:]]*$' | sed -n "${CASE_IDX}p")
read -r IDX LABEL ENT SIZE XLR LAYERS HID GLR PAT <<<"$LINE"
[ -z "${IDX:-}" ] && { echo "no grid row for case ${CASE_IDX} (task ${SLURM_ARRAY_TASK_ID})"; exit 1; }

echo "[$(date)] tune DATASET=$DATASET backbone=$BACKBONE case=$IDX ($LABEL): ent=$ENT size=$SIZE xlr=$XLR L=$LAYERS h=$HID glr=$GLR pat=$PAT (host $(hostname))"

python3 "$PROJECT/MOSE-GNN/run.py" \
    --dataset       "$DATASET" --fold 0 \
    --backbone      "$BACKBONE" --node_encoder "$NODE_ENCODER" \
    --w_feat --w_readout \
    --epochs        "$EPOCHS" \
    --data_root     "$DATA_ROOT" \
    --vocab_root    "$VOCAB_ROOT" \
    --vocab_variant conservative_ertl_ring_mdl_filter \
    --conv_normalize none --graph_pool add --unk_mode fixed \
    --processed_root "$PROCESSED_ROOT" \
    --ent_reg       "$ENT"    --size_reg    "$SIZE" \
    --explainer_lr  "$XLR"    --gnn_lr      "$GLR" \
    --num_layers    "$LAYERS" --hidden_dim  "$HID" \
    --patience      "$PAT" \
    --out_dir       "$OUT_TUNE/conservative_ertl_ring_mdl_filter"

echo "[$(date)] tune DATASET=$DATASET backbone=$BACKBONE case=$IDX ($LABEL) DONE"
