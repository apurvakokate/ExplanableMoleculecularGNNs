#!/bin/bash
#SBATCH -A eecs
#SBATCH -p share,gpu,dgx2
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 10:00:00
#SBATCH -J motune
#SBATCH -o fragmentation_tuning_experiment/logs/tune_%A_%a.out
#SBATCH --array=1-255%51
# ---------------------------------------------------------------------------
# MoSE OFAT sensitivity sweep — conservative_ertl_ring_mdl_filter, GAT.
# ALL FOLDS: 3 datasets x 5 folds x 17 configs (grid.tsv) = 255 tasks, up to 51
# in parallel. No DATASET env needed — everything is derived from the task id:
#   i    = SLURM_ARRAY_TASK_ID - 1          (0..254)
#   ds   = DATASETS[i / 85]                 (85 = 17 configs x 5 folds)
#   rem  = i % 85
#   fold = rem / 17                         (0..4)
#   cfg  = rem % 17 + 1                      (grid.tsv row)
# Ordering is dataset-outer, so Benzene's 5 folds finish first.
#
#   source parallel/final_ertlmdl_env.sh
#   sbatch fragmentation_tuning_experiment/tune_mose.sbatch
#
# Runs land in fragmentation_tuning_experiment/runs/ (NOT the production tree).
# Only HP knobs are overridden; everything else replicates the deployed run_mose
# call, so results are comparable to the deployed (fold-averaged) GAT run.
# ---------------------------------------------------------------------------
set -euo pipefail
BACKBONE="${BACKBONE:-GAT}"

# SLURM runs a SPOOLED COPY of this script, so BASH_SOURCE is not the repo path.
# The job starts in the submission dir (repo root, per README).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source parallel/final_ertlmdl_env.sh >/dev/null   # sets PROJECT + roots + conda + NODE_ENCODER + DATA_ROOT

HERE="$PROJECT/fragmentation_tuning_experiment"
GRID="$HERE/grid.tsv"
OUT_TUNE="$HERE/runs"
mkdir -p "$HERE/logs" "$OUT_TUNE"

DATASETS=(Benzene_Verified_GT Mutagenicity Lipophilicity)
NCFG=17; NFOLD=5; PERDS=$((NCFG * NFOLD))          # 85
i=$((SLURM_ARRAY_TASK_ID - 1))
DS_IDX=$((i / PERDS)); REM=$((i % PERDS)); FOLD=$((REM / NCFG)); CFG=$((REM % NCFG + 1))
DATASET="${DATASETS[$DS_IDX]}"

LINE=$(grep -vE '^[[:space:]]*#' "$GRID" | grep -vE '^[[:space:]]*$' | sed -n "${CFG}p")
read -r IDX KNOB ENT SIZE XLR GLR LAYERS HID PAT WD ES <<<"$LINE"
[ -z "${IDX:-}" ] && { echo "no grid row for cfg ${CFG} (task ${SLURM_ARRAY_TASK_ID})"; exit 1; }

echo "[$(date)] tune DATASET=$DATASET fold=$FOLD bb=$BACKBONE cfg=$IDX knob=$KNOB : ent=$ENT size=$SIZE xlr=$XLR glr=$GLR L=$LAYERS h=$HID pat=$PAT wd=$WD es=$ES (host $(hostname))"

python3 "$PROJECT/MOSE-GNN/run.py" \
    --dataset       "$DATASET" --fold "$FOLD" \
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
    --patience      "$PAT"    --weight_decay "$WD" \
    --early_stop_metric "$ES" \
    --out_dir       "$OUT_TUNE/conservative_ertl_ring_mdl_filter"

echo "[$(date)] tune DATASET=$DATASET fold=$FOLD cfg=$IDX ($KNOB) DONE"
