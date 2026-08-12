#!/bin/bash
#SBATCH -A eecs
#SBATCH -p share,gpu,dgx2
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 10:00:00
#SBATCH -J motune
#SBATCH -o fragmentation_tuning_experiment/logs/tune_%A_%a.out
#SBATCH --array=1-17
# ^ no %throttle: SLURM runs as many of the 17 as there are free GPUs. Submit all
#   three datasets (see README) and idle cores backfill the later datasets; the
#   earliest-submitted (Benzene) keeps scheduling priority. To cap concurrency
#   instead, override at submit: sbatch --array=1-17%N ...
# ---------------------------------------------------------------------------
# MoSE OFAT sensitivity sweep for the conservative_ertl_ring_mdl_filter vocab.
# ONE backbone (GAT), one dataset per submission, fold 0. One array task per
# grid.tsv row (17). Row 1 = fixed center; each other row moves ONE of the
# three explainer knobs (ent_reg, size_reg, explainer_lr), both sides.
# DATASET is passed via env:
#
#   source parallel/final_ertlmdl_env.sh
#   DATASET=Mutagenicity sbatch fragmentation_tuning_experiment/tune_mose.sbatch
#   (also Benzene_Verified_GT, Lipophilicity)
#
# Runs land in fragmentation_tuning_experiment/runs/ (NOT the production tree).
# Only HP knobs are overridden; everything else replicates the deployed
# run_mose() call, so results are comparable to the deployed GAT run.
# ---------------------------------------------------------------------------
set -euo pipefail
: "${DATASET:?set DATASET=Mutagenicity|Benzene_Verified_GT|Lipophilicity}"
BACKBONE="${BACKBONE:-GAT}"

# SLURM executes a SPOOLED COPY of this script, so $0 / BASH_SOURCE is NOT the repo
# path. The job's initial cwd is the submission dir (repo root, per README), so anchor
# on SLURM_SUBMIT_DIR; then $PROJECT (set by the env script) locates everything.
# (stderr is NOT suppressed, so any env/source error lands in the task log.)
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"               # -> PROJECT (repo root)
source parallel/final_ertlmdl_env.sh >/dev/null   # sets PROJECT + roots + conda + NODE_ENCODER + DATA_ROOT

HERE="$PROJECT/fragmentation_tuning_experiment"
GRID="$HERE/grid.tsv"
OUT_TUNE="$HERE/runs"
mkdir -p "$HERE/logs" "$OUT_TUNE"

# SLURM_ARRAY_TASK_ID-th non-comment, non-blank grid row.
LINE=$(grep -vE '^[[:space:]]*#' "$GRID" | grep -vE '^[[:space:]]*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")
read -r IDX KNOB ENT SIZE XLR GLR LAYERS HID PAT WD ES <<<"$LINE"
[ -z "${IDX:-}" ] && { echo "no grid row for task ${SLURM_ARRAY_TASK_ID}"; exit 1; }

echo "[$(date)] tune DATASET=$DATASET bb=$BACKBONE case=$IDX knob=$KNOB : ent=$ENT size=$SIZE xlr=$XLR glr=$GLR L=$LAYERS h=$HID pat=$PAT wd=$WD es=$ES (host $(hostname))"

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
    --patience      "$PAT"    --weight_decay "$WD" \
    --early_stop_metric "$ES" \
    --out_dir       "$OUT_TUNE/conservative_ertl_ring_mdl_filter"

echo "[$(date)] tune DATASET=$DATASET bb=$BACKBONE case=$IDX ($KNOB) DONE"
