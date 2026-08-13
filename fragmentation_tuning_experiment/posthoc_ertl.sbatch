#!/bin/bash
#SBATCH -A eecs
#SBATCH -p share,gpu,dgx2,preempt
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 5:00:00
#SBATCH -J posthoc_ertl
#SBATCH -o fragmentation_tuning_experiment/logs/posthoc_%A_%a.out
#SBATCH --array=1-4
#SBATCH --exclude=cn-gpu3
# ---------------------------------------------------------------------------
# Post-hoc MAGE (mage_v2) + motif-occlusion with the ERTL+MDL vocab
# (conservative_ertl_ring_mdl_filter), scored on the ALREADY-TRAINED vanilla GAT
# models — no retraining, so the model keeps full predictive performance.
# One array task per GT dataset (GAT, 5 folds). preempt included for capacity.
#   Phase A: scaffold baseline eval dirs (run_vanilla --epochs 0 loads the vanilla
#            weights under the ERTL eval vocab).
#   Phase B: score mage_v2 + motif_occlusion via eval_driver_posthoc.
#
#   source parallel/final_ertlmdl_env.sh
#   sbatch fragmentation_tuning_experiment/posthoc_ertl.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source parallel/final_ertlmdl_env.sh >/dev/null

DATASETS=(Benzene_Verified_GT Alkane_Carbonyl_Verified_GT Fluoride_Carbonyl_Verified_GT Mutagenicity)
DS="${DATASETS[$((SLURM_ARRAY_TASK_ID - 1))]}"
mkdir -p fragmentation_tuning_experiment/logs

echo "[$(date)] posthoc DS=$DS bb=GAT (host $(hostname))"

# --- Phase A: scaffold baseline eval dirs from the trained vanilla GAT weights (epochs=0) ---
VOCAB_FOCUS=conservative_ertl_ring_mdl DATASETS="$DS" BACKBONES=GAT FOLDS="0 1 2 3 4" \
    SKIP_EXISTING=1 bash run_experiments.sh phase5_baselines

# Keep ONLY the 88-motif filtered vocab; drop the 2936-motif unfiltered scaffold so it isn't
# scored (exact dir name — does NOT match conservative_ertl_ring_mdl_filter).
for f in 0 1 2 3 4; do
    rm -rf "$OUT_ROOT/baselines/$DS/fold$f/conservative_ertl_ring_mdl"
done

# --- Phase B: score mage_v2 + motif_occlusion on GPU (skips cells already scored) ---
python3 analysis/eval_driver_posthoc.py \
    --out_root       "$OUT_ROOT/baselines" \
    --dest_root      "$OUT_ROOT/posthoc_ertl" \
    --data_root      "$DATA_ROOT" \
    --vocab_root     "$VOCAB_ROOT" \
    --processed_root "$PROCESSED_ROOT" \
    --device cuda \
    --dataset "$DS" \
    --methods motif_occlusion mage_v2 \
    --skip_done

echo "[$(date)] posthoc DS=$DS DONE"
