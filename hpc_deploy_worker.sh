#!/bin/bash
# hpc_deploy_worker.sh — generic per-shard HPC worker (reused/generalised from
# ablation_worker.sh). ONE SLURM array task = ONE shard. It runs a sharded python
# driver over its slice; the driver owns skip-done (never overwrites) + its own
# per-shard failure log, so re-submitting is safe and resumable.
#
# ALL parameters live HERE (env vars with defaults), NOT inside the python driver —
# so the driver stays generic and this file is the single place to configure a run.
# Reuse it for other sharded drivers by overriding SCRIPT + the roots.
#
# Env (the launcher exports these; defaults below are for the node-direct GT-ROC sweep):
#   REPO      repo root on HPC
#   CONDA_SH  path to conda profile.d/conda.sh   ENV_NAME  conda env
#   DEVICE    cpu | cuda
#   SCRIPT    python driver, relative to REPO
#   OUT       checkpoints tree (final_v2)        POSTHOC  saved-atts tree (posthoc_v1)
#   DEST      NEW output tree (path-aligned)     DATA     dataset FOLDS root
#   VOCAB     vocab root
#   SHARD     i/N for this task (REQUIRED; launcher sets it from SLURM_ARRAY_TASK_ID)
set -uo pipefail
REPO="${REPO:-/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor}"
CONDA_SH="${CONDA_SH:-/nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-l2xgnn}"
DEVICE="${DEVICE:-cpu}"
SCRIPT="${SCRIPT:-analysis/validate_posthoc_gtroc.py}"
OUT="${OUT:-$REPO/final_v2}"
POSTHOC="${POSTHOC:-$REPO/posthoc_v1}"
DEST="${DEST:-$REPO/gtroc_nodedirect_v1}"
DATA="${DATA:-/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/}"
VOCAB="${VOCAB:-$REPO/vocab_final_v2}"
: "${SHARD:?set SHARD=i/N (the launcher sets this from SLURM_ARRAY_TASK_ID)}"

source "$CONDA_SH"; conda activate "$ENV_NAME"
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled
[ "$DEVICE" = cpu ] && export CUDA_VISIBLE_DEVICES=""
echo "[worker] START $(date +%s) shard=$SHARD device=$DEVICE script=$SCRIPT host=$(hostname -s) job=${SLURM_JOB_ID:-$$}"
python3 -u "$SCRIPT" \
  --out_root "$OUT" --posthoc_root "$POSTHOC" --dest_root "$DEST" \
  --data_root "$DATA" --vocab_root "$VOCAB" --device "$DEVICE" --shard "$SHARD"
rc=$?
echo "[worker] END $(date +%s) shard=$SHARD rc=$rc"
exit $rc
