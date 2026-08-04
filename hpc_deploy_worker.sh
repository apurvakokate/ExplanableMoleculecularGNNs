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
# SHARD: use an explicit SHARD=i/N (interactive), else derive it from the SLURM array
# task index + SHARD_N (the launcher submits THIS file as the batch script with those set).
if [ -z "${SHARD:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  SHARD="${SLURM_ARRAY_TASK_ID}/${SHARD_N:?set SHARD_N when submitting as a SLURM array}"
fi
: "${SHARD:?set SHARD=i/N, or submit as a SLURM array with SHARD_N set}"

source "$CONDA_SH"; conda activate "$ENV_NAME"
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled
[ "$DEVICE" = cpu ] && export CUDA_VISIBLE_DEVICES=""
# optional: restrict to specific dataset(s) via DATASET env (e.g. DATASET=mutag). MUTAG_DATA_ROOT,
# if exported, is inherited by the driver's env for the mutag loader.
DSARG=(); [ -n "${DATASET:-}" ] && DSARG=(--dataset $DATASET)
FLTARG=(); [ -n "${FILTERED:-}" ] && FLTARG=(--filtered)   # FILTERED=1 -> all-filtered vocab pass
UNKARG=(); [ -n "${UNK_MODE:-}" ] && UNKARG=(--unk_mode "$UNK_MODE")   # zero|half
echo "[worker] START $(date +%s) shard=$SHARD device=$DEVICE script=$SCRIPT dataset=${DATASET:-all} filtered=${FILTERED:-0} unk=${UNK_MODE:-zero} host=$(hostname -s) job=${SLURM_JOB_ID:-$$}"
python3 -u "$SCRIPT" \
  --out_root "$OUT" --posthoc_root "$POSTHOC" --dest_root "$DEST" \
  --data_root "$DATA" --vocab_root "$VOCAB" --device "$DEVICE" --shard "$SHARD" "${DSARG[@]}" "${FLTARG[@]}" "${UNKARG[@]}"
rc=$?
echo "[worker] END $(date +%s) shard=$SHARD rc=$rc"
exit $rc
