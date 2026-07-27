#!/usr/bin/env bash
# One independent CELL = (dataset, variant, backbone, fold, tier).
# Trains the vanilla model then runs the post-hoc explainers for THIS cell only,
# reusing run_experiments.sh's exact (correct) invocation via env-scoping — no flags
# are re-derived here. `real` tier => real-label phases; a gt/dnf tier => gt phases
# scoped by GT_TIER_ONLY. Threads are pinned so N cells on one node don't oversubscribe.
#
# Requires the run env already exported (PROJECT, OUT_ROOT, VOCAB_ROOT, PROCESSED_ROOT,
# RULE_ENGINE, RULE_TIERS, etc. — i.e. the _setup block). Usage:
#   run_cell.sh <dataset> <variant> <backbone> <fold> <tier>
set -uo pipefail
ds=$1; variant=$2; bb=$3; fold=$4; tier=$5

export DATASETS="$ds" DATASETS_CSV="$ds" VOCAB_FOCUS="$variant" BACKBONES="$bb" FOLDS="$fold"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
       OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
cd "${PROJECT:?set PROJECT}"

if [ "$tier" = "real" ]; then
    unset GT_TIER_ONLY
    bash run_experiments.sh phase5_vanilla && bash run_experiments.sh phase5_baselines
else
    export GT_TIER_ONLY="$tier"
    bash run_experiments.sh phase5_vanilla_gt && bash run_experiments.sh phase5_baselines_gt
fi
