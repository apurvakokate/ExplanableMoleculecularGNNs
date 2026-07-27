#!/usr/bin/env bash
# One independent CELL = (dataset, variant, backbone, fold, tier).
# Trains the vanilla model then runs the post-hoc explainers for THIS cell only,
# reusing run_experiments.sh's exact (correct) invocation via env-scoping — no flags
# are re-derived here. `real` tier => real-label phases; a gt/dnf tier => gt phases
# scoped by GT_TIER_ONLY. Threads are pinned to 1 so N cells on one node run
# one-core-each with no oversubscription. FORCE_CPU=1 hides the GPU so get_device()
# falls back to CPU and a cell is a pure 1-core CPU unit (no GPU contention).
#
# Requires the run env already exported (PROJECT, OUT_ROOT, VOCAB_ROOT, PROCESSED_ROOT,
# RULE_ENGINE, RULE_TIERS, ... — i.e. final_v1_env.sh sourced by the caller). Usage:
#   run_cell.sh <dataset> <variant> <backbone> <fold> <tier>
set -uo pipefail
ds=$1; variant=$2; bb=$3; fold=$4; tier=$5

export DATASETS="$ds" DATASETS_CSV="$ds" VOCAB_FOCUS="$variant" BACKBONES="$bb" FOLDS="$fold"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
       OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
# CPU-first: hide the GPU so each cell is a pure 1-core CPU worker (no GPU contention).
[ "${FORCE_CPU:-1}" = "1" ] && export CUDA_VISIBLE_DEVICES=""
cd "${PROJECT:?set PROJECT}"

_t0=$(date +%s)
_rc=0
if [ "$tier" = "real" ]; then
    unset GT_TIER_ONLY
    { bash run_experiments.sh phase5_vanilla && bash run_experiments.sh phase5_baselines; } || _rc=$?
else
    export GT_TIER_ONLY="$tier"
    { bash run_experiments.sh phase5_vanilla_gt && bash run_experiments.sh phase5_baselines_gt; } || _rc=$?
fi
_t1=$(date +%s)

# Timed row: one line per cell → shared timing log (statistics harvest reads this).
# Small O_APPEND writes are atomic, so concurrent workers can share one file.
if [ -n "${CELL_TIMING_LOG:-}" ]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$ds" "$variant" "$bb" "$fold" "$tier" "$((_t1 - _t0))" "$_rc" "$(hostname)" \
        >> "$CELL_TIMING_LOG"
fi
exit "$_rc"
