#!/usr/bin/env bash
# One CELL = (dataset, variant, backbone, fold, tier, celltype).
#
# celltype:
#   vtrain    -> train Vanilla checkpoint only (skip if best_model.pt exists)  [GPU<->CPU]
#   posthoc   -> post-hoc explainers on the Vanilla checkpoint (--epochs 0)    [CPU only]
#   mose      -> train MOSE     (self-explaining, native explanation)          [GPU<->CPU]
#   gsat      -> train GSAT                                                     [GPU<->CPU]
#   motifsat  -> train MotifSAT                                                 [GPU<->CPU]
#
# tier: 'real' = original-labels regime; a 'dnf_*' key = relabelled/synthetic-GT regime.
# DEVICE is chosen by the WORKER via CUDA_VISIBLE_DEVICES (gpu worker pins a GPU; cpu
# worker hides it) — run_cell does NOT force a device, so the same cell runs on either.
# Reuses run_experiments.sh's exact phase invocations via env-scoping. Requires the run
# env exported (PROJECT, OUT_ROOT, VOCAB_ROOT, PROCESSED_ROOT, RULE_*, CELL_TIMING_LOG).
#   run_cell.sh <dataset> <variant> <backbone> <fold> <tier> <celltype>
set -uo pipefail
ds=$1; variant=$2; bb=$3; fold=$4; tier=$5; celltype=$6

export DATASETS="$ds" DATASETS_CSV="$ds" VOCAB_FOCUS="$variant" BACKBONES="$bb" FOLDS="$fold"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
       OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
cd "${PROJECT:?set PROJECT}"

# tier regime scoping ---------------------------------------------------------
#   real     -> original labels:  GT_ONLY=0, REAL_ONLY=1 (no gt), no GT_TIER_ONLY
#   source   -> _Verified_GT datasets (Benzene/Alkane_Carbonyl/Fluoride_Carbonyl): the model
#               is ALSO trained on ORIGINAL labels; GT-ROC vs the planted cause is scored from
#               data.node_label, which the loader bakes in from SOURCE_GT_SMARTS via
#               attach_source_gt (no --use_gt / gt_cache needed). So 'source' must use the SAME
#               routing as 'real' — the real-label phases. The *_gt phases skip these datasets
#               (not in GT_SUPPORTED_DATASETS) and validate_use_gt would reject --use_gt for them.
#   dnf_*    -> relabelled GT:     GT_ONLY=1, GT_TIER_ONLY=<tier> (that tier only)
if [ "$tier" = "real" ] || [ "$tier" = "source" ]; then
    unset GT_TIER_ONLY; export GT_ONLY=0 REAL_ONLY=1; _gt=""
else
    export GT_TIER_ONLY="$tier" GT_ONLY=1 REAL_ONLY=0; _gt="_gt"
fi

_t0=$(date +%s); _rc=0
case "$celltype" in
    vtrain)    bash run_experiments.sh "phase5_vanilla${_gt}"   || _rc=$? ;;
    posthoc)   bash run_experiments.sh "phase5_baselines${_gt}" || _rc=$? ;;
    mose)      bash run_experiments.sh phase5_mose              || _rc=$? ;;
    gsat)      bash run_experiments.sh phase5_gsat              || _rc=$? ;;
    motifsat)  bash run_experiments.sh phase5_motifsat          || _rc=$? ;;
    *) echo "run_cell: unknown celltype '$celltype'" >&2; _rc=2 ;;
esac
_t1=$(date +%s)

# timed row -> shared timing log (statistics/completeness harvest read this)
if [ -n "${CELL_TIMING_LOG:-}" ]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$ds" "$variant" "$bb" "$fold" "$tier" "$celltype" "$((_t1-_t0))" "$_rc" "$(hostname)" \
        >> "$CELL_TIMING_LOG"
fi
exit "$_rc"
