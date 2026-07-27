#!/usr/bin/env bash
# stage0_prereqs.sh — from-scratch Stage 0: build processed graphs + vocab
# (rbrics, fg_first, + *_filter) + phase-4 DNF GT (BBBP, Mutagenicity; esol is
# regression → no GT) into the final_v1 roots. Run ONCE before queue_dispatch.sh.
# Runs inside the biggest holder for compute (srun --overlap); local fallback.
#
#   Smoke test one dataset first:  DATASETS=BBBP bash parallel/stage0_prereqs.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/final_v1_env.sh"
# allow a caller override (e.g. smoke-test one dataset) to win over the env default
[ -n "${DATASETS_OVERRIDE:-}" ] && export DATASETS="$DATASETS_OVERRIDE" DATASETS_CSV="$DATASETS_OVERRIDE"

echo "[stage0] DATASETS=$DATASETS"
echo "[stage0] VOCAB_ROOT=$VOCAB_ROOT  PROCESSED_ROOT=$PROCESSED_ROOT  OUT_ROOT=$OUT_ROOT"

# phase0_4 = phase0 → phase4 for all DATASETS (vocab build → thresholding → DNF GT).
holder="${HOLDERS%% *}"; jobid="${holder%%:*}"
if squeue -j "$jobid" -h -o "%T" 2>/dev/null | grep -q RUNNING; then
    echo "[stage0] running phase0_4 inside holder $jobid"
    srun --overlap --jobid="$jobid" --ntasks=1 bash -c \
        "source '$HERE/final_v1_env.sh'; [ -n '${DATASETS_OVERRIDE:-}' ] && export DATASETS='${DATASETS_OVERRIDE:-}' DATASETS_CSV='${DATASETS_OVERRIDE:-}'; cd '$PROJECT'; bash run_experiments.sh phase0_4"
else
    echo "[stage0] no RUNNING holder $jobid — running phase0_4 locally"
    cd "$PROJECT"; bash run_experiments.sh phase0_4
fi

echo "[stage0] built vocab variants:"
for d in "$VOCAB_ROOT"/*/; do [ -d "$d" ] && printf '  %-28s %s\n' "$(basename "$d")" "$(ls "$d" 2>/dev/null | tr '\n' ' ')"; done
echo "[stage0] rule_tiers (GT tiers) present:"
find "$VOCAB_ROOT" -name rule_tiers.json 2>/dev/null | sed 's/^/  /' | head
echo "[stage0] gt_cache present: $([ -d "$OUT_ROOT/gt_cache" ] && echo yes || echo NO)"
