#!/usr/bin/env bash
# Emit one TAB-separated cell spec per line:  dataset<TAB>variant<TAB>backbone<TAB>fold<TAB>tier
# tiers = {real (unless GT_ONLY=1)} + the dnf keys in that variant's rule_tiers.json.
# Config from env: DATASETS, VARIANTS (or VOCAB_FOCUS), BACKBONES, FOLDS, VOCAB_ROOT, GT_ONLY.
set -uo pipefail
: "${VOCAB_ROOT:?set VOCAB_ROOT}"
DATASETS="${DATASETS:?set DATASETS}"
VARIANTS="${VARIANTS:-${VOCAB_FOCUS:-fg_first}}"; VARIANTS="${VARIANTS//,/ }"
BACKBONES="${BACKBONES:-GIN GCN SAGE GAT PNA}"
FOLDS="${FOLDS:-0 1 2 3 4}"

for ds in $DATASETS; do
  for variant in $VARIANTS; do
    tiers=""
    [ "${GT_ONLY:-0}" != "1" ] && tiers="real"
    rt="$VOCAB_ROOT/$ds/$variant/rule_tiers.json"
    if [ -f "$rt" ]; then
      tiers="$tiers $(python3 -c "import json,sys;print(' '.join(json.load(open('$rt')).keys()))" 2>/dev/null)"
    fi
    for bb in $BACKBONES; do for fold in $FOLDS; do for t in $tiers; do
      printf '%s\t%s\t%s\t%s\t%s\n' "$ds" "$variant" "$bb" "$fold" "$t"
    done; done; done
  done
done
