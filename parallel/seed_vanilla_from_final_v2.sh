#!/usr/bin/env bash
# seed_vanilla_from_final_v2.sh — RUN ON HPC. Symlinks the (vocab-independent) Vanilla base
# checkpoints from final_v2 into final_ertlmdl for the two new fragmentation variants, so
# phase5_vanilla with SKIP_EXISTING=1 reuses them instead of retraining.
#
# Vanilla path (verified): {ROOT}/vanilla/{ds}/fold{F}/{variant}/bb-{BB}_enc-onehot_norm-none_{tier}/
# The Vanilla GNN uses atom features only (no vocab) -> the checkpoint is identical across
# fragmentation variants; we symlink an existing variant's leaf dir to the two new variants.
#
# Usage (on HPC):  bash parallel/seed_vanilla_from_final_v2.sh
set -uo pipefail
SRC_ROOT="${REUSE_VANILLA_FROM:-$PROJECT/final_v2}"
DST_ROOT="${OUT_ROOT:-$PROJECT/final_ertlmdl}"
DATASETS=(Mutagenicity BBBP hERG esol Lipophilicity
          Benzene_Verified_GT Alkane_Carbonyl_Verified_GT Fluoride_Carbonyl_Verified_GT)
BACKBONES=(GIN GCN SAGE GAT PNA)
FOLDS=(0 1 2 3 4)
TIER="real"                                  # planted deferred -> all vanilla trained on original labels
# all 4 fragmentation variants (vanilla is vocab-independent -> same checkpoint symlinked to each)
NEW_VARIANTS=(conservative_ertl_ring_mdl conservative_ertl_ring_bpe \
              conservative_ertl_ring_mdl_filter conservative_ertl_ring_bpe_filter)

n_link=0 n_miss=0
for ds in "${DATASETS[@]}"; do
  for f in "${FOLDS[@]}"; do
    for bb in "${BACKBONES[@]}"; do
      leaf="bb-${bb}_enc-onehot_norm-none_${TIER}"
      # find ANY existing source variant's vanilla leaf for this (ds,fold,bb,tier)
      src=""
      for cand in "$SRC_ROOT/vanilla/$ds/fold$f"/*/"$leaf"; do
        [ -f "$cand/best_model.pt" ] && { src="$cand"; break; }
      done
      if [ -z "$src" ]; then
        echo "[seed] MISS  $ds/fold$f/$bb ($TIER) — no source vanilla in final_v2" >&2
        n_miss=$((n_miss+1)); continue
      fi
      for v in "${NEW_VARIANTS[@]}"; do
        dst="$DST_ROOT/vanilla/$ds/fold$f/$v/$leaf"
        if [ -e "$dst/best_model.pt" ]; then continue; fi
        mkdir -p "$(dirname "$dst")"
        ln -sfn "$src" "$dst"                 # symlink the whole leaf dir (best_model.pt + hparams + summary)
        n_link=$((n_link+1))
      done
    done
  done
done
echo "[seed] linked=$n_link  missing_source=$n_miss  (src=$SRC_ROOT -> dst=$DST_ROOT/vanilla)"
