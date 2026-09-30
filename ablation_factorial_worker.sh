#!/usr/bin/env bash
# ablation_factorial_worker.sh — fill the FULL 3x2x2 MoSE architecture factorial for
# rbrics_filter, REAL tier only. DRAFT — review before running. Launch via
# ablation_factorial_launch.sh (claim-based pull: many workers self-balance, never dup a cell).
#
# GRID (per base cell = dataset x fold x backbone):
#   node_encoder  in {onehot, linear}
#   conv_normalize in {none, l2, layernorm}
#   graph_pool    in {add, mean}
# = 12 combos. Base universe = rbrics_filter REAL: 8 datasets x 5 folds x 5 backbones = 200
# (mutag EXCLUDED). Target = 2,400; ~800 already exist (onehot/none/add, onehot/none/mean,
# onehot/l2/add, linear/none/add) and are skipped; ~1,600 to run.
#
# NO-OVERWRITE — three independent guards:
#   1. variant_tag encodes enc, norm and pool -> every combo lands in a DISTINCT dir.
#   2. cross-tree DONE-check: skip if $DONE_FILE exists for this exact tag in ANY of
#      final_v2 / ablation_v2/{normal,m1} / ablated_completely_v1 (auto-skips existing arms).
#   3. atomic mkdir CLAIM so two workers never take the same cell.
# The layernorm arm uses --conv_normalize layernorm and NEVER --apply_layer_norm (those two
# collapse to the same 'norm-layernorm' tag but are different ops; we want per-conv layernorm).
#
# ENV: POOL_DATASETS (req; space-sep datasets this pool handles — must NOT contain mutag),
#      DEVICE=cuda|cpu, DONE_FILE (opt, default summary_splits.json),
#      DRY_RUN=1 (print run.py commands + counts, run nothing), OUT_ROOT (opt, default in-place).
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
cd "$REPO"; export PYTHONPATH=$REPO WANDB_MODE=disabled
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
[ "${DEVICE:-cuda}" = cpu ] && export CUDA_VISIBLE_DEVICES=""

VOCAB=$REPO/vocab_final_v2; PROC=$REPO/processed_final_v2
BASE=$REPO/ablated_completely_v1
OUT_ROOT="${OUT_ROOT:-$BASE}"                 # new runs land here (in-place next to none/l2)
FOLDS_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/
DISPATCH=$OUT_ROOT/_dispatch_fac; CLAIMS=$DISPATCH/claims; FAILURES=$DISPATCH/failures.tsv
mkdir -p "$CLAIMS"; [ -s "$FAILURES" ] || printf 'ts\thost_job\tcell_id\trc\n' > "$FAILURES"
WHO="$(hostname -s):${SLURM_JOB_ID:-$$}"; DONE_FILE="${DONE_FILE:-summary_splits.json}"
: "${POOL_DATASETS:?set POOL_DATASETS to the space-separated datasets for this pool}"
DRY_RUN="${DRY_RUN:-0}"

VOCABV=rbrics_filter                          # REAL rbrics only
BACKBONES=(GIN GCN GAT PNA SAGE)              # GAT is heads=1 (project default)
FOLDS=(0 1 2 3 4)
ENCS=(onehot linear); NORMS=(none l2 layernorm); POOLS=(add mean)
# trees scanned by the cross-tree DONE-check (do NOT run into these; only OUT_ROOT is written)
SCAN_TREES=("$REPO/final_v2/mose/$VOCABV" \
            "$REPO/ablation_v2/normal/mose/$VOCABV" \
            "$REPO/ablation_v2/m1/mose/$VOCABV" \
            "$REPO/ablated_completely_v1/mose/$VOCABV")

_poolsfx(){ [ "$1" = mean ] && echo "_pool-mean" || echo ""; }
# tag leaf glob for one combo (hp suffix, if any, matched by trailing *)
_tag(){ local bb=$1 enc=$2 norm=$3 pool=$4
  echo "${bb}_${enc}_norm-${norm}$(_poolsfx "$pool")_wf+wr_unk-fixed_real_ep500_${VOCABV}*"; }
# exists anywhere across SCAN_TREES (+ OUT_ROOT) for this ds/fold?
_exists(){ local ds=$1 fold=$2 tag=$3 t
  for t in "${SCAN_TREES[@]}" "$OUT_ROOT/mose/$VOCABV"; do
    compgen -G "$t/$ds/fold$fold/$tag/$DONE_FILE" >/dev/null 2>&1 && return 0
  done; return 1; }

echo "worker $WHO DEVICE=${DEVICE:-cuda} DRY_RUN=$DRY_RUN OUT_ROOT=$OUT_ROOT POOL=[$POOL_DATASETS]"
n_target=0 n_run=0 n_skip=0
for ds in $POOL_DATASETS; do
  [ "$ds" = mutag ] && { echo "REFUSE mutag (excluded)"; continue; }
  case " $POOL_DATASETS " in *" $ds "*) ;; *) continue;; esac
  for fold in "${FOLDS[@]}"; do
    for bb in "${BACKBONES[@]}"; do
      for enc in "${ENCS[@]}"; do for norm in "${NORMS[@]}"; do for pool in "${POOLS[@]}"; do
        n_target=$((n_target+1))
        tag=$(_tag "$bb" "$enc" "$norm" "$pool")
        _exists "$ds" "$fold" "$tag" && { n_skip=$((n_skip+1)); continue; }
        local_out="$OUT_ROOT/mose/$VOCABV"
        cellid="mosefac__${ds}__f${fold}__${bb}__${enc}__${norm}__${pool}"
        cmd=(python3 MOSE-GNN/run.py --dataset "$ds" --fold "$fold" --backbone "$bb"
             --node_encoder "$enc" --conv_normalize "$norm" --graph_pool "$pool"
             --w_feat --w_readout --unk_mode fixed --epochs 500 --per_split_eval
             --data_root "$FOLDS_ROOT" --vocab_root "$VOCAB" --vocab_variant "$VOCABV"
             --processed_root "$PROC" --out_dir "$local_out")
        if [ "$DRY_RUN" = 1 ]; then echo "[dry] $cellid :: ${cmd[*]}"; n_run=$((n_run+1)); continue; fi
        mkdir "$CLAIMS/$cellid" 2>/dev/null || { n_skip=$((n_skip+1)); continue; }   # CLAIM fails -> another worker took it
        echo "$WHO $(date +%s)" > "$CLAIMS/$cellid/info"; echo "[run ${DEVICE:-cuda}] $cellid"; n_run=$((n_run+1))
        "${cmd[@]}"; rc=$?
        if [ "$rc" -eq 0 ] && compgen -G "$local_out/$ds/fold$fold/$tag/$DONE_FILE" >/dev/null 2>&1; then continue; fi
        printf '%s\t%s\t%s\t%s\n' "$(date +%s)" "$WHO" "$cellid" "$rc" >> "$FAILURES"
        touch "$CLAIMS/$cellid/.failed"; echo "[FAIL rc=$rc] $cellid"
      done; done; done
    done
  done
done
echo "worker $WHO DONE. target=$n_target run=$n_run skip=$n_skip"
