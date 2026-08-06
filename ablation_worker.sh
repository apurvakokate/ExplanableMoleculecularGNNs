#!/usr/bin/env bash
# ablation_worker.sh — claim-based PULL worker for the ablation. One SLURM job = one
# worker; launch many (via ablation_launch.sh) and they self-balance over the filesystem.
#
# DYNAMIC: add workers anytime (re-run the launcher); move datasets between the GPU/CPU
# pools freely — claims are CELL-keyed and shared across pools, so a running cell is never
# duplicated and only NOT-yet-started cells relocate.
#
# CONTRACT
#   DONE    = $DONE_FILE (default summary.json) exists in the cell's run dir. This is the
#             project's usual completion marker. NOTE: --per_split_eval writes
#             summary_splits.json AFTER summary.json; to make the done-marker also require
#             the per-split eval, set DONE_FILE=summary_splits.json.
#   CLAIM   = atomic `mkdir` of a cell-keyed dir under _dispatch/claims (prevents duplicate
#             concurrent runs; shared across GPU+CPU pools).
#   FAILURE = run.py rc!=0 (or missing DONE after rc==0) -> appended to
#             _dispatch/failures.tsv (exact cell) + a `.failed` marker in the claim dir.
#   NO AUTO-REDEPLOY: a claimed cell (running / failed / orphaned-by-preemption) is SKIPPED
#             by every worker. Re-kick is MANUAL (delete the claim dir for the cells you
#             want retried); nothing is auto-retried.
#
# ENV (set by the launcher): POOL_DATASETS (req), DEVICE=cuda|cpu, TIER=real|planted,
#   BACKBONES (opt), DONE_FILE (opt).
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
cd "$REPO"; export PYTHONPATH=$REPO WANDB_MODE=disabled
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
[ "${DEVICE:-cuda}" = cpu ] && export CUDA_VISIBLE_DEVICES=""

VOCAB=$REPO/vocab_final_v2; PROC=$REPO/processed_final_v2; OUT=$REPO/ablation_v2
BASE=$REPO/final_v2; GTCACHE=$BASE/gt_cache
FOLDS_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/; MUTAG_ROOT=$REPO/data
DISPATCH=$OUT/_dispatch; CLAIMS=$DISPATCH/claims; FAILURES=$DISPATCH/failures.tsv
mkdir -p "$CLAIMS"; [ -s "$FAILURES" ] || printf 'ts\thost_job\tcell_id\trc\n' > "$FAILURES"
WHO="$(hostname -s):${SLURM_JOB_ID:-$$}"
TIER="${TIER:-real}"; DONE_FILE="${DONE_FILE:-summary_splits.json}"
BACKBONES="${BACKBONES:-GIN GCN SAGE GAT PNA}"
: "${POOL_DATASETS:?set POOL_DATASETS to the space-separated datasets for this pool}"

_data_root(){ [ "$1" = mutag ] && echo "$MUTAG_ROOT" || echo "$FOLDS_ROOT"; }
_folds(){ [ "$1" = mutag ] && echo 0 || echo "0 1 2 3 4"; }
_mutag(){ [ "$1" = mutag ] && echo "--mutag_index_maps_path $MUTAG_ROOT/mutag_$2_index_maps.pkl --mutag_smiles_csv_path $MUTAG_ROOT/mutag_$2.csv --mutag_splits_path $MUTAG_ROOT/mutag_$2_splits.pkl --mutag_seed 42"; }
_inpool(){ case " $POOL_DATASETS " in *" $1 "*) return 0;; *) return 1;; esac; }

# run_cell CELLID FAMILY OUTVAR DATASET FOLD GLOB OROOT -- <run.py argv...>
run_cell(){
  local cellid=$1 family=$2 outvar=$3 ds=$4 f=$5 glob=$6 oroot=$7; shift 7; shift   # drop the '--'
  local parent="$oroot/$family/$outvar/$ds/fold$f"
  compgen -G "$parent/$glob/$DONE_FILE" >/dev/null 2>&1 && return 0          # DONE -> skip
  mkdir "$CLAIMS/$cellid" 2>/dev/null || return 0                           # CLAIM fails -> skip (no redeploy)
  echo "$WHO $(date +%s)" > "$CLAIMS/$cellid/info"
  echo "[run $DEVICE] $cellid"
  "$@"; local rc=$?
  if [ "$rc" -eq 0 ] && compgen -G "$parent/$glob/$DONE_FILE" >/dev/null 2>&1; then return 0; fi
  printf '%s\t%s\t%s\t%s\n' "$(date +%s)" "$WHO" "$cellid" "$rc" >> "$FAILURES"  # FAILURE tracked
  touch "$CLAIMS/$cellid/.failed"
  echo "[FAIL rc=$rc] $cellid"
}

mose_run(){   # setup norm unk enc pool regime   (ablation_v2: MoSE, source-GT/real only)
  local setup=$1 norm=$2 unk=$3 enc=$4 pool=$5 regime=$6 bb f ds dr variant G ptok oroot
  oroot="$REPO/ablation_v2/$regime"                       # normal/m1 in separate top-level trees
  local bal=(); local pat=()
  [ "$regime" = m1 ] && { bal=(BALANCED_SAMPLER=1); pat=(--patience 100); }   # m1 = class-balanced sampler
  # anchor DONE-glob so add (no _pool token) and mean (_pool-mean) never match each other
  ptok=$([ "$pool" = add ] && echo "wf" || echo "pool-${pool}")
  for ds in $POOL_DATASETS; do _inpool "$ds" || continue; dr=$(_data_root "$ds")
    for variant in rbrics_filter rdkit_fg_first_filter ertl_first_filter fg_first_filter; do
      for bb in $BACKBONES; do for f in $(_folds "$ds"); do
        G="${bb}_${enc}_norm-${norm}_${ptok}*unk-${unk}_*"
        run_cell "mose__${regime}__${variant}__${ds}__f${f}__${bb}__${setup}" mose "$variant" "$ds" "$f" "$G" "$oroot" -- \
          env "${bal[@]}" python3 MOSE-GNN/run.py --dataset "$ds" --fold "$f" --backbone "$bb" \
            --node_encoder "$enc" --conv_normalize "$norm" --unk_mode "$unk" --graph_pool "$pool" "${pat[@]}" \
            --w_feat --w_readout --epochs 500 --per_split_eval \
            --data_root "$dr" --vocab_root "$VOCAB" --vocab_variant "$variant" \
            --processed_root "$PROC" --out_dir "$oroot/mose/$variant" $(_mutag "$ds" "$f")
      done; done
    done
  done
}

ms_run(){     # setup norm enc  MODE(real|planted)
  local setup=$1 norm=$2 enc=$3 mode=$4 bb f ds dr variant G
  if [ "$mode" = real ]; then
    for ds in $POOL_DATASETS; do _inpool "$ds" || continue; dr=$(_data_root "$ds")
      for variant in rbrics rdkit_fg_first ertl_first fg_first; do
        for bb in $BACKBONES; do for f in $(_folds "$ds"); do
          G="${bb}_readout_${enc}_norm-${norm}_*"
          run_cell "motifsat__real__${variant}__${ds}__f${f}__${bb}__${setup}" motifsat "$variant" "$ds" "$f" "$G" -- \
            python3 MotifSAT/run.py --dataset "$ds" --fold "$f" --backbone "$bb" \
              --node_encoder "$enc" --conv_normalize "$norm" \
              --motif_method readout --noise motif --info_loss_level motif \
              --w_feat --w_message --w_readout --epochs 500 --per_split_eval \
              --data_root "$dr" --vocab_root "$VOCAB" --vocab_variant "$variant" \
              --processed_root "$PROC" --out_dir "$OUT/motifsat/$variant" $(_mutag "$ds" "$f")
        done; done
      done
    done
  else
    for gtdir in "$BASE"/motifsat/*_relabelled_dnf_*; do [ -d "$gtdir" ] || continue
      local gv=$(basename "$gtdir") bv tier; bv="${gv%%_relabelled_*}"; tier="${gv#*_relabelled_}"
      for dsdir in "$gtdir"/*/; do ds=$(basename "$dsdir"); case "$ds" in ogbg-*) continue;; esac
        _inpool "$ds" || continue; dr=$(_data_root "$ds")
        for bb in $BACKBONES; do for f in $(_folds "$ds"); do [ -d "$gtdir/$ds/fold$f" ] || continue
          G="${bb}_readout_${enc}_norm-${norm}_*"
          run_cell "motifsat__planted__${gv}__${ds}__f${f}__${bb}__${setup}" motifsat "$gv" "$ds" "$f" "$G" -- \
            python3 MotifSAT/run.py --dataset "$ds" --fold "$f" --backbone "$bb" \
              --node_encoder "$enc" --conv_normalize "$norm" \
              --motif_method readout --noise motif --info_loss_level motif \
              --w_feat --w_message --w_readout --epochs 500 --per_split_eval \
              --use_gt --gt_cache "$GTCACHE" --gt_tier "$tier" \
              --data_root "$dr" --vocab_root "$VOCAB" --vocab_variant "$bv" \
              --processed_root "$PROC" --out_dir "$OUT/motifsat/$gv" $(_mutag "$ds" "$f")
        done; done
      done
    done
  fi
}

echo "worker $WHO  DEVICE=${DEVICE:-cuda}  DONE_FILE=$DONE_FILE  POOL=[$POOL_DATASETS]"
# ablation_v2 — MoSE only, source-GT (real), 4 filtered vocabs. 7 (setup x regime) combos.
# base+normal is SKIPPED (already in antehoc_v1); base+m1 IS run.
#         setup    norm unk              enc    pool regime
mose_run base     none fixed            onehot add  m1
mose_run mean     none fixed            onehot mean normal
mose_run mean     none fixed            onehot mean m1
mose_run lunk     none learnable_shared onehot add  normal
mose_run lunk     none learnable_shared onehot add  m1
mose_run meanlunk none learnable_shared onehot mean normal
mose_run meanlunk none learnable_shared onehot mean m1
echo "worker $WHO DONE."
