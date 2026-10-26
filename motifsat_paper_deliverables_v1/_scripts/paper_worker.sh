#!/usr/bin/env bash
# paper_worker.sh — claim-based PULL worker for MotifSAT on the GSAT paper datasets (EXPERIMENT_LOG §7)
# (fork of variants_worker.sh: same DONE / CLAIM / FAILURE contract and training recipe — OUR model,
# 5 backbones; only the datasets, replicates, r table and preset set differ).
# Submitted through sbatch stdin by paper_launch.sh (no file copy to the HPC); this file is the record.
#
# One CELL = (vocab, preset, grouping, dataset, replicate, backbone). Run dir:
#     $OUT/paper_v1/<vocab>/<preset>/<grouping>/<dataset>/<rep>/<backbone>/
#   dataset  mutag (data_root $REPO/data, our disjoint folds = mutag_<k>_splits.pkl) |
#            ogbg-molhiv / molbace / molbbbp / moltox21 / molclintox (GSAT_upstream data, official split)
#   rep      mutag: fold<k> (--fold k, seed 42 as in the 8-dataset campaign);
#            OGB:   seed<k> (--fold 0 --seed k; one official split, so replicates are seeds)
#   vocab    rbrics_v3 = vocab_final_v3/<ds>/rbrics (fold-0 vocabulary)
#            fmbpe_v4  = fmbpe_v4/vocab/fold<f>/<ds>/fm_bpe_full_union (mutag per fold; OGB fold 0)
#   preset -> grouping (best per family, §4):  base_gsat, base_gsat_noanneal -> none;
#            intra_inter_loss, readout_layernorm, readout_nonorm -> pooled;
#            motif_emb_iddesc_chem_res, motif_emb_multihot_chem_res -> copy.
#   base_gsat_noanneal runs on rbrics_v3 only (as in the 8-dataset campaign). base_gsat trains on
#   rbrics_v3; its fmbpe_v4 cell RE-EVALUATES that checkpoint (--eval_only; training is motif-free),
#   and is skipped (left unclaimed) until the rbrics_v3 base run is complete.
# Every run: beta = info_loss_coef 1, ckpt_rule gsat, r from R_TABLE (GSAT's schedule, r_table_paper.sh;
# init_r 0.9, decay_r 0.1), --per_split_eval --occurrence_eval, 500 epochs, patience 50,
# clip_grad = $CLIP if set (else run.py default 2.0).
# DONE (trained) = native_complete.py passes AND history.json AND occurrence_summary.json non-empty.
# DONE (eval-only base) = summary.json + summary_splits.json + occurrence_summary.json non-empty.
#
# ENV: POOL_DATASETS (req), DEVICE=cuda|cpu, optional VOCABS PRESETS BACKBONES REPS EPOCHS PATIENCE
#      CLIP SMOKE PAR.
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
SCRIPTS="$REPO/motifsat_paper_deliverables_v1/_scripts"
OGB=/nfs/hpc/share/kokatea/ChemIntuit/GSAT_upstream_6812f5c/data
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled OGB_DATA_ROOT="$OGB"
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
[ "${DEVICE:-cuda}" = cpu ] && export CUDA_VISIBLE_DEVICES=""

OUT="$REPO/motifsat_paper_deliverables_v1"; RUNS="$OUT/paper_v1"
PROC_BASE="$REPO/processed_paper_v1"      # passed for completeness; mutag/OGB loaders annotate on read
DISPATCH="$OUT/_dispatch_paper"; CLAIMS="$DISPATCH/claims"; FAILURES="$DISPATCH/failures.tsv"
# SMOKE (2-epoch) runs go to a separate tree + claim dir so they can never count as DONE.
if [ "${SMOKE:-0}" = 1 ]; then RUNS="$OUT/paper_v1_smoke"; CLAIMS="$DISPATCH/claims_smoke"; fi
mkdir -p "$CLAIMS"
[ -s "$FAILURES" ] || printf 'ts\thost\tjobid\tdevice\tcell_id\trc\n' > "$FAILURES"
WHO="$(hostname -s):${SLURM_JOB_ID:-$$}"; JOBID="${SLURM_JOB_ID:-$$}"

EPOCHS="${EPOCHS:-500}"; PATIENCE="${PATIENCE:-50}"
BACKBONES="${BACKBONES:-GIN GCN GAT SAGE PNA}"
REPS="${REPS:-0 1 2 3 4}"
VOCABS="${VOCABS:-rbrics_v3}"
PRESETS="${PRESETS:-base_gsat base_gsat_noanneal intra_inter_loss readout_layernorm readout_nonorm motif_emb_iddesc_chem_res motif_emb_multihot_chem_res}"
: "${POOL_DATASETS:?set POOL_DATASETS to the datasets for this pool}"
# lists may arrive ':'-joined from paper_launch.sh
POOL_DATASETS="${POOL_DATASETS//:/ }"; VOCABS="${VOCABS//:/ }"; PRESETS="${PRESETS//:/ }"; BACKBONES="${BACKBONES//:/ }"; REPS="${REPS//:/ }"
if [ "${SMOKE:-0}" = 1 ]; then REPS="0"; BACKBONES="GIN"; EPOCHS=2; fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
       OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# r schedule per (dataset, backbone): "<final_r> <decay_interval>" (GSAT's values, r_table_paper.sh).
R_TABLE_FILLED_AT_LAUNCH

grouping_of(){
    case "$1" in
        base_gsat|base_gsat_noanneal) echo none ;;
        intra_inter_loss|readout_layernorm|readout_nonorm) echo pooled ;;
        motif_emb_*) echo copy ;;
        *) return 1 ;;
    esac
}

_complete(){
    python3 "$SCRIPTS/native_complete.py" "$1" "$2" --quiet >/dev/null 2>&1 \
        && [ -s "$1/history.json" ] && [ -s "$1/occurrence_summary.json" ]
}
_evaluated(){ [ -s "$1/summary.json" ] && [ -s "$1/summary_splits.json" ] && [ -s "$1/occurrence_summary.json" ]; }

run_cell(){
    local vocab=$1 preset=$2 grouping=$3 ds=$4 k=$5 bb=$6
    local stem=motifsat; [[ "$preset" == base_gsat* || "$preset" == motif_emb* ]] && stem=gsat   # run.py _method
    local rep fold seed=() droot
    if [ "$ds" = mutag ]; then rep="fold$k"; fold=$k; droot="$REPO/data"
    else rep="seed$k"; fold=0; seed=(--seed "$k"); droot="$OGB"; fi
    local cid="${vocab}__${preset}__${grouping}__${ds}__${rep}__${bb}"
    local dir="$RUNS/$vocab/$preset/$grouping/$ds/$rep/$bb"
    local evalonly=0; [ "$preset" = base_gsat ] && [ "$vocab" != rbrics_v3 ] && evalonly=1
    if [ "$evalonly" = 1 ]; then _evaluated "$dir" && return 0; else _complete "$dir" "$stem" && return 0; fi
    local rr; rr="$(r_for "$ds" "$bb")" || { echo "[skip] no r for $ds/$bb" >&2; return 0; }
    local fr=${rr% *} di=${rr#* }
    local vroot var
    if [ "$vocab" = rbrics_v3 ]; then vroot="$REPO/vocab_final_v3"; var=rbrics
    else vroot="$REPO/fmbpe_v4/vocab/fold$fold"; var=fm_bpe_full_union; fi
    local src="$RUNS/rbrics_v3/base_gsat/none/$ds/$rep/$bb"
    if [ "$evalonly" = 1 ]; then _complete "$src" gsat || return 0; fi        # base not trained yet -> later
    local gargs=(); [ "$grouping" != none ] && gargs=(--motif_grouping "$grouping")
    local eargs=(); [ "$evalonly" = 1 ] && eargs=(--eval_only --load_weights_from "$src/best_model.pt")
    mkdir "$CLAIMS/$cid" 2>/dev/null || return 0                         # CLAIM fails -> skip
    printf '%s\t%s\t%s\n' "$WHO" "$JOBID" "$(date +%s)" > "$CLAIMS/$cid/info"
    mkdir -p "$dir"
    [ "$evalonly" = 1 ] && printf '{"source_run": "%s", "final_r": %s, "decay_interval": %s, "vocab": "%s"}\n' \
        "$src" "$fr" "$di" "$vocab" > "$dir/source.json"
    echo "[run ${DEVICE:-cuda}] $cid (final_r=$fr decay_interval=$di clip_grad=${CLIP:-default} eval_only=$evalonly)"
    python3 MotifSAT/run.py --config "MotifSAT/configs/${preset}.yaml" \
        --dataset "$ds" --fold "$fold" "${seed[@]}" --backbone "$bb" \
        --data_root "$droot" --vocab_root "$vroot" --vocab_variant "$var" \
        --processed_root "$PROC_BASE/$vocab" "${gargs[@]}" \
        --info_loss_coef 1 --init_r 0.9 --decay_r 0.1 --final_r "$fr" --decay_interval "$di" \
        --ckpt_rule gsat ${CLIP:+--clip_grad "$CLIP"} \
        --out_dir "$dir" --final_out_dir --per_split_eval --occurrence_eval \
        --epochs "$EPOCHS" --patience "$PATIENCE" "${eargs[@]}" > "$dir/run.log" 2>&1
    local rc=$?
    if [ "$evalonly" = 1 ] && grep -q "\[eval_only\] loaded with" "$dir/run.log"; then rc=3; fi   # non-strict load
    if [ "$rc" -eq 0 ]; then
        if [ "$evalonly" = 1 ]; then _evaluated "$dir" && return 0; else _complete "$dir" "$stem" && return 0; fi
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date +%s)" "$(hostname -s)" "$JOBID" "${DEVICE:-cuda}" "$cid" "$rc" >> "$FAILURES"
    touch "$CLAIMS/$cid/.failed"
    echo "[FAIL rc=$rc] $cid"; tail -5 "$dir/run.log"
}

echo "worker $WHO DEVICE=${DEVICE:-cuda} POOL=[$POOL_DATASETS] VOCABS=[$VOCABS] PRESETS=[$PRESETS] REPS=[$REPS] EPOCHS=$EPOCHS CLIP=${CLIP:-default}"
# dataset outermost so the ground-truth set (mutag, listed first in POOL_DATASETS) finishes first;
# base_gsat first inside each dataset so the fmbpe_v4 base re-evaluations find their checkpoint.
work(){
for ds in $POOL_DATASETS; do
  for preset in $PRESETS; do
    grouping="$(grouping_of "$preset")" || { echo "[skip] unknown preset $preset" >&2; continue; }
    if [ "$preset" = base_gsat_noanneal ]; then vocs=rbrics_v3; else vocs=$VOCABS; fi
    for vocab in $vocs; do
      for k in $REPS; do
        for bb in $BACKBONES; do
          run_cell "$vocab" "$preset" "$grouping" "$ds" "$k" "$bb"
        done
      done
    done
  done
done
}
# PAR loops per job share the job's cores (one single-threaded process per core; -c 2 -> PAR 2,
# so a GPU job still holds 2 cores per GPU). Claims keep the loops from duplicating cells.
PAR="${PAR:-1}"
for _k in $(seq 1 "$PAR"); do work & sleep 5; done
wait
echo "worker $WHO DONE."
