#!/usr/bin/env bash
# variants_worker.sh — claim-based PULL worker for the MotifSAT variant campaign
# (fork of rsweep_worker.sh: same DONE / CLAIM / FAILURE contract, no auto-redeploy).
# Submitted through sbatch stdin by variants_launch.sh (no file copy to the HPC); this file
# is the record. Caches must exist first: variants_preprocess.sh.
#
# One CELL = (vocab, preset, grouping, dataset, fold, backbone). Run dir:
#     $OUT/variants_v1/<vocab>/<preset>/<grouping>/<dataset>/fold<f>/<backbone>/
#   vocab    rbrics_v3 = vocab_final_v3/<ds>/rbrics            (fold-0 vocabulary)
#            fmbpe_v4  = fmbpe_v4/vocab/fold<f>/<ds>/fm_bpe_full_union (per-fold vocabulary)
#   grouping pooled | copy  (--motif_grouping; one gate / consistency group per motif or per
#            occurrence). base_gsat_noanneal has no motif grouping -> grouping "none".
# Every run: beta = info_loss_coef 1 (GSAT's 1/|E|), ckpt_rule gsat, r from R_TABLE below
# (init_r 0.9, decay_r 0.1), --per_split_eval --occurrence_eval, 500 epochs, patience 50,
# clip_grad = $CLIP if set (else run.py default 2.0).
# DONE = native_complete.py passes AND history.json AND occurrence_summary.json non-empty.
#
# ENV: POOL_DATASETS (req), DEVICE=cuda|cpu, optional VOCABS PRESETS GROUPINGS BACKBONES FOLDS
#      EPOCHS PATIENCE CLIP SMOKE.
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
SCRIPTS="$REPO/motifsat_paper_deliverables_v1/_scripts"
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
[ "${DEVICE:-cuda}" = cpu ] && export CUDA_VISIBLE_DEVICES=""

OUT="$REPO/motifsat_paper_deliverables_v1"; RUNS="$OUT/variants_v1"
PROC_BASE="$REPO/processed_variants_v1"
FOLDS_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/
DISPATCH="$OUT/_dispatch_variants"; CLAIMS="$DISPATCH/claims"; FAILURES="$DISPATCH/failures.tsv"
# SMOKE (2-epoch) runs go to a separate tree + claim dir so they can never count as DONE.
if [ "${SMOKE:-0}" = 1 ]; then RUNS="$OUT/variants_v1_smoke"; CLAIMS="$DISPATCH/claims_smoke"; fi
mkdir -p "$CLAIMS"
[ -s "$FAILURES" ] || printf 'ts\thost\tjobid\tdevice\tcell_id\trc\n' > "$FAILURES"
WHO="$(hostname -s):${SLURM_JOB_ID:-$$}"; JOBID="${SLURM_JOB_ID:-$$}"

EPOCHS="${EPOCHS:-500}"; PATIENCE="${PATIENCE:-50}"
BACKBONES="${BACKBONES:-GIN GCN GAT SAGE PNA}"
FOLDS="${FOLDS:-0 1 2 3 4}"
VOCABS="${VOCABS:-rbrics_v3 fmbpe_v4}"
PRESETS="${PRESETS:-intra_consistency inter_loss intra_inter_loss readout readout_inter readout_layernorm readout_nonorm base_gsat_noanneal}"
GROUPINGS="${GROUPINGS:-pooled copy}"
: "${POOL_DATASETS:?set POOL_DATASETS to the space-separated datasets for this pool}"
# lists may arrive ':'-joined from variants_launch.sh
POOL_DATASETS="${POOL_DATASETS//:/ }"; VOCABS="${VOCABS//:/ }"; PRESETS="${PRESETS//:/ }"; GROUPINGS="${GROUPINGS//:/ }"; BACKBONES="${BACKBONES//:/ }"
if [ "${SMOKE:-0}" = 1 ]; then POOL_DATASETS="BBBP"; FOLDS="0"; BACKBONES="GIN"; EPOCHS=2; fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
       OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# r schedule per (dataset, backbone): "<final_r> <decay_interval>". Chosen from the r sweep
# (rsweep_runs/, validation node GT-ROC; non-GT datasets take their backbone's setting).
R_TABLE_FILLED_AT_LAUNCH

_complete(){
    python3 "$SCRIPTS/native_complete.py" "$1" "$2" --quiet >/dev/null 2>&1 \
        && [ -s "$1/history.json" ] && [ -s "$1/occurrence_summary.json" ]
}

run_cell(){
    local vocab=$1 preset=$2 grouping=$3 ds=$4 f=$5 bb=$6
    local stem=motifsat; [[ "$preset" == base_gsat* || "$preset" == motif_emb* ]] && stem=gsat   # run.py _method: gsat unless readout/loss
    local cid="${vocab}__${preset}__${grouping}__${ds}__f${f}__${bb}"
    local dir="$RUNS/$vocab/$preset/$grouping/$ds/fold$f/$bb"
    _complete "$dir" "$stem" && return 0                                  # DONE  -> skip
    local rr; rr="$(r_for "$ds" "$bb")" || { echo "[skip] no r for $ds/$bb" >&2; return 0; }
    local fr=${rr% *} di=${rr#* }
    local vroot var
    if [ "$vocab" = rbrics_v3 ]; then vroot="$REPO/vocab_final_v3"; var=rbrics
    else vroot="$REPO/fmbpe_v4/vocab/fold$f"; var=fm_bpe_full_union; fi
    local gargs=(); [ "$grouping" != none ] && gargs=(--motif_grouping "$grouping")
    mkdir "$CLAIMS/$cid" 2>/dev/null || return 0                         # CLAIM fails -> skip
    printf '%s\t%s\t%s\n' "$WHO" "$JOBID" "$(date +%s)" > "$CLAIMS/$cid/info"
    echo "[run ${DEVICE:-cuda}] $cid (final_r=$fr decay_interval=$di clip_grad=${CLIP:-default})"
    python3 MotifSAT/run.py --config "MotifSAT/configs/${preset}.yaml" \
        --dataset "$ds" --fold "$f" --backbone "$bb" \
        --data_root "$FOLDS_ROOT" --vocab_root "$vroot" --vocab_variant "$var" \
        --processed_root "$PROC_BASE/$vocab" "${gargs[@]}" \
        --info_loss_coef 1 --init_r 0.9 --decay_r 0.1 --final_r "$fr" --decay_interval "$di" \
        --ckpt_rule gsat ${CLIP:+--clip_grad "$CLIP"} \
        --out_dir "$dir" --final_out_dir --per_split_eval --occurrence_eval \
        --epochs "$EPOCHS" --patience "$PATIENCE"
    local rc=$?
    if [ "$rc" -eq 0 ] && _complete "$dir" "$stem"; then return 0; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date +%s)" "$(hostname -s)" "$JOBID" "${DEVICE:-cuda}" "$cid" "$rc" >> "$FAILURES"
    touch "$CLAIMS/$cid/.failed"
    echo "[FAIL rc=$rc] $cid"
}

echo "worker $WHO DEVICE=${DEVICE:-cuda} POOL=[$POOL_DATASETS] VOCABS=[$VOCABS] PRESETS=[$PRESETS] GROUPINGS=[$GROUPINGS] EPOCHS=$EPOCHS CLIP=${CLIP:-default}"
# dataset outermost so the ground-truth sets (listed first in POOL_DATASETS) finish first
work(){
for ds in $POOL_DATASETS; do
  for preset in $PRESETS; do
    if [ "$preset" = base_gsat_noanneal ]; then vocs=rbrics_v3; grps=none; else vocs=$VOCABS; grps=$GROUPINGS; fi
    for vocab in $vocs; do
      for grouping in $grps; do
        for f in $FOLDS; do
          for bb in $BACKBONES; do
            run_cell "$vocab" "$preset" "$grouping" "$ds" "$f" "$bb"
          done
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
