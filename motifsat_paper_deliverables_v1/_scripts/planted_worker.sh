#!/usr/bin/env bash
# planted_worker.sh — claim-based PULL worker for the MotifSAT planted-GT campaign (P2, approved 2026-10-26;
# fork of variants_worker.sh: same DONE / CLAIM / FAILURE contract). Submitted through sbatch stdin by
# planted_launch.sh (no file copy); this file is the record. Caches must exist first: planted_prep.sh (P1).
#
# One CELL = (rule, preset, grouping, dataset, fold, backbone). Run dir:
#     $OUT/planted_v3/runs/<rule>/<preset>/<grouping>/<dataset>/fold<f>/<backbone>/
# Presets (best of each family + GSAT): base_gsat/none, intra_inter_loss/pooled, readout_nonorm/pooled,
# motif_emb_iddesc_gin_res/copy. Vocabulary: rBRICS v3 (planted_v3/vocab = symlinks to vocab_final_v3 +
# the planted rule_tiers.json). Labels and ground truth: --use_gt --gt_cache planted_v3/gt_cache --gt_tier <rule>.
# Every run: beta = info_loss_coef 1, ckpt_rule gsat, r from R_TABLE (init 0.9, decay 0.1), clip = run.py
# default 2.0 unless $CLIP, --per_split_eval --occurrence_eval, 500 epochs, patience 50.
# DONE = native_complete.py passes AND history.json AND occurrence_summary.json non-empty.
#
# ENV: POOL_DATASETS (req), DEVICE=cuda|cpu, optional PRESETS BACKBONES FOLDS RULESET EPOCHS PATIENCE CLIP SMOKE PAR.
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
SCRIPTS="$REPO/motifsat_paper_deliverables_v1/_scripts"
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
[ "${DEVICE:-cuda}" = cpu ] && export CUDA_VISIBLE_DEVICES=""

OUT="$REPO/motifsat_paper_deliverables_v1"; P3="$OUT/planted_v3"; RUNS="$P3/runs"
PROC="$REPO/processed_variants_v1/rbrics_v3"
FOLDS_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/
DISPATCH="$OUT/_dispatch_planted"; CLAIMS="$DISPATCH/claims"; FAILURES="$DISPATCH/failures.tsv"
if [ "${SMOKE:-0}" = 1 ]; then RUNS="$P3/runs_smoke"; CLAIMS="$DISPATCH/claims_smoke"; fi
mkdir -p "$CLAIMS"
[ -s "$FAILURES" ] || printf 'ts\thost\tjobid\tdevice\tcell_id\trc\n' > "$FAILURES"
WHO="$(hostname -s):${SLURM_JOB_ID:-$$}"; JOBID="${SLURM_JOB_ID:-$$}"

EPOCHS="${EPOCHS:-500}"; PATIENCE="${PATIENCE:-50}"
BACKBONES="${BACKBONES:-GIN GCN GAT SAGE PNA}"; FOLDS="${FOLDS:-0 1 2 3 4}"
PRESETS="${PRESETS:-base_gsat intra_inter_loss readout_nonorm motif_emb_iddesc_gin_res}"
: "${POOL_DATASETS:?set POOL_DATASETS}"
POOL_DATASETS="${POOL_DATASETS//:/ }"; PRESETS="${PRESETS//:/ }"; BACKBONES="${BACKBONES//:/ }"; FOLDS="${FOLDS//:/ }"
declare -A RULES=(
  [BBBP]="dnf_k1_r1 dnf_k1_r2 dnf_k3_r8 dnf_k2_r6 dnf_k2_r7 dnf_k2_r10 dnf_k3_r1 dnf_k3_r2 dnf_k3_r3 dnf_k3_r4"
  [hERG]="dnf_k1_r1 dnf_k1_r2 dnf_k2_r2 dnf_k2_r6 dnf_k3_r12 dnf_k2_r9 dnf_k2_r1 dnf_k3_r1 dnf_k3_r2 dnf_k3_r3"
  [Mutagenicity]="dnf_k1_r1 dnf_k1_r2 dnf_k1_r3 dnf_k3_r1 dnf_k3_r2 dnf_k3_r3 dnf_k3_r4 dnf_k3_r6 dnf_k3_r7 dnf_k3_r8 dnf_k2_r4"
)
if [ "${SMOKE:-0}" = 1 ]; then FOLDS="0"; BACKBONES="GIN"; EPOCHS=2; fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
       OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

R_TABLE_FILLED_AT_LAUNCH

_complete(){
    python3 "$SCRIPTS/native_complete.py" "$1" "$2" --quiet >/dev/null 2>&1 \
        && [ -s "$1/history.json" ] && [ -s "$1/occurrence_summary.json" ]
}
grouping_of(){ case "$1" in base_gsat) echo none;; motif_emb_*) echo copy;; *) echo pooled;; esac; }

run_cell(){
    local rule=$1 preset=$2 ds=$3 f=$4 bb=$5 grouping; grouping="$(grouping_of "$preset")"
    local stem=motifsat; [[ "$preset" == base_gsat* || "$preset" == motif_emb* ]] && stem=gsat
    local cid="${rule}__${preset}__${grouping}__${ds}__f${f}__${bb}"
    local dir="$RUNS/$rule/$preset/$grouping/$ds/fold$f/$bb"
    _complete "$dir" "$stem" && return 0
    [ -s "$P3/gt_cache/$ds/fold$f/rbrics/relabel_$rule/test_with_gt.pt" ] || { echo "[skip] no cache $ds/$rule/f$f" >&2; return 0; }
    python3 -c "import json,sys; sys.exit(0 if json.load(open('$P3/_prep/${ds}__${rule}__f${f}.json'))['ok'] else 1)" 2>/dev/null \
        || { echo "[skip] prep check not passed: $ds/$rule/f$f" >&2; return 0; }
    local rr; rr="$(r_for "$ds" "$bb")" || { echo "[skip] no r for $ds/$bb" >&2; return 0; }
    local fr=${rr% *} di=${rr#* }
    local gargs=(); [ "$grouping" != none ] && gargs=(--motif_grouping "$grouping")
    mkdir "$CLAIMS/$cid" 2>/dev/null || return 0
    printf '%s\t%s\t%s\n' "$WHO" "$JOBID" "$(date +%s)" > "$CLAIMS/$cid/info"
    mkdir -p "$dir"
    echo "[run ${DEVICE:-cuda}] $cid (final_r=$fr decay_interval=$di clip_grad=${CLIP:-default})"
    python3 MotifSAT/run.py --config "MotifSAT/configs/${preset}.yaml" \
        --dataset "$ds" --fold "$f" --backbone "$bb" \
        --data_root "$FOLDS_ROOT" --vocab_root "$P3/vocab" --vocab_variant rbrics \
        --processed_root "$PROC" "${gargs[@]}" \
        --use_gt --gt_cache "$P3/gt_cache" --gt_tier "$rule" \
        --info_loss_coef 1 --init_r 0.9 --decay_r 0.1 --final_r "$fr" --decay_interval "$di" \
        --ckpt_rule gsat ${CLIP:+--clip_grad "$CLIP"} \
        --out_dir "$dir" --final_out_dir --per_split_eval --occurrence_eval \
        --epochs "$EPOCHS" --patience "$PATIENCE" > "$dir/run.log" 2>&1
    local rc=$?
    if [ "$rc" -eq 0 ] && _complete "$dir" "$stem"; then return 0; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%s)" "$(hostname -s)" "$JOBID" "${DEVICE:-cuda}" "$cid" "$rc" >> "$FAILURES"
    touch "$CLAIMS/$cid/.failed"
    echo "[FAIL rc=$rc] $cid"; tail -5 "$dir/run.log"
}

echo "worker $WHO DEVICE=${DEVICE:-cuda} POOL=[$POOL_DATASETS] PRESETS=[$PRESETS] EPOCHS=$EPOCHS CLIP=${CLIP:-default}"
work(){
for ds in $POOL_DATASETS; do
  read -ra rs <<< "${RULES[$ds]}"; [ "${SMOKE:-0}" = 1 ] && rs=("${rs[0]}")
  for rule in "${rs[@]}"; do
    for preset in $PRESETS; do
      for f in $FOLDS; do
        for bb in $BACKBONES; do run_cell "$rule" "$preset" "$ds" "$f" "$bb"; done
      done
    done
  done
done
}
PAR="${PAR:-1}"
for _k in $(seq 1 "$PAR"); do work & sleep 5; done
wait
echo "worker $WHO DONE."
