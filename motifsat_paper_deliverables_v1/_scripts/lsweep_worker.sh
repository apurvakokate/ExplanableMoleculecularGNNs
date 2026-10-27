#!/usr/bin/env bash
# lsweep_worker.sh — claim-based PULL worker for the consistency-loss coefficient sweep (approved 2026-10-27).
# Fork of variants_worker.sh (same DONE / CLAIM / FAILURE contract). Submitted through sbatch stdin by
# lsweep_launch.sh (no file copy to the HPC); this file is the record.
#
# Preset intra_inter_loss, pooled grouping, rBRICS v3. The loss is
#     motif_loss_coef * (within_node_coef * within_var - between_motif_coef * between_var)
# and the yaml sets all three to 1.0; explicitly typed CLI flags override the yaml (run.py _cli_provided_dests).
# SETTINGS (tag -> motif_loss_coef within_node_coef between_motif_coef):
#     lam0.1 0.1 1 1 | lam0.3 0.3 1 1 | lam3 3 1 1 | lam10 10 1 1      (overall strength; lam1 = existing runs)
#     btw0.3 1 1 0.3 | btw3 1 1 3                                      (within:between ratio at strength 1;
#                                                                       intra-only / inter-only = existing runs)
# One CELL = (setting, dataset, fold, backbone). Run dir:
#     $OUT/lsweep_v1/rbrics_v3_<tag>/intra_inter_loss/pooled/<dataset>/fold<f>/<backbone>/
# (the tag sits in the vocab slot so variants_collect.py lsweep_v1 reads the tree unchanged).
# Datasets: our eight (FOLDS data, r_table.sh) and mutag (data/, r_table_paper.sh, as paper_worker.sh).
# Every run otherwise identical to the existing intra_inter_loss runs: info_loss_coef 1, init_r 0.9,
# decay_r 0.1, r from the tables, ckpt_rule gsat, clip = run.py default 2.0, --per_split_eval
# --occurrence_eval, 500 epochs, patience 50.
# DONE = native_complete.py passes AND history.json AND occurrence_summary.json non-empty.
#
# ENV: POOL_DATASETS (req), DEVICE=cuda|cpu, optional SETTINGS BACKBONES FOLDS EPOCHS PATIENCE SMOKE PAR.
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
SCRIPTS="$REPO/motifsat_paper_deliverables_v1/_scripts"
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
[ "${DEVICE:-cuda}" = cpu ] && export CUDA_VISIBLE_DEVICES=""

OUT="$REPO/motifsat_paper_deliverables_v1"; RUNS="$OUT/lsweep_v1"
FOLDS_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/
DISPATCH="$OUT/_dispatch_lsweep"; CLAIMS="$DISPATCH/claims"; FAILURES="$DISPATCH/failures.tsv"
# SMOKE (2-epoch) runs go to a separate tree + claim dir so they can never count as DONE.
if [ "${SMOKE:-0}" = 1 ]; then RUNS="$OUT/lsweep_v1_smoke"; CLAIMS="$DISPATCH/claims_smoke"; fi
mkdir -p "$CLAIMS"
[ -s "$FAILURES" ] || printf 'ts\thost\tjobid\tdevice\tcell_id\trc\n' > "$FAILURES"
WHO="$(hostname -s):${SLURM_JOB_ID:-$$}"; JOBID="${SLURM_JOB_ID:-$$}"

EPOCHS="${EPOCHS:-500}"; PATIENCE="${PATIENCE:-50}"
BACKBONES="${BACKBONES:-GIN GCN GAT SAGE PNA}"
FOLDS="${FOLDS:-0 1 2 3 4}"
SETTINGS="${SETTINGS:-lam0.1 lam0.3 lam3 lam10 btw0.3 btw3}"
: "${POOL_DATASETS:?set POOL_DATASETS to the space-separated datasets for this pool}"
# lists may arrive ':'-joined from lsweep_launch.sh
POOL_DATASETS="${POOL_DATASETS//:/ }"; SETTINGS="${SETTINGS//:/ }"; BACKBONES="${BACKBONES//:/ }"; FOLDS="${FOLDS//:/ }"
if [ "${SMOKE:-0}" = 1 ]; then FOLDS="0"; BACKBONES="GIN"; EPOCHS=2; fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
       OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# r schedules: r_for_own (r_table.sh, our eight datasets) and r_for_paper (r_table_paper.sh, mutag).
R_TABLE_FILLED_AT_LAUNCH
r_for(){ if [ "$1" = mutag ]; then r_for_paper "$1" "$2"; else r_for_own "$1" "$2"; fi; }

coefs_of(){  # tag -> "<motif_loss_coef> <within_node_coef> <between_motif_coef>"; unknown -> rc 1
    case "$1" in
        lam0.1) echo "0.1 1 1" ;;  lam0.3) echo "0.3 1 1" ;;  lam3) echo "3 1 1" ;;  lam10) echo "10 1 1" ;;
        btw0.3) echo "1 1 0.3" ;;  btw3) echo "1 1 3" ;;
        *) return 1 ;;
    esac
}

_complete(){
    python3 "$SCRIPTS/native_complete.py" "$1" "$2" --quiet >/dev/null 2>&1 \
        && [ -s "$1/history.json" ] && [ -s "$1/occurrence_summary.json" ]
}

run_cell(){
    local tag=$1 ds=$2 f=$3 bb=$4
    local cid="rbrics_v3_${tag}__intra_inter_loss__pooled__${ds}__f${f}__${bb}"
    local dir="$RUNS/rbrics_v3_${tag}/intra_inter_loss/pooled/$ds/fold$f/$bb"
    _complete "$dir" motifsat && return 0                                 # DONE  -> skip
    local cc; cc="$(coefs_of "$tag")" || { echo "[skip] unknown setting $tag" >&2; return 0; }
    local lam wn bm; read -r lam wn bm <<< "$cc"
    local rr; rr="$(r_for "$ds" "$bb")" || { echo "[skip] no r for $ds/$bb" >&2; return 0; }
    local fr=${rr% *} di=${rr#* }
    local droot proc
    if [ "$ds" = mutag ]; then droot="$REPO/data"; proc="$REPO/processed_paper_v1/rbrics_v3"
    else droot="$FOLDS_ROOT"; proc="$REPO/processed_variants_v1/rbrics_v3"; fi
    mkdir "$CLAIMS/$cid" 2>/dev/null || return 0                         # CLAIM fails -> skip
    printf '%s\t%s\t%s\n' "$WHO" "$JOBID" "$(date +%s)" > "$CLAIMS/$cid/info"
    mkdir -p "$dir"
    echo "[run ${DEVICE:-cuda}] $cid (coef=$lam within=$wn between=$bm final_r=$fr decay_interval=$di)"
    python3 MotifSAT/run.py --config MotifSAT/configs/intra_inter_loss.yaml \
        --dataset "$ds" --fold "$f" --backbone "$bb" \
        --data_root "$droot" --vocab_root "$REPO/vocab_final_v3" --vocab_variant rbrics \
        --processed_root "$proc" --motif_grouping pooled \
        --motif_loss_coef "$lam" --within_node_coef "$wn" --between_motif_coef "$bm" \
        --info_loss_coef 1 --init_r 0.9 --decay_r 0.1 --final_r "$fr" --decay_interval "$di" \
        --ckpt_rule gsat \
        --out_dir "$dir" --final_out_dir --per_split_eval --occurrence_eval \
        --epochs "$EPOCHS" --patience "$PATIENCE" > "$dir/run.log" 2>&1
    local rc=$?
    if [ "$rc" -eq 0 ] && _complete "$dir" motifsat; then return 0; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date +%s)" "$(hostname -s)" "$JOBID" "${DEVICE:-cuda}" "$cid" "$rc" >> "$FAILURES"
    touch "$CLAIMS/$cid/.failed"
    echo "[FAIL rc=$rc] $cid"; tail -5 "$dir/run.log"
}

# Requeue recovery (added 2026-10-27 after the smoke lost 2 cells this way): a preempted array task is requeued
# with the SAME job id, so the claims it took before preemption look live to the janitor and block its own
# loops. On a restart (SLURM_RESTART_COUNT > 0), before any loop starts, this job's unfinished claims are
# MOVED (never deleted) to claims_orphaned/<stamp>_requeue_<jobid>/, their partial run dirs to
# <RUNS>_interrupted/<same stamp>/, so the loops below pick the cells up again.
if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
  T="$(date +%Y%m%d%H%M%S)_requeue_${JOBID}"
  for c in "$CLAIMS"/*; do
    [ -d "$c" ] || continue; [ -e "$c/.failed" ] && continue
    [ "$(cut -f2 "$c/info" 2>/dev/null)" = "$JOBID" ] || continue
    IFS='|' read -r v p g ds f bb <<< "$(basename "$c" | sed 's/__/|/g')"
    run="$RUNS/$v/$p/$g/$ds/fold${f#f}/$bb"
    _complete "$run" motifsat && continue
    mkdir -p "$DISPATCH/claims_orphaned/$T"; mv "$c" "$DISPATCH/claims_orphaned/$T/"
    if [ -d "$run" ]; then mkdir -p "${RUNS}_interrupted/$T/$v/$p/$g/$ds/fold${f#f}"; mv "$run" "${RUNS}_interrupted/$T/$v/$p/$g/$ds/fold${f#f}/"; fi
    echo "[requeue] released $(basename "$c") (restart ${SLURM_RESTART_COUNT})"
  done
fi

echo "worker $WHO DEVICE=${DEVICE:-cuda} POOL=[$POOL_DATASETS] SETTINGS=[$SETTINGS] BACKBONES=[$BACKBONES] FOLDS=[$FOLDS] EPOCHS=$EPOCHS"
# dataset outermost so the ground-truth sets (listed first in POOL_DATASETS) finish first
work(){
for ds in $POOL_DATASETS; do
  for tag in $SETTINGS; do
    for f in $FOLDS; do
      for bb in $BACKBONES; do
        run_cell "$tag" "$ds" "$f" "$bb"
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
