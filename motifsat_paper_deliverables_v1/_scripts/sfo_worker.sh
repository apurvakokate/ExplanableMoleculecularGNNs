#!/usr/bin/env bash
# sfo_worker.sh — claim-based PULL worker for MotifSAT on the SFO vocabulary
# (size_frequency_optimization). Fork of base_worker.sh; the ONLY differences are
# the PER-FOLD vocab/processed roots (fragmentation_v2/{vocab,processed}/fold<f>),
# the vocab variant, the output tree (sfo_runs) and an isolated _dispatch_sfo.
# Requires bash >= 4 (associative array).
#
# One CELL = (preset, dataset, fold, backbone). Deterministic run dir:
#     $OUT/sfo_runs/<preset>/<dataset>/fold<f>/<backbone>/
# DONE/CLAIM/FAILURE/skip semantics identical to base_worker (native_complete.py).
#
# ENV (from the launcher): POOL_DATASETS (req), DEVICE=cuda|cpu, plus optional
#   BACKBONES, FOLDS, PRESETS, EPOCHS, PATIENCE, SMOKE.
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
SCRIPTS="$REPO/motifsat_paper_deliverables_v1/_scripts"
FRAG="$REPO/fragmentation_v2"                              # SFO vocab/processed tree
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
[ "${DEVICE:-cuda}" = cpu ] && export CUDA_VISIBLE_DEVICES=""

OUT="$REPO/motifsat_paper_deliverables_v1"; RUNS="$OUT/sfo_runs"
FOLDS_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/
DISPATCH="$OUT/_dispatch_sfo"; CLAIMS="$DISPATCH/claims"; FAILURES="$DISPATCH/failures.tsv"
mkdir -p "$CLAIMS"
[ -s "$FAILURES" ] || printf 'ts\thost\tjobid\tdevice\tcell_id\trc\n' > "$FAILURES"
WHO="$(hostname -s):${SLURM_JOB_ID:-$$}"; JOBID="${SLURM_JOB_ID:-$$}"

EPOCHS="${EPOCHS:-500}"
PATIENCE="${PATIENCE:-50}"
BACKBONES="${BACKBONES:-GIN GCN GAT SAGE PNA}"
FOLDS="${FOLDS:-0 1 2 3 4}"
: "${POOL_DATASETS:?set POOL_DATASETS to the space-separated datasets for this pool}"

# SMOKE: exercise all 9 presets on BBBP / fold 0 / GIN only.
if [ "${SMOKE:-0}" = 1 ]; then POOL_DATASETS="BBBP"; FOLDS="0"; BACKBONES="GIN"; fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
       OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# preset -> stem, ordered — single source of truth = native_complete.py
PRESET_ORDER=(); declare -A STEM
while IFS=$'\t' read -r _p _s; do
    [ -n "$_p" ] || continue
    PRESET_ORDER+=("$_p"); STEM["$_p"]="$_s"
done < <(python3 "$SCRIPTS/native_complete.py" --list-presets)
PRESETS="${PRESETS:-${PRESET_ORDER[*]}}"

_complete(){ python3 "$SCRIPTS/native_complete.py" "$1" "$2" --quiet >/dev/null 2>&1; }

run_cell(){
    local preset=$1 ds=$2 f=$3 bb=$4
    local stem="${STEM[$preset]:-}"
    [ -n "$stem" ] || { echo "[skip] unknown preset '$preset'" >&2; return 0; }
    local cid="${preset}__${ds}__f${f}__${bb}"
    local dir="$RUNS/$preset/$ds/fold$f/$bb"
    # PER-FOLD sfo roots (the base_worker uses a single fold-agnostic rbrics root)
    local vroot="$FRAG/vocab/fold$f"
    local proot="$FRAG/processed/fold$f"
    _complete "$dir" "$stem" && return 0                       # DONE  -> skip
    mkdir "$CLAIMS/$cid" 2>/dev/null || return 0              # CLAIM fails -> skip
    printf '%s\t%s\t%s\n' "$WHO" "$JOBID" "$(date +%s)" > "$CLAIMS/$cid/info"
    echo "[run ${DEVICE:-cuda}] $cid"
    python3 MotifSAT/run.py --config "MotifSAT/configs/${preset}.yaml" \
        --dataset "$ds" --fold "$f" --backbone "$bb" \
        --data_root "$FOLDS_ROOT" --vocab_root "$vroot" \
        --vocab_variant size_frequency_optimization \
        --processed_root "$proot" \
        --out_dir "$dir" --final_out_dir --per_split_eval --epochs "$EPOCHS" \
        --patience "$PATIENCE"
    local rc=$?
    if [ "$rc" -eq 0 ] && _complete "$dir" "$stem"; then return 0; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date +%s)" "$(hostname -s)" "$JOBID" "${DEVICE:-cuda}" "$cid" "$rc" >> "$FAILURES"
    touch "$CLAIMS/$cid/.failed"
    echo "[FAIL rc=$rc] $cid"
}

echo "worker $WHO DEVICE=${DEVICE:-cuda} POOL=[$POOL_DATASETS] FOLDS=[$FOLDS] BB=[$BACKBONES] EPOCHS=$EPOCHS (SFO)"
for preset in $PRESETS; do
    for ds in $POOL_DATASETS; do
        for f in $FOLDS; do
            for bb in $BACKBONES; do
                run_cell "$preset" "$ds" "$f" "$bb"
            done
        done
    done
done
echo "worker $WHO DONE."
