#!/usr/bin/env bash
# base_worker.sh — claim-based PULL worker for the MotifSAT paper base-runs (rbrics).
# One SLURM job = one worker; launch many (via base_launch.sh) and they self-balance
# over the filesystem. Requires bash >= 4 (associative array).
#
# One CELL = (preset, dataset, fold, backbone). Deterministic run dir (no variant_tag
# glob — see native_complete.py):
#     $OUT/base_runs/<preset>/<dataset>/fold<f>/<backbone>/
#
# CONTRACT
#   DONE    = native_complete.py passes on the cell dir (ALL artifacts present +
#             non-empty + content-valid). NOT a marker-file existence check.
#   CLAIM   = atomic `mkdir` of $CLAIMS/<cell_id> (prevents duplicate concurrent
#             runs; shared across the GPU + CPU pools).
#   FAILURE = run.py rc!=0 OR incomplete after rc==0 -> appended to failures.tsv +
#             a `.failed` marker in the claim dir.
#   NO AUTO-REDEPLOY: a claimed cell (running / failed / orphaned-by-preemption) is
#             SKIPPED by every worker. Re-kick is MANUAL via base_status.py.
#
# ENV (from the launcher): POOL_DATASETS (req), DEVICE=cuda|cpu, plus optional
#   BACKBONES, FOLDS, PRESETS, EPOCHS, SMOKE.
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
SCRIPTS="$REPO/motifsat_paper_deliverables_v1/_scripts"
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
[ "${DEVICE:-cuda}" = cpu ] && export CUDA_VISIBLE_DEVICES=""

VOCAB="$REPO/vocab_final_v2"; PROC="$REPO/processed_final_v2"
OUT="$REPO/motifsat_paper_deliverables_v1"; RUNS="$OUT/base_runs"
FOLDS_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/
DISPATCH="$OUT/_dispatch"; CLAIMS="$DISPATCH/claims"; FAILURES="$DISPATCH/failures.tsv"
mkdir -p "$CLAIMS"
[ -s "$FAILURES" ] || printf 'ts\thost\tjobid\tdevice\tcell_id\trc\n' > "$FAILURES"
WHO="$(hostname -s):${SLURM_JOB_ID:-$$}"; JOBID="${SLURM_JOB_ID:-$$}"

EPOCHS="${EPOCHS:-500}"
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
    _complete "$dir" "$stem" && return 0                       # DONE  -> skip
    mkdir "$CLAIMS/$cid" 2>/dev/null || return 0              # CLAIM fails -> skip
    printf '%s\t%s\t%s\n' "$WHO" "$JOBID" "$(date +%s)" > "$CLAIMS/$cid/info"
    echo "[run ${DEVICE:-cuda}] $cid"
    python3 MotifSAT/run.py --config "MotifSAT/configs/${preset}.yaml" \
        --dataset "$ds" --fold "$f" --backbone "$bb" \
        --data_root "$FOLDS_ROOT" --vocab_root "$VOCAB" --vocab_variant rbrics \
        --processed_root "$PROC" \
        --out_dir "$dir" --final_out_dir --per_split_eval --epochs "$EPOCHS"
    local rc=$?
    if [ "$rc" -eq 0 ] && _complete "$dir" "$stem"; then return 0; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date +%s)" "$(hostname -s)" "$JOBID" "${DEVICE:-cuda}" "$cid" "$rc" >> "$FAILURES"
    touch "$CLAIMS/$cid/.failed"
    echo "[FAIL rc=$rc] $cid"
}

echo "worker $WHO DEVICE=${DEVICE:-cuda} POOL=[$POOL_DATASETS] FOLDS=[$FOLDS] BB=[$BACKBONES] EPOCHS=$EPOCHS"
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
