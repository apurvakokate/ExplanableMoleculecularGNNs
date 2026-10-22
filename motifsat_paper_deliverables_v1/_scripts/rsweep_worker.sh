#!/usr/bin/env bash
# rsweep_worker.sh — claim-based PULL worker for the base_gsat r-schedule sweep (rbrics).
# Fork of base_worker.sh: preset fixed to base_gsat, beta = info_loss_coef 1 on ALL datasets
# (GSAT's 1/|E|), ckpt_rule=gsat; loops over r SETTINGS = final_r x decay_interval.
# One SLURM job = one worker; launch many (via base_launch.sh) and they self-balance
# over the filesystem. Requires bash >= 4 (associative array).
#
# One CELL = (preset, dataset, fold, backbone). Deterministic run dir (no variant_tag
# glob — see native_complete.py):
#     $OUT/rsweep_runs/<setting>/<dataset>/fold<f>/<backbone>/   setting = r<final>_i<interval>
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
#   BACKBONES, FOLDS, PRESETS, EPOCHS, PATIENCE, SMOKE.
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
SCRIPTS="$REPO/motifsat_paper_deliverables_v1/_scripts"
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
[ "${DEVICE:-cuda}" = cpu ] && export CUDA_VISIBLE_DEVICES=""

VOCAB="$REPO/vocab_final_v2"; PROC="$REPO/processed_final_v2"
OUT="$REPO/motifsat_paper_deliverables_v1"; RUNS="$OUT/rsweep_runs"
FOLDS_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/
DISPATCH="$OUT/_dispatch_rsweep"; CLAIMS="$DISPATCH/claims"; FAILURES="$DISPATCH/failures.tsv"
mkdir -p "$CLAIMS"
[ -s "$FAILURES" ] || printf 'ts\thost\tjobid\tdevice\tcell_id\trc\n' > "$FAILURES"
WHO="$(hostname -s):${SLURM_JOB_ID:-$$}"; JOBID="${SLURM_JOB_ID:-$$}"

EPOCHS="${EPOCHS:-500}"
PATIENCE="${PATIENCE:-50}"
BACKBONES="${BACKBONES:-GIN GCN GAT SAGE PNA}"
FOLDS="${FOLDS:-0 1 2 3 4}"
: "${POOL_DATASETS:?set POOL_DATASETS to the space-separated datasets for this pool}"

# SMOKE: all 5 settings on BBBP / fold 0 / GIN only.
if [ "${SMOKE:-0}" = 1 ]; then POOL_DATASETS="BBBP"; FOLDS="0"; BACKBONES="GIN"; fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
       OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# r settings: final_r x decay_interval (final_r 0.9 = constant prior; one interval)
SETTINGS="${SETTINGS:-r0.5_i10 r0.5_i20 r0.7_i10 r0.7_i20 r0.9_i10}"
_complete(){ python3 "$SCRIPTS/native_complete.py" "$1" "$2" --quiet >/dev/null 2>&1; }

run_cell(){
    local setting=$1 ds=$2 f=$3 bb=$4 stem=gsat
    local fr="${setting%_i*}"; fr="${fr#r}"; local di="${setting#*_i}"
    local cid="${setting}__${ds}__f${f}__${bb}"
    local dir="$RUNS/$setting/$ds/fold$f/$bb"
    _complete "$dir" "$stem" && [ -s "$dir/history.json" ] && return 0   # DONE -> skip
    mkdir "$CLAIMS/$cid" 2>/dev/null || return 0                         # CLAIM fails -> skip
    printf '%s\t%s\t%s\n' "$WHO" "$JOBID" "$(date +%s)" > "$CLAIMS/$cid/info"
    echo "[run ${DEVICE:-cuda}] $cid (final_r=$fr decay_interval=$di)"
    python3 MotifSAT/run.py --config "MotifSAT/configs/base_gsat.yaml" \
        --dataset "$ds" --fold "$f" --backbone "$bb" \
        --data_root "$FOLDS_ROOT" --vocab_root "$VOCAB" --vocab_variant rbrics \
        --processed_root "$PROC" \
        --info_loss_coef 1 --init_r 0.9 --decay_r 0.1 --final_r "$fr" --decay_interval "$di" \
        --ckpt_rule gsat \
        --out_dir "$dir" --final_out_dir --per_split_eval --epochs "$EPOCHS" \
        --patience "$PATIENCE"
    local rc=$?
    if [ "$rc" -eq 0 ] && _complete "$dir" "$stem" && [ -s "$dir/history.json" ]; then return 0; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date +%s)" "$(hostname -s)" "$JOBID" "${DEVICE:-cuda}" "$cid" "$rc" >> "$FAILURES"
    touch "$CLAIMS/$cid/.failed"
    echo "[FAIL rc=$rc] $cid"
}
echo "worker $WHO DEVICE=${DEVICE:-cuda} POOL=[$POOL_DATASETS] FOLDS=[$FOLDS] BB=[$BACKBONES] EPOCHS=$EPOCHS"
for setting in $SETTINGS; do
    for ds in $POOL_DATASETS; do
        for f in $FOLDS; do
            for bb in $BACKBONES; do
                run_cell "$setting" "$ds" "$f" "$bb"
            done
        done
    done
done
echo "worker $WHO DONE."
