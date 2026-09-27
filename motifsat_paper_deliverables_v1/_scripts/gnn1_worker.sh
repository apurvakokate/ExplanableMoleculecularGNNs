#!/usr/bin/env bash
# gnn1_worker.sh — claim-based PULL worker for the GNN1-design ablation (mechanism ③,
# motif_emb, full rbrics vocab). Fork of base_worker.sh. Requires bash >= 4.
#
# One CELL = (config, dataset, fold, backbone), where `config` is one of the 32
# factorial GNN1-design configs enumerated by gnn1_grid.py. Deterministic run dir
# (the config_id path carries all five knobs — NOT variant_tag; --final_out_dir):
#     $OUT/gnn1_ablation_runs/<config_id>/<dataset>/fold<f>/<backbone>/
#
# CONTRACT
#   DONE    = native_complete.py passes on the cell dir with stem 'gsat' (motif_emb
#             is written with method stem gsat). ALL artifacts present + non-empty +
#             content-valid. NOT a marker-file check.
#   CLAIM   = atomic `mkdir` of $CLAIMS/<cell_id> (shared across GPU + CPU pools).
#   FAILURE = run.py rc!=0 OR incomplete after rc==0 -> failures.tsv + `.failed`.
#   NO AUTO-REDEPLOY: claimed cells are skipped; re-kick is MANUAL via gnn1_status.py.
#
# The 32 configs share MotifSAT/configs/motif_emb_base.yaml (method family only);
# each cell's five knobs are set EXPLICITLY by the flags gnn1_grid.py emits. NO
# --use_gt: GT-ROC auto-triggers via node_label on the *_Verified_GT datasets.
#
# ENV (from the launcher): POOL_DATASETS (req), DEVICE=cuda|cpu, plus optional
#   BACKBONES, FOLDS, CONFIGS (space-separated config_ids to restrict to), EPOCHS,
#   PATIENCE, SMOKE, BASE_PRESET.
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
SCRIPTS="$REPO/motifsat_paper_deliverables_v1/_scripts"
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
[ "${DEVICE:-cuda}" = cpu ] && export CUDA_VISIBLE_DEVICES=""

VOCAB="$REPO/vocab_final_v2"; PROC="$REPO/processed_final_v2"
OUT="$REPO/motifsat_paper_deliverables_v1"; RUNS="$OUT/gnn1_ablation_runs"
FOLDS_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/
DISPATCH="$OUT/_dispatch_gnn1"; CLAIMS="$DISPATCH/claims"; FAILURES="$DISPATCH/failures.tsv"
mkdir -p "$CLAIMS"
[ -s "$FAILURES" ] || printf 'ts\thost\tjobid\tdevice\tcell_id\trc\n' > "$FAILURES"
WHO="$(hostname -s):${SLURM_JOB_ID:-$$}"; JOBID="${SLURM_JOB_ID:-$$}"

STEM=gsat                                    # motif_emb -> run.py _method == 'gsat'
BASE_PRESET="${BASE_PRESET:-MotifSAT/configs/motif_emb_base.yaml}"
EPOCHS="${EPOCHS:-500}"
PATIENCE="${PATIENCE:-50}"
BACKBONES="${BACKBONES:-GIN GCN GAT SAGE PNA}"
FOLDS="${FOLDS:-0 1 2 3 4}"
: "${POOL_DATASETS:?set POOL_DATASETS to the space-separated datasets for this pool}"

# Load the 32 configs as "config_id<TAB>flags" lines (single source = gnn1_grid.py).
declare -A CFG_FLAGS; CONFIG_ORDER=()
while IFS=$'\t' read -r _cid _flags; do
    [ -n "$_cid" ] || continue
    CONFIG_ORDER+=("$_cid"); CFG_FLAGS["$_cid"]="$_flags"
done < <(python3 "$SCRIPTS/gnn1_grid.py" --list-configs)
# Optional restriction to a subset of config_ids (targeted reruns).
CONFIGS="${CONFIGS:-${CONFIG_ORDER[*]}}"

# SMOKE: anchor + the richest cell (id_desc + D-chemistry + LayerNorm + residual)
# — exercises every knob incl. the chem edge feature and confirms all 20 artifacts
# land for motif_emb. BBBP (no-GT contract) + one GT set (Benzene) so GT-ROC emits.
if [ "${SMOKE:-0}" = 1 ]; then
    POOL_DATASETS="${POOL_DATASETS:-BBBP Benzene_Verified_GT}"; FOLDS="0"; BACKBONES="GIN"
    CONFIGS="mf-multihot__gin__n-none__r-off__L2 mf-id_desc__ginechem__n-layer__r-on__L2"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
       OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

_complete(){ python3 "$SCRIPTS/native_complete.py" "$1" "$STEM" --quiet >/dev/null 2>&1; }

run_cell(){
    local cfgid=$1 ds=$2 f=$3 bb=$4
    local flags="${CFG_FLAGS[$cfgid]:-}"
    # Unknown config_id = operator typo in $CONFIGS. Fail LOUD, do not silently skip.
    [ -n "$flags" ] || { echo "[FATAL] unknown config_id '$cfgid' (not in gnn1_grid.py --list-configs)" >&2; exit 1; }
    local cid="${cfgid}__${ds}__f${f}__${bb}"
    local dir="$RUNS/$cfgid/$ds/fold$f/$bb"
    _complete "$dir" && return 0                                # DONE  -> skip
    mkdir "$CLAIMS/$cid" 2>/dev/null || return 0               # CLAIM fails -> skip
    printf '%s\t%s\t%s\n' "$WHO" "$JOBID" "$(date +%s)" > "$CLAIMS/$cid/info"
    echo "[run ${DEVICE:-cuda}] $cid"
    # $flags is intentionally unquoted: each token (e.g. --motif_feat, multihot) is a
    # separate argv entry with no internal spaces.
    python3 MotifSAT/run.py --config "$BASE_PRESET" \
        $flags \
        --dataset "$ds" --fold "$f" --backbone "$bb" \
        --data_root "$FOLDS_ROOT" --vocab_root "$VOCAB" --vocab_variant rbrics \
        --processed_root "$PROC" \
        --out_dir "$dir" --final_out_dir --per_split_eval --epochs "$EPOCHS" \
        --patience "$PATIENCE"
    local rc=$?
    if [ "$rc" -eq 0 ] && _complete "$dir"; then return 0; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date +%s)" "$(hostname -s)" "$JOBID" "${DEVICE:-cuda}" "$cid" "$rc" >> "$FAILURES"
    touch "$CLAIMS/$cid/.failed"
    echo "[FAIL rc=$rc] $cid"
}

echo "worker $WHO DEVICE=${DEVICE:-cuda} POOL=[$POOL_DATASETS] FOLDS=[$FOLDS] BB=[$BACKBONES] EPOCHS=$EPOCHS NCFG=$(echo $CONFIGS | wc -w)"
for cfgid in $CONFIGS; do
    for ds in $POOL_DATASETS; do
        for f in $FOLDS; do
            for bb in $BACKBONES; do
                run_cell "$cfgid" "$ds" "$f" "$bb"
            done
        done
    done
done
echo "worker $WHO DONE."
