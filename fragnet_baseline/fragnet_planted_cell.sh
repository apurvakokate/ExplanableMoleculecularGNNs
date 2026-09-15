#!/usr/bin/env bash
# One FragNet PLANTED cell = one (dataset, rule, fold), run END-TO-END:
#   dump_context(l2xgnn) -> prep(fragnet) -> finetune(fragnet) -> export(fragnet) -> eval(l2xgnn, x2 unk)
# Self-contained (no cross-phase SLURM dependency), so a preemption only re-runs THIS cell.
# IDEMPOTENT: skips if both unk eval CSVs for this fold already exist -> safe under --requeue and
# safe to re-launch the whole array to mop up incompletes. Writes an atomic per-cell .status (rc+secs).
#
# Usage: fragnet_planted_cell.sh DS RULE FOLD [DEV=gpu|cpu]
set -uo pipefail
ds=${1:?DS}; rid=${2:?RULE}; fold=${3:?FOLD}; DEV=${4:-gpu}

# ── coordinates (mirror run_base_runs.sh / prior deploy) ──────────────────────
FB=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/fragnet_baseline
VENDOR=$FB/vendor/FragNet
PT=$VENDOR/fragnet/exps/pt/unimol_exp1s4/pt.pt
SA=$FB/stage_a_fragnet_env; SB=$FB/stage_b_adapter
DATA_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS
PROC_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/processed_final_v2
VOCAB_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/vocab_final_v2
PV=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/planted_v2
BASE=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/fragnet/planted
VOCAB=rbrics
BATCH_SIZE=${BATCH_SIZE:-256}
CONDA=/nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh

UW=$BASE/$ds/$rid/fold$fold
EVAL_ROOT=$BASE/$ds/$rid/eval
TRK=$BASE/_status; mkdir -p "$UW" "$TRK"
ST=$TRK/${ds}_${rid}_f${fold}.status

# device mapping: gpu -> train/export on GPU; cpu -> CPU + pin threads to the 2-core alloc
if [ "$DEV" = cpu ]; then
  FT_DEV=cpu; EXP_DEV=cpu
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
else
  FT_DEV=gpu; EXP_DEV=auto
fi

# ── skip-if-done: completion == both unk eval CSVs present for this fold ──────
done_csv() { find "$EVAL_ROOT/unk-$1" -path "*fold${fold}*" -name fragnet_frag_perlayer_metrics.csv 2>/dev/null | grep -q .; }
if done_csv include && done_csv exclude; then
  echo "SKIP $ds $rid f$fold (eval CSVs present)"; echo "$ds $rid f$fold SKIP done" > "$ST"; exit 0
fi

t0=$(date +%s)
echo "=== cell $ds $rid fold$fold DEV=$DEV -> $UW ==="
source "$CONDA"
fail() { echo "$ds $rid f$fold FAIL $1 rc=$2 secs=$(( $(date +%s)-t0 ))" > "$ST"; exit 1; }

# ── dump (l2xgnn) ─────────────────────────────────────────────────────────────
conda activate l2xgnn
python $SB/align_and_aggregate.py dump_context --dataset "$ds" --fold "$fold" --vocab $VOCAB \
  --regime planted --planted_root $PV --rule_id "$rid" \
  --data_root $DATA_ROOT --processed_root $PROC_ROOT --vocab_root $VOCAB_ROOT \
  --out "$UW/graph_context.json"; rc=$?; [ $rc -ne 0 ] && fail dump $rc
conda deactivate

# ── prep + finetune + export (fragnet) ────────────────────────────────────────
conda activate fragnet
python $SA/prep_data.py --graph_context "$UW/graph_context.json" --out_dir "$UW" \
  --vendor $VENDOR --frag_type custom; rc=$?; [ $rc -ne 0 ] && fail prep $rc
python $SA/finetune_fragnet.py --work "$UW" --pt_ckpt $PT --vendor $VENDOR \
  --task clf --epochs 10000 --es_patience 100 --batch_size $BATCH_SIZE --device $FT_DEV
rc=$?; [ $rc -ne 0 ] && fail finetune $rc
python $SA/export_frag_attention.py --work "$UW" --vendor $VENDOR --device $EXP_DEV \
  --graph_context "$UW/graph_context.json" --out "$UW/fragnet_frag_neutral.json" 2>&1 | grep -v "bond mask value"
rc=${PIPESTATUS[0]}
{ [ $rc -ne 0 ] || [ ! -s "$UW/fragnet_frag_neutral.json" ]; } && fail export $rc
rm -f "$UW/train.pkl" "$UW/val.pkl" "$UW/test.pkl"   # transient — free per cell
conda deactivate

# ── eval (l2xgnn), both unk modes ─────────────────────────────────────────────
conda activate l2xgnn
erc=0
for unk in include exclude; do
  echo "--- eval unk=$unk ---"
  python $SB/eval_frag_attention.py --dataset "$ds" --fold "$fold" --vocab $VOCAB --unk "$unk" \
    --regime planted --planted_root $PV --rule_id "$rid" \
    --data_root $DATA_ROOT --processed_root $PROC_ROOT --vocab_root $VOCAB_ROOT \
    --neutral "$UW/fragnet_frag_neutral.json" --dest_root "$EVAL_ROOT" || erc=$?
done
echo "$ds $rid f$fold rc=$erc secs=$(( $(date +%s)-t0 )) dev=$DEV end=$(date +%H:%M:%S)" > "$ST"
exit $erc
