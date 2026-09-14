#!/usr/bin/env bash
# FragNet base_runs orchestrator — 8 datasets × 5 folds = 40 units, per-layer attention.
#   source datasets → GT-ROC + Pearson + pred ; none datasets → Pearson + pred only (no GT-ROC).
#   task (clf/regr) is AUTO from graph_context __meta__.task_type (esol/Lipophilicity → regression).
# RUN ON THE HPC. This script only SUBMITS SLURM jobs (via `submit`); it computes nothing itself.
# Phases per unit, chained per-unit with SLURM aftercorr (a unit's failure never blocks the others):
#   dump_context(l2xgnn,CPU) → prep(fragnet,CPU) → finetune(fragnet,GPU) → export(fragnet,GPU) → eval(l2xgnn,CPU)
# Everything lands under the fragnet track folder; pkls are deleted after export. NEVER touches
# _fragnet_poc_scratch or mose_replication_v2.
set -euo pipefail

# ── coordinates (verified 2026-09-13) ───────────────────────────────────────────
FB=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/fragnet_baseline
VENDOR=$FB/vendor/FragNet
PT=$VENDOR/fragnet/exps/pt/unimol_exp1s4/pt.pt
SA=$FB/stage_a_fragnet_env
SB=$FB/stage_b_adapter
DATA_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS
PROC_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/processed_final_v2
VOCAB_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/vocab_final_v2
TRACK=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/fragnet     # NEW track (sibling of mose_replication_v2)
BASE=$TRACK/base_runs
VOCAB=rbrics
CONDA=/nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh

GPU_PART=${GPU_PART:-preempt}     # GPU phases
CPU_PART=${CPU_PART:-share}       # CPU phases (share/eecs dodge the GPU-group CPU quota; preempt starves CPU arrays)
BATCH_SIZE=${BATCH_SIZE:-256}
MAXCC=${MAXCC:-40}                # SLURM array concurrency cap (40 GPUs)
PREP_CORES=${PREP_CORES:-4}       # conformer generation is the CPU pole; GPU phases stay at 2 (HARD rule)

# dataset:regime:task  (task = clf | regr; only esol/Lipophilicity are regression)
DATASETS=(
  "Benzene_Verified_GT:source:clf"
  "Alkane_Carbonyl_Verified_GT:source:clf"
  "Fluoride_Carbonyl_Verified_GT:source:clf"
  "Mutagenicity:none:clf"
  "BBBP:none:clf"
  "hERG:none:clf"
  "esol:none:regr"
  "Lipophilicity:none:regr"
)
FOLDS_DEFAULT=(0 1 2 3 4)

SBD=$BASE/_sbatch
UNITS=$SBD/units.txt

write_units() {   # ONLY_DS / ONLY_FOLD env filters allow a 1-unit dry run
  mkdir -p "$SBD" "$BASE/_logs"
  : > "$UNITS"
  local folds=("${FOLDS_DEFAULT[@]}"); [ -n "${ONLY_FOLD:-}" ] && folds=($ONLY_FOLD)
  for dr in "${DATASETS[@]}"; do
    local ds reg task; IFS=: read -r ds reg task <<< "$dr"
    [ -n "${ONLY_DS:-}" ] && [ "$ds" != "$ONLY_DS" ] && continue
    for f in "${folds[@]}"; do echo "$ds $reg $task $f" >> "$UNITS"; done
  done
  echo "[units] $(wc -l < "$UNITS") units -> $UNITS"
}

_preamble() {   # emitted verbatim into each sbatch: resolve this array task's unit
  cat <<'PRE'
set -uo pipefail
read ds reg task fold < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" __UNITS__)
UW=__BASE__/$reg/$ds/fold$fold
mkdir -p "$UW"
echo "=== unit: $ds fold$fold regime=$reg -> $UW ==="
source __CONDA__
PRE
}

emit() {
  write_units
  local pre; pre=$(_preamble | sed "s#__UNITS__#$UNITS#; s#__BASE__#$BASE#; s#__CONDA__#$CONDA#")

  cat > "$SBD/dump.sbatch" <<EOF
#!/bin/bash
#SBATCH -J fn_dump
#SBATCH -p $CPU_PART
#SBATCH --gres=gpu:0
#SBATCH -c 2
#SBATCH -t 1:00:00
#SBATCH -o $BASE/_logs/dump_%A_%a.log
$pre
conda activate l2xgnn
python $SB/align_and_aggregate.py dump_context --dataset "\$ds" --fold "\$fold" --vocab $VOCAB --regime "\$reg" \\
  --data_root $DATA_ROOT --processed_root $PROC_ROOT --vocab_root $VOCAB_ROOT --out "\$UW/graph_context.json"
EOF

  cat > "$SBD/prep.sbatch" <<EOF
#!/bin/bash
#SBATCH -J fn_prep
#SBATCH -p $CPU_PART
#SBATCH --gres=gpu:0
#SBATCH -c $PREP_CORES
#SBATCH -t 10:00:00
#SBATCH -o $BASE/_logs/prep_%A_%a.log
$pre
conda activate fragnet
python $SA/prep_data.py --graph_context "\$UW/graph_context.json" --out_dir "\$UW" \\
  --vendor $VENDOR --frag_type custom
EOF

  cat > "$SBD/finetune.sbatch" <<EOF
#!/bin/bash
#SBATCH -J fn_ft
#SBATCH -p $GPU_PART
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -t 4:00:00
#SBATCH -o $BASE/_logs/ft_%A_%a.log
$pre
conda activate fragnet
python $SA/finetune_fragnet.py --work "\$UW" --pt_ckpt $PT --vendor $VENDOR \\
  --task "\$task" --batch_size $BATCH_SIZE
EOF

  cat > "$SBD/export.sbatch" <<EOF
#!/bin/bash
#SBATCH -J fn_exp
#SBATCH -p $GPU_PART
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -t 2:00:00
#SBATCH -o $BASE/_logs/export_%A_%a.log
$pre
conda activate fragnet
python $SA/export_frag_attention.py --work "\$UW" --vendor $VENDOR \\
  --graph_context "\$UW/graph_context.json" --out "\$UW/fragnet_frag_neutral.json" 2>&1 | grep -v "bond mask value"
rc=\${PIPESTATUS[0]}
if [ "\$rc" -eq 0 ] && [ -s "\$UW/fragnet_frag_neutral.json" ]; then
  rm -f "\$UW/train.pkl" "\$UW/val.pkl" "\$UW/test.pkl"    # pkls transient — free ~0.3-0.7GB/fold
  echo "[export] rc=0, neutral written, pkls cleaned"
else
  echo "[export] FAILED rc=\$rc"; exit 1
fi
EOF

  cat > "$SBD/eval.sbatch" <<EOF
#!/bin/bash
#SBATCH -J fn_eval
#SBATCH -p $CPU_PART
#SBATCH --gres=gpu:0
#SBATCH -c 2
#SBATCH -t 1:00:00
#SBATCH -o $BASE/_logs/eval_%A_%a.log
$pre
conda activate l2xgnn
for unk in include exclude; do
  echo "--- eval unk=\$unk ---"
  python $SB/eval_frag_attention.py --dataset "\$ds" --fold "\$fold" --vocab $VOCAB --unk "\$unk" --regime "\$reg" \\
    --data_root $DATA_ROOT --processed_root $PROC_ROOT --vocab_root $VOCAB_ROOT \\
    --neutral "\$UW/fragnet_frag_neutral.json" --dest_root $BASE/\$reg/\$ds/eval
done
EOF

  echo "[emit] wrote 5 sbatch scripts -> $SBD"
}

submit() {   # full chain, per-unit aftercorr so one unit's failure doesn't block the others
  emit
  local N A; N=$(wc -l < "$UNITS"); A="0-$((N-1))%$MAXCC"
  echo "[submit] array $A ($N units)"
  local jd jp jf je jv
  jd=$(sbatch --parsable --array="$A" "$SBD/dump.sbatch")
  jp=$(sbatch --parsable --dependency=aftercorr:"$jd" --array="$A" "$SBD/prep.sbatch")
  jf=$(sbatch --parsable --dependency=aftercorr:"$jp" --array="$A" "$SBD/finetune.sbatch")
  je=$(sbatch --parsable --dependency=aftercorr:"$jf" --array="$A" "$SBD/export.sbatch")
  jv=$(sbatch --parsable --dependency=aftercorr:"$je" --array="$A" "$SBD/eval.sbatch")
  echo "submitted: dump=$jd prep=$jp finetune=$jf export=$je eval=$jv"
}

phase() {   # resubmit ONE phase standalone (no deps) after a fix, e.g.: $0 phase eval
  emit
  local N A; N=$(wc -l < "$UNITS"); A="0-$((N-1))%$MAXCC"
  sbatch --array="$A" "$SBD/$1.sbatch"
}

status()      { squeue -u "$USER" -o '%.10i %.9P %.9j %.2t %.10M %.6D %R'; }
rollup()      { python "$FB/rollup_base_runs.py" --base "$BASE"; }
clean_scratch() { echo "[clean] rm -rf /nfs/stak/users/kokatea/hpc-share/ChemIntuit/Claude+Cursor/_fragnet_poc_scratch";
                  rm -rf /nfs/stak/users/kokatea/hpc-share/ChemIntuit/Claude+Cursor/_fragnet_poc_scratch; }

case "${1:-help}" in
  emit)          emit ;;
  submit)        submit ;;
  phase)         phase "${2:?usage: $0 phase {dump|prep|finetune|export|eval}}" ;;
  status)        status ;;
  rollup)        rollup ;;
  clean_scratch) clean_scratch ;;
  *)
    echo "usage: $0 {emit|submit|phase <name>|status|rollup|clean_scratch}" >&2
    echo "  emit    — write units.txt + 5 sbatch scripts to $SBD (no submit; review them)" >&2
    echo "  submit  — emit + submit the full per-unit chain (dump→prep→finetune→export→eval)" >&2
    echo "  phase   — resubmit ONE phase standalone after a fix" >&2
    echo "  rollup  — aggregate per-unit metrics into $BASE/rollup.csv + coverage report" >&2
    echo "  DRY RUN: ONLY_DS=Benzene_Verified_GT ONLY_FOLD=0 $0 submit   (1 unit end-to-end)" >&2
    echo "  clean_scratch — rm -rf _fragnet_poc_scratch (AFTER experiments validate)" >&2
    ;;
esac
