#!/usr/bin/env bash
# FragNet PLANTED orchestrator — 3 datasets × ~50 DNF rules × 5 folds ≈ 750 units, per-layer attention.
# regime=planted: relabelled graphs carry the fired-clause cause as node_label → GT-ROC IS computed
# (vs the KNOWN planted cause) + Pearson + pred. Single-architecture FragNet → no backbone blow-up.
# RUN ON THE HPC. Only SUBMITS SLURM jobs; computes nothing itself. Everything is scheduled UPFRONT
# (one submission of all units) so queue position is held. Per-unit aftercorr chain:
#   dump_context(l2xgnn,CPU) → prep(fragnet,CPU) → finetune(fragnet,GPU) → export(fragnet,GPU) → eval(l2xgnn,CPU)
# Outputs to the fragnet track: fragnet/planted/<dataset>/<rule_id>/... ; transient pkls deleted post-export.
set -euo pipefail

# ── coordinates (verified) ───────────────────────────────────────────────────
FB=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/fragnet_baseline
VENDOR=$FB/vendor/FragNet
PT=$VENDOR/fragnet/exps/pt/unimol_exp1s4/pt.pt
SA=$FB/stage_a_fragnet_env
SB=$FB/stage_b_adapter
DATA_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS
PROC_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/processed_final_v2
VOCAB_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/vocab_final_v2
PV=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/planted_v2          # planted GT input
TRACK=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/fragnet
BASE=$TRACK/planted                                                    # sibling of base_runs
VOCAB=rbrics
CONDA=/nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh

GPU_PART=${GPU_PART:-preempt}
CPU_PART=${CPU_PART:-share}
BATCH_SIZE=${BATCH_SIZE:-256}
MAXCC=${MAXCC:-75}                # array concurrency — 75 GPUs (COMPUTE=gpu) or ~87 CPU slots (COMPUTE=cpu)
PREP_CORES=${PREP_CORES:-2}
COMPUTE=${COMPUTE:-gpu}           # gpu = finetune/export on GPU; cpu = on CPU (more slots available)
FT_CPU_CORES=${FT_CPU_CORES:-2}   # cores for finetune/export when COMPUTE=cpu
DATASETS="BBBP hERG Mutagenicity"     # all regime=planted, task=clf (DNF targets are binary)

SBD=$BASE/_sbatch
UNITS=$SBD/units.txt

write_units() {   # units = "dataset rule_id fold"; ONLY_DS/ONLY_RULE/ONLY_FOLD filters for dry runs
  mkdir -p "$SBD" "$BASE/_logs"
  : > "$UNITS"
  local folds=(0 1 2 3 4); [ -n "${ONLY_FOLD:-}" ] && folds=($ONLY_FOLD)
  for ds in $DATASETS; do
    [ -n "${ONLY_DS:-}" ] && [ "$ds" != "$ONLY_DS" ] && continue
    for rid in $(ls -1 "$PV/$ds" 2>/dev/null | grep -E "^dnf_"); do
      [ -n "${ONLY_RULE:-}" ] && [ "$rid" != "$ONLY_RULE" ] && continue
      for f in "${folds[@]}"; do echo "$ds $rid $f" >> "$UNITS"; done
    done
  done
  echo "[units] $(wc -l < "$UNITS") units -> $UNITS"
}

_preamble() {
  cat <<'PRE'
set -uo pipefail
read ds rid fold < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" __UNITS__)
UW=__BASE__/$ds/$rid/fold$fold
mkdir -p "$UW"
echo "=== unit: $ds $rid fold$fold (planted) -> $UW ==="
source __CONDA__
PRE
}

emit() {
  write_units
  local pre; pre=$(_preamble | sed "s#__UNITS__#$UNITS#; s#__BASE__#$BASE#; s#__CONDA__#$CONDA#")
  # finetune/export resources + device depend on COMPUTE (gpu vs cpu)
  local FT_PART FT_GRES FT_CORES FT_DEV EXP_DEV
  if [ "$COMPUTE" = "cpu" ]; then
    FT_PART=$CPU_PART; FT_GRES="gpu:0"; FT_CORES=$FT_CPU_CORES; FT_DEV=cpu; EXP_DEV=cpu
  else
    FT_PART=$GPU_PART; FT_GRES="gpu:1"; FT_CORES=2; FT_DEV=gpu; EXP_DEV=auto
  fi
  echo "[emit] COMPUTE=$COMPUTE -> finetune/export: -p $FT_PART --gres=$FT_GRES -c $FT_CORES (ft device=$FT_DEV, export device=$EXP_DEV)"

  cat > "$SBD/dump.sbatch" <<EOF
#!/bin/bash
#SBATCH -J fp_dump
#SBATCH -p $CPU_PART
#SBATCH --gres=gpu:0
#SBATCH -c 2
#SBATCH -t 1:00:00
#SBATCH -o $BASE/_logs/dump_%A_%a.log
$pre
conda activate l2xgnn
python $SB/align_and_aggregate.py dump_context --dataset "\$ds" --fold "\$fold" --vocab $VOCAB \\
  --regime planted --planted_root $PV --rule_id "\$rid" \\
  --data_root $DATA_ROOT --processed_root $PROC_ROOT --vocab_root $VOCAB_ROOT --out "\$UW/graph_context.json"
EOF

  cat > "$SBD/prep.sbatch" <<EOF
#!/bin/bash
#SBATCH -J fp_prep
#SBATCH -p $CPU_PART
#SBATCH --gres=gpu:0
#SBATCH -c $PREP_CORES
#SBATCH -t 12:00:00
#SBATCH -o $BASE/_logs/prep_%A_%a.log
$pre
conda activate fragnet
python $SA/prep_data.py --graph_context "\$UW/graph_context.json" --out_dir "\$UW" \\
  --vendor $VENDOR --frag_type custom
EOF

  cat > "$SBD/finetune.sbatch" <<EOF
#!/bin/bash
#SBATCH -J fp_ft
#SBATCH -p $FT_PART
#SBATCH --gres=$FT_GRES
#SBATCH -c $FT_CORES
#SBATCH -t 4:00:00
#SBATCH -o $BASE/_logs/ft_%A_%a.log
$pre
conda activate fragnet
python $SA/finetune_fragnet.py --work "\$UW" --pt_ckpt $PT --vendor $VENDOR \\
  --task clf --batch_size $BATCH_SIZE --device $FT_DEV
EOF

  cat > "$SBD/export.sbatch" <<EOF
#!/bin/bash
#SBATCH -J fp_exp
#SBATCH -p $FT_PART
#SBATCH --gres=$FT_GRES
#SBATCH -c $FT_CORES
#SBATCH -t 2:00:00
#SBATCH -o $BASE/_logs/export_%A_%a.log
$pre
conda activate fragnet
python $SA/export_frag_attention.py --work "\$UW" --vendor $VENDOR --device $EXP_DEV \\
  --graph_context "\$UW/graph_context.json" --out "\$UW/fragnet_frag_neutral.json" 2>&1 | grep -v "bond mask value"
rc=\${PIPESTATUS[0]}
if [ "\$rc" -eq 0 ] && [ -s "\$UW/fragnet_frag_neutral.json" ]; then
  rm -f "\$UW/train.pkl" "\$UW/val.pkl" "\$UW/test.pkl"    # transient — free per unit
  echo "[export] rc=0, neutral written, pkls cleaned"
else
  echo "[export] FAILED rc=\$rc"; exit 1
fi
EOF

  cat > "$SBD/eval.sbatch" <<EOF
#!/bin/bash
#SBATCH -J fp_eval
#SBATCH -p $CPU_PART
#SBATCH --gres=gpu:0
#SBATCH -c 2
#SBATCH -t 1:00:00
#SBATCH -o $BASE/_logs/eval_%A_%a.log
$pre
conda activate l2xgnn
for unk in include exclude; do
  echo "--- eval unk=\$unk ---"
  python $SB/eval_frag_attention.py --dataset "\$ds" --fold "\$fold" --vocab $VOCAB --unk "\$unk" \\
    --regime planted --planted_root $PV --rule_id "\$rid" \\
    --data_root $DATA_ROOT --processed_root $PROC_ROOT --vocab_root $VOCAB_ROOT \\
    --neutral "\$UW/fragnet_frag_neutral.json" --dest_root $BASE/\$ds/\$rid/eval
done
EOF

  echo "[emit] wrote 5 sbatch scripts -> $SBD"
}

submit() {   # UPFRONT: all units in one 5-phase aftercorr chain (per-unit isolation)
  emit
  local N A; N=$(wc -l < "$UNITS"); A="0-$((N-1))%$MAXCC"
  if [ "$N" -gt 1000 ]; then echo "WARN: $N > MaxArraySize 1001 — split by dataset (ONLY_DS)"; fi
  echo "[submit] array $A ($N units, upfront)"
  local jd jp jf je jv
  jd=$(sbatch --parsable --array="$A" "$SBD/dump.sbatch")
  jp=$(sbatch --parsable --dependency=aftercorr:"$jd" --array="$A" "$SBD/prep.sbatch")
  jf=$(sbatch --parsable --dependency=aftercorr:"$jp" --array="$A" "$SBD/finetune.sbatch")
  je=$(sbatch --parsable --dependency=aftercorr:"$jf" --array="$A" "$SBD/export.sbatch")
  jv=$(sbatch --parsable --dependency=aftercorr:"$je" --array="$A" "$SBD/eval.sbatch")
  echo "submitted: dump=$jd prep=$jp finetune=$jf export=$je eval=$jv"
}

phase()  { emit; local N A; N=$(wc -l < "$UNITS"); A="0-$((N-1))%$MAXCC"; sbatch --array="$A" "$SBD/${1:?usage: phase <dump|prep|finetune|export|eval>}.sbatch"; }
status() { squeue -u "$USER" -o '%.10i %.9P %.9j %.2t %.10M %.6D %R'; }
rollup() { python "$FB/rollup_base_runs.py" --base "$BASE"; }

case "${1:-help}" in
  emit)    emit ;;
  submit)  submit ;;
  phase)   phase "${2:-}" ;;
  status)  status ;;
  rollup)  rollup ;;
  *)
    echo "usage: $0 {emit|submit|phase <name>|status|rollup}" >&2
    echo "  emit   — write units.txt + 5 sbatch to $SBD (review; no submit)" >&2
    echo "  submit — emit + submit ALL units upfront (dump→prep→ft→export→eval, per-unit aftercorr)" >&2
    echo "  DRY RUN: ONLY_DS=BBBP ONLY_RULE=dnf_k1_r1 ONLY_FOLD=0 $0 submit  (1 unit)" >&2
    echo "  rollup — aggregate $BASE metrics + coverage/drops" >&2
    ;;
esac
