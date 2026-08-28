#!/usr/bin/env bash
# One cell of the SFO-vs-rBRICS post-hoc + GSAT reuse run. Args: VOCAB(sfo|rbrics) DATASET FOLD BACKBONE
# Writes unique per-cell rollup CSVs (no append races) + an atomic per-cell .status file (rc + secs).
set -uo pipefail
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh
conda activate l2xgnn
VOCAB=$1; D=$2; F=$3; BB=$4
P=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
EXP=$P/fragmentation_v2
DATA=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS
OUT=$EXP/eval_reuse; TRK=$EXP/_tracking_reuse; ART=$EXP/eval_reuse_art
mkdir -p $OUT/$VOCAB $TRK $ART
export PYTHONPATH=$P WANDB_MODE=disabled
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CAP="python3 $EXP/_deploy/run_capped.py"
STATUS=$TRK/${VOCAB}_${D}_f${F}_${BB}.status
T0=$(date +%s); rc_all=0
case "$D" in
  Benzene_Verified_GT|Alkane_Carbonyl_Verified_GT|Fluoride_Carbonyl_Verified_GT) TIER=source;;
  *) TIER=none;;
esac
ev() { $CAP $P/analysis/evaluate.py "$@"; }

if [ "$VOCAB" = "sfo" ]; then
  VR=$EXP/vocab/fold$F; PR=$EXP/processed/fold$F
  SFO=$EXP/models/baselines/$D/fold$F/size_frequency_optimization/bb-${BB}_enc-onehot_norm-none_real
  if [ "$BB" = "GAT" ]; then REUSE=$P/mose_replication/final_v2_gath1/baselines/$D/fold$F/rbrics/bb-GAT_enc-onehot_norm-none_real; GCK=$P/final_v2_gath1
  else REUSE=$P/posthoc_v1/baselines/$D/fold$F/rbrics/bb-${BB}_enc-onehot_norm-none_real; GCK=$P/final_v2; fi
  $CAP $P/analysis/eval_driver_posthoc.py --out_root $EXP/models --data_root $DATA \
    --vocab_root $VR --processed_root $PR --dest_root $EXP/posthoc --device cpu --dataset $D \
    --only $SFO --reuse_atts_dir $REUSE --methods gnnexplainer pgexplainer motif_occlusion --no_skip_done
  rc=$?; [ $rc -ne 0 ] && rc_all=$rc
  for M in gnnexplainer pgexplainer motif_occlusion; do
    ev --method $M --dataset $D --vocab size_frequency_optimization --gt_tier $TIER --unk exclude \
      --ckpt_root $EXP/models --artifacts_root $EXP/posthoc --dest_root $ART \
      --dest $OUT/sfo/${M}_${D}_f${F}_${BB}.csv --data_root $DATA --vocab_root $VR --processed_root $PR \
      --fold $F --backbone $BB --device cpu; rc=$?; [ $rc -ne 0 ] && rc_all=$rc
  done
  ev --method gsat --dataset $D --vocab size_frequency_optimization --weight_vocab rbrics --gt_tier $TIER \
    --unk exclude --ckpt_root $GCK --dest_root $ART --dest $OUT/sfo/gsat_${D}_f${F}_${BB}.csv \
    --data_root $DATA --vocab_root $VR --processed_root $PR --fold $F --backbone $BB --device cpu
  rc=$?; [ $rc -ne 0 ] && rc_all=$rc
else
  VR=$P/vocab_final_v2; PR=$P/processed_final_v2
  if [ "$BB" = "GAT" ]; then CK=$P/mose_replication/final_v2_gath1; ACK=$P/mose_replication/eval_gath1; GCK=$P/final_v2_gath1
  else CK=$P/mose_replication/final_v2; ACK=$P/posthoc_v1; GCK=$P/final_v2; fi
  for M in gnnexplainer pgexplainer motif_occlusion; do
    ev --method $M --dataset $D --vocab rbrics --gt_tier $TIER --unk exclude \
      --ckpt_root $CK --artifacts_root $ACK --dest_root $ART \
      --dest $OUT/rbrics/${M}_${D}_f${F}_${BB}.csv --data_root $DATA --vocab_root $VR --processed_root $PR \
      --fold $F --backbone $BB --device cpu; rc=$?; [ $rc -ne 0 ] && rc_all=$rc
  done
  ev --method gsat --dataset $D --vocab rbrics --gt_tier $TIER --unk exclude --ckpt_root $GCK \
    --dest_root $ART --dest $OUT/rbrics/gsat_${D}_f${F}_${BB}.csv \
    --data_root $DATA --vocab_root $VR --processed_root $PR --fold $F --backbone $BB --device cpu
  rc=$?; [ $rc -ne 0 ] && rc_all=$rc
fi
T1=$(date +%s)
echo "${VOCAB} ${D} f${F} ${BB} rc=${rc_all} secs=$((T1-T0)) end=$(date +%H:%M:%S)" > $STATUS
exit $rc_all
