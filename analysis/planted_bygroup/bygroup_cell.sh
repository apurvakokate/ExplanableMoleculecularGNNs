#!/usr/bin/env bash
# One planted task: DS RULE GROUP(native|posthoc).  Same invocation as
# fragmentation_v2/_deploy/gtroc_cell.sh (the run behind the planted GT-ROC tables), with
# analysis/evaluate_gtroc_bygroup.py in place of evaluate_gtroc.py and a NEW output tree.
# Pure READ of the persisted per-atom scores; writes only under mose_replication_v2/gtroc_bygroup.
# SKIP-IF-DONE: each per-method CSV gets a <csv>.done sentinel on a clean rc==0 run with rows.
set -uo pipefail
DS=$1; RULE=$2; GROUP=$3
P=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
DATA=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
export PYTHONPATH=$P WANDB_MODE=disabled OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
OUT=$P/mose_replication_v2/gtroc_bygroup; TRK=$OUT/_tracking/planted
CK=$P/planted_v2/$DS/$RULE; VR=$P/planted_v2/$DS/_shared/vocab; PR=$P/planted_v2/$DS/_shared/processed
D=$OUT/planted/$DS/$RULE; mkdir -p $D $TRK
G="python3 -u $P/analysis/evaluate_gtroc_bygroup.py --dataset $DS --gt_tier planted --ckpt_root $CK \
  --data_root $DATA --vocab_root $VR --processed_root $PR --device cpu"
T0=$(date +%s); rc_all=0; nskip=0
run_one(){
  local dest=$1; shift
  if [ -f "$dest.done" ]; then echo "SKIP $(basename $dest) (done)"; nskip=$((nskip+1)); return; fi
  $G "$@"; local r=$?
  if [ "$r" -eq 0 ] && [ -s "$dest" ] && [ "$(grep -vc '^tier,' "$dest")" -gt 0 ]; then
    touch "$dest.done"
  else
    rc_all=$r; rm -f "$dest.done"
  fi
}
if [ "$GROUP" = native ]; then
  run_one $D/mose_fixed.csv --method mose --vocab rbrics_filter --unk_mode fixed --artifacts_root $CK/eval/artifacts --dest $D/mose_fixed.csv
  run_one $D/gsat.csv       --method gsat --vocab rbrics                         --artifacts_root $CK/eval/artifacts --dest $D/gsat.csv
else
  for M in gnnexplainer pgexplainer motif_occlusion mage; do
    run_one $D/$M.csv --method $M --vocab rbrics --artifacts_root $CK/posthoc_gtroc --dest $D/$M.csv
  done
fi
T1=$(date +%s)
rows=$(cat $D/*.csv 2>/dev/null | grep -vc '^tier,' || echo 0)
echo "planted $DS $RULE $GROUP rc=$rc_all rows=$rows skip=$nskip secs=$((T1-T0)) end=$(date +%H:%M:%S)" > $TRK/${DS}_${RULE}_${GROUP}.status
exit $rc_all
