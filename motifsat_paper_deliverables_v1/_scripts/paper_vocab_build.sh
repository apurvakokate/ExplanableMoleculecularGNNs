#!/usr/bin/env bash
#SBATCH -J paper_vocab
#SBATCH -p share,preempt
#SBATCH --gres=gpu:0
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 11:00:00
#SBATCH --requeue
#SBATCH --array=0-31
#SBATCH -o /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/vocab_final_v3/_logs/paper_vocab_%A_%a.out
# paper_vocab_build.sh — vocabularies for the GSAT paper datasets in scope (EXPERIMENT_LOG §7.1):
# mutag + ogbg-molhiv / molbace / molbbbp / moltox21 / molclintox. Same generator and parameters as
# the v3/v4 rebuild of our 8 datasets (fixed rBRICS stamping §3.5, per-copy sidecar), new datasets:
#   0-11   rBRICS  ds(6) x {rbrics, rbrics_filter}, fold 0      -> vocab_final_v3/<ds>/<variant>/
#   12-21  FM-BPE  mutag fold(5) x {unfiltered, _filter}        -> fmbpe_v4/vocab/fold<k>/mutag/
#   22-31  FM-BPE  OGB ds(5) x {unfiltered, _filter}, fold 0    -> fmbpe_v4/vocab/fold0/<ds>/
# threshold_pct = SharedModules/data/threshold_config.py UNIFIED_FILTER_THRESHOLDS; FM-BPE lambda =
# lambda_star from fmbpe_v4/lambda_star/{<ds>.json | folds/<ds>_f<k>.json} (paper_lambda_star.sh).
# CSVs: mutag -> $P/data/mutag_<k>.csv; OGB -> GSAT_upstream_6812f5c/data/<ds>_0.csv (official split).
# Submitted via sbatch stdin; this file is the record.
set -uo pipefail
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh
conda activate l2xgnn
P=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
OGB=/nfs/hpc/share/kokatea/ChemIntuit/GSAT_upstream_6812f5c/data
export PYTHONPATH=$P OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
DS=(mutag ogbg-molhiv ogbg-molbace ogbg-molbbbp ogbg-moltox21 ogbg-molclintox)
OGBDS=(ogbg-molhiv ogbg-molbace ogbg-molbbbp ogbg-moltox21 ogbg-molclintox)
i=$SLURM_ARRAY_TASK_ID
die() { echo "FATAL: $*" >&2; exit 3; }
root_of() { [ "$1" = mutag ] && echo "$P/data" || echo "$OGB"; }
pct_of() { python3 -c "import sys;sys.path.insert(0,'$P/SharedModules');from data.threshold_config import UNIFIED_FILTER_THRESHOLDS as T;print(T['$1'])"; }
cd $P/MotifBreakdown
if [ $i -lt 12 ]; then
  ds=${DS[$((i/2))]}; OUT=$P/vocab_final_v3; DATA=$(root_of $ds)
  if [ $((i%2)) -eq 0 ]; then V=rbrics; EXTRA=""
  else V=rbrics_filter; pct=$(pct_of $ds) || die "pct $ds"; EXTRA="--variant rbrics_filter --apply_threshold --threshold_pct $pct"; fi
  [ -s "$OUT/$ds/$V/${ds}_${V}_mol_fragment_copies.pickle" ] && { echo "SKIP $ds $V"; exit 0; }
  python3 -u generate_vocab_rules.py --datasets $ds --data_root $DATA --fold 0 --method rbrics --out_dir $OUT $EXTRA
  rc=$?; echo "VOCAB_DONE rbrics $ds $V rc=$rc"; exit $rc
fi
if [ $i -lt 22 ]; then j=$((i-12)); ds=mutag; fold=$((j/2))
else j=$((i-22)); ds=${OGBDS[$((j/2))]}; fold=0; fi
ARM=fm_bpe_full_union; DATA=$(root_of $ds)
if [ "$fold" = 0 ]; then LJ=$P/fmbpe_v4/lambda_star/$ds.json; else LJ=$P/fmbpe_v4/lambda_star/folds/${ds}_f$fold.json; fi
[ -s "$LJ" ] || { echo "LAMBDA_PENDING $LJ"; exit 4; }
lam=$(python3 -c "import json;d=json.load(open('$LJ'));assert d.get('accepted'),d.get('status');print(repr(d['lambda_star']))") || die "lambda $LJ"
pct=$(pct_of $ds) || die "pct $ds"
OUT=$P/fmbpe_v4/vocab/fold$fold
if [ $((j%2)) -eq 0 ]; then V=$ARM; EXTRA=""; else V=${ARM}_filter; EXTRA="--apply_threshold --threshold_pct $pct"; fi
[ -s "$OUT/$ds/$V/${ds}_${V}_mol_fragment_copies.pickle" ] && { echo "SKIP $ds f$fold $V"; exit 0; }
python3 -u generate_vocab_rules.py --datasets $ds --data_root $DATA --out_dir $OUT --fold $fold \
    --method $ARM --variant $V $EXTRA --fm_lambda $lam
rc=$?; echo "VOCAB_DONE fmbpe $ds f$fold $V lam=$lam rc=$rc"; exit $rc
