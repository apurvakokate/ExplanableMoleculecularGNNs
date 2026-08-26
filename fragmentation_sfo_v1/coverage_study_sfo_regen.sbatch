#!/usr/bin/env bash
# =============================================================================
# coverage_study_sfo_regen.sbatch   (fragmentation_sfo_v1)
#   Regenerate the SFO vocabulary at EACH threshold, per dataset, for the
#   threshold-vs-coverage study.  SFO's S_min gates the partition (eligibility,
#   xcover/exp1_candidate_inventory.py), so the fragments themselves change with
#   the threshold -> one regen per (dataset, threshold).  SFO_SMIN_PCT overrides
#   S_min; SFO_WORKERS=1 keeps each task single-core (1 core / setup).
#
#   Array index -> (dataset, threshold):  idx//6 = dataset, idx%6 = threshold.
#   8 datasets x 6 thresholds = 48 tasks.
#
#   Submit:  sbatch fragmentation_sfo_v1/coverage_study_sfo_regen.sbatch
#   (mkdir -p the log dir first — see EXECUTION.md)
# =============================================================================
#SBATCH -J sfo_cov
#SBATCH -p preempt
#SBATCH --gres=gpu:0
#SBATCH -c 1
#SBATCH -t 10:00:00
#SBATCH --array=0-47
#SBATCH -o /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/fragmentation_sfo_v1/_logs/sfo_cov_%a.out
set -uo pipefail
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh
conda activate l2xgnn

P=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
FOLDS=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS
OUT=$P/fragmentation_sfo_v1/sfo_vocabs
export PYTHONPATH=$P

DATASETS=(BBBP hERG Mutagenicity esol Lipophilicity \
          Benzene_Verified_GT Alkane_Carbonyl_Verified_GT Fluoride_Carbonyl_Verified_GT)
THRS=(0.001 0.005 0.01 0.02 0.03 0.05)

i=$SLURM_ARRAY_TASK_ID
ds=${DATASETS[$((i / 6))]}
thr=${THRS[$((i % 6))]}
dest=$OUT/thr${thr}

# .done sentinel so a preempted array can be resubmitted without redoing finished cells
sentinel=$dest/$ds/size_frequency_optimization/.done
if [ -f "$sentinel" ]; then
  echo "SKIP ds=$ds thr=$thr (already .done)"; exit 0
fi

# S_min = this threshold; single-core; dataset name for the S_min lookup fallback
export SFO_SMIN_PCT=$thr SFO_WORKERS=1 SFO_DATASET=$ds
echo "=== SFO regen  ds=$ds  thr=$thr  (task $i)  $(date) ==="
cd $P/MotifBreakdown
python3 generate_vocab_rules.py \
    --datasets "$ds" --data_root "$FOLDS" --fold 0 \
    --method size_frequency_optimization \
    --out_dir "$dest" \
  && [ -f "$dest/$ds/size_frequency_optimization/matrix_columns.csv" ] \
  && touch "$sentinel" \
  && echo "OK ds=$ds thr=$thr" \
  || echo "FAIL ds=$ds thr=$thr"
