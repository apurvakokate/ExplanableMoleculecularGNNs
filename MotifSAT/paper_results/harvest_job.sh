#!/usr/bin/env bash
# harvest_job.sh — one (dataset, fold) harvest task. HDS/HFOLD come from the
# submitting env (run_harvest.sh sets them inline; sbatch --export=ALL forwards).
# sbatch-a-script pattern (this SLURM build rejects --wrap).
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
export PYTHONPATH="$REPO"
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
PR="$REPO/MotifSAT/paper_results"
DELIV="$REPO/motifsat_paper_deliverables_v1"
FOLDS_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS
: "${HDS:?set HDS}"; : "${HFOLD:?set HFOLD}"
cd "$PR"
python3 harvest.py --dataset "$HDS" --fold "$HFOLD" \
    --runs "$DELIV/base_runs" --vocab_root "$REPO/vocab_final_v2" \
    --data_root "$FOLDS_ROOT" --processed_root "$REPO/processed_final_v2" \
    --out "$DELIV/paper_tables/partials"
