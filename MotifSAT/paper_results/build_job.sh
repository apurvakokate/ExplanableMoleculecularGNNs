#!/usr/bin/env bash
# build_job.sh — reduce the partials into the 3 tables. sbatch-a-script pattern.
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
export PYTHONPATH="$REPO"
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
DELIV="$REPO/motifsat_paper_deliverables_v1"
cd "$REPO/MotifSAT/paper_results"
python3 build_tables.py --partials "$DELIV/paper_tables/partials" --out "$DELIV/paper_tables"
