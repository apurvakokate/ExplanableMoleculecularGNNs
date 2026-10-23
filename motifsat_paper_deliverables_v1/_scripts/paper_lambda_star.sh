#!/usr/bin/env bash
#SBATCH -J paper_lambda
#SBATCH -p share,preempt
#SBATCH --gres=gpu:0
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 11:00:00
#SBATCH --requeue
#SBATCH --array=0-9
#SBATCH -o /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/fmbpe_v4/lambda_star_paper_logs/%A_%a.out
# paper_lambda_star.sh — FM-BPE lambda* for the GSAT paper datasets in scope (EXPERIMENT_LOG §7.1):
# mutag folds 0-4 (per-fold vocabularies, as for our 8 datasets) + ogbg-molhiv / molbace / molbbbp /
# moltox21 / molclintox (one official OGB split = fold 0). Same settings as fmbpe_v3:
# --criterion cov_eps --eps_cov 0.25, defaults otherwise (start union, 8 refine iters, rings frozen).
# threshold_pct from SharedModules/data/threshold_config.py UNIFIED_FILTER_THRESHOLDS.
# Out: fmbpe_v4/lambda_star/<ds>.json (fold 0), fmbpe_v4/lambda_star/folds/<ds>_f<k>.json (k >= 1).
# Label-free (SMILES + split membership only). Submitted via sbatch stdin; this file is the record.
set -euo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
cd "$REPO/MotifBreakdown"; export PYTHONPATH="$REPO:$REPO/MotifBreakdown"
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
OGB=/nfs/hpc/share/kokatea/ChemIntuit/GSAT_upstream_6812f5c/data
CELLS=("mutag 0 $REPO/data 0.005" "mutag 1 $REPO/data 0.005" "mutag 2 $REPO/data 0.005"
       "mutag 3 $REPO/data 0.005" "mutag 4 $REPO/data 0.005"
       "ogbg-molhiv 0 $OGB 0.004" "ogbg-molbace 0 $OGB 0.005" "ogbg-molbbbp 0 $OGB 0.006"
       "ogbg-moltox21 0 $OGB 0.005" "ogbg-molclintox 0 $OGB 0.005")
read -r ds f root thr <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
OUT="$REPO/fmbpe_v4/lambda_star"; mkdir -p "$OUT/folds"
if [ "$f" = 0 ]; then out="$OUT/$ds.json"; else out="$OUT/folds/${ds}_f$f.json"; fi
[ -s "$out" ] && { echo "[done] $out"; exit 0; }
echo "[lambda*] $ds fold $f thr $thr root $root -> $out"
python3 fm_bpe_lambda_star.py --dataset "$ds" --data_root "$root" --fold "$f" \
    --threshold_pct "$thr" --criterion cov_eps --eps_cov 0.25 --out "$out"
