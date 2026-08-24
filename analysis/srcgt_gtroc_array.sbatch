#!/usr/bin/env bash
#SBATCH -J srcgt_arr
#SBATCH -p share,preempt
#SBATCH --gres=gpu:0
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH -t 4:00:00
#SBATCH --array=0-41
#SBATCH --requeue
#SBATCH --open-mode=append
set -uo pipefail
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh
conda activate l2xgnn
B=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
MR=$B/mose_replication
DEST=$B/eval_srcgt_gtroc_v1
FOLDS=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/
cd $B; export PYTHONPATH=$B WANDB_MODE=disabled OMP_NUM_THREADS=4
DS=(Benzene_Verified_GT Fluoride_Carbonyl_Verified_GT Alkane_Carbonyl_Verified_GT)
# build 42-combo list: unk | ds | group | tree
COMBOS=()
for U in exclude include; do for D in "${DS[@]}"; do
  COMBOS+=("$U|$D|posthoc|final_v2" "$U|$D|posthoc|final_v2_gath1"
           "$U|$D|gsat|final_v2" "$U|$D|gsat|final_v2_gath1"
           "$U|$D|mose|final_v2" "$U|$D|mose|ablated_completely_v1" "$U|$D|mose|final_v2_gath1")
done; done
IFS='|' read -r U D G CK <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"
DR=$DEST/unk-$U; mkdir -p $DR/_rollups
run(){ echo "== $1 $D $CK unk=$U $(date) =="
  python3 analysis/evaluate.py --method "$1" --dataset "$D" --vocab "$2" --gt_tier source --unk "$U" \
    --ckpt_root $MR/$CK $3 --dest_root $DR --dest $DR/_rollups/${1}_${D}_${CK}.csv \
    --data_root $FOLDS --vocab_root $B/vocab_final_v2 || echo "FAIL $1 $D $CK $U rc=$?"; }
case "$G" in
  posthoc) for M in gnnexplainer pgexplainer mage motif_occlusion; do run "$M" rbrics "--artifacts_root $MR/posthoc_v1"; done;;
  gsat)    run gsat rbrics "";;
  mose)    run mose rbrics_filter "";;
esac
echo "== TASK $SLURM_ARRAY_TASK_ID DONE $(date) =="
