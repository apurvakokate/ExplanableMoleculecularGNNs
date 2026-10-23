#!/usr/bin/env bash
#SBATCH -J variants_baseeval
#SBATCH -p share,preempt
#SBATCH --gres=gpu:0
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH --array=0-399%60
#SBATCH -o /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/motifsat_paper_deliverables_v1/_dispatch_variants/baseeval_logs/%A_%a.out
# variants_base_eval.sh — base_gsat for the variant campaign WITHOUT retraining: the r-sweep
# checkpoint at the chosen r (r_table.sh) is re-evaluated on each vocabulary, with
# --eval_only --load_weights_from --per_split_eval --occurrence_eval (training is motif-free
# and clip 2.0 = campaign setting, so the sweep model IS the base_gsat model).
# Task = (vocab, dataset, fold, backbone): 2 x 8 x 5 x 5 = 400.
# Out: variants_v1/<vocab>/base_gsat/none/<ds>/fold<f>/<bb>/ (+ source.json = checkpoint used).
# DONE = summary.json + summary_splits.json + occurrence_summary.json non-empty.
# Fails (rc 3) if the checkpoint loads with missing/unexpected keys (eval_only is non-strict).
# Submitted through sbatch stdin by the launcher; r_for() is filled in from r_table.sh.
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
OUT="$REPO/motifsat_paper_deliverables_v1"
FOLDS_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/
DSS=(Benzene_Verified_GT Alkane_Carbonyl_Verified_GT Fluoride_Carbonyl_Verified_GT BBBP hERG Mutagenicity esol Lipophilicity)
BBS=(GIN GCN GAT SAGE PNA)
R_TABLE_FILLED_AT_LAUNCH
i=${SLURM_ARRAY_TASK_ID}
v=$(( i / 200 )); r=$(( i % 200 )); ds=${DSS[$(( r / 25 ))]}; r=$(( r % 25 )); f=$(( r / 5 )); bb=${BBS[$(( r % 5 ))]}
if [ "$v" = 0 ]; then vocab=rbrics_v3; vroot="$REPO/vocab_final_v3"; var=rbrics
else vocab=fmbpe_v4; vroot="$REPO/fmbpe_v4/vocab/fold$f"; var=fm_bpe_full_union; fi
rr="$(r_for "$ds" "$bb")" || { echo "[skip] no r for $ds/$bb (pending)"; exit 0; }
fr=${rr% *}; di=${rr#* }
src="$OUT/rsweep_runs/r${fr}_i${di}/$ds/fold$f/$bb"
dir="$OUT/variants_v1/$vocab/base_gsat/none/$ds/fold$f/$bb"
if [ -s "$dir/summary.json" ] && [ -s "$dir/summary_splits.json" ] && [ -s "$dir/occurrence_summary.json" ]; then
    echo "[done] $dir"; exit 0; fi
[ -s "$src/best_model.pt" ] || { echo "[FAIL] no sweep checkpoint $src"; exit 2; }
mkdir -p "$dir"
printf '{"source_run": "%s", "final_r": %s, "decay_interval": %s, "vocab": "%s"}\n' "$src" "$fr" "$di" "$vocab" > "$dir/source.json"
echo "[eval] $vocab $ds fold$f $bb <- $src"
python3 MotifSAT/run.py --config MotifSAT/configs/base_gsat.yaml \
    --dataset "$ds" --fold "$f" --backbone "$bb" \
    --data_root "$FOLDS_ROOT" --vocab_root "$vroot" --vocab_variant "$var" \
    --processed_root "$REPO/processed_variants_v1/$vocab" \
    --info_loss_coef 1 --init_r 0.9 --decay_r 0.1 --final_r "$fr" --decay_interval "$di" \
    --ckpt_rule gsat --out_dir "$dir" --final_out_dir --per_split_eval --occurrence_eval \
    --eval_only --load_weights_from "$src/best_model.pt" > "$dir/eval.log" 2>&1
rc=$?
if grep -q "\[eval_only\] loaded with" "$dir/eval.log"; then
    echo "[FAIL] non-strict load: $(grep '\[eval_only\] loaded with' "$dir/eval.log")"; exit 3; fi
[ "$rc" = 0 ] && [ -s "$dir/summary_splits.json" ] && [ -s "$dir/occurrence_summary.json" ] \
    || { echo "[FAIL rc=$rc] $dir"; tail -20 "$dir/eval.log"; exit 1; }
echo "[ok] $dir"
