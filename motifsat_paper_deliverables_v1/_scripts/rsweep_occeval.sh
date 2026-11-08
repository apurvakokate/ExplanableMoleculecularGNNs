#!/usr/bin/env bash
#SBATCH -J rsw_occ
#SBATCH -p share,preempt
#SBATCH --gres=gpu:0
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH -o /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/motifsat_paper_deliverables_v1/_dispatch_rsweep/occ_logs/%A_%a.out
# rsweep_occeval.sh — add the OCCURRENCE outputs to the base-GSAT r sweep WITHOUT retraining.
#
# WHY: rsweep_runs holds base GSAT at the five r schedules with best_model.pt, but it predates
# --occurrence_eval, so it has no occurrence_<split>.csv. Those files are the only place the
# per-(graph, motif, copy) SCORE and IMPACT are stored together with a graph index, which is what
# the within-molecule instance Pearson needs. The impact caches on disk are mask-ALL at
# (motif, graph) granularity, so they cannot substitute.
#
# Eval-only is valid here: best_model.pt is written once after training exits
# (MotifSAT/train.py), so its presence proves the loop completed and the GSAT-eligible epoch was
# selected. Verified 2026-11-07 to reproduce a full run's metrics to ~1e-6.
#
# VOCAB: evaluated with the values variants_v2 used for base_gsat (vocab_final_v3 / rbrics /
# processed_variants_v1/rbrics_v3/rbrics), NOT rsweep's original processed_final_v2/rbrics, so the
# numbers are directly comparable to the variant cells the selected schedule is applied to. The
# base-GSAT weights are vocabulary-independent (motif_method=none, noise=none): the vocabulary
# only groups atoms for the evaluation.
#
# ARMS: the five r-grid points. r0.5_i10_clip0 is EXCLUDED -- it is a gradient-clipping arm, not
# an r-grid point.
#
# One array task per cell, indexed into a sorted glob so the mapping is deterministic.
# Idempotent: a cell that already has occurrence_summary.json is skipped.
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
export MOSE_GAT_HEADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""

OUT="$REPO/motifsat_paper_deliverables_v1"
SWEEP="$OUT/rsweep_runs"
FOLDS_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/
VROOT="$REPO/vocab_final_v3"
PROC="$REPO/processed_variants_v1/rbrics_v3/rbrics"
DISPATCH="$OUT/_dispatch_rsweep"; FAILURES="$DISPATCH/occ_failures.tsv"
mkdir -p "$DISPATCH/occ_logs"
[ -s "$FAILURES" ] || printf 'ts\thost\tjobid\tarm\tdataset\tfold\tbackbone\trc\n' > "$FAILURES"

# deterministic cell list: the five r-grid arms only
mapfile -t CELLS < <(for arm in r0.5_i10 r0.5_i20 r0.7_i10 r0.7_i20 r0.9_i10; do
    find "$SWEEP/$arm" -name best_model.pt 2>/dev/null
done | sort)
N=${#CELLS[@]}
i="${SLURM_ARRAY_TASK_ID:-0}"
echo "[cells] $N total; this task index $i"
if [ "$i" -ge "$N" ]; then echo "[skip] index $i >= $N"; exit 0; fi

ckpt="${CELLS[$i]}"
dir="$(dirname "$ckpt")"
bb="$(basename "$dir")"
fold="$(basename "$(dirname "$dir")")"
ds="$(basename "$(dirname "$(dirname "$dir")")")"
arm="$(basename "$(dirname "$(dirname "$(dirname "$dir")")")")"
f="${fold#fold}"
# r schedule from the arm directory name: r<final_r>_i<decay_interval>
fr="${arm#r}"; fr="${fr%%_i*}"
di="${arm##*_i}"
echo "[cell] arm=$arm ds=$ds fold=$f bb=$bb final_r=$fr decay_interval=$di"

if [ -s "$dir/occurrence_summary.json" ]; then echo "[done] $dir"; exit 0; fi
[ -s "$ckpt" ] || { echo "[FAIL] no checkpoint $ckpt"; exit 2; }

python3 MotifSAT/run.py --config MotifSAT/configs/base_gsat.yaml \
    --dataset "$ds" --fold "$f" --backbone "$bb" \
    --data_root "$FOLDS_ROOT" --vocab_root "$VROOT" --vocab_variant rbrics \
    --processed_root "$PROC" \
    --info_loss_coef 1 --init_r 0.9 --decay_r 0.1 --final_r "$fr" --decay_interval "$di" \
    --ckpt_rule gsat --clip_grad 2.0 \
    --out_dir "$dir" --final_out_dir --per_split_eval --occurrence_eval \
    --epochs 500 --patience 50 \
    --eval_only --load_weights_from "$ckpt" > "$dir/occeval.log" 2>&1
rc=$?

# A non-strict load means the checkpoint does not match this architecture: reject the cell.
if grep -q "\[eval_only\] loaded with" "$dir/occeval.log"; then
    echo "[FAIL] non-strict load: $(grep '\[eval_only\] loaded with' "$dir/occeval.log")"; rc=3
fi
if [ "$rc" = 0 ] && [ -s "$dir/occurrence_summary.json" ] && [ -s "$dir/occurrence_valid.csv" ]; then
    echo "[ok] $arm/$ds/fold$f/$bb  files=$(ls "$dir" | wc -l)"; exit 0
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date +%s)" "$(hostname -s)" "${SLURM_JOB_ID:-$$}" "$arm" "$ds" "$f" "$bb" "$rc" >> "$FAILURES"
echo "[FAIL rc=$rc] $arm/$ds/fold$f/$bb"; tail -20 "$dir/occeval.log"; exit 1
