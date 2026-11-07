#!/usr/bin/env bash
#SBATCH -J v2_spass_eval
#SBATCH -p gpu,preempt
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH -o /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/motifsat_paper_deliverables_v1/_dispatch_variants/secondpass_logs/%A_%a.out
# variants_secondpass_eval.sh — second pass for the variants_v2 cells whose TRAINING finished
# (best_model.pt + history.json on disk) but whose downstream evaluation never ran, so
# occurrence_summary.json is missing. Re-runs ONLY the eval pipeline against the cell's OWN
# checkpoint: --eval_only --load_weights_from <cell>/best_model.pt --per_split_eval --occurrence_eval.
#
# WHY this is not a retrain: best_model.pt is written once, AFTER the training loop exits
# (MotifSAT/train.py:306); during training the best state lives only in memory. Its presence
# therefore proves the loop ran to completion and the GSAT-eligible best epoch was selected.
# All 7 cells carry no ckpt_fallback, so none is a last-epoch salvage.
#
# Device: GPU, matching the campaign (the complete siblings ran DEVICE=cuda), so the numbers
# stay comparable and the SMOKE diff below is a clean test of the eval_only path rather than
# of CPU-vs-GPU float noise.
#
# r: --eval_only never calls anneal_r, so model.r stays at init_r 0.9 instead of final_r.
# r reaches only the aux dict and info_loss (MotifSAT/model.py:692) -- never the attention
# sampling, the logits or any reported metric -- so this cannot move a number. SMOKE proves it.
#
# SMOKE=1 validates the path WITHOUT touching anything real: cells are three COMPLETE cells,
# out_dir is redirected to a throwaway tree, no claim is taken and no failure is recorded.
# The regenerated metrics are then diffed against the ones already on disk.
#
# ENV: SMOKE=0|1.  Arrays: SMOKE=1 -> --array=0-2, SMOKE=0 -> --array=0-6.
# Submitted through sbatch stdin (no file copy to the HPC); this file is the record.
# r_for() is filled in from r_table_v2.sh at submit time.
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
SCRIPTS="$REPO/motifsat_paper_deliverables_v1/_scripts"
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
# Match the campaign environment exactly (see any sibling run_manifest.json "env").
export MOSE_GAT_HEADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

OUT="$REPO/motifsat_paper_deliverables_v1"
TREE=variants_v2
RUNS="$OUT/$TREE"
SMOKE_RUNS="$OUT/${TREE}_secondpass_smoke"
PROC_BASE="$REPO/processed_variants_v1"
FOLDS_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/
DISPATCH="$OUT/_dispatch_variants"; FAILURES="$DISPATCH/failures.tsv"
CLAIMS="$DISPATCH/claims_$TREE"
SMOKE="${SMOKE:-0}"
JOBID="${SLURM_JOB_ID:-$$}"

R_TABLE_FILLED_AT_LAUNCH

# The 7 cells to finish: "<preset> <grouping> <dataset> <fold> <backbone>".
# All have best_model.pt + history.json and no occurrence_summary.json.
REAL_CELLS=(
  "base_gsat none hERG 3 PNA"
  "base_gsat none hERG 4 GAT"
  "base_gsat none hERG 4 PNA"
  "base_gsat none hERG 4 SAGE"
  "readout_layernorm_poolmax copy hERG 2 SAGE"
  "readout_layernorm_poolmax copy hERG 4 GCN"
  "readout_layernorm_poolmulti copy hERG 0 GAT"
)
# Validation cells: already COMPLETE, one per preset shape above, so the regenerated
# metrics can be diffed against the on-disk ones. Nothing real is written.
SMOKE_CELLS=(
  "base_gsat none hERG 0 GAT"
  "readout_layernorm_poolmax copy hERG 0 GCN"
  "readout_layernorm_poolmulti copy hERG 0 GCN"
)

if [ "$SMOKE" = 1 ]; then CELLS=("${SMOKE_CELLS[@]}"); else CELLS=("${REAL_CELLS[@]}"); fi
i="${SLURM_ARRAY_TASK_ID:-0}"
[ "$i" -lt "${#CELLS[@]}" ] || { echo "[skip] task $i beyond cell list (${#CELLS[@]})"; exit 0; }
read -r preset grouping ds f bb <<< "${CELLS[$i]}"

vocab=rbrics_v3; vroot="$REPO/vocab_final_v3"; var=rbrics
stem=motifsat; [[ "$preset" == base_gsat* || "$preset" == motif_emb* ]] && stem=gsat
rel="$vocab/$preset/$grouping/$ds/fold$f/$bb"
cid="${vocab}__${preset}__${grouping}__${ds}__f${f}__${bb}"
real="$RUNS/$rel"
ckpt="$real/best_model.pt"
if [ "$SMOKE" = 1 ]; then dir="$SMOKE_RUNS/$rel"; else dir="$real"; fi

echo "[cell] $rel  stem=$stem  SMOKE=$SMOKE"
[ -s "$ckpt" ] || { echo "[FAIL] no checkpoint $ckpt"; exit 2; }
rr="$(r_for "$ds" "$bb")" || { echo "[FAIL] no r for $ds/$bb"; exit 2; }
fr=${rr% *}; di=${rr#* }

if [ "$SMOKE" != 1 ]; then
    # Already finished by an earlier pass -> nothing to do.
    if [ -s "$dir/occurrence_summary.json" ]; then echo "[done] $rel"; exit 0; fi
    mkdir "$CLAIMS/$cid" 2>/dev/null || { echo "[skip] claimed $cid"; exit 0; }
    printf '%s\t%s\t%s\n' "$(hostname -s):$JOBID" "$JOBID" "$(date +%s)" > "$CLAIMS/$cid/info"
    # run.py rewrites run_manifest.json; keep the training one as the provenance record.
    if [ -s "$dir/run_manifest.json" ] && [ ! -e "$dir/run_manifest_train.json" ]; then
        cp -p "$dir/run_manifest.json" "$dir/run_manifest_train.json"
    fi
fi

gargs=(); [ "$grouping" != none ] && gargs=(--motif_grouping "$grouping")
mkdir -p "$dir"
echo "[eval] $rel  final_r=$fr decay_interval=$di  <- $ckpt"
python3 MotifSAT/run.py --config "MotifSAT/configs/${preset}.yaml" \
    --dataset "$ds" --fold "$f" --backbone "$bb" \
    --data_root "$FOLDS_ROOT" --vocab_root "$vroot" --vocab_variant "$var" \
    --processed_root "$PROC_BASE/$vocab" "${gargs[@]}" \
    --info_loss_coef 1 --init_r 0.9 --decay_r 0.1 --final_r "$fr" --decay_interval "$di" \
    --ckpt_rule gsat --clip_grad 2.0 \
    --out_dir "$dir" --final_out_dir --per_split_eval --occurrence_eval \
    --eval_only --load_weights_from "$ckpt" > "$dir/eval.log" 2>&1
rc=$?

# A non-strict load means the checkpoint does not match this architecture: reject the cell.
if grep -q "\[eval_only\] loaded with" "$dir/eval.log"; then
    echo "[FAIL] non-strict load: $(grep '\[eval_only\] loaded with' "$dir/eval.log")"; rc=3
fi

ok=0
if [ "$rc" = 0 ] \
   && python3 "$SCRIPTS/native_complete.py" "$dir" "$stem" --quiet >/dev/null 2>&1 \
   && [ -s "$dir/occurrence_summary.json" ]; then ok=1; fi
# The real cells must still carry their training history; the smoke tree never has one.
if [ "$ok" = 1 ] && [ "$SMOKE" != 1 ] && [ ! -s "$dir/history.json" ]; then
    echo "[FAIL] history.json lost"; ok=0; rc=4
fi

if [ "$ok" = 1 ]; then
    echo "[ok] $rel  files=$(ls "$dir" | wc -l)"; exit 0
fi
echo "[FAIL rc=$rc] $rel"; tail -25 "$dir/eval.log"
if [ "$SMOKE" != 1 ]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date +%s)" "$(hostname -s)" "$JOBID" "secondpass" "$cid" "$rc" >> "$FAILURES"
    touch "$CLAIMS/$cid/.failed"
fi
exit 1
