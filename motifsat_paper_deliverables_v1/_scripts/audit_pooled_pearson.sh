#!/usr/bin/env bash
#SBATCH -J audit_pp
#SBATCH -p share,preempt
#SBATCH --gres=gpu:0
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH --requeue
#SBATCH --array=0-399%60
#SBATCH -o /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/motifsat_paper_deliverables_v1/_audit/pooled_pearson_logs/%A_%a.out
# audit_pooled_pearson.sh — control for the granularity confound of the occurrence Pearson metrics.
# Base GSAT checkpoints (the ones variants_v1/rbrics_v3/base_gsat was scored from, source.json) are
# re-evaluated with run.py --eval_only --occurrence_eval exactly as variants_base_eval.sh did, with ONE
# change: occurrence_eval's attention W is GSAT's own soft node attention AVERAGED within each motif
#   type  : per (graph, motif id)          — analogue of the pooled readout presets
#   copy  : per (graph, motif id, copy)    — analogue of the per-copy motif-graph presets
# Pooling keeps every motif's mean score; only p(W) and p(W \ m) change. If pooled GSAT reaches the
# MotifSAT Pearson, the gain is the attention's granularity, not motif-level training.
# Task = mode(2) x dataset(8) x fold(5) x backbone(5) = 400. Out: _audit/pooled_pearson/<mode>/<ds>/fold<f>/<bb>/
# Read-only for the campaign trees. Submitted via sbatch stdin; this file is the record.
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
cd "$REPO"; export PYTHONPATH="$REPO:$REPO/MotifSAT" WANDB_MODE=disabled OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
OUT="$REPO/motifsat_paper_deliverables_v1"
FOLDS_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/
DSS=(Benzene_Verified_GT Alkane_Carbonyl_Verified_GT Fluoride_Carbonyl_Verified_GT BBBP hERG Mutagenicity esol Lipophilicity)
BBS=(GIN GCN GAT SAGE PNA); MODES=(type copy)
i=${SLURM_ARRAY_TASK_ID}
mode=${MODES[$(( i / 200 ))]}; r=$(( i % 200 )); ds=${DSS[$(( r / 25 ))]}; r=$(( r % 25 )); f=$(( r / 5 )); bb=${BBS[$(( r % 5 ))]}
base="$OUT/variants_v1/rbrics_v3/base_gsat/none/$ds/fold$f/$bb"
dir="$OUT/_audit/pooled_pearson/$mode/$ds/fold$f/$bb"
[ -s "$dir/occurrence_summary.json" ] && { echo "[done] $dir"; exit 0; }
src=$(python3 -c "import json;print(json.load(open('$base/source.json'))['source_run'])") || { echo "[FAIL] no source.json $base"; exit 2; }
read -r fr di < <(python3 -c "import json;d=json.load(open('$base/source.json'));print(d['final_r'],d['decay_interval'])")
[ -s "$src/best_model.pt" ] || { echo "[FAIL] no checkpoint $src"; exit 2; }
mkdir -p "$dir"
echo "[audit $mode] $ds fold$f $bb <- $src"
POOL_MODE=$mode python3 - "$ds" "$f" "$bb" "$fr" "$di" "$src" "$dir" "$FOLDS_ROOT" "$REPO" > "$dir/eval.log" 2>&1 <<'PY'
import os, sys, torch
ds, f, bb, fr, di, src, out, folds, repo = sys.argv[1:10]
MODE = os.environ['POOL_MODE']
import occurrence_eval as oe
from SharedModules.evaluation.motif_eval import model_node_att_fn as _base_fn

def pooled_fn(model, device):
    fn = _base_fn(model, device)
    def g(d):
        W = fn(d)
        if W is None:
            return None
        W = W.view(-1).float()
        n2m = d.nodes_to_motifs.view(-1).to(W.device).long()
        n2c = d.nodes_to_copy.view(-1).to(W.device).long()
        known = n2m >= 0
        out = W.clone()
        if bool(known.any()):
            key = n2m[known] * (int(n2c.max()) + 1) + n2c[known] if MODE == 'copy' else n2m[known]
            _, inv = torch.unique(key, return_inverse=True)
            s = torch.zeros(int(inv.max()) + 1, device=W.device).index_add_(0, inv, W[known])
            c = torch.zeros_like(s).index_add_(0, inv, torch.ones_like(W[known]))
            out[known] = (s / c)[inv]
        return out
    return g

oe.model_node_att_fn = pooled_fn            # occurrence_tables looks it up in its own module
import run
sys.argv = ['run.py', '--config', f'{repo}/MotifSAT/configs/base_gsat.yaml', '--dataset', ds, '--fold', f,
            '--backbone', bb, '--data_root', folds, '--vocab_root', f'{repo}/vocab_final_v3',
            '--vocab_variant', 'rbrics', '--processed_root', f'{repo}/processed_variants_v1/rbrics_v3',
            '--info_loss_coef', '1', '--init_r', '0.9', '--decay_r', '0.1', '--final_r', fr,
            '--decay_interval', di, '--ckpt_rule', 'gsat', '--out_dir', out, '--final_out_dir',
            '--occurrence_eval', '--eval_only', '--load_weights_from', f'{src}/best_model.pt']
run.main()
PY
rc=$?
if grep -q "\[eval_only\] loaded with" "$dir/eval.log"; then echo "[FAIL] non-strict load"; exit 3; fi
[ "$rc" = 0 ] && [ -s "$dir/occurrence_summary.json" ] || { echo "[FAIL rc=$rc] $dir"; tail -20 "$dir/eval.log"; exit 1; }
echo "[ok] $dir"
