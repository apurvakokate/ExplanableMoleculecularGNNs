#!/usr/bin/env bash
#SBATCH -J variants_preproc
#SBATCH -p share,preempt
#SBATCH --gres=gpu:0
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH --array=0-79
#SBATCH -o /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/motifsat_paper_deliverables_v1/_dispatch_variants/preproc_logs/%A_%a.out
# variants_preprocess.sh — build the PyG processed caches for the MotifSAT variant campaign
# BEFORE the GPU workers start, so hundreds of workers never race to build the same cache.
# One array task = one (vocab, dataset, fold); 2 vocabs x 8 datasets x 5 folds = 80 tasks.
#   idx 0-39  rBRICS v3   vocab_final_v3/<ds>/rbrics                  (fold-0 vocabulary)
#   idx 40-79 FM-BPE v4   fmbpe_v4/vocab/fold<k>/<ds>/fm_bpe_full_union (per-fold vocabulary)
# Caches: processed_variants_v1/<vocab_tag>/<variant>/<ds>_fold<k>  (fresh roots; rBRICS v3
# atoms differ from v2 after the atom-tracking fix, so v2 caches are NOT reused).
# Also a check: iterates every split once so AttachCopies validates every copy record
# against nodes_to_motifs; writes _dispatch_variants/preproc_ok/<cell>.json on success.
# Submitted through sbatch stdin (no file copy to the HPC); this file is the record.
set -euo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled OMP_NUM_THREADS=2
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
DSS=(Benzene_Verified_GT Alkane_Carbonyl_Verified_GT Fluoride_Carbonyl_Verified_GT BBBP hERG Mutagenicity esol Lipophilicity)
i=${SLURM_ARRAY_TASK_ID}
v=$(( i / 40 )); r=$(( i % 40 )); ds=${DSS[$(( r / 5 ))]}; f=$(( r % 5 ))
if [ "$v" = 0 ]; then
    TAG=rbrics_v3; VROOT="$REPO/vocab_final_v3"; VAR=rbrics
else
    TAG=fmbpe_v4; VROOT="$REPO/fmbpe_v4/vocab/fold$f"; VAR=fm_bpe_full_union
fi
OKDIR="$REPO/motifsat_paper_deliverables_v1/_dispatch_variants/preproc_ok"; mkdir -p "$OKDIR"
echo "preproc $TAG $ds fold$f vocab=$VROOT/$ds/$VAR"
python3 - "$TAG" "$VROOT" "$VAR" "$ds" "$f" "$OKDIR" <<'PYEND'
import json, sys, time
import torch
from SharedModules.data.vocab import load_vocab
from SharedModules.data.loader import get_loaders, TASK_TYPE
from SharedModules.data.dataset_routing import variant_processed_root
tag, vroot, var, ds, f, okdir = sys.argv[1:7]; f = int(f)
REPO = '/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor'
FOLDS = '/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/'
proc = variant_processed_root(f'{REPO}/processed_variants_v1/{tag}', var)
t0 = time.time()
vocab = load_vocab(vroot, ds, var)
loaders, _, meta = get_loaders(dataset=ds, data_root=FOLDS, fold=f, vocab=vocab, processed_root=proc,
                               batch_size=256, normalize=(TASK_TYPE.get(ds, 'BinaryClass') == 'Regression'))
out = {'tag': tag, 'dataset': ds, 'fold': f, 'variant': var, 'vocab_root': vroot, 'processed_root': proc,
       'num_motifs': vocab.num_motifs}
for split, ld in loaders.items():
    n_g = n_at = n_in_copy = n_rep_graphs = 0
    for d in ld:
        assert hasattr(d, 'nodes_to_copy'), 'nodes_to_copy missing: copy sidecar not attached'
        n_g += d.num_graphs; n_at += d.num_nodes
        n_in_copy += int((d.nodes_to_copy >= 0).sum())
        rep = torch.zeros(d.num_graphs, dtype=torch.bool)
        rep[d.batch[d.nodes_to_copy > 0]] = True
        n_rep_graphs += int(rep.sum())
    out[split] = {'graphs': n_g, 'atoms': n_at, 'atoms_in_copy': n_in_copy,
                  'graphs_with_repeated_motif': n_rep_graphs}
out['seconds'] = round(time.time() - t0, 1)
print(json.dumps(out))
json.dump(out, open(f'{okdir}/{tag}__{ds}__f{f}.json', 'w'), indent=1)
PYEND
