#!/usr/bin/env bash
# setup.sh — run ONCE on the LOGIN node (needs internet). Idempotent.
#   1. clone upstream GSAT at the pinned commit (cells.py UPSTREAM_COMMIT); refuse if any
#      tracked upstream file is modified.
#   2. symlink raw files that already exist on the HPC (no copies):
#        mutag      <- ChemIntuit/GSAT/data/mutag/raw            (8 upstream raw files)
#        Graph-SST2 <- L2XGNN/xai_evaluation/datasets/Graph-SST2/raw (6 files)
#        mnist      <- $MNIST_RAW (only if set; mnist_75sp_{train,test}.pkl)
#   3. download + process the auto-download sets through upstream's own dataset classes:
#        ba_2motifs (DIG_storage/BA_2Motifs.pkl, 6.0 MB), 6 x ogbg_mol* (snap.stanford.edu, 2.9 MB)
# Heavy processing (mutag, SST2, mnist, spmotif generation) runs in prep_job.sh on a CPU node.
set -euo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
S="$REPO/gsat_replication_v1/_scripts"
UP=/nfs/hpc/share/kokatea/ChemIntuit/GSAT_upstream_6812f5c  # pristine upstream clone, outside the synced repo
URL=$(python3 -c "import sys; sys.path.insert(0,'$S'); import cells; print(cells.UPSTREAM_REPO)")
COMMIT=$(python3 -c "import sys; sys.path.insert(0,'$S'); import cells; print(cells.UPSTREAM_COMMIT)")
mkdir -p "$REPO/gsat_replication_v1/_dispatch/logs" "$REPO/gsat_replication_v1/results"

if [ ! -d "$UP/.git" ]; then
    git clone "$URL" "$UP"
    git -C "$UP" checkout -q "$COMMIT"
fi
HEAD=$(git -C "$UP" rev-parse --short=7 HEAD)
[ "$HEAD" = "${COMMIT:0:7}" ] || { echo "ABORT: upstream HEAD $HEAD != $COMMIT"; exit 1; }
# only the CODE must be pristine: the repo's data/ogbg_*/.gitignore placeholders are
# replaced by ogb's own download (it rmtree's the folder), as for any upstream user.
git -C "$UP" diff --quiet HEAD -- src || { echo "ABORT: upstream src/ modified"; git -C "$UP" diff --stat -- src; exit 1; }
echo "upstream OK: $HEAD, unmodified"

D="$UP/data"
link_all(){  # src_dir dst_dir file...
    local src=$1 dst=$2; shift 2; mkdir -p "$dst"
    for f in "$@"; do
        [ -e "$src/$f" ] || { echo "ABORT: missing $src/$f"; exit 1; }
        ln -sfn "$src/$f" "$dst/$f"
    done
}
link_all /nfs/hpc/share/kokatea/ChemIntuit/GSAT/data/mutag/raw "$D/mutag/raw" \
    Mutagenicity_A.txt Mutagenicity_edge_gt.txt Mutagenicity_edge_labels.txt \
    Mutagenicity_graph_indicator.txt Mutagenicity_graph_labels.txt \
    Mutagenicity_label_readme.txt Mutagenicity_node_labels.txt Mutagenicity.pkl
link_all /nfs/hpc/share/kokatea/L2XGNN/xai_evaluation/datasets/Graph-SST2/raw "$D/Graph-SST2/raw" \
    Graph-SST2_edge_index.txt Graph-SST2_graph_labels.txt Graph-SST2_node_features.pkl \
    Graph-SST2_node_indicator.txt Graph-SST2_sentence_tokens.json Graph-SST2_split_indices.txt
if [ -n "${MNIST_RAW:-}" ]; then
    link_all "$MNIST_RAW" "$D/mnist/raw" mnist_75sp_train.pkl mnist_75sp_test.pkl \
        mnist_75sp_train_superpixels.pkl mnist_75sp_test_superpixels.pkl
else
    echo "MNIST_RAW not set: mnist cells will fail until raw files are linked"
fi
echo "raw symlinks OK"

source ~/hpc-share/anaconda3/etc/profile.d/conda.sh
conda activate l2xgnn
cd "$UP/src"
# ogb asks "Will you update the dataset now? (y/N)" for the placeholder folders; answer N
# (its download() replaces the folder anyway). The script goes via a file so stdin is free.
cat > "$UP/_setup_download.py" <<'EOF'
from pathlib import Path
from datasets import SynGraphDataset
from ogb.graphproppred import PygGraphPropPredDataset
d = Path('../data')
print('ba_2motifs', len(SynGraphDataset(d, 'ba_2motifs')))
for n in ['molhiv', 'molbace', 'molbbbp', 'molclintox', 'moltox21', 'molsider']:
    print(n, len(PygGraphPropPredDataset(root=d, name='ogbg-' + n)))
EOF
yes N | python "$UP/_setup_download.py"
echo "=== setup done. Next: sbatch $S/prep_job.sh ==="
