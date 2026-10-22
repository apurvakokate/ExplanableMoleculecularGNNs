#!/usr/bin/env bash
#SBATCH -J gsat_repl_prep
#SBATCH -p preempt
#SBATCH --gres=gpu:0
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH -o /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/gsat_replication_v1/_dispatch/logs/prep_%j.out
# prep_job.sh — process every dataset ONCE through upstream's own get_data_loaders, so the
# 210 parallel seed jobs never race on first-time processing. Generates Spurious-Motif
# (upstream gen_dataset), processes mutag (6.3 GB pkl -> 64G mem), Graph-SST2, mnist.
# Uses seed 0 and the upstream config's batch_size/splits; nothing is trained.
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
UP=/nfs/hpc/share/kokatea/ChemIntuit/GSAT_upstream_6812f5c  # pristine upstream clone, outside the synced repo
source ~/hpc-share/anaconda3/etc/profile.d/conda.sh
conda activate l2xgnn
cd "$UP/src"
python - <<'EOF'
import yaml, traceback
from pathlib import Path
from utils import get_local_config_name, get_data_loaders
d = Path(yaml.safe_load(open('configs/global_config.yml'))['data_dir'])
for ds in ['spmotif_0.5', 'spmotif_0.7', 'spmotif_0.9', 'mutag', 'Graph-SST2', 'mnist']:
    try:
        cfg = yaml.safe_load(open('configs/' + get_local_config_name('GIN', ds)))['data_config']
        loaders, test_set, x_dim, e_dim, ncls, aux = get_data_loaders(
            d, ds, cfg['batch_size'], cfg.get('splits'), 0, cfg.get('mutag_x', False))
        print(f'PREP OK {ds}: train={len(loaders["train"].dataset)} '
              f'valid={len(loaders["valid"].dataset)} test={len(loaders["test"].dataset)} '
              f'x_dim={x_dim} edge_attr_dim={e_dim} num_class={ncls}', flush=True)
    except Exception:
        print(f'PREP FAIL {ds}', flush=True); traceback.print_exc()
EOF
