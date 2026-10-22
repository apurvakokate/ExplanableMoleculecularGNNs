#!/usr/bin/env python
"""run_one_seed.py — run ONE seed of upstream GSAT exactly as upstream main() does.

Upstream main() loops `for random_state in range(num_seeds)` in one process; this
wrapper runs a single random_state so the 10 seeds run as parallel jobs. Everything
else is upstream: same config files (get_local_config_name), same data_dir
(global_config.yml), same train_gsat_one_seed(). No upstream file is modified.

Must be run with cwd = <GSAT_upstream>/src (upstream uses relative ./configs, ../data).

Output: <out> JSON = {metric_dict, hparam_dict, backbone, dataset, seed, config_file,
upstream_commit, log_dir, elapsed_s}. Written atomically (tmp -> rename).
TensorBoard events + checkpoints land in data/<dataset>/logs/<tag> as upstream does.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, os.getcwd())
from run_gsat import train_gsat_one_seed          # noqa: E402  (upstream)
from utils import get_local_config_name           # noqa: E402  (upstream)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--backbone', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--cuda', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    # upstream sets 5; we hold 2 cores per GPU (thread count does not change the math)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '2')))
    config_dir = Path('./configs')
    method_name = 'GSAT'
    global_config = yaml.safe_load((config_dir / 'global_config.yml').open('r'))
    local_config_name = get_local_config_name(a.backbone, a.dataset)
    local_config = yaml.safe_load((config_dir / local_config_name).open('r'))
    data_dir = Path(global_config['data_dir'])
    device = torch.device(f'cuda:{a.cuda}' if a.cuda >= 0 else 'cpu')

    # Deterministic log dir; a preempted partial run is wiped so TB events are not mixed.
    tag = f'repl_v1-{a.dataset}-{a.backbone}-seed{a.seed}-{method_name}'
    log_dir = data_dir / a.dataset / 'logs' / tag
    if log_dir.exists():
        shutil.rmtree(log_dir)

    t0 = time.time()
    hparam_dict, metric_dict = train_gsat_one_seed(
        local_config, data_dir, log_dir, a.backbone, a.dataset, method_name, device, a.seed)

    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    rec = {'backbone': a.backbone, 'dataset': a.dataset, 'seed': a.seed,
           'config_file': local_config_name, 'upstream_commit': commit,
           'log_dir': str(log_dir.resolve()), 'elapsed_s': time.time() - t0,
           'metric_dict': {k: float(v) for k, v in metric_dict.items()},
           'hparam_dict': hparam_dict}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(rec, indent=2, default=str))
    tmp.rename(out)
    print(f'[run_one_seed] wrote {out}')


if __name__ == '__main__':
    main()
