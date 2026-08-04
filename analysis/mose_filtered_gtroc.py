#!/usr/bin/env python3
"""mose_filtered_gtroc.py — FILTERED node-direct GT-ROC for MoSE under a chosen UNK mode.

MoSE is already trained on the _filter vocab (its meta vocab_variant ends in _filter), so
NO variant patch — the loaders are already filtered. Forwards MoSE to get its per-node
attention and applies --unk_mode:
  exclude : drop UNK atoms from the AUC (masked GT-ROC)  -- needed for variant (A)
  half    : UNK atoms -> 0.5  (MoSE's own native value; matches variant (B))
  zero    : UNK atoms -> 0.0
Writes <dest>/<relpath>/gtroc.json = {"mose": {split: {gtroc_instance, gtroc_global}}}.
Same shard / skip-done / failure-log contract as the other drivers.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.environ.get('REPO', str(Path(__file__).resolve().parent.parent)))

from analysis.eval_driver_common import build_gt_loaders, split_lists_and_gt, iter_runs, SPLITS
from analysis.eval_driver_antehoc import load_native_model_and_scores
from SharedModules.evaluation.motif_eval import model_node_att_fn
from SharedModules.evaluation.split_eval import gt_roc_block
from analysis.gtroc_unk import gtroc_masked

INST, GLOB = 'instance_gt_roc_node_auc_mean', 'global_gt_roc_node_auc_mean'
SOURCE_GT = {'Benzene_Verified_GT', 'Alkane_Carbonyl_Verified_GT',
             'Fluoride_Carbonyl_Verified_GT', 'mutag'}


def _fill_fn(model, device, fill):
    base = model_node_att_fn(model, device)
    def fn(data):
        a = base(data)
        if a is None:
            return None
        a = a.view(-1).clone()
        n2m = getattr(data, 'nodes_to_motifs', None)
        if n2m is not None:
            a[n2m.view(-1).to(a.device) < 0] = fill
        return a
    return fn


def process(run_dir, meta, args, device):
    # MoSE meta variant is ALREADY the _filter variant -> use as-is (no patch).
    loaders, vocab, dmeta, tt = build_gt_loaders(
        meta, args.data_root, args.vocab_root, None, args.loader_batch_size)
    sl_all, gt = split_lists_and_gt(loaders, meta)
    if all(gt.get(s) is None for s in SPLITS):
        return None
    model, _ = load_native_model_and_scores(
        run_dir, meta, 'mose', loaders, vocab, dmeta, device, tt)
    mode = getattr(args, 'unk_mode', 'exclude')
    per_split = {}
    for split in SPLITS:
        gl = gt.get(split)
        if not gl:
            continue
        if mode == 'exclude':
            base = model_node_att_fn(model, device)
            att_by_i = {i: base(g) for i, g in enumerate(gl) if base(g) is not None}
            inst, glob, _ = gtroc_masked(att_by_i, gl)
            per_split[split] = {'gtroc_instance': inst, 'gtroc_global': glob}
        else:
            fn = _fill_fn(model, device, 0.5 if mode == 'half' else 0.0)
            b = gt_roc_block(model, gl, device, fn)
            per_split[split] = {'gtroc_instance': b.get(INST, float('nan')),
                                'gtroc_global': b.get(GLOB, float('nan'))}
    return {'mose': per_split} if per_split else {}


def _log_fail(fail_log, relpath, reason):
    fail_log.parent.mkdir(parents=True, exist_ok=True)
    with open(fail_log, 'a') as f:
        f.write(f'{relpath}\tmose\t{reason}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', required=True, help='final_v2 (mose checkpoints)')
    ap.add_argument('--dest_root', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--vocab_root', required=True)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--loader_batch_size', type=int, default=128)
    ap.add_argument('--dataset', nargs='*', default=None)
    ap.add_argument('--shard', default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--unk_mode', default='exclude', choices=['zero', 'half', 'exclude'])
    ap.add_argument('--posthoc_root', default=None)   # accepted-but-unused (worker compat)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--filtered', action='store_true')
    args = ap.parse_args()
    device = torch.device(args.device)
    dest_root = Path(args.dest_root)
    fail_log = dest_root / '_failures' / f"mose_{(args.shard or 'single').replace('/', '_')}.log"

    ok = failed = skipped = 0
    for run_dir, meta, fam in iter_runs(args.out_root, {'mose'}, args.dataset, args.shard):
        if args.limit and ok >= args.limit:
            break
        v = meta.get('vocab_variant', '')
        if '_filter' not in v:                        # MoSE runs are all filtered; skip any non-filter
            continue
        if not (meta['dataset'] in SOURCE_GT or '_relabelled_dnf' in v or meta.get('use_gt')):
            continue
        relpath = run_dir.relative_to(args.out_root)
        out_dir = dest_root / relpath
        gtroc_path = out_dir / 'gtroc.json'
        if gtroc_path.exists() and gtroc_path.stat().st_size > 2:
            skipped += 1
            continue
        tag = f"{meta.get('dataset')}/{meta.get('backbone')}/{v}/fold{meta.get('fold')}"
        try:
            res = process(run_dir, meta, args, device)
            if not res:
                skipped += 1
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            gtroc_path.write_text(json.dumps(res, indent=2))
            ok += 1
            print(f'  [ok] {tag}')
        except Exception as e:
            failed += 1
            _log_fail(fail_log, relpath, f'{type(e).__name__}: {e}')
            print(f'  [FAIL] {tag}: {type(e).__name__}: {e}')
    print(f'\nDONE mose-filtered shard={args.shard or "single"}  ok={ok} failed={failed} skipped={skipped}')


if __name__ == '__main__':
    main()
