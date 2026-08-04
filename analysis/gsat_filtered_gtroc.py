#!/usr/bin/env python3
"""gsat_filtered_gtroc.py — FILTERED node-direct GT-ROC for GSAT.

Loads each base_gsat checkpoint, builds loaders on the _filter vocab/processed tree,
forwards GSAT to get its per-node attention, ZEROES the att on UNK-motif atoms
(nodes_to_motifs < 0) so GSAT is scored only on the kept-motif atom set — apples-to-
apples with MoSE — then computes instance/global GT-ROC. NO retraining (inference).

Output: <dest>/<relpath>/gtroc.json = {"gsat": {split: {gtroc_instance, gtroc_global}}}.
Same shard / skip-done / per-shard failure-log contract as validate_posthoc_gtroc.py.

  python analysis/gsat_filtered_gtroc.py --out_root final_v2 --dest_root gtroc_filtered_v1 \
      --data_root <FOLDS> --vocab_root <VOCAB> --device cpu --shard i/N
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.environ.get('REPO', str(Path(__file__).resolve().parent.parent)))

from analysis.eval_driver_common import build_gt_loaders, split_lists_and_gt, iter_runs, SPLITS
from analysis.eval_driver_antehoc import load_native_model_and_scores
from SharedModules.evaluation.motif_eval import model_node_att_fn
from SharedModules.evaluation.split_eval import gt_roc_block

INST, GLOB = 'instance_gt_roc_node_auc_mean', 'global_gt_roc_node_auc_mean'
SOURCE_GT = {'Benzene_Verified_GT', 'Alkane_Carbonyl_Verified_GT',
             'Fluoride_Carbonyl_Verified_GT', 'mutag'}


def _to_filter_variant(v):
    m = re.search(r'(_relabelled_dnf_k\d+_r\d+)$', v)
    base, suf = (v[:m.start()], m.group(1)) if m else (v, '')
    return base + '_filter' + suf


def _unk_zeroed_fn(model, device):
    """GSAT per-node att with UNK-motif atoms zeroed (kept-atom scope)."""
    base = model_node_att_fn(model, device)
    def fn(data):
        a = base(data)
        if a is None:
            return None
        a = a.view(-1).clone()
        n2m = getattr(data, 'nodes_to_motifs', None)
        if n2m is not None:
            a[n2m.view(-1).to(a.device) < 0] = 0.0
        return a
    return fn


def process(run_dir, meta, args, device):
    fmeta = dict(meta)
    fmeta['vocab_variant'] = _to_filter_variant(meta['vocab_variant'])
    base_proc = str(Path(meta['processed_root']).parent) if meta.get('processed_root') else None
    fmeta.pop('processed_root', None)
    loaders, vocab, dmeta, tt = build_gt_loaders(
        fmeta, args.data_root, args.vocab_root, base_proc, args.loader_batch_size)
    sl_all, gt = split_lists_and_gt(loaders, fmeta)
    if all(gt.get(s) is None for s in SPLITS):
        return None
    model, _ = load_native_model_and_scores(
        run_dir, fmeta, 'gsat', loaders, vocab, dmeta, device, tt)
    fn = _unk_zeroed_fn(model, device)
    per_split = {}
    for split in SPLITS:
        gl = gt.get(split)
        if not gl:
            continue
        b = gt_roc_block(model, gl, device, fn)
        per_split[split] = {'gtroc_instance': b.get(INST, float('nan')),
                            'gtroc_global': b.get(GLOB, float('nan'))}
    return {'gsat': per_split} if per_split else {}


def _log_fail(fail_log, relpath, reason):
    fail_log.parent.mkdir(parents=True, exist_ok=True)
    with open(fail_log, 'a') as f:
        f.write(f'{relpath}\tgsat\t{reason}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', required=True, help='final_v2 (base_gsat checkpoints)')
    ap.add_argument('--dest_root', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--vocab_root', required=True)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--loader_batch_size', type=int, default=128)
    ap.add_argument('--dataset', nargs='*', default=None)
    ap.add_argument('--shard', default=None)
    ap.add_argument('--limit', type=int, default=None)
    # accepted-but-unused: lets the generic worker drive this with the same arg set as the
    # post-hoc driver (GSAT is always filtered; it reads no posthoc atts).
    ap.add_argument('--posthoc_root', default=None)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--filtered', action='store_true')
    args = ap.parse_args()
    device = torch.device(args.device)
    dest_root = Path(args.dest_root)
    fail_log = dest_root / '_failures' / f"gsat_{(args.shard or 'single').replace('/', '_')}.log"

    ok = failed = skipped = 0
    for run_dir, meta, fam in iter_runs(args.out_root, {'gsat'}, args.dataset, args.shard):
        if args.limit and ok >= args.limit:
            break
        v = meta.get('vocab_variant', '')
        if '_filter' in v:
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
    print(f'\nDONE gsat-filtered shard={args.shard or "single"}  ok={ok} failed={failed} skipped={skipped}')


if __name__ == '__main__':
    main()
