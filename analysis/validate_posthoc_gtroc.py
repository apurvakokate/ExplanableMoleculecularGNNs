#!/usr/bin/env python3
"""validate_posthoc_gtroc.py — shardable worker: node-direct Instance + Global GT-ROC
for the post-hoc methods over source-GT + synthetic(planted)-GT setups in posthoc_v1.

Consumes a shard (``--shard i/N``) so N workers run disjoint slices. Per setup:
  * writes ``<dest>/<relpath>/gtroc.json`` =
      {method: {split: {"gtroc_instance": float, "gtroc_global": float}}}
    computed by feeding each method's PER-GRAPH node attribution DIRECTLY to
    gt_roc_block (instance = ANY one fired disjunct; global = UNION of all disjuncts;
    per graph, then averaged).
  * for mage_v2 ALSO writes ``<dest>/<relpath>/mage_v2_node_atts.json`` =
      {split: {graph_idx: [per-node att]}}  (recomputed from mage_attention.pt so it
    never has to be recomputed again).

Per-node attribution source per method:
  gnnexplainer / pgexplainer / motif_occlusion  ->  saved explainer_importances.json
  mage_v2                                        ->  score_mage(attention_mean) from
                                                     saved mage_attention.pt

NEVER overwrites: skips a setup whose gtroc.json already exists (resumable, safe to add
workers). Any setup that fails to process OR write is appended to the worker's failure
log ``<dest>/_failures/<shard>.log`` (relpath<TAB>error).

  REPO=<repo> python3 analysis/validate_posthoc_gtroc.py \
      --out_root <REPO>/final_v2 --posthoc_root <REPO>/posthoc_v1 \
      --dest_root <REPO>/gtroc_nodedirect_v1 \
      --data_root <FOLDS> --vocab_root <REPO>/vocab_final_v2 --shard 0/300
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.environ.get('REPO', str(Path(__file__).resolve().parent.parent)))

from analysis.eval_driver_common import (
    build_gt_loaders, split_lists_and_gt, iter_runs, SPLITS)
from analysis.eval_driver_posthoc import load_vanilla_model
from SharedModules.baselines.mage import _MotifAttention, score_mage
from SharedModules.evaluation.split_eval import gt_roc_block, _pi_to_node_atts

SAVED_ATT = ('gnnexplainer', 'pgexplainer', 'motif_occlusion')   # per-node atts on disk
ALL_METHODS = SAVED_ATT + ('mage_v2',)
INST, GLOB = 'instance_gt_roc_node_auc_mean', 'global_gt_roc_node_auc_mean'
# source-GT datasets (planted are selected by meta use_gt); together = the in-scope set
SOURCE_GT = {'Benzene_Verified_GT', 'Alkane_Carbonyl_Verified_GT',
             'Fluoride_Carbonyl_Verified_GT', 'mutag'}


def _node_att_fn(gl, att_by_i):
    """Attach each graph's per-node att (aligned by split position) and return a
    node_att_fn reading it. Length guard fails loud on misalignment."""
    for g in gl:
        g._raw_att = None
    n = 0
    for i, g in enumerate(gl):
        a = att_by_i.get(i)
        if a is None:
            continue
        t = a.float() if torch.is_tensor(a) else torch.tensor(a, dtype=torch.float32)
        if t.numel() != g.num_nodes:
            raise RuntimeError(f'att/nodes mismatch graph {i}: {t.numel()} vs {g.num_nodes}')
        g._raw_att = t
        n += 1

    def fn(d):
        a = getattr(d, '_raw_att', None)
        return a.to(d.x.device) if a is not None else None
    return (fn if n else None)


def _load_mage(posthoc_dir, device):
    p = Path(posthoc_dir) / 'mage_attention.pt'
    if not p.exists():
        return None
    sd = torch.load(p, map_location='cpu', weights_only=True)
    attn = _MotifAttention(int(sd['a_q'].shape[0]))
    attn.load_state_dict(sd)
    return attn.to(device).eval()


def process(run_dir, meta, args, device):
    """Return (gtroc, mage_atts, issues). Each METHOD is isolated in its own try/except
    so one method failing never sinks the others; ``issues`` collects (method, reason)
    for methods that were MISSING (no atts) or ERRORed. Returns (None, None, None) when
    the setup carries no GT / was not evaluated (out of scope)."""
    relpath = run_dir.relative_to(args.out_root)
    posthoc_dir = Path(args.posthoc_root) / relpath
    if not (posthoc_dir / 'summary_splits.json').exists():
        return None, None, None
    loaders, vocab, dmeta, tt = build_gt_loaders(
        meta, args.data_root, args.vocab_root, None, args.loader_batch_size)
    sl_all, gt = split_lists_and_gt(loaders, meta)
    if all(gt.get(s) is None for s in SPLITS):
        return None, None, None
    model = load_vanilla_model(run_dir, meta, dmeta, device)
    imp_p = posthoc_dir / 'explainer_importances.json'
    imp = (json.loads(imp_p.read_text()).get('importances_by_split', {})
           if imp_p.exists() else {})
    attn = _load_mage(posthoc_dir, device)
    pc = 0 if meta['dataset'] == 'mutag' else 1

    gtroc, mage_atts, issues = {}, {}, []
    for method in ALL_METHODS:
        try:
            per_split, got_att = {}, False
            for split in SPLITS:
                gl, sl = gt.get(split), sl_all.get(split)
                if not gl or not sl:
                    continue
                if method in SAVED_ATT:
                    # saved atts are keyed to the FULL split (split_lists[split]); gt[split]
                    # IS that same list for every dataset EXCEPT mutag (a subset). Guard so
                    # a subset can never silently misalign the atts.
                    if gl is not sl:
                        raise RuntimeError('GT list != split list (e.g. mutag subset): saved '
                                           'atts are keyed to the full split, remap needed')
                    a = (imp.get(split, {}) or {}).get(method)
                    att_by_i = {int(k): v for k, v in a.items()} if a else None
                elif attn is None:                        # mage_v2 needs the saved attention
                    att_by_i = None
                else:
                    # score on gl (the GT eval list): per_instance keys are the index into
                    # the SCORED list (kept_gi), so scoring gl makes them align with the
                    # gl that gt_roc_block iterates — correct for mutag's subset too.
                    _, pi = score_mage(model, attn, gl, vocab, device, tt,
                                       positive_class=pc, return_per_instance=True,
                                       verbose=False, embed_batch_size=args.batch_size,
                                       score_mode='attention_mean')
                    att_by_i = {int(k): v for k, v in _pi_to_node_atts(pi, gl).items()} or None
                if not att_by_i:
                    continue
                got_att = True
                if method == 'mage_v2':
                    mage_atts[split] = {str(i): t.detach().cpu().view(-1).tolist()
                                        for i, t in att_by_i.items()}
                fn = _node_att_fn(gl, att_by_i)
                if fn is None:
                    continue
                b = gt_roc_block(model, gl, device, fn)
                per_split[split] = {'gtroc_instance': b.get(INST, float('nan')),
                                    'gtroc_global': b.get(GLOB, float('nan'))}
            if per_split:
                gtroc[method] = per_split
            else:
                issues.append((method, 'MISSING: no attributions available' if not got_att
                               else 'MISSING: atts present but no GT-ROC produced'))
        except Exception as e:                            # isolate: one method never sinks the rest
            issues.append((method, f'ERROR: {type(e).__name__}: {e}'))
    return gtroc, mage_atts, issues


def _log_fail(fail_log: Path, relpath, method, reason):
    """Append one line: relpath<TAB>method<TAB>reason. method='*' for setup-level."""
    fail_log.parent.mkdir(parents=True, exist_ok=True)
    with open(fail_log, 'a') as f:
        f.write(f'{relpath}\t{method}\t{reason}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', required=True, help='final_v2 (checkpoints + meta)')
    ap.add_argument('--posthoc_root', required=True, help='posthoc_v1 (saved atts + summary)')
    ap.add_argument('--dest_root', required=True, help='NEW output tree (path-aligned)')
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--vocab_root', required=True)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--loader_batch_size', type=int, default=128)
    ap.add_argument('--dataset', nargs='*', default=None)
    ap.add_argument('--shard', default=None, help='i/N — this worker processes runs[i::N]')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    device = torch.device(args.device)
    dest_root = Path(args.dest_root)
    shard_tag = (args.shard or 'single').replace('/', '_')
    fail_log = dest_root / '_failures' / f'{shard_tag}.log'

    ok = failed = skipped = 0
    for run_dir, meta, fam in iter_runs(args.out_root, {'baselines'}, args.dataset, args.shard):
        if args.limit and ok >= args.limit:
            break
        if '_filter' in meta.get('vocab_variant', ''):
            continue
        # scope: source-GT datasets + planted (use_gt) runs only
        if not (meta.get('use_gt') or meta['dataset'] in SOURCE_GT):
            continue
        relpath = run_dir.relative_to(args.out_root)
        out_dir = dest_root / relpath
        gtroc_path = out_dir / 'gtroc.json'
        if gtroc_path.exists() and gtroc_path.stat().st_size > 2:     # never overwrite
            skipped += 1
            continue
        tag = f"{meta.get('dataset')}/{meta.get('backbone')}/{meta.get('vocab_variant')}/fold{meta.get('fold')}/{meta.get('gt_tier')}"
        try:
            gtroc, mage_atts, issues = process(run_dir, meta, args, device)
            if gtroc is None:                              # out of scope / no GT
                skipped += 1
                continue
            for method, reason in issues:                  # log missing/errored methods
                _log_fail(fail_log, relpath, method, reason)
            if not gtroc:                                  # every method missing/errored
                failed += 1
                print(f'  [FAIL] {tag}: all methods missing/errored (logged)')
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            gtroc_path.write_text(json.dumps(gtroc, indent=2))
            if mage_atts:
                (out_dir / 'mage_v2_node_atts.json').write_text(json.dumps(mage_atts))
            ok += 1
            print(f'  [ok] {tag}' + (f'  ({len(issues)} method issue(s) logged)' if issues else ''))
        except Exception as e:                             # setup-level (loaders/model/write)
            failed += 1
            _log_fail(fail_log, relpath, '*', f'{type(e).__name__}: {e}')
            print(f'  [FAIL] {tag}: {type(e).__name__}: {e}')

    print(f'\nDONE shard={args.shard or "single"}  ok={ok} failed={failed} skipped={skipped}'
          f'  (failures logged to {fail_log})')


if __name__ == '__main__':
    main()
