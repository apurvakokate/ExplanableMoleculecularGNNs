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

--mage_refit: mage_v2 is RE-FIT from scratch (fit_mage; the patched _motif_graph_from_smarts
restores the `ring:`-keyed motifs the saved attention was fit without) and is the ONLY method
scored, over FG-first setups only. Runs IN-PLACE — point --dest_root at the EXISTING gtroc tree:
the corrected mage_v2 is MERGED into each gtroc.json (other methods' cells kept) and the buggy
posthoc mage_attention.pt is OVERWRITTEN with the corrected full-vocab fit. A `_mage_refit.done`
sentinel per setup makes it resumable. Run the FULL pass only (no --filtered).

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
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.environ.get('REPO', str(Path(__file__).resolve().parent.parent)))

from analysis.eval_driver_common import (
    build_gt_loaders, split_lists_and_gt, iter_runs, SPLITS)
from analysis.eval_driver_posthoc import load_vanilla_model
from SharedModules.baselines.mage import _MotifAttention, score_mage, fit_mage
from SharedModules.evaluation.split_eval import gt_roc_block, _pi_to_node_atts
from analysis.gtroc_unk import gtroc_masked

SAVED_ATT = ('gnnexplainer', 'pgexplainer', 'motif_occlusion')   # per-node atts on disk
ALL_METHODS = SAVED_ATT + ('mage_v2',)
INST, GLOB = 'instance_gt_roc_node_auc_mean', 'global_gt_roc_node_auc_mean'
# source-GT datasets (planted are selected by meta use_gt); together = the in-scope set
SOURCE_GT = {'Benzene_Verified_GT', 'Alkane_Carbonyl_Verified_GT',
             'Fluoride_Carbonyl_Verified_GT', 'mutag'}
# FG-first fragmentations key ring motifs as `ring:<smiles>`; the pre-fix MAGE parser
# dropped them, so --mage_refit re-fits MAGE (only these are affected — see mage.py).
FG_FIRST = {'fg_first', 'rdkit_fg_first', 'ertl_first'}


def _base_frag(v):
    """Base fragmentation: drop a planted `_relabelled_dnf...` suffix and `_filter`."""
    return re.sub(r'_relabelled_dnf.*$', '', v).replace('_filter', '')


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


def _to_filter_variant(v):
    """Insert `_filter` (before any planted `_relabelled_dnf` suffix). For baselines the
    vocab_variant is the base fragmentation, so this just appends `_filter`."""
    m = re.search(r'(_relabelled_dnf_k\d+_r\d+)$', v)
    base, suf = (v[:m.start()], m.group(1)) if m else (v, '')
    return base + '_filter' + suf


def _fill_unk(att_by_i, gl, fill):
    """FILTERED mode: set each graph's per-node att to ``fill`` on atoms whose (filtered)
    motif is UNK (nodes_to_motifs < 0), applied to EVERY method so the UNK-atom treatment
    is identical across methods. fill=0.0 -> zero; fill=0.5 -> neutral (matches MoSE).
    Handles both list atts (baselines) and tensor atts (MAGE)."""
    out = {}
    for i, a in att_by_i.items():
        n2m = getattr(gl[i], 'nodes_to_motifs', None)
        if n2m is None:
            out[i] = a
            continue
        n2m = n2m.view(-1).tolist()
        if torch.is_tensor(a):
            a = a.clone().float()
            for j, m in enumerate(n2m):
                if m < 0 and j < a.numel():
                    a[j] = fill
            out[i] = a
        else:
            out[i] = [fill if (j < len(n2m) and n2m[j] < 0) else v for j, v in enumerate(a)]
    return out


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
    lmeta, proc_arg = meta, None            # FULL: use each run's stored processed_root
    if args.filtered:                       # FILTERED: build on the _filter vocab/processed tree
        lmeta = dict(meta)
        lmeta['vocab_variant'] = _to_filter_variant(meta['vocab_variant'])
        # base processed tree (parent of the stored full-variant dir); _proc_root appends _filter
        proc_arg = str(Path(meta['processed_root']).parent) if meta.get('processed_root') else None
        lmeta.pop('processed_root', None)
    loaders, vocab, dmeta, tt = build_gt_loaders(
        lmeta, args.data_root, args.vocab_root, proc_arg, args.loader_batch_size)
    sl_all, gt = split_lists_and_gt(loaders, lmeta)
    if all(gt.get(s) is None for s in SPLITS):
        return None, None, None
    model = load_vanilla_model(run_dir, meta, dmeta, device)
    imp_p = posthoc_dir / 'explainer_importances.json'
    imp = (json.loads(imp_p.read_text()).get('importances_by_split', {})
           if imp_p.exists() else {})
    # --mage_refit: RE-FIT MAGE from scratch (patched parser restores the `ring:` motifs
    # the saved attention was fit without) and score ONLY mage_v2. Fits on whatever vocab
    # this invocation uses — full vocab (FULL pass) or _filter vocab (--filtered pass).
    if args.mage_refit:
        tr = sl_all.get('train')
        attn = (fit_mage(model, tr, vocab, device, tt, attn_epochs=args.attn_epochs,
                         verbose=False, embed_batch_size=args.batch_size) if tr else None)
        methods = ('mage_v2',)
        # OVERWRITE the buggy (ring-less) saved model in posthoc with the corrected full-vocab
        # fit (FULL pass only; the --filtered pass fits a different, filter-vocab attention).
        if attn is not None and not args.filtered:
            torch.save(attn.state_dict(), posthoc_dir / 'mage_attention.pt')
    else:
        attn = _load_mage(posthoc_dir, device)
        methods = ALL_METHODS
    pc = 0 if meta['dataset'] == 'mutag' else 1

    gtroc, mage_atts, issues = {}, {}, []
    for method in methods:
        try:
            per_split, got_att = {}, False
            for split in SPLITS:
                gl, sl = gt.get(split), sl_all.get(split)
                if not gl or not sl:
                    continue
                if method in SAVED_ATT:
                    # saved atts are keyed to the FULL split (split_lists[split]). gt[split]
                    # IS that same list for every dataset EXCEPT mutag, where it is a subset
                    # (mutag_gt_eval_graphs, which returns the SAME graph objects, filtered).
                    a = (imp.get(split, {}) or {}).get(method)
                    if not a:
                        att_by_i = None
                    elif gl is sl:
                        att_by_i = {int(k): v for k, v in a.items()}      # keyed by split pos
                    else:                                                 # subset: remap by identity
                        ja = {int(k): v for k, v in a.items()}
                        pos = {id(g): i for i, g in enumerate(sl)}
                        att_by_i = {j: ja[pos[id(g)]] for j, g in enumerate(gl)
                                    if id(g) in pos and pos[id(g)] in ja} or None
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
                if method == 'mage_v2':          # save RAW per-node atts (pre fill/mask)
                    mage_atts[split] = {str(i): (t.detach().cpu().view(-1).tolist()
                                        if torch.is_tensor(t) else list(t))
                                        for i, t in att_by_i.items()}
                if args.filtered and args.unk_mode == 'exclude':   # drop UNK atoms from the AUC
                    inst, glob, _ = gtroc_masked(att_by_i, gl)
                    per_split[split] = {'gtroc_instance': inst, 'gtroc_global': glob}
                    continue
                atts = (_fill_unk(att_by_i, gl, 0.5 if args.unk_mode == 'half' else 0.0)
                        if args.filtered else att_by_i)          # zero|half fill UNK for all methods
                fn = _node_att_fn(gl, atts)
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
    ap.add_argument('--filtered', action='store_true',
                    help='FILTERED vocab: build loaders on the _filter variant; UNK-motif atoms '
                         'get the --unk_mode treatment for every method (apples-to-apples).')
    ap.add_argument('--unk_mode', default='zero', choices=['zero', 'half', 'exclude'],
                    help="UNK-atom treatment under --filtered: 'zero' (0.0), 'half' (0.5, matches "
                         "MoSE), or 'exclude' (drop UNK atoms from the AUC entirely).")
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--mage_refit', action='store_true',
                    help='RE-FIT MAGE (patched parser restores ring motifs) instead of loading '
                         'the saved ring-less attention; scores ONLY mage_v2, FG-first setups only.')
    ap.add_argument('--attn_epochs', type=int, default=300,
                    help='MAGE re-fit epochs when --mage_refit (matches fit_mage default)')
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
        # --mage_refit only affects FG-first vocabs (the `ring:`-keyed ones); skip the rest
        if args.mage_refit and _base_frag(meta.get('vocab_variant', '')) not in FG_FIRST:
            continue
        relpath = run_dir.relative_to(args.out_root)
        out_dir = dest_root / relpath
        gtroc_path = out_dir / 'gtroc.json'
        refit_done = out_dir / '_mage_refit.done'
        if args.mage_refit:
            if refit_done.exists():               # resumable: this setup was already re-fit
                skipped += 1
                continue
        elif gtroc_path.exists() and gtroc_path.stat().st_size > 2:   # normal mode: never overwrite
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
            if args.mage_refit and gtroc_path.exists() and gtroc_path.stat().st_size > 2:
                merged = json.loads(gtroc_path.read_text())   # OVERWRITE mage_v2 in place,
                merged.update(gtroc)                          # keep the other methods' cells
                gtroc_path.write_text(json.dumps(merged, indent=2))
            else:
                gtroc_path.write_text(json.dumps(gtroc, indent=2))
            if mage_atts:
                (out_dir / 'mage_v2_node_atts.json').write_text(json.dumps(mage_atts))
            if args.mage_refit:
                refit_done.write_text('')                     # sentinel -> resumable skip
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
