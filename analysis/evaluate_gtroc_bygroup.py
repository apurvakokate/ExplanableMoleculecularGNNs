#!/usr/bin/env python3
"""evaluate_gtroc_bygroup.py — planted GT-ROC split by whether a rule-positive molecule also
holds a "class-1-only" motif outside the rule.

It is analysis/evaluate_gtroc.py with extra columns. Run discovery, loaders, ground truth, the
saved per-atom scores, the keep mask and every AUC come from that script and analysis/evaluate.py
unchanged; evaluate_gtroc.eval_one is called first, and the per-molecule values computed here
must average to the gtroc_{full,filt}_all it returns (asserted for every run, both conditions).
Same command line as evaluate_gtroc.py. Pure read: no model is loaded, nothing is re-run, and
the only file written is --dest.

DEFINITIONS (per run = dataset x rule x method x backbone x fold; all molecules of the fold,
train + valid + test combined, as in gtroc_*_all)
  rule-positive molecule : at least one clause fires (asserted to be exactly the molecules
                           with planted label 1)
  rule motifs            : the motif ids of the atoms flagged in node_label_clauses
  motifs of a molecule   : read under the UNFILTERED vocabulary for every method, so
                           MoSE-GNN (evaluated under the filtered one) shares the grouping
  class-1-only motif     : a motif id, not a rule motif, found in at least one rule-positive
                           molecule and in NO other molecule of the dataset ("pure" motif)
  WITH / WITHOUT         : rule-positive molecules that do / do not hold a pure motif

COLUMNS ADDED (for cond in full, filt; filt drops atoms whose motif is not in the fold's kept
list, exactly as evaluate_gtroc.py does)
  gt_with_<cond>, gt_without_<cond>   mean per-molecule planted GT-ROC (max over fired
                                      clauses) in each group, with n_with_<cond>, n_without_<cond>
  gtx_with_<cond>                     the same molecules scored against the rule EXTENDED by
                                      OR with the pure motifs: each pure motif in the molecule
                                      is one more clause, its atoms leave the negatives, and
                                      the molecule's value is the max over all clauses
  pvr_with_<cond>, n_pvr_<cond>       AUC of the pure-motif atoms against the rule atoms
                                      (the SpurROC contrast with the pure motifs as the
                                      competitor): below 0.5 the rule outranks them
  n_pos, n_pos_with, n_pure_motifs, n_pure_kept
                                      counts on the whole dataset (independent of the method);
                                      n_pure_kept = pure motifs inside the fold's kept list
"""
import numpy as np
import torch

import analysis.evaluate_gtroc as E
from analysis.evaluate import _auc, _keep_fn, _np1

_ORIG_EVAL_ONE = E.eval_one
_ORIG_ATTS_READ = E._atts_read
_ATTS = {}          # one entry: the scores of the run being evaluated (read once, used twice)
_GROUPS = {}        # id(first graph) -> (first graph, group info, n graphs): shared by the backbones of a fold


def _atts_read_once(art_dir, method):
    key = (str(art_dir), method)
    if key not in _ATTS:
        _ATTS.clear()
        _ATTS[key] = _ORIG_ATTS_READ(art_dir, method)
    return _ATTS[key]


def _clauses(g):
    nlc = getattr(g, 'node_label_clauses', None)
    if nlc is None:
        raise ValueError('graph without node_label_clauses: not a planted dataset')
    nlc = nlc.detach().cpu().numpy() if torch.is_tensor(nlc) else np.asarray(nlc)
    return nlc, [c for c in (np.where(nlc[:, j] > 0)[0].tolist()
                             for j in range(nlc.shape[1])) if c]


def _base_graphs(meta, args, gl):
    """The same molecules under the UNFILTERED vocabulary. A run evaluated under <vocab>_filter
    (MoSE-GNN) gets its graphs from a loader that hides the motifs outside the kept list, so
    the pure motifs must be read from the base vocabulary for every method to share one
    grouping. The two lists are checked to be the same molecules, atom for atom."""
    v = str(meta.get('vocab_variant', ''))
    if not v.endswith('_filter'):
        return gl
    base = v[:-len('_filter')]
    split_lists, gt = E._loaders_for({**meta, 'vocab_variant': base, 'gt_vocab_variant': base},
                                     args)
    bl = [g for s_ in E.SPLITS for g in (gt.get(s_) or [])]
    if len(bl) != len(gl):
        raise ValueError(f'base-vocabulary list has {len(bl)} graphs, the run has {len(gl)}')
    for a, b in zip(gl, bl):
        na, nb = _clauses(a)[0], _clauses(b)[0]
        if na.shape != nb.shape or (na > 0).tolist() != (nb > 0).tolist() \
                or int(_np1(a.y)[0]) != int(_np1(b.y)[0]):
            raise ValueError('base-vocabulary graphs do not line up with the run\'s graphs')
    return bl


def _groups(gl):
    """Pure motifs of the dataset and, per graph, the atoms of each pure motif it holds.
    `gl` must be the base-vocabulary graph list (see _base_graphs)."""
    hit = _GROUPS.get(id(gl[0]))
    if hit is not None and hit[0] is gl[0] and hit[2] == len(gl):
        return hit[1]
    n2m = [_np1(getattr(g, 'nodes_to_motifs')).astype(int) for g in gl]
    fired = [_clauses(g)[1] for g in gl]
    pos = [bool(f) for f in fired]
    y = [int(_np1(g.y)[0]) for g in gl]
    if [int(p) for p in pos] != y:
        raise ValueError('rule-positive molecules are not exactly the label-1 molecules: '
                         f'{sum(pos)} fired vs {sum(y)} labelled')
    rule = set()
    for m, f in zip(n2m, fired):
        for c in f:
            rule.update(int(x) for x in m[c])
    in_pos, in_neg = set(), set()
    for m, p in zip(n2m, pos):
        (in_pos if p else in_neg).update(int(x) for x in m if x >= 0)
    pure = in_pos - in_neg - rule
    atoms = []                                   # per graph: {pure motif id: [atom indices]}
    for m, p in zip(n2m, pos):
        d = {}
        if p:
            for k, x in enumerate(m):
                if int(x) in pure:
                    d.setdefault(int(x), []).append(k)
        atoms.append(d)
    info = dict(pos=pos, pure=pure, atoms=atoms,
                n_pos=sum(pos), n_pos_with=sum(1 for d in atoms if d))
    _GROUPS.clear()
    _GROUPS[id(gl[0])] = (gl[0], info, len(gl))
    return info


def _by_group(att_by_i, gl, keep_fn, info):
    """Per-molecule values; the first block repeats evaluate._dnf_gtroc line for line."""
    inst, ext, pvr, group = {}, {}, {}, {}
    for i, g in enumerate(gl):
        a = att_by_i.get(i)
        if a is None:
            continue
        nlc, clauses = _clauses(g)
        if not clauses:
            continue
        att = _np1(a); keep = keep_fn(g)
        U = set().union(*[set(c) for c in clauses])
        neg = [k for k in range(nlc.shape[0]) if k not in U and keep[k]]
        per = [v for v in (_auc(att, [k for k in c if keep[k]], neg) for c in clauses) if v == v]
        if not per:
            continue
        inst[i] = max(per)
        pure_atoms = info['atoms'][i]
        group[i] = bool(pure_atoms)
        if not pure_atoms:
            continue
        P = set().union(*[set(v) for v in pure_atoms.values()])
        neg_x = [k for k in neg if k not in P]
        per_x = [v for v in (_auc(att, [k for k in c if keep[k]], neg_x)
                             for c in clauses + list(pure_atoms.values())) if v == v]
        if per_x:
            ext[i] = max(per_x)
        v = _auc(att, [k for k in sorted(P) if keep[k]], [k for k in sorted(U) if keep[k]])
        if v == v:
            pvr[i] = v
    return inst, ext, pvr, group


def _mean(v):
    v = list(v)
    return float(np.mean(v)) if v else float('nan')


def eval_one(method, run_dir, art_dir, meta, args, device, kept):
    row = _ORIG_EVAL_ONE(method, run_dir, art_dir, meta, args, device, kept)
    split_lists, gt = E._loaders_for(meta, args)
    atts = E._atts_read(art_dir, method)
    gl, abi = E._combined(gt, atts, split_lists)
    info = _groups(_base_graphs(meta, args, gl))
    row.update(n_pos=info['n_pos'], n_pos_with=info['n_pos_with'],
               n_pure_motifs=len(info['pure']),
               n_pure_kept=(len(info['pure'] & kept) if kept is not None else len(info['pure'])))
    for cond, unk in (('full', 'include'), ('filt', 'exclude')):
        keep_fn = _keep_fn(kept if cond == 'filt' else None, unk)
        inst, ext, pvr, group = _by_group(abi, gl, keep_fn, info)
        ref = row[f'gtroc_{cond}_all']
        if not np.isclose(_mean(inst.values()), ref, rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError(f'per-molecule mean {_mean(inst.values())} != gtroc_{cond}_all {ref}')
        w = [i for i in inst if group[i]]
        wo = [i for i in inst if not group[i]]
        row[f'n_with_{cond}'], row[f'n_without_{cond}'] = len(w), len(wo)
        row[f'gt_with_{cond}'] = _mean(inst[i] for i in w)
        row[f'gt_without_{cond}'] = _mean(inst[i] for i in wo)
        row[f'gtx_with_{cond}'] = _mean(ext[i] for i in w if i in ext)
        row[f'n_pvr_{cond}'] = len(pvr)
        row[f'pvr_with_{cond}'] = _mean(pvr.values())
    return row


E.eval_one = eval_one
E._atts_read = _atts_read_once

if __name__ == '__main__':
    E.main()
