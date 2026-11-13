#!/usr/bin/env python3
"""Where in the GNN does hetero-atom identity get lost, and does the gate cause it?

TWO CHANGES vs analysis/probe_element_matched.py (which is left untouched):

1. HETERO-ONLY, TRAINED HETERO-ONLY. Carbon is dropped from the probe's training
   set AND its evaluation. In the all-element probe, carbon is ~75% of atoms, so
   a degenerate low-gate embedding makes the probe emit "carbon" for everything:
   recall_low(C) -> 1 and recall_low(hetero) -> 0 BY CONSTRUCTION. That manufactured
   both the positive hetero gaps and the negative carbon gaps. Restricting the task
   to "which hetero element is this" flattens the prior so there is no 75% class to
   collapse onto. ``frac_pred_majority`` measures what is left of the effect.

2. PER DEPTH. Representations are captured at every layer, so the question becomes
   WHERE identity dies rather than whether it is gone at the end:

     pre : backbone.encode(x)            -- before MoSE touches anything (reference)
     0   : encode(x) * gate              -- w_feat applied, ZERO message passing
     i   : input to convs[i]             -- post-norm, post-ReLU output of layer i
     L   : the returned node_emb         -- what probe_element_matched.py probed

   Captured with forward hooks on backbone.convs[i] (their INPUT is layer i-1's
   output), so SharedModules is not modified and convolve() is not re-implemented.

PRE-REGISTERED PREDICTION (the point of doing this per depth):

   l2 arm, layer 0  ->  gap ~= 0.
       w_feat multiplies by a SCALAR, and l2(g*v) == l2(v) to 1.1e-16, so it cannot
       change direction before message passing. A non-zero gap here falsifies the
       mechanism account.  CAVEAT: for atoms with gate EXACTLY 0 the layer-0 vector
       is the zero vector and the direction is genuinely destroyed -- hence
       ``frac_gate_zero`` is reported per bucket so layer 0 is interpretable.
   raw arm, layer 0 ->  gap LARGE. The magnitude really is scaled down.
   l2 arm, layers 1..L -> gap RISES with depth, as unscaled neighbour messages
       (w_message=False for MoSE) progressively swamp the atom's own contribution.

Eval only. One cell (dataset x backbone x fold) per invocation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import warnings

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_REPO), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from probe_element_matched import (        # noqa: E402
    IDX2SYM, _gap_se, _load_graphs, _strata_gap, _z_to_p, l2, resolve_cell,
)
from probe_vs_vanilla import _atom_class, _load_model_and_data   # noqa: E402
from SharedModules.data.dataset import ATOMS                      # noqa: E402

CARBON = ATOMS['C']


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def layer_reps(model, data, device):
    """Per-layer node representations + gate + element for one graph.

    Returns (dict{layer_name: [N,H]}, gate [N], Y [N], deg [N]) or None.
    """
    model.eval()
    data = data.to(device)
    n = data.x.size(0)
    batch = getattr(data, 'batch', None)
    if batch is None:
        batch = torch.zeros(n, dtype=torch.long, device=device)

    # --- gate (node attention) from the model's own forward ------------------
    node_att = None
    out = model(data.x, data.edge_index, batch,
                getattr(data, 'nodes_to_motifs', None),
                getattr(data, 'edge_attr', None))
    if isinstance(out, (tuple, list)) and len(out) >= 2 and out[1] is not None:
        node_att = out[1].view(-1)
        if len(out) >= 3 and isinstance(out[2], dict) and \
                out[2].get('node_att_soft') is not None:
            node_att = out[2]['node_att_soft'].view(-1)
    if node_att is None:
        return None

    backbone = model
    for attr in ('backbone', 'clf'):
        cand = getattr(model, attr, None)
        if cand is not None and hasattr(cand, 'get_embedding'):
            backbone = cand
            break
    if not hasattr(backbone, 'get_embedding'):
        return None

    reps = {}
    # 'pre': before any gating. encode() is a public method; no hook needed.
    reps['pre'] = backbone.encode(data.x).detach()

    handles = []
    def _mk(i):
        def _hook(_mod, inp, _out):
            reps[str(i)] = inp[0].detach()     # input to conv i == layer i output
        return _hook
    for i, conv in enumerate(backbone.convs):
        handles.append(conv.register_forward_hook(_mk(i)))
    try:
        _, node_emb = backbone.get_embedding(
            data.x, data.edge_index, node_att=node_att.unsqueeze(-1),
            w_feat=bool(getattr(model, 'w_feat', False)),
            w_message=bool(getattr(model, 'w_message', False)),
            w_readout=bool(getattr(model, 'w_readout', False)),
            batch=batch)
    finally:
        for h in handles:
            h.remove()
    reps[str(len(backbone.convs))] = node_emb.detach()

    ei = data.edge_index.cpu().numpy()
    deg = np.bincount(ei[0], minlength=n)[:n] if ei.size else np.zeros(n, int)
    return ({k: v.cpu().numpy() for k, v in reps.items()},
            node_att.cpu().numpy(), _atom_class(data.x.cpu().numpy()), deg)


def collect(model, graphs, device, max_graphs=0):
    acc, GA, Y, DEG = {}, [], [], []
    for d in (graphs[:max_graphs] if max_graphs else graphs):
        r = layer_reps(model, d, device)
        if r is None:
            continue
        reps, ga, y, deg = r
        for k, v in reps.items():
            acc.setdefault(k, []).append(v)
        GA.append(ga); Y.append(y); DEG.append(deg)
    if not GA:
        return None
    return (({k: np.concatenate(v) for k, v in acc.items()}),
            np.concatenate(GA), np.concatenate(Y), np.concatenate(DEG))


# ─────────────────────────────────────────────────────────────────────────────
def analyse(X, Y, low, high, mid, layer, arm, min_n, seed):
    """Hetero-only element-matched gap for one (layer, arm)."""
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', ConvergenceWarning)
        clf = LogisticRegression(max_iter=1000).fit(X[mid], Y[mid])
        conv = not any(issubclass(x.category, ConvergenceWarning) for x in w)
    train_classes = set(np.unique(Y[mid]).tolist())
    pred = clf.predict(X)
    correct = (pred == Y)

    # majority class of the PROBE'S TRAINING SET -- what a collapsed probe emits
    vals, cnts = np.unique(Y[mid], return_counts=True)
    maj = int(vals[cnts.argmax()])

    # held-out mid ceiling (2-fold inside mid; main probe untouched)
    mi = np.flatnonzero(mid); perm = rng.permutation(len(mi))
    h1, h2 = mi[perm[:len(perm)//2]], mi[perm[len(perm)//2:]]
    pmid = np.full(len(Y), -1, dtype=np.int64)
    for tr, te in ((h1, h2), (h2, h1)):
        if len(tr) < 10 or len(np.unique(Y[tr])) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            pmid[te] = LogisticRegression(max_iter=1000).fit(X[tr], Y[tr]).predict(X[te])
    sm = pmid >= 0
    cmid = sm & (pmid == Y)

    def ceil(e):
        m = mid & sm & (Y == e)
        return float(cmid[m].mean()) if m.sum() else float('nan')

    def prior(e):
        m = mid & sm
        return float((Y[m] == e).mean()) if m.sum() else float('nan')

    row = dict(layer=layer, arm=arm, converged=bool(conv),
               majority_elem=IDX2SYM.get(maj, str(maj)),
               frac_pred_majority_low=float((pred[low] == maj).mean()),
               frac_pred_majority_high=float((pred[high] == maj).mean()),
               frac_pred_majority_mid=float((pred[mid] == maj).mean()))

    elems = [int(e) for e in np.unique(Y)
             if e in train_classes
             and int((low & (Y == e)).sum()) >= min_n
             and int((high & (Y == e)).sum()) >= min_n]
    usable = [e for e in elems if ceil(e) > prior(e)]
    row.update(n_elems=len(elems), n_elems_usable=len(usable),
               elems=','.join(IDX2SYM.get(e, str(e)) for e in elems),
               elems_usable=','.join(IDX2SYM.get(e, str(e)) for e in usable))
    per = []
    for keys, tag in ((elems, 'elem'), (usable, 'usable')):
        if not keys:
            row[f'gap_{tag}'] = row[f'se_{tag}'] = row[f'p_{tag}'] = float('nan')
            continue
        gp, d = _strata_gap(correct, Y, low, high, keys)
        se = _gap_se(d, Y, low, high, keys)
        row[f'gap_{tag}'] = gp
        row[f'se_{tag}'] = se
        row[f'p_{tag}'] = _z_to_p(gp / se) if se > 0 else float('nan')
        if tag == 'elem':
            for e in keys:
                rh, rl, g = d[e]
                per.append(dict(layer=layer, arm=arm, element=IDX2SYM.get(e, str(e)),
                                recall_high=rh, recall_low=rl, gap=g,
                                recall_mid=ceil(e), prior_mid=prior(e),
                                usable=bool(ceil(e) > prior(e)),
                                n_low=int((low & (Y == e)).sum()),
                                n_high=int((high & (Y == e)).sum()),
                                n_mid=int((mid & (Y == e)).sum())))
    row['recall_mid_mean'] = float(np.mean([ceil(e) for e in usable])) if usable else float('nan')
    return row, per


def process(rd, args, device):
    ds = rd.parts[rd.parts.index('rbrics_filter') + 1]
    fold = rd.parent.name
    bb = rd.name.split('_')[0]
    tag = dict(dataset=ds, backbone=bb, fold=fold)

    model, test_list, st = _load_model_and_data(rd, args.data_root, args.vocab_root, device)
    if st != 'ok':
        raise SystemExit(f'FATAL {ds}/{bb}/{fold}: {st}')
    lists = {}
    if 'test' in args.splits:
        lists['test'] = test_list
    extra = [s for s in args.splits if s != 'test']
    if extra:
        lists.update(_load_graphs(rd, args.data_root, args.vocab_root, extra))

    got = {}
    for sp in args.splits:
        if sp in lists:
            g = collect(model, lists[sp], device, args.max_graphs)
            if g is not None:
                got[sp] = g
    if not got:
        raise SystemExit(f'FATAL {ds}/{bb}/{fold}: no embeddings')

    units = list(got.items())
    if len(got) > 1:
        keys = list(got.values())[0][0].keys()
        units.append(('pooled', (
            {k: np.concatenate([v[0][k] for v in got.values()]) for k in keys},
            np.concatenate([v[1] for v in got.values()]),
            np.concatenate([v[2] for v in got.values()]),
            np.concatenate([v[3] for v in got.values()]))))

    rows, pers = [], []
    for sp, (reps, GA, Y, DEG) in units:
        het = Y != CARBON                      # <- carbon excluded entirely
        low = (GA < args.lo) & het
        high = (GA > args.hi) & het
        mid = het & ~(GA < args.lo) & ~(GA > args.hi)
        base = dict(**tag, split=sp,
                    n_hetero=int(het.sum()), n_all=int(len(GA)),
                    n_low=int(low.sum()), n_high=int(high.sum()), n_mid=int(mid.sum()),
                    frac_gate_zero_low=float((GA[low] == 0).mean()) if low.sum() else float('nan'),
                    frac_gate_zero_all=float((GA == 0).mean()))
        if low.sum() < args.min_n or high.sum() < args.min_n or mid.sum() < 100 \
                or len(np.unique(Y[mid])) < 2:
            rows.append(dict(base, layer='-', arm='-', skip_reason=(
                f'hetero extremes too thin (n_low={int(low.sum())} '
                f'n_high={int(high.sum())} n_mid={int(mid.sum())} '
                f'classes={len(np.unique(Y[mid]))})')))
            continue
        for layer in sorted(reps, key=lambda k: (k != 'pre', k)):
            E = reps[layer]
            for arm, X in (('l2', l2(E)), ('raw', E)):
                r, pr = analyse(X, Y, low, high, mid, layer, arm, args.min_n, args.seed)
                rows.append(dict(base, **r))
                pers.extend(dict(**tag, split=sp, **p) for p in pr)
    return rows, pers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--backbone', required=True, choices=['GCN', 'GIN', 'GAT', 'SAGE', 'PNA'])
    ap.add_argument('--fold', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--vocab_root', required=True)
    ap.add_argument('--splits', default='all')
    ap.add_argument('--max_graphs', type=int, default=0)
    ap.add_argument('--lo', type=float, default=0.1)
    ap.add_argument('--hi', type=float, default=0.9)
    ap.add_argument('--min_n', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', required=True)
    a = ap.parse_args()
    V = ('train', 'valid', 'test')
    a.splits = list(V) if a.splits.strip().lower() == 'all' else \
        [s.strip() for s in a.splits.split(',') if s.strip()]
    bad = [s for s in a.splits if s not in V]
    if bad:
        ap.error(f'unknown split(s) {bad}')
    a.splits = [s for s in V if s in a.splits]

    rd = resolve_cell(a.dataset, a.backbone, a.fold)
    print(f'cell={a.dataset}/{a.backbone}/{a.fold} splits={",".join(a.splits)}', flush=True)
    print(f'run_dir={rd}', flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows, pers = process(rd, a, device)

    for r in rows:
        if r.get('layer') == '-':
            print(f'  [{r["split"]:>6}] {r["skip_reason"]}', flush=True)
            continue
        if r['split'] != 'pooled':
            continue
        print(f'  [{r["split"]:>6}] L{r["layer"]:>3} {r["arm"]:>3}  '
              f'gap={r["gap_usable"]:+.3f}+-{r["se_usable"]:.3f} p={r["p_usable"]:.4f}  '
              f'ceil={r["recall_mid_mean"]:.3f}  '
              f'majpred lo/hi={r["frac_pred_majority_low"]:.2f}/{r["frac_pred_majority_high"]:.2f}'
              f'  [{r["elems_usable"]}]', flush=True)

    out = Path(a.out_dir) / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    stem = f'{a.backbone}_{a.fold}'
    pd.DataFrame(rows).to_csv(out / f'{stem}_depth.csv', index=False)
    if pers:
        pd.DataFrame(pers).to_csv(out / f'{stem}_depthelem.csv', index=False)
    (out / f'{stem}.done').write_text(f'{a.dataset}/{a.backbone}/{a.fold}\nrows={len(rows)}\n')
    print(f'OK -> {out}/{stem}_depth.csv  rows={len(rows)}', flush=True)


if __name__ == '__main__':
    main()
