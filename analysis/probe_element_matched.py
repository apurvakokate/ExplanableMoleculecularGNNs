#!/usr/bin/env python3
"""Within-MoSE, element-matched node-identity probe.

QUESTION
    Are atoms MoSE gates DOWN harder to identify (by element) from their node
    embedding than atoms it gates UP -- after controlling for the fact that the
    two buckets contain different elements?

WHY NO VANILLA ARM / NO DiD
    The question is within one model. The vanilla arm in ``probe_vs_vanilla.py``
    controls for "low-gate atoms are intrinsically harder atoms"; stratifying by
    element controls for the same thing inside a single model, and the
    within-element permutation null (below) controls for "the gap is noise".

WHAT IS PROBED
    ``BaseGNN.get_embedding(...)[1]`` -- the conv-stack output, post-ReLU, BEFORE
    pooling. For MoSE (injection 101: w_feat on, w_message off, w_readout on) the
    gate enters this tensor exactly once, scaling the *encoded input features*
    before message passing. ``w_readout`` scales a separate tensor that feeds only
    ``graph_emb`` and never touches ``node_emb``.

    Unlike ``probe_vs_vanilla.py`` this does NOT re-multiply by the gate. That
    multiply is cancelled by L2 normalisation (||g*e|| = g*||e||, verified to
    1e-16), so the pair of operations was a no-op that read as if the gate were
    being applied.

TWO FEATURE ARMS
    l2  : unit-length embeddings  -> is the element still in the DIRECTION?
    raw : embeddings as produced  -> can a linear readout still recover it?
          (MOSE-GNN/config.py sets conv_normalize='none', so the gate's
          shrinking of the input is NOT renormalised away layer by layer; the
          magnitude channel is real and L2 discards it.)

STATISTIC
    Probe = logistic regression fit on MID-gate atoms only (never on the atoms
    it is scored on). Then, per element e that has >= --min_n atoms in BOTH
    buckets AND was present in the probe's training classes:

        gap(e) = P(correct | high, element e) - P(correct | low, element e)

    headline = unweighted mean of gap(e) over qualifying elements.

CEILING (which elements count)
    An element enters the primary headline only if the probe beats PRIOR-GUESSING
    on it, measured on held-out mid atoms. Numerous-but-illegible elements
    otherwise contribute floor noise with full weight and cancel real effects.

INFERENCE
    Closed-form SE of a difference in proportions, summed over strata (see
    ``_gap_se``). A 200-rep within-element permutation null was built first and
    then removed: the formula reproduces it (ratio 0.979, never disagreeing at
    p<0.05) because the ``min_n`` floor licenses the normal approximation, and
    within-molecule correlation cancels out of the within-element difference.

Eval only. Writes three CSVs. Must live in <repo>/analysis/ (sibling import).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.exceptions import ConvergenceWarning
import warnings

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse the loaders VERBATIM from the existing probe -- not reimplemented, not
# replaced. probe_vs_vanilla.py is left untouched.
from probe_vs_vanilla import (            # noqa: E402
    BASE,
    _atom_class,
    _load_model_and_data,
    _node_emb_and_att,
    l2,
    run_dirs,
)

from SharedModules.data.dataset import ATOMS   # noqa: E402

IDX2SYM = {v: k for k, v in ATOMS.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 -- per-atom collection
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def collect(mose, tl, device, max_graphs: int):
    """Walk the test list once; return per-atom (emb, gate, element, degree, graph).

    No gate multiplication is applied to the embedding: see module docstring.
    """
    E, GA, Y, DEG, GID = [], [], [], [], []
    for gi, d in enumerate(tl[:max_graphs]):
        ne, na, x = _node_emb_and_att(mose, d, device, gated=True)
        if ne is None:
            continue
        n = ne.shape[0]
        ei = d.edge_index.cpu().numpy()
        # edge_index is stored both directions, so row 0 counts give the degree.
        deg = np.bincount(ei[0], minlength=n)[:n] if ei.size else np.zeros(n, int)
        E.append(ne)
        GA.append(na)
        Y.append(_atom_class(x))
        DEG.append(deg)
        GID.append(np.full(n, gi, dtype=np.int64))
    if not E:
        return None
    return dict(E=np.concatenate(E), GA=np.concatenate(GA), Y=np.concatenate(Y),
                DEG=np.concatenate(DEG), GID=np.concatenate(GID))


# ─────────────────────────────────────────────────────────────────────────────
# Steps 5-8 -- stratified gap, ceiling, standard error, degree check
# ─────────────────────────────────────────────────────────────────────────────
def _strata_gap(correct, strat, low, high, keys):
    """Mean over strata of  P(correct|high,stratum) - P(correct|low,stratum).

    ``correct`` is a fixed boolean per atom (probe right or wrong).
    Returns (headline, {key: (rec_high, rec_low, gap)}).

    Unequal bucket sizes need no equalising here: ``_gap_se`` prices the
    differing precision of each stratum into the SE of the mean directly.
    """
    per = {}
    gaps = []
    for k in keys:
        m = strat == k
        hi = np.flatnonzero(m & high)
        lo = np.flatnonzero(m & low)
        rh = float(correct[hi].mean())
        rl = float(correct[lo].mean())
        per[k] = (rh, rl, rh - rl)
        gaps.append(rh - rl)
    return (float(np.mean(gaps)) if gaps else float('nan')), per


def _gap_se(per, strat, low, high, keys):
    """Closed-form SE of the element-matched gap.

    Each gap(e) is a difference in two independent proportions:
        var(gap_e) = p_h(1-p_h)/n_h + p_l(1-p_l)/n_l
    The headline is their unweighted mean over K strata, so
        se(headline) = sqrt(sum_e var(gap_e)) / K

    This replaces a 200-rep within-stratum permutation null, which it reproduces:
    over 60 random scenarios se_formula/sd_perm was 0.979 on average (0.847-1.052),
    p-values agreed to 0.024, and the two never disagreed at p<0.05. The
    ``min_n`` floor is what licenses the normal approximation, so the two are
    redundant and the formula is kept.

    Within-molecule correlation does NOT break it. Simulating a shared
    per-molecule difficulty offset (rho = 0 / 0.8 / 1.6) gave se_formula /
    true_sd = 1.054 / 1.024 / 1.017, and a molecule-level bootstrap agreed to
    4 decimals. The high-vs-low comparison happens WITHIN an element and
    within the same molecules, so the molecule offset cancels out of the
    difference -- effectively a paired design. No cluster bootstrap needed.
    """
    v = 0.0
    for k in keys:
        ph, pl, _ = per[k]
        m = strat == k
        nh = int((m & high).sum())
        nl = int((m & low).sum())
        v += ph * (1 - ph) / nh + pl * (1 - pl) / nl
    return float(np.sqrt(v) / len(keys))


def _z_to_p(z):
    """One-sided upper-tail normal p (H1: high > low)."""
    from math import erfc, sqrt
    return float(0.5 * erfc(z / sqrt(2)))


def analyse_arm(X, Y, low, high, mid, DEG, arm, min_n, seed):
    """Fit the probe on MID, then score high-vs-low element-matched. One arm."""
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', ConvergenceWarning)
        clf = LogisticRegression(max_iter=1000).fit(X[mid], Y[mid])
        converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)

    train_classes = set(np.unique(Y[mid]).tolist())
    pred = clf.predict(X)
    correct = (pred == Y)

    # --- held-out MID reference = the CEILING -------------------------------
    # The main probe is fit on ALL of mid, so its accuracy there is training
    # accuracy and inflated. For an honest "how legible is this element at all"
    # number we refit on half of mid and score the other half, both ways. This
    # is reporting only: `clf`, `pred` and `correct` above are untouched, so the
    # high/low comparison is unaffected.
    mid_idx = np.flatnonzero(mid)
    _perm = rng.permutation(len(mid_idx))
    _h1, _h2 = mid_idx[_perm[:len(_perm) // 2]], mid_idx[_perm[len(_perm) // 2:]]
    pred_mid = np.full(len(Y), -1, dtype=np.int64)
    for _tr, _te in ((_h1, _h2), (_h2, _h1)):
        if len(_tr) < 10 or len(np.unique(Y[_tr])) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            pred_mid[_te] = LogisticRegression(max_iter=1000).fit(
                X[_tr], Y[_tr]).predict(X[_te])
    scored_mid = pred_mid >= 0
    correct_mid = scored_mid & (pred_mid == Y)

    def _ceil(e):
        """Held-out mid recall for element e = can the probe read it at all."""
        m = mid & scored_mid & (Y == e)
        return float(correct_mid[m].mean()) if m.sum() else float('nan')

    def _prior(e):
        """Element e's share of the mid bucket = the prior-guessing recall.
        A parameter-free chance floor: a classifier that samples from the label
        prior gets element e right at rate P(e). A ceiling below this means the
        probe has learned nothing about e, so its high/low gap is floor noise."""
        m = mid & scored_mid
        return float((Y[m] == e).mean()) if m.sum() else float('nan')

    # --- Step 6: the old, unstratified number, for contrast only -------------
    keep = np.isin(Y, list(train_classes))
    def _bal(msk):
        m = msk & keep
        return (balanced_accuracy_score(Y[m], pred[m])
                if m.sum() >= 30 else float('nan'))
    bal_h, bal_l = _bal(high), _bal(low)

    # Ceiling over the SAME elements the gap will be averaged over (filled below
    # once `elems` is known); overall held-out mid accuracy for context.
    _mid_sc = mid & scored_mid
    bal_mid = (balanced_accuracy_score(Y[_mid_sc], pred_mid[_mid_sc])
               if _mid_sc.sum() >= 30 else float('nan'))

    # --- Step 5: element strata ---------------------------------------------
    elems = [int(e) for e in np.unique(Y)
             if e in train_classes
             and int((low & (Y == e)).sum()) >= min_n
             and int((high & (Y == e)).sum()) >= min_n]
    row = dict(arm=arm, converged=bool(converged),
               bal_high_unstrat=bal_h, bal_low_unstrat=bal_l,
               gap_unstrat=bal_h - bal_l,
               bal_mid=bal_mid,
               n_elems=len(elems),
               elems=','.join(IDX2SYM.get(e, str(e)) for e in elems))
    per_rows = []
    if not elems:
        row.update(gap_elem=float('nan'),
                   gap_elem_deg=float('nan'), n_strata_deg=0,
                   recall_mid_mean=float('nan'), gap_elem_usable=float('nan'),
                   n_elems_usable=0, elems_usable='',
                   se_elem=float('nan'), p_elem=float('nan'),
                   se_usable=float('nan'), p_usable=float('nan'),
                   skip_reason=f'no element has >={min_n} in both buckets')
        return row, per_rows

    gap_elem, per = _strata_gap(correct, Y, low, high, elems)
    row['gap_elem'] = gap_elem
    row['recall_mid_mean'] = float(np.mean([_ceil(e) for e in elems]))

    # An element enters the PRIMARY headline only if the probe beats prior-
    # guessing on it. Both headlines are reported: a large gap between them
    # means illegible-but-numerous elements were diluting the average.
    usable = [e for e in elems if _ceil(e) > _prior(e)]
    row['n_elems_usable'] = len(usable)
    row['elems_usable'] = ','.join(IDX2SYM.get(e, str(e)) for e in usable)
    row['gap_elem_usable'] = (_strata_gap(correct, Y, low, high, usable)[0]
                              if usable else float('nan'))

    # --- Step 7: standard error + one-sided p (H0: gap = 0) -----------------
    row['se_elem'] = _gap_se(per, Y, low, high, elems)
    row['p_elem'] = _z_to_p(gap_elem / row['se_elem']) if row['se_elem'] > 0 \
        else float('nan')

    # Same, for the PRIMARY (usable-only) statistic.
    if usable:
        _, per_u = _strata_gap(correct, Y, low, high, usable)
        row['se_usable'] = _gap_se(per_u, Y, low, high, usable)
        row['p_usable'] = (_z_to_p(row['gap_elem_usable'] / row['se_usable'])
                           if row['se_usable'] > 0 else float('nan'))
    else:
        row['se_usable'] = row['p_usable'] = float('nan')

    # --- Step 8: (element, degree) strata -----------------------------------
    degb = np.minimum(DEG, 4)
    strat2 = Y.astype(np.int64) * 10 + degb.astype(np.int64)
    _ok = set(usable) if usable else set(elems)
    keys2 = [k for k in np.unique(strat2)
             if int(k // 10) in _ok
             and int((low & (strat2 == k)).sum()) >= min_n
             and int((high & (strat2 == k)).sum()) >= min_n]
    if keys2:
        row['gap_elem_deg'] = _strata_gap(correct, strat2, low, high, keys2)[0]
    else:
        row['gap_elem_deg'] = float('nan')
    row['n_strata_deg'] = len(keys2)

    # --- per-element detail --------------------------------------------------
    for e in elems:
        rh, rl, g = per[e]
        _me = mid & scored_mid & (Y == e)
        per_rows.append(dict(arm=arm, element=IDX2SYM.get(e, str(e)),
                             recall_high=rh, recall_low=rl, gap=g,
                             recall_mid=_ceil(e),
                             prior_mid=_prior(e),
                             usable=bool(_ceil(e) > _prior(e)),
                             n_mid_scored=int(_me.sum()),
                             n_low=int((low & (Y == e)).sum()),
                             n_high=int((high & (Y == e)).sum()),
                             n_mid=int((mid & (Y == e)).sum())))
    return row, per_rows


# ─────────────────────────────────────────────────────────────────────────────
# Per-run driver
# ─────────────────────────────────────────────────────────────────────────────
def process(mose_dir: Path, data_root, vocab_root, device, args):
    ds = mose_dir.parts[mose_dir.parts.index('rbrics_filter') + 1]
    fold = mose_dir.parent.name
    bb = mose_dir.name.split('_')[0]
    tag = dict(dataset=ds, backbone=bb, fold=fold)

    mose, tl, st = _load_model_and_data(mose_dir, data_root, vocab_root, device)
    if st != 'ok':
        return None, None, None, 'mose ' + st

    got = collect(mose, tl, device, args.max_graphs)
    if got is None:
        return None, None, None, 'no embeddings'
    E, GA, Y, DEG = got['E'], got['GA'], got['Y'], got['DEG']

    low = GA < args.lo
    high = GA > args.hi
    mid = (~low) & (~high)

    # Composition over ALL elements, incl. those excluded from the gap.
    comp = [dict(**tag, element=IDX2SYM.get(int(e), str(e)),
                 n_low=int((low & (Y == e)).sum()),
                 n_high=int((high & (Y == e)).sum()),
                 n_mid=int((mid & (Y == e)).sum()))
            for e in np.unique(Y)]

    base = dict(**tag, n_atoms=int(len(GA)), n_low=int(low.sum()),
                n_high=int(high.sum()), n_mid=int(mid.sum()),
                gate_min=float(GA.min()), gate_med=float(np.median(GA)),
                gate_max=float(GA.max()))

    if low.sum() < args.min_n or high.sum() < args.min_n or mid.sum() < 100 \
            or len(np.unique(Y[mid])) < 2:
        # Informative, not a failure: this backbone does not use the extremes.
        r = dict(base, arm='n/a', skip_reason=(
            f'gate extremes too thin (n_low={int(low.sum())} '
            f'n_high={int(high.sum())} n_mid={int(mid.sum())})'))
        return [r], [], comp, 'thin'

    gap_rows, per_rows = [], []
    for arm, X in (('l2', l2(E)), ('raw', E)):
        r, pr = analyse_arm(X, Y, low, high, mid, DEG, arm,
                            args.min_n, args.seed)
        gap_rows.append(dict(base, **r))
        per_rows.extend(dict(**tag, **p) for p in pr)
    return gap_rows, per_rows, comp, 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--vocab_root', required=True)
    ap.add_argument('--max_graphs', type=int, default=800)
    ap.add_argument('--lo', type=float, default=0.1, help='gate < lo  => LOW bucket')
    ap.add_argument('--hi', type=float, default=0.9, help='gate > hi  => HIGH bucket')
    ap.add_argument('--min_n', type=int, default=30,
                    help='an element must have this many atoms in BOTH buckets')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_prefix', default=None)
    a = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G, P, C = [], [], []
    for rd in run_dirs(a.dataset):
        gr, pr, cp, st = process(rd, a.data_root, a.vocab_root, device, a)
        bb = rd.name.split('_')[0]
        fold = rd.parent.name
        if gr is None:
            print(f'  [skip] {bb} {fold}: {st}', flush=True)
            continue
        G.extend(gr)
        P.extend(pr or [])
        C.extend(cp or [])
        if st == 'thin':
            print(f'  [{bb} {fold}] {gr[0]["skip_reason"]}', flush=True)
            continue
        for r in gr:
            print(f'  [{bb} {fold}] {r["arm"]:>3}  '
                  f'unstrat={r["gap_unstrat"]:+.3f}  '
                  f'USABLE={r["gap_elem_usable"]:+.3f}+-{r["se_usable"]:.3f} '
                  f'(p={r["p_usable"]:.4f}, {r["n_elems_usable"]}/{r["n_elems"]} elems)  '
                  f'all_elem={r["gap_elem"]:+.3f}  '
                  f'ceil={r["recall_mid_mean"]:.3f}  '
                  f'elem+deg={r["gap_elem_deg"]:+.3f}  '
                  f'[{r["elems"]}]', flush=True)

    if not G:
        print('no runs')
        return
    pre = a.out_prefix or str(BASE / 'final_v2' / f'probe_elemmatched_{a.dataset}')
    os.makedirs(os.path.dirname(pre) or '.', exist_ok=True)
    pd.DataFrame(G).to_csv(f'{pre}_gap.csv', index=False)
    if P:
        pd.DataFrame(P).to_csv(f'{pre}_perelem.csv', index=False)
    pd.DataFrame(C).to_csv(f'{pre}_composition.csv', index=False)
    print('wrote', f'{pre}_gap.csv', len(G), '| perelem', len(P), '| comp', len(C))


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# Provenance: within-MoSE replacement for the paired MoSE-vs-Vanilla DiD in
# analysis/probe_vs_vanilla.py, which is left untouched and still importable.
# Changes vs that probe: no vanilla arm, no DiD; node_emb is NOT re-multiplied
# by the gate (L2 cancels it); elements are stratified with a min_n floor and a
# prior-beating ceiling test; inference is a closed-form SE.
# ---------------------------------------------------------------------------
