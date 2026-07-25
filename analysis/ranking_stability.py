#!/usr/bin/env python3
"""ranking_stability.py — is the explainer ranking stable across fragmentations?

The core validity test for treating fragmentation as a *controlled axis*: does the choice
of FG detector (fg_first / ertl_first / rdkit_fg_first) change WHICH explainers look best?
We rank the explainers by GT-ROC within each fragmentation and Spearman-correlate the
rankings across fragmentations. High rho => the conclusions do not depend on the FG list =>
fg_first is a validated axis, not an arbitrary knob.

Input: the results_tidy.csv emitted by aggregate_experiments.py (which already carries the
`fragmentation`, `tier`, `synthetic` and `family` columns). Example:

    python analysis/aggregate_experiments.py --out_root <results> --save_dir <tables> \
        --metrics gt_roc_node_fired_auc_mean pearson_motif spurious_roc_node_auc_mean
    python analysis/ranking_stability.py --tidy <tables>/results_tidy.csv
"""
import argparse
from itertools import combinations

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _spearman(a, b):
    """Spearman rho (+ two-sided p if scipy present)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if _HAVE_SCIPY:
        rho, p = spearmanr(a, b)
        return float(rho), float(p)
    ra = pd.Series(a).rank().to_numpy(); rb = pd.Series(b).rank().to_numpy()
    rho = float(np.corrcoef(ra, rb)[0, 1])
    return rho, float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tidy', required=True, help='results_tidy.csv from aggregate_experiments.py')
    ap.add_argument('--metric', default='gt_roc_node_fired_auc_mean__mean',
                    help='column to rank explainers by (default: Mode-2 node GT-ROC)')
    ap.add_argument('--explainer_col', default='family')
    ap.add_argument('--frag_col', default='fragmentation')
    ap.add_argument('--dataset', default=None, help='restrict to one dataset (optional)')
    ap.add_argument('--tier', default=None, help='restrict to one tier (optional)')
    args = ap.parse_args()

    df = pd.read_csv(args.tidy)
    if args.metric not in df.columns:
        raise SystemExit(f"metric '{args.metric}' not in tidy columns: {list(df.columns)}")
    if 'synthetic' in df.columns:
        df = df[df['synthetic'].astype(str) == 'gt']          # GT-ROC only exists for planted runs
    if args.dataset and 'dataset' in df.columns:
        df = df[df['dataset'] == args.dataset]
    if args.tier and 'tier' in df.columns:
        df = df[df['tier'].astype(str) == args.tier]
    df = df.dropna(subset=[args.metric])
    if df.empty:
        raise SystemExit("no GT-bearing rows with the metric after filtering")

    # mean GT-ROC per (explainer, fragmentation), averaged over tier/backbone/fold
    piv = (df.groupby([args.explainer_col, args.frag_col])[args.metric]
             .mean().unstack(args.frag_col))
    frags = list(piv.columns)
    print(f"=== explainer {args.metric} by fragmentation "
          f"(dataset={args.dataset or 'all'}, tier={args.tier or 'all'}) ===")
    print(piv.round(3).to_string())
    print("\n=== ranking per fragmentation (1 = best GT-ROC) ===")
    print(piv.rank(ascending=False).astype('Int64').to_string())

    print("\n=== Spearman rho of the explainer ranking between fragmentations ===")
    if len(frags) < 2:
        print("  need >=2 fragmentations in the data; found:", frags)
        return
    for a, b in combinations(frags, 2):
        sub = piv[[a, b]].dropna()
        if len(sub) < 3:
            print(f"  {a} vs {b}: too few common explainers ({len(sub)})")
            continue
        rho, p = _spearman(sub[a], sub[b])
        verdict = 'STABLE (ranking robust to FG choice)' if rho >= 0.8 else \
                  ('mostly stable' if rho >= 0.5 else 'UNSTABLE — conclusions depend on the FG list')
        pstr = '' if np.isnan(p) else f", p={p:.3f}"
        print(f"  {a:16} vs {b:16}: rho={rho:+.3f}{pstr}  (n={len(sub)} explainers)  -> {verdict}")


if __name__ == '__main__':
    main()
