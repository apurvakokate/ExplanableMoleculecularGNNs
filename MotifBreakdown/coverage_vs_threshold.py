#!/usr/bin/env python3
"""coverage_vs_threshold.py — vocab size and node coverage vs support threshold.

Matches the notebook (CreateMotifVocab) exactly:

  Count signal:  weighted_count = sum of (1/motif_length) per node-slot.
                 One motif occurrence always contributes 1.0, regardless of size.

  Global cutoff: int(pct/100 * N_trainval)  where N_trainval is the exact
                 train+val count read from vocab_meta.json.

  Minority pass: for class-imbalanced binary datasets (one class >= 60%),
                 rescue motifs that are frequent in the minority class even
                 if rare globally:
                   mb_cutoff = int(pct/100 * N_minority)
                   keep m if wt_count[minority][m] >= mb_cutoff

  Node coverage: fraction of train+val nodes assigned to a kept motif,
                 computed as (sum of weighted_count for kept motifs) /
                 (sum of weighted_count for all motifs).
                 Test coverage uses the same kept-motif set applied to
                 test-set atom occurrences (approximated from n_mols_test
                 column if available, else reported as N/A).

  Plot:          x = threshold % of N_trainval (bottom) + molecule count (top)
                 Panel 0: vocabulary size (linear)
                 Panel 1: train+val coverage and test coverage on same axes
                 Panel 2: fraction of >=1%-support motifs retained

X-axis range: 0.01% to 1.0% (matches CONSTANTS.PERCENT_THRESHOLDS range).

Usage:
    python coverage_vs_threshold.py \\
        --vocab_root ./vocab_output \\
        --dataset Mutagenicity \\
        --variant rbrics \\
        --out_dir ./results/coverage_plots
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

import pickle


IMBALANCE_MARGIN = 0.6

def _lp(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def _load(vocab_root, dataset, variant):
    vdir = Path(vocab_root) / dataset / variant
    cols_path = vdir / 'matrix_columns.csv'
    meta_path = vdir / 'vocab_meta.json'

    if not cols_path.exists():
        raise FileNotFoundError(f"Not found: {cols_path}")

    cols = pd.read_csv(cols_path)

    # Load exact split sizes from vocab_meta.json
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    n_tv  = meta.get('n_trainval')
    n_total = meta.get('n_total', n_tv)
    n0_tv = meta.get('n0_trainval')
    n1_tv = meta.get('n1_trainval')

    # Fallback if vocab_meta.json predates this change
    if n_tv is None:
        n_tv = int(cols['n_mols'].max()) if 'n_mols' in cols else 1
        print(f"  [warn] vocab_meta.json not found — using support.max()={n_tv} as N_trainval. "
              f"Re-run phase1 to get exact counts.")
        n0_tv = n1_tv = None

    base        = str(vdir / f'{dataset}_{variant}')
    lookup_tv   = _lp(Path(base + '_graph_lookup.pickle'))
    valid_path  = Path(base + '_valid_graph_lookup.pickle')
    if valid_path.exists():
        lookup_tv = {**lookup_tv, **_lp(valid_path)}
    lookup_test = _lp(Path(base + '_test_graph_lookup.pickle'))

    return cols, n_tv, n0_tv, n1_tv, n_total, lookup_tv, lookup_test
# insert this new function before compute_sweep

def compute_node_coverage(data_lookup, kept_motifs):
    """Fraction of nodes per graph whose motif is in kept_motifs, averaged across graphs.
    Matches old Utils_vocab.compute_node_coverage exactly."""
    coverages = []
    for node2motif in data_lookup.values():
        total = len(node2motif)
        if total == 0:
            continue
        covered = sum(
            1 for v in node2motif.values()
            if (v[0] if isinstance(v, tuple) else v) in kept_motifs
        )
        coverages.append(covered / total)
    return sum(coverages) / len(coverages) if coverages else 0.0

def compute_sweep(vocab_root, dataset, variant, thresholds=None):
    cols, n_tv, n0_tv, n1_tv, n_total, lookup_tv, lookup_test = _load(vocab_root, dataset, variant)

    # ── Count signal: weighted_count (notebook: Σ 1/length per node-slot) ──
    if 'weighted_count' in cols.columns:
        support = cols['weighted_count']
    elif 'n_occurrences' in cols.columns:
        support = cols['n_occurrences'].astype(float)
    else:
        support = cols['n_mols'].astype(float)

    # Per-class weighted counts for minority rescue
    wt0 = cols['wt_count_0'] if 'wt_count_0' in cols.columns else None
    wt1 = cols['wt_count_1'] if 'wt_count_1' in cols.columns else None

    total_support = float(support.sum())

    # Infer minority class
    minority = None
    n_minority = None
    if n0_tv is not None and n1_tv is not None:
        r0, r1 = n0_tv / n_tv, n1_tv / n_tv
        if r0 >= IMBALANCE_MARGIN:
            minority, n_minority = 1, n1_tv
        elif r1 >= IMBALANCE_MARGIN:
            minority, n_minority = 0, n0_tv

    if thresholds is None:
        thresholds = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]

    rows = []
    for thr in thresholds:
        # Use n_tv (train+val) — not n_total — as the denominator so that
        # the displayed threshold % matches the actual cutoff semantics.
        # n_total includes test molecules which are never seen during vocab selection.
        global_cut = int(thr * n_tv)

        # Global pass: keep motifs with weighted_count >= global_cutoff
        mask_global = support >= global_cut
        motif_post  = mask_global.copy()

        # Minority rescue pass
        mb_cut = 0
        if minority is not None and n_minority is not None and wt0 is not None:
            mb_cut   = int(thr * n_minority)
            wt_min   = wt1 if minority == 1 else wt0
            motif_post = mask_global | (wt_min >= mb_cut)

        n_rescued = int(motif_post.sum()) - int(mask_global.sum())

        # # Node coverage: weighted sum for kept motifs / total weighted sum
        # cov_tv = float(support[motif_post].sum()) / total_support if total_support > 0 else 0.0

        # # Test coverage: n_mols_test is the per-motif test occurrence count.
        # # Denominator = total test occurrences across ALL motifs (kept + filtered).
        # if 'n_mols_test' in cols.columns:
        #     test_occ   = cols['n_mols_test'].astype(float)
        #     total_test = float(test_occ.sum())
        #     cov_test   = float(test_occ[motif_post].sum()) / total_test if total_test > 0 else 0.0
        # else:
        #     cov_test = float('nan')
        kept_motifs = set(cols.loc[motif_post, 'motif_identity'])
        cov_tv   = compute_node_coverage(lookup_tv,   kept_motifs)
        cov_test = compute_node_coverage(lookup_test, kept_motifs)

        # Fraction of >=1% motifs kept
        if 'above_1pct' in cols.columns:
            a1   = cols['above_1pct'].astype(bool)
            n_a1 = int(a1.sum())
            pck  = float((motif_post & a1).sum()) / n_a1 if n_a1 > 0 else 1.0
        else:
            pck = float('nan')

        rows.append({
            'threshold':        thr,
            'min_count':        global_cut,
            'mb_cut':           mb_cut,
            'vocab_size_global': int(mask_global.sum()),
            'vocab_size':       int(motif_post.sum()),
            'n_rescued':        n_rescued,
            'coverage_tv':      round(cov_tv,   4),
            'coverage_test':    round(cov_test,  4) if not np.isnan(cov_test) else float('nan'),
            'unk_rate':         round(1 - cov_tv, 4),
            'pct_common_kept':  round(pck, 4),
            'minority_class':   minority if minority is not None else -1,
            'n_trainval':       n_tv,
        })

    return pd.DataFrame(rows)


def print_table(df, dataset, variant):
    n_tv  = int(df['n_trainval'].iloc[0])
    minor = int(df['minority_class'].iloc[0])
    minor_str = f'  minority=class{minor}' if minor >= 0 else '  balanced'
    print(f"\n  {dataset} / {variant}  (N_trainval={n_tv:,}{minor_str})")
    print(f"  {'thr%':>8}  {'N_cut':>6}  {'vocab':>6}  {'rescued':>7}  "
          f"{'cov_tv%':>8}  {'cov_test%':>9}  {'common%':>8}")
    print(f"  {'-'*65}")
    for _, r in df.iterrows():
        flag = ''
        if 0.78 <= r['coverage_tv'] <= 0.82: flag = ' <- ~80%'
        if 0.88 <= r['coverage_tv'] <= 0.92: flag = ' <- ~90%'
        ct = f"{r['coverage_test']*100:8.1f}%" if not np.isnan(r['coverage_test']) else '       N/A'
        print(f"  {r['threshold']*100:7.3f}%  {int(r['min_count']):6d}  "
              f"{int(r['vocab_size']):6d}  {int(r['n_rescued']):7d}  "
              f"{r['coverage_tv']*100:7.1f}%  {ct}  "
              f"{r['pct_common_kept']*100:7.1f}%{flag}")

    cands = df[df['coverage_tv'] >= 0.80]
    if not cands.empty:
        b = cands.iloc[-1]
        print(f"\n  Suggested: {b['threshold']*100:.3f}% "
              f"= {int(b['min_count'])} molecules  "
              f"vocab={int(b['vocab_size'])}  "
              f"coverage={b['coverage_tv']*100:.1f}%"
              f"  rescued={int(b['n_rescued'])}")
        print(f"  -> Set THRESHOLD={b['threshold']:.4f} in experiment_config.sh")
    print()


def plot_sweep(df, dataset, variant, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("  matplotlib not available — text table only")
        return

    n_tv  = int(df['n_trainval'].iloc[0])
    minor = int(df['minority_class'].iloc[0])
    has_test = not df['coverage_test'].isna().all()

    xi    = list(range(len(df)))
    xlbls = [f"{v*100:.3f}%" for v in df['threshold']]
    mcnts = [str(int(v)) for v in df['min_count']]

    def _fmt_x(ax):
        ax.set_xticks(xi)
        ax.set_xticklabels(xlbls, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Min support threshold (% of N_trainval)', fontsize=9)
        ax.set_xlim(-0.3, len(xi) - 0.7)
        top = ax.twiny()
        top.set_xlim(ax.get_xlim())
        top.set_xticks(xi)
        top.set_xticklabels(mcnts, fontsize=7.5)
        top.set_xlabel(f'Min support count  (N_trainval={n_tv:,})', fontsize=9)

    minor_str = f'minority=class{minor}' if minor >= 0 else 'balanced'
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'{dataset} / {variant}  [{minor_str}]', fontsize=13)

    # Panel 0: vocabulary size (linear)
    ax0 = axes[0]
    ax0.plot(xi, df['vocab_size_global'], 'o--', color='steelblue', lw=1.5, ms=5,
             alpha=0.5, label='global only')
    ax0.plot(xi, df['vocab_size'], 'o-', color='steelblue', lw=2, ms=6,
             label='+ minority rescue')
    vmax = int(df['vocab_size'].max())
    mag  = max(0, len(str(vmax)) - 2)
    step = max(1, round(vmax / 8 / 10**mag) * 10**mag)
    ax0.yaxis.set_major_locator(mticker.MultipleLocator(step))
    ax0.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax0.set_ylabel('Vocabulary size')
    ax0.set_title('Vocabulary size')
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)
    _fmt_x(ax0)

    # Panel 1: node coverage (train+val and test)
    ax1 = axes[1]
    ax1.plot(xi, df['coverage_tv']*100, 's-', color='seagreen', lw=2, ms=6,
             label='train+val')
    if has_test:
        ax1.plot(xi, df['coverage_test']*100, 's--', color='seagreen', lw=1.5, ms=5,
                 alpha=0.6, label='test')
    ax1.set_ylim(0, 105)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax1.axhline(80, color='orange', ls='--', alpha=0.7, label='80%')
    ax1.axhline(90, color='red',    ls='--', alpha=0.7, label='90%')
    ax1.legend(fontsize=8)
    ax1.set_ylabel('Node coverage (%)')
    ax1.set_title('Node coverage  (% atoms in known motifs)')
    ax1.grid(True, alpha=0.3)
    _fmt_x(ax1)

    # Panel 2: common motifs retained
    ax2 = axes[2]
    if not df['pct_common_kept'].isna().all():
        ax2.plot(xi, df['pct_common_kept']*100, '^-', color='darkorange', lw=2, ms=6)
        ax2.set_ylim(0, 105)
        ax2.yaxis.set_major_locator(mticker.MultipleLocator(10))
        ax2.set_ylabel('Common motifs kept (%)')
        ax2.set_title('Fraction of >=1%-support motifs retained')
        ax2.grid(True, alpha=0.3)
        _fmt_x(ax2)
    else:
        axes[2].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vocab_root', required=True)
    ap.add_argument('--dataset',    required=True)
    ap.add_argument('--variant',    required=True)
    ap.add_argument('--out_dir',    default='./results/coverage_plots')
    ap.add_argument('--thresholds', nargs='*', type=float, default=None)
    args = ap.parse_args()

    try:
        df = compute_sweep(args.vocab_root, args.dataset, args.variant, args.thresholds)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print_table(df, args.dataset, args.variant)

    out = Path(args.out_dir) / f'{args.dataset}_{args.variant}_coverage.png'
    plot_sweep(df, args.dataset, args.variant, out)

    csv = Path(args.out_dir) / f'{args.dataset}_{args.variant}_coverage.csv'
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    print(f"  CSV:  {csv}")


if __name__ == '__main__':
    main()
