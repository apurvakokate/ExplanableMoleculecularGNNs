#!/usr/bin/env python3
"""coverage_vs_threshold.py — vocab size and node coverage vs support threshold.

Matches the notebook (CreateMotifVocab) exactly:

  Count signal:  weighted_count — flat +1.0 per fragment occurrence in
                 train+val (same as build_vocab in generate_vocab_rules.py).
                 Cutoff: int(thr * N_trainval).

  Global cutoff: int(thr * N_trainval)  where `thr` is the fraction in
                 CHOSEN_THRESHOLD (e.g. 0.002 = 0.2%) and N_trainval is the
                 exact train+val count read from vocab_meta.json. (No /100 — thr
                 is already a fraction; the displayed % is thr*100.)

  Minority pass: for class-imbalanced binary datasets (one class >= 60%),
                 rescue motifs that are frequent in the minority class even
                 if rare globally:
                   mb_cutoff = int(thr * N_minority)
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


# Single source of truth: the same margin used by mining and per-fold re-apply
# (SharedModules/data/threshold_config.py). Imported rather than redefined so the
# phase-2 coverage plots can never drift from the actual threshold policy.
try:
    from SharedModules.data.threshold_config import IMBALANCE_MARGIN
except Exception:  # script run without repo root on sys.path — keep in sync
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

    task_type = meta.get('task_type')
    if task_type is None:
        try:
            from SharedModules.data.dataset_schema import TASK_TYPE
            task_type = TASK_TYPE.get(dataset, 'Classification')
        except ImportError:
            task_type = 'Classification'

    return cols, n_tv, n0_tv, n1_tv, n_total, lookup_tv, lookup_test, task_type
# insert this new function before compute_sweep

def compute_node_coverage(data_lookup, kept_motifs):
    """Fraction of nodes per graph whose motif is in kept_motifs, averaged across graphs.
    Matches old Utils_vocab.compute_node_coverage exactly."""
    return _coverage_stats(data_lookup, kept_motifs)['mean_node_cov']


def _coverage_stats(data_lookup, kept_motifs):
    """Strict coverage / UNKNOWN accounting when a motif not in kept_motifs falls to UNK.

    Returns both views because they answer different questions:
      mean_node_cov   macro: mean over graphs of (covered nodes / nodes)  [the legacy metric]
      micro_node_cov  micro: total covered nodes / total nodes            [atom-weighted]
      n_unk_nodes     ABSOLUTE count of atoms that become UNKNOWN at this threshold
      frac_mols_full  fraction of molecules with ZERO unknown atoms (fully explained)
    A node's motif is v[0] when the lookup stores (motif, ...) tuples, else v."""
    per_graph = []
    n_nodes = n_cov = n_mols = n_full = 0
    for node2motif in data_lookup.values():
        total = len(node2motif)
        if total == 0:
            continue
        covered = sum(
            1 for v in node2motif.values()
            if (v[0] if isinstance(v, tuple) else v) in kept_motifs
        )
        per_graph.append(covered / total)
        n_nodes += total; n_cov += covered
        n_mols += 1; n_full += int(covered == total)
    return {
        'mean_node_cov':  (sum(per_graph) / len(per_graph)) if per_graph else 0.0,
        'micro_node_cov': (n_cov / n_nodes) if n_nodes else 0.0,
        'n_nodes':        n_nodes,
        'n_unk_nodes':    n_nodes - n_cov,
        'n_mols':         n_mols,
        'frac_mols_full': (n_full / n_mols) if n_mols else 0.0,
    }

def compute_sweep(vocab_root, dataset, variant, thresholds=None, large_min=8):
    cols, n_tv, n0_tv, n1_tv, n_total, lookup_tv, lookup_test, task_type = _load(vocab_root, dataset, variant)

    # ── Count signal: weighted_count (flat +1.0 per fragment occurrence) ──
    if 'weighted_count' in cols.columns:
        support = cols['weighted_count']
    elif 'n_occurrences' in cols.columns:
        support = cols['n_occurrences'].astype(float)
    else:
        support = cols['n_mols'].astype(float)

    # ── Fragment-size signal for the singleton / large breakdown ──
    # n_atoms and ring ship in matrix_columns.csv for every variant (rbrics & sfo),
    # so the size distribution of the KEPT vocab needs no SMILES parsing.
    frag_na  = cols['n_atoms'].astype(float) if 'n_atoms' in cols.columns else None
    frag_rng = cols['ring'].astype(bool)     if 'ring'    in cols.columns else None

    # Per-class weighted counts for minority rescue
    wt0 = cols['wt_count_0'] if 'wt_count_0' in cols.columns else None
    wt1 = cols['wt_count_1'] if 'wt_count_1' in cols.columns else None

    total_support = float(support.sum())

    # Infer minority class (classification only — regression has no class split)
    minority = None
    n_minority = None
    if task_type != 'Regression' and n0_tv is not None and n1_tv is not None:
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
        st_tv   = _coverage_stats(lookup_tv,   kept_motifs)
        st_test = _coverage_stats(lookup_test, kept_motifs)
        cov_tv   = st_tv['mean_node_cov']
        cov_test = st_test['mean_node_cov'] if st_test['n_mols'] else float('nan')

        # Fraction of >=1% motifs kept
        if 'above_1pct' in cols.columns:
            a1   = cols['above_1pct'].astype(bool)
            n_a1 = int(a1.sum())
            pck  = float((motif_post & a1).sum()) / n_a1 if n_a1 > 0 else 1.0
        else:
            pck = float('nan')

        # ── Size breakdown of the KEPT vocab at this threshold ──
        #   singleton = 1 heavy atom ; large = >= large_min atoms ; ring = whole ring system.
        if frag_na is not None:
            kept_na       = frag_na[motif_post]
            n_singleton   = int((kept_na == 1).sum())
            n_large       = int((kept_na >= large_min).sum())
            frag_size_med = float(kept_na.median()) if len(kept_na) else float('nan')
            frag_size_avg = float(kept_na.mean())   if len(kept_na) else float('nan')
        else:
            n_singleton = n_large = -1
            frag_size_med = frag_size_avg = float('nan')
        n_ring = int((frag_rng & motif_post).sum()) if frag_rng is not None else -1
        vs = int(motif_post.sum())

        rows.append({
            'threshold':        thr,
            'min_count':        global_cut,
            'mb_cut':           mb_cut,
            'vocab_size_global': int(mask_global.sum()),
            'vocab_size':       vs,
            'n_rescued':        n_rescued,
            'n_singleton':      n_singleton,
            'n_large':          n_large,
            'n_ring':           n_ring,
            'frac_singleton':   round(n_singleton / vs, 4) if (vs > 0 and n_singleton >= 0) else float('nan'),
            'frac_large':       round(n_large / vs, 4) if (vs > 0 and n_large >= 0) else float('nan'),
            'frag_size_median': frag_size_med,
            'frag_size_mean':   round(frag_size_avg, 3) if not np.isnan(frag_size_avg) else float('nan'),
            'large_min':        large_min,
            'coverage_tv':      round(cov_tv,   4),            # macro: mean per-graph node coverage
            'coverage_test':    round(cov_test,  4) if not np.isnan(cov_test) else float('nan'),
            'unk_rate':         round(1 - cov_tv, 4),          # macro node UNK rate (train+val)
            # ── strict UNKNOWN accounting once the threshold re-applies to the vocab ──
            'micro_cov_tv':     round(st_tv['micro_node_cov'], 4),      # atom-weighted coverage
            'n_nodes_tv':       int(st_tv['n_nodes']),
            'n_unk_nodes_tv':   int(st_tv['n_unk_nodes']),             # ATOMS that fall to UNK
            'unk_rate_micro_tv': round(1 - st_tv['micro_node_cov'], 4),
            'frac_mols_full_tv': round(st_tv['frac_mols_full'], 4),    # molecules with 0 UNK atoms
            'micro_cov_test':   round(st_test['micro_node_cov'], 4) if st_test['n_mols'] else float('nan'),
            'n_unk_nodes_test': int(st_test['n_unk_nodes']) if st_test['n_mols'] else -1,
            'frac_mols_full_test': round(st_test['frac_mols_full'], 4) if st_test['n_mols'] else float('nan'),
            'pct_common_kept':  round(pck, 4),
            'minority_class':   minority if minority is not None else -1,
            'n_trainval':       n_tv,
        })

    return pd.DataFrame(rows)


def print_table(df, dataset, variant):
    n_tv  = int(df['n_trainval'].iloc[0])
    minor = int(df['minority_class'].iloc[0])
    minor_str = f'  minority=class{minor}' if minor >= 0 else '  balanced'
    lm = int(df['large_min'].iloc[0]) if 'large_min' in df.columns else 8
    print(f"\n  {dataset} / {variant}  (N_trainval={n_tv:,}{minor_str})")
    print(f"  {'thr%':>8}  {'N_cut':>6}  {'vocab':>6}  {'singl':>6}  {'large>='+str(lm):>8}  "
          f"{'ring':>5}  {'medsz':>5}  {'cov_tv%':>8}  {'unk_nodes':>9}  {'mols_full%':>10}  {'cov_test%':>9}")
    print(f"  {'-'*104}")
    for _, r in df.iterrows():
        flag = ''
        if 0.78 <= r['coverage_tv'] <= 0.82: flag = ' <- ~80%'
        if 0.88 <= r['coverage_tv'] <= 0.92: flag = ' <- ~90%'
        ct = f"{r['coverage_test']*100:8.1f}%" if not np.isnan(r['coverage_test']) else '       N/A'
        msz = f"{r['frag_size_median']:.0f}" if not np.isnan(r.get('frag_size_median', float('nan'))) else '  -'
        unkn = int(r['n_unk_nodes_tv']) if 'n_unk_nodes_tv' in r else -1
        mf   = r.get('frac_mols_full_tv', float('nan'))
        mfs  = f"{mf*100:9.1f}%" if not (isinstance(mf, float) and np.isnan(mf)) else '      N/A'
        print(f"  {r['threshold']*100:7.3f}%  {int(r['min_count']):6d}  "
              f"{int(r['vocab_size']):6d}  {int(r['n_singleton']):6d}  {int(r['n_large']):8d}  "
              f"{int(r['n_ring']):5d}  {msz:>5}  "
              f"{r['coverage_tv']*100:7.1f}%  {unkn:9d}  {mfs}  {ct}{flag}")

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
    fig, axes = plt.subplots(1, 4, figsize=(21, 5))
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

    # Panel 3: kept-vocab size breakdown (distinct / singletons / large / rings)
    ax3 = axes[3]
    lm = int(df['large_min'].iloc[0]) if 'large_min' in df.columns else 8
    if 'n_singleton' in df.columns and (df['n_singleton'] >= 0).any():
        ax3.plot(xi, df['vocab_size'],  'o-', color='steelblue', lw=2,   ms=6, label='distinct (all kept)')
        ax3.plot(xi, df['n_singleton'], 's-', color='crimson',   lw=1.8, ms=5, label='singletons (1 atom)')
        ax3.plot(xi, df['n_large'],     'D-', color='seagreen',  lw=1.8, ms=5, label=f'large (>={lm} atoms)')
        ax3.plot(xi, df['n_ring'],      '^-', color='purple',    lw=1.5, ms=4, alpha=0.7, label='ring systems')
        ax3.set_ylabel('Motif count')
        ax3.set_title('Kept-vocab size breakdown')
        ax3.legend(fontsize=7.5, loc='best')
        ax3.grid(True, alpha=0.3)
        _fmt_x(ax3)
    else:
        ax3.set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_combined_sweep(sweeps: dict, variant: str, out_path: Path):
    """Overlay coverage / vocab curves for multiple datasets on one figure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from matplotlib import cm
    except ImportError:
        print("  matplotlib not available — skipping combined plot")
        return

    if not sweeps:
        print("  [warn] no datasets for combined plot")
        return

    # Align on the union of thresholds (sorted); reindex each df.
    all_thr = sorted({t for df in sweeps.values() for t in df['threshold']})
    aligned = {}
    for ds, df in sweeps.items():
        aligned[ds] = df.set_index('threshold').reindex(all_thr).reset_index()

    xi = list(range(len(all_thr)))
    xlbls = [f"{v*100:.3f}%" for v in all_thr]

    datasets = sorted(sweeps.keys())
    # Fixed palette: same dataset → same color in every panel (cmap(int) is unreliable
    # on resampled/continuous colormaps — values >1 clip to the last color).
    try:
        _palette = list(plt.colormaps['tab10'].colors)
    except AttributeError:
        _palette = [cm.get_cmap('tab10')(i / 9.0) for i in range(10)]
    ds_colors = {ds: _palette[i % len(_palette)] for i, ds in enumerate(datasets)}

    def _fmt_x(ax):
        ax.set_xticks(xi)
        ax.set_xticklabels(xlbls, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Min support threshold (% of N_trainval)', fontsize=9)
        ax.set_xlim(-0.3, len(xi) - 0.7)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(f'All datasets / {variant}', fontsize=13)

    # Panel 0 — vocabulary size
    ax0 = axes[0]
    for ds in datasets:
        df = aligned[ds]
        c = ds_colors[ds]
        ax0.plot(xi, df['vocab_size'], 'o-', color=c, lw=1.8, ms=4, label=ds)
    ax0.set_ylabel('Vocabulary size')
    ax0.set_title('Vocabulary size (+ minority rescue)')
    ax0.legend(fontsize=7, loc='best')
    ax0.grid(True, alpha=0.3)
    _fmt_x(ax0)

    # Panel 1 — train+val node coverage
    ax1 = axes[1]
    for ds in datasets:
        df = aligned[ds]
        c = ds_colors[ds]
        ax1.plot(xi, df['coverage_tv'] * 100, 's-', color=c, lw=1.8, ms=4, label=ds)
    ax1.set_ylim(0, 105)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax1.axhline(80, color='orange', ls='--', alpha=0.6, lw=1)
    ax1.axhline(90, color='red', ls='--', alpha=0.6, lw=1)
    ax1.set_ylabel('Node coverage (%)')
    ax1.set_title('Node coverage (train+val)')
    ax1.legend(fontsize=7, loc='lower left')
    ax1.grid(True, alpha=0.3)
    _fmt_x(ax1)

    # Panel 2 — common motifs retained
    ax2 = axes[2]
    any_common = False
    for ds in datasets:
        df = aligned[ds]
        if df['pct_common_kept'].isna().all():
            continue
        any_common = True
        c = ds_colors[ds]
        ax2.plot(xi, df['pct_common_kept'] * 100, '^-', color=c, lw=1.8, ms=4, label=ds)
    if any_common:
        ax2.set_ylim(0, 105)
        ax2.yaxis.set_major_locator(mticker.MultipleLocator(10))
        ax2.set_ylabel('Common motifs kept (%)')
        ax2.set_title('Fraction of ≥1%-support motifs retained')
        ax2.legend(fontsize=7, loc='lower left')
        ax2.grid(True, alpha=0.3)
        _fmt_x(ax2)
    else:
        ax2.set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved combined: {out_path}")


def _run_single(vocab_root, dataset, variant, out_dir, thresholds, large_min=8):
    df = compute_sweep(vocab_root, dataset, variant, thresholds, large_min=large_min)
    print_table(df, dataset, variant)
    out_dir = Path(out_dir)
    plot_sweep(df, dataset, variant, out_dir / f'{dataset}_{variant}_coverage.png')
    csv = out_dir / f'{dataset}_{variant}_coverage.csv'
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    print(f"  CSV:  {csv}")
    return df


def assemble_regen_curve(regen_root_tmpl, dataset, variant, thresholds, out_dir, large_min=8):
    """For methods whose vocabulary is REGENERATED per threshold (SFO: S_min gates the
    partition), the curve is one point per regenerated vocab, NOT a post-hoc sweep.

    For each threshold X, read the vocab at regen_root_tmpl with '{thr}' -> f'{X:g}',
    take its FULL-vocab measurement (compute_sweep at threshold 0 keeps everything, so
    the reported vocab_size/coverage/singleton/large is exactly what SFO SELECTS at X),
    stamp the real threshold X, and emit a curve CSV+plot in the post-hoc-sweep schema."""
    rows = []
    for thr in thresholds:
        root = regen_root_tmpl.replace('{thr}', f'{thr:g}')
        vdir = Path(root) / dataset / variant
        if not (vdir / 'matrix_columns.csv').exists():
            print(f"  [skip] {dataset}/{variant} @ thr={thr:g}: no vocab at {vdir}")
            continue
        one = compute_sweep(root, dataset, variant, thresholds=[0.0], large_min=large_min)
        r = one.iloc[0].to_dict()
        r['threshold'] = thr
        r['min_count'] = int(thr * int(r['n_trainval']))
        rows.append(r)
    if not rows:
        print(f"  [warn] no regenerated vocabs found for {dataset}/{variant}")
        return None
    df = pd.DataFrame(rows).sort_values('threshold').reset_index(drop=True)
    print_table(df, dataset, variant)
    out_dir = Path(out_dir)
    plot_sweep(df, dataset, variant, out_dir / f'{dataset}_{variant}_coverage.png')
    csv = out_dir / f'{dataset}_{variant}_coverage.csv'
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    print(f"  CSV:  {csv}")
    return df


def plot_variant_compare(out_dir: Path, datasets, variants, tag='rbrics_vs_sfo'):
    """Overlay the per-setup CSVs of >=2 variants for each dataset: coverage, vocab
    size, and singleton fraction on shared threshold axes. Reads the CSVs already
    written by _run_single (no vocab needed), so it runs after the fan-out."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping variant-compare"); return
    out_dir = Path(out_dir)
    loaded = {}   # (ds, var) -> df
    for ds in datasets:
        for var in variants:
            p = out_dir / f'{ds}_{var}_coverage.csv'
            if p.exists():
                loaded[(ds, var)] = pd.read_csv(p)
    ds_have = [ds for ds in datasets if any((ds, v) in loaded for v in variants)]
    if not ds_have:
        print("  [warn] variant-compare: no CSVs found"); return
    vcol = {variants[0]: 'tab:blue', variants[-1]: 'tab:red'}
    for i, v in enumerate(variants):
        vcol.setdefault(v, f'C{i}')
    ncol = min(4, len(ds_have)); nrow = (len(ds_have) + ncol - 1) // ncol
    for metric, ylab, scale in (('coverage_tv', 'Node coverage (train+val) %', 100.0),
                                ('vocab_size', 'Distinct kept motifs', 1.0),
                                ('frac_singleton', 'Singleton fraction %', 100.0)):
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2*ncol, 3.2*nrow), squeeze=False)
        for k, ds in enumerate(ds_have):
            ax = axes[k // ncol][k % ncol]
            for v in variants:
                df = loaded.get((ds, v))
                if df is None or metric not in df.columns:
                    continue
                ax.plot(df['threshold']*100, df[metric]*scale, 'o-', ms=4, lw=1.8,
                        color=vcol[v], label=v.replace('size_frequency_optimization', 'sfo'))
            ax.axvline(0.5, color='0.5', ls=':', lw=1)   # the current 0.5% threshold
            ax.set_title(ds, fontsize=9); ax.grid(True, alpha=0.3)
            ax.set_xlabel('threshold %', fontsize=8)
            if k % ncol == 0: ax.set_ylabel(ylab, fontsize=8)
            if k == 0: ax.legend(fontsize=7)
        for k in range(len(ds_have), nrow*ncol):
            axes[k // ncol][k % ncol].set_visible(False)
        fig.suptitle(f'{tag}: {ylab}', fontsize=12)
        fig.tight_layout()
        p = out_dir / f'compare_{tag}_{metric}.png'
        fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
        print(f"  Saved compare: {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vocab_root', required=True)
    ap.add_argument('--dataset',    default=None,
                    help='Single dataset (legacy mode)')
    ap.add_argument('--datasets',   nargs='*', default=None,
                    help='Multiple datasets; use with --combine_plot for overlay')
    ap.add_argument('--variant',    required=True)
    ap.add_argument('--out_dir',    default='./results/coverage_plots')
    ap.add_argument('--thresholds', nargs='*', type=float, default=None)
    ap.add_argument('--large_min', type=int, default=8,
                    help='heavy-atom count at/above which a kept motif counts as "large"')
    ap.add_argument('--combine_plot', action='store_true',
                    help='Save one PNG overlaying all --datasets on the same axes')
    ap.add_argument('--compare', action='store_true',
                    help='Read already-written per-setup CSVs in --out_dir and overlay '
                         '--variants per dataset (rbrics vs sfo). No vocab needed.')
    ap.add_argument('--variants', nargs='*', default=None,
                    help='variant list for --compare (e.g. rbrics size_frequency_optimization)')
    ap.add_argument('--assemble_regen', action='store_true',
                    help='Assemble a per-threshold-REGENERATED curve (SFO): one point per '
                         'vocab read from --regen_root_tmpl (with {thr}), for --thresholds.')
    ap.add_argument('--regen_root_tmpl', default=None,
                    help="vocab-root template with a literal {thr}, e.g. "
                         "'.../sfo_vocabs/thr{thr}'  ({thr} -> f'{X:g}' per threshold)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    # ── comparison mode: overlay variants from existing CSVs ──
    if args.compare:
        if not args.datasets or not args.variants:
            ap.error('--compare needs --datasets and --variants')
        plot_variant_compare(out_dir, args.datasets, args.variants)
        return

    # ── assemble mode: SFO curve from per-threshold regenerated vocabs ──
    if args.assemble_regen:
        if not args.regen_root_tmpl or not args.thresholds:
            ap.error('--assemble_regen needs --regen_root_tmpl and --thresholds')
        dss = args.datasets or ([args.dataset] if args.dataset else None)
        if not dss:
            ap.error('--assemble_regen needs --dataset or --datasets')
        for ds in dss:
            assemble_regen_curve(args.regen_root_tmpl, ds, args.variant,
                                 args.thresholds, args.out_dir, args.large_min)
        return

    if args.datasets:
        sweeps = {}
        for ds in args.datasets:
            vdir = Path(args.vocab_root) / ds / args.variant
            if not (vdir / 'matrix_columns.csv').exists():
                print(f"  [skip] {ds}/{args.variant} — no vocab (missing matrix_columns.csv)")
                continue
            try:
                sweeps[ds] = _run_single(args.vocab_root, ds, args.variant,
                                         args.out_dir, args.thresholds, args.large_min)
            except FileNotFoundError as e:
                print(f"  [skip] {ds}: {e}", file=sys.stderr)
        if args.combine_plot and sweeps:
            plot_combined_sweep(
                sweeps, args.variant,
                out_dir / f'all_datasets_{args.variant}_coverage.png')
        elif args.combine_plot:
            print("  [warn] --combine_plot: no datasets had vocab output")
        return

    if not args.dataset:
        ap.error('Provide --dataset or --datasets')

    try:
        _run_single(args.vocab_root, args.dataset, args.variant,
                    args.out_dir, args.thresholds, args.large_min)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
