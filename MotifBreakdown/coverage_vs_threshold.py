#!/usr/bin/env python3
"""coverage_vs_threshold.py — vocab size and node coverage vs support threshold.

Matches the notebook (CreateMotifVocab) exactly:

  Count signal:  weighted_count = sum of (1/motif_length) per node-slot, which
                 nets to 1.0 per occurrence — i.e. a plain train+val occurrence
                 count, regardless of motif size. This is the SAME signal the
                 vocab filter (generate_vocab_rules.run_dataset) thresholds on.

  Global cutoff: int(thr * N_trainval)  where `thr` is the fraction in
                 CHOSEN_THRESHOLD (e.g. 0.002 = 0.2%) and N_trainval is the
                 exact train+val count read from vocab_meta.json. (No /100 — thr
                 is already a fraction; the displayed % is thr*100.)

  Minority pass: for class-imbalanced binary datasets (one class >= 60%),
                 rescue motifs that are frequent in the minority class even
                 if rare globally:
                   mb_cutoff = int(thr * N_minority)
                   keep m if wt_count[minority][m] >= mb_cutoff

  Node coverage modes (--coverage_mode):
    graph_average  — mean per-molecule atom fraction (default)
    weighted_atoms — global covered atoms / all atoms (lookup)
    matrix_ratio   — sum(weighted_count kept) / sum(all weighted_count);
                     test uses n_mols_test the same way (legacy docstring)

  Use --compare_coverage_modes to print graph / atoms / matrix_ratio side by side.

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
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

import pickle


IMBALANCE_MARGIN = 0.6

# Production default: graph_average (each molecule counts equally).
# Pass --coverage_mode weighted_atoms to reproduce notebook / legacy curves.
DEFAULT_COVERAGE_MODE = os.environ.get('COVERAGE_MODE', 'graph_average')

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

def compute_node_coverage_graph_average(data_lookup, kept_motifs):
    """Mean per-molecule node fraction (each graph weighted equally)."""
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


def compute_node_coverage_atom_weighted(data_lookup, kept_motifs):
    """Global atom fraction: each atom counts equally across all graphs."""
    n_total = n_covered = 0
    for node2motif in data_lookup.values():
        for v in node2motif.values():
            n_total += 1
            if (v[0] if isinstance(v, tuple) else v) in kept_motifs:
                n_covered += 1
    return n_covered / n_total if n_total > 0 else 0.0


def compute_coverage_matrix_ratio(support, motif_post, cols):
    """Legacy matrix support ratio from matrix_columns (not lookup atoms)."""
    total = float(support.sum())
    if total <= 0:
        return 0.0, float('nan')
    cov_tv = float(support[motif_post].sum()) / total
    if 'n_mols_test' in cols.columns:
        test_occ = cols['n_mols_test'].astype(float)
        total_test = float(test_occ.sum())
        cov_test = (float(test_occ[motif_post].sum()) / total_test
                    if total_test > 0 else float('nan'))
    else:
        cov_test = float('nan')
    return cov_tv, cov_test


def compute_coverage(mode, lookup_tv, lookup_test, kept_motifs, *,
                     support=None, motif_post=None, cols=None):
    """Dispatch coverage metrics by mode."""
    if mode == 'matrix_ratio':
        return compute_coverage_matrix_ratio(support, motif_post, cols)
    if mode == 'weighted_atoms':
        return (compute_node_coverage_atom_weighted(lookup_tv, kept_motifs),
                compute_node_coverage_atom_weighted(lookup_test, kept_motifs))
    return (compute_node_coverage_graph_average(lookup_tv, kept_motifs),
            compute_node_coverage_graph_average(lookup_test, kept_motifs))


def _mode_suffix(coverage_mode: str) -> str:
    return '' if coverage_mode == 'graph_average' else f'_{coverage_mode}'


# Back-compat alias
compute_node_coverage = compute_node_coverage_graph_average

def compute_sweep(vocab_root, dataset, variant, thresholds=None,
                  coverage_mode=None, compare_coverage_modes=False):
    mode = coverage_mode or DEFAULT_COVERAGE_MODE
    cols, n_tv, n0_tv, n1_tv, n_total, lookup_tv, lookup_test, task_type = _load(
        vocab_root, dataset, variant)

    # ── Count signal: weighted_count (1.0 per train+val occurrence) ──
    if 'weighted_count' in cols.columns:
        support = cols['weighted_count']
    elif 'n_occurrences' in cols.columns:
        support = cols['n_occurrences'].astype(float)
    else:
        support = cols['n_mols'].astype(float)

    # Per-class weighted counts for minority rescue
    wt0 = cols['wt_count_0'] if 'wt_count_0' in cols.columns else None
    wt1 = cols['wt_count_1'] if 'wt_count_1' in cols.columns else None

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

        kept_motifs = set(cols.loc[motif_post, 'motif_identity'])
        cov_tv, cov_test = compute_coverage(
            mode, lookup_tv, lookup_test, kept_motifs,
            support=support, motif_post=motif_post, cols=cols)

        if 'above_1pct' in cols.columns:
            a1   = cols['above_1pct'].astype(bool)
            n_a1 = int(a1.sum())
            pck  = float((motif_post & a1).sum()) / n_a1 if n_a1 > 0 else 1.0
        else:
            pck = float('nan')

        row = {
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
            'coverage_mode':    mode,
        }
        if compare_coverage_modes:
            g_tv, g_te = compute_coverage(
                'graph_average', lookup_tv, lookup_test, kept_motifs)
            a_tv, a_te = compute_coverage(
                'weighted_atoms', lookup_tv, lookup_test, kept_motifs)
            m_tv, m_te = compute_coverage_matrix_ratio(support, motif_post, cols)
            row['coverage_tv_graph'] = round(g_tv, 4)
            row['coverage_test_graph'] = round(g_te, 4) if not np.isnan(g_te) else float('nan')
            row['coverage_tv_atoms'] = round(a_tv, 4)
            row['coverage_test_atoms'] = round(a_te, 4) if not np.isnan(a_te) else float('nan')
            row['coverage_tv_matrix'] = round(m_tv, 4)
            row['coverage_test_matrix'] = round(m_te, 4) if not np.isnan(m_te) else float('nan')
        rows.append(row)

    return pd.DataFrame(rows)


def print_table(df, dataset, variant, compare_coverage_modes=False):
    n_tv  = int(df['n_trainval'].iloc[0])
    minor = int(df['minority_class'].iloc[0])
    mode  = df['coverage_mode'].iloc[0] if 'coverage_mode' in df.columns else 'graph_average'
    minor_str = f'  minority=class{minor}' if minor >= 0 else '  balanced'
    print(f"\n  {dataset} / {variant}  (N_trainval={n_tv:,}{minor_str})  "
          f"[coverage_mode={mode}]")

    if compare_coverage_modes and 'coverage_tv_graph' in df.columns:
        print(f"  {'thr%':>8}  {'vocab':>6}  {'graph%':>8}  {'atoms%':>8}  {'matrix%':>8}")
        print(f"  {'-'*52}")
        for _, r in df.iterrows():
            print(f"  {r['threshold']*100:7.3f}%  {int(r['vocab_size']):6d}  "
                  f"{r['coverage_tv_graph']*100:7.1f}%  "
                  f"{r['coverage_tv_atoms']*100:7.1f}%  "
                  f"{r['coverage_tv_matrix']*100:7.1f}%")
        print()
        return

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
    cov_mode = df['coverage_mode'].iloc[0] if 'coverage_mode' in df.columns else 'graph_average'
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'{dataset} / {variant}  [{minor_str}]  cov={cov_mode}', fontsize=13)

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
    ax1.set_ylabel('Coverage (%)')
    cov_titles = {
        'graph_average': 'Node coverage (% atoms in kept motifs, per-graph mean)',
        'weighted_atoms': 'Atom coverage (global atom fraction, lookup)',
        'matrix_ratio': 'Matrix ratio (kept weighted_count / total)',
    }
    ax1.set_title(cov_titles.get(cov_mode, 'Coverage (train+val)'))
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


def write_combined_tables(sweeps: dict, variant: str, out_dir: Path,
                          coverage_mode: str) -> None:
    """Write long + wide CSV summaries for all datasets in one variant."""
    if not sweeps:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    for ds in sorted(sweeps):
        part = sweeps[ds].copy()
        part.insert(0, 'dataset', ds)
        part.insert(1, 'variant', variant)
        part['coverage_mode'] = coverage_mode
        parts.append(part)

    long_df = pd.concat(parts, ignore_index=True)
    long_path = out_dir / f'all_datasets_{variant}{_mode_suffix(coverage_mode)}_coverage.csv'
    long_df.to_csv(long_path, index=False)
    print(f"  Saved combined table: {long_path}")

    for metric, suffix, scale in (
        ('coverage_tv', 'cov_tv_pct', 100.0),
        ('coverage_test', 'cov_test_pct', 100.0),
        ('vocab_size', 'vocab_size', 1.0),
    ):
        if metric == 'coverage_test' and long_df[metric].isna().all():
            continue
        wide = long_df.pivot(index='dataset', columns='threshold', values=metric)
        wide = wide.reindex(sorted(wide.index), axis=0)
        wide = wide.reindex(sorted(wide.columns), axis=1)
        wide.columns = [f'{c * 100:.3f}%' for c in wide.columns]
        if scale != 1.0:
            wide = (wide * scale).round(1)
        else:
            wide = wide.round(0)
        wide_path = out_dir / f'all_datasets_{variant}{_mode_suffix(coverage_mode)}_{suffix}.csv'
        wide.reset_index().to_csv(wide_path, index=False)
        print(f"  Saved combined table: {wide_path}")

    _print_combined_summary(long_df, variant, coverage_mode)


def _print_combined_summary(long_df: pd.DataFrame, variant: str,
                            coverage_mode: str) -> None:
    """Print a compact threshold × dataset coverage matrix to stdout."""
    if long_df.empty:
        return

    piv = long_df.pivot(index='dataset', columns='threshold', values='coverage_tv')
    piv = piv.reindex(sorted(piv.index), axis=0)
    piv = piv.reindex(sorted(piv.columns), axis=1)

    thr_hdrs = [f'{t * 100:6.3f}%' for t in piv.columns]
    ds_w = max(len('dataset'), max(len(str(d)) for d in piv.index))

    print(f"\n  Combined coverage (train+val %, {coverage_mode}) — {variant}")
    print(f"  {'dataset':<{ds_w}}  " + '  '.join(f'{h:>8}' for h in thr_hdrs))
    print(f"  {'-' * (ds_w + 2 + 10 * len(thr_hdrs))}")
    for ds in piv.index:
        cells = []
        for t in piv.columns:
            v = piv.loc[ds, t]
            cells.append(f'{v * 100:7.1f}%' if pd.notna(v) else '      N/A')
        print(f"  {ds:<{ds_w}}  " + '  '.join(cells))
    print()


def _run_single(vocab_root, dataset, variant, out_dir, thresholds,
                coverage_mode=None, compare_coverage_modes=False):
    mode = coverage_mode or DEFAULT_COVERAGE_MODE
    df = compute_sweep(vocab_root, dataset, variant, thresholds,
                       coverage_mode=mode,
                       compare_coverage_modes=compare_coverage_modes)
    print_table(df, dataset, variant, compare_coverage_modes=compare_coverage_modes)
    tag = _mode_suffix(mode)
    cmp_tag = '_compare' if compare_coverage_modes else ''
    out_dir = Path(out_dir)
    plot_sweep(df, dataset, variant,
               out_dir / f'{dataset}_{variant}{tag}{cmp_tag}_coverage.png')
    csv = out_dir / f'{dataset}_{variant}{tag}{cmp_tag}_coverage.csv'
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    print(f"  CSV:  {csv}")
    return df


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
    ap.add_argument('--combine_plot', action='store_true',
                    help='Save one PNG overlaying all --datasets on the same axes')
    ap.add_argument('--coverage_mode',
                    choices=('weighted_atoms', 'graph_average', 'matrix_ratio'),
                    default=None,
                    help='Coverage: graph_average (default), weighted_atoms (lookup), '
                         'or matrix_ratio (sum weighted_count kept / total)')
    ap.add_argument('--compare_coverage_modes', action='store_true',
                    help='Print/plot with graph, atoms, and matrix_ratio columns')
    args = ap.parse_args()

    coverage_mode = args.coverage_mode or DEFAULT_COVERAGE_MODE
    out_dir = Path(args.out_dir)
    tag = _mode_suffix(coverage_mode)
    cmp_tag = '_compare' if args.compare_coverage_modes else ''

    if args.datasets:
        sweeps = {}
        for ds in args.datasets:
            vdir = Path(args.vocab_root) / ds / args.variant
            if not (vdir / 'matrix_columns.csv').exists():
                print(f"  [skip] {ds}/{args.variant} — no vocab (missing matrix_columns.csv)")
                continue
            try:
                sweeps[ds] = _run_single(args.vocab_root, ds, args.variant,
                                         args.out_dir, args.thresholds,
                                         coverage_mode=coverage_mode,
                                         compare_coverage_modes=args.compare_coverage_modes)
            except FileNotFoundError as e:
                print(f"  [skip] {ds}: {e}", file=sys.stderr)
        if args.combine_plot and sweeps:
            plot_combined_sweep(
                sweeps, args.variant,
                out_dir / f'all_datasets_{args.variant}{tag}{cmp_tag}_coverage.png')
            write_combined_tables(sweeps, args.variant, out_dir, coverage_mode)
        elif args.combine_plot:
            print("  [warn] --combine_plot: no datasets had vocab output")
        return

    if not args.dataset:
        ap.error('Provide --dataset or --datasets')

    try:
        _run_single(args.vocab_root, args.dataset, args.variant,
                    args.out_dir, args.thresholds,
                    coverage_mode=coverage_mode,
                    compare_coverage_modes=args.compare_coverage_modes)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
