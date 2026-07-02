#!/usr/bin/env python3
"""plot_score_vs_impact.py — binned impact box-plot grid + count histogram.

One figure **per (dataset, algorithm, label-regime)** with a subplot grid:
  * columns : GNN backbone (architecture)
  * rows    : vocab variant (fragmentation / threshold)
  * x-axis  : equal-width learned-score bins (not quantile bins), on the
              algorithm's own score range by default
  * y-axis  : motif IMPACT distribution (box plot)
  * top strip : motif count per score bin (orange bars)

Real-label and relabelled/GT runs are kept in SEPARATE figures (``_real`` /
``_gt`` in the filename) — never pooled into one box.

MOSE / MotifSAT write ``score_vs_impact.csv`` directly (GSAT / Vanilla only if
the run emits it). Baseline explainers (GNNExplainer, PGExplainer, MAGE) join
``{explainer}_motif_scores_{agg}.csv`` with ``motif_impact.csv``; each
explainer is its own algorithm figure.

Usage
-----
    python analysis/plot_score_vs_impact.py --out_root results \\
        --save_dir results/plots --nbins 6 [--agg mean|max]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis.aggregate_experiments import FAMILIES, resolve_family

_COUNT_COLOR = '#e8a24c'
_BOX_COLOR = '#1f77b4'

# Preferred row / column order (unknown values sort alphabetically after these).
_VARIANT_ORDER = (
    'rbrics_old', 'rbrics', 'rbrics_filter',
    'all_fallback_bpe', 'all_fallback_bpe_filter',
)
_BACKBONE_ORDER = ('GIN', 'GCN', 'GraphSAGE', 'SAGE', 'GAT', 'PNA')

_FAMILY_TITLES = {
    'mose': 'MOSE-GNN',
    'motifsat': 'MotifSAT',
    'gsat': 'GSAT',
    'base_gsat': 'GSAT',
    'vanilla': 'Vanilla GNN',
    'gnnexplainer': 'GNNExplainer',
    'pgexplainer': 'PGExplainer',
    'mage': 'MAGE',
}


def _meta(run_dir: Path):
    fam = dataset = backbone = variant = None
    synthetic = 'real'
    for part in run_dir.parts:
        if part in FAMILIES:
            fam = 'gsat' if part == 'base_gsat' else part
            break
    sj = run_dir / 'summary.json'
    if sj.exists():
        try:
            with open(sj, encoding='utf-8') as f:
                d = json.load(f)
            dataset, backbone, variant = (
                d.get('dataset'), d.get('backbone'), d.get('vocab_variant'))
            # use_gt from summary is authoritative for real vs relabelled — the
            # vocab_variant field is stripped of '_relabelled', so it cannot be
            # inferred from the variant (would pool real + relabelled runs).
            synthetic = 'gt' if bool(d.get('use_gt')) else 'real'
            if not fam:
                fam = resolve_family(d, str(run_dir))
        except Exception:
            pass
    return (fam or 'unknown', dataset or 'unknown', backbone or 'unknown',
            variant or 'unknown', synthetic)


def _baseline_explainers(run_dir: Path):
    """List explainer prefixes present in a baseline run dir, by score files."""
    names = set()
    for f in run_dir.glob('*_motif_scores_*.csv'):
        stem = f.name.rsplit('_motif_scores_', 1)[0]
        names.add(stem)
    return sorted(names)


def collect(out_root: Path, agg: str = 'mean',
            datasets: set[str] | None = None) -> pd.DataFrame:
    """Gather per-motif (score, impact) rows for every model family."""
    from analysis.aggregate_experiments import dataset_allowed

    recs = []
    for csv in out_root.rglob('score_vs_impact.csv'):
        if datasets and not dataset_allowed(csv.parent, datasets):
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if df.empty or 'score' not in df or 'impact' not in df:
            continue
        fam, ds, bb, var, syn = _meta(csv.parent)
        df = df.assign(family=fam, dataset=ds, backbone=bb, variant=var,
                       synthetic=syn, run_dir=str(csv.parent))
        recs.append(df)

    for impact_csv in out_root.rglob('motif_impact.csv'):
        run_dir = impact_csv.parent
        if datasets and not dataset_allowed(run_dir, datasets):
            continue
        explainers = _baseline_explainers(run_dir)
        if not explainers:
            continue
        try:
            imp = pd.read_csv(impact_csv)[['motif_id', 'impact']]
        except Exception:
            continue
        _, ds, bb, var, syn = _meta(run_dir)
        for expl in explainers:
            score_file = run_dir / f'{expl}_motif_scores_{agg}.csv'
            if not score_file.exists():
                continue
            try:
                sc = pd.read_csv(score_file)
            except Exception:
                continue
            score_col = f'score_{agg}' if f'score_{agg}' in sc else (
                'score' if 'score' in sc else None)
            if score_col is None or 'motif_id' not in sc:
                continue
            merged = sc[['motif_id', score_col]].merge(imp, on='motif_id', how='inner')
            merged = merged.rename(columns={score_col: 'score'})
            if merged.empty:
                continue
            merged = merged.assign(family=expl, dataset=ds, backbone=bb,
                                   variant=var, synthetic=syn, run_dir=str(run_dir))
            recs.append(merged)

    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()


def _ordered(values, preferred: tuple[str, ...]) -> list[str]:
    pref = {v: i for i, v in enumerate(preferred)}
    return sorted(set(values), key=lambda x: (pref.get(x, len(preferred)), str(x)))


def _score_bins_equal(score_min: float, score_max: float, nbins: int) -> np.ndarray:
    """Equal-width bin edges on [score_min, score_max]."""
    if not np.isfinite(score_min):
        score_min = 0.0
    if not np.isfinite(score_max) or score_max <= score_min:
        score_max = score_min + 1.0
    return np.linspace(score_min, score_max, nbins + 1)


def _family_score_range(scores: np.ndarray,
                        score_min: float | None = None,
                        score_max: float | None = None) -> tuple[float, float]:
    """Shared x-axis range for one algorithm.

    Defaults to that algorithm's OWN data range (``[0, data_max]`` for
    non-negative scores, else ``[data_min, data_max]``) so small-scale
    explainers (e.g. GNNExplainer, max ~0.06) are not crushed against a
    fixed ``[0, 1]`` axis. Pass ``--score_min`` / ``--score_max`` to override
    (e.g. ``--score_max 1`` to force the full sigmoid range)."""
    finite = scores[np.isfinite(scores)]
    if len(finite):
        data_lo, data_hi = float(np.min(finite)), float(np.max(finite))
    else:
        data_lo, data_hi = 0.0, 1.0

    lo = float(score_min) if score_min is not None else (0.0 if data_lo >= 0.0 else data_lo)
    hi = float(score_max) if score_max is not None else data_hi
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _bin_label(edges, i):
    return f'{edges[i]:.2g}-{edges[i+1]:.2g}'


def _assign_bins(scores: np.ndarray, edges: np.ndarray) -> np.ndarray:
    nbin = len(edges) - 1
    # Slightly extend the top edge so score == score_max lands in the last bin.
    edges_ext = edges.copy()
    edges_ext[-1] += 1e-9
    return np.clip(np.digitize(scores, edges_ext) - 1, 0, nbin - 1)


def plot_cell(ax, ax_top, cell: pd.DataFrame, edges: np.ndarray):
    """Single-run panel: impact boxes + motif-count strip."""
    nbin = len(edges) - 1
    if cell.empty:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes, fontsize=8, color='0.5')
        ax.set_xlim(-0.5, nbin - 0.5)
        ax_top.set_xlim(-0.5, nbin - 0.5)
        ax_top.set_ylim(0, 1)
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        return [0] * nbin

    sub = cell.dropna(subset=['score', 'impact']).copy()
    if sub.empty:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes, fontsize=8, color='0.5')
        ax.set_xlim(-0.5, nbin - 0.5)
        ax_top.set_xlim(-0.5, nbin - 0.5)
        ax_top.set_ylim(0, 1)
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        return [0] * nbin

    sub['_bin'] = _assign_bins(sub['score'].values, edges)
    data, pos = [], []
    counts = []
    for b in range(nbin):
        vals = sub.loc[sub['_bin'] == b, 'impact'].values
        data.append(vals if len(vals) else [np.nan])
        pos.append(float(b))
        counts.append(int((sub['_bin'] == b).sum()))

    bp = ax.boxplot(data, positions=pos, widths=0.55,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker='o', markersize=2,
                                    markerfacecolor='lightgray',
                                    markeredgecolor='none', alpha=0.4))
    for box in bp['boxes']:
        box.set(facecolor=_BOX_COLOR, edgecolor='black', linewidth=0.5, alpha=0.85)
    for med in bp['medians']:
        med.set(color='black', linewidth=1)

    ax.set_xlim(-0.5, nbin - 0.5)
    ax.set_xticks(range(nbin))
    ax.set_xticklabels([_bin_label(edges, i) for i in range(nbin)],
                       rotation=45, fontsize=6, ha='right')
    ax.tick_params(axis='y', labelsize=7)

    cmax = max(counts) if counts and max(counts) > 0 else 1
    ax_top.bar(pos, counts, width=0.55, color=_COUNT_COLOR,
               alpha=0.9, edgecolor='white', linewidth=0.3)
    for x, c in zip(pos, counts):
        if c > 0:
            ax_top.text(x, c + cmax * 0.04, str(c), ha='center', va='bottom',
                        fontsize=5.5, color='#8a5a10')
    ax_top.set_xlim(-0.5, nbin - 0.5)
    ax_top.set_ylim(0, cmax * 1.35)
    ax_top.set_xticks([])
    ax_top.set_ylabel('#', fontsize=6, labelpad=1)
    ax_top.tick_params(axis='y', labelsize=5, length=2)
    ax_top.set_yticks([0, cmax] if cmax > 0 else [0])
    for spine in ('top', 'right'):
        ax_top.spines[spine].set_visible(False)
    return counts


def _family_title(family: str) -> str:
    return _FAMILY_TITLES.get(family, family.replace('_', ' ').title())


def _safe_slug(text: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in text)


def plot_algorithm_figure(
    ds: str,
    family: str,
    synthetic: str,
    fsub: pd.DataFrame,
    *,
    edges: np.ndarray,
    save_dir: Path,
    agg: str,
    count_rows: list[dict],
) -> Path | None:
    """Render one (dataset × algorithm × label-regime) grid and return the path."""
    variants = _ordered(fsub['variant'].unique(), _VARIANT_ORDER)
    backbones = _ordered(fsub['backbone'].unique(), _BACKBONE_ORDER)
    if not variants or not backbones:
        return None

    nrow, ncol = len(variants), len(backbones)
    fig = plt.figure(figsize=(3.6 * ncol, 3.1 * nrow))
    outer = fig.add_gridspec(nrow, ncol, hspace=0.42, wspace=0.28)

    for ri, var in enumerate(variants):
        for ci, bb in enumerate(backbones):
            inner = outer[ri, ci].subgridspec(2, 1, height_ratios=[1, 3.2], hspace=0.06)
            ax_top = fig.add_subplot(inner[0, 0])
            ax = fig.add_subplot(inner[1, 0], sharex=ax_top)
            cell = fsub[(fsub['variant'] == var) & (fsub['backbone'] == bb)]
            counts = plot_cell(ax, ax_top, cell, edges)

            if ri == 0:
                ax_top.set_title(str(bb), fontsize=9, pad=2)
            if ci == 0:
                ax.set_ylabel(f'{var}\nimpact', fontsize=7)
            if ri == nrow - 1:
                ax.set_xlabel('score bin', fontsize=7)
            else:
                ax.set_xlabel('')
                plt.setp(ax.get_xticklabels(), visible=False)

            for b, ct in enumerate(counts):
                count_rows.append({
                    'dataset': ds,
                    'family': family,
                    'synthetic': synthetic,
                    'variant': var,
                    'backbone': bb,
                    'score_bin': _bin_label(edges, b),
                    'bin_index': b,
                    'motif_count': ct,
                })

    lo, hi = float(edges[0]), float(edges[-1])
    _reg = {'real': 'real labels', 'gt': 'relabelled / GT'}.get(synthetic, synthetic)
    fig.suptitle(
        f'{ds} — {_family_title(family)}  ({_reg})\n'
        f'motif impact by equal-width score bins [{lo:.3g}, {hi:.3g}]',
        fontsize=11, y=1.01,
    )
    out = save_dir / f'score_vs_impact_{ds}_{_safe_slug(family)}_{synthetic}_{agg}.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out


def write_demo_runs(out_root: Path) -> None:
    """Create a tiny synthetic run tree for smoke-testing the plot layout."""
    rng = np.random.default_rng(0)
    datasets = ('BBBP', 'mutag')
    families = {
        'mose': {'score_scale': 1.0},
        'motifsat': {'score_scale': 1.0},
        'gnnexplainer': {'score_scale': 1.0, 'baseline': True},
    }
    variants = ('rbrics', 'all_fallback_bpe')
    backbones = ('GIN', 'GCN')

    for ds in datasets:
        for fam, opts in families.items():
            for var in variants:
                for bb in backbones:
                    run_dir = out_root / fam / ds / 'fold0' / var / f'{bb}_demo'
                    run_dir.mkdir(parents=True, exist_ok=True)
                    summary = {
                        'dataset': ds,
                        'backbone': bb,
                        'vocab_variant': var,
                        'motif_method': (
                            'mose' if fam == 'mose'
                            else 'readout' if fam == 'motifsat'
                            else 'none'),
                    }
                    (run_dir / 'summary.json').write_text(json.dumps(summary))

                    n = 120
                    scores = rng.uniform(0, opts['score_scale'], n)
                    impacts = 0.15 * scores + rng.normal(0, 0.04, n)
                    motif_ids = np.arange(n)
                    pd.DataFrame({
                        'motif_id': motif_ids,
                        'impact': impacts,
                    }).to_csv(run_dir / 'motif_impact.csv', index=False)

                    if opts.get('baseline'):
                        pd.DataFrame({
                            'motif_id': motif_ids,
                            'score_mean': scores,
                        }).to_csv(run_dir / f'{fam}_motif_scores_mean.csv', index=False)
                    else:
                        pd.DataFrame({
                            'motif_id': motif_ids,
                            'score': scores,
                            'impact': impacts,
                            'abs_disc': rng.uniform(0, 0.5, n),
                        }).to_csv(run_dir / 'score_vs_impact.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--save_dir', default=None)
    ap.add_argument('--counts_table', default=None)
    ap.add_argument('--nbins', type=int, default=6)
    ap.add_argument('--score_min', type=float, default=None,
                    help='Shared score-axis minimum per algorithm figure '
                         '(default: 0 when scores are non-negative, else data min).')
    ap.add_argument('--score_max', type=float, default=None,
                    help="Shared score-axis maximum per algorithm figure "
                         "(default: THIS algorithm's data max, so small-scale "
                         "explainers are not crushed; pass 1 to force full [0,1]).")
    ap.add_argument('--agg', default='mean', choices=['mean', 'max'],
                    help='Baseline node-score aggregation (mean|max).')
    ap.add_argument('--dataset', nargs='*', default=None,
                    help='Only plot these dataset(s), e.g. --dataset mutag BBBP')
    ap.add_argument('--demo', action='store_true',
                    help='Write synthetic demo runs under out_root/demo_runs '
                         'and plot from there (smoke test).')
    args = ap.parse_args()

    out_root = Path(args.out_root)
    if args.demo:
        demo_root = out_root / 'demo_runs'
        write_demo_runs(demo_root)
        out_root = demo_root
        print('wrote demo runs under', demo_root)

    save_dir = Path(args.save_dir) if args.save_dir else out_root / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)

    datasets = set(args.dataset) if args.dataset else None
    df = collect(out_root, agg=args.agg, datasets=datasets)
    if df.empty:
        print('No score_vs_impact.csv (or baseline score files) found under',
              out_root)
        raise SystemExit(1)

    count_rows: list[dict] = []
    written: list[Path] = []

    # One shared score range PER ALGORITHM (across all its datasets/regimes), so
    # every figure for that algorithm uses the same x-axis and small-scale
    # explainers get their own range instead of a fixed [0, 1].
    fam_range = {
        family: _family_score_range(
            fsub['score'].values, score_min=args.score_min, score_max=args.score_max)
        for family, fsub in df.groupby('family')
    }

    # Separate real-label from relabelled/GT runs — never pool them in one box.
    for (ds, family, synthetic), fsub in df.groupby(['dataset', 'family', 'synthetic']):
        lo, hi = fam_range[family]
        edges = _score_bins_equal(lo, hi, args.nbins)
        out = plot_algorithm_figure(
            ds, family, synthetic, fsub,
            edges=edges,
            save_dir=save_dir,
            agg=args.agg,
            count_rows=count_rows,
        )
        if out is not None:
            written.append(out)
            print('wrote', out)

    if count_rows:
        ct_df = pd.DataFrame(count_rows)
        ct_path = (Path(args.counts_table) if args.counts_table
                   else save_dir / f'score_impact_counts_{args.agg}.csv')
        ct_df.to_csv(ct_path, index=False)
        print('wrote', ct_path, f'({len(ct_df)} rows)')
        try:
            ct_df.to_markdown(str(ct_path).replace('.csv', '.md'), index=False)
            print('wrote', str(ct_path).replace('.csv', '.md'))
        except Exception:
            pass

    if not written:
        print('No figures produced (check dataset / family filters).')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
