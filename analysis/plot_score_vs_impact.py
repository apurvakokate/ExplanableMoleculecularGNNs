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
import re
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

from analysis.aggregate_experiments import ARCHIVE_PREFIXES, FAMILIES, resolve_family

_COUNT_COLOR = '#e8a24c'
_BOX_COLOR = '#1f77b4'

# Preferred row / column order (unknown values sort alphabetically after these).
_VARIANT_ORDER = (
    'rbrics_old', 'rbrics', 'rbrics_filter',
    'rbrics_protected', 'rbrics_protected_filter',
    'all_fallback_bpe', 'all_fallback_bpe_filter',
    'all_fallback_bpe_protected', 'all_fallback_bpe_protected_filter',
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


def _path_excluded(rel_parts: tuple[str, ...]) -> bool:
    return any(p.startswith(ARCHIVE_PREFIXES) for p in rel_parts)


def _meta(run_dir: Path):
    fam = dataset = backbone = variant = fold = None
    synthetic = 'real'
    m = re.search(r'fold(\d+)', str(run_dir))
    if m:
        fold = int(m.group(1))
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
            if d.get('fold') is not None:
                fold = int(d['fold'])
        except Exception:
            pass
    return (fam or 'unknown', dataset or 'unknown', backbone or 'unknown',
            variant or 'unknown', synthetic, fold)


def collect(out_root: Path | list[Path], agg: str = 'mean',
            datasets: set[str] | None = None,
            impact_kind: str = 'own') -> pd.DataFrame:
    """Gather per-(motif, graph) (score, impact) rows for every model family.

    ``impact_kind`` selects which baseline impact to plot when a per-explainer
    table carries both (``method`` column): ``'own'`` = the explainer's own
    leave-one-out, ``'agnostic'`` = the original uniform-weight impact. Motif-
    aware models (MOSE/MotifSAT) have a single impact and are unaffected."""
    from analysis.aggregate_experiments import dataset_allowed

    roots = [out_root] if isinstance(out_root, Path) else list(out_root)
    recs = []
    for root in roots:
        root = Path(root)
        for csv in root.rglob('score_vs_impact.csv'):
            rel = csv.relative_to(root)
            if _path_excluded(rel.parts):
                continue
            if datasets and not dataset_allowed(csv.parent, datasets):
                continue
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue
            if df.empty or 'score' not in df or 'impact' not in df:
                continue
            fam, ds, bb, var, syn, fold = _meta(csv.parent)
            df = df.assign(family=fam, dataset=ds, backbone=bb, variant=var,
                           synthetic=syn, fold=fold, run_dir=str(csv.parent),
                           results_root=str(root))
            recs.append(df)

        # Baselines: each post-hoc explainer writes its OWN instance-level
        # score-vs-impact table (its attribution used as the LOO weight vector
        # W → the explainer's own per-graph impact), one file per (explainer,
        # agg). Read the file matching the requested agg so the plotted impact
        # is the explainer's own LOO impact — NOT the vanilla model's shared,
        # explainer-agnostic motif_impact.csv (graph-removal) it used to reuse.
        for expl in ('gnnexplainer', 'pgexplainer', 'mage'):
            for svi in root.rglob(f'{expl}_{agg}_score_vs_impact.csv'):
                run_dir = svi.parent
                rel = svi.relative_to(root)
                if _path_excluded(rel.parts):
                    continue
                if datasets and not dataset_allowed(run_dir, datasets):
                    continue
                try:
                    df = pd.read_csv(svi)
                except Exception:
                    continue
                if df.empty or 'score' not in df or 'impact' not in df:
                    continue
                # Long-form file carries both impact definitions tagged by
                # 'method'; keep only the requested one.
                if 'method' in df.columns:
                    df = df[df['method'] == impact_kind]
                    if df.empty:
                        continue
                _, ds, bb, var, syn, fold = _meta(run_dir)
                df = df.assign(family=expl, dataset=ds, backbone=bb, variant=var,
                               synthetic=syn, fold=fold, run_dir=str(run_dir),
                               results_root=str(root))
                recs.append(df)

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
    has_mid = 'motif_id' in sub.columns
    for b in range(nbin):
        in_bin = sub['_bin'] == b
        # Boxplot is instance-level: every (motif, graph) impact is a point.
        vals = sub.loc[in_bin, 'impact'].values
        data.append(vals if len(vals) else [np.nan])
        pos.append(float(b))
        # Count strip stays a *motif* count (unique motifs in the bin), not the
        # per-graph instance count, so the label remains accurate.
        counts.append(int(sub.loc[in_bin, 'motif_id'].nunique()) if has_mid
                      else int(in_bin.sum()))

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
    impact_kind: str = 'own',
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
            cell_folds = (
                sorted(int(f) for f in cell['fold'].dropna().unique())
                if 'fold' in cell.columns else [])
            fold_label = ','.join(map(str, cell_folds)) if cell_folds else ''

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
                    'fold': fold_label,
                    'score_bin': _bin_label(edges, b),
                    'bin_index': b,
                    'motif_count': ct,
                })

    lo, hi = float(edges[0]), float(edges[-1])
    _reg = {'real': 'real labels', 'gt': 'relabelled / GT'}.get(synthetic, synthetic)
    _kind = {'own': "explainer's own LOO",
             'agnostic': 'uniform-weight (original)'}.get(impact_kind, impact_kind)
    fig.suptitle(
        f'{ds} — {_family_title(family)}  ({_reg}) — impact: {_kind}\n'
        f'motif impact by equal-width score bins [{lo:.3g}, {hi:.3g}]',
        fontsize=11, y=1.01,
    )
    out = (save_dir /
           f'score_vs_impact_{ds}_{_safe_slug(family)}_{synthetic}_{agg}_{impact_kind}.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--extra_out_root', nargs='*', default=None,
                    help='additional result trees to include (e.g. results_motifsat_ib)')
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
    ap.add_argument('--impact_kind', default='own', choices=['own', 'agnostic'],
                    help="Baseline impact to plot: 'own' (each explainer's own "
                         "leave-one-out) or 'agnostic' (original uniform-weight "
                         'impact, shared across explainers). Motif-aware models '
                         'are unaffected.')
    ap.add_argument('--dataset', nargs='*', default=None,
                    help='Only plot these dataset(s), e.g. --dataset mutag BBBP')
    args = ap.parse_args()

    out_root = Path(args.out_root)

    save_dir = Path(args.save_dir) if args.save_dir else Path(args.out_root) / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)

    datasets = set(args.dataset) if args.dataset else None
    roots = [out_root]
    for p in args.extra_out_root or []:
        ep = Path(p)
        if ep not in roots:
            roots.append(ep)
    df = collect(roots, agg=args.agg, datasets=datasets,
                 impact_kind=args.impact_kind)
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
            impact_kind=args.impact_kind,
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
