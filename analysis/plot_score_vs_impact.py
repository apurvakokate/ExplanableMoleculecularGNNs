#!/usr/bin/env python3
"""plot_score_vs_impact.py — binned impact box-plot grid + count histogram.

Reproduces the "Counts / MOSE / Vanilla / Impact" grid style:
  * x-axis  : learned-score bins (low -> high score)
  * y-axis  : motif IMPACT distribution, drawn as box plots
  * series  : one coloured box per group at each score bin (e.g. model family
              or score-aggregation), placed side-by-side
  * orange  : a COUNT histogram at the bottom of each panel -- how many motifs
              fall in each score bin (the "Counts" series)
  * grid    : one panel per (dataset x facet); facet defaults to vocab variant

The per-bin motif COUNTS are also written to a CSV/markdown table.

Inputs (per run, written by EvalPipeline.to_dataframe):
    score_vs_impact.csv   columns: motif_id, score, impact, abs_disc, ...
Run dirs are discovered by rglob; meta read from sibling summary.json.

Usage
-----
    python analysis/plot_score_vs_impact.py --out_root results \
        --save_dir results/plots --counts_table results/score_impact_counts.csv \
        [--group family] [--facet vocab_variant] [--nbins 6]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_SERIES_COLORS = ['#d62728', '#2ca02c', '#7f7f7f', '#1f77b4', '#9467bd', '#8c564b']
_COUNT_COLOR = '#e8a24c'


def _meta(run_dir: Path):
    fam = dataset = backbone = variant = None
    sj = run_dir / 'summary.json'
    if sj.exists():
        try:
            d = json.load(open(sj))
            dataset, backbone, variant = d.get('dataset'), d.get('backbone'), d.get('vocab_variant')
            # Family from motif_method/model_type (reliable), NOT the path —
            # exp_dir layout is inconsistent (some start with the dataset name).
            mt = (d.get('model_type') or '').lower()
            mm = (d.get('motif_method') or '').lower()
            if 'mose' in mt or mm == 'mose':
                fam = 'mose'
            elif 'motifsat' in mt or 'gsat' in mt or mm in ('readout', 'node_emb', 'motif_emb', 'loss'):
                fam = 'motifsat'
            elif 'vanilla' in mt or mm == 'none':
                fam = 'vanilla'
            else:
                fam = mt or mm or 'unknown'
        except Exception:
            pass
    return fam or 'unknown', dataset or 'unknown', backbone or 'unknown', variant or 'unknown'


def _baseline_explainers(run_dir: Path):
    """List explainer prefixes present in a baseline run dir, by score files."""
    names = set()
    for f in run_dir.glob('*_motif_scores_*.csv'):
        # {explainer}_motif_scores_{mean|max}.csv
        stem = f.name.rsplit('_motif_scores_', 1)[0]
        names.add(stem)
    return sorted(names)


def collect(out_root: Path, agg: str = 'mean') -> pd.DataFrame:
    """Gather per-motif (score, impact) rows for every model family.

    mose / motifsat write score_vs_impact.csv directly. Baseline explainers
    (gnnexplainer, pgexplainer, mage, ...) instead write
    {explainer}_motif_scores_{agg}.csv (score) + motif_impact.csv (impact);
    we join them on motif_id so each explainer becomes its own group, named
    "{explainer}_{agg}" (e.g. gnnexplainer_mean).
    """
    recs = []
    # 1) native score_vs_impact.csv (mose, motifsat)
    for csv in out_root.rglob('score_vs_impact.csv'):
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if df.empty or 'score' not in df or 'impact' not in df:
            continue
        fam, ds, bb, var = _meta(csv.parent)
        df = df.assign(family=fam, dataset=ds, backbone=bb, variant=var,
                       run_dir=str(csv.parent))
        recs.append(df)

    # 2) baseline explainers: join {explainer}_motif_scores_{agg}.csv + motif_impact.csv
    for impact_csv in out_root.rglob('motif_impact.csv'):
        run_dir = impact_csv.parent
        explainers = _baseline_explainers(run_dir)
        if not explainers:
            continue  # only baseline dirs have these score files
        try:
            imp = pd.read_csv(impact_csv)[['motif_id', 'impact']]
        except Exception:
            continue
        _, ds, bb, var = _meta(run_dir)
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
                                   variant=var, run_dir=str(run_dir))
            recs.append(merged)

    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()


def _score_bins(scores: np.ndarray, nbins: int):
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return np.array([0, 1.0])
    edges = np.unique(np.quantile(scores, np.linspace(0, 1, nbins + 1)))
    if len(edges) < 3:
        edges = np.linspace(scores.min(), scores.max() + 1e-9, nbins + 1)
    return edges


def _bin_label(edges, i):
    return f'{edges[i]:.2g}-{edges[i+1]:.2g}'


def plot_panel(ax, ax_top, sub: pd.DataFrame, group_col: str, edges: np.ndarray):
    """Draw impact boxes on `ax` and a per-group motif-count bar strip on
    `ax_top` (a separate axis above the boxes, with a real labeled scale)."""
    groups = sorted(sub[group_col].unique())
    nbin = len(edges) - 1
    width = 0.8 / max(1, len(groups))
    bin_idx = np.clip(np.digitize(sub['score'].values, edges) - 1, 0, nbin - 1)
    sub = sub.assign(_bin=bin_idx)

    for gi, g in enumerate(groups):
        color = _SERIES_COLORS[gi % len(_SERIES_COLORS)]
        data, pos = [], []
        for b in range(nbin):
            vals = sub[(sub[group_col] == g) & (sub['_bin'] == b)]['impact'].dropna().values
            data.append(vals if len(vals) else [np.nan])
            pos.append(b + (gi - (len(groups) - 1) / 2) * width)
        bp = ax.boxplot(data, positions=pos, widths=width * 0.9,
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker='o', markersize=2,
                                        markerfacecolor='lightgray',
                                        markeredgecolor='none', alpha=0.4))
        for box in bp['boxes']:
            box.set(facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.9)
        for med in bp['medians']:
            med.set(color='black', linewidth=1)

    ax.set_xlim(-0.5, nbin - 0.5)
    ax.set_xticks(range(nbin))
    ax.set_xticklabels([_bin_label(edges, i) for i in range(nbin)],
                       rotation=45, fontsize=6, ha='right')
    ax.tick_params(axis='y', labelsize=7)

    # ── per-group motif counts: grouped bars on a dedicated top axis ──────────
    per_group_counts = {}
    for g in groups:
        per_group_counts[g] = [int(((sub[group_col] == g) & (sub['_bin'] == b)).sum())
                               for b in range(nbin)]
    all_counts = [c for gc in per_group_counts.values() for c in gc]
    cmax = max(all_counts) if all_counts and max(all_counts) > 0 else 1
    for gi, g in enumerate(groups):
        color = _SERIES_COLORS[gi % len(_SERIES_COLORS)]
        pos = [b + (gi - (len(groups) - 1) / 2) * width for b in range(nbin)]
        ax_top.bar(pos, per_group_counts[g], width=width * 0.9, color=color,
                   alpha=0.85, edgecolor='white', linewidth=0.3)
        # annotate only bars that are tall enough to avoid label collisions when
        # several groups crowd a bin (e.g. 4 explainer families).
        fs = 5.0 if len(groups) <= 3 else 4.0
        thresh = cmax * (0.12 if len(groups) >= 4 else 0.0)
        for x, c in zip(pos, per_group_counts[g]):
            if c > thresh:
                ax_top.text(x, c + cmax * 0.04, str(c), ha='center', va='bottom',
                            fontsize=fs, color=color, rotation=90 if len(groups) >= 4 else 0)
    ax_top.set_xlim(-0.5, nbin - 0.5)
    ax_top.set_ylim(0, cmax * (1.5 if len(groups) >= 4 else 1.25))
    ax_top.set_xticks([])
    ax_top.set_ylabel('# motifs', fontsize=7)
    ax_top.tick_params(axis='y', labelsize=6)
    ax_top.set_yticks([0, cmax])
    for spine in ('top', 'right'):
        ax_top.spines[spine].set_visible(False)
    return per_group_counts, groups
    return per_group_counts, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--save_dir', default=None)
    ap.add_argument('--counts_table', default=None)
    ap.add_argument('--group', default='backbone',
                    help='Colour side-by-side boxes by this column '
                         '(backbone|family|variant). Default backbone, so the '
                         'different architectures tested show as distinct boxes.')
    ap.add_argument('--facet', default='variant')
    ap.add_argument('--nbins', type=int, default=6)
    ap.add_argument('--agg', default='mean', choices=['mean', 'max'],
                    help='Baseline node-score aggregation to plot (mean|max). '
                         'Affects baseline explainer groups only; outputs are '
                         'suffixed with the agg so mean/max do not overwrite.')
    args = ap.parse_args()

    out_root = Path(args.out_root)
    save_dir = Path(args.save_dir) if args.save_dir else out_root / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)

    df = collect(out_root, agg=args.agg)
    if df.empty:
        print('No score_vs_impact.csv found under', out_root,
              '\n(Train with the updated code so runs emit it.)')
        return

    edges = _score_bins(df['score'].values, args.nbins)
    count_rows = []

    for ds, dsub in df.groupby('dataset'):
        facets = sorted(dsub[args.facet].unique())
        ncol = max(1, len(facets))
        fig = plt.figure(figsize=(4.4 * ncol, 4.3))
        # Two rows per column: a short count strip on top, the box panel below.
        gs = fig.add_gridspec(2, ncol, height_ratios=[1, 3.4], hspace=0.08,
                              wspace=0.28)
        groups = []
        for c, fv in enumerate(facets):
            ax_top = fig.add_subplot(gs[0, c])
            ax = fig.add_subplot(gs[1, c], sharex=ax_top)
            panel = dsub[dsub[args.facet] == fv]
            per_group_counts, groups = plot_panel(ax, ax_top, panel, args.group, edges)
            ax_top.set_title(f'{args.facet}={fv}', fontsize=8)
            if c == 0:
                ax.set_ylabel('motif impact')
            else:
                ax_top.set_ylabel('')
            ax.set_xlabel('learned-score bin')
            for g, gc in per_group_counts.items():
                for b, ct in enumerate(gc):
                    count_rows.append({'dataset': ds, args.facet: fv,
                                       'group': g,
                                       'score_bin': _bin_label(edges, b),
                                       'bin_index': b, 'motif_count': ct})
        handles = [plt.Line2D([0], [0], marker='s', linestyle='',
                              markerfacecolor=_SERIES_COLORS[i % len(_SERIES_COLORS)],
                              markeredgecolor='black', markersize=8, label=str(g))
                   for i, g in enumerate(groups)]
        fig.legend(handles=handles, ncol=len(handles), loc="upper center",
                   bbox_to_anchor=(0.5, 0.95), fontsize=8, frameon=False,
                   title=f'boxes / bars coloured by {args.group}', title_fontsize=8)
        fig.suptitle(f'{ds}: motif impact by learned-score bin  '
                     f'(top strip = # motifs per bin, per {args.group})',
                     fontsize=10, y=1.02)
        out = save_dir / f'score_vs_impact_{ds}_{args.agg}.png'
        fig.savefig(out, dpi=140, bbox_inches='tight')
        plt.close(fig)
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


if __name__ == '__main__':
    main()
