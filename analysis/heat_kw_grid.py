#!/usr/bin/env python3
"""(k, width) heatmap as a dataset x backbone GRID (no dataset pooling).

Rows = datasets (BBBP, Mutagenicity, hERG); columns = backbones (GAT/GCN/PNA/SAGE).
Each of the 12 panels is a heatmap: y = methods, x = (k,width) buckets (width-major),
cell = mean GT-ROC x100 over the rules in that (dataset, backbone, bucket), folds
averaged. Sequential viridis (CB-safe), linear scale clipped to the observed range,
chance marked on the shared colorbar. Per-dataset bucket rule-counts are shown in the
row labels (they differ by dataset).

Usage: python heat_kw_grid.py --root <planted_v2> --out <dir> [--methods mose gsat mage gnnexplainer]
"""
import argparse, csv, glob, json, os, statistics as st
from collections import defaultdict

DS = ['BBBP', 'Mutagenicity', 'hERG']
BB = ['GAT', 'GCN', 'PNA', 'SAGE']
PR = {'mose': 'MoSE', 'gsat': 'GSAT', 'gnnexplainer': 'GNNExpl',
      'mage': 'MAGE', 'motif_occlusion': 'Occl', 'pgexplainer': 'PGExpl'}
KW = [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)]  # width-major


def fl(x):
    try:
        v = float(x); return v if v == v else None
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', default='paper_deliverables')
    ap.add_argument('--methods', nargs='+',
                    default=['mose', 'gsat', 'mage', 'gnnexplainer', 'pgexplainer'])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import Normalize
    import matplotlib.cm as mcm
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.linewidth': 0.6,
                         'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    CMAP = 'viridis'

    kw = {}
    for d in DS:
        t = json.load(open(f'{a.root}/{d}/_shared/vocab/{d}/rbrics/rule_tiers.json'))
        for rule, r in t.items():
            kw[(d, rule)] = (r['k'], max(len(c['motifs']) for c in r['clauses']))
    cell = defaultdict(list)
    for f in glob.glob(f'{a.root}/*/*/eval/metrics_*.csv'):
        for r in csv.DictReader(open(f)):
            if r['method'] not in a.methods or r['unk'] != 'exclude':
                continue
            v = fl(r.get('gtroc_instance'))
            if v is not None:
                cell[(r['dataset'], r['gt_rule'], r['method'], r['backbone'])].append(v)
    cm = {k: st.mean(v) for k, v in cell.items() if v}

    buckets = [b for b in KW if sum(1 for k in kw if kw[k] == b) > 0]
    # per-dataset bucket counts
    dsn = {d: [sum(1 for (dd, rl) in kw if dd == d and kw[(dd, rl)] == b) for b in buckets]
           for d in DS}

    # pass 1: all panel matrices + global observed range
    mats = {}
    for d in DS:
        for bb in BB:
            M = np.full((len(a.methods), len(buckets)), np.nan)
            for yi, m in enumerate(a.methods):
                for xi, bkt in enumerate(buckets):
                    vals = [cm[(dd, rl, mm, b)] for (dd, rl, mm, b) in cm
                            if dd == d and mm == m and b == bb and kw[(dd, rl)] == bkt]
                    if vals:
                        M[yi, xi] = st.mean(vals)
            mats[(d, bb)] = M
    allv = np.concatenate([mats[k][~np.isnan(mats[k])] for k in mats])
    gmin, gmax = float(allv.min()), float(allv.max())
    NORM = Normalize(vmin=gmin, vmax=gmax)
    _cmap = mcm.get_cmap(CMAP)

    def tc(val):
        r, g, b, _ = _cmap(NORM(val))
        return ('white', 'black') if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else ('black', 'white')

    fig, axs = plt.subplots(len(DS), len(BB),
                            figsize=(1.05 * len(buckets) * len(BB) + 2.2, 0.92 * len(a.methods) * len(DS) + 2),
                            sharex=False, sharey=False)
    fig.subplots_adjust(left=0.075, right=0.9, top=0.93, bottom=0.06, wspace=0.06, hspace=0.42)
    im = None
    w1 = sum(1 for (k, w) in buckets if w == 1)
    for ri, d in enumerate(DS):
        for ci, bb in enumerate(BB):
            ax = axs[ri, ci]
            M = mats[(d, bb)]
            im = ax.imshow(M, cmap=CMAP, norm=NORM, aspect='auto')
            ax.set_xticks(np.arange(-.5, len(buckets), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(a.methods), 1), minor=True)
            ax.grid(which='minor', color='white', linewidth=1.2)
            ax.tick_params(which='both', length=0)
            if 0 < w1 < len(buckets):
                ax.axvline(w1 - 0.5, color='k', lw=1.4)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_yticks(range(len(a.methods)))
            ax.set_yticklabels([PR[m] for m in a.methods] if ci == 0 else [], fontsize=9.5)
            if ri == 0:
                ax.set_title(bb, fontsize=12, fontweight='bold', pad=5)
            # x labels: bucket + this dataset's per-bucket n, directly under each panel
            ax.set_xticks(range(len(buckets)))
            ax.set_xticklabels([f'$k{k}w{w}$\n$n{{=}}{n}$' for (k, w), n in zip(buckets, dsn[d])],
                               fontsize=7.8, linespacing=1.05)
            if ci == 0:   # dataset row label (horizontal, tight to the panel)
                ax.annotate(d, xy=(-0.30, 0.5), xycoords='axes fraction', ha='right', va='center',
                            rotation=90, fontsize=12.5, fontweight='bold')
            for yi in range(len(a.methods)):
                for xi in range(len(buckets)):
                    if M[yi, xi] == M[yi, xi]:
                        fg, hc = tc(M[yi, xi])
                        ax.text(xi, yi, f'{M[yi,xi]*100:.0f}', ha='center', va='center',
                                fontsize=9.5, fontweight='bold', color=fg,
                                path_effects=[pe.withStroke(linewidth=1.4, foreground=hc)])
    fig.supxlabel('rule structural complexity  —  $k$ clauses, width $w$  '
                  '(simpler $\\rightarrow$ harder;  black rule separates $w{=}1$ from $w{=}2$;  '
                  '$n$ = rules in that dataset\'s bucket)', fontsize=10, y=0.005)
    fig.suptitle('Explanation correctness (GT-ROC$\\,\\times100$) vs rule structure — '
                 'dataset (rows) $\\times$ backbone (cols)', fontsize=13)
    cb = fig.colorbar(im, ax=axs, shrink=.6, pad=.015)
    cb.set_label('GT-ROC $\\times100$', fontsize=10, labelpad=20)
    ticks = [round(x, 2) for x in np.arange(np.ceil(gmin * 20) / 20, gmax + 1e-9, 0.1)]
    if gmin < 0.5 < gmax:
        cb.ax.axhline(0.5, color='k', lw=1.0)
        cb.ax.text(0.5, 0.5, 'chance', ha='center', va='center', fontsize=7,
                   transform=cb.ax.transAxes,
                   bbox=dict(fc='white', ec='none', alpha=0.75, pad=0.5))
        if 0.5 not in ticks:
            ticks.append(0.5)
    cb.set_ticks(sorted(ticks)); cb.set_ticklabels([f'{int(round(x*100))}' for x in sorted(ticks)])
    out = os.path.join(a.out, 'heat_kw_grid.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print('wrote', out, '| per-dataset bucket n:', dsn)


if __name__ == '__main__':
    main()
