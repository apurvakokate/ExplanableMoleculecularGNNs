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
    ap.add_argument('--root', required=True,
                    help='metrics rollup root (reads <root>/<ds>/<rule>/all/metrics_*_unk-exclude.csv)')
    ap.add_argument('--vocab_root', default=None,
                    help='root holding <ds>/_shared/vocab/<ds>/rbrics/rule_tiers.json '
                         '(the planted_v2 source tree). Defaults to --root.')
    ap.add_argument('--out', default='paper_deliverables')
    ap.add_argument('--methods', nargs='+',
                    default=['mose', 'gsat', 'mage', 'gnnexplainer', 'pgexplainer'])
    a = ap.parse_args()
    vocab_root = a.vocab_root or a.root
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
        t = json.load(open(f'{vocab_root}/{d}/_shared/vocab/{d}/rbrics/rule_tiers.json'))
        for rule, r in t.items():
            kw[(d, rule)] = (r['k'], max(len(c['motifs']) for c in r['clauses']))
    cell = defaultdict(list)
    seen = set()   # drop concurrent-write duplicate (ds,rule,method,bb,fold) rows
    for f in glob.glob(f'{a.root}/*/*/all/metrics_*_unk-exclude.csv'):
        for r in csv.DictReader(open(f)):
            if r['method'] not in a.methods or r['unk'] != 'exclude':
                continue
            dk = (r['dataset'], r['gt_rule'], r['method'], r['backbone'], r['fold'])
            if dk in seen:
                continue
            seen.add(dk)
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
                            figsize=(0.55 * len(buckets) * len(BB) + 1.3, 0.46 * len(a.methods) * len(DS) + 1.0),
                            sharex=False, sharey=False)
    fig.subplots_adjust(left=0.11, right=0.885, top=0.905, bottom=0.13, wspace=0.045, hspace=0.26)
    im = None
    w1 = sum(1 for (k, w) in buckets if w == 1)
    for ri, d in enumerate(DS):
        for ci, bb in enumerate(BB):
            ax = axs[ri, ci]
            M = mats[(d, bb)]
            im = ax.imshow(M, cmap=CMAP, norm=NORM, aspect='auto')
            ax.set_xticks(np.arange(-.5, len(buckets), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(a.methods), 1), minor=True)
            ax.grid(which='minor', color='white', linewidth=1.4)
            ax.tick_params(which='both', length=0)
            if 0 < w1 < len(buckets):
                ax.axvline(w1 - 0.5, color='k', lw=1.6)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_yticks(range(len(a.methods)))
            ax.set_yticklabels([PR[m] for m in a.methods] if ci == 0 else [], fontsize=11.5)
            if ri == 0:
                ax.set_title(bb, fontsize=14, fontweight='bold', pad=5)
            # x labels: bucket + this dataset's per-bucket n, directly under each panel
            ax.set_xticks(range(len(buckets)))
            ax.set_xticklabels([f'$k{k}w{w}$\n$n{{=}}{n}$' for (k, w), n in zip(buckets, dsn[d])],
                               fontsize=9.5, linespacing=1.15)
            if ci == 0:   # dataset row label, clear of the method tick labels
                ax.annotate(d, xy=(-0.62, 0.5), xycoords='axes fraction', ha='center', va='center',
                            rotation=90, fontsize=15, fontweight='bold')
            for yi in range(len(a.methods)):
                for xi in range(len(buckets)):
                    if M[yi, xi] == M[yi, xi]:
                        fg, hc = tc(M[yi, xi])
                        ax.text(xi, yi, f'{M[yi,xi]*100:.0f}', ha='center', va='center',
                                fontsize=13, fontweight='bold', color=fg,
                                path_effects=[pe.withStroke(linewidth=1.6, foreground=hc)])
    fig.supxlabel('rule structural complexity  —  $k$ clauses, width $w$  '
                  '(black rule separates $w{=}1$ from $w{=}2$;  '
                  '$n$ = rules in that dataset\'s bucket)', fontsize=12, y=0.035)
    fig.suptitle('Explanation correctness (GT-ROC$\\,\\times100$) vs rule structure — '
                 'dataset (rows) $\\times$ backbone (cols)', fontsize=15, y=0.965)
    cax = fig.add_axes([0.905, 0.20, 0.014, 0.55])       # tall, tight to the panels
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('GT-ROC $\\times100$', fontsize=12, labelpad=8)
    cb.ax.tick_params(labelsize=10)
    ticks = [round(x, 2) for x in np.arange(np.ceil(gmin * 20) / 20, gmax + 1e-9, 0.1)]
    if gmin < 0.5 < gmax:
        cb.ax.axhline(0.5, color='k', lw=1.2)
        if 0.5 not in ticks:
            ticks.append(0.5)
    ticks = sorted(ticks)
    cb.set_ticks(ticks)
    cb.set_ticklabels(['chance' if abs(t - 0.5) < 1e-9 else f'{int(round(t*100))}' for t in ticks])
    out = os.path.join(a.out, 'heat_kw_grid.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print('wrote', out, '| per-dataset bucket n:', dsn)


if __name__ == '__main__':
    main()
