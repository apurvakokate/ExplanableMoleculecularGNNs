#!/usr/bin/env python3
"""(k, width) structural-complexity heatmap for the monotone methods.

Each rule -> one bucket by (#clauses k, conjunction width), ordered by increasing
complexity: k1w1 < k1w2 < k2w1 < k2w2 < k3w1 < k3w2. x = bucket, y = method x
backbone (folds averaged), cell = mean GT-ROC x100 (annotated); column header shows
#rules. k and width are dataset-comparable, so pooling datasets here is safe (no range
confound). Methods restricted to those with monotone GT-ROC on this axis.

Usage: python heat_kw.py --root <planted_v2> --out <dir> [--methods mose gnnexplainer gsat mage]
"""
import argparse, csv, glob, json, os, statistics as st
from collections import defaultdict

DS = ['BBBP', 'Mutagenicity', 'hERG']
BB = ['GAT', 'GCN', 'PNA', 'SAGE']
PR = {'mose': 'MoSE', 'gsat': 'GSAT', 'gnnexplainer': 'GNNExpl',
      'mage': 'MAGE', 'motif_occlusion': 'Occl'}
KW = [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)]  # width-major: all w1 then w2
KWLAB = {(k, w): f'k{k}\\,w{w}' for (k, w) in KW}


def fl(x):
    try:
        v = float(x); return v if v == v else None
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', default='paper_deliverables')
    ap.add_argument('--methods', nargs='+', default=['mose', 'gsat', 'mage', 'gnnexplainer'])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import Normalize
    import matplotlib.cm as mcm
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.linewidth': 0.6,
                         'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    CMAP = 'viridis'   # sequential, perceptually-uniform, colour-blind-safe (standard for scores)

    kw = {}
    for d in DS:
        t = json.load(open(f'{a.root}/{d}/_shared/vocab/{d}/rbrics/rule_tiers.json'))
        for rule, r in t.items():
            w = max(len(c['motifs']) for c in r['clauses'])
            kw[(d, rule)] = (r['k'], w)
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
    nrules = [sum(1 for k in kw if kw[k] == b) for b in buckets]
    halo = [pe.withStroke(linewidth=2.0, foreground='white')]

    # pass 1: build every panel's matrix, and find the ACTUAL observed value range so
    # the colour scale shows only values that occur (no dead scale).
    mats = {}
    for bb in BB:
        M = np.full((len(a.methods), len(buckets)), np.nan)
        Ncell = np.zeros((len(a.methods), len(buckets)), int)
        for yi, m in enumerate(a.methods):
            for xi, bkt in enumerate(buckets):
                vals = [cm[(d, rl, mm, b)] for (d, rl, mm, b) in cm
                        if mm == m and b == bb and kw[(d, rl)] == bkt]
                Ncell[yi, xi] = len(vals)
                if vals:
                    M[yi, xi] = st.mean(vals)
        mats[bb] = (M, Ncell)
    allv = np.concatenate([mats[bb][0][~np.isnan(mats[bb][0])] for bb in BB])
    gmin, gmax = float(allv.min()), float(allv.max())
    NORM = Normalize(vmin=gmin, vmax=gmax)        # linear, clipped to observed range
    _cmap = mcm.get_cmap(CMAP)

    def txt_color(val):                            # luminance-adaptive so text stays legible
        r, g, b, _ = _cmap(NORM(val))
        return ('white', 'black') if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else ('black', 'white')

    fig, axs = plt.subplots(1, len(BB),
                            figsize=(1.15 * len(buckets) * len(BB) + 2, 0.62 * len(a.methods) + 2.8),
                            sharey=True)
    im = None
    for bi, bb in enumerate(BB):
        ax = axs[bi]
        M, Ncell = mats[bb]
        im = ax.imshow(M, cmap=CMAP, norm=NORM, aspect='auto')
        # crisp white cell borders
        ax.set_xticks(np.arange(-.5, len(buckets), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(a.methods), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.4)
        ax.tick_params(which='minor', length=0)
        # width-1 | width-2 divider (between the last w1 bucket and first w2 bucket)
        w1 = sum(1 for (k, w) in buckets if w == 1)
        if 0 < w1 < len(buckets):
            ax.axvline(w1 - 0.5, color='k', lw=1.8)
        ax.set_xticks(range(len(buckets)))
        ax.set_xticklabels([f'$k{k}\\,w{w}$\n$n{{=}}{n}$' for (k, w), n in zip(buckets, nrules)],
                           fontsize=9.5)
        if bi == 0:
            ax.set_yticks(range(len(a.methods)))
            ax.set_yticklabels([PR[m] for m in a.methods], fontsize=11.5)
        ax.set_title(bb, fontsize=13, fontweight='bold', pad=6)
        ax.tick_params(axis='both', which='both', length=0)   # no tick dashes
        for sp in ax.spines.values():
            sp.set_visible(False)
        for yi in range(len(a.methods)):
            for xi in range(len(buckets)):
                if M[yi, xi] == M[yi, xi]:
                    fg, hc = txt_color(M[yi, xi])
                    hl = [pe.withStroke(linewidth=1.8, foreground=hc)]
                    ax.text(xi, yi, f'{M[yi,xi]*100:.0f}', ha='center', va='center',
                            fontsize=13, fontweight='bold', color=fg, path_effects=hl)
    # rule-count row under the shared x, once (buckets identical across panels)
    fig.supxlabel('rule structural complexity  —  $k$ clauses, conjunction width $w$  '
                  '(simpler $\\rightarrow$ harder;  vertical rule separates $w{=}1$ from $w{=}2$;  '
                  '$n$ = rules per cell)', fontsize=10)
    fig.suptitle('Explanation correctness (GT-ROC$\\,\\times100$) vs rule structure, by backbone',
                 fontsize=13, y=1.02)
    cb = fig.colorbar(im, ax=axs, shrink=.72, pad=.015)         # bar = observed range only
    cb.set_label('GT-ROC $\\times100$', fontsize=10)
    ticks = [round(x, 2) for x in np.arange(np.ceil(gmin * 20) / 20, gmax + 1e-9, 0.1)]
    if gmin < 0.5 < gmax:                                        # chance as a reference tick+line
        cb.ax.axhline(0.5, color='k', lw=1.0)
        cb.ax.text(1.5, 0.5, 'chance', va='center', fontsize=7.5, transform=cb.ax.transAxes)
        if 0.5 not in ticks:
            ticks.append(0.5)
    cb.set_ticks(sorted(ticks))
    cb.set_ticklabels([f'{int(round(x*100))}' for x in sorted(ticks)])
    out = os.path.join(a.out, 'heat_kw_structure.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print('wrote per-backbone facets ->', out, ' buckets:',
          [f'k{k}w{w}' for k, w in buckets], ' nrules:', nrules)


if __name__ == '__main__':
    main()
