#!/usr/bin/env python3
"""
analyse_results.py
==================
Analyse the outputs of generate_vocab_rules.py.
Produces motif distribution plots and rule distribution summaries.

Usage:
    python analyse_results.py --out_dir ./motifsat_output --dataset Mutagenicity
    python analyse_results.py --out_dir ./motifsat_output --all

Outputs (one directory per dataset/variant):
    motif_distribution.png   — top-N motif support bar chart
    rule_distribution.png    — rule coverage histogram + tier breakdown
    motif_stats.csv          — full motif table (support, ring, size, class split)
    rule_stats.csv           — full rule table (coverage, clauses, motifs)
    summary_report.txt       — text summary
"""

import sys, os, json, pickle, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.sparse as sp


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_variant(vdir: Path, dataset: str, variant: str) -> dict:
    base = str(vdir / f'{dataset}_{variant}')

    meta    = json.load(open(vdir / 'meta.json'))
    rules   = json.load(open(vdir / 'rules.json'))
    cols    = pd.read_csv(vdir / 'matrix_columns.csv')
    X       = sp.load_npz(str(vdir / 'matrix.npz'))
    smdf    = pd.read_csv(vdir / 'smiles_labels.csv')

    ml   = pickle.load(open(base + '_motif_list.pickle',   'rb'))
    mc   = pickle.load(open(base + '_motif_counts.pickle', 'rb'))
    mlen = pickle.load(open(base + '_motif_length.pickle', 'rb'))
    mcls = pickle.load(open(base + '_motif_class.pickle',  'rb'))

    return dict(meta=meta, rules=rules, cols=cols, X=X, smdf=smdf,
                motif_list=ml, motif_counts=mc, motif_length=mlen,
                motif_class=mcls, vdir=vdir, dataset=dataset, variant=variant)


# ─────────────────────────────────────────────────────────────────────────────
# MOTIF STATS
# ─────────────────────────────────────────────────────────────────────────────

def motif_stats_df(d: dict) -> pd.DataFrame:
    n      = d['X'].shape[0]
    labels = d['smdf']['label'].values if 'label' in d['smdf'].columns \
             else np.zeros(n, int)
    n_pos  = int(labels.sum())
    n_neg  = n - n_pos

    # True per-motif ring flag from matrix_columns.csv (frag.has_ring at vocab
    # build time) — keyed by SMARTS. Avoids guessing ring membership from the
    # SMILES string or using heavy-atom count as a proxy.
    ring_map = {}
    if 'ring' in d['cols'].columns and 'motif_identity' in d['cols'].columns:
        ring_map = {str(s): bool(r) for s, r in
                    zip(d['cols']['motif_identity'], d['cols']['ring'])}

    rows = []
    for i, smi in enumerate(d['motif_list']):
        cnt  = d['motif_counts'][i]
        na   = d['motif_length'][i]
        cls  = d['motif_class'].get(i, {0: 0, 1: 0})
        sup  = round(cnt / n * 100, 2)
        p1   = round(cls.get(1,0) / max(cnt,1) * 100, 1)
        enr  = round((cls.get(1,0)/max(n_pos,1)) /
                     (max(cls.get(0,0),1)/max(n_neg,1)), 3) if n_neg else 0
        rows.append({
            'motif_id':   i,
            'smarts':     smi,
            'n_mols':     cnt,
            'support_%':  sup,
            'n_atoms':    na,
            'ring':       ring_map.get(str(smi), False),
            'n_pos':      cls.get(1,0),
            'n_neg':      cls.get(0,0),
            'pct_pos':    p1,
            'enrichment': enr,
        })
    return pd.DataFrame(rows).sort_values('support_%', ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# RULE STATS
# ─────────────────────────────────────────────────────────────────────────────

def rule_stats_df(d: dict) -> pd.DataFrame:
    rows = []
    for rank, r in enumerate(d['rules']):
        motifs = [m for c in r['clauses'] for m in c['motifs']]
        rule_str = ' ∨ '.join(
            '(' + ' ∧ '.join(c['motifs']) + ')' for c in r['clauses'])
        rows.append({
            'rank':      rank,
            'pct1':      r['pct1'],
            'pct0':      r['pct0'],
            'n1':        r['n1'],
            'n0':        r['n0'],
            'n_clauses': r['n_clauses'],
            'n_motifs':  len(set(motifs)),
            'rule_str':  rule_str,
            'motifs':    '|'.join(motifs),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    'ring':    '#185FA5',
    'noRing':  '#BA7517',
    'grid':    '#e8e8e8',
    'pos':     '#15803D',
    'neg':     '#A32D2D',
    'neutral': '#6B7280',
}


def plot_motif_distribution(ms: pd.DataFrame, d: dict, out: Path,
                             top_n: int = 20):
    top  = ms.head(top_n).copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              gridspec_kw={'width_ratios': [3, 1]})

    # ── Panel 1: support bar chart ──────────────────────────────────────────
    ax = axes[0]
    colors = [PALETTE['ring'] if r else PALETTE['noRing']
              for r in top['ring']]
    bars = ax.barh(range(len(top)), top['support_%'][::-1].values,
                   color=colors[::-1], edgecolor='white', linewidth=0.4)

    # Annotate bars
    for i, (_, row) in enumerate(top.iloc[::-1].iterrows()):
        x = row['support_%']
        label = f"{row['support_%']:.1f}%  {row['smarts'][:30]}"
        ax.text(x + 0.2, i, label, va='center', fontsize=7.5,
                color='#1a1a1a')

    ax.set_yticks([])
    ax.set_xlabel('Molecular support (%)', fontsize=11)
    ax.set_title(f"{d['dataset']} — {d['variant']}\nTop-{top_n} motifs by support",
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, top['support_%'].max() * 1.4)
    ax.grid(axis='x', color=PALETTE['grid'], linewidth=0.5)
    ax.spines[['top','right','left']].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=PALETTE['ring'],   label='Ring-containing'),
        mpatches.Patch(color=PALETTE['noRing'], label='Acyclic / functional group'),
    ]
    ax.legend(handles=legend_patches, fontsize=9, frameon=False,
              loc='lower right')

    # ── Panel 2: size distribution ─────────────────────────────────────────
    ax2 = axes[1]
    bins  = [(1,2,'1-2a'), (3,5,'3-5a'), (6,9,'6-9a'), (10,15,'10-15a'), (16,99,'16+a')]
    sizes = ms['n_atoms'].values
    counts = [int(((sizes >= lo) & (sizes <= hi)).sum()) for lo,hi,_ in bins]
    labels = [lab for _,_,lab in bins]
    bar_colors = [PALETTE['noRing'], PALETTE['noRing'],
                  PALETTE['ring'],   PALETTE['ring'],   PALETTE['neutral']]
    ax2.barh(labels, counts, color=bar_colors, edgecolor='white', linewidth=0.4)
    for i, c in enumerate(counts):
        ax2.text(c + 0.5, i, str(c), va='center', fontsize=9)
    ax2.set_xlabel('Fragment count', fontsize=10)
    ax2.set_title('Size distribution', fontsize=11)
    ax2.grid(axis='x', color=PALETTE['grid'], linewidth=0.5)
    ax2.spines[['top','right','left']].set_visible(False)

    plt.tight_layout()
    path = out / 'motif_distribution.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    → {path}")


def plot_rule_distribution(rs: pd.DataFrame, d: dict, out: Path):
    if rs.empty:
        print("    (no rules to plot)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Panel 1: coverage histogram ────────────────────────────────────────
    ax = axes[0]
    ax.hist(rs['pct1'], bins=20, color=PALETTE['pos'],
            edgecolor='white', linewidth=0.5, alpha=0.85)
    ax.set_xlabel('Rule coverage — positive class (%)', fontsize=10)
    ax.set_ylabel('Number of rules', fontsize=10)
    ax.set_title('Rule coverage distribution', fontsize=11, fontweight='bold')
    ax.axvline(rs['pct1'].max(), color='#A32D2D', linestyle='--',
               linewidth=1.2, label=f"best: {rs['pct1'].max():.1f}%")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis='y', color=PALETTE['grid'], linewidth=0.5)
    ax.spines[['top','right']].set_visible(False)

    # ── Panel 2: tier breakdown ────────────────────────────────────────────
    ax2 = axes[1]
    tier_counts = rs['n_clauses'].value_counts().sort_index()
    colors = [PALETTE['ring'], PALETTE['noRing'],
              PALETTE['pos'],  PALETTE['neg']][:len(tier_counts)]
    bars = ax2.bar(tier_counts.index.astype(str), tier_counts.values,
                   color=colors, edgecolor='white', linewidth=0.5)
    for bar, cnt in zip(bars, tier_counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(cnt), ha='center', va='bottom', fontsize=9)
    ax2.set_xlabel('Number of clauses (OR branches)', fontsize=10)
    ax2.set_ylabel('Rules', fontsize=10)
    ax2.set_title('Rules by complexity tier', fontsize=11, fontweight='bold')
    ax2.grid(axis='y', color=PALETTE['grid'], linewidth=0.5)
    ax2.spines[['top','right']].set_visible(False)

    # ── Panel 3: top-5 rules as scatter (pct1 vs pct0) ────────────────────
    ax3 = axes[2]
    ax3.scatter(rs['pct0'], rs['pct1'], alpha=0.4, s=18,
                color=PALETTE['ring'], label='all rules')
    top5 = rs.head(5)
    ax3.scatter(top5['pct0'], top5['pct1'], s=60, zorder=5,
                color=PALETTE['neg'], label='top-5')
    for _, row in top5.iterrows():
        motifs = row['motifs'].split('|')[:2]
        label  = ' ∨ '.join(m[:20] for m in motifs) + ('…' if len(motifs) > 2 else '')
        ax3.annotate(label, (row['pct0'], row['pct1']),
                     fontsize=6.5, textcoords='offset points',
                     xytext=(4, 2), color='#333')
    ax3.set_xlabel('Negative class coverage (%)', fontsize=10)
    ax3.set_ylabel('Positive class coverage (%)', fontsize=10)
    ax3.set_title('Coverage: positive vs negative', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9, frameon=False)
    ax3.grid(color=PALETTE['grid'], linewidth=0.5)
    ax3.spines[['top','right']].set_visible(False)

    fig.suptitle(f"{d['dataset']} — {d['variant']} — {len(rs)} rules",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = out / 'rule_distribution.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(d: dict, ms: pd.DataFrame, rs: pd.DataFrame, out: Path):
    meta  = d['meta']
    n     = d['X'].shape[0]
    ab1   = int((ms['support_%'] >= 1.0).sum())
    ab5   = int((ms['support_%'] >= 5.0).sum())

    lines = [
        f"{'='*60}",
        f"  {d['dataset']} — {d['variant']}",
        f"{'='*60}",
        f"",
        f"MOLECULES",
        f"  Total:         {n}",
        f"  Training:      {int((d['smdf']['group']=='training').sum()) if 'group' in d['smdf'].columns else n}",
        f"  Test:          {int((d['smdf']['group']=='test').sum()) if 'group' in d['smdf'].columns else 0}",
        f"",
        f"VOCABULARY",
        f"  Total motifs:  {meta['n_vocab_motifs']}",
        f"  Above 1%:      {ab1}",
        f"  Above 5%:      {ab5}",
        f"  Mean size:     {ms['n_atoms'].mean():.1f} atoms",
        f"  Ring-containing: {int(ms['ring'].sum())} of {len(ms)}",
        f"",
        f"  Top-10 motifs by support:",
    ]
    for _, row in ms.head(10).iterrows():
        lines.append(f"    {row['support_%']:5.1f}%  {row['n_atoms']:2d}a  "
                     f"{'R' if row['ring'] else ' '}  {row['smarts']}")

    lines += [
        f"",
        f"RULES",
        f"  Total rules:   {len(rs)}",
    ]
    if not rs.empty:
        tier_counts = rs['n_clauses'].value_counts().sort_index()
        for t, c in tier_counts.items():
            lines.append(f"  {t}-clause rules: {c}")
        lines += [
            f"",
            f"  Best rule:     {rs.iloc[0]['pct1']:.1f}% positive coverage",
            f"  ({rs.iloc[0]['rule_str'][:80]})",
            f"",
            f"  Top-5 rules:",
        ]
        for _, row in rs.head(5).iterrows():
            lines.append(f"    rank {row['rank']:3d}  pct1={row['pct1']:5.1f}%  "
                         f"pct0={row['pct0']:4.1f}%  "
                         f"clauses={row['n_clauses']}  "
                         f"{row['rule_str'][:60]}")

    txt = '\n'.join(lines)
    path = out / 'summary_report.txt'
    path.write_text(txt)
    print(f"    → {path}")
    print(txt)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def analyse(out_dir: Path, dataset: str, variant: str):
    vdir = out_dir / dataset / variant
    if not vdir.exists():
        print(f"  SKIP {dataset}/{variant} — directory not found")
        return

    print(f"\n{'='*55}\n  {dataset} / {variant}\n{'='*55}")
    d = load_variant(vdir, dataset, variant)

    ms = motif_stats_df(d)
    rs = rule_stats_df(d)

    ms.to_csv(vdir / 'motif_stats.csv', index=False)
    rs.to_csv(vdir / 'rule_stats.csv',  index=False)
    print(f"    → {vdir}/motif_stats.csv  ({len(ms)} motifs)")
    print(f"    → {vdir}/rule_stats.csv   ({len(rs)} rules)")

    plot_motif_distribution(ms, d, vdir)
    plot_rule_distribution(rs, d, vdir)
    write_summary(d, ms, rs, vdir)


def main():
    p = argparse.ArgumentParser(description='Analyse MotifSAT output')
    p.add_argument('--out_dir',  required=True)
    p.add_argument('--dataset',  default=None)
    p.add_argument('--variant',  default=None)
    p.add_argument('--all',      action='store_true',
                   help='Analyse all dataset/variant combinations found')
    p.add_argument('--top_n',    type=int, default=20,
                   help='Top N motifs to plot (default 20)')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        print(f"Error: {out_dir} not found"); sys.exit(1)

    if args.all:
        # Discover all dataset/variant directories
        pairs = []
        for ds_dir in sorted(out_dir.iterdir()):
            if not ds_dir.is_dir(): continue
            for v_dir in sorted(ds_dir.iterdir()):
                if not v_dir.is_dir(): continue
                if (v_dir / 'meta.json').exists():
                    pairs.append((ds_dir.name, v_dir.name))
        if not pairs:
            print("No analysable directories found."); sys.exit(1)
    elif args.dataset and args.variant:
        pairs = [(args.dataset, args.variant)]
    elif args.dataset:
        # All variants for this dataset
        ds_dir = out_dir / args.dataset
        pairs = [(args.dataset, v.name)
                 for v in sorted(ds_dir.iterdir())
                 if v.is_dir() and (v / 'meta.json').exists()]
    else:
        print("Specify --dataset, --dataset + --variant, or --all"); sys.exit(1)

    for ds, variant in pairs:
        analyse(out_dir, ds, variant)

    print(f"\nDone. Results in {out_dir}")


if __name__ == '__main__':
    main()
