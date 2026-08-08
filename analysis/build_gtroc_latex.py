#!/usr/bin/env python3
"""build_gtroc_latex.py — node-direct GT-ROC tables in the make_acm_tables LaTeX style.

booktabs tables, cells $mean\\pm std$ across folds, best-in-row bolded (GT-ROC is
higher-is-better). Post-hoc methods as columns; datasets grouped in rows.

  * SOURCE-GT (real tier): one table per fragmentation. instance==global, so one
    metric. Rows = (dataset, backbone).
  * PLANTED (dnf tiers): one table per (fragmentation, metric in {instance,global}).
    Rows = (dataset, rule, backbone) — planted rules as sub-rows (ruled style).

  python analysis/build_gtroc_latex.py --csv gtroc_long.csv --out-dir <dir> [--split test]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

METHODS = ['gnnexplainer', 'pgexplainer', 'mage_v2', 'motif_occlusion', 'gsat', 'mose']
HEAD = {'gnnexplainer': 'GNNExpl.', 'pgexplainer': 'PGExpl.', 'mage_v2': 'MAGE',
        'motif_occlusion': 'Occlusion', 'gsat': 'GSAT',
        'mose': r'MoSE\textsuperscript{$\dagger$}'}   # dagger = filtered vocab
BACKBONE_ORDER = ['GIN', 'GCN', 'GAT', 'SAGE', 'PNA']
FRAGS = ['rbrics', 'fg_first', 'rdkit_fg_first', 'ertl_first']
FRAG_TEX = {'rbrics': 'rBRICS', 'fg_first': 'FG-first', 'rdkit_fg_first': 'FG-first (RDKit)',
            'ertl_first': 'Ertl-first'}
SRC = ['Benzene_Verified_GT', 'Alkane_Carbonyl_Verified_GT',
       'Fluoride_Carbonyl_Verified_GT', 'mutag']
PLANTED = ['BBBP', 'hERG', 'Mutagenicity', 'ogbg-molbace']
DATASET_TEX = {'Benzene_Verified_GT': 'Benzene', 'Alkane_Carbonyl_Verified_GT': 'Alkane-Carbonyl',
               'Fluoride_Carbonyl_Verified_GT': 'Fluoride-Carbonyl', 'mutag': 'mutag',
               'BBBP': 'BBBP', 'hERG': 'hERG', 'Mutagenicity': 'Mutagenicity',
               'ogbg-molbace': 'molbace'}
TIER_TEX = {'dnf_k2_r1': 'k2r1', 'dnf_k2_r2': 'k2r2', 'dnf_k3_r1': 'k3r1', 'dnf_k3_r2': 'k3r2'}


def _fmt(mean, std, best):
    if not np.isfinite(mean):
        return '--'
    body = f'{mean:.2f}\\pm{std:.2f}' if np.isfinite(std) else f'{mean:.2f}'
    return f'$\\mathbf{{{body}}}$' if best else f'${body}$'


def _agg(df, metric):
    """mean/std/n over folds, keyed (dataset,gt_tier,fragmentation,backbone,method)."""
    col = f'gtroc_{metric}'
    g = (df.groupby(['dataset', 'gt_tier', 'fragmentation', 'backbone', 'method'])[col]
           .agg(['mean', 'std', 'count']).reset_index())
    return g


def _rowcells(sub, ds, tier, bb):
    r = sub[(sub.dataset == ds) & (sub.gt_tier == tier) & (sub.backbone == bb)]
    vals = {}
    for m in METHODS:
        rr = r[r.method == m]
        if len(rr):
            vals[m] = (float(rr['mean'].iloc[0]), float(rr['std'].iloc[0]))
    finite = {k: v[0] for k, v in vals.items() if np.isfinite(v[0])}
    best = max(finite, key=finite.get) if finite else None
    return [_fmt(*vals[m], m == best) if m in vals else '--' for m in METHODS], bool(vals)


def build_table(sub, frag, datasets, tiers, ruled, caption, label):
    heads = [HEAD[m] for m in METHODS]
    pre = 'lll' if ruled else 'll'
    ncols = pre + 'c' * len(METHODS)
    hdr = ('Dataset & Rule & BB & ' if ruled else 'Dataset & BB & ') + ' & '.join(heads) + r' \\'
    lines = [r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
             f'\\caption{{{caption}}}', f'\\label{{{label}}}',
             f'\\begin{{tabular}}{{{ncols}}}', r'\toprule', hdr, r'\midrule']
    any_row = False
    for di, ds in enumerate(datasets):
        if ds not in set(sub.dataset):
            continue
        if di > 0 and any_row:
            lines.append(r'\midrule')
        first_ds = True
        for tier in tiers:
            bbs = [b for b in BACKBONE_ORDER
                   if len(sub[(sub.dataset == ds) & (sub.gt_tier == tier) & (sub.backbone == b)])]
            first_tier = True
            for bb in bbs:
                cells, ok = _rowcells(sub, ds, tier, bb)
                if not ok:
                    continue
                any_row = True
                dcol = DATASET_TEX.get(ds, ds) if first_ds else ''
                if ruled:
                    tcol = TIER_TEX.get(tier, tier) if first_tier else ''
                    lines.append(f'{dcol} & {tcol} & {bb} & ' + ' & '.join(cells) + r' \\')
                else:
                    lines.append(f'{dcol} & {bb} & ' + ' & '.join(cells) + r' \\')
                first_ds = False; first_tier = False
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    return '\n'.join(lines) + '\n' if any_row else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--all_filtered', action='store_true',
                    help='every method on the filtered vocab (drop MoSE dagger; note kept-atom scope)')
    ap.add_argument('--note', default='', help='extra caption note (e.g. "UNK=0.5")')
    ap.add_argument('--methods', nargs='*', default=None,
                    help='override/subset the method columns (e.g. drop gsat while it finishes)')
    args = ap.parse_args()
    if args.methods:
        METHODS[:] = [m for m in args.methods]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    df = df[df.split == args.split]
    vocab_note = (f' All methods on the filtered vocab (kept-motif atom set{("; " + args.note) if args.note else ""}).'
                  if args.all_filtered else r' $\dagger$: MoSE filtered vocab; others full.')
    if args.all_filtered:
        HEAD['mose'] = 'MoSE'

    inputs = []
    for frag in FRAGS:
        # SOURCE-GT: instance (== global); one table
        sub = _agg(df[df.fragmentation == frag], 'instance')
        cap = (f'Node-direct GT-ROC (source-GT, split={args.split}), fragmentation '
               f'{FRAG_TEX.get(frag, frag)}. Mean$\\pm$std across folds; best per row in bold. '
               r'Instance$=$Global (single-clause GT).' + vocab_note)
        t = build_table(sub[sub.gt_tier == 'real'], frag, SRC, ['real'], False,
                        cap, f'tab:gtroc-src-{frag}')
        if t:
            fn = f'gtroc_source_{frag}.tex'; (out / fn).write_text(t); inputs.append(fn)
        # PLANTED: instance + global, ruled (rules as sub-rows)
        tiers = ['dnf_k2_r1', 'dnf_k2_r2', 'dnf_k3_r1', 'dnf_k3_r2']
        for metric in ('instance', 'global'):
            subm = _agg(df[df.fragmentation == frag], metric)
            cap = (f'Node-direct GT-ROC (planted DNF, {metric.upper()}, split={args.split}), '
                   f'fragmentation {FRAG_TEX.get(frag, frag)}. Rules k$X$r$Y$ as sub-rows. '
                   r'Mean$\pm$std across folds; best per row in bold.' + vocab_note)
            t = build_table(subm[subm.gt_tier.isin(tiers)], frag, PLANTED, tiers, True,
                            cap, f'tab:gtroc-planted-{metric}-{frag}')
            if t:
                fn = f'gtroc_planted_{metric}_{frag}.tex'
                (out / fn).write_text(t); inputs.append(fn)

    il = out / '_INPUT_LIST.tex'
    il.write_text('\n'.join(f'\\input{{{f}}}' for f in inputs) + '\n')
    (out / 'PREAMBLE_NOTE.txt').write_text(
        'Requires: \\usepackage{booktabs}. GT-ROC higher is better; bold = best per row.\n')
    print(f'wrote {len(inputs)} tables + _INPUT_LIST.tex to {out}')
    for f in inputs:
        print(' ', f)


if __name__ == '__main__':
    main()
