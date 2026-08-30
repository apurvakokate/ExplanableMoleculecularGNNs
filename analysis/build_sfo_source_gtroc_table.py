#!/usr/bin/env python3
r"""build_sfo_source_gtroc_table.py — SFO source-GT instance GT-ROC table (matches tab:gtroc-tvt).

INPUT  : fragmentation_v2/results/gtroc_source/<ds>/<method>_f<F>.csv (from evaluate_gtroc.py;
         one row per backbone for that fold; columns gtroc_{full,filt}_{...}).
VALUE  : gtroc_filt_all / gtroc_full_all -> Filt/Full rows, pooled train+valid+test.
AGG    : per (dataset, backbone, method) -> mean +/- std over the 5 FOLDS (unit=fold, like the
         rBRICS source table), x100, 1 dp. Significance via analysis.table_sig (Welch two-sided +
         Holm over the 5 folds); higher is better -> bold=max.
LAYOUT : rows = {Benzene, Fluoride-Carbonyl, Alkane-Carbonyl} x {GIN,GCN,GAT,SAGE,PNA} x Vocab{Filt,Full};
         cols = GNNExpl, PGExpl, MAGE, Occlusion, GSAT, MOSE, MOSE_unk  (matches tab:gtroc-tvt).
OUTPUT : <out>/sfo_gtroc_source.tex (tab:sfo-gtroc-tvt) + sfo_gtroc_source_agg.csv.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

try:
    from analysis import table_sig as ts
except ImportError:
    import table_sig as ts

# display label -> (short ds name for the table)
DATASETS = [('Benzene_Verified_GT', 'Benzene'),
            ('Fluoride_Carbonyl_Verified_GT', 'Fluoride-Carbonyl'),
            ('Alkane_Carbonyl_Verified_GT', 'Alkane-Carbonyl')]
BACKBONES = ['GIN', 'GCN', 'GAT', 'SAGE', 'PNA']
# (display label, method-file stem)
METHODS = [('GNNExpl.', 'gnnexplainer'), ('PGExpl.', 'pgexplainer'), ('MAGE', 'mage'),
           ('Occlusion', 'motif_occlusion'), ('GSAT', 'gsat'), ('MOSE', 'mose'),
           ('MOSE$_{unk}$', 'mose_U')]
ALPHA = 0.05
COLS = ['gtroc_filt_all', 'gtroc_full_all']
CONDITIONS = [('gtroc_filt_all', 'Filt'), ('gtroc_full_all', 'Full')]


def load(root):
    """Long frame: one row per (dataset, method-stem, backbone, fold) with the value columns."""
    rows = []
    stems = sorted({m[1] for m in METHODS})
    for ds, _short in DATASETS:
        for stem in stems:
            for f in glob.glob(f'{root}/{ds}/{stem}_f*.csv'):
                try:
                    df = pd.read_csv(f)
                except Exception:
                    continue
                for _, r in df.iterrows():
                    rec = dict(dataset=ds, method=stem, backbone=str(r.get('backbone')),
                               fold=r.get('fold'))
                    for c in COLS:
                        rec[c] = pd.to_numeric(pd.Series([r.get(c)]), errors='coerce').iloc[0]
                    rows.append(rec)
    return pd.DataFrame(rows)


def fold_vector(df, ds, bb, stem, col):
    """The per-fold values (one per fold) for one cell, NaNs dropped."""
    sub = df[(df.dataset == ds) & (df.backbone == bb) & (df.method == stem)]
    if sub.empty or col not in sub.columns:
        return np.array([])
    return np.asarray(pd.to_numeric(sub[col], errors='coerce').dropna().values, float)


def fmt(mean, std):
    return f'{mean * 100:.1f}_{{{std * 100:.1f}}}'


def _row_cells(df, ds, bb, col):
    cells = []
    for _, stem in METHODS:
        v = fold_vector(df, ds, bb, stem, col)
        cells.append(dict(key=stem,
                          mean=(float(v.mean()) if v.size else float('nan')),
                          std=(float(v.std(ddof=1)) if v.size > 1 else 0.0),
                          n=int(v.size)))
    tags = ts.decide(cells, higher_better=True, alpha=ALPHA)
    out = []
    for c in cells:
        if not np.isfinite(c['mean']):
            out.append('--')
            continue
        s = f"${fmt(c['mean'], c['std'])}$"
        out.append(ts.wrap(s, tags.get(c['key'], 'plain'), 'bold_underline'))
    return out


def build_tex(df):
    ncol = len(METHODS)
    lastcol = 3 + ncol
    rows_per_ds = len(BACKBONES) * len(CONDITIONS)
    L = ['% SFO source-GT instance GT-ROC; Filt=exclude-unk, Full=include-unk; pooled tvt.',
         '% cell = mean_{std} x100 (1 dp); unit=fold (5 folds). Matches tab:gtroc-tvt.',
         '\\begin{tabular}{lll' + 'c' * ncol + '}', '\\toprule',
         'Dataset & BB & Vocab & ' + ' & '.join(lbl for lbl, _ in METHODS) + ' \\\\', '\\midrule']
    for di, (ds, short) in enumerate(DATASETS):
        for bi, bb in enumerate(BACKBONES):
            for ci, (col, clab) in enumerate(CONDITIONS):
                cells = _row_cells(df, ds, bb, col)
                if bi == 0 and ci == 0:
                    lead = (f'\\multirow{{{rows_per_ds}}}{{*}}{{{short}}} & '
                            f'\\multirow{{2}}{{*}}{{{bb}}} & {clab}')
                elif ci == 0:
                    lead = f' & \\multirow{{2}}{{*}}{{{bb}}} & {clab}'
                else:
                    lead = f' &  & {clab}'
                L.append(f'{lead} & ' + ' & '.join(cells) + ' \\\\')
            if bi != len(BACKBONES) - 1:
                L.append(f'\\cmidrule(l){{2-{lastcol}}}')
        if di != len(DATASETS) - 1:
            L.append('\\midrule')
    L += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(L)


CAPTION = (
    'Instance GT-ROC on the source-GT datasets under the \\textbf{SFO} (size-frequency-optimization) '
    'vocabulary, pooled over train+valid+test (graph-weighted), $\\times100$; subscript $=$ std over '
    'the 5 folds, per (dataset, backbone). The \\textbf{Vocab} column: \\textbf{Filt} (exclude-unk) '
    'restricts to kept-motif nodes, \\textbf{Full} (include-unk) scores all-node recovery. Per '
    '(dataset, backbone) and within each condition, the best is in \\textbf{bold} and methods not '
    'significantly worse (Welch two-sided $t$-test over the 5 folds, Holm, $\\alpha{=}0.05$) are '
    '\\underline{underlined}. MOSE$_{unk}$ = MOSE with learnable unknown. GSAT atts are the '
    'vocab-independent rBRICS-trained GSAT node attentions scored under the SFO vocabulary '
    '(byte-identical to the canonical bundle). Directly comparable to the rBRICS source table.')


def floatwrap(body, caption, label):
    return ('\\begin{table*}[t]\n\\centering\n\\small\n' + body +
            f'\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table*}}\n')


def build_agg_csv(df):
    out = []
    for ds, _short in DATASETS:
        for bb in BACKBONES:
            for lbl, stem in METHODS:
                rec = dict(dataset=ds, backbone=bb, method=lbl, stem=stem)
                for c in COLS:
                    v = fold_vector(df, ds, bb, stem, c)
                    rec[f'{c}_mean'] = round(float(v.mean()) * 100, 2) if v.size else np.nan
                    rec[f'{c}_std'] = round(float(v.std(ddof=1)) * 100, 2) if v.size > 1 else 0.0
                    rec[f'{c}_n'] = int(v.size)
                out.append(rec)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='results/gtroc_source (reads <ds>/<method>_f*.csv)')
    ap.add_argument('--out', required=True, help='output dir (created; nothing overwritten)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = load(args.root)
    if df.empty:
        raise SystemExit(f'no CSVs under {args.root}')
    cov = (df.dropna(subset=['gtroc_filt_all']).groupby(['dataset', 'method'])['fold']
           .nunique().unstack(fill_value=0))
    print('folds with finite gtroc_filt_all (per dataset x method):')
    print(cov.to_string())
    open(os.path.join(args.out, 'sfo_gtroc_source.tex'), 'w').write(
        floatwrap(build_tex(df), CAPTION, 'tab:sfo-gtroc-tvt'))
    build_agg_csv(df).to_csv(os.path.join(args.out, 'sfo_gtroc_source_agg.csv'), index=False)
    print(f'\nwrote sfo_gtroc_source.tex + sfo_gtroc_source_agg.csv -> {args.out}')


if __name__ == '__main__':
    main()
