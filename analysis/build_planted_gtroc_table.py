#!/usr/bin/env python3
r"""build_planted_gtroc_table.py — planted GT-ROC LaTeX table from evaluate_gtroc.py CSVs.

INPUT  : mose_replication_v2/gtroc_paper_artifacts/planted/<ds>/<rule>/<method>.csv
         (produced by analysis/evaluate_gtroc.py; one row per backbone x fold, columns
          gtroc_{full,filt}_{train,valid,test,all}).
VALUE  : gtroc_filt_all  -- exclude-unk (kept vocabulary; all methods share one motif set),
         pooled over train+valid+test (matches the source-GT table tab:gtroc-tvt convention).
AGG    : per rule -> fold-average; per (dataset, backbone, method) -> mean +/- std over the
         50 rules (each rule fold-averaged), x100, 1 dp.
LAYOUT : rows = {BBBP, Mutagenicity, hERG} x {GCN, GAT, SAGE, PNA};
         cols = GNNExpl, PGExpl, MAGE, Occlusion, GSAT, MOSE.  No GIN, no MOSE_unk
         (mose_learnable_shared produced 0 planted rows -> not available).
SIG    : per (dataset, backbone), best mean in \boldmath; methods NOT significantly worse than
         the best (Welch two-sided t-test over the 50 rule values, Holm across the other methods,
         alpha=0.05) in \underline.
OUTPUT : <out>/planted_gtroc.tex  and  <out>/planted_gtroc_agg.csv  (validation: per-cell
         mean/std/n plus the Full-all counterpart for a Full-vs-Filt sanity check).
         Nothing existing is overwritten; both land under gtroc_paper_artifacts/tables/.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

# reuse the EXACT significance module the source-GT table (tab:gtroc-tvt) uses -> identical
# Welch two-sided + Holm step-down (best bold, not-sig-worse underlined). Sample unit here is
# the 50 per-rule values (vs 5 folds for the source-GT table); the test itself is unchanged.
try:
    from analysis import table_sig as ts
except ImportError:                      # when run from inside analysis/
    import table_sig as ts

VALUE = 'gtroc_filt_all'
AUX = 'gtroc_full_all'                 # reported in the validation CSV only (Full vs Filt gap)
DATASETS = ['BBBP', 'Mutagenicity', 'hERG']
BACKBONES = ['GCN', 'GAT', 'SAGE', 'PNA']
# (display label, method-file stem)
METHODS = [('GNNExpl.', 'gnnexplainer'), ('PGExpl.', 'pgexplainer'), ('MAGE', 'mage'),
           ('Occlusion', 'motif_occlusion'), ('GSAT', 'gsat'), ('MOSE', 'mose_fixed')]
ALPHA = 0.05


def load(root):
    """Long frame: one row per (dataset, rule, method-stem, backbone, fold, value_col)."""
    rows = []
    stems = sorted({m[1] for m in METHODS})
    for ds in DATASETS:
        for stem in stems:
            for f in glob.glob(f'{root}/{ds}/*/{stem}.csv'):
                rule = os.path.basename(os.path.dirname(f))
                try:
                    df = pd.read_csv(f)
                except Exception:
                    continue
                for _, r in df.iterrows():
                    rows.append(dict(
                        dataset=ds, rule=rule, method=stem, backbone=str(r.get('backbone')),
                        fold=r.get('fold'),
                        filt=pd.to_numeric(pd.Series([r.get(VALUE)]), errors='coerce').iloc[0],
                        full=pd.to_numeric(pd.Series([r.get(AUX)]), errors='coerce').iloc[0]))
    return pd.DataFrame(rows)


def rule_vector(df, ds, bb, stem, col='filt'):
    """The 50 per-rule values (each rule fold-averaged), NaNs dropped."""
    sub = df[(df.dataset == ds) & (df.backbone == bb) & (df.method == stem)]
    if sub.empty:
        return np.array([])
    v = sub.groupby('rule')[col].mean().dropna().values
    return np.asarray(v, float)


def fmt(mean, std):
    return f'{mean * 100:.1f}_{{{std * 100:.1f}}}'


CONDITIONS = [('filt', 'Filt'), ('full', 'Full')]   # exclude-unk / include-unk, both pooled tvt


def _row_cells(df, ds, bb, col):
    r"""One table row for a Vocab condition. Best/tie decided by analysis.table_sig.decide (the
    SAME Welch two-sided + Holm used by tab:gtroc-tvt); unit = the 50 per-rule values. Best ->
    \boldmath, not-significantly-worse -> \underline, via table_sig.wrap(style='bold_underline')."""
    cells = []
    for _, stem in METHODS:
        v = rule_vector(df, ds, bb, stem, col)
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
    """Mirror the source-GT table tab:gtroc-tvt: rows = Dataset x BB x Vocab{Filt,Full},
    columns = explainers, cell = mean_{std} x100. Bold/underline computed per row (condition)."""
    ncol = len(METHODS)
    lastcol = 3 + ncol                                   # Dataset, BB, Vocab + methods
    rows_per_ds = len(BACKBONES) * len(CONDITIONS)
    L = []
    L.append('% planted GT-ROC (instance); Filt=gtroc_filt_all (exclude-unk), Full=gtroc_full_all')
    L.append('% (include-unk); both pooled train+valid+test. cell = mean_{std} x100 (1 dp);')
    L.append('% unit=rule (50 rules, each fold-averaged). Matches tab:gtroc-tvt layout.')
    L.append('\\begin{tabular}{lll' + 'c' * ncol + '}')
    L.append('\\toprule')
    L.append('Dataset & BB & Vocab & ' + ' & '.join(lbl for lbl, _ in METHODS) + ' \\\\')
    L.append('\\midrule')
    for di, ds in enumerate(DATASETS):
        for bi, bb in enumerate(BACKBONES):
            for ci, (col, clab) in enumerate(CONDITIONS):
                cells = _row_cells(df, ds, bb, col)
                if bi == 0 and ci == 0:
                    lead = f'\\multirow{{{rows_per_ds}}}{{*}}{{{ds}}} & \\multirow{{2}}{{*}}{{{bb}}} & {clab}'
                elif ci == 0:
                    lead = f' & \\multirow{{2}}{{*}}{{{bb}}} & {clab}'
                else:
                    lead = f' &  & {clab}'
                L.append(f'{lead} & ' + ' & '.join(cells) + ' \\\\')
            if bi != len(BACKBONES) - 1:
                L.append(f'\\cmidrule(l){{2-{lastcol}}}')
        if di != len(DATASETS) - 1:
            L.append('\\midrule')
    L.append('\\bottomrule')
    L.append('\\end{tabular}')
    return '\n'.join(L)


CAPTION = (
    'Instance GT-ROC between motif attribution and the planted cause, pooled over '
    'train+valid+test (graph-weighted), $\\times100$; subscript $=$ std over the 50 rules '
    '(each rule fold-averaged), per (dataset, backbone). The \\textbf{Vocab} column gives the '
    'evaluation setting: \\textbf{Filt} (exclude-unk) restricts to kept-motif nodes so all '
    'methods share one motif set, \\textbf{Full} (include-unk) scores all-node recovery. Per '
    '(dataset, backbone) and within each condition, the best is in \\textbf{bold} and methods '
    'not significantly worse (Welch two-sided $t$-test over the 50 rule values, Holm, '
    '$\\alpha{=}0.05$) are \\underline{underlined}. The GIN backbone and a learnable-unknown '
    'MoSE arm (MOSE$_{unk}$) were not trained in this campaign; PGExplainer entries rest on its '
    'available valid rules.')


def floatwrap(body, caption, label):
    return ('\\begin{table*}[t]\n\\centering\n\\small\n' + body +
            f'\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table*}}\n')


def build_agg_csv(df):
    """Per-cell aggregate for validation: mean/std/n for both Filt-all and Full-all."""
    out = []
    for ds in DATASETS:
        for bb in BACKBONES:
            for lbl, stem in METHODS:
                vf = rule_vector(df, ds, bb, stem, 'filt')
                vfu = rule_vector(df, ds, bb, stem, 'full')
                out.append(dict(
                    dataset=ds, backbone=bb, method=lbl, stem=stem,
                    n_rules=int(vf.size),
                    filt_all_mean=round(float(vf.mean()) * 100, 2) if vf.size else np.nan,
                    filt_all_std=round(float(vf.std(ddof=1)) * 100, 2) if vf.size > 1 else 0.0,
                    full_all_mean=round(float(vfu.mean()) * 100, 2) if vfu.size else np.nan,
                    full_all_std=round(float(vfu.std(ddof=1)) * 100, 2) if vfu.size > 1 else 0.0))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True,
                    help='gtroc_paper_artifacts/planted (reads <root>/<ds>/<rule>/<method>.csv)')
    ap.add_argument('--out', required=True, help='output dir (created; nothing overwritten)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load(args.root)
    if df.empty:
        raise SystemExit(f'no CSVs found under {args.root}')

    # coverage report -> stdout, so a short cell count is visible before trusting the table
    cov = (df.dropna(subset=['filt']).groupby(['dataset', 'method'])['rule']
           .nunique().unstack(fill_value=0))
    print('rules with a finite gtroc_filt_all, per dataset x method:')
    print(cov.to_string())

    tex = floatwrap(build_tex(df), CAPTION, 'tab:planted_gtroc')
    with open(os.path.join(args.out, 'planted_gtroc.tex'), 'w') as fh:
        fh.write(tex)
    build_agg_csv(df).to_csv(os.path.join(args.out, 'planted_gtroc_agg.csv'), index=False)
    print(f'\nwrote {args.out}/planted_gtroc.tex')
    print(f'wrote {args.out}/planted_gtroc_agg.csv')


if __name__ == '__main__':
    main()
