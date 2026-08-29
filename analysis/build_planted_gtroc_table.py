#!/usr/bin/env python3
r"""build_planted_gtroc_table.py — planted GT-ROC / Spurious-ROC / Family-ROC LaTeX tables from
evaluate_gtroc.py CSVs.

INPUT  : mose_replication_v2/gtroc_paper_artifacts/planted/<ds>/<rule>/<method>.csv
         (one row per backbone x fold; columns gtroc_{full,filt}_{...}, spur_{full,filt}_{...},
          fam_{full,filt}_{...}).
AGG    : per rule -> fold-average; per (dataset, backbone, method) -> mean +/- std over the 50
         rules, x100, 1 dp. Significance via analysis.table_sig (Welch two-sided + Holm), the SAME
         module the source-GT table uses. GT-ROC: higher is better (bold=max). Spurious/Family-ROC:
         lower is better (bold=min).

TABLES (all under <out>/, nothing existing overwritten):
  planted_gtroc.tex    : tab:planted_gtroc  -- Dataset x BB x Vocab{Filt,Full} rows, value gtroc_*_all.
  planted_spurroc.tex  : tab:planted_spurroc -- Dataset x BB rows (exclude-unk / spur_filt_all), with a
                         per-dataset column r_bar_sp = mean_std of the strongest planted confounder's
                         correlation over the 50 rules, read from rule_tiers.json (NOT the eval run).
  planted_famroc.tex   : tab:planted_famroc -- Dataset x BB rows (exclude-unk / fam_filt_all).
  planted_gtroc_agg.csv: per-cell mean/std/n for gtroc/spur/fam, Filt + Full (validation).
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from analysis import table_sig as ts
except ImportError:                      # when run from inside analysis/
    import table_sig as ts

DATASETS = ['BBBP', 'Mutagenicity', 'hERG']
BACKBONES = ['GCN', 'GAT', 'SAGE', 'PNA']
# (display label, method-file stem)
METHODS = [('GNNExpl.', 'gnnexplainer'), ('PGExpl.', 'pgexplainer'), ('MAGE', 'mage'),
           ('Occlusion', 'motif_occlusion'), ('GSAT', 'gsat'), ('MOSE', 'mose_fixed')]
ALPHA = 0.05
# value columns pulled from each per-cell CSV (the pooled "all" split, both conditions)
COLS = ['gtroc_filt_all', 'gtroc_full_all', 'spur_filt_all', 'spur_full_all',
        'fam_filt_all', 'fam_full_all']


def load(root):
    """Long frame: one row per (dataset, rule, method-stem, backbone, fold) with every value col."""
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
                    rec = dict(dataset=ds, rule=rule, method=stem,
                               backbone=str(r.get('backbone')), fold=r.get('fold'))
                    for c in COLS:
                        rec[c] = pd.to_numeric(pd.Series([r.get(c)]), errors='coerce').iloc[0]
                    rows.append(rec)
    return pd.DataFrame(rows)


def rule_vector(df, ds, bb, stem, col):
    """The 50 per-rule values (each rule fold-averaged) for one value column, NaNs dropped."""
    sub = df[(df.dataset == ds) & (df.backbone == bb) & (df.method == stem)]
    if sub.empty or col not in sub.columns:
        return np.array([])
    return np.asarray(sub.groupby('rule')[col].mean().dropna().values, float)


def fmt(mean, std):
    return f'{mean * 100:.1f}_{{{std * 100:.1f}}}'


def _row_cells(df, ds, bb, col, higher_better):
    """One table row for a value column; best/tie via table_sig (Welch+Holm). higher_better=True for
    GT-ROC (bold=max), False for Spurious/Family-ROC (bold=min). bold=\\boldmath, tie=\\underline."""
    cells = []
    for _, stem in METHODS:
        v = rule_vector(df, ds, bb, stem, col)
        cells.append(dict(key=stem,
                          mean=(float(v.mean()) if v.size else float('nan')),
                          std=(float(v.std(ddof=1)) if v.size > 1 else 0.0),
                          n=int(v.size)))
    tags = ts.decide(cells, higher_better=higher_better, alpha=ALPHA)
    out = []
    for c in cells:
        if not np.isfinite(c['mean']):
            out.append('--')
            continue
        s = f"${fmt(c['mean'], c['std'])}$"
        out.append(ts.wrap(s, tags.get(c['key'], 'plain'), 'bold_underline'))
    return out


# ── r_bar_sp : strongest-confounder correlation, averaged over the 50 rules (read, not evaluated) ──
def r_sp_bar(planted_root, ds):
    """(mean, std, n) of spurious_pos[0]['corr'] over the rules in rule_tiers.json. spurious_pos is
    stored corr-descending (rule_dnf.py), so entry 0 is the strongest confounder for that rule."""
    p = Path(planted_root) / ds / '_shared' / 'vocab' / ds / 'rbrics' / 'rule_tiers.json'
    if not p.exists():
        return (float('nan'), 0.0, 0)
    d = json.loads(p.read_text())
    vals = []
    for _rule, spec in d.items():
        sp = (spec or {}).get('spurious_pos') or []
        if sp and 'corr' in sp[0]:
            vals.append(float(sp[0]['corr']))
    vals = [v for v in vals if v == v]
    if not vals:
        return (float('nan'), 0.0, 0)
    return (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, len(vals))


# ── GT-ROC table: Dataset x BB x Vocab{Filt,Full}, explainer cols, bold=max ──
CONDITIONS = [('gtroc_filt_all', 'Filt'), ('gtroc_full_all', 'Full')]


def build_gtroc_tex(df):
    ncol = len(METHODS)
    lastcol = 3 + ncol
    rows_per_ds = len(BACKBONES) * len(CONDITIONS)
    L = ['% planted GT-ROC (instance); Filt=exclude-unk, Full=include-unk; both pooled tvt.',
         '% cell = mean_{std} x100 (1 dp); unit=rule (50 rules, each fold-averaged).',
         '\\begin{tabular}{lll' + 'c' * ncol + '}', '\\toprule',
         'Dataset & BB & Vocab & ' + ' & '.join(lbl for lbl, _ in METHODS) + ' \\\\', '\\midrule']
    for di, ds in enumerate(DATASETS):
        for bi, bb in enumerate(BACKBONES):
            for ci, (col, clab) in enumerate(CONDITIONS):
                cells = _row_cells(df, ds, bb, col, higher_better=True)
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
    L += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(L)


# ── Spurious-ROC table: Dataset x r_bar_sp x BB, explainer cols, bold=min, one row per bb ──
def build_spurroc_tex(df, planted_root):
    ncol = len(METHODS)
    L = ['% planted Spurious-ROC; cell = mean_{std} x100 (1 dp); unk=exclude; unit=rule; lower better',
         '\\begin{tabular}{lll' + 'c' * ncol + '}', '\\toprule',
         'Dataset & $\\bar r_{sp}$ & BB & ' + ' & '.join(lbl for lbl, _ in METHODS) + ' \\\\',
         '\\midrule']
    for di, ds in enumerate(DATASETS):
        m, sd, _n = r_sp_bar(planted_root, ds)
        rsp = f'${m:.2f}_{{{sd:.2f}}}$' if m == m else '--'
        for bi, bb in enumerate(BACKBONES):
            cells = _row_cells(df, ds, bb, 'spur_filt_all', higher_better=False)
            if bi == 0:
                lead = (f'\\multirow{{{len(BACKBONES)}}}{{*}}{{{ds}}} & '
                        f'\\multirow{{{len(BACKBONES)}}}{{*}}{{{rsp}}} & {bb}')
            else:
                lead = f' &  & {bb}'
            L.append(f'{lead} & ' + ' & '.join(cells) + ' \\\\')
        if di != len(DATASETS) - 1:
            L.append('\\midrule')
    L += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(L)


# ── Family-ROC table: Dataset x BB, explainer cols, bold=min, one row per bb (no r_bar_sp) ──
def build_famroc_tex(df):
    ncol = len(METHODS)
    L = ['% planted Family-ROC; cell = mean_{std} x100 (1 dp); unk=exclude; unit=rule; lower better',
         '\\begin{tabular}{ll' + 'c' * ncol + '}', '\\toprule',
         'Dataset & BB & ' + ' & '.join(lbl for lbl, _ in METHODS) + ' \\\\', '\\midrule']
    for di, ds in enumerate(DATASETS):
        for bi, bb in enumerate(BACKBONES):
            cells = _row_cells(df, ds, bb, 'fam_filt_all', higher_better=False)
            lead = f'\\multirow{{{len(BACKBONES)}}}{{*}}{{{ds}}} & {bb}' if bi == 0 else f' & {bb}'
            L.append(f'{lead} & ' + ' & '.join(cells) + ' \\\\')
        if di != len(DATASETS) - 1:
            L.append('\\midrule')
    L += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(L)


GTROC_CAP = (
    'Instance GT-ROC between motif attribution and the planted cause, pooled over '
    'train+valid+test (graph-weighted), $\\times100$; subscript $=$ std over the 50 rules '
    '(each rule fold-averaged), per (dataset, backbone). The \\textbf{Vocab} column: \\textbf{Filt} '
    '(exclude-unk) restricts to kept-motif nodes so all methods share one motif set, \\textbf{Full} '
    '(include-unk) scores all-node recovery. Per (dataset, backbone) and within each condition, the '
    'best is in \\textbf{bold} and methods not significantly worse (Welch two-sided $t$-test over the '
    '50 rule values, Holm, $\\alpha{=}0.05$) are \\underline{underlined}. The GIN backbone and a '
    'learnable-unknown MoSE arm (MOSE$_{unk}$) were not trained in this campaign.')
SPUR_CAP = (
    'Spurious-ROC between motif attribution and the planted cause, \\emph{lower is better}, '
    '$\\times100$; subscript $=$ std over the 50 rules (each rule fold-averaged), per (dataset, '
    'backbone). Evaluated on the kept vocabulary (\\emph{exclude-unk}), so all methods share one '
    'motif set. Per (dataset, backbone) the best (lowest) is in \\textbf{bold} and methods not '
    'significantly worse (Welch two-sided $t$-test over the 50 rule values, Holm, $\\alpha{=}0.05$) '
    'are \\underline{underlined}. The per-dataset column $\\bar r_{sp}$ gives the mean$_{std}$ '
    'strength of the strongest planted confounder over the 50 rules (the correlation a shortcut '
    'could exploit). The GIN backbone and a learnable-unknown MoSE arm (MOSE$_{unk}$) were not '
    'trained in this campaign.')
FAM_CAP = (
    'Family-ROC between motif attribution and the planted cause, \\emph{lower is better}, '
    '$\\times100$; subscript $=$ std over the 50 rules (each rule fold-averaged), per (dataset, '
    'backbone). Evaluated on the kept vocabulary (\\emph{exclude-unk}), so all methods share one '
    'motif set. Per (dataset, backbone) the best (lowest) is in \\textbf{bold} and methods not '
    'significantly worse (Welch two-sided $t$-test over the 50 rule values, Holm, $\\alpha{=}0.05$) '
    'are \\underline{underlined}. The GIN backbone and a learnable-unknown MoSE arm (MOSE$_{unk}$) '
    'were not trained in this campaign.')


def floatwrap(body, caption, label):
    return ('\\begin{table*}[t]\n\\centering\n\\small\n' + body +
            f'\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table*}}\n')


def build_agg_csv(df, planted_root):
    out = []
    for ds in DATASETS:
        for bb in BACKBONES:
            for lbl, stem in METHODS:
                rec = dict(dataset=ds, backbone=bb, method=lbl, stem=stem)
                for c in COLS:
                    v = rule_vector(df, ds, bb, stem, c)
                    rec[f'{c}_mean'] = round(float(v.mean()) * 100, 2) if v.size else np.nan
                    rec[f'{c}_std'] = round(float(v.std(ddof=1)) * 100, 2) if v.size > 1 else 0.0
                    rec[f'{c}_n'] = int(v.size)
                out.append(rec)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True,
                    help='gtroc_paper_artifacts/planted (reads <root>/<ds>/<rule>/<method>.csv)')
    ap.add_argument('--out', required=True, help='output dir (created; nothing overwritten)')
    ap.add_argument('--planted_root', required=True,
                    help='planted_v2 base (reads <ds>/_shared/vocab/<ds>/rbrics/rule_tiers.json '
                         'for the r_bar_sp column)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load(args.root)
    if df.empty:
        raise SystemExit(f'no CSVs found under {args.root}')

    for metric, col in (('gtroc', 'gtroc_filt_all'), ('spur', 'spur_filt_all'), ('fam', 'fam_filt_all')):
        cov = (df.dropna(subset=[col]).groupby(['dataset', 'method'])['rule']
               .nunique().unstack(fill_value=0))
        print(f'== rules with finite {col} (per dataset x method) ==')
        print(cov.to_string())
    print('\n== r_bar_sp (per dataset, from rule_tiers.json) ==')
    for ds in DATASETS:
        m, sd, n = r_sp_bar(args.planted_root, ds)
        print(f'  {ds}: {m:.3f} +/- {sd:.3f}  (n={n})')

    open(os.path.join(args.out, 'planted_gtroc.tex'), 'w').write(
        floatwrap(build_gtroc_tex(df), GTROC_CAP, 'tab:planted_gtroc'))
    open(os.path.join(args.out, 'planted_spurroc.tex'), 'w').write(
        floatwrap(build_spurroc_tex(df, args.planted_root), SPUR_CAP, 'tab:planted_spurroc'))
    open(os.path.join(args.out, 'planted_famroc.tex'), 'w').write(
        floatwrap(build_famroc_tex(df), FAM_CAP, 'tab:planted_famroc'))
    build_agg_csv(df, args.planted_root).to_csv(
        os.path.join(args.out, 'planted_gtroc_agg.csv'), index=False)
    print(f'\nwrote planted_gtroc.tex / planted_spurroc.tex / planted_famroc.tex / '
          f'planted_gtroc_agg.csv -> {args.out}')


if __name__ == '__main__':
    main()
