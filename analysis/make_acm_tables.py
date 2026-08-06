#!/usr/bin/env python3
"""make_acm_tables.py — fold-aggregate the per-fold metric set (build_metric_set.py
output) and emit ACM LaTeX tables.

AGGREGATION RULE (enforced): values are aggregated ONLY across folds. Every other
dimension — dataset, backbone, model, fragmentation, filtered, gt_tier, gt_rule,
model_type, impact_basis — is a grouping key, so each cell aggregates at most the
5 CV folds. fold_aggregate() ASSERTS max-folds-per-cell <= 5 and prints the
distribution; anything above 5 means a dimension leaked and it aborts.

Every table is a single clean slice — one metric, one fragmentation, one filtered
value, one split (and one planted rule for synthetic) — never mixed.

SPLITS: every metric is emitted per split (train/valid/test) SEPARATELY, EXCEPT
grouped Pearson/Spearman, which are computed jointly over train+val+test by the
calculator (pooled 'alltest') and therefore have no per-split form.

filtered is a SEPARATE table dimension (models shown on their native vocab only):
  full (filtered=False): GNNExpl, PGExpl, MAGE, Occlusion, GSAT, MotifSAT
  filt (filtered=True):  GNNExpl, PGExpl, MAGE, Occlusion, MOSE
predictive tables use the prediction-making models: full = Vanilla/GSAT/MotifSAT,
filt = MOSE. Datasets with no finite value for a metric (e.g. AUC on a regression
set) are dropped from that table.

Output tree (out-dir):
  fold_aggregated_main.csv, fold_aggregated_synthetic.csv
  main/{faith_grouped,faith_instance,gtroc,predictive}/*.tex (+ _coverage.csv)
  synthetic/{gtroc,diagnostics}/*.tex (+ _coverage.csv)

Usage:
  python analysis/make_acm_tables.py --main main_metrics_per_fold.csv \
      --supp supp_metrics_per_fold.csv --out-dir tables [--std-ddof 1]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CELL_KEYS = ['dataset', 'backbone', 'model', 'model_type', 'impact_basis',
             'fragmentation', 'filtered', 'gt_tier', 'gt_rule', 'synthetic']

FULL_MODELS = ['GNNExplainer', 'PGExplainer', 'MAGE', 'MotifOcclusion', 'GSAT', 'MotifSAT']
FULL_HEAD = ['GNNExpl.', 'PGExpl.', 'MAGE', 'Occlusion', 'GSAT', 'MotifSAT']
FILT_MODELS = ['GNNExplainer', 'PGExplainer', 'MAGE', 'MotifOcclusion', 'MoSE']
FILT_HEAD = ['GNNExpl.', 'PGExpl.', 'MAGE', 'Occlusion', 'MOSE']
PRED_FULL_MODELS, PRED_FULL_HEAD = ['Vanilla', 'GSAT', 'MotifSAT'], ['Vanilla', 'GSAT', 'MotifSAT']
PRED_FILT_MODELS, PRED_FILT_HEAD = ['MoSE'], ['MOSE']

BACKBONE_ORDER = ['GIN', 'GCN', 'GAT', 'SAGE', 'PNA']
DATASET_ORDER = ['BBBP', 'Mutagenicity', 'esol', 'hERG', 'Lipophilicity',
                 'ogbg-molbace', 'Benzene_Verified_GT', 'Fluoride_Carbonyl_Verified_GT',
                 'Alkane_Carbonyl_Verified_GT', 'mutag']
DATASET_TEX = {'ogbg-molbace': 'molbace', 'Benzene_Verified_GT': 'Benzene',
               'Fluoride_Carbonyl_Verified_GT': 'Fluoride+Carbonyl',
               'Alkane_Carbonyl_Verified_GT': 'Alkane+Carbonyl'}
FRAGS = ['rbrics', 'FGFirst-RDKIT', 'ertl', 'custom-FG']
FRAG_TEX = {'rbrics': 'rBRICS', 'FGFirst-RDKIT': 'FGFirst-RDKIT', 'ertl': 'ertl', 'custom-FG': 'custom-FG'}
SPLITS = ['train', 'valid', 'test']


def fold_aggregate(df: pd.DataFrame, ddof: int, tag: str) -> pd.DataFrame:
    non_metric = set(CELL_KEYS + ['fold', 'n_folds', 'run_path', 'base_vocab'])
    metric_cols = [c for c in df.columns
                   if c not in non_metric and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(CELL_KEYS, dropna=False)
    sizes = g.size()
    mx = int(sizes.max())
    dist = {int(k): int(v) for k, v in sizes.value_counts().sort_index().items()}
    print(f'[{tag}] folds-per-cell distribution {dist}  MAX={mx}')
    assert mx <= 5, f'CONTAMINATION: a cell aggregates {mx} > 5 folds — a dimension leaked'
    mean = g[metric_cols].mean().add_prefix('mean_')
    std = g[metric_cols].std(ddof=ddof).add_prefix('std_')
    cnt = g[metric_cols].count().add_prefix('n_')     # count() excludes NaN (vectorised)
    return pd.concat([mean, std, cnt], axis=1).reset_index()


def _fmt(mean, std, best):
    if not np.isfinite(mean):
        return '--'
    body = f'{mean:.2f}\\pm{std:.2f}' if np.isfinite(std) else f'{mean:.2f}'
    return f'$\\mathbf{{{body}}}$' if best else f'${body}$'


def _fmt_mean(mean, best):
    """Mean-only cell (for the dense combined-rule longtables where mean±std would
    overflow the page width)."""
    if not np.isfinite(mean):
        return '--'
    return f'$\\mathbf{{{mean:.2f}}}$' if best else f'${mean:.2f}$'


# ---- per-scheme merged model specs: (model, head, filtered_vocab) --------------
# One table per fragmentation scheme; every method shown on its NATIVE vocab so
# MoSE (filtered-only) sits beside the rest. A dagger marks filtered-vocab columns.
DAG = r'\textsuperscript{$\dagger$}'
# predictive: the prediction-making models. Vanilla/GSAT/MotifSAT = full vocab
# (Vanilla is nominally fragmentation-independent — shown per scheme as-is), MoSE = filtered.
PRED_SPEC = [('Vanilla', 'Vanilla', False), ('GSAT', 'GSAT', False),
             ('MotifSAT', 'MotifSAT', False), ('MoSE', 'MOSE', True)]
# MAIN explanation: post-hoc baselines on FILTERED (apples-to-apples with MoSE),
# ante-hoc GSAT/MotifSAT full (no filtered variant exists), MoSE filtered.
EXPL_SPEC_MAIN = [('GNNExplainer', 'GNNExpl.', True), ('PGExplainer', 'PGExpl.', True),
                  ('MAGE', 'MAGE', True), ('MotifOcclusion', 'Occlusion', True),
                  ('GSAT', 'GSAT', False), ('MotifSAT', 'MotifSAT', False), ('MoSE', 'MOSE', True)]
# SYNTHETIC (planted): baselines have NO filtered runs in supp -> all full except MoSE.
EXPL_SPEC_SYNTH = [('GNNExplainer', 'GNNExpl.', False), ('PGExplainer', 'PGExpl.', False),
                   ('MAGE', 'MAGE', False), ('MotifOcclusion', 'Occlusion', False),
                   ('GSAT', 'GSAT', False), ('MotifSAT', 'MotifSAT', False), ('MoSE', 'MOSE', True)]


# grouped faithfulness: ALL methods on the FILTERED vocab (baselines already filt;
# GSAT/MotifSAT via the validated offline re-score; MoSE native filtered).
GROUPED_SPEC = [('GNNExplainer', 'GNNExpl.', True), ('PGExplainer', 'PGExpl.', True),
                ('MAGE', 'MAGE', True), ('MotifOcclusion', 'Occlusion', True),
                ('GSAT', 'GSAT', True), ('MotifSAT', 'MotifSAT', True), ('MoSE', 'MOSE', True)]
# full-vocab grouped: everyone EXCEPT MoSE (which has no full-vocab run).
FULL_GROUPED_SPEC = [('GNNExplainer', 'GNNExpl.', False), ('PGExplainer', 'PGExpl.', False),
                     ('MAGE', 'MAGE', False), ('MotifOcclusion', 'Occlusion', False),
                     ('GSAT', 'GSAT', False), ('MotifSAT', 'MotifSAT', False)]


def _spec_parts(specs):
    models = [m for m, _, _ in specs]
    mixed = len({f for _, _, f in specs}) > 1        # dagger only disambiguates a mix
    head = [(h + DAG if f else h) if mixed else h for _, h, f in specs]
    mfilt = {m: f for m, _, f in specs}
    return models, head, mfilt


def _vocab_note(specs):
    has_f = any(f for _, _, f in specs)
    has_u = any(not f for _, _, f in specs)
    if has_f and has_u:
        return r' $\dagger$=filtered vocab, unmarked=full vocab.'
    return ' All filtered vocab.' if has_f else ' All full vocab.'


def build_table(agg, metric, frag, gt_tiers, specs, caption, label, hib):
    mcol, scol, ncol = f'mean_{metric}', f'std_{metric}', f'n_{metric}'
    if mcol not in agg.columns:
        return None, None
    models, head, mfilt = _spec_parts(specs)
    sub = agg[(agg['fragmentation'] == frag) & (agg['gt_tier'].isin(gt_tiers))
              & (agg['model'].isin(models))]
    sub = sub[sub[mcol].notna()]
    if sub.empty:
        return None, None
    datasets = [d for d in DATASET_ORDER if d in set(sub['dataset'])]
    body, cov, nvals = [], [], []
    for di, d in enumerate(datasets):
        bbs = [b for b in BACKBONE_ORDER if b in set(sub[sub['dataset'] == d]['backbone'])]
        for bi, bb in enumerate(bbs):
            vals = {}
            for mod in models:                 # each model on its OWN vocab
                r = sub[(sub['dataset'] == d) & (sub['backbone'] == bb)
                        & (sub['model'] == mod) & (sub['filtered'] == mfilt[mod])]
                if len(r):
                    r0 = r.iloc[0]
                    vals[mod] = (r0[mcol], r0[scol])
                    n = int(r0[ncol]) if np.isfinite(r0[ncol]) else 0
                    nvals.append(n)
                    cov.append(dict(dataset=d, backbone=bb, model=mod, filtered=mfilt[mod],
                                    mean=r0[mcol], std=r0[scol], n_folds=n))
            finite = {k: v[0] for k, v in vals.items() if np.isfinite(v[0])}
            best = (max(finite, key=finite.get) if hib else min(finite, key=finite.get)) if finite else None
            cells = [_fmt(*vals[m], m == best) if m in vals else '--' for m in models]
            dcol = DATASET_TEX.get(d, d) if bi == 0 else ''
            body.append((dcol, bb, cells, bi == 0 and di > 0))
    if not cov:
        return None, None
    nmin, nmax = min(nvals), max(nvals)
    foldnote = f'{nmin}' if nmin == nmax else f'{nmin}--{nmax}'
    ncols = 'll' + 'c' * len(models)
    env = 'table*' if len(models) > 4 else 'table'
    lines = [f'\\begin{{{env}}}[t]', r'\centering',
             f'\\caption{{{caption} Mean$\\pm$std across folds ({foldnote} folds/cell); '
             f"rows are backbone.{_vocab_note(specs)} '--' = no completed cell.}}",
             f'\\label{{{label}}}', r'\scriptsize', r'\setlength{\tabcolsep}{3pt}',
             f'\\begin{{tabular}}{{{ncols}}}', r'\toprule',
             'Dataset & BB & ' + ' & '.join(head) + r' \\', r'\midrule']
    for dcol, bb, cells, sep in body:
        if sep:
            lines.append(r'\midrule')
        lines.append(f'{dcol} & {bb} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', f'\\end{{{env}}}']
    return '\n'.join(lines), pd.DataFrame(cov)


def emit(agg, metric, frag, gt_tiers, specs, hib, outdir, stem, cap_metric, cap_extra=''):
    cap = f'{cap_metric} --- {FRAG_TEX[frag]} fragmentation{cap_extra}.'
    tex, cov = build_table(agg, metric, frag, gt_tiers, specs, cap, f'tab:{stem}', hib)
    if tex is None:
        return 0
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f'{stem}.tex').write_text(tex + '\n')
    cov.to_csv(outdir / f'{stem}_coverage.csv', index=False)
    return 1


def build_split_table(agg, base, is_prefix, frag, gt_tiers, specs, caption, label, hib):
    """One table per fragmentation with train/valid/test as SUB-COLUMNS under each
    method; each method pulled on its OWN vocab (specs). Best is bolded PER split."""
    def cols(split):
        c = f'{split}_{base}' if is_prefix else f'{base}_{split}'
        return f'mean_{c}', f'std_{c}', f'n_{c}'
    if not any(cols(s)[0] in agg.columns for s in SPLITS):
        return None, None
    models, head, mfilt = _spec_parts(specs)
    sub = agg[(agg['fragmentation'] == frag) & (agg['gt_tier'].isin(gt_tiers))
              & (agg['model'].isin(models))]
    if sub.empty:
        return None, None

    def has_any(d):
        s = sub[sub['dataset'] == d]
        return any(cols(sp)[0] in s.columns and s[cols(sp)[0]].notna().any() for sp in SPLITS)
    datasets = [d for d in DATASET_ORDER if d in set(sub['dataset']) and has_any(d)]
    if not datasets:
        return None, None

    body, cov, nvals = [], [], []
    for di, d in enumerate(datasets):
        bbs = [b for b in BACKBONE_ORDER if b in set(sub[sub['dataset'] == d]['backbone'])]
        for bi, bb in enumerate(bbs):
            cm = {}                               # (model, split) -> (mean, std)
            for mod in models:
                r = sub[(sub['dataset'] == d) & (sub['backbone'] == bb)
                        & (sub['model'] == mod) & (sub['filtered'] == mfilt[mod])]
                if not len(r):
                    continue
                r0 = r.iloc[0]
                for sp in SPLITS:
                    mc, sc, nc = cols(sp)
                    if mc not in r0:
                        continue
                    mv, sv = r0[mc], r0.get(sc, np.nan)
                    cm[(mod, sp)] = (mv, sv)
                    if np.isfinite(mv):
                        n = int(r0[nc]) if (nc in r0 and np.isfinite(r0[nc])) else 0
                        nvals.append(n)
                        cov.append(dict(dataset=d, backbone=bb, model=mod, filtered=mfilt[mod],
                                        split=sp, mean=mv, std=sv, n_folds=n))
            best = {}
            for sp in SPLITS:
                fin = {m: cm[(m, sp)][0] for m in models
                       if (m, sp) in cm and np.isfinite(cm[(m, sp)][0])}
                best[sp] = (max(fin, key=fin.get) if hib else min(fin, key=fin.get)) if fin else None
            cells = [_fmt(*cm[(mod, sp)], mod == best[sp]) if (mod, sp) in cm else '--'
                     for mod in models for sp in SPLITS]
            dcol = DATASET_TEX.get(d, d) if bi == 0 else ''
            body.append((dcol, bb, cells, bi == 0 and di > 0))
    if not cov:
        return None, None

    nmin, nmax = min(nvals), max(nvals)
    foldnote = f'{nmin}' if nmin == nmax else f'{nmin}--{nmax}'
    nm = len(models)
    ncols = 'll' + 'ccc' * nm
    grp = ' & '.join(f'\\multicolumn{{3}}{{c}}{{{h}}}' for h in head)
    cmids = ' '.join(f'\\cmidrule(lr){{{3 + i * 3}-{5 + i * 3}}}' for i in range(nm))
    subhdr = ' & '.join(['Tr & Va & Te'] * nm)
    wide = nm * 3 > 3
    env = 'table*' if wide else 'table'
    resize = nm * 3 > 9                           # >=12-col tables: fit to \textwidth
    lines = [f'\\begin{{{env}}}[t]', r'\centering',
             f'\\caption{{{caption} Mean$\\pm$std across folds ({foldnote} folds/cell); '
             f"rows are backbone; Tr/Va/Te = train/valid/test sub-columns.{_vocab_note(specs)} '--' = no completed cell.}}",
             f'\\label{{{label}}}', r'\scriptsize', r'\setlength{\tabcolsep}{2pt}']
    if resize:
        lines.append(r'\resizebox{\textwidth}{!}{%')
    lines += [f'\\begin{{tabular}}{{{ncols}}}', r'\toprule',
              'Dataset & BB & ' + grp + r' \\', cmids,
              ' &  & ' + subhdr + r' \\', r'\midrule']
    for dcol, bb, cells, sep in body:
        if sep:
            lines.append(r'\midrule')
        lines.append(f'{dcol} & {bb} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    if resize:
        lines.append(r'}')
    lines.append(f'\\end{{{env}}}')
    return '\n'.join(lines), pd.DataFrame(cov)


def emit_split(agg, base, is_prefix, tex, hib, frag, gt_tiers, specs, outdir, stem, cap_extra=''):
    cap = f'{tex} --- {FRAG_TEX[frag]} fragmentation{cap_extra}.'
    t, cov = build_split_table(agg, base, is_prefix, frag, gt_tiers, specs, cap, f'tab:{stem}', hib)
    if t is None:
        return 0
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f'{stem}.tex').write_text(t + '\n')
    cov.to_csv(outdir / f'{stem}_coverage.csv', index=False)
    return 1


def _rule_short(rule):
    return rule.replace('dnf_k', 'k').replace('_r', 'r')     # dnf_k2_r1 -> k2r1


def build_ruled_table(agg, base, is_prefix, frag, specs, caption, label, hib):
    """SYNTHETIC table: ALL planted DNF rules in one table, each rule a SUB-ROW
    under (dataset, backbone); train/valid/test as sub-columns; each method on its
    OWN vocab (specs). Best model is bolded per (row, split)."""
    def cols(split):
        c = f'{split}_{base}' if is_prefix else f'{base}_{split}'
        return f'mean_{c}', f'std_{c}', f'n_{c}'
    if not any(cols(s)[0] in agg.columns for s in SPLITS):
        return None, None
    models, head, mfilt = _spec_parts(specs)
    sub = agg[(agg['fragmentation'] == frag) & (agg['gt_tier'] == 'planted')
              & (agg['model'].isin(models))]
    if sub.empty:
        return None, None
    rules = sorted(r for r in sub['gt_rule'].dropna().unique() if r)
    if not rules:
        return None, None

    def has_any(d):
        s = sub[sub['dataset'] == d]
        return any(cols(sp)[0] in s.columns and s[cols(sp)[0]].notna().any() for sp in SPLITS)
    datasets = [d for d in DATASET_ORDER if d in set(sub['dataset']) and has_any(d)]
    if not datasets:
        return None, None

    body, cov, nvals = [], [], []
    for di, d in enumerate(datasets):
        bbs = [b for b in BACKBONE_ORDER if b in set(sub[sub['dataset'] == d]['backbone'])]
        for bi, bb in enumerate(bbs):
            for ri, rule in enumerate(rules):
                cm = {}
                for mod in models:
                    r = sub[(sub['dataset'] == d) & (sub['backbone'] == bb)
                            & (sub['model'] == mod) & (sub['filtered'] == mfilt[mod])
                            & (sub['gt_rule'] == rule)]
                    if not len(r):
                        continue
                    r0 = r.iloc[0]
                    for sp in SPLITS:
                        mc, sc, nc = cols(sp)
                        if mc not in r0:
                            continue
                        mv, sv = r0[mc], r0.get(sc, np.nan)
                        cm[(mod, sp)] = (mv, sv)
                        if np.isfinite(mv):
                            n = int(r0[nc]) if (nc in r0 and np.isfinite(r0[nc])) else 0
                            nvals.append(n)
                            cov.append(dict(dataset=d, backbone=bb, rule=rule, model=mod,
                                            filtered=mfilt[mod], split=sp, mean=mv, std=sv,
                                            n_folds=n))
                best = {}
                for sp in SPLITS:
                    fin = {m: cm[(m, sp)][0] for m in models
                           if (m, sp) in cm and np.isfinite(cm[(m, sp)][0])}
                    best[sp] = (max(fin, key=fin.get) if hib else min(fin, key=fin.get)) if fin else None
                cells = [_fmt_mean(cm[(mod, sp)][0], mod == best[sp]) if (mod, sp) in cm else '--'
                         for mod in models for sp in SPLITS]
                dcol = DATASET_TEX.get(d, d) if (bi == 0 and ri == 0) else ''
                bbcol = bb if ri == 0 else ''
                body.append((dcol, bbcol, _rule_short(rule), cells,
                             di > 0 and bi == 0 and ri == 0,      # dataset separator
                             bi > 0 and ri == 0))                 # backbone separator
    if not cov:
        return None, None

    nmin, nmax = min(nvals), max(nvals)
    foldnote = f'{nmin}' if nmin == nmax else f'{nmin}--{nmax}'
    nm = len(models)
    lastcol = 3 + nm * 3
    ncols = 'lll' + 'ccc' * nm
    grp = ' & '.join(f'\\multicolumn{{3}}{{c}}{{{h}}}' for h in head)
    cmids = ' '.join(f'\\cmidrule(lr){{{4 + i * 3}-{6 + i * 3}}}' for i in range(nm))
    subhdr = ' & '.join(['Tr & Va & Te'] * nm)
    cap = (f'{caption} All planted DNF rules as sub-rows (k$X$r$Y$ $=$ dnf\\_k$X$\\_r$Y$). '
           f'Mean across folds ({foldnote} folds/cell); std omitted for width '
           f'(see per-fold / \\_coverage.csv);{_vocab_note(specs)} '
           f"Tr/Va/Te = train/valid/test. '--' = no completed cell.")
    # These tables are BOTH wide (15--18 data cols) AND tall (80 rows). A float
    # (table/table*) cannot page-break; \resizebox cannot wrap a longtable; and
    # `landscape' does NOT widen \textwidth under geometry. So: a portrait longtable
    # (page-breaks, repeats the header) with MEAN-ONLY cells, which fits the column
    # width. REQUIRES \usepackage{longtable} + \usepackage{booktabs}; place in a
    # single-column region / appendix in the ACM twocolumn template (longtable needs
    # one column). std for these cells lives in the {stem}_coverage.csv.
    head_block = ['\\toprule',
                  'Dataset & BB & Rule & ' + grp + r' \\', cmids,
                  ' &  &  & ' + subhdr + r' \\', r'\midrule']
    lines = [r'\begingroup', r'\scriptsize', r'\setlength{\tabcolsep}{2pt}',
             f'\\begin{{longtable}}{{{ncols}}}',
             f'\\caption{{{cap}}}\\label{{{label}}}\\\\']
    lines += head_block + [r'\endfirsthead']
    lines += [f'\\multicolumn{{{lastcol}}}{{c}}{{\\tablename\\ \\thetable{{}}: continued}} \\\\']
    lines += head_block + [r'\endhead']
    lines += [r'\midrule',
              f'\\multicolumn{{{lastcol}}}{{r}}{{\\scriptsize continued on next page}} \\\\',
              r'\endfoot', r'\bottomrule', r'\endlastfoot']
    for dcol, bbcol, rulecol, cells, dsep, bbsep in body:
        if dsep:
            lines.append(r'\midrule')
        elif bbsep:
            lines.append(f'\\cmidrule(l){{2-{lastcol}}}')
        lines.append(f'{dcol} & {bbcol} & {rulecol} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\end{longtable}', r'\endgroup']
    return '\n'.join(lines), pd.DataFrame(cov)


def emit_ruled(agg, base, is_prefix, tex, hib, frag, specs, outdir, stem):
    cap = f'{tex} (planted GT) --- {FRAG_TEX[frag]} fragmentation.'
    t, cov = build_ruled_table(agg, base, is_prefix, frag, specs, cap, f'tab:{stem}', hib)
    if t is None:
        return 0
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f'{stem}.tex').write_text(t + '\n')
    cov.to_csv(outdir / f'{stem}_coverage.csv', index=False)
    return 1


# grouped Pearson/Spearman: JOINT (no split), weighted & unweighted only.
# The incl-UNK / excl-UNK axis is DEGENERATE: UNK (motif_id<0) is dropped upstream
# by the mask cache before the grouped point cloud is formed, so n_motifs_inclunk ==
# n_motifs_exclunk in every run and incl-UNK is bit-identical to excl-UNK (verified
# across all 20,832 harvested rows). We therefore emit UNK-excluded only; the label
# notes incl-UNK is identical so the omission is explicit, not silent.
GROUPED_VARIANTS = [('w', 'exclunk', 'weighted, UNK excl. (incl-UNK identical)'),
                    ('u', 'exclunk', 'unweighted, UNK excl. (incl-UNK identical)')]
# per-split explanation metrics (suffix _<split>): (col-base, TeX, higher_is_better)
INSTANCE_METRICS = [('instance_pearson_agnostic', 'Per-instance faithfulness (Pearson)', True),
                    ('instance_spearman_agnostic', 'Per-instance faithfulness (Spearman)', True)]
GTROC_METRICS = [('grouped_gtroc_fired', 'Grouped GT-ROC (fired clauses)', True),
                 ('instance_gtroc_fired', 'Per-instance GT-ROC (fired disjuncts)', True)]
DIAG_METRICS = [('spurious_roc', 'Spurious-motif ROC (lower better)', False),
                ('family_roc', 'Family-motif ROC (lower better)', False)]
# predictive (prefix <split>_): (col-suffix, TeX, higher_is_better)
PRED_METRICS = [('auc', 'AUC', True), ('rmse_orig', 'RMSE (orig units)', False),
                ('mae_orig', 'MAE (orig units)', False), ('rmse', 'RMSE (normalised)', False),
                ('mae', 'MAE (normalised)', False)]


def gen_main(agg, out):
    """One table per (metric, fragmentation) — all methods merged, each on its native
    vocab. Predictive uses PRED_SPEC (Vanilla/GSAT/MotifSAT + MoSE); explanation uses
    EXPL_SPEC_MAIN (post-hoc baselines filtered + GSAT/MotifSAT full + MoSE filtered)."""
    n = 0
    for frag in FRAGS:
        # 1) grouped Pearson & Spearman — JOINT (no split)
        for stat, statname in (('pearson', 'Pearson'), ('spearman', 'Spearman')):
            for w, unk, desc in GROUPED_VARIANTS:
                col = f'grouped_{stat}_agnostic_{w}_{unk}'
                cap = f'Grouped faithfulness ({statname}, {desc}), joint train+val+test'
                n += emit(agg, col, frag, ['none', 'source'], FULL_GROUPED_SPEC, True,
                          out / 'main' / 'faith_grouped', f'{col}__{frag}_full', cap)
                n += emit(agg, col, frag, ['none', 'source'], GROUPED_SPEC, True,
                          out / 'main' / 'faith_grouped', f'{col}__{frag}_filt', cap)
        # 2) per-split metrics — ONE table each, train/valid/test as sub-columns
        for base, tex, hib in INSTANCE_METRICS:
            n += emit_split(agg, base, False, tex, hib, frag, ['none', 'source'],
                            EXPL_SPEC_MAIN, out / 'main' / 'faith_instance', f'{base}__{frag}')
        for base, tex, hib in GTROC_METRICS:          # source-GT only in MAIN
            n += emit_split(agg, base, False, tex, hib, frag, ['source'],
                            EXPL_SPEC_MAIN, out / 'main' / 'gtroc', f'{base}__{frag}', ', source GT')
        for suf, tex, hib in PRED_METRICS:            # predictive: prefix <split>_
            n += emit_split(agg, suf, True, tex, hib, frag, ['none', 'source'],
                            PRED_SPEC, out / 'main' / 'predictive', f'{suf}__{frag}')
    return n


def gen_synthetic(agg, out):
    n = 0
    for frag in FRAGS:                    # ONE table per (metric, frag); rules = sub-rows
        for base, tex, hib in GTROC_METRICS:
            n += emit_ruled(agg, base, False, tex, hib, frag, EXPL_SPEC_SYNTH,
                            out / 'synthetic' / 'gtroc', f'{base}__{frag}')
        for base, tex, hib in DIAG_METRICS:
            n += emit_ruled(agg, base, False, tex, hib, frag, EXPL_SPEC_SYNTH,
                            out / 'synthetic' / 'diagnostics', f'{base}__{frag}')
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main', required=True)
    ap.add_argument('--supp', default=None)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--std-ddof', type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    magg = fold_aggregate(pd.read_csv(args.main, low_memory=False), args.std_ddof, 'main')
    magg.to_csv(out / 'fold_aggregated_main.csv', index=False)
    nm = gen_main(magg, out)
    ns = 0
    if args.supp:
        sagg = fold_aggregate(pd.read_csv(args.supp, low_memory=False), args.std_ddof, 'supp')
        sagg.to_csv(out / 'fold_aggregated_synthetic.csv', index=False)
        ns = gen_synthetic(sagg, out)
    print(f'wrote {nm} MAIN + {ns} SYNTHETIC tables into {out}')


if __name__ == '__main__':
    main()
