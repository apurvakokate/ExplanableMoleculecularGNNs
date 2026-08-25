#!/usr/bin/env python3
"""Compact publication tables (table*) for GT-ROC, Spurious-ROC, Family-ROC,
matching the grouped-Pearson house style.

Aggregation: per (dataset, backbone). The unit is one rule (its 5 folds averaged,
the only permitted aggregation); the cell is mean$_{\\text{std}}$ over the 50 rules,
$\\times100$, 1 dp. Per (dataset, backbone) row the best method is \\textbf{bold} and
methods not significantly worse (Welch two-sided t-test, Holm, alpha=0.05) are
\\underline{underlined}. Direction: GT-ROC higher is better; Spurious/Family-ROC
lower is better.

The Spurious-ROC table carries the spurious STRENGTH: a per-dataset column
$\\bar r_{sp}$ = mean$_{std}$ of the strongest-confounder correlation over the 50
rules (backbone-independent, so multirow'd with the dataset).

Columns: GNNExpl. PGExpl. MAGE Occlusion GSAT MOSE.
  (No GIN backbone and no MoSE$_U$ arm exist in this campaign.)
PGExplainer is INCOMPLETE (952/3000 valid); its cells rest on available valid rules.

Usage: python build_paper_tables.py --root <planted_v2> --out <dir>
"""
import argparse, csv, glob, json, math, os, statistics as st
from collections import defaultdict

DS = ['BBBP', 'Mutagenicity', 'hERG']
BB = ['GCN', 'GAT', 'SAGE', 'PNA']
COLS = [('GNNExpl.', 'gnnexplainer'), ('PGExpl.', 'pgexplainer'), ('MAGE', 'mage'),
        ('Occlusion', 'motif_occlusion'), ('GSAT', 'gsat'), ('MOSE', 'mose')]
METRICS = {'gtroc': ('gtroc_instance', True), 'spurroc': ('spurious_roc', False),
           'famroc': ('family_roc', False)}
DS_TEX = {'BBBP': 'BBBP', 'Mutagenicity': 'Mutagenicity', 'hERG': 'hERG'}


def fl(x):
    try:
        v = float(x); return v if v == v else None
    except (TypeError, ValueError):
        return None


def rulekey(r):
    return (int(r.split('_k')[1].split('_')[0]), int(r.split('_r')[1]))


def load(root, vocab_root):
    # rule_tiers.json (spurious strength) lives in the planted SOURCE tree (planted_v2),
    # which is a DIFFERENT tree from the metrics rollups (mose_replication_v2/rollups/planted).
    strength = {}
    for d in DS:
        t = json.load(open(f'{vocab_root}/{d}/_shared/vocab/{d}/rbrics/rule_tiers.json'))
        for rule, r in t.items():
            strength[(d, rule)] = r['correlate_r']
    # perfold[(ds,rule,bb,method,metric)] -> list over folds (unk=exclude only)
    perfold = defaultdict(lambda: defaultdict(list))
    # A concurrent-write race left some rollups with duplicate (ds,rule,bb,method,fold)
    # rows (identical to ~1e-8); dedup so a fold is not double-counted in its mean.
    seen = set()
    for f in glob.glob(f'{root}/*/*/all/metrics_*_unk-exclude.csv'):
        for r in csv.DictReader(open(f)):
            if r['unk'] != 'exclude':
                continue
            dk = (r['dataset'], r['gt_rule'], r['backbone'], r['method'], r['fold'])
            if dk in seen:
                continue
            seen.add(dk)
            k = (r['dataset'], r['gt_rule'], r['backbone'], r['method'])
            for mk, (col, _) in METRICS.items():
                v = fl(r.get(col))
                if v is not None:
                    perfold[k][mk].append(v)
    return strength, perfold


def rule_values(perfold, d, b, method, mk):
    """one fold-averaged value per rule -> list over rules (the significance sample)."""
    out = []
    for (dd, rule, bb, m), md in perfold.items():
        if dd == d and bb == b and m == method and md.get(mk):
            out.append(st.mean(md[mk]))
    return out


def welch_p(x, y):
    from scipy import stats
    if len(x) < 2 or len(y) < 2:
        return 1.0
    try:
        return float(stats.ttest_ind(x, y, equal_var=False, nan_policy='omit').pvalue)
    except Exception:
        return 1.0


def holm(pairs, alpha=0.05):
    """pairs: {name: p}. returns set of names that are NOT significantly worse (fail to reject)."""
    items = sorted(pairs.items(), key=lambda kv: kv[1])
    m = len(items)
    notworse = set()
    rejected_all_below = True
    for i, (name, p) in enumerate(items):
        thresh = alpha / (m - i)
        if p < thresh and rejected_all_below:
            pass  # significantly worse than best
        else:
            rejected_all_below = False
            notworse.add(name)
    return notworse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True,
                    help='metrics rollup root, e.g. mose_replication_v2/rollups/planted '
                         '(reads <root>/<ds>/<rule>/all/metrics_*_unk-exclude.csv)')
    ap.add_argument('--vocab_root', default=None,
                    help='root holding <ds>/_shared/vocab/<ds>/rbrics/rule_tiers.json '
                         '(the planted_v2 source tree). Defaults to --root for the '
                         'legacy single-tree layout.')
    ap.add_argument('--out', default='paper_deliverables')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    strength, perfold = load(a.root, a.vocab_root or a.root)

    for mk, (col, higher) in METRICS.items():
        with_strength = (mk == 'spurroc')
        lines = []
        for d in DS:
            # per-dataset spurious strength (over 50 rules, backbone-independent)
            svals = [strength[(dd, rl)] for (dd, rl) in strength if dd == d]
            smean, sstd = st.mean(svals), st.pstdev(svals)
            for bi, b in enumerate(BB):
                # gather rule-value samples per method
                samp = {name: rule_values(perfold, d, b, method, mk) for name, method in COLS}
                stat = {name: (st.mean(v) * 100 if v else None,
                               st.pstdev(v) * 100 if len(v) > 1 else (0.0 if v else None))
                        for name, v in samp.items()}
                # best + significance
                valid = {n: s[0] for n, s in stat.items() if s[0] is not None}
                bold = None; under = set()
                if valid:
                    best = (max if higher else min)(valid, key=valid.get)
                    bold = best
                    pairs = {n: welch_p(samp[best], samp[n]) for n in valid if n != best}
                    under = holm(pairs)
                cells = []
                for name, _ in COLS:
                    m_, s_ = stat[name]
                    if m_ is None:
                        cells.append('--'); continue
                    txt = f'${m_:.1f}_{{{s_:.1f}}}$'
                    if name == bold:
                        txt = '{\\boldmath ' + txt + '}'
                    elif name in under:
                        txt = '\\underline{' + txt + '}'
                    cells.append(txt)
                # row prefix (multirow dataset [+ strength] on first backbone row)
                if bi == 0:
                    pre = ['\\multirow{%d}{*}{%s}' % (len(BB), DS_TEX[d])]
                    if with_strength:
                        pre.append('\\multirow{%d}{*}{$%.2f_{%.2f}$}' % (len(BB), smean, sstd))
                    pre.append(b)
                else:
                    pre = ['']
                    if with_strength:
                        pre.append('')
                    pre.append(b)
                lines.append(' & '.join(pre + cells) + ' \\\\')
            lines.append('\\midrule')
        if lines and lines[-1] == '\\midrule':
            lines[-1] = '\\bottomrule'

        ncols_lead = 3 if with_strength else 2
        colspec = 'l' + ('l' if with_strength else '') + 'l' + 'c' * len(COLS)
        head = (['Dataset'] + (['$\\bar r_{sp}$'] if with_strength else []) + ['BB']
                + [c for c, _ in COLS])
        direction = 'higher is better' if higher else 'lower is better'
        name = {'gtroc': 'GT-ROC (instance)', 'spurroc': 'Spurious-ROC',
                'famroc': 'Family-ROC'}[mk]
        tex = []
        tex.append('\\begin{table*}[t]\n\\centering\n\\small')
        tex.append(f'% {name}; cell = mean_{{std}} x100 (1 dp); unk=exclude; unit=rule (folds avg)')
        tex.append('\\begin{tabular}{' + colspec + '}')
        tex.append('\\toprule')
        tex.append(' & '.join(head) + ' \\\\')
        tex.append('\\midrule')
        tex += lines
        tex.append('\\end{tabular}')
        strength_note = (' The per-dataset column $\\bar r_{sp}$ gives the mean$_{std}$ '
                         'strength of the strongest planted confounder over the 50 rules '
                         '(the correlation a shortcut could exploit).' if with_strength else '')
        cap = (f'\\caption{{{name} between motif attribution and the planted cause, '
               f'{direction}, $\\times100$; subscript $=$ std over the 50 rules '
               f'(each rule fold-averaged), per (dataset, backbone). Evaluated on the '
               f'kept vocabulary (\\emph{{exclude-unk}}), so all methods share one motif '
               f'set. Per (dataset, backbone) the best is in \\textbf{{bold}} and methods '
               f'not significantly worse (Welch two-sided $t$-test over the 50 rule values, '
               f'Holm, $\\alpha{{=}}0.05$) are \\underline{{underlined}}.{strength_note} '
               f'A learnable-unknown MoSE arm (MoSE$_U$) and the GIN backbone were not '
               f'trained in this campaign; PGExplainer is incomplete '
               f'($952/3000$ valid cells) and its entries rest on the available valid rules.}}')
        tex.append(cap)
        tex.append('\\label{tab:planted_%s}' % mk)
        tex.append('\\end{table*}')
        out = os.path.join(a.out, f'paper_table_{mk}.tex')
        open(out, 'w').write('\n'.join(tex) + '\n')
        print('wrote', out)


if __name__ == '__main__':
    main()
