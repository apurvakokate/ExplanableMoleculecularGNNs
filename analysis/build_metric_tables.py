#!/usr/bin/env python3
"""Final metric tables: GT-ROC, Spurious-ROC, Family-ROC.

Row granularity: (dataset, backbone, rule). Aggregation: over FOLDS ONLY, reported
as mean +/- std. Columns: one per method. The Spurious-ROC table additionally
carries the spurious STRENGTH (the strongest non-rule confounder correlation r,
from the rule descriptor) so the ROC is read against how strong the shortcut was.

Methods:
  MoSE-fix/filt  = MoSE (unk-fixed training) scored in the filtered view (--unk exclude)
  MoSE-fix/full  = MoSE (unk-fixed training) scored in the full view    (--unk include)
    (a learnable-UNK MoSE arm was NOT trained in this campaign; it would be a
     separate run. The two MoSE columns are the two METRIC views, not train modes.)
  GSAT, GNNExplainer, MAGE, Motif-Occl., PGExplainer : filtered view (--unk exclude)

Outputs (per metric): a tidy CSV (all rows) and a LaTeX longtable.
PGExplainer is emitted but flagged incomplete (many cells missing/degenerate).

Usage: python build_metric_tables.py --root <planted_v2> --out <dir>
"""
import argparse, csv, glob, json, math, os, statistics as st
from collections import defaultdict

DS = ['BBBP', 'Mutagenicity', 'hERG']
BB = ['GAT', 'GCN', 'PNA', 'SAGE']
# (column key, method, metric-unk view)
COLS = [('MoSE-fix/filt', 'mose', 'exclude'),
        ('MoSE-fix/full', 'mose', 'include'),
        ('GSAT', 'gsat', 'exclude'),
        ('GNNExplainer', 'gnnexplainer', 'exclude'),
        ('MAGE', 'mage', 'exclude'),
        ('Motif-Occl.', 'motif_occlusion', 'exclude'),
        ('PGExplainer', 'pgexplainer', 'exclude')]
METRICS = {'gtroc': 'gtroc_instance', 'spurroc': 'spurious_roc', 'famroc': 'family_roc'}


def fl(x):
    try:
        v = float(x); return v if v == v else None
    except (TypeError, ValueError):
        return None


def rulekey(rule):
    return (int(rule.split('_k')[1].split('_')[0]), int(rule.split('_r')[1]))


def load(root):
    strength = {}
    for d in DS:
        t = json.load(open(f'{root}/{d}/_shared/vocab/{d}/rbrics/rule_tiers.json'))
        for rule, r in t.items():
            strength[(d, rule)] = r['correlate_r']
    # folds[(ds,rule,bb,method,unk,metric)] -> list over folds
    folds = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(f'{root}/*/*/eval/metrics_*.csv'):
        for r in csv.DictReader(open(f)):
            key = (r['dataset'], r['gt_rule'], r['backbone'], r['method'], r['unk'])
            for mk, col in METRICS.items():
                v = fl(r.get(col))
                if v is not None:
                    folds[key][mk].append(v)
    return strength, folds


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = st.mean(vals)
    s = st.pstdev(vals) if len(vals) > 1 else 0.0
    return (m, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', default='paper_deliverables')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    strength, folds = load(a.root)

    rules_by_ds = {}
    for d in DS:
        rs = sorted({rl for (dd, rl) in strength if dd == d}, key=rulekey)
        rules_by_ds[d] = rs

    for mk in METRICS:
        # ---- CSV ----
        cpath = os.path.join(a.out, f'table_{mk}.csv')
        with open(cpath, 'w', newline='') as fh:
            w = csv.writer(fh)
            hdr = ['dataset', 'backbone', 'rule']
            if mk == 'spurroc':
                hdr += ['spur_strength_r']
            for col, _, _ in COLS:
                hdr += [f'{col}_mean', f'{col}_std']
            w.writerow(hdr)
            for d in DS:
                for b in BB:
                    for rule in rules_by_ds[d]:
                        row = [d, b, rule]
                        if mk == 'spurroc':
                            row.append(f'{strength[(d, rule)]:.4f}')
                        for col, method, unk in COLS:
                            r = ms(folds[(d, rule, b, method, unk)][mk])
                            row += [f'{r[0]:.4f}', f'{r[1]:.4f}'] if r else ['', '']
                        w.writerow(row)
        # ---- LaTeX longtable ----
        lpath = os.path.join(a.out, f'table_{mk}.tex')
        lower = ' (lower is better)' if mk in ('spurroc', 'famroc') else ''
        with open(lpath, 'w') as fh:
            ncol = 3 + (1 if mk == 'spurroc' else 0) + len(COLS)
            fh.write('% ' + f'{mk.upper()} mean$\\pm$std over 5 folds, per (dataset,backbone,rule){lower}.\n')
            fh.write('% MoSE-fix = unk-fixed training; /filt = --unk exclude, /full = --unk include.\n')
            fh.write('% PGExplainer arm is INCOMPLETE (952/3000 valid); blanks = no valid cell.\n')
            fh.write('\\begin{longtable}{lll' + ('r' if mk == 'spurroc' else '') + 'r' * len(COLS) + '}\n')
            head = ['DS', 'BB', 'rule'] + (['$r_{\\mathrm{sp}}$'] if mk == 'spurroc' else []) \
                + [c.replace('_', '\\_') for c, _, _ in COLS]
            fh.write(' & '.join(head) + ' \\\\\n\\hline\n\\endhead\n')
            for d in DS:
                for b in BB:
                    for rule in rules_by_ds[d]:
                        cells = [d, b, rule.replace('_', '\\_')]
                        if mk == 'spurroc':
                            cells.append(f'{strength[(d, rule)]:.2f}')
                        for col, method, unk in COLS:
                            r = ms(folds[(d, rule, b, method, unk)][mk])
                            cells.append(f'{r[0]:.2f}\\,$\\pm$\\,{r[1]:.2f}' if r else '--')
                        fh.write(' & '.join(cells) + ' \\\\\n')
            fh.write('\\end{longtable}\n')
        nrow = sum(len(rules_by_ds[d]) for d in DS) * len(BB)
        print(f'{mk}: wrote {cpath} and {lpath}  ({nrow} rows)')


if __name__ == '__main__':
    main()
