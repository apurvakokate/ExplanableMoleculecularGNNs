#!/usr/bin/env python3
"""motifsat_bygroup.py DS RULE OUT — planted GT-ROC of the MotifSAT planted runs, split by
whether a rule-positive molecule also holds a class-1-only motif outside the rule.

The MotifSAT counterpart of analysis/evaluate_gtroc_bygroup.py, for the tree
motifsat_paper_deliverables_v1/planted_v3 (trees runs, runs_eps0, runs_newr; every arm and
grouping found). Like spur_msat_v1/spur.py it loads no model: it reads each run's saved per-atom
scores (explainer_importances.json, train + valid + test) and the ground-truth graphs of the
rule (gt_cache/.../<split>_with_gt.pt). The only file written is OUT ("-" prints instead).

Per run (tree, arm, grouping, fold, backbone), over all molecules of the fold:
  gt_all                  mean per-molecule planted GT-ROC: max over the fired clauses of
                          AUC(scores; clause atoms, atoms outside every fired clause) — the same
                          computation as analysis/evaluate.py _dnf_gtroc, no atom dropped
  stored                  instance_gt_roc_node_auc_mean_all from the run's summary.json; gt_all
                          must equal it (checked when the results are aggregated)
  gt_with / gt_without    gt_all restricted to rule-positive molecules that do / do not hold a
                          pure motif (a motif id, not a rule motif, found in rule-positive
                          molecules and in no other molecule of the dataset)
  gtx_with                the WITH molecules scored against the rule extended by OR with the
                          pure motifs (each one more clause; its atoms leave the negatives)
  pvr_with                AUC of the pure-motif atoms against the rule atoms; below 0.5 the rule
                          outranks them
"""
import glob, json, os, sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

B = '/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor'
P3 = B + '/motifsat_paper_deliverables_v1/planted_v3'
TREES = ('runs', 'runs_eps0', 'runs_newr')
SPL = ('train', 'valid', 'test')


def _auc(scores, pos, neg):                       # analysis/evaluate.py _auc
    if not pos or not neg:
        return float('nan')
    y = [1] * len(pos) + [0] * len(neg)
    s = [float(scores[i]) for i in pos] + [float(scores[i]) for i in neg]
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float('nan')


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def load_fold(ds, rule, fold):
    """Per split the graphs of the fold, each reduced to (clauses, motif id per atom, label)."""
    out = {}
    for s in SPL:
        p = '%s/gt_cache/%s/fold%d/rbrics/relabel_%s/%s_with_gt.pt' % (P3, ds, fold, rule, s)
        gl = []
        for d in torch.load(p, weights_only=False):
            nlc = _np(d.node_label_clauses)
            clauses = [c for c in (np.where(nlc[:, j] > 0)[0].tolist()
                                   for j in range(nlc.shape[1])) if c]
            gl.append((clauses, _np(d.nodes_to_motifs).astype(int).reshape(-1),
                       int(_np(d.y).reshape(-1)[0]), nlc.shape[0]))
        out[s] = gl
    return out


def groups(fold_graphs):
    """Pure motif ids of the dataset and, per graph, the atoms of each pure motif it holds."""
    allg = [g for s in SPL for g in fold_graphs[s]]
    pos = [bool(g[0]) for g in allg]
    if [int(p) for p in pos] != [g[2] for g in allg]:
        raise ValueError('rule-positive molecules are not exactly the label-1 molecules')
    rule, in_pos, in_neg = set(), set(), set()
    for (clauses, n2m, _, _), p in zip(allg, pos):
        for c in clauses:
            rule.update(int(x) for x in n2m[c])
        (in_pos if p else in_neg).update(int(x) for x in n2m if x >= 0)
    pure = in_pos - in_neg - rule
    atoms = {}
    for s in SPL:
        for i, (clauses, n2m, _, _) in enumerate(fold_graphs[s]):
            d = {}
            if clauses:
                for k, x in enumerate(n2m):
                    if int(x) in pure:
                        d.setdefault(int(x), []).append(k)
            atoms[(s, i)] = d
    return dict(n_pos=sum(pos), n_pos_with=sum(1 for d in atoms.values() if d),
                n_pure_motifs=len(pure), atoms=atoms)


def score_run(path, fold_graphs, info):
    ibs = json.load(open(path))['importances_by_split']
    inst, grp, ext, pvr = [], [], [], []
    for s in SPL:
        st = ibs.get(s) or {}
        if not st:
            continue
        if len(st) != 1:
            raise ValueError('%s: expected one score stem, found %s' % (path, sorted(st)))
        arr = st[next(iter(st))]
        for i, (clauses, n2m, _, n) in enumerate(fold_graphs[s]):
            a = arr.get(str(i))
            if a is None or not clauses:
                continue
            att = np.asarray(a, float).reshape(-1)
            if len(att) != n:
                raise ValueError('%s %s graph %d: %d scores for %d atoms' % (path, s, i, len(att), n))
            U = set().union(*[set(c) for c in clauses])
            neg = [k for k in range(n) if k not in U]
            per = [v for v in (_auc(att, c, neg) for c in clauses) if v == v]
            if not per:
                continue
            inst.append(max(per))
            pure_atoms = info['atoms'][(s, i)]
            grp.append(bool(pure_atoms))
            if not pure_atoms:
                ext.append(None); pvr.append(None)
                continue
            P = set().union(*[set(v) for v in pure_atoms.values()])
            neg_x = [k for k in neg if k not in P]
            per_x = [v for v in (_auc(att, c, neg_x)
                                 for c in clauses + list(pure_atoms.values())) if v == v]
            ext.append(max(per_x) if per_x else None)
            v = _auc(att, sorted(P), sorted(U))
            pvr.append(v if v == v else None)
    m = lambda v: float(np.mean(v)) if len(v) else None
    w = [x for x, g in zip(inst, grp) if g]
    wo = [x for x, g in zip(inst, grp) if not g]
    return dict(gt_all=m(inst), n_all=len(inst), gt_with=m(w), n_with=len(w),
                gt_without=m(wo), n_without=len(wo),
                gtx_with=m([x for x in ext if x is not None]),
                pvr_with=m([x for x in pvr if x is not None]),
                n_pvr=sum(x is not None for x in pvr))


def main(ds, rule, out):
    fh = sys.stdout if out == '-' else open(out, 'w')
    folds, infos = {}, {}
    n = 0
    for tree in TREES:
        pat = '%s/%s/%s/*/*/%s/fold*/*/explainer_importances.json' % (P3, tree, rule, ds)
        for f in sorted(glob.glob(pat)):
            parts = f.split('/')
            preset, grouping, fold, bb = parts[-6], parts[-5], int(parts[-3][4:]), parts[-2]
            if fold not in folds:
                folds[fold] = load_fold(ds, rule, fold)
                infos[fold] = groups(folds[fold])
            info = infos[fold]
            row = dict(tree=tree, dataset=ds, rule=rule, preset=preset, grouping=grouping,
                       fold=fold, backbone=bb, n_pos=info['n_pos'],
                       n_pos_with=info['n_pos_with'], n_pure_motifs=info['n_pure_motifs'])
            row.update(score_run(f, folds[fold], info))
            sp = os.path.join(os.path.dirname(f), 'summary.json')
            stored = json.load(open(sp)).get('instance_gt_roc_node_auc_mean_all') \
                if os.path.exists(sp) else None
            row['stored'] = stored
            fh.write(json.dumps(row) + '\n'); fh.flush()
            n += 1
    if out != '-':
        fh.close()
        open(out + '.done', 'w').write('%d\n' % n)
    print('done %s %s runs=%d' % (ds, rule, n), file=sys.stderr)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
