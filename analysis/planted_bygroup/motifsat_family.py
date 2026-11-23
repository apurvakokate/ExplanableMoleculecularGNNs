#!/usr/bin/env python3
"""motifsat_family.py DS RULE OUT — the MotifSAT planted runs of one rule, per split.

Companion of motifsat_bygroup.py (same files read, same per-molecule GT-ROC) with three changes:
every value is reported per split (train, valid, test) and for the three together; the extended
ground truth is reported over all rule-positive molecules; and SpurROC is added. Only the tree
runs_newr is read (the runs behind the dissertation's planted GT-ROC table), every arm and
grouping found. No model is loaded. The only file written is OUT ("-" prints instead).

Per run and per split S (and "all" = the molecules of the three splits together):
  n_S, gt_S       rule-positive molecules scored, and their mean planted GT-ROC: max over the
                  fired clauses of AUC(scores; clause atoms, atoms outside every fired clause)
                  — analysis/evaluate.py _dnf_gtroc. gt_all must equal the run's stored
                  instance_gt_roc_node_auc_mean_all (written as `stored`; checked on aggregation).
  ngtx_S, gtx_S   GT-ROC against the rule EXTENDED by OR with the class-1-only motifs (a motif id,
                  not a rule motif, found in rule-positive molecules and in no other molecule of
                  the dataset): each such motif in the molecule is one more clause and its atoms
                  leave the negatives. A molecule without such a motif keeps its rule value. A
                  molecule whose atoms ALL belong to the rule or to a class-1-only motif has no
                  negative atom left, so its extended value is undefined and it is left out
                  (ngtx_S counts the molecules that enter gtx_S; n_S - ngtx_S were left out).
  nwith_S         how many of the n_S molecules hold a class-1-only motif
  nspur_S, spur_S SpurROC as spur_msat_v1/spur.py computes it: per molecule that has atoms of the
                  correlated motif (node_label_spurious) and rule atoms (node_label),
                  AUC(scores; correlated-motif atoms, rule atoms); mean over those molecules.
"""
import glob, json, os, sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

B = '/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor'
P3 = B + '/motifsat_paper_deliverables_v1/planted_v3'
TREE = 'runs_newr'
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
    """Per split the graphs of the fold: (clauses, motif id per atom, label, atoms, spur) where
    spur = (correlated-motif atoms, rule atoms) or None when the graph carries no such labels."""
    out = {}
    for s in SPL:
        p = '%s/gt_cache/%s/fold%d/rbrics/relabel_%s/%s_with_gt.pt' % (P3, ds, fold, rule, s)
        gl = []
        for d in torch.load(p, weights_only=False):
            nlc = _np(d.node_label_clauses)
            clauses = [c for c in (np.where(nlc[:, j] > 0)[0].tolist()
                                   for j in range(nlc.shape[1])) if c]
            nl, sp = getattr(d, 'node_label', None), getattr(d, 'node_label_spurious', None)
            spur = None
            if nl is not None and sp is not None:
                nl = _np(nl).reshape(-1)
                sp = _np(sp)
                sp = sp.max(axis=1) if sp.ndim > 1 else sp.reshape(-1)
                spur = (np.flatnonzero(sp > 0).tolist(), np.flatnonzero(nl > 0).tolist())
            gl.append((clauses, _np(d.nodes_to_motifs).astype(int).reshape(-1),
                       int(_np(d.y).reshape(-1)[0]), nlc.shape[0], spur))
        out[s] = gl
    return out


def groups(fold_graphs):
    """Class-1-only motif ids of the dataset and, per graph, the atoms of each one it holds."""
    allg = [g for s in SPL for g in fold_graphs[s]]
    pos = [bool(g[0]) for g in allg]
    if [int(p) for p in pos] != [g[2] for g in allg]:
        raise ValueError('rule-positive molecules are not exactly the label-1 molecules')
    rule, in_pos, in_neg = set(), set(), set()
    for g, p in zip(allg, pos):
        for c in g[0]:
            rule.update(int(x) for x in g[1][c])
        (in_pos if p else in_neg).update(int(x) for x in g[1] if x >= 0)
    pure = in_pos - in_neg - rule
    atoms = {}
    for s in SPL:
        for i, g in enumerate(fold_graphs[s]):
            d = {}
            if g[0]:
                for k, x in enumerate(g[1]):
                    if int(x) in pure:
                        d.setdefault(int(x), []).append(k)
            atoms[(s, i)] = d
    return dict(n_pos=sum(pos), n_pure_motifs=len(pure), atoms=atoms)


def score_run(path, fold_graphs, info):
    ibs = json.load(open(path))['importances_by_split']
    row, tot = {}, dict(n=0, gt=0.0, ngtx=0, gtx=0.0, nwith=0, nspur=0, spur=0.0)
    for s in SPL:
        acc = dict(n=0, gt=0.0, ngtx=0, gtx=0.0, nwith=0, nspur=0, spur=0.0)
        st = ibs.get(s) or {}
        if st:
            if len(st) != 1:
                raise ValueError('%s: expected one score stem, found %s' % (path, sorted(st)))
            arr = st[next(iter(st))]
            for i, (clauses, n2m, _, n, spur) in enumerate(fold_graphs[s]):
                a = arr.get(str(i))
                if a is None:
                    continue
                att = np.asarray(a, float).reshape(-1)
                if len(att) != n:
                    raise ValueError('%s %s graph %d: %d scores for %d atoms' % (path, s, i, len(att), n))
                if spur is not None and spur[0] and spur[1]:
                    v = _auc(att, spur[0], spur[1])
                    if v == v:
                        acc['nspur'] += 1; acc['spur'] += v
                if not clauses:
                    continue
                U = set().union(*[set(c) for c in clauses])
                neg = [k for k in range(n) if k not in U]
                per = [v for v in (_auc(att, c, neg) for c in clauses) if v == v]
                if not per:
                    continue
                inst = max(per)
                x = inst
                pure_atoms = info['atoms'][(s, i)]
                if pure_atoms:
                    P = set().union(*[set(v) for v in pure_atoms.values()])
                    neg_x = [k for k in neg if k not in P]
                    per_x = [v for v in (_auc(att, c, neg_x)
                                         for c in clauses + list(pure_atoms.values())) if v == v]
                    x = max(per_x) if per_x else None          # None: no negative atom left
                    acc['nwith'] += 1
                acc['n'] += 1; acc['gt'] += inst
                if x is not None:
                    acc['ngtx'] += 1; acc['gtx'] += x
        for k in tot:
            tot[k] += acc[k]
        for tag, d in ((s, acc),):
            row.update(_cols(tag, d))
    row.update(_cols('all', tot))
    return row


def _cols(tag, d):
    return {'n_' + tag: d['n'], 'gt_' + tag: (d['gt'] / d['n'] if d['n'] else None),
            'ngtx_' + tag: d['ngtx'], 'gtx_' + tag: (d['gtx'] / d['ngtx'] if d['ngtx'] else None), 'nwith_' + tag: d['nwith'],
            'nspur_' + tag: d['nspur'], 'spur_' + tag: (d['spur'] / d['nspur'] if d['nspur'] else None)}


def main(ds, rule, out):
    fh = sys.stdout if out == '-' else open(out, 'w')
    folds, infos, n = {}, {}, 0
    pat = '%s/%s/%s/*/*/%s/fold*/*/explainer_importances.json' % (P3, TREE, rule, ds)
    for f in sorted(glob.glob(pat)):
        parts = f.split('/')
        preset, grouping, fold, bb = parts[-6], parts[-5], int(parts[-3][4:]), parts[-2]
        if fold not in folds:
            folds[fold] = load_fold(ds, rule, fold)
            infos[fold] = groups(folds[fold])
        info = infos[fold]
        row = dict(tree=TREE, dataset=ds, rule=rule, preset=preset, grouping=grouping, fold=fold,
                   backbone=bb, n_pos=info['n_pos'], n_pure_motifs=info['n_pure_motifs'])
        row.update(score_run(f, folds[fold], info))
        sp = os.path.join(os.path.dirname(f), 'summary.json')
        row['stored'] = json.load(open(sp)).get('instance_gt_roc_node_auc_mean_all') if os.path.exists(sp) else None
        fh.write(json.dumps(row) + '\n'); fh.flush()
        n += 1
    if out != '-':
        fh.close()
        open(out + '.done', 'w').write('%d\n' % n)
    print('done %s %s runs=%d' % (ds, rule, n), file=sys.stderr)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
