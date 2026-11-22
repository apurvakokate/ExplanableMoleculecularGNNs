#!/usr/bin/env python3
"""random_baseline.py TREE DS RULE OUT — what a RANDOM explainer scores on the planted measures.

The planted GT-ROC of a molecule is a maximum over its fired clauses, so its chance level is
not 0.5: it depends on how many clauses enter the maximum and on their sizes, and it rises when
the rule is extended with more clauses. This script scores random explanations exactly as
analysis/evaluate_gtroc_bygroup.py and motifsat_bygroup.py score real ones, on the same
ground-truth graphs. No model, no run directory: only gt_cache is read. The only file written
is OUT ("-" prints instead).

TREE  v2 = planted_v2 (MoSE-GNN and baselines; conditions full and filt, filt drops atoms whose
           motif is outside the fold's kept list, taken from analysis.evaluate_gtroc)
      v3 = motifsat_paper_deliverables_v1/planted_v3 (MotifSAT; condition full only)

Two random explainers, DRAWS draws each, seeded by (dataset, rule, fold):
  atom   an independent uniform score per atom
  motif  an independent uniform score per motif OCCURRENCE (a connected set of atoms with the
         same motif id), shared by its atoms

One JSON line per (fold, level, cond) with the mean over draws of
  gt_all, gt_with, gt_without   mean per-molecule planted GT-ROC (max over fired clauses)
  gtx_with, gtx_all             the same against the rule extended by OR with the class-1-only
                                motifs (gtx_all: every rule-positive molecule)
  pvr_with                      AUC of class-1-only motif atoms against rule atoms
and the counts n_pos, n_pos_with, n_pure_motifs, n_multi (rule-positive molecules in which two
or more clauses fire).
"""
import json, os, sys, zlib
import numpy as np
import torch
from scipy.stats import rankdata

B = '/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor'
DATA = '/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS'
SPL = ('train', 'valid', 'test')
DRAWS = 20


def gt_path(tree, ds, rule, fold, split):
    root = ('%s/planted_v2/%s/%s' % (B, ds, rule)) if tree == 'v2' else \
        (B + '/motifsat_paper_deliverables_v1/planted_v3')
    return '%s/gt_cache/%s/fold%d/rbrics/relabel_%s/%s_with_gt.pt' % (root, ds, fold, rule, split)


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def occurrences(n2m, edge_index):
    """Occurrence id per atom: connected atoms that share a motif id."""
    n = len(n2m)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in zip(edge_index[0], edge_index[1]):
        a, b = int(a), int(b)
        if n2m[a] == n2m[b]:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    roots = [find(i) for i in range(n)]
    ids = {r: j for j, r in enumerate(sorted(set(roots)))}
    return np.array([ids[r] for r in roots])


def auc(s, pos, neg):
    """AUC with average ranks for ties (equals sklearn.metrics.roc_auc_score)."""
    if not len(pos) or not len(neg):
        return None
    r = rankdata(np.concatenate([s[pos], s[neg]]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def load_fold(tree, ds, rule, fold):
    gl = []
    for s in SPL:
        for d in torch.load(gt_path(tree, ds, rule, fold, s), weights_only=False):
            nlc = _np(d.node_label_clauses)
            clauses = [c for c in (np.where(nlc[:, j] > 0)[0] for j in range(nlc.shape[1])) if len(c)]
            n2m = _np(d.nodes_to_motifs).astype(int).reshape(-1)
            gl.append(dict(clauses=clauses, n2m=n2m, y=int(_np(d.y).reshape(-1)[0]), n=nlc.shape[0],
                           occ=occurrences(n2m, _np(d.edge_index)) if clauses else None))
    return gl


def groups(gl):
    pos = [bool(g['clauses']) for g in gl]
    if [int(p) for p in pos] != [g['y'] for g in gl]:
        raise ValueError('rule-positive molecules are not exactly the label-1 molecules')
    rule, in_pos, in_neg = set(), set(), set()
    for g, p in zip(gl, pos):
        for c in g['clauses']:
            rule.update(int(x) for x in g['n2m'][c])
        (in_pos if p else in_neg).update(int(x) for x in g['n2m'] if x >= 0)
    pure = in_pos - in_neg - rule
    for g in gl:
        d = {}
        if g['clauses']:
            for k, x in enumerate(g['n2m']):
                if int(x) in pure:
                    d.setdefault(int(x), []).append(k)
        g['pure'] = [np.array(v) for v in d.values()]
    return dict(n_pos=sum(pos), n_pos_with=sum(1 for g in gl if g['pure']), n_pure_motifs=len(pure),
                n_multi=sum(1 for g in gl if len(g['clauses']) > 1))


def score(gl, kept, level, rng):
    """One random draw: (gt_all, gt_with, gt_without, gtx_with, gtx_all, pvr_with)."""
    inst, grp, ext, pvr = [], [], [], []
    for g in gl:
        cl = g['clauses']
        if not cl:
            continue
        n = g['n']
        s = rng.random(n) if level == 'atom' else rng.random(int(g['occ'].max()) + 1)[g['occ']]
        keep = np.ones(n, bool) if kept is None else np.array([int(m) in kept for m in g['n2m']])
        inU = np.zeros(n, bool)
        for c in cl:
            inU[c] = True
        neg = np.where(~inU & keep)[0]
        per = [v for v in (auc(s, c[keep[c]], neg) for c in cl) if v is not None]
        if not per:
            continue
        inst.append(max(per)); grp.append(bool(g['pure']))
        if not g['pure']:
            ext.append(max(per)); pvr.append(None)
            continue
        inP = np.zeros(n, bool)
        for c in g['pure']:
            inP[c] = True
        neg_x = np.where(~inU & ~inP & keep)[0]
        per_x = [v for v in (auc(s, c[keep[c]], neg_x) for c in cl + g['pure']) if v is not None]
        ext.append(max(per_x) if per_x else None)
        pvr.append(auc(s, np.where(inP & keep)[0], np.where(inU & keep)[0]))
    m = lambda v: float(np.mean(v)) if len(v) else None
    w = [x for x, q in zip(inst, grp) if q]
    wo = [x for x, q in zip(inst, grp) if not q]
    return (m(inst), m(w), m(wo), m([x for x, q in zip(ext, grp) if q and x is not None]),
            m([x for x in ext if x is not None]), m([x for x in pvr if x is not None]))


def main(tree, ds, rule, out, folds=range(5), draws=DRAWS):
    fh = sys.stdout if out == '-' else open(out, 'w')
    n = 0
    for fold in folds:
        gl = load_fold(tree, ds, rule, fold)
        info = groups(gl)
        conds = [('full', None)]
        if tree == 'v2':
            from analysis.evaluate_gtroc import _kept_for_fold
            kept = _kept_for_fold(ds, fold, '%s/planted_v2/%s/_shared/vocab' % (B, ds), DATA, 'rbrics')
            conds.append(('filt', kept))
        for level in ('atom', 'motif'):
            for cond, kept in conds:
                rng = np.random.default_rng(zlib.crc32(('%s|%s|%d|%s' % (ds, rule, fold, level)).encode()))
                acc = [score(gl, kept, level, rng) for _ in range(draws)]
                cols = ('gt_all', 'gt_with', 'gt_without', 'gtx_with', 'gtx_all', 'pvr_with')
                row = dict(tree=tree, dataset=ds, rule=rule, fold=fold, level=level, cond=cond, draws=draws, **info)
                for j, c in enumerate(cols):
                    v = [a[j] for a in acc if a[j] is not None]
                    row[c] = float(np.mean(v)) if v else None
                fh.write(json.dumps(row) + '\n'); fh.flush()
                n += 1
    if out != '-':
        fh.close()
        open(out + '.done', 'w').write('%d\n' % n)
    print('done %s %s %s rows=%d' % (tree, ds, rule, n), file=sys.stderr)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
