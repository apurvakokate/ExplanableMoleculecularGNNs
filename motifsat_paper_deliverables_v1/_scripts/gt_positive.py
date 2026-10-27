"""gt_positive.py — READ-ONLY node GT-ROC on POSITIVE molecules only, for every run (written 2026-10-27; NOT YET
APPROVED — awaiting the user's sign-off before it is run).

Why: on Alkane- and Fluoride-Carbonyl (*_Verified_GT), ground-truth atoms are also marked on negative molecules that
contain one rule component (76% / 85% of the molecules the stored GT-ROC averages over). The positive-only view is the
one where the ground truth is the cause of the label. case_studies.py computed it for GSAT / consistency / readout
(layer norm) only; this does it for every setting, the coefficient sweep and the averaged-GSAT control.

Run on the HPC (l2xgnn env, CPU):  python3 gt_positive.py TASK_ID   (array 0-19; JSON lines on stdout)
TASK_ID -> (dataset, fold): Benzene, Alkane-Carbonyl, Fluoride-Carbonyl, mutag x 5 folds; rBRICS v3 only.
Test graphs exactly as gt_aupr.py / case_studies.py (same loaders, same processed roots). For every run directory
  variants_v1/rbrics_v3/<preset>/<grouping>/<ds>/fold<f>/<bb>      (our datasets; every preset incl. KL arms)
  paper_v1/rbrics_v3/<preset>/<grouping>/mutag/fold<f>/<bb>        (Mutag)
  lsweep_v1/rbrics_v3_<tag>/intra_inter_loss/pooled/<ds>/fold<f>/<bb>  (coefficient sweep)
with saved test gates (explainer_importances.json), one line:
  roc_all  mean node GT-ROC over test molecules with mixed ground truth (must equal the stored gt_roc_node_auc_mean;
           |diff| reported, > 2e-3 flagged misaligned)
  roc_pos  the same mean over POSITIVE molecules only (label 1; mutag: label 0 = mutagen, which is where its ground
           truth lives, so roc_pos = roc_all there)
  n_all, n_pos
  base_gsat only: roc_pos_type / roc_pos_copy = positives, gates averaged per motif type / occurrence (control).
Labels: the graph's own y (test loader), so no external label file is needed.
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

R = '/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor'
P = f'{R}/motifsat_paper_deliverables_v1'
GT = ['Benzene_Verified_GT', 'Alkane_Carbonyl_Verified_GT', 'Fluoride_Carbonyl_Verified_GT', 'mutag']
TASKS = [(ds, f) for ds in GT for f in range(5)]


def group_mean(s, keys):
    out = s.copy()
    for k in np.unique(keys):
        m = keys == k
        out[m] = s[m].mean()
    return out


def main(i):
    sys.path.insert(0, R); os.chdir(R)
    from SharedModules.data.loader import get_loaders
    from SharedModules.data.vocab import load_vocab
    ds, f = TASKS[i]
    droot = f'{R}/data' if ds == 'mutag' else '/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/'
    proc = f'{R}/processed_paper_v1/rbrics_v3' if ds == 'mutag' else f'{R}/processed_variants_v1/rbrics_v3'
    L, _, _ = get_loaders(dataset=ds, data_root=droot, fold=f, vocab=load_vocab(f'{R}/vocab_final_v3', ds, 'rbrics'),
                          processed_root=proc, batch_size=256)
    test = list(L['test'].dataset)
    pos_label = 0 if ds == 'mutag' else 1
    keep = []
    for gi, g in enumerate(test):
        y = getattr(g, 'node_label', None)
        if y is None:
            continue
        y = y.view(-1).numpy().astype(int)
        if y.sum() in (0, len(y)):
            continue
        keep.append((gi, y, int(float(g.y.view(-1)[0]) == pos_label),
                     g.nodes_to_motifs.view(-1).numpy(), g.nodes_to_copy.view(-1).numpy()))
    tree = 'paper_v1' if ds == 'mutag' else 'variants_v1'
    dirs = sorted(glob.glob(f'{P}/{tree}/rbrics_v3/*/*/{ds}/fold{f}/*')) + \
        sorted(glob.glob(f'{P}/lsweep_v1/rbrics_v3_*/intra_inter_loss/pooled/{ds}/fold{f}/*'))
    for d in dirs:
        vocab, preset, grouping = d.split('/')[-6], d.split('/')[-5], d.split('/')[-4]
        out = dict(tree=d.split('/')[-7], vocab=vocab, preset=preset, grouping=grouping, dataset=ds, fold=f,
                   backbone=d.split('/')[-1])
        try:
            imp = json.load(open(f'{d}/explainer_importances.json'))['importances_by_split']['test']
            (key,) = imp.keys(); imp = imp[key]
            ref = next(iter(json.load(open(f'{d}/summary_splits.json')).values()))['test']['gt_roc_node_auc_mean']
        except Exception as e:                                   # incomplete run: report, never hide
            print(json.dumps(dict(out, error=repr(e))), flush=True); continue
        assert len(imp) == len(test), (d, len(imp), len(test))
        acc = {k: [] for k in ('all', 'pos', 'pos_type', 'pos_copy')}
        for gi, y, is_pos, n2m, n2c in keep:
            s = np.asarray(imp[str(gi)], float)
            assert len(s) == len(y), (d, gi)
            r = roc_auc_score(y, s)
            acc['all'].append(r)
            if is_pos:
                acc['pos'].append(r)
                if preset == 'base_gsat':
                    acc['pos_type'].append(roc_auc_score(y, group_mean(s, n2m)))
                    acc['pos_copy'].append(roc_auc_score(y, group_mean(s, n2m * 100000 + n2c)))
        res = {f'roc_{k}': float(np.mean(v)) for k, v in acc.items() if v}
        res.update(n_all=len(acc['all']), n_pos=len(acc['pos']), ref_roc=ref)
        res['roc_diff'] = abs(res['roc_all'] - ref) if ref is not None and 'roc_all' in res else None
        res['misaligned'] = not (res['roc_diff'] is not None and res['roc_diff'] <= 2e-3)
        print(json.dumps(dict(out, **res)), flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
