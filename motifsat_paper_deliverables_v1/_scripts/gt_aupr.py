"""gt_aupr.py — READ-ONLY node GT-AUPR from saved test gates (S3, approved 2026-10-26).

Run on the HPC (l2xgnn env, CPU):  python3 gt_aupr.py TASK_ID   (array 0-39; one JSON line per run on stdout)
TASK_ID -> (vocab, dataset, fold): 3 Verified_GT datasets x 5 folds x 2 vocabs (variants_v1) + mutag x 5 folds
x 2 vocabs (paper_v1). Same procedure as the granularity audit (job 21940128):
  gates  = explainer_importances.json['importances_by_split']['test'][<only key>] (per test graph, per atom;
           motif gates already broadcast to atoms)
  labels = node_label of the same test graph from get_loaders (same vocab / fold / processed root as the run)
Per graph with GT that is neither all-0 nor all-1: ROC-AUC, average precision (AUPR), prevalence (= the AUPR
of a random scorer) and normalised AUPR (AUPR - prev) / (1 - prev); each averaged over graphs.
ALIGNMENT CHECK: the recomputed ROC must equal the stored gt_roc_node_auc_mean (|diff| reported; a run with
|diff| > 1e-6 is flagged 'misaligned' and must not be used). For base_gsat the motif-averaged control
(gates averaged per motif type and per occurrence) is reported too.
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

R = '/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor'
P = f'{R}/motifsat_paper_deliverables_v1'
GT = ['Benzene_Verified_GT', 'Alkane_Carbonyl_Verified_GT', 'Fluoride_Carbonyl_Verified_GT']
VOCABS = ['rbrics_v3', 'fmbpe_v4']
TASKS = [(v, ds, f) for ds in GT for f in range(5) for v in VOCABS] + [(v, 'mutag', f) for f in range(5) for v in VOCABS]


def main(i):
    sys.path.insert(0, R); os.chdir(R)
    from SharedModules.data.loader import get_loaders
    from SharedModules.data.vocab import load_vocab
    vocab, ds, f = TASKS[i]
    tree, rep = ('paper_v1', f'fold{f}') if ds == 'mutag' else ('variants_v1', f'fold{f}')
    vroot, var = (f'{R}/vocab_final_v3', 'rbrics') if vocab == 'rbrics_v3' else (f'{R}/fmbpe_v4/vocab/fold{f}',
                                                                                    'fm_bpe_full_union')
    droot = f'{R}/data' if ds == 'mutag' else '/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/'
    proc = f'{R}/processed_paper_v1/{vocab}' if ds == 'mutag' else f'{R}/processed_variants_v1/{vocab}'
    L, _, _ = get_loaders(dataset=ds, data_root=droot, fold=f, vocab=load_vocab(vroot, ds, var),
                          processed_root=proc, batch_size=256)
    test = list(L['test'].dataset)
    for d in sorted(glob.glob(f'{P}/{tree}/{vocab}/*/*/{ds}/{rep}/*')):
        preset, grouping, bb = d.split('/')[-5], d.split('/')[-4], d.split('/')[-1]
        out = dict(vocab=vocab, preset=preset, grouping=grouping, dataset=ds, fold=f, backbone=bb)
        try:
            imp = json.load(open(f'{d}/explainer_importances.json'))['importances_by_split']['test']
            (key,) = imp.keys(); imp = imp[key]
            ref = next(iter(json.load(open(f'{d}/summary_splits.json')).values()))['test']['gt_roc_node_auc_mean']
        except Exception as e:                                   # incomplete run: report, never hide
            print(json.dumps(dict(out, error=repr(e))), flush=True); continue
        acc = {k: [] for k in ('roc', 'ap', 'prev', 'nap', 'ap_type', 'ap_copy', 'roc_type', 'roc_copy')}
        for gi, g in enumerate(test):
            y = getattr(g, 'node_label', None)
            if y is None:
                continue
            y = y.view(-1).numpy().astype(int)
            if y.sum() in (0, len(y)):
                continue
            s = np.asarray(imp[str(gi)], float)
            assert len(s) == len(y), (d, gi, len(s), len(y))
            prev = y.mean(); ap = average_precision_score(y, s)
            acc['roc'].append(roc_auc_score(y, s)); acc['ap'].append(ap); acc['prev'].append(prev)
            acc['nap'].append((ap - prev) / (1 - prev))
            if preset == 'base_gsat':
                n2m = g.nodes_to_motifs.view(-1).numpy(); n2c = g.nodes_to_copy.view(-1).numpy()
                for tag, keys in (('type', n2m), ('copy', n2m * 100000 + n2c)):
                    sp = s.copy()
                    for k in np.unique(keys):
                        m = keys == k; sp[m] = s[m].mean()
                    acc[f'ap_{tag}'].append(average_precision_score(y, sp))
                    acc[f'roc_{tag}'].append(roc_auc_score(y, sp))
        res = {k: float(np.mean(v)) for k, v in acc.items() if v}
        res['n_graphs'] = len(acc['roc']); res['ref_roc'] = ref
        res['roc_diff'] = abs(res.get('roc', np.nan) - ref) if ref is not None else None
        res['misaligned'] = not (res['roc_diff'] is not None and res['roc_diff'] <= 1e-6)
        print(json.dumps(dict(out, **res)), flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
