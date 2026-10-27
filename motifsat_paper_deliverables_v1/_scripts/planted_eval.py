"""planted_eval.py — READ-ONLY evaluation of the planted-GT campaign (P3, approved 2026-10-26).

Run on the HPC (l2xgnn env, CPU):  python3 planted_eval.py TASK_ID   (array 0-30 = one (dataset, rule);
one JSON line per run on stdout). Optional 2nd arg: tree name under planted_v3/ (default runs).
Per run dir planted_v3/runs/<rule>/<preset>/<grouping>/<ds>/fold<f>/<bb>:
  collector fields (as variants_collect.py): pred (test ROC-AUC), gtroc (stored gt_roc_node_auc_mean = fired-clause
  GT-ROC), pi_all / pm_all (instance / grouped Pearson, mask-all), score_mean / score_std, best_epoch, final_r, ...
  from the saved test gates (explainer_importances.json, test, only key) and the run's own label cache
  planted_v3/gt_cache/<ds>/fold<f>/rbrics/relabel_<rule>/test_with_gt.pt (same order as the run's test loader):
    roc / ap / prev / nap : per graph with ground truth that is neither all-0 nor all-1 (= molecules on which the
                            rule fired), ROC-AUC, average precision, GT prevalence, (ap - prev) / (1 - prev); means.
    ALIGNMENT: roc must equal the stored gtroc (|diff| reported; > 2e-3 flagged misaligned).
    base_gsat only: roc/ap with gates averaged per motif type and per occurrence (granularity control).
    shortcut_roc__<motif> : per spurious correlate j (node_label_spurious[:, j], rule record order): on graphs where
                            the rule fired and motif j is present outside the cause, ROC of the gates with positives =
                            motif-j atoms, negatives = cause atoms (overlap excluded). > 0.5 = explainer ranks the
                            shortcut above the cause. Mean over graphs, with the graph count.
"""
import glob
import json
import math
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

R = '/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor'
P3 = f'{R}/motifsat_paper_deliverables_v1/planted_v3'
RULES = {
    'BBBP': 'dnf_k1_r1 dnf_k1_r2 dnf_k3_r8 dnf_k2_r6 dnf_k2_r7 dnf_k2_r10 dnf_k3_r1 dnf_k3_r2 dnf_k3_r3 dnf_k3_r4',
    'hERG': 'dnf_k1_r1 dnf_k1_r2 dnf_k2_r2 dnf_k2_r6 dnf_k3_r12 dnf_k2_r9 dnf_k2_r1 dnf_k3_r1 dnf_k3_r2 dnf_k3_r3',
    'Mutagenicity': 'dnf_k1_r1 dnf_k1_r2 dnf_k1_r3 dnf_k3_r1 dnf_k3_r2 dnf_k3_r3 dnf_k3_r4 dnf_k3_r6 dnf_k3_r7 '
                    'dnf_k3_r8 dnf_k2_r4',
}
TASKS = [(ds, r) for ds in ('BBBP', 'hERG', 'Mutagenicity') for r in RULES[ds].split()]


def num(x):
    return x if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)) else None


def collect(d):
    s = json.load(open(f'{d}/summary.json'))
    t = next(iter(json.load(open(f'{d}/summary_splits.json')).values()))['test']
    return dict(pred=num(t.get('auc')), gtroc=num(t.get('gt_roc_node_auc_mean')),
                gtroc_instance=num(t.get('instance_gt_roc_node_auc_mean')),
                pm_all=num(s.get('occ_pearson_motif_maskall_test')), pi_all=num(s.get('occ_pearson_instance_maskall_test')),
                score_mean=num(s.get('score_mean')), score_std=num(s.get('score_std')),
                best_epoch=s.get('best_epoch'), final_r=s.get('final_r'), decay_interval=s.get('decay_interval'),
                use_gt=s.get('use_gt'), gt_tier=s.get('gt_tier'))


def pooled(s, keys):
    out = s.copy()
    for k in np.unique(keys):
        m = keys == k
        out[m] = s[m].mean()
    return out


def main(i, tree='runs'):
    import torch
    ds, rule = TASKS[i]
    caches = {}
    for d in sorted(glob.glob(f'{P3}/{tree}/{rule}/*/*/{ds}/fold*/*')):
        preset, grouping, fold, bb = d.split('/')[-5], d.split('/')[-4], int(d.split('/')[-2][4:]), d.split('/')[-1]
        out = dict(dataset=ds, rule=rule, preset=preset, grouping=grouping, fold=fold, backbone=bb)
        try:
            out.update(collect(d))
            imp = json.load(open(f'{d}/explainer_importances.json'))['importances_by_split']['test']
            (key,) = imp.keys(); imp = imp[key]
        except Exception as e:                                   # incomplete run: report, never hide
            print(json.dumps(dict(out, error=repr(e))), flush=True); continue
        if fold not in caches:
            caches[fold] = torch.load(f'{P3}/gt_cache/{ds}/fold{fold}/rbrics/relabel_{rule}/test_with_gt.pt',
                                      weights_only=False)
        test = caches[fold]
        assert len(imp) == len(test), (d, len(imp), len(test))
        acc = {k: [] for k in ('roc', 'ap', 'prev', 'nap', 'roc_type', 'ap_type', 'roc_copy', 'ap_copy')}
        spur_keys = None; spur = {}
        for gi, g in enumerate(test):
            y = g.node_label.view(-1).numpy().astype(int)
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
                    sp = pooled(s, keys)
                    acc[f'roc_{tag}'].append(roc_auc_score(y, sp)); acc[f'ap_{tag}'].append(average_precision_score(y, sp))
            ns = getattr(g, 'node_label_spurious', None)
            if ns is not None and ns.numel():
                ns = ns.view(len(y), -1).numpy().astype(int)
                if spur_keys is None:
                    spur_keys = str(getattr(g, 'spurious_motif_keys', '')).split('\t')
                for j in range(ns.shape[1]):
                    pos = (ns[:, j] == 1) & (y == 0); neg = (y == 1)
                    if pos.any() and neg.any():
                        lab = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
                        spur.setdefault(j, []).append(roc_auc_score(lab, np.r_[s[pos], s[neg]]))
        res = {k: float(np.mean(v)) for k, v in acc.items() if v}
        res['n_graphs'] = len(acc['roc'])
        res['roc_diff'] = (abs(res['roc'] - out['gtroc']) if 'roc' in res and out.get('gtroc') is not None else None)
        res['misaligned'] = not (res['roc_diff'] is not None and res['roc_diff'] <= 2e-3)
        for j, v in spur.items():
            name = spur_keys[j] if spur_keys and j < len(spur_keys) else f'col{j}'
            res[f'shortcut_roc__{name}'] = float(np.mean(v)); res[f'shortcut_n__{name}'] = len(v)
        print(json.dumps(dict(out, **res)), flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else 'runs')
