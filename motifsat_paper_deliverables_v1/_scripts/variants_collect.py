"""variants_collect.py — READ-ONLY collector for the MotifSAT variant campaign.

Run on the HPC with stdin (no file copy):  ssh osuhpc 'python3 -' < variants_collect.py > runs.jsonl
One JSON line per COMPLETE run under variants_v1/<vocab>/<preset>/<grouping>/<ds>/fold<f>/<bb>/
(complete = summary.json + summary_splits.json + occurrence_summary.json present). Test split only:
  pred        ROC-AUC (classification) or RMSE in original units (ESOL, Lipophilicity)
  gtroc       node GT-ROC per graph, mean (summary_splits test gt_roc_node_auc_mean; same as sweep)
  edge_auc    GSAT pooled edge explanation AUC (summary gsat_edge_auc_test)
  pm_all / pm_one   grouped Pearson, impact = mask ALL copies / mask ONE copy at a time
  pi_all / pi_one   instance Pearson, same two impact modes
  pi_all_multi / pi_one_multi   instance Pearson on (motif, molecule) pairs with >1 copy
  score_mean / score_std        motif score distribution (summary score_mean / score_std)
plus best_epoch, final_r, decay_interval, fix_r, motif_grouping for checking.
"""
import glob
import json
import math
import os

ROOT = '/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/motifsat_paper_deliverables_v1/variants_v1'


def num(x):
    return x if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)) else None


for occ in glob.glob(f'{ROOT}/*/*/*/*/fold*/*/occurrence_summary.json'):
    d = os.path.dirname(occ)
    vocab, preset, grouping, ds, fold, bb = d.split('/')[-6:]
    try:
        s = json.load(open(f'{d}/summary.json'))
        sp = json.load(open(f'{d}/summary_splits.json'))
    except (OSError, ValueError):
        continue
    t = next(iter(sp.values()))['test']
    auc = num(t.get('auc'))
    row = {
        'vocab': vocab, 'preset': preset, 'grouping': grouping, 'dataset': ds,
        'fold': int(fold[4:]), 'backbone': bb,
        'task': 'cls' if auc is not None else 'reg',
        'pred': auc if auc is not None else num(t.get('rmse_orig')),
        'gtroc': num(t.get('gt_roc_node_auc_mean')),
        'edge_auc': num(s.get('gsat_edge_auc_test')),
        'pm_all': num(s.get('occ_pearson_motif_maskall_test')),
        'pm_one': num(s.get('occ_pearson_motif_maskone_test')),
        'pi_all': num(s.get('occ_pearson_instance_maskall_test')),
        'pi_one': num(s.get('occ_pearson_instance_maskone_test')),
        'pi_all_multi': num(s.get('occ_pearson_instance_maskall_multi_test')),
        'pi_one_multi': num(s.get('occ_pearson_instance_maskone_multi_test')),
        'score_mean': num(s.get('score_mean')), 'score_std': num(s.get('score_std')),
        'best_epoch': s.get('best_epoch'), 'final_r': s.get('final_r'),
        'decay_interval': s.get('decay_interval'), 'fix_r': s.get('fix_r'),
        'motif_grouping': s.get('motif_grouping'),
        **{k: s.get(k) for k in ('motif_feat', 'motif_edge_feat', 'motif_edge_chem', 'motif_gnn_layers',
                                 'motif_gnn_norm', 'motif_gnn_residual')},
    }
    print(json.dumps(row))
