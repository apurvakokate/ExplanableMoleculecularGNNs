"""variants_collect.py — READ-ONLY collector for the MotifSAT variant campaign.

Run on the HPC with stdin (no file copy):  ssh osuhpc 'python3 - [TREE] [SPLIT]' < variants_collect.py > runs.jsonl
SPLIT = test (default) or valid / train: every metric below is read from that split instead (model selection uses valid).
TREE = variants_v1 (default) or paper_v1 (GSAT paper datasets; replicate dirs fold<k> for mutag,
seed<k> for OGB — both reported as fold=k, with rep='fold'|'seed').
One JSON line per COMPLETE run under <TREE>/<vocab>/<preset>/<grouping>/<ds>/<rep><k>/<bb>/
(complete = summary.json + summary_splits.json + occurrence_summary.json present). Test split only:
  pred        ROC-AUC (classification; MultiLabel = mean per-task ROC-AUC, auc_per_task kept) or
              RMSE in original units (ESOL, Lipophilicity)
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
import sys

TREE = sys.argv[1] if len(sys.argv) > 1 else 'variants_v1'
SPLIT = sys.argv[2] if len(sys.argv) > 2 else 'test'          # test (default) | valid | train
ROOT = f'/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/motifsat_paper_deliverables_v1/{TREE}'


def num(x):
    return x if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)) else None


for occ in glob.glob(f'{ROOT}/*/*/*/*/fold*/*/occurrence_summary.json') + \
        glob.glob(f'{ROOT}/*/*/*/*/seed*/*/occurrence_summary.json'):
    d = os.path.dirname(occ)
    vocab, preset, grouping, ds, fold, bb = d.split('/')[-6:]
    try:
        s = json.load(open(f'{d}/summary.json'))
        sp = json.load(open(f'{d}/summary_splits.json'))
    except (OSError, ValueError):
        continue
    t = next(iter(sp.values()))[SPLIT]
    auc = num(t.get('auc'))
    if auc is None:
        auc = num(t.get('auc_mean'))                       # MultiLabel (Tox21, ClinTox)
    row = {
        'vocab': vocab, 'preset': preset, 'grouping': grouping, 'dataset': ds,
        'fold': int(fold[4:]), 'rep': fold[:4], 'backbone': bb,
        'auc_per_task': t.get('auc_per_task'),
        'task': 'cls' if auc is not None else 'reg',
        'pred': auc if auc is not None else num(t.get('rmse_orig')),
        'gtroc': num(t.get('gt_roc_node_auc_mean')),
        'edge_auc': num(s.get('gsat_edge_auc_' + SPLIT)),
        'pm_all': num(s.get('occ_pearson_motif_maskall_' + SPLIT)),
        'pm_one': num(s.get('occ_pearson_motif_maskone_' + SPLIT)),
        'pi_all': num(s.get('occ_pearson_instance_maskall_' + SPLIT)),
        'pi_one': num(s.get('occ_pearson_instance_maskone_' + SPLIT)),
        'pi_all_multi': num(s.get('occ_pearson_instance_maskall_multi_' + SPLIT)),
        'pi_one_multi': num(s.get('occ_pearson_instance_maskone_multi_' + SPLIT)),
        'score_mean': num(s.get('score_mean')), 'score_std': num(s.get('score_std')),
        'best_epoch': s.get('best_epoch'), 'final_r': s.get('final_r'),
        'decay_interval': s.get('decay_interval'), 'fix_r': s.get('fix_r'),
        'motif_grouping': s.get('motif_grouping'),
        **{k: s.get(k) for k in ('motif_feat', 'motif_edge_feat', 'motif_edge_chem', 'motif_gnn_layers',
                                 'motif_gnn_norm', 'motif_gnn_residual')},
    }
    print(json.dumps(row))
