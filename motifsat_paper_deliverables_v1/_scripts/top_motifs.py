"""top_motifs.py — READ-ONLY top-10 motifs per (dataset, backbone, preset) (S2, approved 2026-10-26).

Run on the HPC (l2xgnn env):  python3 top_motifs.py OUT_CSV
rBRICS v3 only (one vocabulary for all folds, so motif ids agree across folds).
Inputs: <TREE>/rbrics_v3/<preset>/<grouping>/<ds>/<rep><k>/<bb>/motif_allmask_test.csv, one row per
(test molecule, motif type) with the model's own motif score (mean gate over the motif's atoms) and its
mask-all impact |p(W) - p(W \\ m)|.
Per (dataset, backbone, preset): pool the test rows of the 5 folds / seeds, then per motif type
  n_mol      rows (molecule x fold) containing it;  kept if n_mol >= 10
  score      mean score;  impact  mean impact;  top1  share of its molecules where it is the top-scored motif
  pos_share  share of its molecules with the positive label (classification; mutag positive = label 0;
             Tox21 / ClinTox left empty) or mean target (ESOL, Lipophilicity); base = dataset-wide value
  gt_flag    key contains the dataset's source-GT SMARTS (3 GT datasets) or a nitro / amine-N pattern (Mutag)
Rank by mean score, keep the top 10. Also writes, per (dataset, backbone, preset), the Spearman between score
and impact over all kept motifs (rank = 0 rows).
"""
import glob
import pickle
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.stats import spearmanr

RDLogger.DisableLog('rdApp.*')
R = '/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor'
P = f'{R}/motifsat_paper_deliverables_v1'
FOLDS = '/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS'
OGB = '/nfs/hpc/share/kokatea/ChemIntuit/GSAT_upstream_6812f5c/data'
MAIN = ['Benzene_Verified_GT', 'Alkane_Carbonyl_Verified_GT', 'Fluoride_Carbonyl_Verified_GT', 'BBBP', 'hERG',
        'Mutagenicity', 'esol', 'Lipophilicity']
PAPER = ['mutag', 'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp', 'ogbg-moltox21', 'ogbg-molclintox']
PRESETS = [('base_gsat', 'none'), ('intra_inter_loss', 'pooled'), ('readout_layernorm', 'pooled'),
           ('readout_nonorm', 'pooled')]
GRAPH = {'variants_v1': ('motif_emb_iddesc_chem_res', 'copy'), 'paper_v1': ('motif_emb_iddesc_gin_res', 'copy')}
BB = ['GIN', 'GCN', 'GAT', 'SAGE', 'PNA']
LABEL_COL = {'BBBP': 'BBBP', 'hERG': 'hERG', 'Mutagenicity': 'Mutagenicity', 'esol':
             'measured log solubility in mols per litre', 'Lipophilicity': 'Lipophilicity'}
GT_SMARTS = {'Benzene_Verified_GT': ['c1ccccc1'], 'Fluoride_Carbonyl_Verified_GT': ['[FX1]', '[CX3]=O'],
             'Alkane_Carbonyl_Verified_GT': ['[R0;D2,D1][R0;D2][R0;D2,D1]', '[CX3]=O'],
             'mutag': ['N(~O)~O', '[NX3;!$(N~O);!$(N-C=O)]']}       # mutag: nitro / amine-N (heuristic flag)


def labels(ds):
    """smiles -> label (positive = 1, or the regression target); None for multi-task."""
    if ds in ('ogbg-moltox21', 'ogbg-molclintox'):
        return None
    if ds == 'mutag':
        d = pd.read_csv(f'{R}/data/mutag_0.csv'); y = (d['label'] == 0).astype(float)    # mutagen = label 0
    elif ds.startswith('ogbg'):
        d = pd.read_csv(f'{OGB}/{ds}_0.csv'); y = d['label'].astype(float)
    else:
        d = pd.read_csv(f'{FOLDS}/{ds}_0.csv'); y = d[LABEL_COL.get(ds, 'label')].astype(float)
    return dict(zip(d['smiles'], y))


def gt_flag(ds, key, cache={}):
    pats = GT_SMARTS.get(ds)
    if not pats:
        return ''
    if (ds, key) not in cache:
        m = Chem.MolFromSmiles(key, sanitize=False) or Chem.MolFromSmarts(key)
        if m is None:
            cache[(ds, key)] = '?'
        else:
            m.UpdatePropertyCache(strict=False); Chem.FastFindRings(m)
            hits = [p for p in pats if m.HasSubstructMatch(Chem.MolFromSmarts(p))]
            cache[(ds, key)] = '+'.join(hits)
    return cache[(ds, key)]


def main(out):
    rows = []
    for tree, dsets in (('variants_v1', MAIN), ('paper_v1', PAPER)):
        for ds in dsets:
            keys = list(pickle.load(open(f'{R}/vocab_final_v3/{ds}/rbrics/{ds}_rbrics_motif_list.pickle', 'rb')))
            lab = labels(ds)
            base = float(np.mean(list(lab.values()))) if lab else np.nan
            for preset, grp in PRESETS + [GRAPH[tree]]:
                for bb in BB:
                    fs = sorted(glob.glob(f'{P}/{tree}/rbrics_v3/{preset}/{grp}/{ds}/*/{bb}/motif_allmask_test.csv'))
                    if not fs:
                        continue
                    d = pd.concat([pd.read_csv(f).assign(rep=i) for i, f in enumerate(fs)], ignore_index=True)
                    d['top'] = d.groupby(['rep', 'graph_idx']).score.transform('max') == d.score
                    d['y'] = d.smiles.map(lab) if lab else np.nan
                    g = d.groupby('motif_id').agg(n_mol=('score', 'size'), score=('score', 'mean'),
                                                  impact=('impact', 'mean'), top1=('top', 'mean'),
                                                  pos_share=('y', 'mean'))
                    g = g[g.n_mol >= 10]
                    rho = spearmanr(g.score, g.impact)[0] if len(g) > 2 else np.nan
                    rows.append(dict(dataset=ds, backbone=bb, preset=preset, rank=0, n_runs=len(fs),
                                     n_kept=len(g), spearman_score_impact=rho, base=base,
                                     label_join=float(d.y.notna().mean()) if lab else np.nan))
                    for r, (mid, x) in enumerate(g.sort_values('score', ascending=False).head(10).iterrows(), 1):
                        k = keys[int(mid)] if 0 <= int(mid) < len(keys) else f'id{mid}'
                        rows.append(dict(dataset=ds, backbone=bb, preset=preset, rank=r, motif_id=int(mid), key=k,
                                         n_mol=int(x.n_mol), score=x.score, impact=x.impact, top1=x.top1,
                                         pos_share=x.pos_share, base=base, gt_flag=gt_flag(ds, k)))
                    print(ds, bb, preset, len(fs), len(g), flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == '__main__':
    main(sys.argv[1])
