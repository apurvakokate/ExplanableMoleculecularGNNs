#!/usr/bin/env python3
"""Model-performance tables from the mose_replication bundle.
Vanilla / MoSE / MoSE_U / GSAT, per dataset x backbone, mean+-std over folds.
Classification -> test AUC ; Regression -> RMSE (orig units) + RMSE (standardized).
Source = each model's own summary.json in the bundle. GAT from final_v2_gath1 (heads=1).
"""
import json, glob, os, numpy as np, pandas as pd
B = os.environ.get('BUNDLE', '/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/mose_replication')
OUTDIR = os.environ.get('OUTDIR', '/tmp/perf')
CLS = ['BBBP','Mutagenicity','hERG','Benzene_Verified_GT','Fluoride_Carbonyl_Verified_GT','Alkane_Carbonyl_Verified_GT']
REG = ['esol','Lipophilicity']
DISP = {'Benzene_Verified_GT':'Benzene','Fluoride_Carbonyl_Verified_GT':'Fluoride-Carbonyl',
        'Alkane_Carbonyl_Verified_GT':'Alkane-Carbonyl'}
BB = ['GIN','GCN','GAT','SAGE','PNA']
MODELS = ['Vanilla','MoSE','MoSE_U','GSAT']
SPECS = [
  dict(model='Vanilla', roots=['final_v2','final_v2_gath1'],        family='vanilla',   vocab='rbrics',        unk=None),
  dict(model='MoSE',    roots=['final_v2','final_v2_gath1'],         family='mose',      vocab='rbrics_filter', unk='unk-fixed'),
  dict(model='MoSE_U',  roots=['ablated_completely_v1','final_v2_gath1'], family='mose', vocab='rbrics_filter', unk='unk-learnable_shared'),
  dict(model='GSAT',    roots=['final_v2','final_v2_gath1'],         family='base_gsat', vocab='rbrics',        unk=None),
]
ALLDS = CLS + REG

rows = []
for sp in SPECS:
    for root in sp['roots']:
        is_gath1 = root.endswith('gath1')
        base = f'{B}/{root}/{sp["family"]}'
        for f in glob.glob(f'{base}/**/summary.json', recursive=True):
            # scope: vocab, real, dataset, unk, and GAT<->gath1 split
            if f'/{sp["vocab"]}/' not in f and f'_{sp["vocab"]}_' not in f: continue
            if '_real' not in f and '/real' not in f and 'real' not in os.path.basename(os.path.dirname(f)): pass
            if any(bad in f for bad in ('gt_dnf','relabelled','/mutag/','ogbg')): continue
            if sp['unk'] and sp['unk'] not in f: continue
            # mose fixed leaf must NOT be the learnable one and vice-versa
            if sp['model']=='MoSE' and 'unk-learnable_shared' in f: continue
            try: d = json.load(open(f))
            except Exception: continue
            ds, bb, fold = d.get('dataset'), d.get('backbone'), d.get('fold')
            if ds not in ALLDS or bb not in BB: continue
            if is_gath1 and bb != 'GAT':  continue      # gath1 only supplies GAT
            if (not is_gath1) and bb == 'GAT': continue  # GAT never from non-gath1 tree
            rows.append(dict(model=sp['model'], dataset=ds, backbone=bb, fold=fold,
                             auc=d.get('auc'), rmse=d.get('rmse'), rmse_orig=d.get('rmse_orig')))
df = pd.DataFrame(rows).drop_duplicates(['model','dataset','backbone','fold'])

def agg(ds, bb, model, col):
    s = df[(df.dataset==ds)&(df.backbone==bb)&(df.model==model)]
    v = pd.to_numeric(s[col], errors='coerce').dropna().values
    return (np.nan, 0, 0) if len(v)==0 else (float(np.mean(v)), float(np.std(v, ddof=1) if len(v)>1 else 0.0), len(v))

def cell(ds, bb, model, col, scale100=False, dec=2):
    m,s,n = agg(ds,bb,model,col)
    if np.isnan(m): return '--'
    if scale100:
        return f'${m*100:.0f}_{{{s*100:.0f}}}$'.replace('.0','')
    return f'${m:.{dec}f}_{{{s:.{dec}f}}}$'

def build(datasets, col, title, scale100=False, dec=2):
    L = [f'% {title}  (bundle; mean_{{std}} over folds)',
         r'\begin{tabular}{ll'+'c'*len(MODELS)+'}', r'\toprule',
         'Dataset & BB & '+' & '.join(MODELS)+r' \\', r'\midrule']
    for ds in datasets:
        disp = DISP.get(ds, ds)
        for i,bb in enumerate(BB):
            c = [rf'\multirow{{5}}{{*}}{{{disp}}}' if i==0 else '', bb]
            for mo in MODELS: c.append(cell(ds,bb,mo,col,scale100,dec))
            L.append(' & '.join(c)+r' \\')
        L.append(r'\midrule')
    L[-1]=r'\bottomrule'; L.append(r'\end{tabular}')
    return '\n'.join(L)

def pretty(datasets, col, title, dec=3):
    out=[f'### {title}', 'Dataset            BB    '+'  '.join(f'{m:>12}' for m in MODELS)]
    for ds in datasets:
        for bb in BB:
            cells=[]
            for mo in MODELS:
                m,s,n=agg(ds,bb,mo,col)
                cells.append('     --     ' if np.isnan(m) else f'{m:.{dec}f}±{s:.{dec}f}({n})')
            out.append(f'{DISP.get(ds,ds)[:18]:18}{bb:5} '+'  '.join(f'{c:>12}' for c in cells))
    return '\n'.join(out)

import sys
os.makedirs(OUTDIR, exist_ok=True)
open(f'{OUTDIR}/classification_auc.tex','w').write(build(CLS,'auc','Classification — test ROC-AUC',scale100=True))
open(f'{OUTDIR}/regression_rmse_orig.tex','w').write(build(REG,'rmse_orig','Regression — RMSE (original units)',dec=3))
open(f'{OUTDIR}/regression_rmse_std.tex','w').write(build(REG,'rmse','Regression — RMSE (standardized)',dec=3))
print(pretty(CLS,'auc','CLASSIFICATION — test ROC-AUC'))
print(); print(pretty(REG,'rmse_orig','REGRESSION — RMSE (original units)'))
print(); print(pretty(REG,'rmse','REGRESSION — RMSE (standardized)'))
print('\n[fold coverage] min/median/max folds per populated (ds,bb,model) cell:')
cov=df.groupby(['model','dataset','backbone']).fold.nunique()
print(f'  n_cells={len(cov)}  folds: min={cov.min()} median={int(cov.median())} max={cov.max()}')
print('  cells with <5 folds:');
for k,v in cov[cov<5].items(): print(f'    {k} -> {v} folds')
