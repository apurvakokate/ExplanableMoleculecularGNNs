#!/usr/bin/env python3
"""GT-ROC (instance) table, pooled over train+valid+test, from the source-GT
recompute (eval_srcgt_gtroc_v1/unk-{exclude,include}). Per run we read per-split
instance_gt_roc_node_auc_mean from summary_splits.json and pool by graph count
(from each fold CSV's `group` column). exclude=kept-node, include=all-node GT.
Columns: GNNExpl/PGExpl/MAGE/Occlusion/GSAT/MoSE/MoSE_U, each Filt+Full.
"""
import json, glob, re, os, numpy as np, pandas as pd
B=os.environ.get('ROOT','/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor')
SRC=os.environ.get('SRCGT', f'{B}/eval_srcgt_gtroc_v1')
FOLDS=os.environ.get('FOLDS','/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS')
OUTDIR=os.environ.get('OUTDIR','/tmp/gtroc')
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import table_sig as _ts
ALPHA=float(os.environ.get('SIG_ALPHA','0.05')); STYLE=os.environ.get('SIG_STYLE','bold_underline')
SIG=os.environ.get('SIG','1') not in ('0','','false','no')
DATASETS=[('Benzene','Benzene_Verified_GT'),
          ('Fluoride-Carbonyl','Fluoride_Carbonyl_Verified_GT'),
          ('Alkane-Carbonyl','Alkane_Carbonyl_Verified_GT')]
BB=['GIN','GCN','GAT','SAGE','PNA']
# display method key -> summary_splits key
POSTHOC={'gnnexplainer':'gnnexplainer','pgexplainer':'pgexplainer','mage':'mage_v2','motif_occlusion':'motif_occlusion'}
COLS=[('GNNExpl.','gnnexplainer',['filt','full']),('PGExpl.','pgexplainer',['filt','full']),
      ('MAGE','mage',['filt','full']),('Occlusion','motif_occlusion',['filt','full']),
      ('GSAT','gsat',['filt','full']),('MOSE','mose:fixed',['filt','full']),
      ('MOSE$_U$','mose:learn',['filt','full'])]
UNK={'filt':'exclude','full':'include'}

# --- per (dataset,fold) split graph counts from the FOLDS CSV `group` column ---
def split_counts(ds, fold):
    p=f'{FOLDS}/{ds}_{fold}.csv'
    if not os.path.exists(p): return None
    g=pd.read_csv(p)['group'].astype(str).str.lower()
    n={'train':int((g.str.startswith('train')).sum()),
       'valid':int((g.str.startswith('val')).sum()),
       'test': int((g.str.startswith('test')).sum())}
    return n

def pooled(splitvals, counts):
    num=den=0.0
    for s in ('train','valid','test'):
        v=splitvals.get(s); n=counts.get(s,0)
        if v is None or not np.isfinite(v) or n<=0: continue
        num+=n*v; den+=n
    return num/den if den>0 else float('nan')

def backbone_of(run_dir, fam):
    leaf=os.path.basename(run_dir)
    if fam=='baselines':
        m=re.match(r'bb-([^_]+)',leaf); return m.group(1) if m else None
    return leaf.split('_')[0]   # gsat / mose leaves start with <BB>_

def fold_of(run_dir):
    m=re.search(r'/fold(\d+)/',run_dir+'/'); return int(m.group(1)) if m else None

def dataset_of(run_dir):
    for _,ds in DATASETS:
        if f'/{ds}/' in run_dir+'/': return ds
    return None

def mose_variant(run_dir):
    return 'mose:learn' if 'unk-learnable_shared' in run_dir else 'mose:fixed'

# --- collect rows ---
rows=[]
for unk in ('exclude','include'):
    root=f'{SRC}/unk-{unk}'
    for sj in glob.glob(f'{root}/**/summary_splits.json', recursive=True):
        run_dir=os.path.dirname(sj)
        ds=dataset_of(run_dir); fold=fold_of(run_dir)
        if ds is None or fold is None: continue
        fam = 'baselines' if '/baselines/' in run_dir else ('base_gsat' if '/base_gsat/' in run_dir else ('mose' if '/mose/' in run_dir else None))
        if fam is None: continue
        bb=backbone_of(run_dir, fam)
        if bb not in BB: continue
        cnt=split_counts(ds, fold)
        if cnt is None: continue
        try: d=json.load(open(sj))
        except Exception: continue
        # which method keys live in this leaf
        if fam=='baselines': mkeys=list(POSTHOC.values())      # 4 post-hoc
        elif fam=='base_gsat': mkeys=['gsat']
        else: mkeys=['mose']
        for mk in mkeys:
            if mk not in d: continue
            sv={s:d[mk].get(s,{}).get('instance_gt_roc_node_auc_mean') for s in ('train','valid','test')}
            val=pooled(sv, cnt)
            disp_m = ({v:k for k,v in POSTHOC.items()}.get(mk, mk))     # mage_v2->mage
            if disp_m=='mose': disp_m=mose_variant(run_dir)
            rows.append(dict(method=disp_m, dataset=ds, backbone=bb, fold=fold, unk=unk, gtroc=val))
df=pd.DataFrame(rows).drop_duplicates(['method','dataset','backbone','fold','unk'])

def agg_g(ds, bb, mkey, unk):
    s=df[(df.dataset==ds)&(df.backbone==bb)&(df.method==mkey)&(df.unk==unk)]
    v=pd.to_numeric(s['gtroc'],errors='coerce').dropna().values
    if len(v)==0: return (float('nan'),0.0,0)
    return (float(np.mean(v)), float(np.std(v,ddof=1) if len(v)>1 else 0.0), len(v))

def cell(ds, bb, mkey, unk):
    m,sd,n=agg_g(ds,bb,mkey,unk)
    if n==0: return '--'
    return f'${m*100:.1f}_{{{sd*100:.1f}}}$'    # x100, one decimal, subscript=std (14 cols -> compact)

def build():
    # ROW layout: a 'Vocab' column labels Filt (kept-node) vs Full (all-node GT); each
    # (dataset,backbone) spans two rows, one per condition. Best/ties decided per row.
    methods=[mk for _,mk,_ in COLS]
    lastcol=3+len(COLS)
    L=['% GT-ROC instance, Filt/Full as labelled rows (Vocab column)',
       r'\begin{tabular}{lll'+'c'*len(COLS)+'}', r'\toprule',
       ' & '.join(['Dataset','BB','Vocab']+[n for n,_,_ in COLS])+r' \\',
       r'\midrule']
    for disp,ds in DATASETS:
        for bi,bb in enumerate(BB):
            for si,(sc,slab) in enumerate((('filt','Filt'),('full','Full'))):
                cells=[dict(key=mk, **dict(zip(('mean','std','n'), agg_g(ds,bb,mk,UNK[sc])))) for mk in methods]
                tags=_ts.decide(cells, higher_better=True, alpha=ALPHA) if SIG else {}
                dcell=rf'\multirow{{10}}{{*}}{{{disp}}}' if (bi==0 and si==0) else ''
                bcell=rf'\multirow{{2}}{{*}}{{{bb}}}' if si==0 else ''
                row=[dcell, bcell, slab]
                for _,mk,_ in COLS:
                    s=cell(ds,bb,mk,UNK[sc])
                    row.append(_ts.wrap(s, tags.get(mk,'plain'), STYLE) if SIG else s)
                L.append(' & '.join(row)+r' \\')
            if bi < len(BB)-1:
                L.append(rf'\cmidrule(l){{2-{lastcol}}}')   # separate backbone blocks
        L.append(r'\midrule')
    L[-1]=r'\bottomrule'; L.append(r'\end{tabular}')
    return '\n'.join(L)

def pretty():
    out=['DS               BB   '+'  '.join(f'{n}/{x}' for n,_,s in COLS for x in s)]
    for disp,ds in DATASETS:
        for bb in BB:
            cs=[]
            for _,mk,subs in COLS:
                for sc in subs:
                    s=df[(df.dataset==ds)&(df.backbone==bb)&(df.method==mk)&(df.unk==UNK[sc])]
                    v=pd.to_numeric(s['gtroc'],errors='coerce').dropna().values
                    cs.append('--' if len(v)==0 else f'{np.mean(v):.2f}')
            out.append(f'{disp[:16]:16} {bb:5}'+' '.join(f'{x:>5}' for x in cs))
    return '\n'.join(out)

def floatwrap(tab, caption, label):
    return (f'\\begin{{table*}}[t]\n\\centering\n\\small\n{tab}\n'
            f'\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table*}}\n')
_SIG = ((r' Per (dataset, backbone) and within each condition, the best is in '
         r'\textbf{bold} and results not significantly worse (Welch two-sided $t$-test, '
         rf'$\alpha={ALPHA:g}$, Holm) are \underline{{underlined}}.') if SIG else '')
CAP = (r'Instance GT-ROC on the source-GT datasets, pooled over train+valid+test '
       r'(graph-weighted), $\times100$; subscript\,=\,std\,($\times100$). The \textbf{Vocab} '
       r'column gives the evaluation setting: \textbf{Filt} (exclude-unk) restricts to '
       r'kept-motif nodes, \textbf{Full} (include-unk) scores all-node ground-truth '
       r'recovery. The Full$-$Filt gap is large for MoSE/MoSE$_U$ (filtered-vocab models) '
       r'but small for the baselines. MoSE$_U$=MoSE with learnable unknown; BB=backbone; '
       r'cells with $<2$ folds are not tie-tested.' + _SIG)
os.makedirs(OUTDIR,exist_ok=True)
open(f'{OUTDIR}/gtroc_instance_tvt.tex','w').write(floatwrap(build(), CAP, 'tab:gtroc-tvt'))
print(pretty()); print()
cov=df.groupby(['method','dataset','backbone','unk']).fold.nunique()
print('cells:',len(cov),' folds min/med/max:',cov.min(),int(cov.median()),cov.max())
print('cells with <5 folds:', int((cov<5).sum()))
print('\n===TEX===\n'); print(build())
