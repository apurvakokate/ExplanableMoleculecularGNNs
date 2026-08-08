import pandas as pd, numpy as np, glob
from collections import defaultdict
BASE='/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor'
BB=['GIN','GCN','GAT','SAGE','PNA']
DS=['BBBP','Mutagenicity','hERG','Benzene_Verified_GT','Alkane_Carbonyl_Verified_GT','Fluoride_Carbonyl_Verified_GT']
def foldruns(ds,bb):
    o={}
    for f in range(5):
        g=glob.glob(f'{BASE}/final_v2/mose/rbrics_filter/{ds}/fold{f}/{bb}_*real*/')
        if g:o[f]=g[0]
    return o
def ftab(rd):
    s=pd.read_csv(rd+'score_vs_impact.csv').drop_duplicates('motif_id')
    try:disc=pd.read_csv(rd+'discriminativeness.csv').set_index('motif_id')
    except:disc=None
    s=s.sort_values('score',ascending=False).drop_duplicates('motif_smarts').copy()
    s['imp']=s['score']; s['impact']=s['impact_mean']
    s['dp']=[float(disc.loc[m,'delta_p1']) if (disc is not None and m in disc.index) else np.nan for m in s['motif_id']]
    s['prev']=[float(disc.loc[m,'prevalence']) if (disc is not None and m in disc.index) else np.nan for m in s['motif_id']]
    return s[['motif_smarts','imp','impact','dp','prev']]
def per_fold_selection(ds,bb,k=10):
    # returns list of (fold, class, section, smarts, imp, impact, dp, prev)
    rows=[]
    for f,rd in foldruns(ds,bb).items():
        t=ftab(rd)
        for cls,sub in [('pos',t[t.dp>0]),('neg',t[t.dp<0])]:
            byimp=sub.sort_values('imp',ascending=False)
            for sec,sel in [('top',byimp.head(k)),('bottom',byimp.tail(k))]:
                for _,r in sel.iterrows():
                    rows.append((f,cls,sec,r['motif_smarts'],r['imp'],r['impact'],r['dp'],r['prev']))
    return rows
def aggregate(rows, extra_keys=()):
    # accumulate by (class, section, smarts) [+ extra]
    agg=defaultdict(lambda:dict(nf=0,bbset=set(),simpact=0,simp=0,imp=[],impact=[],dp=[],prev=[]))
    for r in rows:
        f,cls,sec,sm,imp,impact,dp,prev=r[:8]; key=(cls,sec,sm)
        a=agg[key]; a['nf']+=1; a['simpact']+=(impact if impact==impact else 0); a['simp']+=imp
        a['imp'].append(imp); a['impact'].append(impact); a['dp'].append(dp); a['prev'].append(prev)
        if len(r)>8: a['bbset'].add(r[8])
    out=[]
    for (cls,sec,sm),a in agg.items():
        out.append(dict(cls=cls,section=sec,smarts=sm,nf=a['nf'],n_backbone=len(a['bbset']),
            simpact=round(a['simpact'],4),mean_impact=round(np.nanmean(a['impact']),4),std_impact=round(np.nanstd(a['impact']),4),
            mean_importance=round(np.nanmean(a['imp']),4),std_importance=round(np.nanstd(a['imp']),4),
            simport=round(a['simp'],4),mean_dp=round(np.nanmean(a['dp']),4),mean_prev=round(np.nanmean(a['prev']),4)))
    return out
grid=[]; pooled=[]
for ds in DS:
    ds_rows=[]
    for bb in BB:
        rows=per_fold_selection(ds,bb)
        for o in aggregate(rows):
            o=dict(dataset=ds,backbone=bb,**o); grid.append(o)
        ds_rows+=[ (r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],bb) for r in rows ]  # add backbone as 9th
    for o in aggregate(ds_rows, extra_keys=('bb',)):
        pooled.append(dict(dataset=ds,**o))
pd.DataFrame(grid).to_csv('analysis/_me_probe_logs/robust_grid.csv',index=False)
pd.DataFrame(pooled).to_csv('analysis/_me_probe_logs/robust_pooled.csv',index=False)
print('grid rows',len(grid),'pooled rows',len(pooled))
print('datasets',sorted(set(r['dataset'] for r in pooled)))
