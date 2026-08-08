import os, glob
import pandas as pd, numpy as np

BASE="/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor"
LOG=BASE+"/analysis/_me_probe_logs"
DATASETS=["BBBP","Benzene_Verified_GT","Alkane_Carbonyl_Verified_GT","Fluoride_Carbonyl_Verified_GT",
          "hERG","Mutagenicity","esol","Lipophilicity","mutag","ogbg-molbace","ogbg-molhiv"]
BACKBONES=["GIN","GCN","GAT","SAGE","PNA"]
# (col, kind, tree, stem, probe_family, probe_key)
METHODS=[
 ("MOSE","native",  "final_v2/mose/rbrics_filter","mose",           "antehoc","/mose/rbrics_filter/"),
 ("MotifSAT","native","final_v2/motifsat/rbrics", "motifsat",       "antehoc","/motifsat/rbrics/"),
 ("GSAT","native",  "final_v2/base_gsat/rbrics",  "gsat",           "antehoc","/base_gsat/rbrics/"),
 ("GNNExplainer","posthoc","posthoc_v1/baselines","gnnexplainer",   "vanilla","gnnexplainer"),
 ("PGExplainer","posthoc","posthoc_v1/baselines", "pgexplainer",    "vanilla","pgexplainer"),
 ("MotifOcclusion","posthoc","posthoc_v1/baselines","motif_occlusion","vanilla","motif_occlusion"),
 ("MAGE","posthoc", "posthoc_v1/baselines",       "mage_v2",        None,     None),
]
def native_runs(tree,ds):
    return [r for r in glob.glob(f"{BASE}/{tree}/{ds}/fold*/*") if "_real" in os.path.basename(r) and os.path.isdir(r)]
def posthoc_runs(tree,ds):
    return [r for r in glob.glob(f"{BASE}/{tree}/{ds}/fold*/rbrics/*") if os.path.basename(r).endswith("_real") and "/rbrics/" in r and os.path.isdir(r)]
def bb_of(run,kind):
    b=os.path.basename(run); return (b.replace("bb-","").split("_")[0]) if kind=="posthoc" else b.split("_")[0]
def rd(f):
    try: return pd.read_csv(f)
    except Exception: return None

def disc_map(ds):
    fr=[]
    for tree in ["final_v2/mose/rbrics_filter","final_v2/motifsat/rbrics","final_v2/base_gsat/rbrics"]:
        for r in native_runs(tree,ds):
            d=rd(f"{r}/score_vs_impact.csv")
            if d is not None and {"motif_smarts","abs_disc"}<=set(d.columns): fr.append(d[["motif_smarts","abs_disc"]])
    if not fr: return {}
    return pd.concat(fr).groupby("motif_smarts")["abs_disc"].mean().to_dict()

def grouped_native(runs):
    fr=[]
    for r in runs:
        d=rd(f"{r}/score_vs_impact.csv")
        if d is not None and "motif_smarts" in d.columns: fr.append(d)
    if not fr: return None
    g=pd.concat(fr,ignore_index=True)
    agg={"score":"mean","impact":"mean"}
    if "abs_disc" in g.columns: agg["abs_disc"]="mean"
    out=g.groupby("motif_smarts").agg(agg).reset_index()
    out=out.rename(columns={"score":"importance","abs_disc":"disc"})
    if "disc" not in out.columns: out["disc"]=np.nan
    return out

def pool(runs,stem,suffix):
    fr=[]
    for r in runs:
        for f in glob.glob(f"{r}/{stem}_{suffix}_*.csv"):
            d=rd(f)
            if d is not None and "motif_smarts" in d.columns: fr.append(d)
    return pd.concat(fr,ignore_index=True) if fr else None

def grouped_posthoc(runs,stem,dmap):
    imp=pool(runs,stem,"importance")
    if imp is None: return None
    out=imp.groupby("motif_smarts").agg(importance=("score","mean")).reset_index()
    imp2=pool(runs,stem,"impact")
    if imp2 is not None:
        out=out.merge(imp2.groupby("motif_smarts")["impact"].mean().reset_index(),on="motif_smarts",how="left")
    else: out["impact"]=np.nan
    out["disc"]=out["motif_smarts"].map(dmap)
    return out

def probe_val(ds,bb,fam,key,kind):
    p=f"{BASE}/final_v2/masked_node_probe_{ds}.csv"
    if fam is None: return np.nan
    d=rd(p)
    if d is None: return np.nan
    d["run_dir"]=d["run_dir"].astype(str)
    if kind=="posthoc":
        m=(d["family"]=="vanilla")&(d["method"].astype(str)==key)&(d["run_dir"].str.contains(f"bb-{bb}_",regex=False))&(d["run_dir"].str.contains("/rbrics/",regex=False))
        gap="expl_gap_unmasked_minus_masked"
    else:
        m=(d["family"]=="antehoc")&(d["run_dir"].str.contains(key,regex=False))&(d["run_dir"].str.contains(f"/{bb}_",regex=False))&(d["run_dir"].str.contains("real",regex=False))
        gap="gated_gap_unmasked_minus_masked"
    v=d.loc[m,gap]
    return float(v.mean()) if len(v) else np.nan

rows_top=[]; rows_bot=[]; cov=[]
for ds in DATASETS:
    dmap=disc_map(ds)
    for bb in BACKBONES:
        for col,kind,tree,stem,pfam,pkey in METHODS:
            runs=[r for r in (native_runs(tree,ds) if kind=="native" else posthoc_runs(tree,ds)) if bb_of(r,kind)==bb]
            pv=probe_val(ds,bb,pfam,pkey,kind)
            g=(grouped_native(runs) if kind=="native" else grouped_posthoc(runs,stem,dmap)) if runs else None
            n=0 if g is None else len(g); cov.append((ds,bb,col,len(runs),n))
            if not n: continue
            g["probe"]=pv
            for lst,asc in [(rows_top,False),(rows_bot,True)]:
                gg=g.sort_values("importance",ascending=asc).head(10).reset_index(drop=True)
                for rank,rw in gg.iterrows():
                    lst.append(dict(dataset=ds,backbone=bb,rank=rank+1,method=col,smarts=rw.motif_smarts,
                        importance=rw.importance,impact=rw.get("impact",np.nan),disc=rw.get("disc",np.nan),node_mask_probe=pv))

def to_wide(rows,name):
    df=pd.DataFrame(rows); df.to_csv(f"{LOG}/{name}_long.csv",index=False)
    piv=df.pivot_table(index=["dataset","backbone","rank"],columns="method",
        values=["smarts","importance","impact","disc","node_mask_probe"],aggfunc="first")
    piv.columns=[f"{m}_{v}" for v,m in piv.columns]
    order=[f"{m}_{v}" for m in ["GNNExplainer","PGExplainer","MAGE","MotifOcclusion","MOSE","MotifSAT","GSAT"]
           for v in ["smarts","importance","impact","disc","node_mask_probe"] if f"{m}_{v}" in piv.columns]
    piv[order].reset_index().to_csv(f"{LOG}/{name}_wide.csv",index=False)
    print(f"WROTE {name}_wide.csv rows={len(piv)} long={len(df)} datasets={df.dataset.nunique()} methods={sorted(df.method.unique())}")

to_wide(rows_top,"topbottom_top"); to_wide(rows_bot,"topbottom_bottom")
c=pd.DataFrame(cov,columns=["dataset","backbone","method","n_runs","n_motifs"]); c.to_csv(f"{LOG}/topbottom_coverage.csv",index=False)
miss=c[c.n_motifs==0]
print("EMPTY cells:",len(miss)); 
print(miss.groupby(["method"]).size().to_string() if len(miss) else "none")
print("EMPTY by dataset:"); print(miss.groupby("dataset").size().to_string() if len(miss) else "none")
