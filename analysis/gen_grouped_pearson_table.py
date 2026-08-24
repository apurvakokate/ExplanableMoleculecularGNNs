#!/usr/bin/env python3
"""Real-label grouped-Pearson (and GT-ROC) table generator with an EXPLICIT provenance map.

The results are spread across many trees produced by DIFFERENT drivers. This script:
  (1) encodes WHERE each cell lives  (SOURCE_MAP below — the validation table),
  (2) fetches each cell from its resolved location (multi-format readers),
  (3) flags every metric NOT produced by analysis/evaluate.py (the single pipeline), and
  (4) provides compute_via_evaluate() to (re)compute any such cell ON evaluate.py.

Cell key = (dataset, backbone, explainer, regime['source'|'planted'], unk['filt'|'full'|'learn'], metric).
GAT is ALWAYS read from the heads=1 tree (final_v2_gath1 / eval_gath1); every other backbone from final_v2.

Usage:
  # provenance report (what is / isn't on evaluate.py, + the exact evaluate.py commands to fix gaps)
  python gen_grouped_pearson_table.py --report_provenance
  # build the real-label grouped-Pearson LaTeX table from the evaluate.py rollups
  python gen_grouped_pearson_table.py --build \
     --gat_rollups eval_gath1/gath1_rollup_all.csv eval_gath1/gath1_rollup_include.csv \
     --other_rollups eval_real/eval_real_rollup_all.csv --out grouped_pearson_real.tex
"""
import argparse, glob, os, json, subprocess
import pandas as pd, numpy as np

BASE = '/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor'
FOLDS = '/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/'
VOCAB_ROOT = f'{BASE}/vocab_final_v2'

# ── table layout (matches the uploaded template) ────────────────────────────────
DATASETS = [('BBBP','BBBP'), ('Mutagenicity','Mutagenicity'), ('hERG','hERG'),
            ('esol','esol'), ('Lipophilicity','Lipophilicity'),
            ('Benzene','Benzene_Verified_GT'),
            ('Fluoride--Carbonyl','Fluoride_Carbonyl_Verified_GT'),
            ('Alkane--Carbonyl','Alkane_Carbonyl_Verified_GT')]
BACKBONES = ['GIN','GCN','GAT','SAGE','PNA']
COLUMNS = [('GNNExpl.','gnnexplainer',['Filt','Full']), ('PGExpl.','pgexplainer',['Filt','Full']),
           ('MAGE','mage',['Filt','Full']), ('Occlusion','motif_occlusion',['Filt','Full']),
           ('GSAT','gsat',['Filt','Full']), ('MOSE','mose:fixed',['Filt']),
           ('MOSE$_U$','mose:learn',['Filt'])]
UNK = {'Filt':'exclude', 'Full':'include'}
METRIC = 'grouped_pearson_u'

# ── PROVENANCE MAP: (metric, regime) -> per-method source of record ──────────────
# via_ep=True means the tree was produced by analysis/evaluate.py; else the driver named.
# GAT is overridden to the heads=1 source everywhere.
SOURCE_MAP = {
 ('grouped_pearson','source'): {
   'gnnexplainer|pgexplainer|mage|motif_occlusion': dict(
       nongat=('eval_real','evaluate.py rollup', True),
       gat=('eval_gath1','evaluate.py rollup', True),
       legacy=('posthoc_v1/baselines','{stem}_grouped_corr_pooled_alltest.csv', False)),
   'gsat': dict(nongat=('eval_real','rollup',True), gat=('eval_gath1','rollup',True),
                legacy=('antehoc_recompute_v1/base_gsat','gsat_grouped_corr_pooled_alltest.csv',False)),
   'mose:fixed': dict(nongat=('eval_real','rollup',True), gat=('eval_gath1','rollup',True),
                legacy=('antehoc_recompute_v1/mose (unk-fixed)','mose_grouped_corr_pooled_alltest.csv',False)),
   'mose:learn': dict(nongat=('antehoc_recompute_v1/mose (unk-learnable_shared)','mose_grouped_corr_pooled_alltest.csv',False),
                gat=('eval_gath1','rollup',True), legacy=None),
 },
 ('grouped_pearson','planted'): {
   'mose|gsat': dict(any=('planted_{ds}_rbrics_v1/all_results.csv','pearson_motif column',False)),
   'gnnexplainer|pgexplainer|mage|motif_occlusion': dict(any=('MISSING','not computed',False)),
 },
 ('gt_roc','source'): {
   'mose|gsat': dict(nongat=('gtroc_filtered_exclunk_v1/{fam}','gtroc.json',False),
                gat=('eval_gath1','summary_splits gt_roc_node_*',True),
                note='eval_real used gt_tier=none -> no GT-ROC'),
   'gnnexplainer|pgexplainer|mage|motif_occlusion': dict(
                nongat=('gtroc_nodedirect_v1/baselines (Full) | gtroc_filtered_exclunk_v1/baselines (Filt)','gtroc.json',False),
                gat=('eval_gath1','summary_splits',True)),
 },
 ('gt_roc','planted'): {
   'mose|gsat': dict(any=('planted_{ds}_rbrics_v1/all_results.csv','gt_roc_node_fired_auc_mean',False)),
   'gnnexplainer|pgexplainer|mage|motif_occlusion': dict(any=('planted_{ds}_rbrics_v1/posthoc_gtroc/... (NEEDS RECOMPUTE, reads 0)','summary_splits.json',False)),
 },
}

# ── which ckpt/artifact roots each cell computes from (for compute_via_evaluate) ─
def ckpt_root(backbone, regime, dataset, method=None):
    if regime == 'planted':
        return f'{BASE}/planted_{dataset.lower().split("_")[0]}_rbrics_v1'  # planted trees are heads=1
    if backbone == 'GAT':
        return f'{BASE}/final_v2_gath1'                       # heads=1 GAT
    # non-GAT: MoSE learn-unk models live in ablated_completely_v1 (final_v2 has only unk-fixed);
    # everything else (fixed MoSE, GSAT, baselines) in final_v2.
    if method == 'mose:learn':
        return f'{BASE}/ablated_completely_v1'
    return f'{BASE}/final_v2'

def artifacts_root(method, regime, dataset):
    if method not in ('gnnexplainer','pgexplainer','mage','motif_occlusion'):
        return None
    return ckpt_root('X', regime, dataset) if regime == 'planted' else f'{BASE}/posthoc_v1'

def compute_via_evaluate(method, dataset, regime, unk, *, run=False):
    """Emit (and optionally run) the analysis/evaluate.py command that puts this cell ON the
    single pipeline. GAT vs non-GAT differ by ckpt_root, so a full cell is two calls."""
    gt_tier = 'planted' if regime == 'planted' else ('source' if dataset.endswith('_Verified_GT') else 'none')
    vocab = 'rbrics_filter' if method.startswith('mose') else 'rbrics'
    unk_flag = {'filt':'exclude','full':'include','learn':'exclude'}[unk]
    m = 'mose' if method.startswith('mose') else method
    cmds = []
    for bbgrp, bb in [('gat', 'GAT'), ('nongat', 'GCN')]:
        root = ckpt_root(bb, regime, dataset, method)   # GAT->gath1; nongat mose:learn->ablated_completely_v1
        art = artifacts_root(m, regime, dataset)
        cmd = (f"python analysis/evaluate.py --method {m} --dataset {dataset} --vocab {vocab} "
               f"--gt_tier {gt_tier} --unk {unk_flag} --ckpt_root {root} "
               + (f"--artifacts_root {art} " if art else "")
               + f"--dest_root {BASE}/eval_unified --dest {BASE}/eval_unified/_rollups/{m}_{dataset}_{bbgrp}_{unk_flag}.csv "
               f"--data_root {FOLDS} --vocab_root {VOCAB_ROOT}")
        cmds.append(cmd)
        if regime == 'planted':
            break  # planted has one heads=1 tree; no gat/nongat split
    if run:
        for c in cmds:
            subprocess.run(c, shell=True, check=False)
    return cmds

def report_provenance():
    print('# PROVENANCE — where each cell lives (EP=via evaluate.py)\n')
    for (metric, regime), methods in SOURCE_MAP.items():
        print(f'## {metric}  |  regime={regime}')
        for mkey, src in methods.items():
            for role, val in src.items():
                if role == 'note' or val is None:
                    if role == 'note': print(f'   {mkey}: NOTE {val}')
                    continue
                tree, f, ep = val
                tag = 'EP✓' if ep else 'not-EP'
                print(f'   {mkey:45s} [{role:6s}] {tag:7s} {tree}  ::  {f}')
        print()
    print('# METRICS NOT ON evaluate.py -> run compute_via_evaluate to fix. Examples:')
    for (meth, ds, reg, unk) in [('mose:learn','BBBP','source','learn'),
                                 ('gsat','Benzene_Verified_GT','source','filt'),
                                 ('mage','BBBP','planted','filt')]:
        print(f'\n## {meth} / {ds} / {reg} / unk={unk}')
        for c in compute_via_evaluate(meth, ds, reg, unk):
            print('   ' + c)

# ── build the real-label grouped-Pearson LaTeX table from evaluate.py rollups ────
def method_key(row):
    m = row['method']
    if m != 'mose': return m
    return 'mose:learn' if 'unk-learnable_shared' in str(row['run_path']) else 'mose:fixed'

def load(gat_paths, other_paths, vocab):
    def read_many(paths):
        fs = [pd.read_csv(f) for p in paths for f in glob.glob(p)]
        return pd.concat(fs, ignore_index=True) if fs else pd.DataFrame()
    gat, oth = read_many(gat_paths), read_many(other_paths)
    gat = gat[gat.backbone == 'GAT'] if len(gat) else gat
    oth = oth[oth.backbone != 'GAT'] if len(oth) else oth
    df = pd.concat([gat, oth], ignore_index=True)
    if vocab and len(df):
        df = df[df.vocab.isin([vocab, f'{vocab}_filter'])]
    if len(df): df['mkey'] = df.apply(method_key, axis=1)
    return df

def cell(df, ds_id, bb, mkey, unk_eval, metric=METRIC):
    if not len(df) or metric not in df.columns: return '--'
    sel = df[(df.dataset==ds_id)&(df.backbone==bb)&(df.mkey==mkey)&(df.unk==unk_eval)]
    v = pd.to_numeric(sel[metric], errors='coerce').dropna().values
    if len(v)==0: return '--'
    m, s = float(np.mean(v)), float(np.std(v, ddof=1) if len(v)>1 else 0.0)
    sign = '-' if m<0 else ''
    return f'${sign}.{min(round(abs(m)*100),99):02d}_{{{min(round(s*100),99):02d}}}$'

def build(df, metric=METRIC):
    ncol = sum(len(sub) for _,_,sub in COLUMNS)
    L = [rf'% metric={metric} (real labels); cell=.mean_{{std}} x100; Filt=exclude Full=include; GAT=heads1(eval_gath1)',
         r'\begin{tabular}{ll'+'c'*ncol+'}', r'\toprule']
    hdr = ['','']+[rf'\multicolumn{{{len(sub)}}}{{c}}{{{n}}}' for n,_,sub in COLUMNS]
    L.append(' & '.join(hdr)+r' \\')
    sub=['Dataset','BB']; [sub.extend(s) for _,_,s in COLUMNS]
    L.append(' & '.join(sub)+r' \\'); L.append(r'\midrule')
    for disp, ds_id in DATASETS:
        for i,bb in enumerate(BACKBONES):
            cells=[rf'\multirow{{5}}{{*}}{{{disp}}}' if i==0 else '', bb]
            for _,mkey,subs in COLUMNS:
                for sc in subs: cells.append(cell(df, ds_id, bb, mkey, UNK[sc], metric))
            L.append(' & '.join(cells)+r' \\')
        L.append(r'\midrule')
    L[-1]=r'\bottomrule'; L.append(r'\end{tabular}')
    return '\n'.join(L)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--report_provenance', action='store_true',
                    help='print the provenance map + evaluate.py fix-commands, then exit')
    ap.add_argument('--build', action='store_true', help='(default action) build the LaTeX table')
    # DEFAULTS point at the evaluate.py rollups on HPC — so `python gen_..py --build` just works there.
    ap.add_argument('--gat_rollups', nargs='*', default=[            # GAT = heads=1 tree
        f'{BASE}/eval_gath1/gath1_rollup_all.csv',                   #   exclude/Full? -> both files, both unk
        f'{BASE}/eval_gath1/gath1_rollup_include.csv'])
    ap.add_argument('--other_rollups', nargs='*', default=[          # non-GAT
        f'{BASE}/eval_real/eval_real_rollup_all.csv',                #   final_v2: baselines/gsat/mose-fixed
        f'{BASE}/eval_real/eval_real_mose_learn.csv'])               #   ablated_completely_v1: MOSE_U (learn), learnable_shared-only
    ap.add_argument('--vocab', default='rbrics')
    ap.add_argument('--metric', default='grouped_pearson_u',
                    help='rollup column to tabulate (grouped_pearson_u | gtroc_instance | gtroc_global | spurious_roc | family_roc | pred_auc)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.report_provenance:
        report_provenance(); raise SystemExit
    # default action = build the table for --metric
    out = a.out or f'{a.metric}_real.tex'
    df = load(a.gat_rollups, a.other_rollups, a.vocab)
    open(out, 'w').write(build(df, a.metric) + '\n')
    cov = df.groupby('backbone').size().to_dict() if len(df) else {}
    print(f'metric={a.metric} | rows by backbone:', cov, '\nwrote', out)
