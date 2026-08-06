#!/usr/bin/env python3
"""build_ablation_set.py — LOCAL harvester for the ablation_v1 study (OLD format).

The ablation study was run with the PRE-fix (pre-2026-08-02) code, so each run has
a single OLD-format summary.json (NOT the per-split summary_splits.json). Post-hoc
explainers there were fit/scored ON TEST, so this harvester takes the TEST-ONLY
(bare, no `_all` suffix) keys, with the MEAN node-reduction (`_mean_`, not `_max_`).

It emits ONE ROW PER (run x model), with every ablation axis parsed from the path:
  regime(normal|m1), family, dataset, backbone, fold, fragmentation, filtered,
  node_encoder(onehot|linear), conv_normalize(none|l2|layernorm),
  graph_pool(add|mean), unk_mode(fixed|learnable| '' for non-MoSE)
plus the model column (post-hoc explainer OR native model) and impact_basis.

Metric mapping (TEST-ONLY):
  post-hoc {e}:  grp_pearson_agnostic = {e}_mean_pearson_motif_agnostic
                 grp_spearman_agnostic= {e}_mean_spearman_motif_agnostic
                 grp_pearson_own      = {e}_mean_pearson_motif
                 inst_pearson_agnostic= {e}_mean_pearson_instance_agnostic
                 gtroc_fired          = {e}_mean_gt_roc_node_auc_mean  (whole=fired for source-GT)
  native model:  own top-level pearson_motif / spearman_motif / pearson_instance
                 / gt_roc_node_auc_mean   (impact_basis='own')
  predictive (both): auc / train_auc / val_auc / rmse / mae   (top-level)

Baseline ("without ablation") = the IN-STUDY default point: node_encoder=onehot,
conv_normalize=none, graph_pool=add, regime=normal, unk_mode=fixed — it is simply
the row whose ablation axis is at its default value (no separate source needed).

Usage:
  python analysis/build_ablation_set.py --root <ablation_v1> --out ablation_per_fold.csv
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

FRAG_MAP = {'rbrics': 'rbrics', 'rdkit_fg_first': 'FGFirst-RDKIT',
            'ertl_first': 'ertl', 'fg_first': 'custom-FG'}
EXPL_MAP = {'gnnexplainer': 'GNNExplainer', 'pgexplainer': 'PGExplainer',
            'mage_official': 'MAGE', 'motif_occlusion': 'MotifOcclusion'}
FAM_MODEL = {'mose': 'MoSE', 'motifsat': 'MotifSAT', 'gsat': 'GSAT', 'vanilla': 'Vanilla'}
NATIVE = {'mose', 'motifsat', 'gsat'}
BACKBONES = {'GIN', 'GCN', 'GAT', 'SAGE', 'PNA'}
LAYOUT_A = {'baselines', 'vanilla'}          # regime/family/DS/fold/VOCAB/bb-...
LAYOUT_B = {'mose', 'motifsat', 'base_gsat'}  # regime/family/VOCAB/DS/fold/BB_...
NORM_TEX = {'none': 'none', 'l2': 'L2', 'layernorm': 'LayerNorm'}


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _vocab(voc):
    filtered = voc.endswith('_filter')
    base = voc[:-len('_filter')] if filtered else voc
    return FRAG_MAP.get(base, base), filtered


def _parse(root, ss):
    rel = ss.parent.relative_to(root).parts
    if len(rel) < 6:
        return None
    regime, family = rel[0], rel[1]
    fam_key = 'gsat' if family == 'base_gsat' else family
    if family in LAYOUT_A:
        dataset, fold_s, vocab, runid = rel[2], rel[3], rel[4], rel[5]
        mbb = re.match(r'bb-([A-Za-z]+)_', runid)
        backbone = mbb.group(1) if mbb else ''
    elif family in LAYOUT_B:
        vocab, dataset, fold_s, runid = rel[2], rel[3], rel[4], rel[5]
        backbone = runid.split('_', 1)[0]
    else:
        return None
    # encoder by KEYWORD (positional is unsafe: MotifSAT/GSAT run-ids insert a
    # readout/none token before the encoder, e.g. GIN_readout_onehot_..., GAT_none_onehot_...)
    me = re.search(r'(?:^|_|enc-)(atom_encoder|onehot|linear)(?:_|$)', runid)
    enc = me.group(1) if me else ''
    if backbone not in BACKBONES:
        return None
    mn = re.search(r'norm-([a-z0-9]+)', runid); norm = mn.group(1) if mn else 'none'
    mp = re.search(r'pool-([a-z]+)', runid); pool = mp.group(1) if mp else 'add'
    mu = re.search(r'unk-([a-z]+)', runid); unk = mu.group(1) if mu else ''
    mf = re.match(r'fold(\d+)', fold_s); fold = int(mf.group(1)) if mf else -1
    frag, filtered = _vocab(vocab)
    return dict(regime=regime, family=fam_key, dataset=dataset, backbone=backbone,
                fragmentation=frag, filtered=bool(filtered), node_encoder=enc,
                conv_normalize=norm, graph_pool=pool, unk_mode=unk, fold=fold,
                run_path=str(ss.parent))


def _posthoc_metrics(d, e):
    return dict(
        grp_pearson_agnostic=_num(d.get(f'{e}_mean_pearson_motif_agnostic')),
        grp_spearman_agnostic=_num(d.get(f'{e}_mean_spearman_motif_agnostic')),
        grp_pearson_own=_num(d.get(f'{e}_mean_pearson_motif')),
        grp_spearman_own=_num(d.get(f'{e}_mean_spearman_motif')),
        inst_pearson_agnostic=_num(d.get(f'{e}_mean_pearson_instance_agnostic')),
        inst_pearson_own=_num(d.get(f'{e}_mean_pearson_instance')),
        gtroc_fired=_num(d.get(f'{e}_mean_gt_roc_node_auc_mean')))


def _native_metrics(d):
    # native model: own scoring; agnostic axis is meaningless -> mirror own value
    p, s, i = _num(d.get('pearson_motif')), _num(d.get('spearman_motif')), _num(d.get('pearson_instance'))
    g = _num(d.get('gt_roc_node_auc_mean'))
    return dict(grp_pearson_agnostic=p, grp_spearman_agnostic=s, grp_pearson_own=p,
                grp_spearman_own=s, inst_pearson_agnostic=i, inst_pearson_own=i, gtroc_fired=g)


def _pred(d):
    return dict(auc=_num(d.get('auc')), train_auc=_num(d.get('train_auc')),
                val_auc=_num(d.get('val_auc')), rmse=_num(d.get('rmse')), mae=_num(d.get('mae')))


def harvest(root):
    root = Path(root)
    rows = []
    for ss in root.rglob('summary.json'):
        ident = _parse(root, ss)
        if ident is None:
            continue
        try:
            d = json.loads(ss.read_text())
        except Exception:
            continue
        fam = ident.pop('family')
        pred = _pred(d)
        if fam == 'baselines':
            for e, name in EXPL_MAP.items():
                if any(k.startswith(e + '_mean_') for k in d):
                    rows.append({**ident, 'family': fam, 'model': name,
                                 'model_type': 'posthoc', 'impact_basis': 'agnostic',
                                 **_posthoc_metrics(d, e), **pred})
        elif fam in NATIVE:
            rows.append({**ident, 'family': fam, 'model': FAM_MODEL[fam],
                         'model_type': 'antehoc', 'impact_basis': 'own',
                         **_native_metrics(d), **pred})
        elif fam == 'vanilla':
            rows.append({**ident, 'family': fam, 'model': 'Vanilla',
                         'model_type': 'vanilla', 'impact_basis': 'own', **pred})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='ablation_v1 dir (contains normal/ m1/)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rows = harvest(args.root)
    if not rows:
        raise SystemExit('no parseable summary.json under ' + args.root)
    df = pd.DataFrame(rows)
    df['conv_normalize_tex'] = df['conv_normalize'].map(lambda x: NORM_TEX.get(x, x))
    df.to_csv(args.out, index=False)
    print(f'rows={len(df)}  written {args.out}')
    print('by (regime, model_type):')
    print(df.groupby(['regime', 'model_type']).size().to_string())
    print('ablation axes present:')
    for ax in ['conv_normalize', 'graph_pool', 'node_encoder', 'unk_mode', 'filtered']:
        print(f'  {ax}: {sorted(df[ax].dropna().unique().tolist())}')


if __name__ == '__main__':
    main()
