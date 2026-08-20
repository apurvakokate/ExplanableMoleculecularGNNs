#!/usr/bin/env python3
"""build_metric_set.py — LOCAL harvester for the per-split re-eval outputs.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_GT = {'Benzene_Verified_GT', 'Fluoride_Carbonyl_Verified_GT',
             'Alkane_Carbonyl_Verified_GT', 'mutag'}
FRAG_MAP = {'rbrics': 'rbrics', 'rdkit_fg_first': 'FGFirst-RDKIT',
            'ertl_first': 'ertl', 'fg_first': 'custom-FG'}
# MAGE = mage_v2 (attention-mean scoring; drops the label-leaking class-prob P term).
# mage_official (the published S^cm = S·P) is DROPPED from the harvest entirely: its
# scores are label-leaking AND, on FG-first, computed from the buggy ring-less
# attention. Not mapped here => never read into any per-fold CSV or table.
EXPL_MAP = {'gnnexplainer': 'GNNExplainer', 'pgexplainer': 'PGExplainer',
            'mage_v2': 'MAGE', 'motif_occlusion': 'MotifOcclusion'}
FAM_MODEL = {'mose': 'MoSE', 'motifsat': 'MotifSAT', 'gsat': 'GSAT',
             'vanilla': 'Vanilla'}
BACKBONES = {'GIN', 'GCN', 'GAT', 'SAGE', 'PNA'}
SPLITS = ['test', 'valid', 'train']
LAYOUT_A_FAMS = {'baselines', 'vanilla'}          # DATASET/fold/VOCAB/bb-BACKBONE
LAYOUT_B_FAMS = {'mose', 'motifsat', 'base_gsat'}  # VOCAB/DATASET/fold/BACKBONE_...


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _split_vocab(voc: str):
    """Return (fragmentation, filtered, gt_rule, base_vocab). gt_rule is
    'dnf_kK_rR' when the vocab is a _relabelled_dnf variant (planted), else ''."""
    gt_rule = ''
    m = re.search(r'_relabelled_dnf_k(\d+)_r(\d+)$', voc)
    if m:
        gt_rule = f'dnf_k{m.group(1)}_r{m.group(2)}'
        voc = voc[:m.start()]
    filtered = voc.endswith('_filter')
    base = voc[:-len('_filter')] if filtered else voc
    frag = FRAG_MAP.get(base, base)
    return frag, filtered, gt_rule, voc


def _parse_run(root: Path, ss_path: Path):
    """Parse identity from a summary_splits.json path. Handles both layouts.
    Returns a dict of identity fields, or None if unparseable / a tune_* dir."""
    rel = ss_path.parent.relative_to(root)
    parts = rel.parts
    if any(p.startswith('tune') for p in parts):
        return None
    family = parts[0]
    fam_key = 'gsat' if family == 'base_gsat' else family

    if family in LAYOUT_A_FAMS:
        # <family>/<DATASET>/fold<K>/<VOCAB>/bb-<BB>_enc-..._<synth>
        if len(parts) < 5:
            return None
        dataset, fold_s, vocab, runid = parts[1], parts[2], parts[3], parts[4]
        mbb = re.match(r'bb-([A-Za-z]+)_', runid)
        backbone = mbb.group(1) if mbb else ''
        frag, filtered, gt_rule, base_vocab = _split_vocab(vocab)
        # synthetic from run-id suffix
        mg = re.search(r'_gt_dnf_k(\d+)_r(\d+)$', runid)
        if mg:
            gt_rule = gt_rule or f'dnf_k{mg.group(1)}_r{mg.group(2)}'
            synthetic = 'gt'
        else:
            synthetic = 'real'
    elif family in LAYOUT_B_FAMS:
        # <family>/<VOCAB[_relabelled_dnf]>/<DATASET>/fold<K>/<BB>_..._hp-<hash>
        if len(parts) < 5:
            return None
        vocab, dataset, fold_s, runid = parts[1], parts[2], parts[3], parts[4]
        backbone = runid.split('_', 1)[0]
        frag, filtered, gt_rule, base_vocab = _split_vocab(vocab)
        synthetic = 'gt' if gt_rule else 'real'
    else:
        return None

    if backbone not in BACKBONES:
        return None
    mf = re.match(r'fold(\d+)', fold_s)
    fold = int(mf.group(1)) if mf else -1

    # gt_tier: planted (has a dnf rule) > source (source-GT dataset) > none
    if gt_rule:
        gt_tier = 'planted'
    elif dataset in SOURCE_GT:
        gt_tier = 'source'
    else:
        gt_tier = 'none'

    return dict(dataset=dataset, backbone=backbone, fragmentation=frag,
                filtered=bool(filtered), gt_tier=gt_tier, gt_rule=gt_rule,
                synthetic=synthetic, fold=fold, family=fam_key,
                base_vocab=base_vocab, run_path=str(ss_path.parent))


def _read_grouped(run_dir: Path, method: str) -> dict:
    """4 grouped corr variants from the ALLTEST (joint train+val+test) file."""
    p = run_dir / f'{method}_grouped_corr_pooled_alltest.csv'
    out = {}
    if not p.exists():
        return out
    try:
        r = pd.read_csv(p).iloc[0]
    except Exception:
        return out
    for stat in ('pearson', 'spearman'):
        for w in ('w', 'u'):
            for unk in ('inclunk', 'exclunk'):
                out[f'grouped_{stat}_agnostic_{w}_{unk}'] = _num(r.get(f'{stat}_{w}_{unk}'))
    out['grouped_n_motifs_inclunk'] = _num(r.get('n_motifs_inclunk'))
    out['grouped_n_motifs_exclunk'] = _num(r.get('n_motifs_exclunk'))
    return out


def _read_instance(run_dir: Path, method: str) -> dict:
    """Per-split instance corr (pearson/spearman + n)."""
    out = {}
    for s in SPLITS:
        p = run_dir / f'{method}_instance_corr_{s}.csv'
        if not p.exists():
            continue
        try:
            r = pd.read_csv(p).iloc[0]
        except Exception:
            continue
        out[f'instance_pearson_agnostic_{s}'] = _num(r.get('pearson_instance'))
        out[f'instance_spearman_agnostic_{s}'] = _num(r.get('spearman_instance'))
        out[f'n_instances_{s}'] = _num(r.get('n_instances'))
    return out


def _pred_and_gtroc(method_split: dict) -> dict:
    """Predictive + GT-ROC scalars for one method, all splits, from summary_splits."""
    out = {}
    for s in SPLITS:
        d = method_split.get(s) or {}
        out[f'{s}_auc'] = _num(d.get('auc'))
        out[f'{s}_rmse_orig'] = _num(d.get('rmse_orig'))
        out[f'{s}_mae_orig'] = _num(d.get('mae_orig'))
        out[f'{s}_rmse'] = _num(d.get('rmse'))
        out[f'{s}_mae'] = _num(d.get('mae'))
        # GT-ROC (present only when the graphs carry GT node labels)
        out[f'grouped_gtroc_fired_{s}'] = _num(d.get('global_gt_roc_node_auc_mean'))
        # max over fired disjuncts. Renamed from instance_gtroc_fired_* (2026-08): that name
        # meant fired-UNION in the harvesters and max-over-clauses here, and post-Mode-1-removal
        # the union collapses onto grouped_gtroc, leaving the old name with nothing of its own.
        out[f'instance_gtroc_dnf_{s}'] = _num(d.get('instance_gt_roc_node_auc_mean'))
        # spurious_roc_{s} is the MAX over class-1 correlates (single comparable scalar);
        # the per-motif audit lives in spurious_roc__<motif>_{s} below.
        out[f'spurious_roc_{s}'] = _num(d.get('spurious_roc_node_auc_mean'))
        for _k, _v in d.items():
            if _k.startswith('spurious_roc_node_auc_mean__'):
                out[f"spurious_roc__{_k[len('spurious_roc_node_auc_mean__'):]}_{s}"] = _num(_v)
        out[f'family_roc_{s}'] = _num(d.get('family_roc_node_auc_mean'))
        # gt_roc_cause: node_label = the FIRED-CLAUSE cause. Was `gt_roc_whole_{s}` when
        # node_label meant whole-rule; renamed so the old (wrong) semantics can't be read
        # into new numbers, and so a stale archive column can't silently line up with it.
        out[f'gt_roc_cause_{s}'] = _num(d.get('gt_roc_node_auc_mean'))
    return out


def harvest_root(root: Path) -> list:
    rows = []
    for ss in root.rglob('summary_splits.json'):
        if ss.stat().st_size <= 2:
            continue
        ident = _parse_run(root, ss)
        if ident is None:
            continue
        try:
            summ = json.loads(ss.read_text())
        except Exception:
            continue
        run_dir = ss.parent
        for method, method_split in summ.items():
            # model + impact basis + whether explanation metrics apply
            if ident['family'] == 'baselines':          # post-hoc
                if method not in EXPL_MAP:               # DROP mage_official (+ any legacy key)
                    continue                             # — EXPL_MAP is the strict paper-method gate
                model = EXPL_MAP[method]
                model_type, impact_basis, has_expl = 'posthoc', 'agnostic', True
            elif ident['family'] == 'vanilla':
                model, model_type, impact_basis, has_expl = 'Vanilla', 'vanilla', 'own', False
            else:                                        # mose / motifsat / gsat
                model = FAM_MODEL.get(ident['family'], ident['family'])
                model_type, impact_basis, has_expl = 'antehoc', 'own', True
            row = {k: v for k, v in ident.items() if k != 'family'}
            row.update(model=model, model_type=model_type, impact_basis=impact_basis)
            row.update(_pred_and_gtroc(method_split))
            if has_expl:
                row.update(_read_grouped(run_dir, method))
                row.update(_read_instance(run_dir, method))
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--posthoc-root', default=None)
    ap.add_argument('--antehoc-root', default=None)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--grouped-variant', default='u_exclunk',
                    choices=['w_inclunk', 'u_inclunk', 'w_exclunk', 'u_exclunk'],
                    help='which grouped variant fills the headline '
                         'grouped_{pearson,spearman}_agnostic alias')
    args = ap.parse_args()

    rows = []
    for r in (args.posthoc_root, args.antehoc_root):
        if r:
            rows += harvest_root(Path(r))
    if not rows:
        raise SystemExit('no parseable summary_splits.json under the given roots')
    df = pd.DataFrame(rows)

    # headline aliases from the chosen variant
    v = args.grouped_variant
    df['grouped_pearson_agnostic'] = df.get(f'grouped_pearson_agnostic_{v}')
    df['grouped_spearman_agnostic'] = df.get(f'grouped_spearman_agnostic_{v}')

    # n_folds per cell (all identity axes except fold) — the aggregation denominator
    cell = ['dataset', 'backbone', 'model', 'model_type', 'fragmentation',
            'filtered', 'gt_tier', 'gt_rule', 'synthetic']
    df['n_folds'] = df.groupby(cell)['fold'].transform('nunique')

    id_cols = cell + ['impact_basis', 'fold', 'n_folds', 'run_path']
    metric_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + sorted(metric_cols)]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    main_df = df[df['gt_tier'].isin(['none', 'source'])].copy()
    supp_df = df[df['gt_tier'] == 'planted'].copy()
    main_df.to_csv(out / 'main_metrics_per_fold.csv', index=False)
    supp_df.to_csv(out / 'supp_metrics_per_fold.csv', index=False)

    print(f'rows total={len(df)}  main={len(main_df)}  supp={len(supp_df)}')
    print(f'grouped headline variant = {v}')
    print('MAIN by (model_type, gt_tier):')
    print(main_df.groupby(['model_type', 'gt_tier']).size().to_string())
    print(f'wrote {out/"main_metrics_per_fold.csv"} and {out/"supp_metrics_per_fold.csv"}')


if __name__ == '__main__':
    main()
