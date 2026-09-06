#!/usr/bin/env python3
"""make_tables.py — MotifSAT base-runs paper tables (self-contained, artifact-only).

Three tables. Columns = the 9 MotifSAT presets. Rows = Dataset x Backbone [x View]:
  T1  task perf   : AUC (6 classification) / RMSE_orig (2 regression).  NO Full/Filt.
  T2  grouped Pearson (unweighted), all 8 datasets.        Full + Filt.
  T3  instance GT-ROC, 3 source-GT datasets.               Full + Filt.

Both T2 and T3 are computed POOLED over all splits (train+valid+test) and then
averaged over FOLDS only (unit = fold; std = sample std, ddof=1).

Filtered = restrict to the STRICT PER-FOLD kept set, mirroring analysis/evaluate.py
verbatim (build_fold_annotation on the rbrics_filter vocab + the fold CSV,
apply_threshold=True). rbrics and rbrics_filter share the same motif-id space
(verified), so the kept ids apply directly to the rbrics runs.

Highlighting REUSES analysis/table_sig (Welch + Holm; best=bold, tie=underline).
Reads ONLY base_runs artifacts (+ vocab for the kept set; + a data reload via
get_loaders for T3's node labels). No model forward.
"""
from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]   # MotifSAT/paper_results/ -> repo root
sys.path.insert(0, str(_REPO))
from analysis import table_sig as TS  # noqa: E402

# ── campaign constants ───────────────────────────────────────────────────────
PRESET_STEM = {
    'base_gsat': 'gsat', 'base_gsat_noanneal': 'gsat',
    'intra_consistency': 'motifsat', 'inter_loss': 'motifsat',
    'intra_inter_loss': 'motifsat', 'readout': 'motifsat',
    'readout_inter': 'motifsat', 'readout_layernorm': 'motifsat',
    'readout_nonorm': 'motifsat',
}
PRESETS = list(PRESET_STEM)
PRESET_LABEL = {
    'base_gsat': 'GSAT', 'base_gsat_noanneal': 'GSAT-NA',
    'intra_consistency': 'Intra', 'inter_loss': 'Inter',
    'intra_inter_loss': 'Intra+Inter', 'readout': 'Readout',
    'readout_inter': 'Readout+Inter', 'readout_layernorm': 'Readout-LN',
    'readout_nonorm': 'Readout-NN',
}
CLS = ['BBBP', 'hERG', 'Mutagenicity', 'Benzene_Verified_GT',
       'Alkane_Carbonyl_Verified_GT', 'Fluoride_Carbonyl_Verified_GT']
REG = ['esol', 'Lipophilicity']
DATASETS = CLS + REG
SOURCE = ['Benzene_Verified_GT', 'Alkane_Carbonyl_Verified_GT',
          'Fluoride_Carbonyl_Verified_GT']
DS_LABEL = {'BBBP': 'BBBP', 'hERG': 'hERG', 'Mutagenicity': 'Mutagenicity',
            'Benzene_Verified_GT': 'Benzene',
            'Alkane_Carbonyl_Verified_GT': 'Alkane-Carbonyl',
            'Fluoride_Carbonyl_Verified_GT': 'Fluoride-Carbonyl',
            'esol': 'ESOL', 'Lipophilicity': 'Lipophilicity'}
BACKBONES = ['GIN', 'GCN', 'GAT', 'SAGE', 'PNA']
FOLDS = [0, 1, 2, 3, 4]
SPLITS = ['train', 'valid', 'test']


def cell_dir(runs, preset, ds, fold, bb):
    return Path(runs) / preset / ds / f'fold{fold}' / bb


# ── strict per-fold kept set (mirror analysis/evaluate.py 1030-1050) ──────────
_KEPT = {}


def kept_ids(ds, fold, vocab_root, data_root):
    key = (ds, int(fold))
    if key in _KEPT:
        return _KEPT[key]
    from SharedModules.data.vocab import load_vocab
    from SharedModules.data.fold_threshold import build_fold_annotation
    from SharedModules.data.dataset_schema import DATASET_COLUMN
    filt = load_vocab(str(vocab_root), ds, 'rbrics_filter')
    csv = Path(data_root) / f'{ds}_{int(fold)}.csv'
    _, kept, _, _ = build_fold_annotation(
        lookup_all=filt.lookup_all, motif_list=filt.motif_list,
        mol_fragment_smarts=filt.mol_fragment_smarts, csv_path=str(csv),
        label_col=DATASET_COLUMN[ds], dataset=ds,
        variant=filt.variant or 'rbrics_filter',
        vocab_dir=Path(filt.vocab_dir) if filt.vocab_dir else Path('.'),
        apply_threshold=True, threshold_pct=filt.threshold_pct)
    if kept is None:
        raise ValueError(f'no per-fold kept for {ds} fold {fold}')
    kept = set(int(x) for x in kept)
    _KEPT[key] = kept
    return kept


# ── aggregation / formatting ─────────────────────────────────────────────────
def agg(vals):
    a = np.array([v for v in vals if v == v], dtype=float)
    if a.size == 0:
        return (float('nan'), 0.0, 0)
    return (float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0, int(a.size))


def fmt(mean, std, scale=100.0, dp=1):
    if mean != mean:
        return '--'
    return f'${mean*scale:.{dp}f}\\pm{std*scale:.{dp}f}$'


# ── Table 1: task performance ────────────────────────────────────────────────
def harvest_t1(runs, presets, datasets, backbones, folds):
    data = defaultdict(list)   # (ds, bb, preset) -> [per-fold value]
    for preset in presets:
        stem = PRESET_STEM[preset]
        for ds in datasets:
            metric = 'auc' if ds in CLS else 'rmse_orig'
            for bb in backbones:
                for fold in folds:
                    p = cell_dir(runs, preset, ds, fold, bb) / 'summary_splits.json'
                    if not p.exists():
                        continue
                    try:
                        v = json.loads(p.read_text())[stem]['test'][metric]
                    except Exception:
                        continue
                    if v == v:
                        data[(ds, bb, preset)].append(float(v))
    return data


# ── Table 2: grouped Pearson (unweighted), Full + Filt ───────────────────────
def _motif_csv(d, stem, split, kind):
    p = d / f'{stem}_{kind}_{split}.csv'
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def grouped_pearson(d, stem, kept):
    """Pooled (motif,split) points -> (full_pearson_u, filt_pearson_u)."""
    sc, im, mid = [], [], []
    for split in SPLITS:
        imp = _motif_csv(d, stem, split, 'importance')   # motif_id, score
        imc = _motif_csv(d, stem, split, 'impact')       # motif_id, impact
        if imp is None or imc is None:
            continue
        m = imp[['motif_id', 'score']].merge(
            imc[['motif_id', 'impact']], on='motif_id', how='inner')
        sc.extend(m['score'].astype(float).tolist())
        im.extend(m['impact'].astype(float).tolist())
        mid.extend(m['motif_id'].astype(int).tolist())
    if len(sc) < 2:
        return (float('nan'), float('nan'))
    s = np.asarray(sc); i = np.asarray(im); mi = np.asarray(mid)

    def pear(mask):
        a, b = s[mask], i[mask]
        if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
            return float('nan')
        return float(np.corrcoef(a, b)[0, 1])

    full = pear(np.ones(s.size, dtype=bool))
    filt = pear(np.array([int(m) in kept for m in mi]))
    return (full, filt)


def harvest_t2(runs, vocab_root, data_root, presets, datasets, backbones, folds):
    full_d, filt_d = defaultdict(list), defaultdict(list)
    for preset in presets:
        stem = PRESET_STEM[preset]
        for ds in datasets:
            for bb in backbones:
                for fold in folds:
                    d = cell_dir(runs, preset, ds, fold, bb)
                    if not (d / 'summary_splits.json').exists():
                        continue
                    kept = kept_ids(ds, fold, vocab_root, data_root)
                    full, filt = grouped_pearson(d, stem, kept)
                    if full == full:
                        full_d[(ds, bb, preset)].append(full)
                    if filt == filt:
                        filt_d[(ds, bb, preset)].append(filt)
    return full_d, filt_d


# ── Table 3: instance GT-ROC (source datasets), Full + Filt ──────────────────
def instance_gtroc(d, stem, ds, fold, bb, kept, vocab_root, data_root, processed_root):
    from sklearn.metrics import roc_auc_score
    ei = d / 'explainer_importances.json'
    if not ei.exists():
        return (float('nan'), float('nan'))
    atts_by_split = (json.loads(ei.read_text()).get('importances_by_split') or {})
    from SharedModules.data.vocab import load_vocab
    from SharedModules.data.loader import get_loaders
    vocab = load_vocab(str(vocab_root), ds, 'rbrics')
    loaders, test_ds, _meta = get_loaders(
        dataset=ds, data_root=str(data_root), fold=int(fold), vocab=vocab,
        processed_root=str(processed_root), batch_size=128, normalize=False)
    split_lists = {'train': list(loaders['train'].dataset),
                   'valid': list(loaders['valid'].dataset), 'test': list(test_ds)}
    g_full, g_filt = [], []
    for split in SPLITS:
        sl = split_lists.get(split, [])
        matts = (atts_by_split.get(split) or {}).get(stem) or {}
        for gi, g in enumerate(sl):
            a = matts.get(str(gi), matts.get(gi))
            nl = getattr(g, 'node_label', None)
            if a is None or nl is None:
                continue
            y = np.asarray(nl).reshape(-1).astype(float)
            att = np.asarray(a, dtype=float).reshape(-1)
            if y.shape[0] != att.shape[0]:
                continue
            if np.unique(y).size > 1:
                g_full.append(roc_auc_score(y, att))
            n2m = getattr(g, 'nodes_to_motifs', None)
            if n2m is not None:
                mm = np.asarray(n2m).reshape(-1)
                mask = np.array([int(x) in kept for x in mm])
                if mask.sum() > 0 and np.unique(y[mask]).size > 1:
                    g_filt.append(roc_auc_score(y[mask], att[mask]))
    full = float(np.mean(g_full)) if g_full else float('nan')
    filt = float(np.mean(g_filt)) if g_filt else float('nan')
    return (full, filt)


def harvest_t3(runs, vocab_root, data_root, processed_root, presets, backbones, folds):
    full_d, filt_d = defaultdict(list), defaultdict(list)
    for preset in presets:
        stem = PRESET_STEM[preset]
        for ds in SOURCE:
            for bb in backbones:
                for fold in folds:
                    d = cell_dir(runs, preset, ds, fold, bb)
                    if not (d / 'explainer_importances.json').exists():
                        continue
                    kept = kept_ids(ds, fold, vocab_root, data_root)
                    full, filt = instance_gtroc(d, stem, ds, fold, bb, kept,
                                                vocab_root, data_root, processed_root)
                    if full == full:
                        full_d[(ds, bb, preset)].append(full)
                    if filt == filt:
                        filt_d[(ds, bb, preset)].append(filt)
    return full_d, filt_d


# ── LaTeX emit (rows = leading cols + 9 preset cells; per-row sig group) ──────
def emit(path, csv_path, lead_names, rows, presets, higher_better=True, scale=100.0, dp=1):
    """rows: list of (lead_values:tuple, hb:bool|None, {preset:(mean,std,n)})."""
    ncol = len(presets)
    tex = ['% ' + path.name,
           '\\begin{tabular}{' + 'l' * len(lead_names) + 'c' * ncol + '}',
           '\\toprule',
           ' & '.join(lead_names + [PRESET_LABEL[p] for p in presets]) + ' \\\\',
           '\\midrule']
    csv_rows = []
    for lead, hb, cells_by_preset in rows:
        hbv = higher_better if hb is None else hb
        cells = [{'key': p, 'mean': cells_by_preset.get(p, (float('nan'), 0, 0))[0],
                  'std': cells_by_preset.get(p, (float('nan'), 0, 0))[1],
                  'n': cells_by_preset.get(p, (float('nan'), 0, 0))[2]} for p in presets]
        tags = TS.decide(cells, higher_better=hbv)
        out = []
        for p in presets:
            m, s, n = cells_by_preset.get(p, (float('nan'), 0.0, 0))
            out.append(TS.wrap(fmt(m, s, scale, dp), tags.get(p, 'plain')))
            csv_rows.append({**{ln: lv for ln, lv in zip(lead_names, lead)},
                             'preset': p, 'mean': m, 'std': s, 'n': n})
        tex.append(' & '.join(list(lead) + out) + ' \\\\')
    tex += ['\\bottomrule', '\\end{tabular}']
    path.write_text('\n'.join(tex) + '\n')
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f'  wrote {path.name} ({len(rows)} rows) + {csv_path.name}')


def rows_perfview(dfull, dfilt, datasets, backbones, presets, with_view):
    """Build emit-rows for a Full/Filt metric. with_view=False -> Full only, no view col."""
    rows = []
    for ds in datasets:
        for bb in backbones:
            if with_view:
                for view, dd in (('Full', dfull), ('Filt', dfilt)):
                    cells = {p: agg(dd.get((ds, bb, p), [])) for p in presets}
                    rows.append(((DS_LABEL[ds], bb, view), True, cells))
            else:
                cells = {p: agg(dfull.get((ds, bb, p), [])) for p in presets}
                rows.append(((DS_LABEL[ds], bb), True, cells))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', required=True, help='base_runs root')
    ap.add_argument('--vocab_root', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--processed_root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tables', default='1,2,3')
    ap.add_argument('--smoke', action='store_true',
                    help='tiny slice: 2 presets, 1 ds/group, GIN, folds 0-1')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    want = set(a.tables.split(','))

    presets = PRESETS if not a.smoke else ['base_gsat', 'readout']
    backbones = BACKBONES if not a.smoke else ['GIN']
    folds = FOLDS if not a.smoke else [0, 1]
    ds_all = DATASETS if not a.smoke else ['BBBP', 'Benzene_Verified_GT']
    ds_src = SOURCE if not a.smoke else ['Benzene_Verified_GT']

    if '1' in want:
        print('== T1 task performance (AUC + RMSE, split by scale) ==')
        d = harvest_t1(a.runs, presets, ds_all, backbones, folds)
        # AUC table: classification, x100 (%), higher-better
        cls_ds = [x for x in ds_all if x in CLS]
        if cls_ds:
            rows = [((DS_LABEL[ds], bb), True,
                     {p: agg(d.get((ds, bb, p), [])) for p in presets})
                    for ds in cls_ds for bb in backbones]
            emit(out / 't1_auc.tex', out / 't1_auc.csv', ['Dataset', 'Backbone'],
                 rows, presets, higher_better=True, scale=100.0, dp=1)
        # RMSE table: regression, original units (x1), lower-better
        reg_ds = [x for x in ds_all if x in REG]
        if reg_ds:
            rows = [((DS_LABEL[ds], bb), False,
                     {p: agg(d.get((ds, bb, p), [])) for p in presets})
                    for ds in reg_ds for bb in backbones]
            emit(out / 't1_rmse.tex', out / 't1_rmse.csv', ['Dataset', 'Backbone'],
                 rows, presets, higher_better=False, scale=1.0, dp=3)

    if '2' in want:
        print('== T2 grouped Pearson (Full/Filt) ==')
        df, dfl = harvest_t2(a.runs, a.vocab_root, a.data_root, presets, ds_all, backbones, folds)
        rows = rows_perfview(df, dfl, ds_all, backbones, presets, with_view=True)
        emit(out / 't2_grouped_pearson.tex', out / 't2_grouped_pearson.csv',
             ['Dataset', 'Backbone', 'View'], rows, presets)

    if '3' in want:
        print('== T3 instance GT-ROC, source (Full/Filt) ==')
        df, dfl = harvest_t3(a.runs, a.vocab_root, a.data_root, a.processed_root,
                             presets, backbones, folds)
        rows = rows_perfview(df, dfl, ds_src, backbones, presets, with_view=True)
        emit(out / 't3_instance_gtroc.tex', out / 't3_instance_gtroc.csv',
             ['Dataset', 'Backbone', 'View'], rows, presets)

    print(f'done -> {out}')


if __name__ == '__main__':
    main()
