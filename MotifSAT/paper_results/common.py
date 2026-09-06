#!/usr/bin/env python3
"""common.py — shared logic for the MotifSAT base-runs paper tables.

Single source of truth for constants + metric primitives, used by:
  harvest.py     (parallel, one task per (dataset, fold) — data loaded ONCE)
  build_tables.py(reduce: partial CSVs -> 3 .tex/.csv tables)
  make_tables.py (sequential all-in-one; smoke/reference)

Filtered = STRICT per-fold kept set, mirroring analysis/evaluate.py verbatim
(build_fold_annotation on the rbrics_filter vocab + the fold CSV). rbrics and
rbrics_filter share a motif-id space, so kept ids apply directly to rbrics runs.
Both grouped-Pearson and instance-GT-ROC are pooled over ALL splits and averaged
over FOLDS only. Highlighting reuses analysis/table_sig.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
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


# ── Table 1: task performance ────────────────────────────────────────────────
def read_task_perf(d, stem, ds):
    """test AUC (classification) / RMSE_orig (regression); NaN if absent."""
    metric = 'auc' if ds in CLS else 'rmse_orig'
    p = Path(d) / 'summary_splits.json'
    if not p.exists():
        return float('nan')
    try:
        return float(json.loads(p.read_text())[stem]['test'][metric])
    except Exception:
        return float('nan')


# ── Table 2: grouped Pearson (unweighted), Full + Filt ───────────────────────
def _motif_csv(d, stem, split, kind):
    p = Path(d) / f'{stem}_{kind}_{split}.csv'
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


# ── Table 3: instance GT-ROC — data load SPLIT OUT so it is reused per cell ──
def load_fold_split_lists(ds, fold, vocab_root, data_root, processed_root):
    """Reload the graphs ONCE for a (dataset, fold), in the SAME split-local order
    the atts were keyed by at train time. Carries node_label + nodes_to_motifs."""
    from SharedModules.data.vocab import load_vocab
    from SharedModules.data.loader import get_loaders
    vocab = load_vocab(str(vocab_root), ds, 'rbrics')
    loaders, test_ds, _meta = get_loaders(
        dataset=ds, data_root=str(data_root), fold=int(fold), vocab=vocab,
        processed_root=str(processed_root), batch_size=128, normalize=False)
    return {'train': list(loaders['train'].dataset),
            'valid': list(loaders['valid'].dataset), 'test': list(test_ds)}


def gtroc_from_atts(d, stem, split_lists, kept):
    """Per-graph node GT-ROC pooled over all splits -> (full, filt). Uses cached
    split_lists (node_label / nodes_to_motifs) + this cell's cached atts."""
    from sklearn.metrics import roc_auc_score
    ei = Path(d) / 'explainer_importances.json'
    if not ei.exists():
        return (float('nan'), float('nan'))
    atts_by_split = (json.loads(ei.read_text()).get('importances_by_split') or {})
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


# ── aggregation / formatting / LaTeX emit ────────────────────────────────────
def agg(vals):
    a = np.array([v for v in vals if v == v], dtype=float)
    if a.size == 0:
        return (float('nan'), 0.0, 0)
    return (float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0, int(a.size))


def fmt(mean, std, scale=100.0, dp=1):
    if mean != mean:
        return '--'
    return f'${mean*scale:.{dp}f}\\pm{std*scale:.{dp}f}$'


def emit(tex_path, csv_path, lead_names, rows, presets,
         higher_better=True, scale=100.0, dp=1):
    """rows: list of (lead_values:tuple, hb:bool|None, {preset:(mean,std,n)}).
    Per-row significance group across the preset columns (table_sig)."""
    ncol = len(presets)
    tex = ['% ' + Path(tex_path).name,
           '\\begin{tabular}{' + 'l' * len(lead_names) + 'c' * ncol + '}',
           '\\toprule',
           ' & '.join(lead_names + [PRESET_LABEL[p] for p in presets]) + ' \\\\',
           '\\midrule']
    csv_rows = []
    for lead, hb, cells_by_preset in rows:
        hbv = higher_better if hb is None else hb
        cells = [{'key': p,
                  'mean': cells_by_preset.get(p, (float('nan'), 0, 0))[0],
                  'std': cells_by_preset.get(p, (float('nan'), 0, 0))[1],
                  'n': cells_by_preset.get(p, (float('nan'), 0, 0))[2]}
                 for p in presets]
        tags = TS.decide(cells, higher_better=hbv)
        out = []
        for p in presets:
            m, s, n = cells_by_preset.get(p, (float('nan'), 0.0, 0))
            out.append(TS.wrap(fmt(m, s, scale, dp), tags.get(p, 'plain')))
            csv_rows.append({**{ln: lv for ln, lv in zip(lead_names, lead)},
                             'preset': p, 'mean': m, 'std': s, 'n': n})
        tex.append(' & '.join(list(lead) + out) + ' \\\\')
    tex += ['\\bottomrule', '\\end{tabular}']
    Path(tex_path).write_text('\n'.join(tex) + '\n')
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f'  wrote {Path(tex_path).name} ({len(rows)} rows) + {Path(csv_path).name}')
