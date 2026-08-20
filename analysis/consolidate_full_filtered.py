#!/usr/bin/env python3
"""consolidate_full_filtered.py — STEP 1 of the full/filtered metric assembly.

Reads the harvested per-fold metrics (build_metric_set output) and MERGES the
offline filtered-grouped GSAT/MotifSAT re-score, producing an augmented per-fold
CSV where `filtered` (False=full / True=filtered) is a complete axis for grouped
Pearson across ALL methods. Prints the coverage matrix (present / offline-to-do /
needs-recompute) so the remaining gaps are explicit.

What this DOES fill (offline, no model):
  - grouped Pearson/Spearman, filtered, for GSAT & MotifSAT (from the re-score).
Everything already present is harvested untouched:
  - baselines: full+filtered (both runs exist); MoSE: filtered; GSAT/MotifSAT: full.
Still MISSING after this (later steps):
  - MotifSAT filtered GT-ROC  -> offline broadcast (step 1b)
  - GSAT filtered GT-ROC      -> node_att_soft dump (step 2)
  - GSAT/MotifSAT filtered instance Pearson -> per-(graph,motif) recompute (step 2)

  python analysis/consolidate_full_filtered.py \
      --per-fold-dir offline_results/latex_results_v2/per_fold \
      --filtered-grouped offline_results/latex_results_v2/filtered_gsat_motifsat/gsat_motifsat_filtered_grouped.csv \
      --out-dir offline_results/latex_results_v2/consolidated_v1
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FRAG_MAP = {'rdkit_fg_first': 'FGFirst-RDKIT', 'fg_first': 'custom-FG',
            'ertl_first': 'ertl', 'rbrics': 'rbrics'}
MODEL_MAP = {'gsat': 'GSAT', 'motifsat': 'MotifSAT'}
GROUPED = [f'grouped_{s}_agnostic_{w}_{u}unk'
           for s in ('pearson', 'spearman') for w in ('w', 'u') for u in ('incl', 'excl')]
KEY = ['dataset', 'backbone', 'fold', 'fragmentation', 'gt_rule', 'model']


def _norm_rule(x):
    return '' if (pd.isna(x) or x == '') else str(x)


def inject_filtered_grouped(pf: pd.DataFrame, fg: pd.DataFrame, model_map: dict,
                            tag: str) -> pd.DataFrame:
    """Create filtered=True rows carrying the re-score's grouped values, by cloning
    the matching full-vocab (filtered=False) rows and overwriting the grouped columns.
    `model_map` maps re-score model names to the per-fold names (identity for
    already-canonical baseline names). Self-checks re-score _full vs harvested full."""
    fg = fg.copy()
    fg['model'] = fg['model'].map(lambda m: model_map.get(m, m))
    fg['fragmentation'] = fg['fragmentation'].map(FRAG_MAP)
    fg['gt_rule'] = fg['gt_rule'].map(_norm_rule)
    pf = pf.copy(); pf['gt_rule'] = pf['gt_rule'].map(_norm_rule)
    models = set(fg['model'])

    base = pf[(pf.model.isin(models)) & (~pf.filtered)].copy()
    chk = base.merge(fg, on=KEY, suffixes=('', '_rs'))
    d = (chk['grouped_pearson_agnostic_w_exclunk'] - chk['grouped_pearson_agnostic_w_full']).abs()
    d = d[np.isfinite(d)]
    if len(d):
        print(f'[self-check {tag}] re-score _full vs harvested full grouped_pearson_w: '
              f'n={len(d)} max|diff|={d.max():.2e}')

    full_rows = pf[(pf.model.isin(models)) & (~pf.filtered)].copy()
    new = full_rows.merge(fg[KEY], on=KEY, how='inner').copy()   # matched full rows
    new = new.reset_index(drop=True)
    if new.empty:
        return new
    fgm = fg.set_index(KEY)
    idx = pd.MultiIndex.from_frame(new[KEY])
    for s in ('pearson', 'spearman'):
        for w in ('w', 'u'):
            val = fgm.loc[idx, f'grouped_{s}_agnostic_{w}_filtered'].values
            # kept ids are all >=0 -> incl == excl
            new[f'grouped_{s}_agnostic_{w}_exclunk'] = val
            new[f'grouped_{s}_agnostic_{w}_inclunk'] = val
    new['filtered'] = True
    # everything NOT grouped is not available filtered for these models yet -> NaN
    keepcols = set(KEY + ['model_type', 'impact_basis', 'gt_tier', 'synthetic',
                          'n_folds', 'run_path', 'filtered'] + GROUPED
                   + ['grouped_pearson_agnostic', 'grouped_spearman_agnostic'])
    for c in new.columns:
        if c not in keepcols and pd.api.types.is_numeric_dtype(new[c]):
            new[c] = np.nan
    return new


def coverage(df: pd.DataFrame):
    """method x filtered presence for each metric family (fraction of cells non-null)."""
    def present(sub, cols):
        cc = [c for c in cols if c in sub.columns]
        if not cc:
            return 0.0
        return float(sub[cc].notna().any(axis=1).mean())
    fams = {
        'grouped_pearson': ['grouped_pearson_agnostic_w_exclunk'],
        'instance_pearson': [f'instance_pearson_agnostic_{s}' for s in ('train', 'valid', 'test')],
        'gtroc_global': [f'grouped_gtroc_fired_{s}' for s in ('train', 'valid', 'test')],
        'gtroc_instance': [f'instance_gtroc_dnf_{s}' for s in ('train', 'valid', 'test')],
    }
    rows = []
    for model in ['GNNExplainer', 'PGExplainer', 'MAGE', 'MotifOcclusion',
                  'GSAT', 'MotifSAT', 'MoSE']:
        for filt in (False, True):
            sub = df[(df.model == model) & (df.filtered == filt)]
            if sub.empty:
                continue
            r = dict(model=model, vocab=('filt' if filt else 'full'), n=len(sub))
            for fam, cols in fams.items():
                r[fam] = round(present(sub, cols), 2)
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-fold-dir', required=True)
    ap.add_argument('--filtered-grouped', required=True, help='gsat/motifsat re-score CSV')
    ap.add_argument('--baselines-planted-grouped', default=None,
                    help='baselines-planted re-score CSV (optional)')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
    fg = pd.read_csv(args.filtered_grouped)                     # gsat/motifsat
    bp = pd.read_csv(args.baselines_planted_grouped) if args.baselines_planted_grouped else None

    for tag in ('main', 'supp'):
        pf = pd.read_csv(Path(args.per_fold_dir) / f'{tag}_metrics_per_fold.csv', low_memory=False)
        adds = [inject_filtered_grouped(pf, fg, MODEL_MAP, 'gsat/motifsat')]
        if bp is not None:                                     # baselines already have canonical names
            adds.append(inject_filtered_grouped(pf, bp, {}, 'baselines-planted'))
        add = pd.concat([a for a in adds if not a.empty], ignore_index=True) if adds else pd.DataFrame()
        con = pd.concat([pf, add], ignore_index=True)
        con.to_csv(outd / f'{tag}_metrics_per_fold_consolidated.csv', index=False)
        print(f'\n=== {tag}: {len(pf)} -> {len(con)} rows (+{len(add)} injected filtered grouped) ===')
        print(coverage(con).to_string(index=False))
    print(f'\nwrote consolidated per-fold CSVs into {outd}')


if __name__ == '__main__':
    main()
