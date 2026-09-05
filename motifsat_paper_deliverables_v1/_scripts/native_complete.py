#!/usr/bin/env python3
"""native_complete.py — ONE definition of "this native (MoSE/GSAT/MotifSAT) run is done".

Used by two callers so they can never disagree:
  1. base_worker.sh  — pre-run SKIP check + post-flight assertion
  2. base_status.py  — campaign audit / reset

Done means, for a native run dir `d` with method stem `M` (gsat | motifsat):
  - ALL required artifacts exist and are NON-EMPTY (size > 0), and
  - content checks that a marker file could otherwise pass while the real
    artifacts are absent (the fallback-created-marker case):
      * summary_splits.json parses and holds method M with train/valid/test
      * explainer_importances.json has importances_by_split.test[M] non-empty
      * impact_cache_own_{split}.json parses and is not an empty object

This is deliberately NOT "the marker file exists": run.py writes summary.json
BEFORE the per-split-eval block, and summary_splits.json / explainer_importances.json
are MERGE-written (read-modify-write), so any of them can exist without this run's
full artifact set having landed. Presence+non-empty of the whole set is the gate.

Stdlib only (no torch/pandas) so it runs anywhere, incl. outside the conda env.
"""
import os
import sys
import json

# ── Campaign definition (single source of truth for worker + status) ──────────
# preset (MotifSAT/configs/<preset>.yaml) -> method stem written on disk by
# run.py (`gsat` for base_gsat*, else `motifsat`; see run.py `_method`).
PRESET_STEM = {
    'base_gsat':          'gsat',
    'base_gsat_noanneal': 'gsat',
    'intra_consistency':  'motifsat',
    'inter_loss':         'motifsat',
    'intra_inter_loss':   'motifsat',
    'readout':            'motifsat',
    'readout_inter':      'motifsat',
    'readout_layernorm':  'motifsat',
    'readout_nonorm':     'motifsat',
}
PRESETS = list(PRESET_STEM)
DATASETS = ['BBBP', 'esol', 'Lipophilicity', 'hERG', 'Mutagenicity',
            'Benzene_Verified_GT', 'Alkane_Carbonyl_Verified_GT',
            'Fluoride_Carbonyl_Verified_GT']
FOLDS = [0, 1, 2, 3, 4]
BACKBONES = ['GIN', 'GCN', 'GAT', 'SAGE', 'PNA']
SPLITS = ('train', 'valid', 'test')


def iter_cells():
    """Yield (cell_id, preset, stem, dataset, fold, backbone, rel_dir) for the
    FULL expected grid (9 presets x 8 datasets x 5 folds x 5 backbones = 1800)."""
    for p in PRESETS:
        stem = PRESET_STEM[p]
        for ds in DATASETS:
            for f in FOLDS:
                for bb in BACKBONES:
                    cid = f'{p}__{ds}__f{f}__{bb}'
                    rel = f'{p}/{ds}/fold{f}/{bb}'
                    yield cid, p, stem, ds, f, bb, rel


def required_files(stem):
    """The exact artifact set a completed native run emits (verified against
    run.py + SharedModules/evaluation/split_eval.py)."""
    f = ['best_model.pt', 'summary.json', 'summary_splits.json',
         'explainer_importances.json', 'importance_global.csv', 'impact_global.csv',
         f'{stem}_grouped_corr_pooled_alltest.csv',
         f'{stem}_grouped_corr_pooled_testonly.csv']
    for s in SPLITS:
        f += [f'impact_cache_own_{s}.json',
              f'{stem}_importance_{s}.csv',
              f'{stem}_impact_{s}.csv',
              f'{stem}_instance_corr_{s}.csv']
    return f


def is_complete(cell_dir, stem):
    """-> (ok: bool, reason: str). reason is '' iff ok; else the first defect."""
    d = str(cell_dir)
    if not os.path.isdir(d):
        return False, 'run dir missing'
    # 1) every required file exists AND is non-empty.
    for fn in required_files(stem):
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            return False, f'missing: {fn}'
        if os.path.getsize(p) == 0:
            return False, f'empty: {fn}'
    # 2) summary_splits.json: parses, has method M with all three splits.
    try:
        ss = json.load(open(os.path.join(d, 'summary_splits.json')))
    except Exception as e:
        return False, f'summary_splits.json unparseable ({type(e).__name__})'
    m = ss.get(stem)
    if not isinstance(m, dict):
        return False, f'summary_splits.json missing method {stem!r}'
    miss = [s for s in SPLITS if s not in m]
    if miss:
        return False, f'summary_splits.json[{stem}] missing splits {miss}'
    # 3) explainer_importances.json: test attributions for M are present + non-empty.
    try:
        ei = json.load(open(os.path.join(d, 'explainer_importances.json')))
    except Exception as e:
        return False, f'explainer_importances.json unparseable ({type(e).__name__})'
    atts = ((ei.get('importances_by_split') or {}).get('test') or {}).get(stem)
    if not atts:
        return False, f'explainer_importances.json test[{stem!r}] empty/absent'
    # 4) impact_cache_own_{split}.json: parses, not an empty object.
    for s in SPLITS:
        try:
            ic = json.load(open(os.path.join(d, f'impact_cache_own_{s}.json')))
        except Exception as e:
            return False, f'impact_cache_own_{s}.json unparseable ({type(e).__name__})'
        if not ic:
            return False, f'impact_cache_own_{s}.json is empty object'
    return True, ''


def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cell_dir', nargs='?', help='a native run dir to check')
    ap.add_argument('stem', nargs='?', help='method stem: gsat | motifsat')
    ap.add_argument('--list-presets', action='store_true',
                    help='print "<preset>\\t<stem>" per line')
    ap.add_argument('--list-cells', action='store_true',
                    help='print the full expected cell grid as TSV')
    ap.add_argument('--quiet', action='store_true', help='no stdout, exit code only')
    a = ap.parse_args(argv)

    if a.list_presets:
        for p in PRESETS:
            print(f'{p}\t{PRESET_STEM[p]}')
        return 0
    if a.list_cells:
        for cid, p, stem, ds, f, bb, rel in iter_cells():
            print(f'{cid}\t{p}\t{stem}\t{ds}\t{f}\t{bb}\t{rel}')
        return 0
    if not a.cell_dir or not a.stem:
        print('usage: native_complete.py <cell_dir> <stem> | --list-presets | --list-cells',
              file=sys.stderr)
        return 2
    ok, reason = is_complete(a.cell_dir, a.stem)
    if not a.quiet:
        print('VALID' if ok else f'INCOMPLETE: {reason}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(_main(sys.argv[1:]))
