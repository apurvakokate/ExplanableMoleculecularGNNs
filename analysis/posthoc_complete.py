#!/usr/bin/env python3
"""ONE definition of "this method is done for this cell".

Used by three callers so they can never disagree:
  1. eval_driver_posthoc  --skip_done      (decide whether to run)
  2. the runner's post-flight assertion    (did it actually land?)
  3. tracker.py                            (campaign progress)

Done means, for method m in destination dir d:
  - all 11 output files exist and are non-empty
  - the method's fitted checkpoint exists (pgexplainer_mlp.pt / mage_attention.pt)
  - {stem}_importance_test.csv parses and holds MORE THAN ONE distinct score

The last clause is deliberate: a collapsed explainer writes a full, well-formed file set
whose scores are all identical. Treating that as done is how 609 PGExplainer cells were
counted as finished work.
"""
import os, csv, math, json

SPLITS = ('train', 'valid', 'test')
_ATT_CACHE = {}


def _test_atts(dest_dir, stem):
    """Per-node test attributions for `stem`, from explainer_importances.json.
    Parsed at most once per directory (the file is multi-MB) and only reached after
    the cheap checks pass, so a MISSING/PARTIAL cell never pays for it."""
    d = str(dest_dir)
    if d not in _ATT_CACHE:
        p = os.path.join(d, 'explainer_importances.json')
        try:
            _ATT_CACHE[d] = (json.load(open(p)).get('importances_by_split') or {}).get('test') or {}
        except Exception:
            _ATT_CACHE[d] = {}
    return _ATT_CACHE[d].get(stem) or {}
DISK_STEM = {'gnnexplainer': 'gnnexplainer', 'motif_occlusion': 'motif_occlusion',
             'pgexplainer': 'pgexplainer', 'mage': 'mage_v2', 'mage_v2': 'mage_v2'}
CKPT = {'pgexplainer': 'pgexplainer_mlp.pt', 'mage': 'mage_attention.pt',
        'mage_v2': 'mage_attention.pt'}


def required_files(method):
    stem = DISK_STEM.get(method, method)
    f = [f'{stem}_importance_{s}.csv' for s in SPLITS]
    f += [f'{stem}_impact_{s}.csv' for s in SPLITS]
    f += [f'{stem}_instance_corr_{s}.csv' for s in SPLITS]
    f += [f'{stem}_grouped_corr_pooled_alltest.csv', f'{stem}_grouped_corr_pooled_testonly.csv']
    if method in CKPT:
        f.append(CKPT[method])
    return f


def method_state(dest_dir, method):
    """-> 'VALID' | 'DEGENERATE' | 'PARTIAL' | 'MISSING'"""
    d = str(dest_dir)
    if not os.path.isdir(d):
        return 'MISSING'
    need = required_files(method)
    got = [f for f in need
           if os.path.exists(os.path.join(d, f)) and os.path.getsize(os.path.join(d, f)) > 0]
    if not got:
        return 'MISSING'
    if len(got) != len(need):
        return 'PARTIAL'
    stem = DISK_STEM.get(method, method)
    try:
        rows = list(csv.DictReader(open(os.path.join(d, f'{stem}_importance_test.csv'))))
    except Exception:
        return 'DEGENERATE'
    if not rows:
        return 'DEGENERATE'
    vals = set()
    for r in rows:
        try:
            v = float(r.get('score'))
        except (TypeError, ValueError):
            return 'DEGENERATE'
        if math.isnan(v):
            return 'DEGENERATE'
        vals.add(round(v, 12))
    if len(vals) <= 1:
        return 'DEGENERATE'
    # Per-ATOM attributions are what GT-ROC consumes. A cell can have varying MOTIF
    # scores yet flat per-atom attributions -- measured on 7 PGExplainer cells -- and is
    # useless for the metric. Checked last so only candidate-VALID dirs pay the JSON read.
    atts = _test_atts(d, stem)
    if not atts:
        return 'DEGENERATE'
    const = 0
    for _, arr in atts.items():
        fin = [x for x in arr if x is not None and not (isinstance(x, float) and math.isnan(x))]
        if not fin or max(fin) == min(fin):
            const += 1
    if const / len(atts) > 0.99:
        return 'DEGENERATE'
    return 'VALID'


def methods_complete(dest_dir, methods):
    """True only if EVERY requested method is VALID here."""
    return all(method_state(dest_dir, m) == 'VALID' for m in methods)
