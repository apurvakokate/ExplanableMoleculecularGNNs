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


def method_state_detail(dest_dir, method):
    """-> (state, reason, n_missing_files).

    state  : 'VALID' | 'DEGENERATE' | 'PARTIAL' | 'MISSING'
    reason : '' unless DEGENERATE/PARTIAL — the specific defect, so a collapsed cell
             can be RECORDED with its cause instead of just being counted.
    """
    d = str(dest_dir)
    if not os.path.isdir(d):
        return 'MISSING', '', 0
    need = required_files(method)
    got = [f for f in need
           if os.path.exists(os.path.join(d, f)) and os.path.getsize(os.path.join(d, f)) > 0]
    if not got:
        return 'MISSING', '', len(need)
    if len(got) != len(need):
        miss = [f for f in need if f not in got]
        return 'PARTIAL', f'{len(miss)}/{len(need)} files absent or empty: ' + ','.join(miss[:3]), len(miss)
    stem = DISK_STEM.get(method, method)
    try:
        rows = list(csv.DictReader(open(os.path.join(d, f'{stem}_importance_test.csv'))))
    except Exception as e:
        return 'DEGENERATE', f'importance_test unreadable ({type(e).__name__})', 0
    if not rows:
        return 'DEGENERATE', 'importance_test has no rows', 0
    vals = set()
    for r in rows:
        try:
            v = float(r.get('score'))
        except (TypeError, ValueError):
            return 'DEGENERATE', 'non-numeric importance score', 0
        if math.isnan(v):
            return 'DEGENERATE', 'NaN importance score', 0
        vals.add(round(v, 12))
    if len(vals) <= 1:
        return 'DEGENERATE', f'constant importance ({vals.pop():.6g} across {len(rows)} motifs)', 0
    # Per-ATOM attributions are what GT-ROC consumes. A cell can have varying MOTIF
    # scores yet flat per-atom attributions -- measured on 7 PGExplainer cells -- and is
    # useless for the metric. Checked last so only candidate-VALID dirs pay the JSON read.
    atts = _test_atts(d, stem)
    if not atts:
        return 'DEGENERATE', 'no test attributions in explainer_importances.json', 0
    const = 0
    for _, arr in atts.items():
        fin = [x for x in arr if x is not None and not (isinstance(x, float) and math.isnan(x))]
        if not fin or max(fin) == min(fin):
            const += 1
    frac = const / len(atts)
    if frac > 0.99:
        return 'DEGENERATE', f'flat per-atom attributions on {const}/{len(atts)} test graphs', 0
    return 'VALID', '', 0


def method_state(dest_dir, method):
    """-> 'VALID' | 'DEGENERATE' | 'PARTIAL' | 'MISSING'"""
    return method_state_detail(dest_dir, method)[0]


def methods_complete(dest_dir, methods):
    """True only if EVERY requested method is VALID here."""
    return all(method_state(dest_dir, m) == 'VALID' for m in methods)
