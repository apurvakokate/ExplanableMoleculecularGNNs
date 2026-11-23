#!/usr/bin/env python3
"""evaluate_gtroc_bysplit.py — evaluate_gtroc_bygroup.py plus the extended-rule GT-ROC per split.

Same command line, same reads, same checks as analysis/evaluate_gtroc_bygroup.py (which is
analysis/evaluate_gtroc.py with the class-1-only-motif columns); that script's eval_one runs
first and nothing it returns is changed. Pure read: no model is loaded and the only file written
is --dest.

COLUMNS ADDED (cond in full, filt; S in train, valid, test, all)
  n_<cond>_<S>       rule-positive molecules scored in the split
  gtx_<cond>_<S>     their mean GT-ROC against the rule EXTENDED by OR with the class-1-only
                     motifs (definition in evaluate_gtroc_bygroup.py); a molecule that holds no
                     such motif keeps its rule value
  nwith_<cond>_<S>   how many of them hold a class-1-only motif
CHECK, every run, both conditions, every split: the mean of the per-molecule rule values of a
split equals the gtroc_<cond>_<S> that evaluate_gtroc.py returns.
"""
import numpy as np

import analysis.evaluate_gtroc_bygroup as G

E = G.E
_PREV_EVAL_ONE = E.eval_one          # the by-group eval_one


def eval_one(method, run_dir, art_dir, meta, args, device, kept):
    row = _PREV_EVAL_ONE(method, run_dir, art_dir, meta, args, device, kept)
    split_lists, gt = E._loaders_for(meta, args)
    atts = E._atts_read(art_dir, method)
    gl, abi = E._combined(gt, atts, split_lists)
    bounds, off = {}, 0
    for s in E.SPLITS:                               # _combined lays the splits out in this order
        n = len(gt.get(s) or [])
        bounds[s] = (off, off + n); off += n
    info = G._groups(G._base_graphs(meta, args, gl))
    for cond, unk in (('full', 'include'), ('filt', 'exclude')):
        keep_fn = G._keep_fn(kept if cond == 'filt' else None, unk)
        inst, ext, _pvr, group = G._by_group(abi, gl, keep_fn, info)
        x = {i: ext.get(i, v) for i, v in inst.items()}
        for s in E.SPLITS:
            lo, hi = bounds[s]
            idx = [i for i in inst if lo <= i < hi]
            ref = row[f'gtroc_{cond}_{s}']
            got = G._mean(inst[i] for i in idx)
            if not np.isclose(got, ref, rtol=0, atol=1e-9, equal_nan=True):
                raise ValueError(f'per-molecule mean {got} != gtroc_{cond}_{s} {ref}')
            row[f'n_{cond}_{s}'] = len(idx)
            row[f'gtx_{cond}_{s}'] = G._mean(x[i] for i in idx)
            row[f'nwith_{cond}_{s}'] = sum(1 for i in idx if group[i])
        row[f'n_{cond}_all'] = len(inst)
        row[f'gtx_{cond}_all'] = G._mean(x.values())
        row[f'nwith_{cond}_all'] = sum(1 for i in inst if group[i])
    return row


E.eval_one = eval_one

if __name__ == '__main__':
    E.main()
