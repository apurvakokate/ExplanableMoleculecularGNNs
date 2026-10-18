#!/usr/bin/env python3
"""size_frequency_optimization.py -- SFO fragmentation arm, pipeline adapter.

Exposes the standard fragmentation interface used by generate_vocab_rules.py:

    build(smiles_all, groups=None, verbose=True) -> [[(motif_key, atom_set), ...], ...]

The algorithm itself lives in xcover/exp1_candidate_inventory.py; this module is only
the corpus driver + the pipeline contract, so the two cannot drift.

METHOD, as settled:
  1 four OFF-THE-SHELF fragmenters (RDKit BRICS, RDKit RECAP incl. the repaired
    ether/amine/urea rules, rBRICS, CRUSH) propose acyclic cut bonds
  2 UNION of all cut sets -> one common finest segmentation (blocks), iterated to a
    fixpoint by re-passing blocks through the fragmenters
  3 candidates = every connected union of blocks (caps: 12 blocks / 24 heavy atoms)
  4 corpus statistics computed ONCE over candidates, BEFORE any selection
        W(F) = S_structure(F) * S_support(key(F))          [multiplicative]
        S_support floor-rescaled so support-1 scores EXACTLY 0
        S_structure size-free: non-triviality floor x boundary quality
  5 MOTIF ELIGIBILITY (spec section 15): a candidate may be a final motif only if
        supported (>= S_min molecules)  OR  irreducible (single block)
        OR  no proper sub-candidate of it is supported
    S_min is the SAME cutoff the downstream vocabulary applies
    (SharedModules/data/threshold_config.py), so eligibility means
    "would survive the vocabulary threshold" -- it is NOT a new hyperparameter.
  6 one exact-cover ILP per molecule (scipy/HiGHS), lambda = 0:
        max SUM_j W(F_j) x_j   s.t.   SUM_{j: a in F_j} x_j == 1  for every atom a

KEYING: whole ring systems keep the substituent-agnostic 'ring:' key (project
standard); every other motif gets the L2 key (isomeric SMILES with [*] dummies +
sorted external boundary context).  Keys are FINAL -- do NOT pass this module's
output through fg_first_frag.rekey_structural, which would overwrite the boundary
context with a plain frag_key.

NOT DONE HERE (deliberate, unresolved): ring-system peeling.  Ring integrity is a
hard guarantee, so large fused systems stay whole even when that costs vocabulary
size.  The baselines being compared against DO cut into ring systems.
"""
from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_XC = os.path.join(_HERE, 'xcover')
if _XC not in sys.path:
    sys.path.insert(0, _XC)

import exp1_candidate_inventory as _X          # the algorithm; single source of truth

# Settled configuration.  Overridable by env var only, so a run's provenance is
# explicit in the launch command rather than hidden in a default.
MAX_BLOCKS = int(os.environ.get('SFO_MAX_BLOCKS', 12))
MAX_HEAVY  = int(os.environ.get('SFO_MAX_HEAVY', 24))
MAX_CANDS  = int(os.environ.get('SFO_MAX_CANDS', 20000))
W_FORM     = os.environ.get('SFO_W_FORM', 'multiplicative')
S_FORM     = os.environ.get('SFO_S_FORM', 'log_floor')
STRUCT     = os.environ.get('SFO_STRUCT', 'sizefree')
LAM        = float(os.environ.get('SFO_LAM', 0.0))
WORKERS    = int(os.environ.get('SFO_WORKERS', os.cpu_count() or 1))
FIXPOINT   = os.environ.get('SFO_FIXPOINT', '1') != '0'      # segmentation fixpoint (SECTION 3)
EM_ROUNDS  = int(os.environ.get('SFO_EM_ROUNDS', 1))         # support-fixpoint rounds (1 = OFF, current)
PRUNE_DOM  = os.environ.get('SFO_PRUNE_DOMINATED', '0') not in ('0', '', 'false', 'no')

METHODS = {'size_frequency_optimization': 'sfo'}

# Per-round context for the parallel solve workers.  On Linux ('fork') a pool created
# AFTER this dict is updated inherits the current W/supported, so the large W dict is
# never pickled per task -- only the (small) molecule record is.
_CUR = {}


def _solve_one(rec):
    """Solve one molecule's exact-cover ILP with the current-round W in _CUR.
    Returns [(key, atom_set), ...]; [] on missing molecule or infeasible ILP."""
    if rec is None:
        return []
    W = _CUR['W']
    mask, _clauses = _X.eligibility(rec['cand_list'], _CUR['supported'])
    if _CUR['prune']:
        mask = _prune_dominated(rec, W, mask)
    got = _X.solve_partition(rec['blocks'], rec['cand_list'], W, rec['n_atoms'], LAM, mask=mask)
    if got is None:
        return []
    sel, _val = got
    frags = []
    for j in sel:
        bidx, key = rec['cand_list'][j]
        atoms = set()
        for i in bidx:
            atoms |= set(rec['blocks'][i])
        frags.append((key, atoms))
    cover = set()                                    # hard guarantees (spec section 16)
    for _k, a in frags:
        assert not (cover & a), f"overlapping fragments in {rec['smiles']}"
        cover |= a
    assert cover == set(range(rec['n_atoms'])), f"incomplete cover in {rec['smiles']}"
    return frags


def _prune_dominated(rec, W, mask):
    """Drop candidate F when covering F's atoms with its OWN sub-pieces already scores
    >= W(F) (split_value).  Single blocks (irreducible, feasibility fallback) never pruned."""
    cand_list, blocks = rec['cand_list'], rec['blocks']
    m = list(mask)
    for j, (bidx, key) in enumerate(cand_list):
        if not m[j] or len(bidx) == 1:
            continue
        sv = _X.split_value(blocks, cand_list, W, m, j)
        if sv is not None and W.get(key, 0.0) < sv:
            m[j] = False
    return m


def _solve_all(records):
    """Solve every molecule with the current _CUR (parallel across molecules)."""
    if WORKERS > 1:
        import multiprocessing as mp
        with mp.Pool(WORKERS) as pool:               # fork inherits _CUR set by the caller
            return pool.map(_solve_one, records, chunksize=16)
    return [_solve_one(r) for r in records]


def _selected_support(out):
    """key -> number of DISTINCT molecules that SELECTED it this round."""
    from collections import Counter
    c = Counter()
    for frags in out:
        for k in {key for key, _a in frags}:
            c[k] += 1
    return c


def _recompute_W(sel_sup, struct, cand_W):
    """W from SELECTED support (support-fixpoint update).  Keys not selected last round
    keep their candidate-prior W, so they stay available for coverage (never zeroed)."""
    max_sup = max(sel_sup.values()) if sel_sup else 1
    _lo = math.log1p(1)
    denom = (math.log1p(max_sup) - _lo) if S_FORM == 'log_floor' else math.log1p(max_sup)
    if denom <= 0:
        denom = 1.0
    W = {}
    for key, (n_atoms, n_attach) in struct.items():
        sup = sel_sup.get(key, 0)
        if sup <= 0:
            W[key] = cand_W.get(key, 0.0)            # not selected -> retain prior (coverage fallback)
            continue
        s_sup = ((math.log1p(sup) - _lo) / denom) if S_FORM == 'log_floor' else (math.log1p(sup) / denom)
        s_sup = max(0.0, s_sup)
        s_str = _X.s_structure(n_atoms, n_attach, form=STRUCT)
        W[key] = s_str * s_sup if W_FORM == 'multiplicative' else (s_str + s_sup)
    return W


def _s_min(dataset, groups):
    """S_min = the DOWNSTREAM vocabulary cutoff, not a new knob."""
    n_tv = sum(1 for g in (groups or []) if g in ('training', 'valid')) or len(groups or [1])
    pct = _X.THRESHOLD_PCT.get(dataset, 0.005)
    if os.environ.get('SFO_SMIN_PCT'):
        pct = float(os.environ['SFO_SMIN_PCT'])
    return max(1, int(pct * n_tv)), pct, n_tv


def build(smiles_all, method='size_frequency_optimization', groups=None,
          dataset=None, verbose=True, **_ignored):
    """Corpus -> one exact partition per molecule.  Returns [[(key, atom_set)], ...]."""
    if groups is None:
        raise ValueError('size_frequency_optimization.build requires `groups`: S_min is '
                         'defined on train+val, so the caller must state the splits.')
    if dataset is None:
        dataset = os.environ.get('SFO_DATASET', '')

    work = [(s, _HERE, MAX_BLOCKS, MAX_HEAVY, MAX_CANDS, FIXPOINT) for s in smiles_all]
    if WORKERS > 1:
        import multiprocessing as mp
        with mp.Pool(WORKERS) as pool:
            results = pool.map(_X.process_molecule, work, chunksize=16)
    else:
        results = [_X.process_molecule(w) for w in work]

    okr = [r for r in results if r['ok']]
    by_smiles = {r['smiles']: r for r in okr}
    records = [by_smiles.get(smi) for smi in smiles_all]   # None for unparseable molecules
    # SFO_SUPPORT_SPLIT (opt-in, 2026-10-18): 'all' (DEFAULT = deployed) counts candidate support and every EM
    # round's selected support over ALL molecules passed (test included -> transductive); 'trainval' counts them
    # over train+val molecules only (test molecules are still partitioned, scored with that train+val W).
    split = os.environ.get('SFO_SUPPORT_SPLIT', 'all')
    if split not in ('all', 'trainval'):
        raise ValueError(f"SFO_SUPPORT_SPLIT must be 'all' or 'trainval', got {split!r}")
    if len(groups) != len(smiles_all):
        raise ValueError(f'groups ({len(groups)}) and smiles_all ({len(smiles_all)}) differ in length')
    stat_idx = [i for i, g in enumerate(groups) if split == 'all' or g in ('training', 'valid')]
    if not stat_idx:
        raise ValueError('SFO_SUPPORT_SPLIT=trainval but no training/valid molecules')
    stat_recs = [records[i] for i in stat_idx if records[i] is not None]

    # ── round-0 corpus statistics (CANDIDATE support) ───────────────────────
    rows, _sups = _X.aggregate(stat_recs, 1.0, 1.0, 0.0, 0.0,
                               w_form=W_FORM, s_form=S_FORM, struct_form=STRUCT)
    struct = {r['key']: (r['n_atoms'], r['n_attach']) for r in rows}
    cand_W = {r['key']: r['W'] for r in rows}
    s_min, pct, n_tv = _s_min(dataset, groups)
    supported = {r['key']: (r['support'] >= s_min) for r in rows}

    if verbose:
        print(f'    [SFO] dataset={dataset or "?"} mols={len(okr)}/{len(smiles_all)} '
              f'candidates={len(rows)} S_min={s_min} (pct={pct} n_tv={n_tv}) '
              f'supported_keys={sum(supported.values())} '
              f'w={W_FORM}/{S_FORM}/{STRUCT} lam={LAM} em_rounds={EM_ROUNDS} '
              f'prune={PRUNE_DOM} workers={WORKERS} support_split={split} stat_mols={len(stat_recs)}', flush=True)

    # ── support-fixpoint: solve all molecules (parallel), recompute support from the
    #    SELECTIONS, repeat.  Round 0 uses candidate support; each later round uses the
    #    previous round's selected support.  Constant WITHIN a round (parallel-safe),
    #    updated BETWEEN rounds (reflects the solution) -- not greedy, not frozen. ──
    _CUR['supported'] = supported
    _CUR['prune'] = PRUNE_DOM
    W_by_key = dict(cand_W)
    out, prev_sig, rounds_run = None, None, 0
    for rnd in range(max(1, EM_ROUNDS)):
        _CUR['W'] = W_by_key
        out = _solve_all(records)
        rounds_run = rnd + 1
        if EM_ROUNDS <= 1:
            break
        sel_sup = _selected_support([out[i] for i in stat_idx])
        sig = tuple(sorted(sel_sup.items()))
        if verbose:
            print(f'    [SFO] EM round {rnd}: selected_keys={len(sel_sup)} '
                  f'max_sel_sup={max(sel_sup.values()) if sel_sup else 0}', flush=True)
        if sig == prev_sig:                                # converged
            break
        prev_sig = sig
        W_by_key = _recompute_W(sel_sup, struct, cand_W)

    n_fail = sum(1 for i, f in enumerate(out) if not f and records[i] is not None)
    if verbose:
        print(f'    [SFO] rounds={rounds_run} solved={len(smiles_all)-sum(1 for f in out if not f)}'
              f'/{len(smiles_all)} infeasible={n_fail}', flush=True)
    return out
