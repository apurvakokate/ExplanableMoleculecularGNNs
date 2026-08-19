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
FIXPOINT   = os.environ.get('SFO_FIXPOINT', '1') != '0'

METHODS = {'size_frequency_optimization': 'sfo'}


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
    # ── corpus statistics, computed ONCE, before any selection ──────────────
    rows, _sups = _X.aggregate(okr, 1.0, 1.0, 0.0, 0.0,
                               w_form=W_FORM, s_form=S_FORM, struct_form=STRUCT)
    W_by_key = {r['key']: r['W'] for r in rows}
    s_min, pct, n_tv = _s_min(dataset, groups)
    supported = {r['key']: (r['support'] >= s_min) for r in rows}

    if verbose:
        print(f'    [SFO] dataset={dataset or "?"} mols={len(okr)}/{len(smiles_all)} '
              f'candidates={len(rows)} S_min={s_min} (pct={pct} n_tv={n_tv}) '
              f'supported_keys={sum(supported.values())} '
              f'w={W_FORM}/{S_FORM}/{STRUCT} lam={LAM}', flush=True)

    # ── one exact-cover ILP per molecule ────────────────────────────────────
    out, n_fail, n_gated, n_cand = [], 0, 0, 0
    by_smiles = {r['smiles']: r for r in okr}
    for smi in smiles_all:
        r = by_smiles.get(smi)
        if r is None:
            out.append([]); continue
        mask, clauses = _X.eligibility(r['cand_list'], supported)
        n_gated += clauses.get('0_gated', 0); n_cand += len(mask)
        got = _X.solve_partition(r['blocks'], r['cand_list'], W_by_key,
                                 r['n_atoms'], LAM, mask=mask)
        if got is None:
            out.append([]); n_fail += 1; continue
        sel, _val = got
        frags = []
        for j in sel:
            bidx, key = r['cand_list'][j]
            atoms = set()
            for i in bidx:
                atoms |= set(r['blocks'][i])
            frags.append((key, atoms))
        # hard guarantees, asserted per molecule (spec section 16)
        cover = set()
        for _k, a in frags:
            assert not (cover & a), f'overlapping fragments in {smi}'
            cover |= a
        assert cover == set(range(r['n_atoms'])), f'incomplete cover in {smi}'
        out.append(frags)

    if verbose:
        print(f'    [SFO] solved={len(smiles_all)-n_fail}/{len(smiles_all)} '
              f'infeasible={n_fail} gated={100.0*n_gated/max(n_cand,1):.1f}% of candidates',
              flush=True)
    return out
