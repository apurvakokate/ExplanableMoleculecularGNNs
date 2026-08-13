"""ring_grow_bpe.py — rings SEED then GROW via BPE; non-ring starts atomic and merges upward.

Design (as requested): separate the rings first, then run BPE over the whole molecule such that the
RINGS ARE ALLOWED TO GROW, while the non-ring remainder starts from its atomic state and is merged
upward.

Concretely, differing from ring_mdl / conservative_ertl_ring_* (which FREEZE rings so they never
change):
  * Ring systems are detected first (whole fused systems, via ring_mdl.make_partition) and become
    SEED tokens — but they are NOT frozen. A ring may merge with an adjacent group and grow.
  * The non-ring remainder is cut at every single acyclic bond ('all_bonds' finest) so it starts
    atomic (double/triple-bonded pairs such as C=O begin as one unit — "single-bond atomic").
  * There is NO FG-head tier.
  * Merging is greedy BPE. DEFAULT objective = MDL-BPE: accept the adjacent type-pair merge that most
    reduces the two-part description length L = dict_bits + corpus_bits (beta=0, i.e. NO rBRICS
    prior), stop when no merge lowers L. This is self-limiting (no support floor to tune) and is the
    project's stronger linker objective (MDL selection beat frequency-BPE on expressibility/GT-ROC).
    A pure frequency-BPE mode (linker_method='bpe') is also provided for a head-to-head.

Keying: structural frag_key throughout. An UN-grown ring keeps its substituent-agnostic ring-canonical
key (ring:...). The moment a ring grows it is re-keyed by frag_key over ring+neighbour, i.e. it becomes
an attachment-aware frag: motif — substituent-specific BY DESIGN (growth trades ring substituent-
invariance for context). This is the intended departure from the ring-frozen variants.

Implementation: reuses the vetted _MolGraph BPE engine unchanged. _MolGraph(freeze='rings') marks ring
nodes inert; we UN-FREEZE them (g.frozen = all False) so BPE can grow rings while every non-ring node
is atomic. No modification to cascade_bpe_linker.py — the growable behaviour lives entirely here.

Top-level entry: build(smiles_all, head_source='none', linker_method='mdl', break_fused_rings=False,
beta=0.0) -> per-mol [(key, set(atoms))] aligned to smiles_all, IDENTICAL format/signature to
ring_mdl.build / mdl_linker.build so the generate_vocab_rules call site is unchanged. head_source is
accepted for call-compatibility and IGNORED (rings-only seeding; no FG heads).

NO FALLBACK: fails loudly on keying errors (mirrors ring_mdl / mdl_only).
"""
from __future__ import annotations

import math
from collections import Counter

from rdkit import Chem

import chemfrag as cf
import ring_mdl                                  # reuse the rings-only partition (same ring keying)
from cascade_bpe_linker import _MolGraph, _dl


# ───────────────────────────── growable per-molecule graph ─────────────────────
def _grow_graph(mol, part_fn):
    """_MolGraph whose ring nodes are SEEDS but NOT frozen (so BPE can grow them); non-ring nodes are
    atomic (single acyclic bonds cut by finest='all_bonds'). freeze='rings' marks rings inert with the
    ring_mdl partition (no FG heads); we then un-freeze every node so all adjacent pairs may merge."""
    g = _MolGraph(mol, finest='all_bonds', freeze='rings', fg_partition=part_fn)
    g.frozen = [False] * len(g.frozen)           # UN-freeze rings + linker atoms -> everything grows
    return g


# ───────────────────────────── MDL-BPE (default) ───────────────────────────────
def _mdl_bpe_cover(mols, part_fn, beta=0.0, max_atoms=None, max_merges=6000, verbose=False):
    """Greedy MDL-BPE over ALL adjacent type-pairs (rings included). Accept the pair-merge with the
    most-negative estimated Delta L; re-count exactly after applying and stop once L stops dropping.
    Returns per-mol cover [(key, atoms)]. beta=0 -> pure MDL (no rBRICS prior)."""
    graphs = [_grow_graph(m, part_fn) for m in mols]

    def recount():
        tc: Counter = Counter(); ta = {}
        for g in graphs:
            for idt, at in g.fragments():
                tc[idt] += 1; ta[idt] = len(at)
        return tc, ta

    tc, ta = recount()
    L_traj = [_dl(tc, ta, 0, beta)]

    for step in range(max_merges):
        pair_k: Counter = Counter()
        pair_z = {}                              # type-pair -> (merged_key, merged_size)
        for g in graphs:
            for i, j, _is_rb in g.linker_adjacencies():
                p = tuple(sorted((g.ident[i], g.ident[j])))
                pair_k[p] += 1
                if p not in pair_z:
                    merged = g.atoms[i] | g.atoms[j]
                    z = cf.frag_key(g.mol, merged)
                    if z is None:
                        raise ValueError(
                            f"ring_grow_bpe: frag_key None for merged atoms {sorted(merged)} "
                            f"in {Chem.MolToSmiles(g.mol)!r}")
                    pair_z[p] = (z, len(merged))
        if not pair_k:
            break

        L0 = _dl(tc, ta, 0, beta)

        def est_dL(p):
            k = pair_k[p]; (a, b) = p; z, az = pair_z[p]
            nc = tc.copy()
            if a == b:
                nc[a] = tc[a] - 2 * k
            else:
                nc[a] -= k; nc[b] -= k
            nc[z] = nc.get(z, 0) + k
            nc = Counter({t: v for t, v in nc.items() if v > 0})
            nta = dict(ta); nta[z] = az
            return _dl(nc, nta, 0, beta) - L0

        best_p, best_dL = None, -1e-9
        for p in pair_k:
            if max_atoms is not None and pair_z[p][1] > max_atoms:
                continue
            dL = est_dL(p)
            if dL < best_dL:
                best_p, best_dL = p, dL
        if best_p is None:
            break

        for g in graphs:
            g.apply_merge(best_p)
        tc, ta = recount()
        L_now = _dl(tc, ta, 0, beta)
        if L_now >= L_traj[-1]:                   # raw-k estimate too optimistic -> stop (rule dropped)
            break
        L_traj.append(L_now)
        if verbose and (step < 10 or step % 25 == 0):
            print(f"    [ring_grow_bpe] merge {step:3d}: {best_p[0]:>12s} + {best_p[1]:<12s} "
                  f"-> {pair_z[best_p][0]:<12s} estDeltaL={best_dL:8.1f} L={L_now:11.1f}", flush=True)

    if verbose:
        print(f"    [ring_grow_bpe/mdl] {len(L_traj) - 1} merges, L={L_traj[-1]:.0f}", flush=True)
    return [[(g.ident[i], frozenset(g.atoms[i])) for i in range(len(g.atoms))] for g in graphs]


# ───────────────────────────── frequency-BPE (optional head-to-head) ───────────
def _freq_bpe_cover(mols, part_fn, floor_frac=0.005, max_atoms=None, max_merges=6000, verbose=False):
    """Classic frequency BPE with growable rings: each round merge the most-frequent adjacent type-pair
    until its corpus count falls below floor_frac * n_mols. If max_atoms is set, a pair whose merged
    motif would exceed max_atoms heavy atoms is SKIPPED (the next-most-frequent in-budget pair is taken
    instead) — this caps growth and curbs the over-merge tail. Returns per-mol cover [(key, atoms)]."""
    gs = [_grow_graph(m, part_fn) for m in mols]
    FLOOR = floor_frac * len(mols)
    n_merges = 0
    for _ in range(max_merges):
        pk: Counter = Counter()
        psize = {}
        for g in gs:
            for i, j, _r in g.linker_adjacencies():
                p = tuple(sorted((g.ident[i], g.ident[j])))
                pk[p] += 1
                if p not in psize:
                    psize[p] = len(g.atoms[i]) + len(g.atoms[j])
        best = None
        for p, k in pk.most_common():
            if k < FLOOR:                                     # descending -> the rest are below floor too
                break
            if max_atoms is not None and psize[p] > max_atoms:
                continue                                      # skip over-budget pair, take next in-budget
            best = p
            break
        if best is None:
            break
        for g in gs:
            g.apply_merge(best)
        n_merges += 1
    if verbose:
        cap = 'none' if max_atoms is None else str(max_atoms)
        print(f"    [ring_grow_bpe/bpe] {n_merges} merges (floor={FLOOR:.0f}, max_atoms={cap})", flush=True)
    return [[(g.ident[i], frozenset(g.atoms[i])) for i in range(len(g.atoms))] for g in gs]


# ───────────────────────────── shoulder-stopped BPE (data-derived floor) ────────
def _knee_index(counts):
    """Kneedle-style max-distance-from-chord knee on the (step, log10 support) decay curve.

    counts = the per-step merge supports from frequency BPE (non-increasing). Support is ~Zipfian, so
    the knee is detected on LOG-frequency (a linear-scale knee would sit at merge ~1-3, too aggressive).
    Returns t* = number of leading merges to KEEP (the steep 'signal' head), dropping the flat 'noise'
    tail. This replaces the arbitrary 0.5%-of-molecules floor with a per-corpus data-derived stop."""
    n = len(counts)
    if n <= 2:
        return n
    ys = [math.log10(max(c, 1)) for c in counts]
    x1 = float(n - 1)
    y0, y1 = ys[0], ys[-1]
    dx, dy = x1, (y1 - y0)
    denom = math.hypot(dx, dy) or 1.0
    best_i, best_d = 0, -1.0
    for i in range(n):
        d = abs(dy * i - dx * (ys[i] - y0)) / denom          # perpendicular distance to the chord
        if d > best_d:
            best_d, best_i = d, i
    return best_i + 1                                          # keep merges up to & incl. the knee


def _shoulder_bpe_cover(mols, part_fn, max_merges=2000, verbose=False):
    """Frequency-ordered BPE with a DATA-DERIVED stop (NO arbitrary floor). Pass 1: run to exhaustion
    (top-pair support < 2, or cap), recording the merge-frequency decay curve c_t. Pass 2: find the knee
    on (step, log10 c_t) and re-cover keeping only the leading t* (steep-head) merges. Frequency still
    chooses WHICH pair to merge; the shoulder chooses WHEN to stop."""
    gs = [_grow_graph(m, part_fn) for m in mols]
    rules, counts = [], []
    for _ in range(max_merges):
        pk = Counter()
        for g in gs:
            for i, j, _r in g.linker_adjacencies():
                pk[tuple(sorted((g.ident[i], g.ident[j])))] += 1
        if not pk:
            break
        best, k = pk.most_common(1)[0]
        if k < 2:                                             # a once-seen pair is not learnable
            break
        rules.append(best); counts.append(k)
        for g in gs:
            g.apply_merge(best)
    tstar = _knee_index(counts)
    if verbose:
        at = counts[tstar - 1] if 0 < tstar <= len(counts) else None
        print(f"    [ring_grow_bpe/shoulder] full trajectory={len(counts)} merges; knee t*={tstar} "
              f"(support@knee={at}); keeping first {tstar}", flush=True)
    gs2 = [_grow_graph(m, part_fn) for m in mols]
    for r in rules[:tstar]:
        for g in gs2:
            g.apply_merge(r)
    return [[(g.ident[i], frozenset(g.atoms[i])) for i in range(len(g.atoms))] for g in gs2]


# ───────────────────────────── top-level entry ─────────────────────────────────
def build(smiles_all, head_source='none', linker_method='mdl', break_fused_rings=False,
          beta=0.0, verbose=True):
    """Return mol_frags_tracked = per-mol [(key, set(atoms))], aligned to smiles_all (empty list for
    unparseable SMILES).

    linker_method : 'mdl' (MDL-BPE, DEFAULT) | 'bpe' (frequency BPE, 0.5%% floor) | 'bpecap8'
    (frequency BPE, 0.5%% floor + motif capped at 8 heavy atoms) | 'shoulder' (frequency BPE stopped
    at the data-derived knee of the merge-frequency decay curve, no floor). head_source accepted for
    call-compatibility with ring_mdl.build / mdl_linker.build but IGNORED (rings-only seeding, no FG
    heads). break_fused_rings freezes each SSSR ring separately."""
    if linker_method not in ('mdl', 'bpe', 'bpecap8', 'shoulder'):
        raise ValueError(f"linker_method must be mdl|bpe|bpecap8|shoulder, got {linker_method!r}")
    part_fn = ring_mdl.make_partition(break_fused_rings)
    mols = [Chem.MolFromSmiles(s) for s in smiles_all]
    idx_valid = [i for i, m in enumerate(mols) if m is not None]
    valid = [mols[i] for i in idx_valid]
    if verbose:
        print(f"    [ring_grow_bpe] rings-seeded GROWABLE (head_source ignored) linker={linker_method} "
              f"break_fused={break_fused_rings} on {len(valid)}/{len(smiles_all)} mols ...", flush=True)
    if not valid:
        return [[] for _ in smiles_all]
    if linker_method == 'mdl':
        covers = _mdl_bpe_cover(valid, part_fn, beta=beta, verbose=verbose)
    elif linker_method == 'shoulder':
        covers = _shoulder_bpe_cover(valid, part_fn, verbose=verbose)
    elif linker_method == 'bpecap8':
        covers = _freq_bpe_cover(valid, part_fn, max_atoms=8, verbose=verbose)
    else:
        covers = _freq_bpe_cover(valid, part_fn, verbose=verbose)
    out = [[] for _ in smiles_all]
    for j, i in enumerate(idx_valid):
        out[i] = [(k, set(at)) for k, at in covers[j]]
    return out
