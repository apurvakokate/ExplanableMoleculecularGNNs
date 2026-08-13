"""ring_grow_mdl.py — rings + BOUNDED local growth + MDL (candidate #2).

Like ring_mdl (rings + KRIMP-MDL linker) BUT rings are NOT hard-frozen. For each ring system we ALSO
offer BOUNDED "grown-ring" candidates and let KRIMP/MDL SELECT one only when it lowers description
length; otherwise the whole ring is used (so rings NEVER shatter).

Grown-ring candidate rule (signed off):
  * 'attach' = the ring's DIRECT non-ring neighbour atoms (ONE hop). Directly-bonded rings are excluded
    (non-ring attachments only) — no partial rings, no ring+ring motifs; a ring-ring bridge stays linker.
  * Emit ring ∪ S for EVERY non-empty subset S of attach: one attachment at a time, pairs, ..., up to
    ring ∪ (all attachments) as a single motif. 2^k - 1 candidates for k attachments — bounded per ring
    by 2^(ring degree); NOT the unbounded all-subgraph enumeration that exploded mdl_only. (A ring with
    > MAX_ATTACH_POWERSET attachments falls back to singletons + the full set.)
  * The cover picks the LARGEST selected grown subset containing the ring, so ring ∪ (all attachments)
    wins when selected; otherwise a smaller selected subset, else the whole ring.

Keying: a SELECTED grown ring -> attachment-aware frag: key; an un-grown ring -> canonical ring: key.

Cover: each ring is covered by its LARGEST selected grown candidate (by atom containment); if none is
selected, by the whole ring. Then non-ring linker atoms are covered by KRIMP-selected HR/rBRICS/BRICS/
RECAP candidates + atom singletons (identical to ring_mdl). Complete + non-overlapping (asserted).

Functions replicated from ring_mdl.py. build(smiles_all, head_source='none', linker_method='mdl',
break_fused_rings=False) -> per-mol [(key, set(atoms))], same signature/format as ring_mdl.build.
head_source ignored; only MDL selection is supported (growth is candidate-selection, not BPE merging).
NO FALLBACK: fails loudly on keying/cover errors.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import combinations

from rdkit import Chem

import ertl_conservative_frag as ec
import chemfrag as cf

K_HR = 5                    # Hussain-Rea connected-subgraph cap for the non-ring linker tier
MAX_ATTACH_POWERSET = 12    # rings with > this many direct attachments -> singletons + full-set only


# ───────────────────────────── ring detection (same as ring_mdl) ───────────────
def _ring_groups(mol, break_fused):
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    if not ring:
        return []
    if not break_fused:
        return ec._components(mol, ring)
    groups, assigned = [], set()
    for r in Chem.GetSymmSSSR(mol):
        g = {a for a in r if a not in assigned}
        if g:
            assigned |= g
            groups.append(g)
    for comp in ec._components(mol, ring - assigned):
        groups.append(set(comp))
    return groups


def _adj(mol):
    adj = defaultdict(set)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i].add(j); adj[j].add(i)
    return adj


# ───────────────────────────── bounded grown-ring candidates ───────────────────
def _grown_candidates(mol, ratoms, ring_all, adj):
    """Ring + ONE-HOP growth over ALL SUBSETS of the ring's direct attachments. 'attach' = the ring's
    direct NON-RING neighbour atoms (directly-bonded rings excluded — non-ring only). Emit ring ∪ S for
    EVERY non-empty subset S of attach: one attachment at a time, pairs, ..., up to ring ∪ (all
    attachments) as a single motif — 2^k - 1 candidates for k attachments (bounded per ring by
    2^(ring degree)). For a pathologically substituted ring (k > MAX_ATTACH_POWERSET) only the
    singletons and the full set are emitted, to avoid a 2^k blow-up. Returns list of frozenset(atoms)."""
    attach = sorted({n for r in ratoms for n in adj[r] if n not in ring_all})
    k = len(attach)
    if k == 0:
        return []
    out = []
    if k <= MAX_ATTACH_POWERSET:
        for size in range(1, k + 1):
            for combo in combinations(attach, size):
                out.append(frozenset(ratoms | set(combo)))
    else:
        for a in attach:                                    # one-at-a-time
            out.append(frozenset(ratoms | {a}))
        out.append(frozenset(ratoms | set(attach)))         # ring + all attachments
    return out


# ───────────────────────────── linker candidate pool (same as ring_mdl) ────────
def _linker_adj(mol, atoms):
    aset = set(atoms)
    adj = defaultdict(set)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in aset and j in aset:
            adj[i].add(j); adj[j].add(i)
    return adj


def _enumerate_connected(mol, comp, K):
    adj = _linker_adj(mol, comp)
    result = set()
    frontier = {frozenset([a]) for a in comp}
    result |= frontier
    for _ in range(K - 1):
        new = set()
        for s in frontier:
            nbrs = set().union(*[adj[a] for a in s]) - s if s else set()
            for nb in nbrs:
                ns = s | {nb}
                if ns not in result:
                    new.add(ns)
        if not new:
            break
        result |= new; frontier = new
    return result


def prep_molecule(mol, break_fused):
    """ring_systems=[(whole_ring_key, ratoms)]; cands=key->{atoms,size,grown}; linker_atoms=nonring."""
    ring_all = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    adj = _adj(mol)
    cands = {}

    def add(atoms, grown):
        atoms = frozenset(atoms)
        if not atoms:
            return
        k = cf.frag_key(mol, atoms)
        if k is None:
            return
        if cands.get(k) is None:
            cands[k] = {'atoms': atoms, 'size': len(atoms), 'grown': grown}

    ring_systems = []
    for rat in _ring_groups(mol, break_fused):
        ratoms = frozenset(rat)
        ring_systems.append((ec._ring_key(mol, ratoms), ratoms))
        for gat in _grown_candidates(mol, ratoms, ring_all, adj):
            add(gat, grown=True)

    linker_atoms = frozenset(a for a in range(mol.GetNumAtoms()) if a not in ring_all)
    comps = ec._components(mol, linker_atoms)
    for a in linker_atoms:                                          # singletons (MDL atomic base)
        add({a}, grown=False)
    for comp in comps:                                             # Hussain-Rea acyclic ceiling
        for s in _enumerate_connected(mol, comp, K_HR):
            if len(s) >= 2:
                add(s, grown=False)
    for comp in comps:                                             # retrosynthetic priors
        if len(comp) < 2:
            continue
        for bondsets in ([cf._rbrics_within(mol, comp)], [cf._brics_within(mol, comp)],
                         cf.recap_split_options(mol, comp)):
            for bonds in bondsets:
                if not bonds:
                    continue
                for seg in cf.comps_in(mol, comp, bonds):
                    if len(seg) >= 2:
                        add(seg, grown=False)
    return ring_systems, linker_atoms, cands


# ───────────────────────────── cover (ring-first, whole-ring fallback) ─────────
def cover_molecule(mol, ring_systems, linker_atoms, cands, allowed, detail=False):
    claimed = set()
    counts = Counter()
    cover = []

    def place(k, atoms):
        nonlocal claimed
        claimed |= atoms; counts[k] += 1
        if detail:
            cover.append((k, frozenset(atoms)))

    # rings: largest SELECTED grown candidate containing the ring, else the whole ring
    for whole_key, ratoms in ring_systems:
        grown = sorted(
            ((e['size'], k, e['atoms']) for k, e in cands.items()
             if e['grown'] and k in allowed and ratoms <= e['atoms'] and not (e['atoms'] & claimed)),
            key=lambda t: (-t[0], t[1]))
        if grown:
            _sz, k, atoms = grown[0]
            place(k, atoms)
        else:
            place(whole_key, ratoms)

    # non-ring linker: selected HR/rBRICS/... candidates (size desc) then singletons
    insts = [(e['size'], k, e['atoms']) for k, e in cands.items()
             if not e['grown'] and e['size'] >= 2 and k in allowed and (e['atoms'] <= linker_atoms)]
    insts.sort(key=lambda t: (-t[0], t[1]))
    for _sz, k, atoms in insts:
        if atoms & claimed:
            continue
        place(k, atoms)
    for a in linker_atoms:
        if a not in claimed:
            k = cf.frag_key(mol, {a})
            assert k is not None, "singleton keyed None"
            place(k, {a})

    assert claimed == set(range(mol.GetNumAtoms())), "cover incomplete/overlap"
    return (counts, cover) if detail else counts


def _two_part_L(type_count, size_map):
    N = sum(type_count.values())
    if N == 0:
        return 0.0
    LOG2_20 = math.log2(20.0)
    dbits = sum(size_map[t] * LOG2_20 for t in type_count)
    cbits = N * math.log2(N) - sum(c * math.log2(c) for c in type_count.values() if c > 0)
    return dbits + cbits


# ───────────────────────────── KRIMP select + prune ────────────────────────────
def _mdl_cover(mols, break_fused, verbose=False):
    prep, size_map = [], {}
    contains = defaultdict(set); support = Counter()
    for mi, mol in enumerate(mols):
        rs, la, cands = prep_molecule(mol, break_fused)
        prep.append((rs, la, cands))
        for k, e in cands.items():
            size_map[k] = e['size']
            if e['size'] >= 2:
                contains[k].add(mi); support[k] += 1        # selectable = grown + multi-atom linker
        for wk, ratoms in rs:
            size_map.setdefault(wk, len(ratoms))             # whole-ring types (always-available)
    mol_counts = [cover_molecule(mols[mi], *prep[mi], set()) for mi in range(len(mols))]
    total = Counter()
    for c in mol_counts:
        total += c
    L = _two_part_L(total, size_map)
    allowed, selected = set(), set()
    for k in sorted(support, key=lambda k: (-support[k], -size_map[k], k)):
        aff = contains[k]
        nc = {mi: cover_molecule(mols[mi], *prep[mi], allowed | {k}) for mi in aff}
        cand = total.copy()
        for mi in aff:
            cand.update(nc[mi]); cand.subtract(mol_counts[mi])
        cand = Counter({t: v for t, v in cand.items() if v > 0})
        Lnew = _two_part_L(cand, size_map)
        if Lnew < L - 1e-9:
            for mi in aff:
                mol_counts[mi] = nc[mi]
            total, L, allowed = cand, Lnew, allowed | {k}
            selected.add(k)
    changed = True                                          # KRIMP post-acceptance prune
    while changed:
        changed = False
        for k in sorted(list(selected), key=lambda k: (support[k], size_map[k])):
            aff = contains[k]; na = allowed - {k}
            nc = {mi: cover_molecule(mols[mi], *prep[mi], na) for mi in aff}
            cand = total.copy()
            for mi in aff:
                cand.update(nc[mi]); cand.subtract(mol_counts[mi])
            cand = Counter({t: v for t, v in cand.items() if v > 0})
            if _two_part_L(cand, size_map) < L - 1e-9:
                for mi in aff:
                    mol_counts[mi] = nc[mi]
                total, L, allowed = cand, _two_part_L(cand, size_map), na
                selected.discard(k); changed = True
    n_grown = sum(1 for k in selected if any(prep[mi][2].get(k, {}).get('grown') for mi in contains[k]))
    if verbose:
        print(f"    [ring_grow_mdl] selected {len(selected)} motifs ({n_grown} grown-ring), "
              f"L={L:.0f}", flush=True)
    covers = []
    for mi in range(len(mols)):
        _c, cov = cover_molecule(mols[mi], *prep[mi], allowed, detail=True)
        covers.append(cov)
    return covers


# ───────────────────────────── top-level entry ─────────────────────────────────
def build(smiles_all, head_source='none', linker_method='mdl', break_fused_rings=False, verbose=True):
    """Return mol_frags_tracked = per-mol [(key, set(atoms))], aligned to smiles_all. head_source ignored
    (rings + bounded growth by design). linker_method must be 'mdl' (KRIMP selection)."""
    if linker_method != 'mdl':
        raise ValueError(f"ring_grow_mdl supports linker_method='mdl' only, got {linker_method!r}")
    mols = [Chem.MolFromSmiles(s) for s in smiles_all]
    idx_valid = [i for i, m in enumerate(mols) if m is not None]
    valid = [mols[i] for i in idx_valid]
    if verbose:
        print(f"    [ring_grow_mdl] rings + 1-hop attachment-subset growth + MDL on "
              f"{len(valid)}/{len(smiles_all)} mols ...", flush=True)
    if not valid:
        return [[] for _ in smiles_all]
    covers = _mdl_cover(valid, break_fused_rings, verbose=verbose)
    out = [[] for _ in smiles_all]
    for j, i in enumerate(idx_valid):
        out[i] = [(k, set(at)) for k, at in covers[j]]
    return out
