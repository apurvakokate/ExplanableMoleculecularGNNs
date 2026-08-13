"""mdl_only.py — MDL step ONLY (NO ring separation, NO FG-head tier).

The whole molecule is handed to the MDL linker: KRIMP selection + prune over the candidate pool
(Hussain-Rea connected subgraphs <=K, rBRICS / BRICS / RECAP cuts, atom singletons) selects every
motif. Nothing is frozen up front — this is the rings-and-heads-removed ablation of mdl_linker /
ring_mdl. Atom singletons remain in the pool solely as MDL's atomic base so the cover stays complete.

Functions are REPLICATED from ring_mdl.py / mdl_linker.py (not imported) so it can iterate
independently. The ONLY difference is make_partition, which freezes nothing.

CONSEQUENCE (by design): the Hussain-Rea cap is K=8, so whole 6-8-atom rings (benzene, pyridine,
7/8-membered rings) CAN now be enumerated as connected subgraphs and selected by MDL as single
frag: motifs when the compression pays off. Larger/fused ring systems (>8 atoms) still cannot be
captured whole and enter only via rBRICS/BRICS/RECAP segments. Note motifs remain keyed structurally
as frag: WITH attachment context (no ring-freezing) — so a substituted ring is still NOT
substituent-agnostic (different decorations -> different motifs), unlike the ring-frozen variants;
K=8 only restores whole-ring EXPRESSIBILITY, it does not make rings substituent-invariant.

Top-level entry: ``build(smiles_all, head_source='none', linker_method='mdl', break_fused_rings=False)``
returns mol_frags_tracked = per-mol [(key, set(atoms))] aligned to smiles_all, IDENTICAL format/signature
to mdl_linker.build so the call site is unchanged. head_source / break_fused_rings are accepted for
call-compatibility but IGNORED (nothing is frozen).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from rdkit import Chem

import ertl_conservative_frag as ec
import chemfrag as cf
from cascade_bpe_linker import _MolGraph, LOG2_20

K_HR = 8          # Hussain-Rea connected-subgraph enumeration cap (heavy atoms)


# ───────────────────────────── partition builder (freeze NOTHING) ──────────────
def make_partition():
    """Return a partition_fn(mol, **kwargs) -> (owner, ids). Freezes NOTHING: every connected
    component of the whole molecule is a linker group (no rings, no FG heads)."""

    def part(mol, **kwargs):
        n = mol.GetNumAtoms()
        owner = [-1] * n
        ids = {}
        fid = 0
        for comp in ec._components(mol, set(range(n))):          # whole molecule -> linker
            pure_c = all(mol.GetAtomWithIdx(a).GetAtomicNum() == 6 for a in comp)
            ids[fid] = 'chain:' if pure_c else 'frag:'
            for a in comp:
                owner[a] = fid
            fid += 1
        if any(o < 0 for o in owner):
            raise ValueError(
                f"mdl_only partition left atom(s) unassigned for {Chem.MolToSmiles(mol)!r}")
        return owner, ids

    return part


# ───────────────────────────── candidate pool ──────────────────────────────
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


def prep_molecule(mol, partition_fn):
    """No frozen tokens (freeze nothing); linker candidate pool over the WHOLE molecule."""
    owner, ids = partition_fn(mol)
    node_atoms = defaultdict(set)
    for a, o in enumerate(owner):
        node_atoms[o].add(a)
    frozen = []
    linker_atoms = set()
    for fid, atoms in node_atoms.items():
        tag = ids[fid]
        if tag.startswith('ring:'):          # mdl_only never emits ring: tags; kept for parity
            frozen.append((tag, frozenset(atoms)))
        else:
            linker_atoms |= atoms
    cands = {}

    def add(atoms):
        atoms = frozenset(atoms)
        if not atoms:
            return
        k = cf.frag_key(mol, atoms)
        if k is None:
            return
        if cands.get(k) is None:
            cands[k] = {'atoms': atoms, 'size': len(atoms)}

    comps = ec._components(mol, linker_atoms)
    for a in linker_atoms:                                   # singletons (MDL atomic base)
        add({a})
    for comp in comps:                                       # Hussain-Rea acyclic-cut ceiling
        for s in _enumerate_connected(mol, comp, K_HR):
            if len(s) >= 2:
                add(s)
    for comp in comps:                                       # retrosynthetic priors (subsets of HR)
        if len(comp) < 2:
            continue
        for bondsets in ([cf._rbrics_within(mol, comp)], [cf._brics_within(mol, comp)],
                         cf.recap_split_options(mol, comp)):
            for bonds in bondsets:
                if not bonds:
                    continue
                for seg in cf.comps_in(mol, comp, bonds):
                    if len(seg) >= 2:
                        add(seg)
    return frozen, linker_atoms, cands


def cover_molecule(mol, frozen, linker_atoms, cands, allowed, detail=False):
    """Greedy deterministic cover (size desc, key) + singleton atomic base. Complete + non-overlapping
    (asserted). Returns Counter(key->count); with detail also [(key, atoms)]."""
    claimed = set()
    counts = Counter()
    cover = []
    for k, atoms in frozen:
        assert not (atoms & claimed), "frozen overlap"
        claimed |= atoms; counts[k] += 1
        if detail:
            cover.append((k, frozenset(atoms)))
    insts = [(e['size'], k, e['atoms']) for k, e in cands.items()
             if e['size'] >= 2 and k in allowed and (e['atoms'] <= linker_atoms)]
    insts.sort(key=lambda t: (-t[0], t[1]))
    for _sz, k, atoms in insts:
        if atoms & claimed:
            continue
        claimed |= atoms; counts[k] += 1
        if detail:
            cover.append((k, frozenset(atoms)))
    for a in linker_atoms:
        if a not in claimed:
            k = cf.frag_key(mol, {a})
            assert k is not None, "singleton keyed None"
            claimed |= {a}; counts[k] += 1
            if detail:
                cover.append((k, frozenset({a})))
    assert claimed == set(range(mol.GetNumAtoms())), "cover incomplete/overlap"
    return (counts, cover) if detail else counts


def _two_part_L(type_count, size_map):
    N = sum(type_count.values())
    if N == 0:
        return 0.0
    dbits = sum(size_map[t] * LOG2_20 for t in type_count)
    cbits = N * math.log2(N) - sum(c * math.log2(c) for c in type_count.values() if c > 0)
    return dbits + cbits


# ───────────────────────────── selection engines ──────────────────────────────
def _mdl_cover(mols, partition_fn, verbose=False):
    """KRIMP select + prune over the corpus; return per-mol detailed cover [(key, atoms)]."""
    prep, size_map = [], {}
    contains = defaultdict(set); support = Counter()
    for mi, mol in enumerate(mols):
        fr, la, cands = prep_molecule(mol, partition_fn)
        prep.append((fr, la, cands))
        for k, e in cands.items():
            size_map[k] = e['size']
            if e['size'] >= 2:
                contains[k].add(mi); support[k] += 1
        for k, a in fr:
            size_map.setdefault(k, len(a))
    mol_counts = [cover_molecule(mols[mi], *prep[mi], set()) for mi in range(len(mols))]
    total = Counter()
    for c in mol_counts:
        total += c
    L = _two_part_L(total, size_map)
    allowed, selected = set(), set()
    for k in sorted((k for k in size_map if size_map[k] >= 2),
                    key=lambda k: (-support[k], -size_map[k], k)):
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
    if verbose:
        print(f"    [mdl_only] selected {len(selected)} motifs, L={L:.0f}", flush=True)
    covers = []
    for mi in range(len(mols)):
        _c, cov = cover_molecule(mols[mi], *prep[mi], allowed, detail=True)
        covers.append(cov)
    return covers


def _bpe_cover(mols, partition_fn, floor_frac=0.005, verbose=False):
    """Greedy-frequency BPE (freeze nothing via partition_fn); return per-mol cover [(key, atoms)]."""
    gs = [_MolGraph(m, finest='all_bonds', freeze='heads', fg_partition=partition_fn) for m in mols]
    FLOOR = floor_frac * len(mols)
    n_merges = 0
    for _ in range(6000):
        pk = Counter()
        for g in gs:
            for i, j, _r in g.linker_adjacencies():
                pk[tuple(sorted((g.ident[i], g.ident[j])))] += 1
        if not pk:
            break
        best, k = pk.most_common(1)[0]
        if k < FLOOR:
            break
        for g in gs:
            g.apply_merge(best)
        n_merges += 1
    if verbose:
        print(f"    [mdl_only/bpe] {n_merges} merges", flush=True)
    return [[(g.ident[i], frozenset(g.atoms[i])) for i in range(len(g.atoms))] for g in gs]


# ───────────────────────────── top-level entry ──────────────────────────────
def build(smiles_all, head_source='none', linker_method='mdl', break_fused_rings=False,
          verbose=True):
    """Return mol_frags_tracked = per-mol [(key, set(atoms))], aligned to smiles_all.

    head_source / break_fused_rings are accepted for call-compatibility with mdl_linker.build but
    IGNORED — this module freezes nothing (MDL step only)."""
    if linker_method not in ('mdl', 'bpe'):
        raise ValueError(f"linker_method must be mdl|bpe, got {linker_method!r}")
    part_fn = make_partition()
    mols = [Chem.MolFromSmiles(s) for s in smiles_all]
    idx_valid = [i for i, m in enumerate(mols) if m is not None]
    valid = [mols[i] for i in idx_valid]
    if verbose:
        print(f"    [mdl_only] no freeze (rings+heads removed) linker={linker_method} "
              f"on {len(valid)}/{len(smiles_all)} mols ...", flush=True)
    if not valid:
        return [[] for _ in smiles_all]
    covers = (_mdl_cover if linker_method == 'mdl' else _bpe_cover)(valid, part_fn, verbose=verbose)
    out = [[] for _ in smiles_all]
    for j, i in enumerate(idx_valid):
        out[i] = [(k, set(at)) for k, at in covers[j]]
    return out
