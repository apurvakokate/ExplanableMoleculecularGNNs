"""mdl_linker.py — the settled FCOL linker tier: MDL SELECTION over a chemistry candidate pool
(KRIMP / GraphMDL), with a BPE alternative, over a frozen head partition.

This REPLACES the old bottom-up ``cascade_bpe_linker.learn`` merge for ``conservative_ertl_ring_mdl``.
Validated (BBBP/Mutagenicity/hERG/Benzene): MDL selection Pareto-beats greedy BPE (lower two-part L,
higher coverage), the conservative-Ertl head freeze is load-bearing (dropping it drives FG-motif
contamination 0-2% -> 23-34%), and the retrosynthetic rules are subsets of the Hussain-Rea acyclic-cut
ceiling. (A MACCS gap-detector was evaluated and removed: ~0.6% unique contribution across 4 datasets.)

Top-level entry: ``build(smiles_all, head_source, linker_method, break_fused_rings)`` returns
``mol_frags_tracked`` = per-molecule ``[(key, set(atoms))]`` (frozen heads + linker cover), aligned to
``smiles_all`` (empty list for unparseable SMILES) — the exact format generate_vocab_rules consumes.

  head_source   : 'ertl' (rings + conservative-Ertl FG heads, DEFAULT) | 'rbrics' (rings +
                  rBRICS-conservative FG heads) | 'none' (rings only, no FG freeze)
  linker_method : 'mdl' (KRIMP selection + prune, DEFAULT) | 'bpe' (greedy-frequency merge)
  break_fused_rings : freeze each SSSR ring separately (shared atoms -> earliest ring) instead of
                  whole fused systems.

NO FALLBACK: fails loudly on partition/keying errors (mirrors ertl_conservative_frag).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from rdkit import Chem

import ertl_conservative_frag as ec
import chemfrag as cf
from cascade_bpe_linker import _MolGraph, LOG2_20

K_HR = 5          # Hussain-Rea connected-subgraph enumeration cap (heavy atoms)


# ───────────────────────────── partition builders ──────────────────────────────
def _functional(mol, s):
    if any(mol.GetAtomWithIdx(a).GetAtomicNum() not in (1, 6) for a in s):
        return True
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in s and j in s and b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE) \
                and not b.GetIsAromatic():
            return True
    return False


def _ring_groups(mol, break_fused):
    """Ring atom groups: whole fused systems (default) or per-SSSR-ring (break_fused; each shared
    atom assigned to the earliest SSSR ring containing it, so groups are non-overlapping)."""
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
    leftover = ring - assigned                      # (shouldn't happen) any ring atom not in SSSR
    for comp in ec._components(mol, leftover):
        groups.append(set(comp))
    return groups


def _fg_heads_ertl(mol, ring):
    out = []
    for s in ec._get_fgs(mol):
        s = set(s)
        if not (s - ring):
            continue
        if ec._kept(mol, s, ring):
            out.append(s)
    return out


def _fg_heads_rbrics(mol, ring):
    out = []
    for frag in cf.part_rbrics(mol):
        s = set(frag) - ring
        if not s:
            continue
        for comp in ec._components(mol, s):
            comp = set(comp)
            if _functional(mol, comp) and ec._kept(mol, comp, ring):
                out.append(comp)
    return out


def make_partition(head_source='ertl', break_fused_rings=False):
    """Return a partition_fn(mol, **kwargs) -> (owner, ids) matching ertl_conservative_frag.partition.
    The DEFAULT (ertl, whole rings) delegates to the validated ec.partition unchanged."""
    if head_source == 'ertl' and not break_fused_rings:
        return ec.partition
    if head_source not in ('ertl', 'rbrics', 'none'):
        raise ValueError(f"head_source must be ertl|rbrics|none, got {head_source!r}")

    def part(mol, **kwargs):
        n = mol.GetNumAtoms()
        ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
        owner = [-1] * n
        ids = {}
        fid = 0
        for comp in _ring_groups(mol, break_fused_rings):       # rings (frozen)
            ids[fid] = ec._ring_key(mol, comp)
            for a in comp:
                owner[a] = fid
            fid += 1
        if head_source == 'ertl':
            heads = _fg_heads_ertl(mol, ring)
        elif head_source == 'rbrics':
            heads = _fg_heads_rbrics(mol, ring)
        else:
            heads = []
        for s in heads:                                          # FG heads (frozen)
            if any(owner[a] >= 0 for a in s):                    # overlaps a ring / prior head
                continue
            ids[fid] = 'fg:' + head_source
            for a in s:
                owner[a] = fid
            fid += 1
        leftover = [a for a in range(n) if owner[a] < 0]         # rest -> linker
        for comp in ec._components(mol, leftover):
            pure_c = all(mol.GetAtomWithIdx(a).GetAtomicNum() == 6 for a in comp)
            ids[fid] = 'chain:' if pure_c else 'frag:'
            for a in comp:
                owner[a] = fid
            fid += 1
        if any(o < 0 for o in owner):
            raise ValueError(
                f"mdl_linker partition left atom(s) unassigned for {Chem.MolToSmiles(mol)!r}")
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
    """Frozen head tokens + linker candidate pool for ONE molecule."""
    owner, ids = partition_fn(mol)
    node_atoms = defaultdict(set)
    for a, o in enumerate(owner):
        node_atoms[o].add(a)
    frozen = []
    linker_atoms = set()
    for fid, atoms in node_atoms.items():
        tag = ids[fid]
        if tag.startswith('ring:'):
            frozen.append((tag, frozenset(atoms)))
        elif tag.startswith('fg:'):
            k = cf.frag_key(mol, atoms)
            if k is None:
                raise ValueError(f"frozen FG keyed to None in {Chem.MolToSmiles(mol)!r}")
            frozen.append((k, frozenset(atoms)))
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
        e = cands.get(k)
        if e is None:
            cands[k] = {'atoms': atoms, 'size': len(atoms)}

    comps = ec._components(mol, linker_atoms)                # shared DFS (was a local duplicate)
    for a in linker_atoms:                                   # singletons (fallback)
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
    """Greedy deterministic cover (size desc, key) + singleton fallback. Complete + non-overlapping
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
        print(f"    [mdl_linker] selected {len(selected)} linker motifs, L={L:.0f}", flush=True)
    covers = []
    for mi in range(len(mols)):
        _c, cov = cover_molecule(mols[mi], *prep[mi], allowed, detail=True)
        covers.append(cov)
    return covers


def _bpe_cover(mols, partition_fn, floor_frac=0.005, verbose=False):
    """Greedy-frequency BPE (freeze heads via partition_fn); return per-mol cover [(key, atoms)]."""
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
        print(f"    [mdl_linker/bpe] {n_merges} merges", flush=True)
    return [[(g.ident[i], frozenset(g.atoms[i])) for i in range(len(g.atoms))] for g in gs]


# ───────────────────────────── top-level entry ──────────────────────────────
def build(smiles_all, head_source='ertl', linker_method='mdl', break_fused_rings=False,
          verbose=True):
    """Return mol_frags_tracked = per-mol [(key, set(atoms))], aligned to smiles_all."""
    if linker_method not in ('mdl', 'bpe'):
        raise ValueError(f"linker_method must be mdl|bpe, got {linker_method!r}")
    part_fn = make_partition(head_source, break_fused_rings)
    mols = [Chem.MolFromSmiles(s) for s in smiles_all]
    idx_valid = [i for i, m in enumerate(mols) if m is not None]
    valid = [mols[i] for i in idx_valid]
    if verbose:
        print(f"    [mdl_linker] head={head_source} linker={linker_method} "
              f"break_fused={break_fused_rings} on {len(valid)}/{len(smiles_all)} mols ...", flush=True)
    if not valid:
        return [[] for _ in smiles_all]
    covers = (_mdl_cover if linker_method == 'mdl' else _bpe_cover)(valid, part_fn, verbose=verbose)
    out = [[] for _ in smiles_all]
    for j, i in enumerate(idx_valid):
        out[i] = [(k, set(at)) for k, at in covers[j]]
    return out
