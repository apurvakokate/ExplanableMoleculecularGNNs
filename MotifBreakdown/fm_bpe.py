"""fm_bpe.py — FM-BPE (Frequency-MDL BPE): greedy agglomerative merge under a fixed-b MDL criterion.

Starts from the FINEST blocks consistent with every fragmentation approach in the motif pool
(rings kept whole — no approach ever cuts inside a ring; every other atom its own block, only
single acyclic bonds cut so C=O/C=N pairs stay together). NOTHING else is pre-grouped: FG heads
are NOT frozen, so the FM-BPE criterion alone decides every merge (amide/ester/... re-form only
if the data pays for them). Rings may GROW outward but never split.

Merge criterion (FM-BPE, fixed-b form). For an adjacent fragment-TYPE pair (M1, M2) with corpus
co-occurrence count f12 and heavy-atom sizes |M1|, |M2|:

    dC(M1,M2) = lambda * (|M1| + |M2|)  -  f12 * b          merge iff dC < 0

derived from the two-part cost  lambda*cost(vocab) + cost(D|vocab)  under a FIXED b bits per token
reference (NOT log|V|): adding the merged type costs its size |M1|+|M2| in the vocab; each of the
f12 co-occurrences then uses ONE reference instead of two, saving b bits apiece. The alone-terms
(M1, M2 used elsewhere) and the rest-of-vocab term cancel under fixed b (see the derivation — the
log|V| variant's +(f1+f2+T)*log(|V+1|/|V|) inflation vanishes when b is constant).

ONLY the ratio theta = lambda/b matters:  dC < 0  <=>  f12 > theta * (|M1|+|M2|).  So this module
fixes b = 1.0 and sweeps lambda == theta = "required co-occurrences per unit merged-motif size".
Small theta -> coarse (aggressive merging); large theta -> fine. theta is the single granularity
knob (the analogue of SFO's size ramp).

Greedy loop: each round score every adjacent type-pair, apply the most-negative-dC merge across the
whole corpus (BPE: merge that type-pair everywhere), recount, stop when no pair has dC < 0.

Reuses the vetted _MolGraph BPE engine from cascade_bpe_linker.py unchanged (same finest='all_bonds'
starting cut, same frag_key re-keying on merge, same connectivity-respecting apply_merge).

Top-level: build(smiles_all, lam, b=1.0, max_atoms=None) -> per-mol [(key, set(atoms))] aligned to
smiles_all, IDENTICAL format to ring_mdl.build / mdl_linker.build so the vocab call site is unchanged.
head_source / linker_method / break_fused_rings are accepted for call-compatibility and ignored.

NO FALLBACK on keying/cover errors (fails loudly, like ring_mdl / mdl_only). Unparseable SMILES are
skipped with a warning and get an empty cover so index alignment to smiles_all is preserved.
"""
from __future__ import annotations

import sys
from collections import Counter
from typing import Dict, List, Sequence, Set, Tuple

from rdkit import Chem, RDLogger

import ring_mdl
from cascade_bpe_linker import _MolGraph

RDLogger.DisableLog('rdApp.*')


# ───────────────────────────── finest starting state ───────────────────────────
def _union_cut_bonds(mol: Chem.Mol):
    """SFO's EXACT union-of-fragmenters cut set — the same blocks SFO's exact-cover starts from —
    via xcover.exp1_candidate_inventory.segment_to_fixpoint (BRICS + rBRICS + RECAP + CRUSH, unioned
    and refined to a fixpoint). Using SFO's own block builder (not a separate scheme set) makes the
    union arm a clean criterion-vs-SFO comparison: identical admissible boundaries, only the merge
    rule differs. These schemes never cut ring bonds. (Deps xcover + crush_smirks_rules are the ones
    SFO itself uses, already deployed — no extra module needed.)"""
    import os, sys
    _mb = os.path.dirname(os.path.abspath(__file__))
    _xc = os.path.join(_mb, 'xcover')
    for _p in (_mb, _xc):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import exp1_candidate_inventory as _X
    rules = _X._ensure_rules(_mb)
    cuts, _base, _rounds = _X.segment_to_fixpoint(mol, rules)
    out = set()
    for bi in cuts:                                     # SFO union-of-fragmenters bonds (fixpoint)
        b = mol.GetBondWithIdx(bi)
        out.add(frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
    return out


def _finest_graph(mol: Chem.Mol, freeze_rings: bool = True, start: str = 'atomic') -> _MolGraph:
    """Starting blocks for the FM-BPE merge. Rings are kept whole (ring_mdl partition, freeze='rings').

    start='atomic' (default): every non-ring atom is its own block (finest='all_bonds' cuts every
      single acyclic bond) — the absolute finest; the merge can reach ANY partition.
    start='union': collapse the atomic linker blocks across every single acyclic bond that NO
      fragmenter proposes, leaving the union-of-fragmenters (SFO-pool) blocks — chemically admissible
      boundaries only; the merge can only coarsen them.

    freeze_rings=True (default): rings stay frozen -> bare substituent-agnostic `ring:` tokens, never
      merged (matches ring_mdl / SFO / the deployed pipeline). freeze_rings=False: rings are un-frozen
      and may grow outward (attachment-aware), the old growable-ring ablation."""
    g = _MolGraph(mol, finest='all_bonds', freeze='rings', fg_partition=ring_mdl.make_partition())
    if start == 'union':
        U = _union_cut_bonds(mol)
        changed = True
        while changed:                                  # merge linker atoms across every NON-voted bond
            changed = False
            a2n = {a: ni for ni, at in enumerate(g.atoms) for a in at}
            for b in mol.GetBonds():
                if b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE:
                    continue                            # rings/double bonds already within one block
                ai, aj = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                i, j = a2n[ai], a2n[aj]
                if i == j or g.frozen[i] or g.frozen[j]:
                    continue                            # skip ring (frozen) nodes during pre-merge
                if frozenset((ai, aj)) in U:
                    continue                            # union boundary -> keep this cut
                g._merge_nodes(i, j); changed = True; break   # indices shift -> restart scan
    if not freeze_rings:
        g.frozen = [False] * len(g.frozen)              # un-freeze -> rings may grow (ablation)
    return g


# ───────────────────────────── greedy advisor-MDL merge ────────────────────────
def _local_pairs(g: _MolGraph, type_size: Dict[str, int]) -> Counter:
    """Counter of this graph's adjacent fragment-TYPE pairs (sorted); also records each type's size."""
    lp: Counter = Counter()
    for i, j, _ in g.linker_adjacencies():                   # all adjacencies (nothing frozen)
        a, c = g.ident[i], g.ident[j]
        type_size.setdefault(a, len(g.atoms[i]))
        type_size.setdefault(c, len(g.atoms[j]))
        lp[tuple(sorted((a, c)))] += 1
    return lp


def learn_fm_bpe(smiles: Sequence[str], lam: float, b: float = 1.0,
                  max_atoms: int | None = None, max_merges: int = 8000, verbose: bool = False,
                  freeze_rings: bool = True, start: str = 'atomic'):
    """Greedy BPE under dC = lam*(|M1|+|M2|) - f12*b. Returns (rules, graphs) where graphs are the
    merged per-molecule covers and rules = [(pair_type, dC, f12, merged_size), ...] in merge order.
    freeze_rings / start are passed to _finest_graph (see there).

    Incremental: the global adjacency-pair count is maintained exactly, re-scanning ONLY the graphs a
    merge actually touches (not the whole corpus each round). Output is identical to a full recount."""
    graphs: List[_MolGraph] = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            graphs.append(_finest_graph(m, freeze_rings=freeze_rings, start=start))

    type_size: Dict[str, int] = {}
    gpairs: List[Counter] = [_local_pairs(g, type_size) for g in graphs]
    f12: Counter = Counter()
    for lp in gpairs:
        f12.update(lp)

    rules: List[Tuple[Tuple[str, str], float, int, int]] = []
    for step in range(max_merges):
        best_p, best_dC, best_msz = None, 0.0, 0             # accept only dC < 0, pick most negative
        for p, c in f12.items():
            if c <= 0:
                continue
            msz = type_size[p[0]] + type_size[p[1]]          # |M1|+|M2| (heavy atoms)
            if max_atoms is not None and msz > max_atoms:
                continue
            dC = lam * msz - c * b
            if dC < best_dC:
                best_p, best_dC, best_msz = p, dC, msz
        if best_p is None:
            break

        cnt = f12[best_p]
        applied = 0
        for gi, g in enumerate(graphs):                      # only graphs containing best_p change
            if gpairs[gi].get(best_p, 0) == 0:
                continue
            f12.subtract(gpairs[gi])                         # remove this graph's old contribution
            applied += g.apply_merge(best_p)                 # BPE: merge best_p everywhere in g
            gpairs[gi] = _local_pairs(g, type_size)          # rescan ONLY this graph
            f12.update(gpairs[gi])                           # add its new contribution
        if applied == 0:
            break                                            # defensive: no progress -> stop
        rules.append((best_p, best_dC, cnt, best_msz))
        if verbose and (step < 15 or step % 25 == 0):
            print(f"  merge {step:3d}: {best_p[0]:>14s} + {best_p[1]:<14s} "
                  f"f12={cnt:5d} size={best_msz:2d} dC={best_dC:9.1f}", flush=True)

    return rules, graphs


# ───────────────────────────── vocab-pipeline entry point ──────────────────────
def build(smiles_all: Sequence[str], head_source: str = 'none', linker_method: str = 'mdl',
          break_fused_rings: bool = False, lam: float = 1.0, b: float = 1.0,
          max_atoms: int | None = None, max_merges: int = 8000, verbose: bool = False,
          freeze_rings: bool = True, start: str = 'atomic'
          ) -> List[List[Tuple[str, Set[int]]]]:
    """Per-mol [(key, set(atoms))] aligned to smiles_all. head_source/linker_method/break_fused_rings
    accepted for call-compatibility and ignored. lam == theta (b fixed to 1.0 is the natural choice).
    freeze_rings (default True) -> bare ring tokens; start 'atomic' (default) or 'union' (SFO pool)."""
    idx: List[int] = []
    valid: List[str] = []
    for i, s in enumerate(smiles_all):
        if Chem.MolFromSmiles(s) is not None:
            valid.append(s)
            idx.append(i)
        else:
            print(f"[fm_bpe] WARN unparseable SMILES at row {i}: {s!r}", file=sys.stderr)

    _, graphs = learn_fm_bpe(valid, lam=lam, b=b, max_atoms=max_atoms, max_merges=max_merges,
                              verbose=verbose, freeze_rings=freeze_rings, start=start)

    covers: List[List[Tuple[str, Set[int]]]] = [[] for _ in smiles_all]
    for gi, i in enumerate(idx):
        covers[i] = [(k, set(at)) for k, at in graphs[gi].fragments()]
    return covers
