"""cascade_bpe_linker.py — data-driven linker assembly for FG-first fragmentation.

FG-first sets causal boundaries by CHEMISTRY (functional-group heads, ring bodies) and is
silent on the connective tissue (linkers). The original design cut every linker at every
rBRICS bond and frequency-POOLED the rare pieces into a generic bucket — data only curated
the vocabulary, it did not choose the cut.

This module lets DATA choose the linker cuts, with rBRICS as a PRIOR (not a hard bound):

  finest   : cut the linker tier at EVERY single acyclic bond -> finest legal segmentation.
             Heads (fg:) and ring bodies (ring:) are FROZEN, never touched. Starting from the
             finest (not from rBRICS cuts) means MDL can reach ANY segmentation — it can SPLIT
             a long chain rBRICS never cut (fixes UNDER-fragmentation) as well as MERGE pieces
             rBRICS over-cut (fixes OVER-fragmentation).
  BPE/MDL  : greedily MERGE adjacent linker fragments whenever it REDUCES the description length

                 L = SUM_m a(m)*log2(20)                 (dictionary: store each motif)
                   + SUM_m c(m)*(-log2(c(m)/N))          (corpus: Shannon-code the tokens)
                   + beta * (#retained cuts at NON-rBRICS bonds)   (rBRICS prior)

             a(m)=atoms in motif m, c(m)=corpus frequency, N=SUM c(m). A merge X,Y->Z is
             accepted iff Delta L < 0. The beta term makes a cut at a chemically-standard
             (rBRICS) bond FREE to keep but charges beta bits to keep a cut at an unusual bond,
             so boundaries prefer rBRICS positions while data can override either way. beta=0
             recovers pure MDL; finest='rbrics' recovers the merge-only (over-frag-only) mode.

Only linker fragments are merged; heads and ring bodies stay inert, so the chemistry-first
boundaries are preserved. Merging is by fragment TYPE (BPE), so learned rules replay
deterministically on unseen molecules via apply_rules().
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

from rdkit import Chem

import fg_first_frag as _FG
import chemfrag as _cf   # frag_key: attachment-aware structural keying (merge matches on this)

try:
    import brics_rbrics as _BR
    _RBRICS_OK = True
except Exception:
    _RBRICS_OK = False

LOG2_20 = math.log2(20.0)          # bits to name one heavy atom (~20-symbol alphabet)


# ── description length ────────────────────────────────────────────────────────
def _dl(type_count: Counter, type_atoms: Dict[str, int],
        n_nonrbrics_cuts: int, beta: float) -> float:
    """Total description length: dictionary + corpus + rBRICS-prior on retained cuts.
    corpus_bits = SUM c*(-log2(c/N)) = N*log2 N - SUM c*log2 c (exact, avoids per-motif N)."""
    N = sum(type_count.values())
    if N == 0:
        return 0.0
    dict_bits = sum(type_atoms[t] * LOG2_20 for t in type_count)
    S = sum(c * math.log2(c) for c in type_count.values() if c > 0)
    corpus_bits = N * math.log2(N) - S
    return dict_bits + corpus_bits + beta * n_nonrbrics_cuts


# ── per-molecule fragment graph ───────────────────────────────────────────────
class _MolGraph:
    """Nodes = fragments (frozen heads/bodies + linker pieces at the chosen finest cut).
    Tracks atom sets, identities, mergeability, and per-linker-bond rBRICS legality."""
    __slots__ = ('mol', 'atoms', 'ident', 'frozen', '_rb')

    def __init__(self, mol: Chem.Mol, finest: str = 'all_bonds', freeze: str = 'heads'):
        """freeze='heads' (default): FG heads AND ring bodies are inert, only linkers merge (the original
        linker-only mode). freeze='rings': ONLY rings are inert; FG heads may merge with any non-ring
        neighbour (FGs compose into amide/ester/... under the SAME deltaL<0 criterion). Rings are never
        merged in either mode."""
        self.mol = mol
        # rBRICS-legal bonds (as frozensets), for the prior and for finest='rbrics'
        self._rb: Set[frozenset] = set()
        if _RBRICS_OK:
            self._rb = {frozenset((a, b)) for a, b in _BR.rbrics_full_bonds(mol)}

        # heads/bodies from the partition WITHOUT linker subcut; linker atoms = leftover.
        # whole_ring_systems=True: fused rings stay whole so every ring: motif is a closed cycle
        # (a disjoint split would open one fused ring into an acyclic 'ring:CCC' remnant).
        owner, ids = _FG.partition(mol, subcut_chains=False, whole_ring_systems=True)
        groups: Dict[int, Set[int]] = {}
        for a, f in enumerate(owner):
            groups.setdefault(f, set()).add(a)
        self.atoms, self.ident, self.frozen = [], [], []
        linker_atoms: Set[int] = set()
        # IDENTITY = frag_key (attachment-aware structural key) for every non-ring node, so the merge
        # matches and the labels agree (same structure -> same key throughout; no 'represented twice').
        # Rings keep their substituent-agnostic ring-canonical key (the one exception). The tier tag
        # (ids[f]) is still used HERE to decide freeze/leftover, then discarded.
        for f, at in groups.items():
            if _FG._is_leftover(ids[f]):
                linker_atoms |= at
            else:
                is_ring = ids[f].startswith('ring:')
                key = ids[f] if is_ring else (_cf.frag_key(mol, at) or ids[f])
                inert = is_ring if freeze == 'rings' else True
                self.atoms.append(set(at)); self.ident.append(key); self.frozen.append(inert)

        # finest linker segmentation: cut all single acyclic bonds ('all_bonds') or rBRICS only
        cut: Set[frozenset] = set()
        for b in mol.GetBonds():
            a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if a not in linker_atoms or c not in linker_atoms:
                continue
            key = frozenset((a, c))
            if finest == 'rbrics':
                if key in self._rb:
                    cut.add(key)
            else:  # all_bonds: every single acyclic bond is a candidate boundary
                if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing():
                    cut.add(key)
        for comp in _FG._components(mol, sorted(linker_atoms), cut):
            self.atoms.append(set(comp))
            self.ident.append(_cf.frag_key(mol, set(comp)) or _FG._leftover_identity(mol, comp))
            self.frozen.append(False)

    def _atom_to_node(self) -> Dict[int, int]:
        return {a: ni for ni, at in enumerate(self.atoms) for a in at}

    def linker_adjacencies(self) -> List[Tuple[int, int, bool]]:
        """(i, j, bond_is_rbrics) for distinct linker nodes joined by a bond. Linkers are
        acyclic so distinct linker nodes share exactly one bond."""
        a2n = self._atom_to_node()
        seen: Set[Tuple[int, int]] = set()
        out = []
        for b in self.mol.GetBonds():
            ai, aj = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            i, j = a2n[ai], a2n[aj]
            if i == j or self.frozen[i] or self.frozen[j]:
                continue
            key = (i, j) if i < j else (j, i)
            if key not in seen:
                seen.add(key)
                out.append((key[0], key[1], frozenset((ai, aj)) in self._rb))
        return out

    def n_nonrbrics_cuts(self) -> int:
        return sum(1 for _, _, is_rb in self.linker_adjacencies() if not is_rb)

    def apply_merge(self, pair_type: Tuple[str, str]) -> int:
        target = tuple(sorted(pair_type))
        n = 0
        while True:
            hit = False
            for i, j, _ in self.linker_adjacencies():
                if tuple(sorted((self.ident[i], self.ident[j]))) == target:
                    self._merge_nodes(i, j); n += 1; hit = True
                    break                                   # indices shifted; restart scan
            if not hit:
                return n

    def _merge_nodes(self, i: int, j: int) -> None:
        self.atoms[i] |= self.atoms[j]
        # Re-key the merged node by frag_key -> the merged TYPE is the same attachment-aware key the
        # next merge round matches on, and the final label. Same structure always -> same key.
        self.ident[i] = _cf.frag_key(self.mol, self.atoms[i]) or _FG._leftover_identity(self.mol, sorted(self.atoms[i]))
        for lst in (self.atoms, self.ident, self.frozen):
            lst.pop(j)

    def fragments(self) -> List[Tuple[str, Set[int]]]:
        # Identities are already frag_key (non-ring) / ring-canonical (ring) — one structure, one key.
        # (The old FG-completeness relabel pass is dropped: it fired 0x and injected fg:<name> labels
        # that conflict with the structural keying.)
        return list(zip(self.ident, self.atoms))


# ── corpus-level learner ──────────────────────────────────────────────────────
def learn(smiles: Sequence[str], finest: str = 'all_bonds', beta: float = 4.0,
          max_atoms: Optional[int] = None, max_merges: int = 4000, verbose: bool = False,
          freeze: str = 'heads'):
    """Greedy MDL-BPE over mergeable adjacencies with an rBRICS prior. finest='all_bonds' fixes
    both over- and under-fragmentation; finest='rbrics' is merge-only. beta = bits charged per
    retained non-rBRICS cut. max_atoms caps a brick's size so it stays a small reusable unit
    (prevents MDL over-merging past a causal sub-boundary). freeze='heads' merges linkers only;
    freeze='rings' also lets FG heads compose (rings always inert). Returns (rules, graphs, info)."""
    graphs = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            graphs.append(_MolGraph(m, finest=finest, freeze=freeze))

    def recount():
        tc: Counter = Counter(); ta: Dict[str, int] = {}; nrb = 0
        for g in graphs:
            for idt, at in g.fragments():
                tc[idt] += 1; ta[idt] = len(at)
            nrb += g.n_nonrbrics_cuts()
        return tc, ta, nrb

    tc, ta, nrb = recount()
    L_traj = [_dl(tc, ta, nrb, beta)]
    rules: List[Tuple[str, str]] = []

    for step in range(max_merges):
        # candidate adjacent linker type-pairs: raw multiplicity, #non-rBRICS bonds, result type
        pair_k: Counter = Counter()
        pair_nrb: Counter = Counter()
        pair_z: Dict[Tuple[str, str], Tuple[str, int]] = {}
        for g in graphs:
            for i, j, is_rb in g.linker_adjacencies():
                p = tuple(sorted((g.ident[i], g.ident[j])))
                pair_k[p] += 1
                if not is_rb:
                    pair_nrb[p] += 1
                if p not in pair_z:
                    pair_z[p] = (_FG._leftover_identity(g.mol, sorted(g.atoms[i] | g.atoms[j])),
                                 len(g.atoms[i]) + len(g.atoms[j]))
        if not pair_k:
            break

        L0 = _dl(tc, ta, nrb, beta)

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
            return _dl(nc, nta, nrb - pair_nrb[p], beta) - L0    # merging removes its non-rBRICS cuts

        best_p, best_dL = None, -1e-9
        for p in pair_k:
            if max_atoms is not None and pair_z[p][1] > max_atoms:
                continue                        # keep linker bricks small reusable units
            dL = est_dL(p)
            if dL < best_dL:
                best_p, best_dL = p, dL
        if best_p is None:
            break

        for g in graphs:
            g.apply_merge(best_p)
        tc, ta, nrb = recount()
        L_now = _dl(tc, ta, nrb, beta)
        if L_now >= L_traj[-1]:                 # raw-k estimate too optimistic -> stop (state merged, rule dropped)
            break
        L_traj.append(L_now); rules.append(best_p)
        if verbose and (step < 10 or step % 25 == 0):
            print(f'  merge {step:3d}: {best_p[0]:>11s} + {best_p[1]:<11s} -> {pair_z[best_p][0]:<11s} '
                  f'estDeltaL={best_dL:8.1f}  L={L_now:11.1f}  nonRBcuts={nrb}')

    return rules, graphs, {'L_traj': L_traj, 'n_merges': len(rules), 'finest': finest, 'beta': beta}


def apply_rules(mol: Chem.Mol, rules: Sequence[Tuple[str, str]],
                finest: str = 'all_bonds', freeze: str = 'heads') -> List[Tuple[str, Set[int]]]:
    """Replay learned merge rules (in order) on a new molecule -> [(identity, atoms)].
    freeze MUST match the value used in learn() or the frozen set (and thus replayability) differs."""
    g = _MolGraph(mol, finest=finest, freeze=freeze)
    for r in rules:
        g.apply_merge(r)
    return g.fragments()
