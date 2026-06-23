#!/usr/bin/env python3
"""
molfragbpe5.py  —  MolFragBPE v5
=================================
Molecular fragmentation + hierarchy + BPE.

Fragmentation pipeline per molecule (first-match-wins at every level)
-----------------------------------------------------------------------
1. Chemistry cascade on whole molecule, first-match-wins:
       rBRICS → BRICS → RECAP → Murcko
   The first method that produces ≥ 2 fragments wins.  Its children are
   each recursed on with the same full chemistry cascade.
2. fragment_recursive applies the same cascade recursively at every depth.
   First method that fires at each node wins; recursion stops when no
   method produces ≥ 2 fragments (the fragment becomes a leaf).
3. Structural fallback — applied at most once, only if the entire chemistry
   cascade left the molecule as exactly one leaf:
       Fallback A: ring/chain boundary cut   (ring AND chain atoms present)
       Fallback B: all single bonds cut      (fully acyclic molecules only)
   The fallback result is never recursed into.

BPE — prevalence-guided upward merging
----------------------------------------
Merges current-vocabulary children back into their hierarchy parent P
when ALL seven guards hold:
  G1. support[P] >= min_abs            — P appeared in enough molecules
  G2. atom_count(P) <= sz_max          — not too large for GNN
  G3. frag_diameter(P) <= 2*gnn_layers — fits in GNN receptive field
  G4. not (pure acyclic AND diam > gnn_layers)  — no chain blowup
  G5. P is not already a current vocabulary leaf
  G6. >= 1 current-vocab child has atom_count < min_atoms
  G7. All current-vocab children of P co-occur in >= min_abs molecules
Priority: n_cooc × support[P].  One merge per iteration to convergence.

Per-function contract:
  strip()             normalises [16*]/bare-* → [*]
  atom_count()        excludes wildcard (atomicNum 0) and H (atomicNum 1)
  frag_diameter()     longest shortest path between any two heavy atoms
  cut_rbrics()        L51 (nitro), L7a/b, L81 + standard BRICS environments
  cut_brics()         standard BRICS; does NOT cut Ar-NO2
  cut_recap()         ArN amine (excludes N+/N=O), amide, ester, urea, sulfonamide
  cut_murcko()        scaffold + ALL side chains; no atoms lost
  cut_ring_chain()    ring/non-ring boundary; guard: ring_atoms ∩ chain_atoms ≠ ∅
  cut_acyclic_bonds() all single bonds; guard: no ring atoms present
  fragment_recursive()  first-match-wins chemistry cascade, recursive
  fragment_molecule()   full pipeline: cascade + identity + fallback
  bpe_merge()           prevalence-guided upward, one merge per iteration
"""

import os, sys, re, json, time, copy, argparse, warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold

try:
    from rBRICS_public import FindrBRICSBonds, FindreBRICSBonds, BreakrBRICSBonds, reBRICS
    RBRICS_OK = True
except ImportError:
    RBRICS_OK = False
    warnings.warn("rBRICS_public.py not found — using BRICS as primary method")

# Shared BRICS / rBRICS bond discovery (single source of truth with chemfrag.py
# and the legacy tracked engine in generate_vocab_rules.py).
import brics_rbrics as _BR

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
MIN_FRAG_ATOMS = 3    # fragments below this are BPE merge candidates
SZ_MAX         = 18   # max atom count after a BPE merge
GNN_LAYERS     = 3    # GNN depth; BPE allows diameter ≤ 2 × GNN_LAYERS
BPE_MIN_ABS    = 5    # minimum absolute support for BPE merge
TOP_N          = 12   # top fragments to report in stats

TRIVIAL: Set[str] = {
    '[*]C[*]', '[*]N[*]', '[*]O[*]', '[*]S[*]',
    '[*]N([*])[*]', '[*]C([*])[*]',
}

# RECAP bond specifications: (compiled_pattern, atom_pos_a, atom_pos_b)
# Bond to cut is between match[atom_pos_a] and match[atom_pos_b].
# Positions verified against GetSubstructMatches output for each pattern.
RECAP_SPECS: List[Tuple] = [s for s in [
    # ArN amine: c(0)–N(1). Excludes N+ (charged/nitro), N bonded to =O
    (Chem.MolFromSmarts('[c:1]-[N;!R;!$(NC=O);!$([N+]);!$([N](=O)):2]'), 0, 1),
    # Amide: C(0)–N(2).  O is at position 1 in the match.
    (Chem.MolFromSmarts('[C;!R:1](=[O:3])-[N;!R:2]'),           0, 2),
    # Ester: C(0)–O(1)
    (Chem.MolFromSmarts('[C;!R:1](=[O:3])-[O;!R;!$(OC=O):2]'), 0, 1),
    # Urea N–C: N(0)–C(1) and C(1)–N(3) — both bonds cut
    (Chem.MolFromSmarts('[N;!R:1]-[C;!R:2](=[O:4])-[N;!R:3]'), 0, 1),
    (Chem.MolFromSmarts('[N;!R:1]-[C;!R:2](=[O:4])-[N;!R:3]'), 1, 3),
    # Sulfonamide: S(0)–N(3).  O atoms are at positions 1 and 2.
    (Chem.MolFromSmarts('[S;!R:1](=[O:3])(=[O:4])-[N;!R:2]'),  0, 3),
] if s[0] is not None]

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

# When True (default), typed dummies ([16*], [4*]) collapse to [*] in motif keys.
# Set False via set_normalize_dummy_wildcards(False) / --preserve_typed_dummies
# in generate_vocab_rules.py to keep attachment-point types distinct.
_NORMALIZE_DUMMY_WILDCARDS = True


def normalize_dummy_wildcards() -> bool:
    return _NORMALIZE_DUMMY_WILDCARDS


def set_normalize_dummy_wildcards(v: bool) -> None:
    global _NORMALIZE_DUMMY_WILDCARDS
    _NORMALIZE_DUMMY_WILDCARDS = v


def strip(s: str) -> str:
    """Normalise wildcard formats to [*] when normalize_dummy_wildcards() is True.

    Input:  any SMARTS/SMILES string
    Output: same string with [16*],[4*],* → [*] (or unchanged when preserving types)
    """
    if not _NORMALIZE_DUMMY_WILDCARDS:
        return s
    s = re.sub(r'\[\d+\*\]', '[*]', s)
    s = re.sub(r'(?<!\[)\*',  '[*]', s)
    return s


def to_mol(smarts: str) -> Optional[Chem.Mol]:
    """Convert fragment SMARTS to a mol by replacing [*] with H.
    Input:  fragment SMARTS string (may contain [*] wildcards)
    Output: RDKit Mol, or None if invalid
    """
    return Chem.MolFromSmiles(smarts.replace('[*]', '[H]'))


def atom_count(smarts: str) -> int:
    """Count heavy atoms, excluding wildcards (atomicNum 0) and H (atomicNum 1).
    Input:  fragment SMARTS string
    Output: integer >= 0
    """
    m = to_mol(smarts)
    if m is None:
        m = Chem.MolFromSmarts(smarts)
    return sum(1 for a in m.GetAtoms()
               if a.GetAtomicNum() not in (0, 1)) if m else 0


def frag_diameter(smarts: str) -> int:
    """Longest shortest path between any two atoms (topological diameter).
    Input:  fragment SMARTS string
    Output: integer >= 0
    """
    try:
        from rdkit.Chem import rdmolops
        m = to_mol(smarts)
        if m is None or m.GetNumAtoms() < 2:
            return 0
        return int(rdmolops.GetDistanceMatrix(m).max())
    except Exception:
        return 0


def has_ring(smarts: str) -> bool:
    """True if the fragment contains at least one ring.
    Input:  fragment SMARTS string
    Output: bool
    """
    m = to_mol(smarts)
    return m is not None and m.GetRingInfo().NumRings() > 0


def is_trivial(smarts: str) -> bool:
    """True if fragment is in TRIVIAL set or has fewer than 2 heavy atoms.
    Input:  fragment SMARTS string
    Output: bool
    """
    return smarts in TRIVIAL or atom_count(smarts) < 2


def canon(mol: Chem.Mol) -> str:
    """Canonical SMILES, no isomeric info.
    Input:  RDKit Mol
    Output: canonical SMILES string
    """
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


# ─────────────────────────────────────────────────────────────────────────────
# HIERARCHY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HNode:
    smarts:     str
    parent:     Optional[str] = None
    children:   Set[str]      = field(default_factory=set)
    support:    int           = 0   # molecules where this fragment appeared
    depth:      int           = 0   # 0=molecule root, 1=rBRICS output, 2+=further
    cut_method: str           = ''  # method that produced this node


class Hierarchy:
    """Global fragment hierarchy built incrementally across all molecules.

    Nodes are fragment SMARTS strings. Molecule roots (depth=0) use canonical
    SMILES. All other nodes are fragment SMARTS with [*] wildcards.
    """

    def __init__(self):
        self.nodes: Dict[str, HNode] = {}

    def touch(self, smarts: str, parent: Optional[str],
              depth: int, method: str = '') -> HNode:
        """Register node if not already present; return it.
        Input:  smarts, parent smarts (or None), depth, cut method label
        Output: HNode
        """
        if smarts not in self.nodes:
            self.nodes[smarts] = HNode(
                smarts=smarts, parent=parent,
                depth=depth, cut_method=method)
        return self.nodes[smarts]

    def add_molecule_root(self, mol_smi: str):
        """Register a molecule root and increment its support.
        Input:  canonical SMILES of the molecule
        """
        node = self.touch(mol_smi, None, depth=0, method='root')
        node.support += 1

    def add_cut(self, parent: str, children: List[str],
                depth: int, method: str):
        """Record a fragmentation step: parent → children.
        Input:  parent smarts, list of child smarts, child depth, method name
        Raises: KeyError if parent has not been registered via touch() first.
        """
        if parent not in self.nodes:
            raise KeyError(
                f"add_cut: parent {parent!r} not in hierarchy. "
                "Call touch() or add_molecule_root() before add_cut().")
        pnode = self.nodes[parent]
        for c in children:
            child_node = self.touch(c, parent, depth, method)
            pnode.children.add(c)
            child_node.support += 1

    def internal_fragment_nodes(self) -> List[str]:
        """Return all fragment-level internal nodes (depth>=1, has [*], has children).
        Output: list of SMARTS strings
        """
        return [s for s, n in self.nodes.items()
                if n.children and '[*]' in s and n.depth >= 1]


# ─────────────────────────────────────────────────────────────────────────────
# CUT PRIMITIVES
# All use FindXBonds + FragmentOnBonds — guaranteed non-overlapping partition.
# Return [] when no cut is possible or fewer than 2 real fragments result.
# ─────────────────────────────────────────────────────────────────────────────

_CACHE: Dict = {}


def _fob(mol: Chem.Mol, bond_idx: List[int]) -> List[str]:
    """FragmentOnBonds → list of fragment SMARTS.
    Input:  RDKit Mol, list of bond indices to cut
    Output: list of SMARTS strings (length >= 2) or [] if cut produces < 2 real frags.
    Dummy-only artefacts ([*][*]) are filtered by atom_count >= 1.
    """
    if not bond_idx:
        return []
    try:
        fm = Chem.FragmentOnBonds(
            mol, bond_idx, addDummies=True,
            dummyLabels=[(0, 0)] * len(bond_idx))
        result = []
        for f in Chem.GetMolFrags(fm, asMols=True):
            fs = strip(canon(f))
            if atom_count(fs) >= 1:
                result.append(fs)
        return result if len(result) >= 2 else []
    except Exception:
        return []


def cut_rbrics(mol: Chem.Mol) -> List[str]:
    """rBRICS: FindrBRICSBonds + BreakrBRICSBonds + reBRICS post-processing.

    Plain (untracked) path used by molfragbpe5 cascades:
      1. FindrBRICSBonds  — environment bonds
      2. BreakrBRICSBonds — break (same breaker as rbrics_old / plot path)
      3. reBRICS          — further split fragments with CCCCCC chains

    Legacy **tracked** ``method='rbrics'`` in generate_vocab_rules.py uses a
    different implementation: pass-1 FragmentOnBonds (_rbrics_pass1_tracked)
    plus _rebrics_pass_tracked on each fragment. Bond sets usually agree; piece
    boundaries and motif keys can differ slightly. Tests compare fragment counts
    as a sanity check, not byte-identical outputs.

    Output uses [*] normalised SMARTS via strip() unless --preserve_typed_dummies.
    Input:  RDKit Mol
    Output: list of fragment SMARTS, or [] if no eligible bonds
    """
    key = ('rb', canon(mol))
    if key in _CACHE:
        return _CACHE[key]
    if not RBRICS_OK:
        _CACHE[key] = []
        return []
    try:
        bonds = list(FindrBRICSBonds(mol))
        if not bonds:
            _CACHE[key] = []
            return []
        broken = BreakrBRICSBonds(mol, bonds)
        frags  = Chem.GetMolFrags(broken, asMols=True)
        frags  = reBRICS(frags)
        res    = [strip(Chem.MolToSmiles(f, isomericSmiles=False, canonical=True))
                  for f in frags if f is not None]
        res    = [r for r in res if r and atom_count(r) >= 1]
        res    = res if len(res) >= 2 else []
    except Exception:
        res = []
    _CACHE[key] = res
    return res



def cut_rbrics_only(mol: Chem.Mol) -> List[str]:
    """rBRICS without reBRICS post-processing.

    Uses FindrBRICSBonds only — no reBRICS chain-breaking step.
    This replicates the old DomainDrivenGlobalExpl codebase behaviour
    (Utils_vocab.fragment_molecule, recursive=True, algorithm='RBRICS')
    which used FindreBRICSBonds (L20/L21 only, cuts almost nothing on
    drug-like molecules). The difference vs cut_rbrics:
      - cut_rbrics:      FindrBRICSBonds + reBRICS (correct, complete usage)
      - cut_rbrics_only: FindrBRICSBonds only, no reBRICS post-processing

    Output uses [*] normalised SMARTS via strip().
    Internal cascade key only (molfragbpe5.build_cascade); not a CLI --method.
    Input:  RDKit Mol
    Output: list of fragment SMARTS, or [] if no eligible bonds
    """
    key = ('rbo', canon(mol))
    if key in _CACHE:
        return _CACHE[key]
    if not RBRICS_OK:
        _CACHE[key] = []
        return []
    try:
        idx = _BR.nonring_bond_indices(mol, _BR.rbrics_bonds(mol))
        if not idx:
            _CACHE[key] = []
            return []
        res = _fob(mol, sorted(idx))
    except Exception:
        res = []
    _CACHE[key] = res
    return res


def cut_brics(mol: Chem.Mol) -> List[str]:
    """Standard BRICS: FindBRICSBonds + FragmentOnBonds.
    Catches biaryl (L16-L16), vinyl/aryl halide, heteroatom pairs.
    Does NOT cut Ar-NO2 (no BRICS environment for nitrogen in nitro group).
    Input:  RDKit Mol
    Output: list of fragment SMARTS, or []
    """
    key = ('br', canon(mol))
    if key in _CACHE:
        return _CACHE[key]
    try:
        idx = _BR.nonring_bond_indices(mol, _BR.brics_bonds(mol))
        if not idx:
            _CACHE[key] = []
            return []
        res = _fob(mol, sorted(idx))
    except Exception:
        res = []
    _CACHE[key] = res
    return res


def cut_recap(mol: Chem.Mol) -> List[str]:
    """RECAP: cut amide, ester, urea, sulfonamide, ArN amine bonds.
    Bond positions in each pattern verified against GetSubstructMatches output.
    ArN amine pattern excludes N+ and N bonded to =O (prevents Ar-NO2 cuts).
    Input:  RDKit Mol
    Output: list of fragment SMARTS, or []
    """
    key = ('rc', canon(mol))
    if key in _CACHE:
        return _CACHE[key]
    bi: Set[int] = set()
    for patt, ai, bj in RECAP_SPECS:
        for match in mol.GetSubstructMatches(patt):
            if ai < len(match) and bj < len(match):
                bond = mol.GetBondBetweenAtoms(match[ai], match[bj])
                if bond and not bond.IsInRing():
                    bi.add(bond.GetIdx())
    res = _fob(mol, list(bi)) if bi else []
    _CACHE[key] = res
    return res


def cut_murcko(mol: Chem.Mol) -> List[str]:
    """Murcko scaffold decomposition — complete, no atoms lost.
    Cuts every bond at the scaffold/substituent boundary.
    Returns [scaffold_frag, side_chain_1, side_chain_2, ...].
    Input:  RDKit Mol (with at least one ring + at least one substituent)
    Output: list of fragment SMARTS, or [] if mol has no substituents
    """
    key = ('mc', canon(mol))
    if key in _CACHE:
        return _CACHE[key]
    try:
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        if (sc is None or sc.GetNumAtoms() == 0
                or sc.GetNumAtoms() >= mol.GetNumAtoms()):
            _CACHE[key] = []
            return []
        match = mol.GetSubstructMatch(sc)
        if not match:
            _CACHE[key] = []
            return []
        sc_idx = set(match)
        bi = [b.GetIdx() for b in mol.GetBonds()
              if (b.GetBeginAtomIdx() in sc_idx) !=
                 (b.GetEndAtomIdx() in sc_idx)]
        res = _fob(mol, bi) if bi else []
    except Exception:
        res = []
    _CACHE[key] = res
    return res


def cut_ring_chain(mol: Chem.Mol) -> List[str]:
    """Fallback A: cut every bond at the ring/non-ring boundary.
    Guard: molecule must have BOTH ring atoms AND chain (non-ring) atoms.
    Fires on substituted rings that survived all chemistry-driven methods.
    Input:  RDKit Mol
    Output: list of fragment SMARTS, or [] if fully cyclic or fully acyclic
    """
    key = ('rc_fb', canon(mol))
    if key in _CACHE:
        return _CACHE[key]
    ring_idx  = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    chain_idx = set(range(mol.GetNumAtoms())) - ring_idx
    if not ring_idx or not chain_idx:
        _CACHE[key] = []
        return []
    bi = [b.GetIdx() for b in mol.GetBonds()
          if (b.GetBeginAtomIdx() in ring_idx) !=
             (b.GetEndAtomIdx() in ring_idx)]
    res = _fob(mol, bi) if bi else []
    _CACHE[key] = res
    return res


def cut_acyclic_bonds(mol: Chem.Mol) -> List[str]:
    """Fallback B: cut every single non-ring bond.
    Guard: molecule must have NO ring atoms (fully acyclic).
    Used for perfluoroalkyl chains, nitrate esters, etc.
    Input:  RDKit Mol (fully acyclic)
    Output: list of fragment SMARTS, or [] if mol contains any ring
    """
    key = ('ac_fb', canon(mol))
    if key in _CACHE:
        return _CACHE[key]
    if any(a.IsInRing() for a in mol.GetAtoms()):
        _CACHE[key] = []
        return []
    bi = [b.GetIdx() for b in mol.GetBonds()
          if not b.IsInRing() and b.GetBondTypeAsDouble() == 1.0]
    res = _fob(mol, bi) if bi else []
    _CACHE[key] = res
    return res


# Ordered cascade — chemistry-driven (including rBRICS at every level) first,
# structural fallbacks last.
CHEMISTRY_CASCADE: List[Tuple] = [
    ('rbrics',  cut_rbrics),
    ('rbrics_only', cut_rbrics_only),
    ('brics',   cut_brics),
    ('recap',   cut_recap),
    ('murcko',  cut_murcko),
]

FALLBACK_CASCADE: List[Tuple] = [
    ('ring_chain',    cut_ring_chain),
    ('acyclic_bonds', cut_acyclic_bonds),
]

# Per-method chemistry cascades.
# 'rbrics' — only rBRICS at every level of recursion
# 'brics'  — only BRICS  at every level of recursion
# 'all'    — full rBRICS → BRICS → RECAP → Murcko cascade (default)
# Chemistry-only cascades used by fragment_recursive (first-match-wins).
# Structural fallback methods are NOT included here — they are applied once
# at the top level in fragment_molecule, only for molecules that remain a
# single leaf after the full chemistry cascade has been exhausted.
_METHOD_CASCADES: Dict[str, List[Tuple]] = {
    'rbrics': [('rbrics', cut_rbrics)],
    'rbrics_only': [('rbrics_only', cut_rbrics_only)],
    'brics':  [('brics',  cut_brics)],
    'all':    CHEMISTRY_CASCADE,
}


def build_cascade(method: str) -> List[Tuple]:
    """Return the chemistry cascade for the given method name.

    Args:
        method: 'rbrics' | 'brics' | 'all'
                'rbrics' — rBRICS at every recursion level
                'brics'  — BRICS  at every recursion level
                'all'    — rBRICS → BRICS → RECAP → Murcko (default)
    Returns:
        List of (name, cut_fn) tuples consumed by fragment_recursive.
    Raises:
        ValueError if method is unknown.
    """
    if method not in _METHOD_CASCADES:
        raise ValueError(
            f"Unknown method {method!r}. Choose from: "
            f"{list(_METHOD_CASCADES.keys())}")
    return _METHOD_CASCADES[method]

# ─────────────────────────────────────────────────────────────────────────────
# FIRST-MATCH-WINS CASCADE FRAGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def fragment_recursive(smarts: str, hier: Hierarchy,
                       cascade: List[Tuple],
                       depth: int,
                       max_depth: int = 12) -> List[str]:
    """Apply chemistry cascade to one fragment, first-match-wins.

    Tries each (method_name, cut_fn) pair in cascade order.  The first
    method that produces ≥ 2 non-empty fragments wins: those children are
    registered in the hierarchy and each is recursed on with the same full
    cascade.  No other method is tried once one fires.

    The cascade contains ONLY chemistry methods (rBRICS / BRICS / RECAP /
    Murcko).  Structural fallback methods (ring_chain, acyclic_bonds) are
    applied exclusively in fragment_molecule Step 3, never here.

    Input:
        smarts    — fragment SMARTS string (must contain [*] for sub-fragments)
        hier      — Hierarchy object (mutated in place)
        cascade   — ordered list of (method_name, cut_fn) tuples
        depth     — current depth in the hierarchy (children get depth+1)
        max_depth — hard recursion limit (default 12)
    Output:
        list of leaf fragment SMARTS — fragments that no cascade method cut
    """
    if depth >= max_depth:
        return [smarts]

    fm = to_mol(smarts)
    if fm is None:
        return [smarts]

    for method_name, cut_fn in cascade:
        cuts = [c for c in cut_fn(fm) if atom_count(c) >= 1]
        if len(cuts) < 2:
            continue
        # First method that fires wins — record and recurse.
        # add_cut() registers each child via touch() internally, so no
        # explicit touch() call is needed here.
        hier.add_cut(smarts, cuts, depth + 1, method_name)
        leaves = []
        for child_smi in cuts:
            leaves.extend(
                fragment_recursive(child_smi, hier, cascade, depth + 1, max_depth))
        return leaves if leaves else [smarts]

    # No cascade method produced a cut — this fragment is a leaf
    return [smarts]


def fragment_molecule(mol: Chem.Mol, hier: Hierarchy,
                      use_fallback: bool = True,
                      method: str = 'all') -> List[str]:
    """Full fragmentation pipeline for one molecule.

    Step 1 + 2 (chemistry cascade, first-match-wins at every level):
        For the whole molecule and for each fragment produced at any depth,
        try the cascade methods in order:  rBRICS → BRICS → RECAP → Murcko.
        The first method that produces ≥ 2 fragments wins.  Its children are
        recursed on with the same full chemistry cascade.

    Step 3 (structural fallback — applied at most once):
        If and only if the entire chemistry cascade left the molecule as a
        single leaf, try the structural fallback methods (ring/chain boundary,
        then all-single-bonds).  The first that fires wins.  The fallback
        result is never recursed into — it is the terminal last resort.
        Skipped entirely when use_fallback=False.

    Input:
        mol          — RDKit Mol (from Chem.MolFromSmiles(original_csv_smiles))
        hier         — Hierarchy object (mutated in place)
        use_fallback — whether to apply the structural fallback in Step 3
        method       — 'rbrics' | 'brics' | 'all'  (controls cascade contents)
    Output:
        non-overlapping list of leaf fragment SMARTS strings
    """
    chemistry_cascade = build_cascade(method)
    mol_smi = canon(mol)
    hier.add_molecule_root(mol_smi)

    # ── Step 1 + 2: chemistry cascade, first-match-wins at every level ────────
    # Try cascade methods on the whole molecule in order; the first that
    # produces ≥ 2 fragments wins.  Each child is then recursed on with
    # the same full chemistry cascade (first-match-wins throughout).
    all_leaves = []
    for method_name, cut_fn in chemistry_cascade:
        cuts = [c for c in cut_fn(mol) if atom_count(c) >= 1]
        if len(cuts) < 2:
            continue
        # add_cut() registers each child via touch() internally.
        hier.add_cut(mol_smi, cuts, depth=1, method=method_name)
        for child_smi in cuts:
            all_leaves.extend(
                fragment_recursive(child_smi, hier, chemistry_cascade, depth=1))
        break   # first method that fires wins; stop trying others

    if not all_leaves:
        # Identity leaf when no chemistry cut fired ([*] caps for attachment).
        frag_smi = f'[*]{mol_smi}[*]' if '[*]' not in mol_smi else mol_smi
        hier.touch(frag_smi, mol_smi, depth=1, method='identity')
        hier.nodes[frag_smi].support += 1
        hier.nodes[mol_smi].children.add(frag_smi)
        all_leaves = [frag_smi]

    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique_leaves = []
    for leaf in all_leaves:
        if leaf not in seen:
            seen.add(leaf)
            unique_leaves.append(leaf)
    all_leaves = unique_leaves

    # ── Step 3: structural fallback — last resort, applied exactly once ───────
    # Only fires when the entire chemistry cascade failed to fragment the
    # molecule (single leaf remains).  Never recurses into the fallback result.
    if use_fallback and len(all_leaves) == 1:
        leaf_smi = all_leaves[0]
        leaf_mol = to_mol(leaf_smi)
        if leaf_mol is not None:
            for method_name, cut_fn in FALLBACK_CASCADE:
                cuts = [c for c in cut_fn(leaf_mol) if atom_count(c) >= 1]
                if len(cuts) >= 2:
                    hier.add_cut(leaf_smi, cuts, depth=2, method=method_name)
                    all_leaves = cuts
                    break

    return all_leaves


# ─────────────────────────────────────────────────────────────────────────────
# BPE — PREVALENCE-GUIDED UPWARD MERGING
# ─────────────────────────────────────────────────────────────────────────────

def bpe_merge(mol_frags: List[List[str]],
              hier: Hierarchy,
              n: int,
              min_atoms: int = MIN_FRAG_ATOMS,
              max_diam:  int = GNN_LAYERS,
              sz_max:    int = SZ_MAX,
              min_abs:   int = BPE_MIN_ABS,
              max_child_sup: float = 0.05
              ) -> Tuple[List[List[str]], List[dict]]:
    """Prevalence-guided upward BPE via the fragment hierarchy.

    For each internal fragment node P (depth>=1, contains [*], has children):
    Merge children into P when ALL conditions hold:
      1. support[P] >= min_abs
      2. atom_count(P) <= sz_max
      3. frag_diameter(P) <= max_diam * 2
      4. NOT (pure acyclic chain AND diameter > max_diam)
      5. (removed — P already in vocab is fine; merge skips those molecules)
      6. >= 1 current-vocab child has atom_count < min_atoms
      7. All current-vocab children co-occur in >= min_abs molecules
      8. No non-trivial child has molecular support > max_child_sup (default 5%)
         Trivial children (single-atom, bare linkers) are always safe to merge.

    Priority: n_cooc × (len(children)−1) — total tokens saved across dataset.
    One merge per iteration to convergence.

    Input:
        mol_frags     — list of lists of leaf fragment SMARTS (one per molecule)
        hier          — Hierarchy with support counts from fragmentation
        n             — number of valid (non-None) molecules
        min_atoms     — fragments below this are candidates for merging
        max_diam      — GNN receptive field parameter
        sz_max        — maximum atom count after merge
        min_abs       — minimum absolute co-occurrence count
        max_child_sup — max child molecular support fraction (default 0.05 = 5%)
    Output:
        (updated mol_frags, history list of merge dicts)
    """
    mol_frags = [list(f) for f in mol_frags]
    history: List[dict] = []
    current_vocab: Set[str] = {f for frags in mol_frags for f in frags}

    for iteration in range(100):
        candidates = []

        for p_smi in hier.internal_fragment_nodes():
            pnode = hier.nodes[p_smi]

            # Guard 1: support
            if pnode.support < min_abs:
                continue
            # Guard 2: atom count
            p_atoms = atom_count(p_smi)
            if p_atoms == 0 or p_atoms > sz_max:
                continue
            # Guard 3: diameter
            p_diam = frag_diameter(p_smi)
            if p_diam > max_diam * 2:
                continue
            # Guard 4: chain blowup
            if not has_ring(p_smi) and p_diam > max_diam:
                continue
            # Guard 5 removed: if P is already in current_vocab some molecules
            # already use it as a leaf; the merge loop skips those molecules.
            # build_vocab uses SMARTS as key so there is no duplication.
            cur_children = pnode.children & current_vocab
            # Guard 6: at least one small child
            if not any(atom_count(c) < min_atoms for c in cur_children):
                continue
            # Guard 7: co-occurrence
            n_cooc = sum(1 for frags in mol_frags
                         if cur_children.issubset(set(frags)))
            if n_cooc < min_abs:
                continue

            # Guard 8: block merge if any child is high-support AND non-trivial.
            # A trivial child (single atom, bare linker like [*]C[*]) is safe
            # to absorb even at high support — it carries no standalone signal.
            # A non-trivial high-support child (e.g. [*][N+](=O)[O-]) is a
            # meaningful motif that should remain in the vocabulary unchanged.
            child_sup_ok = True
            for c in cur_children:
                if is_trivial(c):
                    continue          # trivial → always safe to merge
                c_sup = sum(1 for frags in mol_frags if c in set(frags)) / max(n, 1)
                if c_sup > max_child_sup:
                    child_sup_ok = False
                    break
            if not child_sup_ok:
                continue

            # Priority: encoding_reduction = molecules_affected × (n_children−1)
            # Directly maximises total tokens saved across the dataset.
            encoding_reduction = n_cooc * (len(cur_children) - 1)
            candidates.append(
                (encoding_reduction, p_smi, pnode, cur_children, n_cooc))

        if not candidates:
            break

        # Select and execute the highest-priority merge
        candidates.sort(key=lambda x: -x[0])
        _, p_smi, pnode, cur_children, n_cooc = candidates[0]

        n_merged = 0
        for i, frags in enumerate(mol_frags):
            if cur_children.issubset(set(frags)):
                mol_frags[i] = ([f for f in frags if f not in cur_children]
                                + [p_smi])
                n_merged += 1

        if n_merged > 0:
            current_vocab -= cur_children
            current_vocab.add(p_smi)
            small = {c for c in cur_children if atom_count(c) < min_atoms}
            enc_saved = n_merged * (len(cur_children) - 1)
            history.append({
                'iteration':        iteration,
                'parent':           p_smi,
                'children':         sorted(cur_children),
                'small_children':   sorted(small),
                'cut_method':       pnode.cut_method,
                'n_merged':         n_merged,
                'enc_saved':        enc_saved,
                'parent_supp':      pnode.support,
                'parent_atoms':     p_atoms,
                'parent_diam':      p_diam,
                'has_ring':         has_ring(p_smi),
            })

    return mol_frags, history


# ─────────────────────────────────────────────────────────────────────────────
# VOCABULARY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def vocab_stats(mol_frags: List[List[str]], n: int, label: str) -> dict:
    """Compute vocabulary statistics for a set of per-molecule fragment lists.
    Input:
        mol_frags — list of lists of fragment SMARTS
        n         — number of molecules
        label     — string label for this variant
    Output:
        dict with vocab_size, above_1pct, above_5pct, mean_size, enc_length,
        single_frag_mols, size_dist, rfc (receptive field coverage), top_frags
    """
    vc: Dict[str, int] = defaultdict(int)
    for frags in mol_frags:
        for f in frags:
            vc[f] += 1

    fl = list(vc.keys())
    if not fl:
        return {'label': label, 'vocab_size': 0, 'above_1pct': 0,
                'above_5pct': 0, 'mean_size': 0, 'enc_length': 0,
                'single_frag_mols': 0, 'rfc': {}, 'top_frags': []}

    freqs = np.array([vc[f] for f in fl])
    sizes = np.array([atom_count(f) for f in fl])
    sups  = freqs / n
    top_n = sorted(fl, key=lambda f: -vc[f])[:TOP_N]

    return {
        'label':            label,
        'vocab_size':       len(fl),
        'above_1pct':       int((sups >= 0.01).sum()),
        'above_5pct':       int((sups >= 0.05).sum()),
        'mean_size':        round(float(sizes.mean()), 1),
        'enc_length':       int(sum(len(f) for f in mol_frags)),
        'single_frag_mols': int(sum(1 for frags in mol_frags if len(frags) == 1)),
        'size_dist': {
            '1-2':   int(((sizes >= 1) & (sizes <= 2)).sum()),
            '3-5':   int(((sizes >= 3) & (sizes <= 5)).sum()),
            '6-9':   int(((sizes >= 6) & (sizes <= 9)).sum()),
            '10-15': int(((sizes >= 10) & (sizes <= 15)).sum()),
            '16+':   int((sizes >= 16).sum()),
        },
        'rfc': {L: round(sum(1 for f in top_n if frag_diameter(f) <= L)
                         / max(len(top_n), 1), 3)
                for L in [2, 3, 4, 5]},
        'top_frags': [
            {'frag': f, 'support': round(vc[f]/n*100, 2),
             'n_atoms': atom_count(f), 'diam': frag_diameter(f),
             'ring': has_ring(f)}
            for f in top_n
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# DATASET PIPELINE (CLI entry point)
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(data_root: str, dataset: str):
    base = Path(data_root) / f'{dataset}_fold0'
    df   = pd.read_csv(base / 'smiles_labels.csv')
    mols, labels = [], []
    for smi, lbl in zip(df['smiles'], df.get('label', [0]*len(df))):
        m = Chem.MolFromSmiles(smi)
        if m:
            mols.append(m)
            labels.append(int(lbl))
    return mols, np.array(labels)


def run_dataset(dataset: str, data_root: str, out_dir: str,
                min_atoms: int = MIN_FRAG_ATOMS,
                max_diam:  int = GNN_LAYERS,
                sz_max:    int = SZ_MAX,
                max_child_sup: float = 0.05,
                min_abs:   int = BPE_MIN_ABS):
    t0  = time.time()
    out = Path(out_dir) / dataset
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n  {dataset}\n{'='*60}")
    mols, labels = load_dataset(data_root, dataset)
    n    = len(mols)
    npos = int(labels.sum())
    szs  = [m.GetNumAtoms() for m in mols]
    print(f"  n={n}  pos={npos} ({100*npos/n:.1f}%)  "
          f"atoms μ={np.mean(szs):.1f} [{min(szs)}–{max(szs)}]")

    all_stats = {}

    # Variants to run: (method, use_fallback, label)
    # Corresponds to the 4 fragmentation experiment conditions:
    #   1. rBRICS only, no fallback, no BPE
    #   2. rBRICS only, no fallback, with BPE
    #   3. Full cascade, with fallback, no BPE
    #   4. Full cascade, with fallback, with BPE
    VARIANTS = [
        ('rbrics', False, 'rbrics_nofall'),
        ('all',    True,  'all_fallback'),
    ]

    for method_name, use_fb, fb_label in VARIANTS:
        # Clear cache between variants: fragments can be shared but cut
        # results depend on which bonds rBRICS vs full cascade identifies.
        _CACHE.clear()
        hier      = Hierarchy()
        mol_frags = []

        t1 = time.time()
        for i, mol in enumerate(mols):
            leaves = fragment_molecule(mol, hier,
                                       use_fallback=use_fb,
                                       method=method_name)
            mol_frags.append(leaves)
            if (i + 1) % 2000 == 0:
                print(f"    {i+1}/{n}", flush=True)

        s = vocab_stats(mol_frags, n, fb_label)
        all_stats[fb_label] = s

        mf_bpe, hist = bpe_merge(
            copy.deepcopy(mol_frags), hier, n,
            min_atoms=min_atoms, max_diam=max_diam,
            sz_max=sz_max, min_abs=min_abs,
            max_child_sup=max_child_sup)
        sb = vocab_stats(mf_bpe, n, f'{fb_label}+bpe')
        all_stats[f'{fb_label}+bpe'] = sb

        json.dump(hist, open(out / f'bpe_{fb_label}.json', 'w'), indent=2)

        depth_dist: Dict[int, int] = defaultdict(int)
        method_cnt: Dict[str, int] = defaultdict(int)
        for nd in hier.nodes.values():
            depth_dist[nd.depth] += 1
            if nd.cut_method:
                method_cnt[nd.cut_method] += 1

        print(f"\n  [{fb_label}] {time.time()-t1:.1f}s")
        print(f"    Hierarchy: {len(hier.nodes)} nodes  "
              f"{len(hier.internal_fragment_nodes())} internal")
        print(f"    Depths: " + "  ".join(
            f"d{k}:{v}" for k, v in sorted(depth_dist.items())))
        print(f"    Methods: " + "  ".join(
            f"{k}:{v}" for k, v in sorted(method_cnt.items())
            if k not in ('root',)))
        print(f"    Vocab={s['vocab_size']} ≥1%:{s['above_1pct']} "
              f"≥5%:{s['above_5pct']} sz={s['mean_size']} "
              f"single={s['single_frag_mols']} "
              f"({100*s['single_frag_mols']/n:.1f}%)")
        print(f"    BPE: vocab={sb['vocab_size']} ≥1%:{sb['above_1pct']} "
              f"merges={len(hist)}")
        for h in hist[:3]:
            print(f"      [{h['cut_method']}] {sorted(h['children'])} "
                  f"→ {h['parent']}  "
                  f"(supp={h['parent_supp']}, {h['parent_atoms']}a)")

    print(f"\n  {'Label':<22} {'Vocab':>7} {'≥1%':>5} {'≥5%':>5} "
          f"{'Sz':>5} {'Single':>7} {'RFC@3':>6}")
    print(f"  {'-'*58}")
    for lab, s in all_stats.items():
        r3  = round(s['rfc'].get(3, 0) * 100)
        sfp = f"{s['single_frag_mols']}({100*s['single_frag_mols']/n:.1f}%)"
        print(f"  {lab:<22} {s['vocab_size']:>7} {s['above_1pct']:>5} "
              f"{s['above_5pct']:>5} {s['mean_size']:>5} {sfp:>7} {r3:>5}%")

    result = {
        'dataset': dataset, 'n': n, 'n_pos': npos,
        'atom_mean': round(float(np.mean(szs)), 1),
        'stats': all_stats,
    }
    json.dump(result, open(out / 'results.json', 'w'), indent=2)
    print(f"\n  Total: {time.time()-t0:.1f}s  →  {out}")
    return result


def main():
    p = argparse.ArgumentParser(description='MolFragBPE v5')
    p.add_argument('--datasets',  nargs='+', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--out_dir',   default='./molfragbpe5_output')
    p.add_argument('--min_atoms', type=int, default=MIN_FRAG_ATOMS)
    p.add_argument('--max_diam',  type=int, default=GNN_LAYERS)
    p.add_argument('--sz_max',    type=int, default=SZ_MAX)
    p.add_argument('--min_abs',   type=int, default=BPE_MIN_ABS)
    args = p.parse_args()

    for c in [os.getcwd(), os.path.join(os.getcwd(), 'r-BRICS')]:
        sys.path.insert(0, c)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    all_results = {}
    for ds in args.datasets:
        all_results[ds] = run_dataset(
            ds, args.data_root, args.out_dir,
            min_atoms=args.min_atoms, max_diam=args.max_diam,
            sz_max=args.sz_max, min_abs=args.min_abs)

    rows = []
    for ds, res in all_results.items():
        for lab, s in res['stats'].items():
            rows.append({'dataset': ds, 'label': lab,
                         'vocab': s['vocab_size'],
                         'above_1pct': s['above_1pct'],
                         'above_5pct': s['above_5pct'],
                         'mean_size': s['mean_size'],
                         'single_frag_mols': s['single_frag_mols'],
                         'enc_length': s['enc_length'],
                         'rfc_L3': round(s['rfc'].get(3, 0)*100, 1),
                         'rfc_L5': round(s['rfc'].get(5, 0)*100, 1)})
    pd.DataFrame(rows).to_csv(Path(args.out_dir) / 'summary.csv', index=False)
    json.dump(all_results,
              open(Path(args.out_dir) / 'all_results.json', 'w'), indent=2)
    print(f"\nSummary → {args.out_dir}/summary.csv")


if __name__ == '__main__':
    main()
