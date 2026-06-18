#!/usr/bin/env python3
"""
generate_vocab_rules.py  —  MotifSAT-compatible output
========================================================
Produces all files required by MotifSAT (github.com/apurvakokate/MotifSAT).

Fragmentation pipeline
-----------------------
1. rBRICS on whole molecule → level-1 fragments
2. Exhaustive cascade on each fragment: rBRICS → BRICS → RECAP → Murcko
3. Structural fallbacks (ring/chain boundary, all-single-bonds) for
   molecules still un-fragmented after the full cascade

Atom tracking
-------------
Every cut uses atom-map numbers to track which ORIGINAL atom indices land in
each fragment. The mapping is propagated through ALL recursion levels by
re-stamping fragment mols with original indices before deeper cuts.
The lookup key is always the original CSV SMILES — never re-canonicalized —
so atom indices match the GNN DataLoader exactly.

Stage 4 vocabulary
-------------------
No filtering. All leaf fragments (including 1-atom trivial pieces) receive a
motif_id. Every atom in every molecule is mapped to a fragment.
MIN_SUP filtering is applied only when building rule extraction candidates.

MotifSAT pickle files (per dataset/variant)
--------------------------------------------
{base}_graph_lookup.pickle         smiles -> {node_idx: (smarts, motif_id)}
{base}_test_graph_lookup.pickle    same for test split
{base}_motif_list.pickle           list[str] SMARTS, index = motif_id
{base}_motif_counts.pickle         list[int] per-motif molecule counts
{base}_motif_length.pickle         list[int] per-motif heavy-atom counts
{base}_motif_class.pickle          {motif_id: {0: n_neg, 1: n_pos}}
{base}_graph_motifidx.pickle       smiles -> set[motif_id] (train)
{base}_test_graph_motifidx.pickle  smiles -> set[motif_id] (test)
"""

import os, sys, re, json, time, copy, pickle, argparse, warnings
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

for _p in [os.getcwd(),
           os.path.join(os.getcwd(), 'r-BRICS'),
           os.path.dirname(os.path.abspath(__file__)),
           os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r-BRICS')]:
    sys.path.insert(0, _p)

import molfragbpe5 as frag
import motif_label_pipeline as pipe

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

try:
    from rBRICS_public import FindrBRICSBonds, FindreBRICSBonds
    RBRICS_OK = True
except ImportError:
    RBRICS_OK = False
    warnings.warn("rBRICS_public.py not found — using BRICS as primary")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET CONFIGURATION
# Maps dataset name → label column name in the CSV.
# CSV format: {data_root}/{dataset}_{fold}.csv
# Add new datasets here.
# ─────────────────────────────────────────────────────────────────────────────
# Unified per-dataset label-column schema (single source of truth shared with
# SharedModules/data/loader.py). Falls back to a local copy if SharedModules is
# not importable (e.g. running vocab generation in isolation), but the values
# MUST match SharedModules/data/dataset_schema.py — keep them identical.
_shared = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'SharedModules')
if _shared not in sys.path:
    sys.path.insert(0, _shared)
# Single source of truth — no local fallback. If SharedModules is not importable
# the run must fail loudly rather than silently use a stale duplicate schema.
from data.dataset_schema import DATASET_COLUMN   # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
# CHOSEN THRESHOLD — per variant × dataset
#
# Key = variant name (= output directory, e.g. "all_fallback_bpe_filter").
# These are the thresholds applied when --apply_threshold is set without
# --threshold_pct.  Edit this dict directly; no shell variable needed.
#
# Threshold semantics: percentage of N_trainval.
#   0.002 → motif must appear in ≥ 0.2% of train+val molecules
#
# Suggested values are derived from the coverage vs threshold elbow plots.
# Aim for the last threshold that keeps node coverage ≥ 80%.
# Datasets where rbrics/rbrics_old never reach 80% use 0.001 (minimum).
# ─────────────────────────────────────────────────────────────────────────────
CHOSEN_THRESHOLD: dict = {

    # ── all + fallback + BPE (filtered) ──────────────────────────────────────
    # High coverage: BPE merges tiny fragments → compact vocab + broad coverage.
    # These thresholds reflect per-dataset elbow points (last ≥ 80% coverage).
    'all_fallback_bpe_filter': {
        'Mutagenicity':      0.002,   # elbow: vocab 212→108, cov 87.6%→81.8%
        'Benzene':           0.006,   # elbow: vocab 273→82,  cov 92.2%→80.6%
        'BBBP':              0.006,   # elbow: vocab 426→118, cov 100%→90.3%
        'hERG':              0.003,
        'Alkane_Carbonyl':   0.003,
        'Fluoride_Carbonyl': 0.003,
        'esol':              0.002,
        'Lipophilicity':     0.003,
        'freesolv':          0.003,
        'tox21':             0.002,
    },

    # ── rBRICS + reBRICS (filtered) ──────────────────────────────────────────
    # rBRICS produces more/larger fragments than BPE → coverage drops faster.
    # BBBP: minority rescue keeps vocab large; first ≥80% point is 0.003-0.004.
    # Mutagenicity + Benzene: coverage never reaches 80% — use minimum (0.001).
    'rbrics_filter': {
        'Mutagenicity':      0.001,   # ← never ≥80% at any threshold
        'Benzene':           0.001,   # ← never ≥80% at any threshold
        'BBBP':              0.004,   # vocab=568, cov=80.3%
        'hERG':              0.002,
        'Alkane_Carbonyl':   0.002,
        'Fluoride_Carbonyl': 0.002,
        'esol':              0.001,
        'Lipophilicity':     0.002,
        'freesolv':          0.002,
        'tox21':             0.001,
    },

    # ── rbrics_only / legacy (filtered) ──────────────────────────────────────
    # Virtually identical to rbrics in practice (reBRICS rarely fires on these
    # datasets).  Coverage curves match rbrics to 3 decimal places.
    'rbrics_old_filter': {
        'Mutagenicity':      0.002,   # ← never ≥80%
        'Benzene':           0.005,   # ← never ≥80%
        'BBBP':              0.006,   # vocab=567, cov=80.3%
        'hERG':              0.005,
        'Alkane_Carbonyl':   0.005,
        'Fluoride_Carbonyl': 0.005,
        'esol':              0.002,
        'Lipophilicity':     0.005,
        'freesolv':          0.005,
        'tox21':             0.005,
    },
}

# Helper: look up the threshold for a given variant + dataset combination.
# Called from run_dataset() when --apply_threshold is set without --threshold_pct.
def get_chosen_threshold(variant: str, dataset: str) -> float:
    """Return the threshold from CHOSEN_THRESHOLD[variant][dataset].

    Raises KeyError with a helpful message if the combination is not in the dict.
    Add the entry to CHOSEN_THRESHOLD to fix the error.
    """
    if variant not in CHOSEN_THRESHOLD:
        raise KeyError(
            f"No CHOSEN_THRESHOLD entry for variant='{variant}'. "
            f"Available variants: {list(CHOSEN_THRESHOLD.keys())}. "
            f"Add '{variant}' to the CHOSEN_THRESHOLD dict in generate_vocab_rules.py."
        )
    if dataset not in CHOSEN_THRESHOLD[variant]:
        raise KeyError(
            f"No CHOSEN_THRESHOLD entry for variant='{variant}', dataset='{dataset}'. "
            f"Add '{dataset}' under CHOSEN_THRESHOLD['{variant}'] in generate_vocab_rules.py."
        )
    return CHOSEN_THRESHOLD[variant][dataset]


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
MIN_FRAG_ATOMS = 3
SZ_MAX         = 18
GNN_LAYERS     = 3
BPE_MIN_ABS    = 5
MIN_SUP        = 0.01  # used only for rule extraction, NOT vocabulary filtering
TOP_N          = 10
MIN_COV        = 5.0

# ─────────────────────────────────────────────────────────────────────────────
# ATOM-TRACKED FRAGMENTATION
#
# Core approach: stamp ORIGINAL atom indices as atom-map numbers on the mol
# before every FragmentOnBonds call. After fragmentation, read atom-map numbers
# from each output fragment to recover original indices. At deeper levels,
# re-stamp the fragment mol using the stored {fragment_atom_idx: orig_idx}
# mapping so original indices are always recoverable.
# ─────────────────────────────────────────────────────────────────────────────

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ LEGACY ATOM-TRACKED FRAGMENTATION — RE-ENABLED for one-shot/two-shot      ║
# ║ methods (rbrics / rbrics_old / rbrics_only / brics).                      ║
# ║                                                                          ║
# ║ run_dataset() routes by --method:                                        ║
# ║   'all'         -> v4 cascade + MDL merge (chemfrag_v4_adapter)           ║
# ║   'rbrics'      -> rBRICS + reBRICS  (two-shot, flat, no tree/merge)      ║
# ║   'rbrics_old'/ -> rBRICS only       (one-shot, flat, no tree/merge)      ║
# ║   'rbrics_only'                                                           ║
# ║   'brics'       -> BRICS only        (flat, no tree/merge)                ║
# ║ The functions below implement the flat legacy path and ARE called for    ║
# ║ the legacy methods. They feed build_vocab + support --apply_threshold     ║
# ║ identically to the original pipeline.                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _fob_tracked(mol_mapped: Chem.Mol,
                 bond_indices: List[int]
                 ) -> List[Tuple[str, Set[int], Dict[int, int]]]:
    """FragmentOnBonds on a mol that already has atom-map numbers encoding
    original atom indices (map_num = orig_idx + 1).

    Input:
        mol_mapped   — RDKit Mol with atom-map numbers set to orig_idx + 1
        bond_indices — list of bond indices to cut
    Output:
        list of (fragment_smarts, {orig_atom_indices}, {frag_atom_idx: orig_idx})
        Returns [] if fewer than 2 real fragments result.
    """
    if not bond_indices:
        orig = {a.GetAtomMapNum() - 1 for a in mol_mapped.GetAtoms()
                if a.GetAtomicNum() not in (0, 1) and a.GetAtomMapNum() > 0}
        idx_map = {a.GetIdx(): a.GetAtomMapNum() - 1
                   for a in mol_mapped.GetAtoms()
                   if a.GetAtomicNum() not in (0, 1) and a.GetAtomMapNum() > 0}
        smi = frag.strip(Chem.MolToSmiles(mol_mapped, canonical=True,
                                           isomericSmiles=False))
        return [(smi, orig, idx_map)]

    try:
        fm = Chem.FragmentOnBonds(
            mol_mapped, bond_indices, addDummies=True,
            dummyLabels=[(0, 0)] * len(bond_indices))
    except Exception:
        orig = {a.GetAtomMapNum() - 1 for a in mol_mapped.GetAtoms()
                if a.GetAtomicNum() not in (0, 1) and a.GetAtomMapNum() > 0}
        idx_map = {a.GetIdx(): a.GetAtomMapNum() - 1
                   for a in mol_mapped.GetAtoms()
                   if a.GetAtomicNum() not in (0, 1) and a.GetAtomMapNum() > 0}
        smi = frag.strip(Chem.MolToSmiles(mol_mapped, canonical=True,
                                           isomericSmiles=False))
        return [(smi, orig, idx_map)]

    result = []
    for f in Chem.GetMolFrags(fm, asMols=True):
        orig_set: Set[int] = set()
        frag_idx_map: Dict[int, int] = {}
        for atom in f.GetAtoms():
            if atom.GetAtomicNum() != 0 and atom.GetAtomMapNum() > 0:
                orig_idx = atom.GetAtomMapNum() - 1
                orig_set.add(orig_idx)
                frag_idx_map[atom.GetIdx()] = orig_idx
        if orig_set:
            # Clear atom maps to get clean canonical SMARTS
            rw = Chem.RWMol(f)
            for a in rw.GetAtoms():
                a.SetAtomMapNum(0)
            fs = frag.strip(Chem.MolToSmiles(
                rw.GetMol(), canonical=True, isomericSmiles=False))
            result.append((fs, orig_set, frag_idx_map))

    # A dummy-only fragment (orig_set empty) is already excluded by `if orig_set`
    # above. Return whatever real fragments remain as long as at least one exists.
    # Callers that need >=2 pieces (e.g. _cascade_tracked) check len() themselves.
    return result


def _stamp_mol(smarts: str, idx_map: Dict[int, int]) -> Optional[Chem.Mol]:
    """Create mol from smarts and stamp atom-map numbers from idx_map.
    idx_map: {fragment_atom_idx: original_atom_idx}.
    Used to carry original indices into deeper fragmentation levels.

    Input:
        smarts  — fragment SMARTS (no atom maps)
        idx_map — {frag_atom_idx: orig_idx} from previous _fob_tracked call
    Output:
        RDKit Mol with atom-map numbers set, or None if smarts invalid
    """
    m = frag.to_mol(smarts)
    if m is None:
        return None
    rw = Chem.RWMol(m)
    for atom in rw.GetAtoms():
        fi = atom.GetIdx()
        if fi in idx_map:
            atom.SetAtomMapNum(idx_map[fi] + 1)
    return rw.GetMol()


def _bond_indices_for(mol: Chem.Mol, cut_fn) -> List[int]:
    """Extract bond indices that cut_fn would cut, without cutting.
    Input:  RDKit Mol (clean, no atom maps), cut function
    Output: list of bond indices
    """
    if cut_fn == frag.cut_rbrics:
        if not RBRICS_OK:
            return []
        try:
            bonds = list(FindrBRICSBonds(mol))
            return [mol.GetBondBetweenAtoms(a, b).GetIdx()
                    for (a, b), _ in bonds if mol.GetBondBetweenAtoms(a, b)]
        except Exception:
            return []

    elif cut_fn == frag.cut_brics:
        from rdkit.Chem import BRICS
        try:
            bonds = list(BRICS.FindBRICSBonds(mol))
            return [mol.GetBondBetweenAtoms(a, b).GetIdx()
                    for (a, b), _ in bonds if mol.GetBondBetweenAtoms(a, b)]
        except Exception:
            return []

    elif cut_fn == frag.cut_recap:
        bi: Set[int] = set()
        for patt, ai, bj in frag.RECAP_SPECS:
            for match in mol.GetSubstructMatches(patt):
                if ai < len(match) and bj < len(match):
                    b = mol.GetBondBetweenAtoms(match[ai], match[bj])
                    if b and not b.IsInRing():
                        bi.add(b.GetIdx())
        return list(bi)

    elif cut_fn == frag.cut_murcko:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        try:
            sc = MurckoScaffold.GetScaffoldForMol(mol)
            if sc is None or sc.GetNumAtoms() == 0 or \
               sc.GetNumAtoms() >= mol.GetNumAtoms():
                return []
            match = mol.GetSubstructMatch(sc)
            if not match:
                return []
            sc_idx = set(match)
            return [b.GetIdx() for b in mol.GetBonds()
                    if (b.GetBeginAtomIdx() in sc_idx) !=
                       (b.GetEndAtomIdx() in sc_idx)]
        except Exception:
            return []

    elif cut_fn == frag.cut_ring_chain:
        ri = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
        if not ri or ri == set(range(mol.GetNumAtoms())):
            return []
        return [b.GetIdx() for b in mol.GetBonds()
                if (b.GetBeginAtomIdx() in ri) != (b.GetEndAtomIdx() in ri)]

    elif cut_fn == frag.cut_acyclic_bonds:
        if any(a.IsInRing() for a in mol.GetAtoms()):
            return []
        return [b.GetIdx() for b in mol.GetBonds()
                if not b.IsInRing() and b.GetBondTypeAsDouble() == 1.0]

    elif cut_fn == frag.cut_rbrics_only:
        if not RBRICS_OK:
            return []
        try:
            bonds = list(FindrBRICSBonds(mol))
            return [mol.GetBondBetweenAtoms(a, b).GetIdx()
                    for (a, b), _ in bonds if mol.GetBondBetweenAtoms(a, b)]
        except Exception:
            return []

    return []


# ── _cascade_tracked: DISABLED (legacy is a single chemistry pass). ──────────
# Kept commented for reference — this was the recursive, exhaustive
# first-match-wins cascade used when legacy fragmentation was multi-level. The
# active legacy path (fragment_molecule_tracked) now performs a single BRICS or
# rBRICS cut (+ optional reBRICS sub-pass) only. Re-enable by uncommenting this
# function and the Step 2/Step 3 block in fragment_molecule_tracked.
#
# def _cascade_tracked(smarts: str,
#                      orig_set: Set[int],
#                      idx_map: Dict[int, int],
#                      cascade: List,
#                      depth: int = 0,
#                      max_depth: int = 12
#                      ) -> List[Tuple[str, Set[int]]]:
#     """Apply cascade methods exhaustively, tracking original atom indices."""
#     if depth >= max_depth or not orig_set:
#         return [(smarts, orig_set)]
#     fm_clean = frag.to_mol(smarts)
#     if fm_clean is None:
#         return [(smarts, orig_set)]
#     fm_mapped = _stamp_mol(smarts, idx_map)
#     if fm_mapped is None:
#         return [(smarts, orig_set)]
#     # Strict first-match-wins: first method yielding >=2 non-overlapping pieces
#     # covering orig_set wins.
#     deduped: List[Tuple[str, Set[int], Dict]] = []
#     for cut_fn in cascade:
#         bond_idx = _bond_indices_for(fm_clean, cut_fn)
#         if not bond_idx:
#             continue
#         pieces = _fob_tracked(fm_mapped, bond_idx)
#         if len(pieces) < 2:
#             continue
#         covered: Set[int] = set()
#         ok = True
#         for _, p_orig, _ in pieces:
#             if p_orig & covered:
#                 ok = False
#                 break
#             covered |= p_orig
#         if not ok or covered != orig_set:
#             continue
#         deduped = list(pieces)
#         break
#     if not deduped:
#         return [(smarts, orig_set)]
#     leaves: List[Tuple[str, Set[int]]] = []
#     for p_smi, p_orig, p_map in deduped:
#         leaves.extend(
#             _cascade_tracked(p_smi, p_orig, p_map, cascade, depth + 1, max_depth))
#     return leaves if leaves else [(smarts, orig_set)]


def fragment_molecule_tracked(mol: Chem.Mol,
                               orig_smi: str,
                               use_fallback: bool,
                               method: str = 'rbrics'
                               ) -> List[Tuple[str, Set[int]]]:
    """Single-pass atom-tracked fragmentation for the legacy methods.

    Uses orig_smi (exact CSV SMILES) to create mol, ensuring atom indices
    match the GNN DataLoader (which also calls Chem.MolFromSmiles(orig_smi)).

    Legacy is intentionally ONE chemistry pass (no recursive cascade, no
    structural fallback):
        brics        — BRICS bonds
        rbrics_only  — rBRICS bonds
        rbrics       — rBRICS bonds + reBRICS sub-pass (long aliphatic chains)
    BRICS/rBRICS bond discovery is delegated to the shared ``brics_rbrics``
    module — single source of truth with the v4 cascade (chemfrag.py).

    Input:
        mol          — Chem.MolFromSmiles(orig_smi)
        orig_smi     — original CSV SMILES string (key for lookup dict)
        use_fallback — accepted for signature compatibility; the legacy
                       structural fallback is DISABLED (see commented block).
        method       — 'rbrics' | 'rbrics_only' | 'brics'
    Output:
        list of (fragment_smarts, {original_atom_indices}) covering ALL n atoms
    """
    import brics_rbrics as BR
    n = mol.GetNumAtoms()

    # Stamp all atoms with original indices (idx + 1, 1-indexed) for tracking.
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    mol_mapped = rw.GetMol()

    # Clean (unmapped) mol for bond discovery — same atom indexing as `mol`.
    mol_clean = frag.to_mol(frag.strip(orig_smi)) or mol

    # Single primary cut: one BRICS or rBRICS bond set over the whole molecule.
    if method == 'brics':
        idx1 = BR.nonring_bond_indices(mol_clean, BR.brics_bonds(mol_clean))
    else:  # 'rbrics' / 'rbrics_only'
        idx1 = BR.nonring_bond_indices(mol_clean, BR.rbrics_bonds(mol_clean))
        if not idx1:                       # rBRICS unavailable / nothing to cut
            idx1 = BR.nonring_bond_indices(mol_clean, BR.brics_bonds(mol_clean))

    if idx1:
        level1 = _fob_tracked(mol_mapped, sorted(idx1))
    else:
        level1 = [(frag.strip(frag.canon(mol)), set(range(n)),
                   {i: i for i in range(n)})]

    # reBRICS sub-pass (rbrics only): split long aliphatic CCCCCC chains within
    # each level-1 piece, with atom tracking preserved. Uses the shared module.
    if method == 'rbrics' and BR.RBRICS_OK:
        rebrics_level1 = []
        for p_smi, p_orig, p_map in level1:
            p_mol = frag.to_mol(p_smi)
            if (p_mol is None or p_mol.GetNumHeavyAtoms() <= 5
                    or not p_mol.HasSubstructMatch(Chem.MolFromSmiles('CCCCCC'))):
                rebrics_level1.append((p_smi, p_orig, p_map))
                continue
            p_mapped = _stamp_mol(p_smi, p_map)
            re_idx = BR.nonring_bond_indices(p_mol, BR.rebrics_bonds(p_mol))
            sub_pieces = (_fob_tracked(p_mapped, sorted(re_idx))
                          if (re_idx and p_mapped is not None) else [])
            if len(sub_pieces) >= 2:
                rebrics_level1.extend(sub_pieces)
            else:
                rebrics_level1.append((p_smi, p_orig, p_map))
        level1 = rebrics_level1

    all_pieces: List[Tuple[str, Set[int]]] = [(s, o) for s, o, _ in level1]

    # ── Step 2 (recursive exhaustive cascade via _cascade_tracked) and Step 3
    #    (structural fallback) are DISABLED: legacy is a single chemistry pass.
    #    The previous behaviour is preserved, commented, below (and in the
    #    also-commented _cascade_tracked definition) for reference.
    #
    # chemistry_fns = [cut_fn for _, cut_fn in frag.build_cascade(method)]
    # FALLBACK      = [frag.cut_ring_chain, frag.cut_acyclic_bonds]
    # cascade = chemistry_fns + (FALLBACK if use_fallback else [])
    # all_pieces = []
    # for p_smi, p_orig, p_map in level1:
    #     all_pieces.extend(_cascade_tracked(p_smi, p_orig, p_map, cascade))
    #
    # if use_fallback and len(all_pieces) == 1:
    #     map0 = level1[0][2]
    #     fm0_clean  = frag.to_mol(all_pieces[0][0])
    #     fm0_mapped = _stamp_mol(all_pieces[0][0], map0)
    #     if fm0_clean is not None and fm0_mapped is not None:
    #         for cut_fn in FALLBACK:
    #             bond_idx2 = _bond_indices_for(fm0_clean, cut_fn)
    #             if not bond_idx2:
    #                 continue
    #             pieces2 = _fob_tracked(fm0_mapped, bond_idx2)
    #             if len(pieces2) >= 2:
    #                 covered = set(); ok = True
    #                 for _, p_orig2, _ in pieces2:
    #                     if p_orig2 & covered:
    #                         ok = False; break
    #                     covered |= p_orig2
    #                 if ok and covered == all_pieces[0][1]:
    #                     all_pieces = [(s, o) for s, o, _ in pieces2]
    #                     break

    # Guarantee: every atom is covered exactly once.
    covered = {a for _, atoms in all_pieces for a in atoms}
    missing = set(range(n)) - covered
    if missing and all_pieces:
        all_pieces[0] = (all_pieces[0][0], all_pieces[0][1] | missing)
    elif missing:
        all_pieces = [(frag.strip(frag.canon(mol)), set(range(n)))]

    return all_pieces


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ END LEGACY ATOM-TRACKED FRAGMENTATION (DISABLED).                        ║
# ║ run_dataset() uses chemfrag_v4_adapter.fragment_tracked_v4 instead.      ║
# ╚══════════════════════════════════════════════════════════════════════════╝


# ─────────────────────────────────────────────────────────────────────────────
# VOCABULARY — NO FILTERING
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab(mol_frags_tracked: List[List[Tuple[str, Set[int]]]],
                labels: np.ndarray,
                groups: List[str] = None,
                min_sup_for_rules: float = MIN_SUP
                ) -> Tuple[List[str], Dict[str, int], Dict[str, dict]]:
    """Build full vocabulary from all leaf fragments — no minimum support filter.
    Every fragment (including trivial 1-atom pieces) receives a motif_id.

    Input:
        mol_frags_tracked — per-molecule list of (smarts, {atom_indices})
        labels            — integer label array (same length)
        min_sup_for_rules — threshold used to flag above_min_sup in stats
    Output:
        motif_list   list[str] sorted by count descending (index = motif_id)
        frag_to_id   {smarts: motif_id}
        motif_stats  {smarts: {count, n_pos, n_neg, n_atoms, ring, above_min_sup}}
    """
    n = len(mol_frags_tracked)
    raw: Dict[str, Dict] = defaultdict(
        lambda: {'count': 0, 'n_pos': 0, 'n_neg': 0,
                 'n_occurrences': 0, 'weighted_count': 0.0,
                 'wt_count_0': 0.0, 'wt_count_1': 0.0})

    for i, (mol_frags, lbl) in enumerate(zip(mol_frags_tracked, labels)):
        lbl_int = int(lbl)
        seen: Set[str] = set()
        is_tv = (groups is None) or (groups[i] in ('training', 'valid'))
        for smarts, atom_set in mol_frags:
            n_at = max(frag.atom_count(smarts), 1)
            w    = n_at * (1.0 / n_at)   # = 1.0
            raw[smarts]['n_occurrences'] += 1
            if is_tv:
                raw[smarts]['weighted_count'] += w
                if lbl_int == 0:
                    raw[smarts]['wt_count_0'] += w
                else:
                    raw[smarts]['wt_count_1'] += w
            if smarts not in seen:
                raw[smarts]['count'] += 1
                raw[smarts]['n_pos' if lbl_int == 1 else 'n_neg'] += 1
                seen.add(smarts)

    kept = sorted(raw.keys(), key=lambda s: -raw[s]['count'])
    frag_to_id = {s: i for i, s in enumerate(kept)}
    motif_stats = {s: {
        **raw[s],
        'n_atoms':        frag.atom_count(s),
        'ring':           frag.has_ring(s),
        'above_min_sup':  raw[s]['count'] / n >= min_sup_for_rules,
        'n_occurrences':  raw[s]['n_occurrences'],
        'weighted_count': raw[s]['weighted_count'],
        'wt_count_0':     raw[s]['wt_count_0'],
        'wt_count_1':     raw[s]['wt_count_1'],
    } for s in kept}

    return kept, frag_to_id, motif_stats


def build_lookup(smiles_list: List[str],
                 mol_frags_tracked: List[List[Tuple[str, Set[int]]]],
                 frag_to_id: Dict[str, int],
                 threshold_motifs: Optional[Set[str]] = None
                 ) -> Dict[str, Dict[int, Tuple[str, int]]]:
    """Build MotifSAT lookup dict.
    lookup[original_smiles][node_idx] = (fragment_smarts, motif_id)

    When threshold_motifs is provided (apply_threshold=True):
      - Motifs IN threshold_motifs  → motif_id as normal (≥ 0)
      - Motifs NOT IN threshold_motifs → motif_id = -1  (unknown)
    When threshold_motifs is None → all motifs keep their real motif_id.

    Input:
        smiles_list       — list of original CSV SMILES strings
        mol_frags_tracked — per-molecule list of (smarts, {atom_indices})
        frag_to_id        — {smarts: motif_id}
        threshold_motifs  — set of motifs that pass the threshold, or None
    Output:
        {smiles: {atom_idx: (smarts, motif_id)}}  motif_id=-1 for unknowns
    """
    lookup: Dict[str, Dict[int, Tuple[str, int]]] = {}
    for smi, mol_frags in zip(smiles_list, mol_frags_tracked):
        node_map: Dict[int, Tuple[str, int]] = {}
        for smarts, atom_set in mol_frags:
            if threshold_motifs is not None and smarts not in threshold_motifs:
                mid = -1          # below threshold → unknown
            else:
                mid = frag_to_id[smarts]
            for atom_idx in atom_set:
                node_map[atom_idx] = (smarts, mid)
        lookup[smi] = node_map
    return lookup


def build_matrix(mol_frags_tracked: List[List[Tuple[str, Set[int]]]],
                 frag_to_id: Dict[str, int],
                 n_mols: int) -> sp.csr_matrix:
    """Binary molecule × motif matrix (uint8, scipy sparse CSR).
    Entry[i,j] = 1 if molecule i contains motif j.

    Input:
        mol_frags_tracked — per-molecule (smarts, atom_set) lists
        frag_to_id        — {smarts: motif_id}
        n_mols            — total number of molecules (including invalid)
    Output:
        scipy.sparse.csr_matrix shape (n_mols, n_vocab)
    """
    rows, cols = [], []
    for i, mol_frags in enumerate(mol_frags_tracked):
        seen: Set[str] = set()
        for smarts, _ in mol_frags:
            if smarts in frag_to_id and smarts not in seen:
                rows.append(i)
                cols.append(frag_to_id[smarts])
                seen.add(smarts)
    return sp.csr_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(n_mols, len(frag_to_id)))


# ─────────────────────────────────────────────────────────────────────────────
# RULE EXTRACTION  (MIN_SUP filter applied here only)
# ─────────────────────────────────────────────────────────────────────────────

def extract_rules(motif_list: List[str],
                  motif_stats: Dict[str, dict],
                  X: sp.csr_matrix,
                  labels: np.ndarray,
                  rank_mode: str = 'balanced') -> List[dict]:
    """Extract DNF rules from the fragment × molecule matrix.

    Motifs with motif_id = -1 (below-threshold unknowns) are never rule
    candidates even if they have high coverage — they are an aggregate of many
    rare fragments and not chemically meaningful as a single entity.

    Input:
        motif_list   — full vocabulary list (index = motif_id; -1 rows absent)
        motif_stats  — from build_vocab()
        X            — binary matrix (molecules × motifs)
        labels       — integer label array
        rank_mode    — 'balanced' (default): sort by the structural
                       balance × separation × (1-spurious) score, which targets
                       a 50/50 synthetic split and penalises spurious/subsuming
                       motifs. 'pct1': legacy sort by positive-coverage only.
    Output:
        list of rule dicts. Sorted by the balance-aware score when
        rank_mode='balanced', else by pct1 descending. Every rule carries the
        score components (balance, separation, spurious, score) regardless of
        mode, so the ranking can be inspected and RULE_INDEX overridden.
    """
    n = X.shape[0]
    Xd = X.toarray().astype(bool)
    fi = {s: i for i, s in enumerate(motif_list)}

    # Rule candidates: above MIN_SUP, ≥ 2 atoms, not trivial, not unknown
    rule_cands = [s for s in motif_list
                  if motif_stats[s]['above_min_sup']
                  and frag.atom_count(s) >= 2
                  and s not in frag.TRIVIAL
                  and s != '[UNKNOWN]'   # explicit unknown sentinel excluded
                  and motif_stats[s].get('is_unknown', False) is False]
    if not rule_cands:
        return []

    all_masks = {s: Xd[:, fi[s]] for s in rule_cands}
    all_cands = [(fi[s], s, float(Xd[:, fi[s]].mean())) for s in rule_cands]
    top = [s for s in rule_cands
           if all_masks[s].mean() * 100 >= MIN_COV][:TOP_N]
    if not top:
        return []

    catalog = pipe.build_catalog()
    pipe.compute_alert_families(top, all_cands, catalog)
    sub_fams = pipe.compute_subsuming_families(top, all_cands)
    prof, _ = pipe.cooc_profile(top, all_cands, all_masks)
    proxy   = pipe.build_proxy_lookup(prof)
    clauses = pipe.build_clauses(top, all_masks, prof, n)
    rules   = pipe.build_dnf_rules(
        sorted(clauses, key=lambda x: -x['n1'])[:30],
        all_masks, prof, n, proxy)

    # Per-molecule fragment lists (over the candidate motifs) for structural SNR.
    tv_frags = [[s for s in rule_cands if Xd[r, fi[s]]] for r in range(n)]

    # Always attach the balance-aware score components so the ranking is
    # inspectable; sort by the requested key.
    rules = pipe.score_dnf_rules(rules, all_masks, tv_frags, prof, sub_fams, n)
    if rank_mode == 'pct1':
        return sorted(rules, key=lambda x: -x['pct1'])
    return rules   # already sorted by balance-aware score


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(out_dir: Path, dataset: str, variant: str,
                 smdf: pd.DataFrame,
                 lookup_all: Dict,
                 smiles_all: List[str],
                 groups_all: List[str],
                 motif_list: List[str],
                 motif_stats: Dict[str, dict],
                 frag_to_id: Dict[str, int],
                 X: sp.csr_matrix,
                 lookup_train: dict, lookup_valid: dict, lookup_test: dict,
                 gmi_train: dict, gmi_test: dict,
                 rules: List[dict], labels: np.ndarray,
                 test_occ_ctr: dict = None) -> dict:

    vdir = out_dir / dataset / variant
    vdir.mkdir(parents=True, exist_ok=True)
    base = str(vdir / f'{dataset}_{variant}')
    n    = X.shape[0]

    def dump(obj, path: str):
        with open(path, 'wb') as f:
            pickle.dump(obj, f, protocol=4)

    # MotifSAT pickle files
    dump(lookup_train,  base + '_graph_lookup.pickle')
    dump(lookup_valid,  base + '_valid_graph_lookup.pickle')
    dump(lookup_test,   base + '_test_graph_lookup.pickle')
    dump(motif_list,    base + '_motif_list.pickle')
    dump([motif_stats[s]['count']   for s in motif_list],
                        base + '_motif_counts.pickle')
    dump([motif_stats[s]['n_atoms'] for s in motif_list],
                        base + '_motif_length.pickle')
    dump({i: {0: motif_stats[s]['n_neg'], 1: motif_stats[s]['n_pos']}
          for i, s in enumerate(motif_list)},
                        base + '_motif_class.pickle')
    dump(gmi_train,     base + '_graph_motifidx.pickle')
    dump(gmi_test,      base + '_test_graph_motifidx.pickle')

    # Additional outputs
    sp.save_npz(str(vdir / 'matrix.npz'), X)

    pd.DataFrame([{
        'matrix_col':     frag_to_id[s], 'motif_id': frag_to_id[s],
        'motif_identity': s,
        'support':        round(motif_stats[s]['count'] / n, 6),
        'n_mols':         motif_stats[s]['count'],
        'n_occurrences':  motif_stats[s]['n_occurrences'],
        'weighted_count': motif_stats[s]['weighted_count'],
        'wt_count_0':     motif_stats[s]['wt_count_0'],
        'wt_count_1':     motif_stats[s]['wt_count_1'],
        'n_mols_test':    (test_occ_ctr.get(s, 0) if test_occ_ctr else 0),
        'n_atoms':        motif_stats[s]['n_atoms'],
        'ring':           motif_stats[s]['ring'],
        'above_1pct':     motif_stats[s]['above_min_sup'],
    } for s in motif_list]).to_csv(vdir / 'matrix_columns.csv', index=False)

    pd.DataFrame([{'motif_id': frag_to_id[s], 'motif_identity': s}
                  for s in motif_list]
                 ).to_csv(vdir / 'motif_vocabulary.csv', index=False)

    # Metadata: exact split sizes needed by coverage_vs_threshold.py
    n_tv    = sum(1 for g in groups_all if g in ('training', 'valid'))
    n_test  = sum(1 for g in groups_all if g == 'test')
    n0_tv   = sum(1 for g, l in zip(groups_all, labels)
                  if g in ('training', 'valid') and int(l) == 0)
    n1_tv   = n_tv - n0_tv
    import json as _json
    meta_path = vdir / 'vocab_meta.json'
    with open(meta_path, 'w') as _f:
        _json.dump({
            'n_trainval': n_tv,
            'n_total': n_tv + n_test,
            'n_test':     n_test,
            'n0_trainval': n0_tv,
            'n1_trainval': n1_tv,
            'vocab_size':  len(motif_list),
            'dataset':     dataset,
            'variant':     variant,
        }, _f, indent=2)

    smdf.to_csv(vdir / 'smiles_labels.csv', index=False)

    with open(vdir / 'rules.json', 'w') as f:
        json.dump(rules, f, indent=2)

    rule_rows = []
    for rank, r in enumerate(rules):
        mf = [m for c in r['clauses'] for m in c['motifs']]
        rule_rows.append({
            'rank': rank,
            'score':      r.get('score'),
            'balance':    r.get('balance'),
            'separation': r.get('separation'),
            'spurious':   r.get('spurious'),
            'cover_pct':  r.get('rule_pct_match'),   # coverage over ALL molecules
            'pct1': round(r['pct1'], 2),
            'pct0': round(r['pct0'], 2), 'n1': r['n1'], 'n0': r['n0'],
            'n_clauses': r['n_clauses'],
            'rule_str': ' ∨ '.join(
                '(' + ' ∧ '.join(c['motifs']) + ')' for c in r['clauses']),
            'motifs': '|'.join(mf),
        })
    pd.DataFrame(rule_rows).to_csv(vdir / 'rules_summary.csv', index=False)

    meta = {
        'dataset': dataset, 'variant': variant, 'algorithm': 'molfragbpe5',
        'n_graphs': n, 'n_vocab_motifs': len(motif_list),
        'n_above_1pct': sum(1 for s in motif_list
                            if motif_stats[s]['above_min_sup']),
        'matrix_shape': list(X.shape),
        'n_rules': len(rules),
        'best_rule_pct1': round(rules[0]['pct1'], 2) if rules else 0,
        'best_rule': (' ∨ '.join('(' + ' ∧ '.join(c['motifs']) + ')'
                                  for c in rules[0]['clauses'])
                      if rules else ''),
        'pickle_base': base,
    }
    with open(vdir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # Bool mask cache for motif-removal evaluation (per split, sparse)
    mask_cache = build_mask_cache(smiles_all, groups_all, lookup_all)
    for split_key, split_cache in mask_cache.items():
        with open(str(vdir / f'mask_cache_{split_key}.pickle'), 'wb') as _f:
            pickle.dump(split_cache, _f, protocol=4)

    print(f"    Saved  → {vdir}/")
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────





def build_mask_cache(
    smiles_list: List[str],
    groups_all: List[str],
    lookup_all: Dict[str, Dict[int, Tuple[str, int]]],
) -> Dict[str, Dict[int, Dict[str, "torch.BoolTensor"]]]:
    """Build a compact bool mask cache for motif-removal evaluation.

    For each known motif (motif_id >= 0) and each molecule that contains it,
    stores a BoolTensor of shape [n_atoms] where True = atom belongs to that motif.

    Structure:
        cache[split][motif_id][smiles] = BoolTensor [n_atoms]
        split in {'training', 'valid', 'test', 'all'}

    At evaluation time: x_masked = x * (~mask).float()
    This is ~n_features times smaller than storing full masked feature matrices.
    """
    import torch as _torch

    splits = {'training', 'valid', 'test'}
    cache: Dict[str, Dict[int, Dict[str, Any]]] = {
        'training': {}, 'valid': {}, 'test': {}, 'all': {}
    }

    for smi, grp in zip(smiles_list, groups_all):
        if grp not in splits:
            continue
        node_map = lookup_all.get(smi, {})
        if not node_map:
            continue
        n = max(node_map.keys()) + 1

        motif_atoms: Dict[int, List[int]] = {}
        for atom_idx, (_, mid) in node_map.items():
            if mid >= 0:
                motif_atoms.setdefault(mid, []).append(atom_idx)

        for mid, atom_idxs in motif_atoms.items():
            mask = _torch.zeros(n, dtype=_torch.bool)
            mask[_torch.tensor(atom_idxs, dtype=_torch.long)] = True
            for key in (grp, 'all'):
                cache[key].setdefault(mid, {})[smi] = mask

    return cache

def _load_csv(data_root: str, dataset: str, fold: int) -> pd.DataFrame:
    """Load {data_root}/{dataset}_{fold}.csv using DATASET_COLUMN for label."""
    path = Path(data_root) / f'{dataset}_{fold}.csv'
    if not path.exists():
        raise FileNotFoundError(
            f"Not found: {path}\n"
            f"Expected: {{data_root}}/{{dataset}}_{{fold}}.csv")
    df = pd.read_csv(path)
    label_col = DATASET_COLUMN.get(dataset)
    if label_col is None:
        raise KeyError(
            f"'{dataset}' not in DATASET_COLUMN. "
            f"Add it to the DATASET_COLUMN dict at the top of this file.")
    if label_col not in df.columns:
        raise KeyError(
            f"Label column '{label_col}' not found in {path}. "
            f"Columns: {df.columns.tolist()}")
    df = df.rename(columns={label_col: 'label'})
    if 'group' not in df.columns:
        df['group'] = 'training'
    return df

def compute_stats(dataset: str, variant: str, fold: int,
                  smiles_all: List[str], groups_all: List[str],
                  labels_all: np.ndarray,
                  mol_frags_tracked: List[List[Tuple[str, Set[int]]]],
                  motif_list: List[str],
                  motif_stats: Dict[str, dict],
                  frag_to_id: Dict[str, int],
                  threshold_motifs: Optional[Set[str]],
                  lookup_all: Dict) -> pd.DataFrame:
    """Compute per-motif and per-molecule statistics and return as a DataFrame.

    Columns:
      dataset, variant, fold, split
      — per-motif section (one row per motif):
        motif_id, smarts, n_atoms, ring,
        freq_count (n molecules containing it),
        freq_pct (% of split molecules),
        above_threshold (True/False),
        is_unknown (maps to -1 when threshold applied)
      — per-molecule section (appended as summary rows):
        n_unfragmented, n_unknown_nodes, mean_frags_per_mol,
        frag_count_distribution

    Returns two DataFrames: (motif_df, graph_df)
    """
    n_all = len(smiles_all)

    # ── Per-motif stats ────────────────────────────────────────────────────
    motif_rows = []
    for s in motif_list:
        st = motif_stats[s]
        motif_rows.append({
            'dataset':         dataset,
            'variant':         variant,
            'fold':            fold,
            'motif_id':        frag_to_id[s],
            'smarts':          s,
            'n_atoms':         st['n_atoms'],
            'ring':            st['ring'],
            'freq_count':      st['count'],
            'freq_pct':        round(st['count'] / n_all * 100, 3),
            'above_threshold': threshold_motifs is None or s in threshold_motifs,
        })
    motif_df = pd.DataFrame(motif_rows).sort_values('freq_count', ascending=False)

    # ── Per-graph stats ────────────────────────────────────────────────────
    graph_rows = []
    for smi, grp, lbl, mol_frags in zip(smiles_all, groups_all, labels_all, mol_frags_tracked):
        m = Chem.MolFromSmiles(smi)
        n_atoms = m.GetNumAtoms() if m else 0
        n_frags = len(mol_frags)
        node_map = lookup_all.get(smi, {})

        # Unfragmented = whole molecule is a single fragment (no cut fired)
        unfragmented = (n_frags == 1 and n_atoms > 0 and
                        frag.atom_count(mol_frags[0][0]) == n_atoms)

        # Unknown nodes = nodes mapped to motif_id = -1
        n_unknown = sum(1 for _, mid in node_map.values() if mid == -1)

        graph_rows.append({
            'dataset':      dataset,
            'variant':      variant,
            'fold':         fold,
            'split':        grp,
            'smiles':       smi,
            'label':        int(lbl),
            'n_atoms':      n_atoms,
            'n_frags':      n_frags,
            'unfragmented': unfragmented,
            'n_unknown_nodes': n_unknown,
            'pct_unknown':  round(n_unknown / max(n_atoms, 1) * 100, 1),
        })
    graph_df = pd.DataFrame(graph_rows)

    return motif_df, graph_df


def run_dataset(dataset: str, data_root: str, out_dir: Path,
                method: str, use_fallback: bool, use_bpe: bool,
                min_atoms: int, max_diam: int, sz_max: int, min_abs: int,
                fold: int = 0,
                apply_threshold: bool = False,
                threshold_pct: Optional[float] = None,
                variant_override: Optional[str] = None,
                variant_suffix: str = '',
                shatter: bool = True,
                rule_rank: str = 'balanced'):
    """Run the full pipeline for one dataset with given settings.

    Args:
        method            — 'rbrics' | 'brics' | 'all'
        use_fallback      — apply structural fallbacks to unfragmented molecules
        use_bpe           — apply prevalence-guided BPE merging
        apply_threshold   — if True, apply the chosen threshold:
                            motifs below the cutoff get motif_id = -1 in the
                            lookup and are excluded from rule candidates.
        threshold_pct     — override the CHOSEN_THRESHOLD value (0.1–0.9).
                            If None and apply_threshold=True, the value is
                            looked up from CHOSEN_THRESHOLD[method][fallback][bpe][dataset].
    """
    # variant_override lets the caller set a canonical output name (e.g.
    # "rbrics_old") independently of the internal method string
    # (e.g. "rbrics_only"). When not provided the name is auto-generated.
    if variant_override is not None:
        variant = variant_override + variant_suffix
    else:
        variant = f"{method}{'_fallback' if use_fallback else ''}{'_bpe' if use_bpe else ''}{'_filter' if apply_threshold else ''}{variant_suffix}"

    t0 = time.time()
    smdf = _load_csv(data_root, dataset, fold)

    smiles_all = smdf['smiles'].tolist()
    labels_all = smdf['label'].values.astype(int)
    groups_all = smdf['group'].tolist()
    n_all      = len(smdf)

    # Resolve threshold
    resolved_pct: Optional[float] = None
    if apply_threshold:
        if threshold_pct is not None:
            resolved_pct = threshold_pct
        else:
            # Look up from CHOSEN_THRESHOLD using the output variant name
            # (e.g. "all_fallback_bpe_filter").  Edit CHOSEN_THRESHOLD at the
            # top of this file to change thresholds without touching the shell.
            resolved_pct = get_chosen_threshold(variant, dataset)

    print(f"\n  method={method}  fallback={use_fallback}  bpe={use_bpe}"
          f"  threshold={'off' if not apply_threshold else f'{resolved_pct}%'}"
          f"  → {variant}")

    # ========================================================================
    # v4 FRAGMENTATION + MDL MERGE  (replaces legacy first-match cascade + BPE)
    # ------------------------------------------------------------------------
    # The legacy block below is DISABLED (commented out). v4 learns ONE global
    # MDL rulebook over the corpus, then tokenizes every molecule by a
    # deterministic per-tree rewrite (consistency by construction). `use_bpe`
    # now selects whether the MDL merge is applied (True) or the finest cascade
    # leaves are used (False). `method`/`use_fallback` are accepted for CLI
    # backward-compatibility; v4 always runs the full cascade with structural
    # fallback built in, so they no longer change the algorithm.
    # ========================================================================
    # ========================================================================
    # FRAGMENTATION ROUTING (restores rbrics / rbrics_old semantics)
    # ------------------------------------------------------------------------
    #   method == 'all'                  -> v4 cascade + MDL merge (tree-based,
    #                                       consistency-by-construction). use_bpe
    #                                       selects whether the MDL merge runs.
    #   method in {'rbrics','rbrics_old', -> LEGACY flat fragmenter (no tree, no
    #              'rbrics_only','brics'}   merge): a one-shot/two-shot bond cut.
    #                                         'rbrics'      = rBRICS + reBRICS  (two-shot)
    #                                         'rbrics_old'/ = rBRICS only        (one-shot)
    #                                         'rbrics_only'   (no reBRICS post-pass)
    #                                         'brics'       = BRICS only
    #                                       These feed straight into build_vocab and
    #                                       support --apply_threshold exactly as before.
    #                                       use_bpe controls the legacy prevalence BPE
    #                                       for these methods (NOT the v4 MDL merge).
    # The shatter floor (--shatter) only affects the v4 ('all') path.
    # ========================================================================
    frag._CACHE.clear()
    hier = frag.Hierarchy()
    bpe_history: List[dict] = []

    _LEGACY_METHODS = {'rbrics', 'rbrics_old', 'rbrics_only', 'brics'}
    _legacy_method = ('rbrics_only' if method in ('rbrics_old', 'rbrics_only')
                      else method)

    if method in _LEGACY_METHODS:
        # ---- LEGACY one-shot/two-shot fragmentation (no tree, no merge) -----
        # NOTE: the legacy prevalence-BPE merge is intentionally DISABLED. The
        # plain (untracked) fragmentation that fed `frag.bpe_merge`, and the BPE
        # block itself, are kept COMMENTED below for reference only — we do not
        # run the legacy engine with BPE for now. To re-enable: uncomment the
        # `import copy`, the `mol_frags_plain` computation, and the `if use_bpe`
        # block (and wire `use_bpe` back in at the call site).
        # import copy as _copy
        mol_frags_tracked: List[List[Tuple[str, Set[int]]]] = []
        # mol_frags_plain:   List[List[str]]                   = []
        for orig_smi in smiles_all:
            mol = Chem.MolFromSmiles(orig_smi)
            if mol is None:
                mol_frags_tracked.append([('[INVALID]', {0})])
                # mol_frags_plain.append(['[INVALID]'])
                continue
            mol_frags_tracked.append(
                fragment_molecule_tracked(mol, orig_smi, use_fallback, _legacy_method))
            # mol_frags_plain.append(
            #     frag.fragment_molecule(mol, hier,
            #                            use_fallback=use_fallback, method=_legacy_method))
        n_valid  = sum(1 for s in smiles_all if Chem.MolFromSmiles(s))
        n_single = sum(1 for f in mol_frags_tracked if len(f) == 1)
        print(f"    [legacy:{_legacy_method}] Fragmented: {n_valid-n_single}/{n_valid} "
              f"({100*(n_valid-n_single)/max(n_valid,1):.1f}%)  single-frag: {n_single}")

        # ---- legacy prevalence BPE merge (DISABLED — kept for reference) ----
        # if use_bpe:
        #     mf_copy = _copy.deepcopy(mol_frags_plain)
        #     mf_copy, bpe_history = frag.bpe_merge(
        #         mf_copy, hier, n_valid,
        #         min_atoms=min_atoms, max_diam=max_diam,
        #         sz_max=sz_max, min_abs=min_abs)
        #     if bpe_history:
        #         merge_map: Dict[str, str] = {}
        #         for h in bpe_history:
        #             for child in h['children']:
        #                 merge_map[child] = h['parent']
        #         def _resolve(s: str) -> str:
        #             seen: Set[str] = set()
        #             while s in merge_map and s not in seen:
        #                 seen.add(s); s = merge_map[s]
        #             return s
        #         mol_frags_tracked = [
        #             [(_resolve(smi), atoms) for smi, atoms in mf]
        #             for mf in mol_frags_tracked]
    else:
        # ---- v4 cascade + MDL merge (method == 'all') -----------------------
        import chemfrag_v4_adapter as _v4
        _ruleset, _index = _v4.learn_corpus_rulebook(smiles_all, use_merge=use_bpe,
                                                     shatter=shatter)
        mol_frags_tracked = []
        for orig_smi in smiles_all:
            if Chem.MolFromSmiles(orig_smi) is None:
                mol_frags_tracked.append([('[INVALID]', {0})])
                continue
            mol_frags_tracked.append(
                _v4.fragment_tracked_v4(orig_smi, _ruleset, _index, shatter=shatter))
        n_valid  = sum(1 for s in smiles_all if Chem.MolFromSmiles(s))
        n_single = sum(1 for frags in mol_frags_tracked if len(frags) == 1)
        print(f"    [v4] Fragmented: {n_valid - n_single}/{n_valid} "
              f"({100*(n_valid-n_single)/max(n_valid,1):.1f}%)  "
              f"single-frag: {n_single}  merge_rules={len(_ruleset)}")

    # ---- BEGIN LEGACY FRAGMENTATION + BPE (DISABLED) -----------------------
    # frag._CACHE.clear()
    # hier = frag.Hierarchy()
    # mol_frags_tracked: List[List[Tuple[str, Set[int]]]] = []
    # mol_frags_plain:   List[List[str]]                   = []
    #
    # for orig_smi in smiles_all:
    #     mol = Chem.MolFromSmiles(orig_smi)
    #     if mol is None:
    #         mol_frags_tracked.append([('[INVALID]', {0})])
    #         mol_frags_plain.append(['[INVALID]'])
    #         continue
    #     mol_frags_tracked.append(
    #         fragment_molecule_tracked(mol, orig_smi, use_fallback, method))
    #     mol_frags_plain.append(
    #         frag.fragment_molecule(mol, hier,
    #                                use_fallback=use_fallback, method=method))
    #
    # n_valid  = sum(1 for s in smiles_all if Chem.MolFromSmiles(s))
    # n_single = sum(1 for frags in mol_frags_tracked if len(frags) == 1)
    # print(f"    Fragmented: {n_valid - n_single}/{n_valid} "
    #       f"({100*(n_valid-n_single)/max(n_valid,1):.1f}%)  "
    #       f"single-frag: {n_single}")
    #
    # # BPE
    # bpe_history: List[dict] = []
    # if use_bpe:
    #     mf_copy = copy.deepcopy(mol_frags_plain)
    #     mf_copy, bpe_history = frag.bpe_merge(
    #         mf_copy, hier, n_valid,
    #         min_atoms=min_atoms, max_diam=max_diam,
    #         sz_max=sz_max, min_abs=min_abs)
    #     if bpe_history:
    #         merge_map: Dict[str, str] = {}
    #         for h in bpe_history:
    #             for child in h['children']:
    #                 merge_map[child] = h['parent']
    #
    #         def resolve(s: str) -> str:
    #             seen: Set[str] = set()
    #             while s in merge_map and s not in seen:
    #                 seen.add(s); s = merge_map[s]
    #             return s
    #
    #         mol_frags_tracked = [
    #             [(resolve(smi), atoms) for smi, atoms in mf]
    #             for mf in mol_frags_tracked]
    # ---- END LEGACY FRAGMENTATION + BPE (DISABLED) -------------------------

    # Vocabulary
    motif_list, frag_to_id, motif_stats = build_vocab(
        mol_frags_tracked, labels_all, groups=groups_all, min_sup_for_rules=MIN_SUP)
    # motif_list, frag_to_id, motif_stats = build_vocab(
    #     mol_frags_tracked, labels_all, min_sup_for_rules=MIN_SUP)

    above1 = sum(1 for s in motif_list if motif_stats[s]['above_min_sup'])
    print(f"    Vocab: {len(motif_list)} motifs  ≥1%:{above1}  "
          f"frags/mol: μ={np.mean([len(f) for f in mol_frags_tracked]):.2f}")
    for s in motif_list[:5]:
        st = motif_stats[s]
        print(f"      {st['count']/n_all*100:5.1f}%  {st['n_atoms']:2d}a  "
              f"{'R' if st['ring'] else ' '}  {s}")

    # Threshold motifs set — used to remap below-threshold nodes to -1
    threshold_motifs: Optional[Set[str]] = None
    if apply_threshold and resolved_pct is not None:
        N_tv        = sum(1 for g in groups_all if g in ('training', 'valid'))
        global_cut  = int((resolved_pct / 100) * N_tv)
        n0_tv = sum(1 for g,l in zip(groups_all,labels_all)
                    if g in ('training','valid') and int(l)==0)
        n1_tv = N_tv - n0_tv
        r0, r1 = n0_tv/max(N_tv,1), n1_tv/max(N_tv,1)
        minority   = (1 if r0 >= 0.6 else (0 if r1 >= 0.6 else None))
        minority_n = (n1_tv if minority==1 else (n0_tv if minority==0 else None))

        # mol_counts per trainval molecule (weighted by 1/length, notebook style)
        from collections import Counter as _Counter
        mol_counts  = _Counter()
        wt_counts_0 = _Counter()
        wt_counts_1 = _Counter()
        for smi, lbl, grp, mf in zip(smiles_all, labels_all, groups_all, mol_frags_tracked):
            if grp not in ('training','valid'): continue
            for smarts, _ in mf:
                length = max(motif_stats.get(smarts,{}).get('n_atoms',1), 1)
                mol_counts[smarts] += 1.0 / length
            if int(lbl) == 0:
                for smarts, _ in mf:
                    length = max(motif_stats.get(smarts,{}).get('n_atoms',1), 1)
                    wt_counts_0[smarts] += 1.0 / length
            else:
                for smarts, _ in mf:
                    length = max(motif_stats.get(smarts,{}).get('n_atoms',1), 1)
                    wt_counts_1[smarts] += 1.0 / length

        threshold_motifs = {m for m,c in mol_counts.items() if c >= global_cut}
        if minority is not None and minority_n is not None:
            mb_cut = int((resolved_pct / 100) * minority_n)
            wt = wt_counts_1 if minority == 1 else wt_counts_0
            for m, cnt in wt.items():
                if cnt >= mb_cut:
                    threshold_motifs.add(m)

        n_below = len(motif_list) - len(threshold_motifs & set(motif_list))
        print(f"    Threshold {resolved_pct}%: cutoff={global_cut}  "
              f"kept={len(threshold_motifs & set(motif_list))}  "
              f"below (→ -1)={n_below}")

    # Lookup + matrix
    lookup_all = build_lookup(smiles_all, mol_frags_tracked, frag_to_id,
                               threshold_motifs)
    X          = build_matrix(mol_frags_tracked, frag_to_id, n_all)

    # Coverage stats
    n_tr  = sum(1 for g in groups_all if g == 'training')
    n_cov = sum(1 for smi, g in zip(smiles_all, groups_all)
                if g == 'training' and smi in lookup_all
                and Chem.MolFromSmiles(smi) is not None
                and all(v[1] != -1 for v in lookup_all[smi].values()))
    n_cov_any = sum(1 for smi, g in zip(smiles_all, groups_all)
                    if g == 'training' and smi in lookup_all
                    and Chem.MolFromSmiles(smi) is not None
                    and len(lookup_all[smi]) ==
                        Chem.MolFromSmiles(smi).GetNumAtoms())
    print(f"    Node coverage: total={n_cov_any}/{n_tr}  "
          f"fully-known={n_cov}/{n_tr} "
          f"({100*n_cov/max(n_tr,1):.1f}%)")

    # GMI dicts — exclude -1 from motifidx sets
    lookup_train = {s: lookup_all[s] for s, g in zip(smiles_all, groups_all)
                    if g == 'training' and s in lookup_all}
    lookup_valid = {s: lookup_all[s] for s, g in zip(smiles_all, groups_all)
                    if g == 'valid'    and s in lookup_all}
    lookup_test  = {s: lookup_all[s] for s, g in zip(smiles_all, groups_all)
                    if g == 'test'     and s in lookup_all}
    gmi_train    = {s: {mid for _, mid in lookup_all[s].values() if mid != -1}
                    for s, g in zip(smiles_all, groups_all)
                    if g == 'training' and s in lookup_all}
    gmi_test     = {s: {mid for _, mid in lookup_all[s].values() if mid != -1}
                    for s, g in zip(smiles_all, groups_all)
                    if g == 'test' and s in lookup_all}

    # Rules
    rules = extract_rules(motif_list, motif_stats, X, labels_all,
                          rank_mode=rule_rank)
    if rules:
        r0 = rules[0]
        print(f"    Best rule [{rule_rank}] "
              f"cover={r0.get('rule_pct_match', '?')}%  "
              f"balance={r0.get('balance', '?')}  "
              f"sep={r0.get('separation', '?')}  "
              f"spurious={r0.get('spurious', '?')}  "
              f"score={r0.get('score', '?')}:")
        print(f"      {' ∨ '.join('('+' ∧ '.join(c['motifs'])+')'for c in r0['clauses'])}")
    print(f"    Rules: {len(rules)}")
    if bpe_history:
        print(f"    BPE: {len(bpe_history)} merges")
        for h in bpe_history[:3]:
            print(f"      {sorted(h['children'])} → {h['parent']}  "
                  f"(supp={h['parent_supp']}, {h['parent_atoms']}a)")

    # Per-motif test occurrence count — computed here where mol_frags_tracked is in scope
    from collections import Counter as _Counter2
    _test_occ: dict = _Counter2()
    for smi, grp, mf in zip(smiles_all, groups_all, mol_frags_tracked):
        if grp != 'test':
            continue
        for smarts, _ in mf:
            if smarts in frag_to_id:
                _test_occ[smarts] += 1.0

    meta = save_outputs(out_dir, dataset, variant, smdf,
                        lookup_all=lookup_all, smiles_all=smiles_all, groups_all=groups_all,
                        motif_list=motif_list, motif_stats=motif_stats, frag_to_id=frag_to_id,
                        X=X, lookup_train=lookup_train, lookup_valid=lookup_valid,
                        lookup_test=lookup_test,
                        gmi_train=gmi_train, gmi_test=gmi_test,
                        rules=rules, labels=labels_all,
                        test_occ_ctr=_test_occ)

    # Statistics CSV
    motif_df, graph_df = compute_stats(
        dataset, variant, fold,
        smiles_all, groups_all, labels_all,
        mol_frags_tracked, motif_list, motif_stats, frag_to_id,
        threshold_motifs, lookup_all)
    vdir = out_dir / dataset / variant
    motif_df.to_csv(vdir / 'stats_motifs.csv',  index=False)
    graph_df.to_csv(vdir / 'stats_graphs.csv',  index=False)
    _write_stats_summary(motif_df, graph_df, vdir)
    print(f"    Stats → {vdir}/stats_motifs.csv  stats_graphs.csv")

    meta['elapsed'] = round(time.time() - t0, 1)
    print(f"    ({meta['elapsed']}s)")
    return meta


def _write_stats_summary(motif_df: pd.DataFrame,
                         graph_df: pd.DataFrame,
                         vdir: Path):
    """Write a human-readable summary of the statistics to stats_summary.txt."""
    lines = []
    add = lines.append

    add("=== MOTIF STATISTICS ===")
    add(f"Total motifs in vocabulary:  {len(motif_df)}")
    add(f"  Above threshold:           {int(motif_df['above_threshold'].sum())}")
    add(f"  Below threshold (→ -1):    {int((~motif_df['above_threshold']).sum())}")

    add("\nMotif size distribution (n_atoms):")
    for (lo, hi, label) in [(1,2,'1-2a'),(3,5,'3-5a'),(6,9,'6-9a'),
                             (10,15,'10-15a'),(16,99,'16+a')]:
        n = int(((motif_df['n_atoms']>=lo)&(motif_df['n_atoms']<=hi)).sum())
        add(f"  {label:>8}: {n:>5}")

    add("\nTop-20 motifs by frequency:")
    add(f"  {'motif_id':>9}  {'freq_count':>11}  {'freq_pct':>9}  "
        f"{'n_atoms':>8}  {'ring':>5}  smarts")
    for _, r in motif_df.head(20).iterrows():
        add(f"  {int(r['motif_id']):>9}  {int(r['freq_count']):>11}  "
            f"{r['freq_pct']:>8.2f}%  {int(r['n_atoms']):>8}  "
            f"{'Y' if r['ring'] else 'N':>5}  {r['smarts']}")

    add("\n=== GRAPH STATISTICS ===")
    for split in ['training','valid','test','all']:
        gd = graph_df if split == 'all' else graph_df[graph_df['split']==split]
        if len(gd) == 0: continue
        add(f"\n[{split}]  n={len(gd)}")
        add(f"  Unfragmented graphs:    {int(gd['unfragmented'].sum())} "
            f"({100*gd['unfragmented'].mean():.1f}%)")
        if 'n_unknown_nodes' in gd.columns:
            g_with_unk = int((gd['n_unknown_nodes'] > 0).sum())
            tot_unk    = int(gd['n_unknown_nodes'].sum())
            add(f"  Graphs with unknown≥1:  {g_with_unk} ({100*g_with_unk/max(len(gd),1):.1f}%)")
            add(f"  Total unknown nodes:    {tot_unk} ({gd['pct_unknown'].mean():.1f}% of atoms avg)")
        add(f"  Frags/graph: mean={gd['n_frags'].mean():.2f}  "
            f"median={gd['n_frags'].median():.0f}  "
            f"min={gd['n_frags'].min()}  max={gd['n_frags'].max()}")
        dist = gd['n_frags'].value_counts().sort_index()
        add("  Frag count distribution:")
        for k, v in dist.items():
            add(f"    {k:>4} frags: {v:>5} graphs")

    (vdir / 'stats_summary.txt').write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(
        description='Generate MotifSAT-compatible vocabulary and rules',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Fragmentation algorithm (--method):
  rbrics  rBRICS only at every level of the cascade
  brics   BRICS  only at every level of the cascade
  all     rBRICS → BRICS → RECAP → Murcko (default, recommended)

--fallback  Also try ring/chain and acyclic-bond cuts on molecules
            that remain as a single fragment after the chemistry cascade.

--bpe       After fragmentation, merge tiny fragments back into their
            parent using prevalence-guided BPE. Reduces vocab size and
            produces more meaningful merged tokens.

Output directory structure:
  {out_dir}/{dataset}/{method}[_fallback][_bpe]/
    {dataset}_{variant}_graph_lookup.pickle
    {dataset}_{variant}_motif_list.pickle
    ... (all MotifSAT pickle files)
    matrix.npz  matrix_columns.csv  rules.json  meta.json

Examples:
  # Full cascade, fallback on, BPE on  (recommended)
  python generate_vocab_rules.py --datasets Mutagenicity BBBP \\
      --data_root /data --out_dir ./out --method all --fallback --bpe

  # rBRICS only, no fallback, no BPE
  python generate_vocab_rules.py --datasets Mutagenicity \\
      --data_root /data --out_dir ./out --method rbrics

  # BRICS only, fallback on
  python generate_vocab_rules.py --datasets Mutagenicity \\
      --data_root /data --out_dir ./out --method brics --fallback
""")
    p.add_argument('--datasets',  nargs='+', required=True,
                   help='Dataset names')
    p.add_argument('--data_root', required=True,
                   help='Directory containing {dataset}_{fold}.csv files '
                        'or {dataset}_fold{fold}/ subdirectories')
    p.add_argument('--fold',      type=int, default=0,
                   help='Fold number (default 0). File: {dataset}_{fold}.csv')
    p.add_argument('--out_dir',   default='./motifsat_output',
                   help='Output root directory')
    p.add_argument('--method',    default='all',
                   choices=['rbrics', 'brics', 'all', 'rbrics_only', 'rbrics_old'],
                   help='Fragmentation algorithm(s) to use (default: all)')
    p.add_argument('--fallback',  action='store_true',
                   help='Apply structural fallbacks to unfragmented molecules')
    p.add_argument('--bpe',       action='store_true',
                   help='Apply BPE merging to reduce tiny fragments')
    p.add_argument('--shatter',   action='store_true',
                   help='Mild-shatter floor: drop the terminal-atom guard in the '
                        'structural stage so every acyclic single bond is cut '
                        '(rings + double bonds still protected), giving the MDL '
                        'merge a finer floor. Measured ~20%% smaller vocab and '
                        'equal-or-better env-consistency. Variant name gets a '
                        '"_shatter" suffix so it does not collide with standard v4.')
    p.add_argument('--min_atoms', type=int, default=MIN_FRAG_ATOMS,
                   help=f'Min atoms for BPE fragment to stand alone (default {MIN_FRAG_ATOMS})')
    p.add_argument('--max_diam',  type=int, default=GNN_LAYERS,
                   help=f'GNN layers = max BPE fragment diameter/2 (default {GNN_LAYERS})')
    p.add_argument('--sz_max',    type=int, default=SZ_MAX,
                   help=f'Max atoms after BPE merge (default {SZ_MAX})')
    p.add_argument('--min_abs',   type=int, default=BPE_MIN_ABS,
                   help=f'Min absolute support for BPE merge (default {BPE_MIN_ABS})')
    p.add_argument('--apply_threshold', action='store_true',
                   help='Apply the chosen threshold: motifs below it get motif_id=-1 '
                        'in the lookup and are excluded from rule candidates. '
                        'Threshold value comes from CHOSEN_THRESHOLD dict or --threshold_pct.')
    p.add_argument('--variant',       default=None,
                   help='Override the auto-generated output variant name '
                        '(e.g. "rbrics_old" for method=rbrics_only). '
                        'Controls the subdirectory under out_dir/{dataset}/.')
    p.add_argument('--threshold_pct', type=float, default=None,
                   help='Override CHOSEN_THRESHOLD with this value (0.1–0.9). '
                        'Only used when --apply_threshold is set.')
    p.add_argument('--rule_rank', default='balanced',
                   choices=['balanced', 'pct1'],
                   help="How to sort rules.json (rule_index 0 = best). "
                        "'balanced' (default): balance × separation × (1-spurious) "
                        "— targets a 50/50 synthetic split and penalises spurious/"
                        "subsuming motifs. 'pct1': legacy positive-coverage sort. "
                        "All score components are written to rules_summary.csv "
                        "either way so you can inspect and override RULE_INDEX.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mild-shatter floor is threaded explicitly into run_dataset → the v4 adapter
    # (no module-global mutation), so concurrent/programmatic callers can't leak
    # state into each other. The output variant name gets a distinct suffix so
    # shatter vocabs never overwrite standard-v4 vocabs.
    _shatter_suffix = '_shatter' if args.shatter else ''

    all_metas = []
    for ds in args.datasets:
        print(f"\n{'='*60}\n  {ds}\n{'='*60}")
        meta = run_dataset(ds, args.data_root, out_dir,
                           method=args.method,
                           use_fallback=args.fallback,
                           use_bpe=args.bpe,
                           min_atoms=args.min_atoms,
                           max_diam=args.max_diam,
                           sz_max=args.sz_max,
                           min_abs=args.min_abs,
                           fold=args.fold,
                           apply_threshold=args.apply_threshold,
                           threshold_pct=args.threshold_pct,
                           variant_override=args.variant,
                           variant_suffix=_shatter_suffix,
                           shatter=bool(args.shatter),
                           rule_rank=args.rule_rank)
        meta['dataset'] = ds
        all_metas.append(meta)

    df = pd.DataFrame([{
        'dataset':      m['dataset'],
        'variant':      m['variant'],
        'n_motifs':     m['n_vocab_motifs'],
        'n_rules':      m['n_rules'],
        'best_pct1':    m['best_rule_pct1'],
        'best_rule':    m['best_rule'][:80],
        'elapsed_s':    m.get('elapsed', 0),
    } for m in all_metas])
    df.to_csv(out_dir / 'summary.csv', index=False)
    print(f"\n{'='*60}\nSummary → {out_dir}/summary.csv")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
