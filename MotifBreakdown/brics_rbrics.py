"""brics_rbrics.py — shared BRICS / rBRICS / reBRICS bond discovery.

Single source of truth for the BRICS and rBRICS cut primitives, reused by BOTH
fragmentation engines so the same chemistry is identified identically everywhere:
  * the v4 cascade          (chemfrag.py: part_brics/part_rbrics, _*_within)
  * the legacy single-pass  (generate_vocab_rules.py: fragment_molecule_tracked)
  * the legacy plain path    (molfragbpe5.py: cut_brics / cut_rbrics_only)

Each ``*_bonds()`` function takes an RDKit Mol (or a submol) and returns the list
of ``(begin_atom_idx, end_atom_idx)`` atom pairs the algorithm would cut, in that
mol's own atom indexing. Callers convert pairs to bond indices in their own
coordinate system via :func:`nonring_bond_indices` and decide how to break them.

rBRICS is an optional vendored dependency (``rBRICS_public``). If it is not
importable, :data:`RBRICS_OK` is False and ``rbrics_bonds`` / ``rebrics_bonds``
return ``[]`` so callers degrade gracefully to BRICS.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Set, Tuple

from rdkit import Chem
from rdkit.Chem import BRICS

try:
    from rBRICS_public import FindrBRICSBonds, FindreBRICSBonds
    RBRICS_OK = True
except Exception:
    RBRICS_OK = False


def brics_bonds(mol: Chem.Mol) -> List[Tuple[int, int]]:
    """Atom pairs that standard BRICS would cut (acyclic by construction)."""
    try:
        return [(a, b) for (a, b), _ in BRICS.FindBRICSBonds(mol)]
    except Exception:
        return []


def rbrics_bonds(mol: Chem.Mol) -> List[Tuple[int, int]]:
    """Atom pairs that rBRICS (FindrBRICSBonds) would cut. [] if rBRICS absent."""
    if not RBRICS_OK:
        return []
    try:
        return [(a, b) for (a, b), _ in FindrBRICSBonds(mol)]
    except Exception:
        return []


def rebrics_bonds(mol: Chem.Mol) -> List[Tuple[int, int]]:
    """Atom pairs reBRICS would cut (long aliphatic CCCCCC chains only).
    [] if rBRICS absent."""
    if not RBRICS_OK:
        return []
    try:
        return [(a, b) for (a, b), _ in FindreBRICSBonds(mol)]
    except Exception:
        return []


def nonring_bond_indices(mol: Chem.Mol,
                         pairs: Iterable[Tuple[int, int]],
                         within: Optional[Set[int]] = None) -> Set[int]:
    """Convert ``(a, b)`` atom pairs to non-ring bond indices in ``mol``.

    Parameters
    ----------
    pairs : iterable of (int, int)
        Atom-index pairs (e.g. from :func:`brics_bonds`).
    within : set[int] or None
        If given, only pairs whose BOTH endpoints lie in the set are kept —
        used by the v4 within-atomset cascade stages.
    """
    idx: Set[int] = set()
    for a, b in pairs:
        if within is not None and (a not in within or b not in within):
            continue
        bd = mol.GetBondBetweenAtoms(a, b)
        if bd is not None and not bd.IsInRing():
            idx.add(bd.GetIdx())
    return idx
