"""ertl_frag.py — Ertl functional-group detection as a drop-in ``partition`` for the
fg_first pipeline (``--method ertl_first`` / ``ertl_first_mdl``).

FG detection uses the **canonical EFGs implementation** shipped in RDKit's Contrib
directory (``Contrib/efgs/efgs.py``, ``get_fgs``) — the "complete and accurate"
implementation of Ertl's algorithm from Lu & Ertl, *EFGs: A Complete and Accurate
Implementation of Ertl's Functional Group Detection Algorithm in RDKit*, J. Chem. Inf.
Model. (2024). If that module is unavailable, we fall back to a faithful reimplementation
of Ertl's reference ``ifg.py`` (Ertl, J. Cheminformatics 9:36, 2017) so the pipeline still
runs — but for publication cite EFGs and rely on the canonical path (``FG_SOURCE`` records
which was used).

Same contract as ``fg_first_frag.partition``: rings are claimed first (reusing fg_first's
ring machinery), then Ertl FGs on the free atoms, then leftover linkers. FGs are keyed by
``chemfrag.frag_key`` (structural — never a ``fg:`` literal); a degenerate FG with no
structural key is left to the leftover tier. Wired into cascade_bpe_linker via its
``fg_partition`` hook, so the MDL merge / keying are identical to fg_first.
"""
import os
import sys
from typing import Dict, List, Set

import rdkit
from rdkit import Chem

import fg_first_frag as _fgf
import chemfrag as _cf

# ── canonical EFGs (RDKit Contrib) with a faithful-reimplementation fallback ──────
_get_fgs_canonical = None
try:
    _efgs_dir = os.path.join(os.path.dirname(rdkit.__file__), 'Contrib', 'efgs')
    if os.path.isdir(_efgs_dir) and _efgs_dir not in sys.path:
        sys.path.insert(0, _efgs_dir)
    from efgs import get_fgs as _get_fgs_canonical   # RDKit Contrib EFGs (Lu & Ertl 2024)
    FG_SOURCE = 'efgs'                                # canonical implementation in use
except Exception:
    FG_SOURCE = 'reimpl'                             # fell back to the reimplementation below

# reference ifg.py marking rules (fallback only)
_PATTS = [Chem.MolFromSmarts(s) for s in
          ['[!#6;!#1]=,#[*]', 'C=,#C', '[CX4](-[O,N,S])-[O,N,S]', '[O,N,S]1CC1']]


def _merge(mol, marked, aset):
    grown = set()
    for idx in aset:
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            j = nbr.GetIdx()
            if j in marked:
                marked.remove(j)
                grown.add(j)
    if grown:
        _merge(mol, marked, grown)
        aset.update(grown)


def _reimpl_fgs(mol) -> List[Set[int]]:
    """Faithful reimplementation of Ertl's ifg.py (fallback when Contrib/efgs is absent)."""
    marked = set(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
    for patt in _PATTS:
        for m in mol.GetSubstructMatches(patt):
            marked.update(m)
    groups = []
    while marked:
        core = {marked.pop()}
        _merge(mol, marked, core)
        groups.append(core)
    return groups


def ertl_fgs(mol) -> List[Set[int]]:
    """Ertl functional-group cores (canonical EFGs if available, else reimplementation)."""
    if _get_fgs_canonical is not None:
        return [set(g) for g in _get_fgs_canonical(mol)]
    return _reimpl_fgs(mol)


def _fg_key(mol, atoms):
    """Structural key for an FG atom set (frag_key, else canonical fragment SMILES), or None.
    Never a ``fg:``/``chain:``/``frag:`` literal, so it is treated as a head and can't leak."""
    k = _cf.frag_key(mol, set(atoms))
    if k:
        return k
    try:
        s = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms), canonical=True)
        return s or None
    except Exception:
        return None


def partition(mol, split_fused_aliphatic=True, subcut_chains=True,
              split_fused_aromatic=False, break_nonarom_min=None,
              whole_ring_systems=False):
    """(owner, ident) partition with Ertl/EFGs FG detection swapped in for the curated
    dictionary. Signature mirrors fg_first_frag.partition so it is a drop-in via the
    cascade_bpe_linker ``fg_partition`` hook."""
    n = mol.GetNumAtoms()
    owner = [-1] * n
    ident: Dict[int, str] = {}

    def claim(atoms, idt):
        free = [a for a in atoms if owner[a] == -1]
        if not free:
            return
        fid = len(ident)
        for a in free:
            owner[a] = fid
        ident[fid] = idt

    # rings first (kept whole, keyed ring:) so ring heteroatoms are not stolen by FG marking
    for rs in _fgf._ring_systems(mol, split_fused_aliphatic, split_fused_aromatic,
                                 break_nonarom_min, whole_ring_systems):
        free_rs = {a for a in rs if owner[a] == -1}
        if free_rs and _fgf._is_ring_set(mol, free_rs):
            claim(free_rs, _fgf._ring_identity(mol, free_rs))

    # Ertl FGs — REQUIRE_WHOLE: an FG is a match ONLY if its ENTIRE atom set is still free. If any
    # atom was already claimed (a ring took it), the FG is NOT present here — skip it and let its
    # atoms fall to the leftover tier (keyed by their real structure). This matches fg_first_frag's
    # claim(require_whole=True) exactly, so all three detectors agree, and it can never emit a
    # partial or disconnected FG (e.g. in CN1C(=O)CCS(=O)(=O)C1... the ring-embedded C=O and the
    # sulfonyl are skipped, their bare =O fall to leftover — never a bogus 'O.O.O' motif).
    for group in ertl_fgs(mol):
        atoms = [a for a in group if 0 <= a < n]
        if not atoms or any(owner[a] != -1 for a in atoms):   # any atom claimed -> not a match
            continue
        key = _fg_key(mol, set(atoms))
        if key is not None:
            claim(atoms, key)

    # everything left -> connected components (chains / linkers)
    rem = [i for i in range(n) if owner[i] == -1]
    for comp in _fgf._components(mol, rem, set()):
        claim(comp, _fgf._leftover_identity(mol, comp))

    return owner, ident
