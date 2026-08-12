"""ertl_conservative_frag.py — conservative Ertl functional-group partition for the
``conservative_ertl_ring_mdl`` fragmentation method.

Pipeline (heads for cascade_bpe_linker): run canonical Ertl ``get_fgs`` on the WHOLE molecule,
subtract ring atoms, then keep ONLY clean, well-formed functional groups as frozen heads and
release everything ambiguous / over-merged / under-merged to the leftover (linker) tier where the
MDL-BPE merge handles it. A region is KEPT iff:

  * complete      — the Ertl set has no ring atom AND no atom double/triple-bonds to a ring atom
                    (drops ring-carbonyl/imine remnants — a bare ``=O`` off a ring carbon is a
                    fragment of a ring carbonyl, not a standalone FG; single bonds to a ring, e.g.
                    a phenol ``-OH``, are fine);
  * single-center — <= 1 hetero-unsaturation island (drops fused composites: sulfonylurea = 2,
                    biuret = 3; keeps recognized single-center composites carbamate/urea/guanidine);
  * bounded       — <= K_MAX_HEAVY (6) heavy atoms (drops conjugated mega-chains / polyphosphates);
  * standard      — no radical / odd valence.

Returns the (owner, ids) contract expected by cascade_bpe_linker._MolGraph / generate_vocab_rules:
ring systems -> ``ring:<canonical smiles>`` (frozen, whole systems), kept FGs -> ``fg:ertl``
(frozen heads; the cascade re-keys them structurally), everything else -> ``chain:``/``frag:``
leftover (flows into the merge).

NO FALLBACK: requires the canonical EFGs implementation (RDKit Contrib/efgs, Lu & Ertl, JCIM 2024).
Import fails loudly if it is unavailable — never substitutes an approximation.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import rdkit
from rdkit import Chem

K_MAX_HEAVY = 6

# ── canonical EFGs — REQUIRED, no fallback ───────────────────────────────────
_EFGS_DIR = os.path.join(os.path.dirname(rdkit.__file__), 'Contrib', 'efgs')
try:
    if os.path.isdir(_EFGS_DIR) and _EFGS_DIR not in sys.path:
        sys.path.insert(0, _EFGS_DIR)
    from efgs import get_fgs as _get_fgs           # RDKit Contrib EFGs (Lu & Ertl 2024)
except Exception as _e:                            # pragma: no cover
    raise ImportError(
        "ertl_conservative_frag requires the canonical EFGs implementation (RDKit Contrib/efgs, "
        "Lu & Ertl, J. Chem. Inf. Model. 2024) and will NOT fall back to an approximation. It is "
        f"not importable ({_e!r}). Install an RDKit build that ships Contrib/efgs "
        "(expected at <rdkit>/Contrib/efgs)."
    ) from _e


def _components(mol: Chem.Mol, atoms) -> List[Set[int]]:
    """Connected components of ``atoms`` under the molecular bond graph."""
    atoms = set(atoms)
    adj: Dict[int, Set[int]] = defaultdict(set)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in atoms and j in atoms:
            adj[i].add(j); adj[j].add(i)
    seen, out = set(), []
    for a in atoms:
        if a in seen:
            continue
        stack, comp = [a], set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); comp.add(x); stack.extend(adj[x] - seen)
        out.append(comp)
    return out


def _dbonds_to_ring(mol: Chem.Mol, atoms: Set[int], ring: Set[int]) -> bool:
    """True if any atom double/triple-bonds to a ring atom outside the set (ring-carbonyl remnant)."""
    for i in atoms:
        ai = mol.GetAtomWithIdx(i)
        for b in ai.GetBonds():
            n = b.GetOtherAtom(ai)
            if (n.GetIdx() in ring and n.GetIdx() not in atoms
                    and b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)):
                return True
    return False


def _hetero_unsat_islands(mol: Chem.Mol, atoms: Set[int]) -> int:
    """# connected islands of nonaromatic multiple-bonded atoms that contain a heteroatom."""
    atoms = set(atoms)
    adj: Dict[int, Set[int]] = defaultdict(set)
    unsat: Set[int] = set()
    for b in mol.GetBonds():
        if b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE) and not b.GetIsAromatic():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in atoms and j in atoms:
                unsat.add(i); unsat.add(j); adj[i].add(j); adj[j].add(i)
    seen, n = set(), 0
    for a in unsat:
        if a in seen:
            continue
        comp, stack = set(), [a]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); comp.add(x); stack.extend(adj[x] - seen)
        if any(mol.GetAtomWithIdx(k).GetAtomicNum() not in (1, 6) for k in comp):
            n += 1
    return n


def _n_heavy(mol: Chem.Mol, atoms) -> int:
    return sum(1 for i in atoms if mol.GetAtomWithIdx(i).GetAtomicNum() > 1)


def _kept(mol: Chem.Mol, s: Set[int], ring: Set[int]) -> bool:
    if (s & ring) or _dbonds_to_ring(mol, s, ring):          # complete
        return False
    if _hetero_unsat_islands(mol, s) > 1:                     # single-center
        return False
    if _n_heavy(mol, s) > K_MAX_HEAVY:                        # bounded
        return False
    if any(mol.GetAtomWithIdx(i).GetNumRadicalElectrons() > 0 for i in s):   # standard
        return False
    return True


def _ring_key(mol: Chem.Mol, comp) -> str:
    """'ring:<canonical, stereo-stripped SMILES>' for a (whole) ring system."""
    km = Chem.Mol(mol)
    Chem.RemoveStereochemistry(km)
    for b in km.GetBonds():
        b.SetBondDir(Chem.BondDir.NONE)
    return 'ring:' + Chem.MolFragmentToSmiles(km, atomsToUse=sorted(comp), canonical=True)


def partition(mol: Chem.Mol, subcut_chains: bool = False,
              whole_ring_systems: bool = True, **kwargs) -> Tuple[List[int], Dict[int, str]]:
    """(owner, ids) partition. ``subcut_chains`` is accepted for signature parity (the linker tier
    is (re)segmented by cascade_bpe_linker at the finest cut, so it is a no-op here). Ring systems
    are always whole (a broken ring remnant would not be a closed ``ring:`` cycle)."""
    n = mol.GetNumAtoms()
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    owner: List[int] = [-1] * n
    ids: Dict[int, str] = {}
    fid = 0
    for comp in _components(mol, ring):                       # ring systems (frozen)
        ids[fid] = _ring_key(mol, comp)
        for a in comp:
            owner[a] = fid
        fid += 1
    for s in _get_fgs(mol):                                   # conservative clean FGs (frozen heads)
        s = set(s)
        if not (s - ring):
            continue
        if _kept(mol, s, ring):
            ids[fid] = 'fg:ertl'
            for a in s:
                owner[a] = fid
            fid += 1
    leftover = [a for a in range(n) if owner[a] < 0]          # rest -> leftover (merges)
    for comp in _components(mol, leftover):
        pure_c = all(mol.GetAtomWithIdx(a).GetAtomicNum() == 6 for a in comp)
        ids[fid] = 'chain:' if pure_c else 'frag:'            # cascade re-keys leftover structurally
        for a in comp:
            owner[a] = fid
        fid += 1
    if any(o < 0 for o in owner):
        raise ValueError(
            f"ertl_conservative partition left atom(s) unassigned for "
            f"{Chem.MolToSmiles(mol)!r} — this is a bug, not a fallback case.")
    return owner, ids
