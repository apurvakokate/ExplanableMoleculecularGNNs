"""rdkit_fg_frag.py — curated functional-group detection sourced from RDKit's *published*
``FunctionalGroups.txt`` (``--method rdkit_fg_first`` / ``rdkit_fg_first_mdl``).

This is the "defensible curated" fragmentation: instead of a bespoke hand-picked SMARTS
list, the functional groups are exactly RDKit's shipped ``$RDDataDir/FunctionalGroups.txt``
(~38 groups). Cite RDKit rather than defending a curation. Everything else is identical to
fg_first — rings first (fg_first_frag machinery), FGs on the free atoms, then linkers; keys
are ``chemfrag.frag_key`` (structural, never a ``fg:`` literal). Wired into cascade via the
``fg_partition`` hook so the MDL merge / keying match fg_first exactly.

The RDKit FG SMARTS carry an explicit ``*`` attachment atom; we exclude those wildcard
positions from the claimed set so an FG owns only its own atoms (partition-clean). Patterns
are applied most-specific-first (by heavy-atom count) so nested matches resolve
deterministically, mirroring fg_first's specific->general ordering.
"""
import os
from typing import Dict, List, Set

from rdkit import Chem, RDConfig

import fg_first_frag as _fgf
import chemfrag as _cf


def _load_rdkit_fgs():
    """[(name, patt_mol, wildcard_positions)] from RDKit's FunctionalGroups.txt, specific-first."""
    path = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
    out = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('//') or s.startswith('#'):
                continue
            parts = [x.strip() for x in s.split('\t') if x.strip()]
            if len(parts) < 2:
                continue
            patt = Chem.MolFromSmarts(parts[1])
            if patt is None:
                continue
            wild = tuple(i for i, a in enumerate(patt.GetAtoms()) if a.GetAtomicNum() == 0)
            n_core = patt.GetNumAtoms() - len(wild)
            out.append((parts[0], patt, wild, n_core))
    out.sort(key=lambda t: -t[3])                    # most-specific (most core atoms) first
    return [(n, p, w) for n, p, w, _ in out]


_FG_PATTS = _load_rdkit_fgs()


def rdkit_fgs(mol) -> List[Set[int]]:
    """FG atom cores (attachment ``*`` atoms excluded), most-specific-first. Overlaps between
    nested groups are resolved by the caller's claim() (free atoms only)."""
    groups = []
    for _name, patt, wild in _FG_PATTS:
        for match in mol.GetSubstructMatches(patt):
            core = set(match[i] for i in range(len(match)) if i not in wild)
            if core:
                groups.append(core)
    return groups


def _fg_key(mol, atoms):
    """Structural key for an FG atom set (frag_key, else canonical fragment SMILES), or None."""
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
    """(owner, ident) partition using RDKit's published FG dictionary. Same contract as
    fg_first_frag.partition (drop-in via cascade_bpe_linker's ``fg_partition`` hook)."""
    n = mol.GetNumAtoms()
    owner = [-1] * n
    ident: Dict[int, str] = {}

    def claim(atoms, idt):
        free = [a for a in atoms if 0 <= a < n and owner[a] == -1]
        if not free:
            return
        fid = len(ident)
        for a in free:
            owner[a] = fid
        ident[fid] = idt

    for rs in _fgf._ring_systems(mol, split_fused_aliphatic, split_fused_aromatic,
                                 break_nonarom_min, whole_ring_systems):
        free_rs = {a for a in rs if owner[a] == -1}
        if free_rs and _fgf._is_ring_set(mol, free_rs):
            claim(free_rs, _fgf._ring_identity(mol, free_rs))

    # REQUIRE_WHOLE: an FG is a match ONLY if its ENTIRE atom set is still free. If any atom was
    # already claimed (a ring took it), the FG is NOT present here — skip it and let its atoms fall
    # to the leftover tier. Matches fg_first_frag's claim(require_whole=True), so all three
    # detectors agree, and no partial/disconnected FG can ever be emitted.
    for group in rdkit_fgs(mol):
        atoms = [a for a in group if 0 <= a < n]
        if not atoms or any(owner[a] != -1 for a in atoms):   # any atom claimed -> not a match
            continue
        key = _fg_key(mol, set(atoms))
        if key is not None:
            claim(atoms, key)

    rem = [i for i in range(n) if owner[i] == -1]
    for comp in _fgf._components(mol, rem, set()):
        claim(comp, _fgf._leftover_identity(mol, comp))

    return owner, ident
