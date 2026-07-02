"""fg_protect.py — protect functional-group toxicophores as explicit, atomic motifs.

Carves protected groups (nitro -NO2, aromatic primary amine / aniline c-NH2) out of
the per-molecule tracked fragments produced by EITHER fragmentation engine (legacy
rbrics or the v4 cascade), emitting each protected group as its OWN fragment and
re-keying the remainder. Purely in atom-index space (indices into
``Chem.MolFromSmiles(orig_smi)`` — the shared convention of both engines), so the
atom-tracking partition invariant is preserved: every atom ends up in exactly one
fragment.

Effect on the vocabulary: the toxicophore becomes a single, high-support motif instead
of being split across variants (nitro) or absent (aromatic amino) — see the mutag
diagnosis in the project docs.
"""
from typing import Callable, List, Optional, Set, Tuple

from rdkit import Chem

import chemfrag as C

# NOTE: the mutag (Mutagenicity) SMILES the pipeline consumes are reconstructed from
# the TUDataset graph with NO bond orders and NO aromaticity (all single bonds, explicit
# H). So bond-order/aromaticity SMARTS (=O, aromatic c) match nothing. Detection is
# therefore CONNECTIVITY-based, which also works on proper-chemistry SMILES:
#   nitro    = an N with >= 2 oxygen neighbours  → protect {N, O, O}
#   aniline  = a ring-attached primary amine: N carrying H, bonded to a ring carbon,
#              and not itself a nitro N        → protect {N}
# (aromaticity is unavailable, so "ring carbon" is the aromatic-ring proxy.)

Frag = Tuple[str, Set[int]]
Keyer = Callable[[Chem.Mol, Set[int]], Optional[str]]


def _h_count(atom) -> int:
    """Total hydrogens whether implicit or explicit-as-atoms (mutag uses explicit H)."""
    return atom.GetTotalNumHs() + sum(nb.GetAtomicNum() == 1 for nb in atom.GetNeighbors())


def protected_atomsets(mol: Chem.Mol) -> List[Tuple[str, frozenset, frozenset]]:
    """Return [(tag, atomset, key_context)] for each protected group, non-overlapping.

    ``atomset``     = atoms the motif OWNS (used for the partition / atom->motif map).
    ``key_context`` = atoms used only to CANONICALISE the motif's identity string
                      (a superset of atomset); its extra atoms stay in their own
                      fragment. Lets us give aniline a distinct label without
                      claiming (or removing) any ring atom.

    Connectivity-based (see module note):
      nitro   = N with >=2 O neighbours → owns {N, O, O}; key_context = same.
      aniline = ring-attached primary amine → owns {N} only (matches the amino GT,
                which in the heavy-atom representation is just the N — the H's are
                implicit / not nodes); key_context = {N, ipso ring C} so the label
                is ``*C(*)N`` (distinct from a generic bridging ``*N``)."""
    nitro_n: Set[int] = set()
    cand: List[Tuple[str, frozenset, frozenset]] = []
    # nitro first (so its N is excluded from the amine pass)
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 7:
            continue
        o_nbrs = [nb.GetIdx() for nb in a.GetNeighbors() if nb.GetAtomicNum() == 8]
        if len(o_nbrs) >= 2:
            nitro_n.add(a.GetIdx())
            s = frozenset({a.GetIdx()} | set(o_nbrs))
            cand.append(('nitro', s, s))
    # ring-attached primary amine (aniline analog)
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 7 or a.GetIdx() in nitro_n:
            continue
        heavy = [nb for nb in a.GetNeighbors() if nb.GetAtomicNum() > 1]
        ring_c = [nb for nb in heavy if nb.GetAtomicNum() == 6 and nb.IsInRing()]
        # primary amine on a ring: exactly one heavy neighbour (the ring C), H filling the rest
        if _h_count(a) >= 1 and len(heavy) == 1 and ring_c:
            cand.append(('aniline', frozenset({a.GetIdx()}),
                         frozenset({a.GetIdx(), ring_c[0].GetIdx()})))
    # dedup identical owned-sets and drop any that overlap an already-owned set
    out: List[Tuple[str, frozenset, frozenset]] = []
    used: Set[int] = set()
    for tag, s, kctx in cand:
        if not s or (s & used):
            continue
        out.append((tag, s, kctx))
        used |= set(s)
    return out


def carve_protected(mol: Chem.Mol, tracked_frags: List[Frag], keyer: Keyer) -> List[Frag]:
    """Emit protected groups as standalone fragments; re-key the remainder.

    ``keyer(mol, atomset) -> smarts`` must be the engine-appropriate fragment keyer
    (chemfrag.frag_key for v4; the legacy canonicaliser for rbrics) so the protected
    and remainder motifs use the same canonical form as the rest of that vocabulary.
    """
    prot = protected_atomsets(mol)
    if not prot:
        return tracked_frags

    # membership uses the OWNED atomsets only (key_context's extra atoms — e.g. the
    # aniline ipso ring C — are NOT removed, so the ring stays intact).
    prot_atoms: Set[int] = set().union(*[set(s) for _, s, _ in prot])

    def _key(aset: Set[int]) -> Optional[str]:
        k = keyer(mol, aset)
        return k if k is not None else C.frag_key(mol, aset)

    out: List[Frag] = []
    # 1) each protected group as its own motif: label from key_context, own atomset
    for _, s, kctx in prot:
        k = _key(set(kctx))
        if k is not None:
            out.append((k, set(s)))
    # 2) remainder of every base fragment, split into connected components
    for _smarts, aset in tracked_frags:
        rem = set(aset) - prot_atoms
        if not rem:
            continue
        for comp in C.comps_in(mol, rem, []):
            k = _key(set(comp))
            if k is not None:
                out.append((k, set(comp)))
    return out
