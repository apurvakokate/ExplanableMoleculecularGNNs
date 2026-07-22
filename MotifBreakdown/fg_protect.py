"""fg_protect.py — protect functional groups as explicit, atomic motifs.

Carves protected groups out of the per-molecule tracked fragments produced by EITHER
fragmentation engine (legacy rbrics or the v4 cascade), emitting each protected group
as its OWN fragment and re-keying the remainder. Purely in atom-index space (indices
into ``Chem.MolFromSmiles(orig_smi)`` — the shared convention of both engines), so the
atom-tracking partition invariant is preserved: every atom ends up in exactly one
fragment.

Effect on the vocabulary: the protected group becomes a single, high-support motif
instead of being split across variants or absent entirely.

TWO DETECTION MODES
-------------------
1. CONNECTIVITY (default, used for mutag): nitro + aniline, detected via neighbour
   counting. Required because mutag's SMILES are reconstructed from TUDataset graphs
   with NO bond orders and NO aromaticity, so `=O` / aromatic-`c` SMARTS match nothing.
2. SMARTS (opt-in, per dataset via PROTECT_SMARTS): for datasets whose SMILES carry
   real chemistry. Used for the *_Verified_GT planted-GT datasets, where we protect
   exactly the ground-truth rule's own SMARTS so the vocabulary can express the causal
   substructure.

WHY THE *_Verified_GT ENTRIES EXIST
-----------------------------------
Measured on logic7 (Fluoride_Carbonyl): the true rule is `[FX1] AND [CX3]=O`, but
NEITHER vocabulary contains a bare fluorine motif — F only ever appears fused into
larger fragments (141 F-containing motifs in rbrics, 105 in all_fallback_bpe, 0 that
are just F; max coverage of any single one: 2.45% / 3.54%). The true rule is therefore
not expressible, and the measured attribution ceiling is 0.913 / 0.935 with 0% of
molecules perfectly explainable. Protecting the GT SMARTS makes the causal group a
first-class motif; the ceiling should then rise toward 1.0. Same molecules, same
labels, same GT, same metric — only the vocabulary changes. That is the controlled
experiment showing the ceiling is CAUSALLY a property of the hypothesis space.
"""
from typing import Callable, Dict, List, Optional, Set, Tuple

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

# ── Per-dataset SMARTS protection ────────────────────────────────────────────
# Patterns are copied VERBATIM from the dataset's own generator:
# google-research/graph-attribution graph_attribution/tasks.py:284-297. Protecting
# the generator's exact SMARTS is what makes the ground-truth substructure
# expressible in the vocabulary.
#
# ORDER MATTERS: earlier entries claim their atoms first; a later pattern that
# overlaps an already-claimed atom is skipped (see the dedup in _dedup_nonoverlapping).
# Carbonyl is listed before the alkane chain so a carbonyl C is never swallowed by the
# chain matcher.
PROTECT_SMARTS: Dict[str, List[Tuple[str, str]]] = {
    'Benzene_Verified_GT': [
        ('benzene', 'c1ccccc1'),
    ],
    'Fluoride_Carbonyl_Verified_GT': [
        ('carbonyl', '[CX3]=O'),
        ('fluoride', '[FX1]'),
    ],
    'Alkane_Carbonyl_Verified_GT': [
        ('carbonyl', '[CX3]=O'),
        ('unbranched_alkane', '[R0;D2,D1][R0;D2][R0;D2,D1]'),
    ],
}


def _dedup_nonoverlapping(
    cand: List[Tuple[str, frozenset, frozenset]],
) -> List[Tuple[str, frozenset, frozenset]]:
    """Keep candidates in order, dropping any that overlap an already-claimed atom.

    The vocabulary requires a partition, so overlapping protected groups cannot all
    survive. First-come-first-served by PROTECT_SMARTS order.

    For a self-overlapping pattern (e.g. the 3-atom chain sliding along a 5-carbon
    chain: {0,1,2},{1,2,3},{2,3,4}) only the first window is protected and the rest of
    the chain stays in its base fragment. That is sufficient: ground truth is graded
    max-over-explanations, so one expressible GT column is enough to reach ceiling 1.0
    for that molecule.
    """
    out: List[Tuple[str, frozenset, frozenset]] = []
    used: Set[int] = set()
    for tag, s, kctx in cand:
        if not s or (s & used):
            continue
        out.append((tag, s, kctx))
        used |= set(s)
    return out


def protected_atomsets_smarts(
    mol: Chem.Mol, specs: List[Tuple[str, str]]
) -> List[Tuple[str, frozenset, frozenset]]:
    """SMARTS-based protection. Owned atomset == the match; key_context == same."""
    cand: List[Tuple[str, frozenset, frozenset]] = []
    for tag, smarts in specs:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            raise ValueError(f'fg_protect: bad SMARTS for {tag!r}: {smarts!r}')
        for match in mol.GetSubstructMatches(patt):
            s = frozenset(match)
            cand.append((tag, s, s))
    return _dedup_nonoverlapping(cand)


def _h_count(atom) -> int:
    """Total hydrogens whether implicit or explicit-as-atoms (mutag uses explicit H)."""
    return atom.GetTotalNumHs() + sum(nb.GetAtomicNum() == 1 for nb in atom.GetNeighbors())


def protected_atomsets(
    mol: Chem.Mol, dataset: Optional[str] = None
) -> List[Tuple[str, frozenset, frozenset]]:
    """Return [(tag, atomset, key_context)] for each protected group, non-overlapping.

    Dispatches on ``dataset``: if it has a PROTECT_SMARTS entry, use SMARTS detection;
    otherwise fall back to the connectivity-based nitro+aniline path (mutag and any
    dataset without an explicit entry). ``dataset=None`` preserves the original
    behaviour exactly.

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
    if dataset is not None and dataset in PROTECT_SMARTS:
        return protected_atomsets_smarts(mol, PROTECT_SMARTS[dataset])
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
    return _dedup_nonoverlapping(cand)


def carve_protected(mol: Chem.Mol, tracked_frags: List[Frag], keyer: Keyer,
                    dataset: Optional[str] = None) -> List[Frag]:
    """Emit protected groups as standalone fragments; re-key the remainder.

    ``keyer(mol, atomset) -> smarts`` must be the engine-appropriate fragment keyer
    (chemfrag.frag_key for v4; the legacy canonicaliser for rbrics) so the protected
    and remainder motifs use the same canonical form as the rest of that vocabulary.

    ``dataset`` selects the detection mode (see :func:`protected_atomsets`).
    """
    prot = protected_atomsets(mol, dataset)
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
