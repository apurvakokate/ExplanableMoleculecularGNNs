"""fg_first_frag.py — functional-group-first molecular fragmentation.

A generalizable alternative to retrosynthetic (rBRICS) fragmentation. The design
principle, established by measurement (see project notes):

    A fragment's IDENTITY is its chemically-canonical functional unit; substitution
    and attachment context are METADATA, not part of the identity. The vocabulary
    therefore has exactly ONE symbol per chemical function.

This fixes the two measured defects of rBRICS / cascade-merge:
  - SHATTERING: rBRICS bakes substitution context into identity, so a benzene ring
    splits into ~946 substitution variants and fluorine never appears as a bare
    motif. FG-first assigns canonical identity, so a benzene ring is one symbol.
  - INCONSISTENCY: retrosynthetic cuts depend on neighbours, so the same group is
    fragmented differently across molecules. Dictionary detection is deterministic
    and context-free.

Two products, deliberately separated:

  partition(mol)  -> atom -> fragment-id, a TRUE PARTITION (every atom in exactly
                     one fragment). Uses MINIMAL functional units (carbonyl = the
                     2-atom C=O, not the 4-atom ester) so a motif-level explainer's
                     attribution ceiling is not diluted. This is what the ceiling
                     and the atom->motif lookup use.

  presence_motifs(mol) -> the SET of motif identities present, which MAY include
                     composite groups (ester, amide) as extra features. Presence
                     features do not need to partition, so multi-resolution identity
                     lives here: both `carbonyl` and `ester` can be present.

Detection is a fixed, inspectable SMARTS dictionary (a pragmatic stand-in for
Ertl, J. Cheminform. 2017, "An algorithm to identify functional groups"). Ordered
specific -> general; earlier entries claim atoms first.

Atom indices are into Chem.MolFromSmiles(smiles) with the exact input string —
never canonicalize downstream or indices desync (same convention as the rest of
the project and the planted-GT export).
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

from rdkit import Chem

# Soft dependency: rBRICS provides (a) saturated ring-junction bonds (L18/L19) used to
# split aliphatic fused ring systems, and (b) reBRICS long-alkyl-chain cuts used to
# sub-fragment linkers into reusable ~C4 bricks. If unavailable, both fall back to the
# whole-system behaviour (aromatic-style fusion, whole chains).
try:
    import brics_rbrics as _BR
    _RBRICS_OK = True
except Exception:
    _RBRICS_OK = False

# ── Minimal functional-group units (for the PARTITION) ────────────────────────
# Each claims the SMALLEST chemically-meaningful atom set for its function, so a
# fragment is pure w.r.t. that function. Listed specific -> general; a match only
# claims atoms not already owned, so order breaks overlaps deterministically.
MINIMAL_FG: List[Tuple[str, str]] = [
    ('nitro',        '[NX3](=O)=O'),
    ('nitro_ion',    '[N+](=O)[O-]'),
    ('nitrile',      '[NX1]#[CX2]'),
    ('sulfonyl',     '[SX4](=O)(=O)'),
    ('sulfoxide',    '[SX3](=O)'),
    ('carbonyl',     '[CX3]=[OX1]'),          # covers ketone/aldehyde/acid/ester/amide C=O
    ('hydroxyl',     '[OX2H]'),
    # ether/thioether: claim ONLY the heteroatom; the flanking carbons are environment
    # (recursive SMARTS) so they are NOT pulled into the match — otherwise the group
    # cannibalises a ring/chain carbon and splits the ring (minimal-unit invariant).
    ('ether_O',      '[OX2;!$([OX2H]);$([OX2]([#6])[#6])]'),
    ('thioether_S',  '[#16X2;$([#16X2]([#6])[#6])]'),
    # amine N is detected everywhere, INCLUDING next to a carbonyl (amide N): the old
    # !$(NC=O) exclusions existed to avoid double-counting near carbonyls, which the MDL
    # contested-cluster layer now resolves properly (carbonyl+amine -> amide iff the data
    # says so). Relaxing them makes amide a genuine two-minimal-FG cluster and drops a
    # special case. An unmerged amide N is labelled amine (chemically loose but rare, since
    # frequent amides merge to fg:amide).
    ('prim_amine',   '[NX3;H2]'),
    ('amine_N',      '[NX3;!$([N+])]'),
    ('halogen',      '[F,Cl,Br,I]'),
    ('phosphate_P',  '[PX4]'),
]

# ── Composite groups (PRESENCE vocabulary only; may overlap minimal units) ─────
# Multi-resolution: these give the rule search access to larger chemical concepts
# without diluting the partition. Not used by partition().
COMPOSITE_FG: List[Tuple[str, str]] = [
    ('carboxyl',   '[CX3](=O)[OX2H1,OX1-]'),
    ('ester',      '[CX3](=O)[OX2][#6]'),
    ('amide',      '[CX3](=O)[NX3]'),
    ('carbamate',  '[NX3][CX3](=O)[OX2]'),
    ('urea',       '[NX3][CX3](=O)[NX3]'),
    ('sulfonamide','[SX4](=O)(=O)[NX3]'),
    ('trifluoromethyl','[CX4]([F])([F])[F]'),
    ('aniline',    '[NX3;H2,H1]c'),
]

_MIN = [(n, Chem.MolFromSmarts(s)) for n, s in MINIMAL_FG]
_COMP = [(n, Chem.MolFromSmarts(s)) for n, s in COMPOSITE_FG]
for (n, _), (_, p) in zip(MINIMAL_FG, _MIN):
    if p is None:
        raise ValueError(f'fg_first_frag: bad minimal SMARTS for {n!r}')


def _fuse(rings: List[Set[int]]) -> List[Set[int]]:
    """Merge sets that share an atom."""
    rings = [set(r) for r in rings]
    changed = True
    while changed:
        changed = False
        for i in range(len(rings)):
            for j in range(i + 1, len(rings)):
                if rings[i] & rings[j]:
                    rings[i] |= rings[j]
                    rings.pop(j)
                    changed = True
                    break
            if changed:
                break
    return rings


def _components(mol: Chem.Mol, atoms: Sequence[int],
                cut: Set[frozenset]) -> List[List[int]]:
    """Connected components of ``atoms`` after removing bonds in ``cut`` (frozensets
    of atom-index pairs). Only edges internal to ``atoms`` are traversed."""
    aset = set(atoms)
    seen: Set[int] = set()
    comps: List[List[int]] = []
    for s in atoms:
        if s in seen:
            continue
        stack, comp = [s], []
        while stack:
            a = stack.pop()
            if a in seen:
                continue
            seen.add(a)
            comp.append(a)
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                b = nb.GetIdx()
                if b in aset and b not in seen and frozenset((a, b)) not in cut:
                    stack.append(b)
        comps.append(comp)
    return comps


def _split_aliphatic_system(mol: Chem.Mol, atoms: Set[int]) -> List[Set[int]]:
    """Split one aliphatic fused ring system at its saturated ring junctions, using
    rBRICS's ring-junction bonds (L18=[R!#1;x3] / L19=[R!#1;x2]). Returns the split
    pieces, or ``[atoms]`` unchanged if rBRICS is unavailable, proposes no internal
    ring cut, or the split degenerates (cage guard: any single-atom piece -> keep
    whole, so e.g. adamantane is not shattered into atoms)."""
    if not _RBRICS_OK:
        return [atoms]
    cut = {frozenset((a, b)) for a, b in _BR.rbrics_bonds(mol)
           if a in atoms and b in atoms
           and mol.GetBondBetweenAtoms(a, b).IsInRing()}
    if not cut:
        return [atoms]
    pieces = _components(mol, sorted(atoms), cut)
    if len(pieces) <= 1 or any(len(p) < 2 for p in pieces):   # cage guard
        return [atoms]
    return [set(p) for p in pieces]


def _decompose_to_sssr(mol: Chem.Mol, atoms: Set[int]) -> List[Set[int]]:
    """Break a fused ring system into its individual SSSR rings; shared fusion atoms go to the
    first ring that claims them (a hard partition can't share them). A single ring (a lone
    macrocycle) has nothing smaller to split into and is returned unchanged. Partition-safe:
    the pieces are a disjoint cover of ``atoms`` (which is the union of its SSSR rings)."""
    rings = [set(r) for r in mol.GetRingInfo().AtomRings() if set(r) <= atoms]
    if len(rings) <= 1:
        return [atoms]
    seen: Set[int] = set()
    pieces: List[Set[int]] = []
    for r in rings:
        p = r - seen
        if p:
            pieces.append(p); seen |= p
    return pieces


def _is_ring_set(mol: Chem.Mol, atoms: Set[int]) -> bool:
    """Whether an atom set is still a genuine (closed) ring: it contains a cycle covering itself,
    i.e. its induced internal-bond count >= its atom count (a tree/open path has count-1 bonds). Used
    to enforce the invariant that ONLY full rings are keyed ``ring:`` — a ring broken by a fused-system
    split (its shared atoms went to another piece) is an open path and is NOT a ring anymore, so it is
    left for the leftover/linker tier and keyed as a fragment (chain:/frag:)."""
    ats = set(atoms)
    nb = sum(1 for b in mol.GetBonds()
             if b.GetBeginAtomIdx() in ats and b.GetEndAtomIdx() in ats)
    return nb >= len(ats)


def _whole_ring_systems(mol: Chem.Mol) -> List[Set[int]]:
    """Whole connected ring SYSTEMS: every maximal set of ring atoms connected through RING bonds
    (fused aromatic+aliphatic kept together — tetralin/decalin/steroid = ONE unit), separated only
    at non-ring bonds (biphenyl -> two benzene rings). Unlike SSSR/junction splitting this can NEVER
    produce an open-path remnant, so every emitted set is a genuine closed (poly)cycle — required for
    the ring-canonical keying to be self-consistent under a HARD partition (fused rings share atoms,
    so a disjoint split must open one of them). See _ring_systems(whole_ring_systems=True)."""
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    if not ring_atoms:
        return []
    cut = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()
           if not b.IsInRing()
           and b.GetBeginAtomIdx() in ring_atoms and b.GetEndAtomIdx() in ring_atoms}
    return [set(p) for p in _components(mol, sorted(ring_atoms), cut)]


def _ring_systems(mol: Chem.Mol,
                  split_fused_aliphatic: bool = True,
                  split_fused_aromatic: bool = False,
                  break_nonarom_min: Optional[int] = None,
                  whole_ring_systems: bool = False) -> List[Set[int]]:
    """Ring systems as units. Aromatic and aliphatic rings are kept SEPARATE (a benzene
    fused to a saturated ring is two distinct chemical units).

    ``split_fused_aromatic`` (default): fused aromatic systems are decomposed into their
    individual SSSR rings (naphthalene -> two 6-rings), so a ring-level planted cause is
    isolable and the fused-system long tail collapses into canonical rings (measured Pareto
    on planted ceiling + ~2x ring reuse). Shared fusion atoms go to the first ring (a hard
    partition can't share them). Set False to keep fused aromatics whole (one pi-system) when
    the causal unit is the whole polycyclic.
    Fused ALIPHATIC systems are split at their saturated ring junctions when
    ``split_fused_aliphatic`` (decalin -> two rings, steroid -> its rings), with a cage
    guard (adamantane stays whole).

    ``break_nonarom_min`` (default None = off): a NON-AROMATIC ring system with >= this many
    atoms is decomposed into its SSSR rings, so oversized fused aliphatics that the junction
    split leaves whole (steroids, alkaloid cages) are broken up. AROMATIC systems are never
    touched by this (PAHs stay whole — they are the causal unit, e.g. in mutagenicity), which
    is why it is aromaticity-selective: it breaks big non-aromatic rings, not all big rings. A
    lone macrocycle (single SSSR ring) has nothing smaller to split into and stays whole."""
    if whole_ring_systems:                    # closed-cycle-safe: never opens a fused ring
        return _whole_ring_systems(mol)
    arom, aliph = [], []
    for r in mol.GetRingInfo().AtomRings():
        rs = set(r)
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r):
            arom.append(rs)
        else:
            aliph.append(rs)
    out = list(arom) if split_fused_aromatic else _fuse(arom)
    for sysat in _fuse(aliph):
        if break_nonarom_min is not None and len(sysat) >= break_nonarom_min:
            out.extend(_decompose_to_sssr(mol, sysat))
        elif split_fused_aliphatic:
            out.extend(_split_aliphatic_system(mol, sysat))
        else:
            out.append(sysat)
    return out


# Ring-identity mode. 'composition' (default) keys a ring by aromatic/aliphatic + element
# formula (ring:aromatic:C6); 'canonical' keys it by its own neutralised tautomer-canonical
# SMILES (ring:c1ccccc1), which DISTINGUISHES isomers (pyrimidine != pyrazine, quinoline !=
# isoquinoline) yet keeps benzene ONE token regardless of substituents. Toggle with
# set_ring_identity(); this changes ONLY ring labels, never the atom partition (so the
# attribution ceiling is identical). See the fragmentation-keying note in the write-ups.
_RING_IDENTITY = 'composition'
_RING_CANON: Dict[str, object] = {}


def set_ring_identity(mode: str) -> None:
    """Select ring keying: 'composition' (default) or 'canonical'. See _ring_identity."""
    global _RING_IDENTITY
    if mode not in ('composition', 'canonical'):
        raise ValueError(f"ring_identity must be 'composition' or 'canonical', got {mode!r}")
    _RING_IDENTITY = mode


def _canon_smiles(mol: Chem.Mol, atoms: Set[int]) -> Optional[str]:
    """Bare canonical SMILES of a sub-fragment: internal bonds only, exocyclic bonds -> implicit H,
    NEUTRALISED + tautomer-canonicalised, no dummies. Shared by ring AND linker keying.

    Robust to aromatic N-heterocycles (imidazole/pyrazole/triazole, N-substituted or not): the
    parent is KEKULISED (explicit single/double bond orders) BEFORE the fragment is cut out, so the
    isolated ring never has to be re-kekulised from bare aromatic flags (which fails for a pyrrole-
    type N that lost its substituent). Exocyclic bonds are then replaced by RDKit-computed implicit
    H, giving the substituent-agnostic skeleton (benzene == chlorobenzene-ring, pyridine-N-oxide ==
    pyridine). Returns None only on genuinely un-sanitisable input; callers must NOT silently fall
    back to a conflating key (see _ring_identity)."""
    if not _RING_CANON:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        _RING_CANON['taut'] = rdMolStandardize.TautomerEnumerator()
        _RING_CANON['unch'] = rdMolStandardize.Uncharger()
    keep = set(atoms)
    try:
        mk = Chem.Mol(mol)
        Chem.Kekulize(mk, clearAromaticFlags=True)    # explicit bond orders before cutting
    except Exception:
        mk = mol                                      # already kekulised / no aromatics
    rw = Chem.RWMol(mk)
    for idx in sorted((a.GetIdx() for a in mk.GetAtoms() if a.GetIdx() not in keep), reverse=True):
        rw.RemoveAtom(idx)
    m2 = rw.GetMol()
    for a in m2.GetAtoms():                            # exocyclic bonds -> RDKit-filled implicit H
        a.SetNoImplicit(False); a.SetNumExplicitHs(0)  # charge dropped by Uncharger after sanitize
    try:
        Chem.SanitizeMol(m2)
        try: m2 = _RING_CANON['unch'].uncharge(m2) or m2
        except Exception: pass
        mt = _RING_CANON['taut'].Canonicalize(m2); m2 = mt if mt else m2
        return Chem.MolToSmiles(m2, canonical=True, isomericSmiles=False)
    except Exception:
        return None


def _ring_canonical(mol: Chem.Mol, atoms: Set[int]) -> Optional[str]:
    """Ring skeleton canonical SMILES (see _canon_smiles), prefixed 'ring:'. None on failure."""
    k = _canon_smiles(mol, atoms)
    return ('ring:' + k) if k is not None else None


def _connected(mol: Chem.Mol, atoms) -> bool:
    """Whether the given atom set induces a single connected subgraph."""
    atoms = set(atoms)
    if len(atoms) <= 1:
        return True
    seen: Set[int] = set(); stack = [next(iter(atoms))]
    while stack:
        a = stack.pop(); seen.add(a)
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            j = nb.GetIdx()
            if j in atoms and j not in seen:
                stack.append(j)
    return seen == atoms


def _ring_identity(mol: Chem.Mol, atoms: Set[int]) -> str:
    """Ring-system identity. Mode via set_ring_identity():
      'composition' (default): aromatic/aliphatic + element formula, NOT substituents, so
        c1ccccc1 and c1ccc(F)cc1 share `ring:aromatic:C6` and pyridine is `ring:aromatic:C5N1`.
        Coarse — conflates isomers (pyrimidine == pyrazine == pyridazine).
      'canonical': neutralised tautomer-canonical ring SMILES, so isomers separate while benzene
        stays ONE token. NO fallback: canonicalisation is robust to aromatic-N heterocycles (see
        _canon_smiles); if it ever fails we RAISE rather than silently emit a conflating
        composition key (which would give one fragment two representations across the dataset)."""
    if _RING_IDENTITY == 'canonical':
        k = _ring_canonical(mol, atoms)
        if k is None:
            raise ValueError(
                f"canonical ring keying failed for atoms {sorted(atoms)} in "
                f"{Chem.MolToSmiles(mol)!r} — refusing to fall back to a conflating composition key")
        return k
    arom = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms)
    from collections import Counter
    elems = Counter(mol.GetAtomWithIdx(i).GetSymbol() for i in atoms)
    formula = ''.join(f'{el}{elems[el]}' for el in sorted(elems))
    return f"ring:{'aromatic' if arom else 'aliphatic'}:{formula}"


def _leftover_identity(mol: Chem.Mol, comp: Sequence[int]) -> str:
    """Linker keyed by its own CANONICAL SMILES (connectivity-preserving; linkers are connected
    components so this is always clean): pure-carbon -> chain:<smiles>, heteroatom -> frag:<smiles>.
    Falls back to composition (chain:C<len> / frag:<elems>) only if canonicalisation fails."""
    pure_c = all(mol.GetAtomWithIdx(a).GetAtomicNum() == 6 for a in comp)
    smi = _canon_smiles(mol, comp)
    if smi is None:
        c = Counter(mol.GetAtomWithIdx(a).GetSymbol() for a in comp)
        smi = f'C{len(comp)}' if pure_c else ''.join(f'{e}{c[e]}' for e in sorted(c))
    return ('chain:' if pure_c else 'frag:') + smi


def partition(mol: Chem.Mol,
              split_fused_aliphatic: bool = True,
              subcut_chains: bool = True,
              split_fused_aromatic: bool = False,
              break_nonarom_min: Optional[int] = None,
              whole_ring_systems: bool = False) -> Tuple[List[int], Dict[int, str]]:
    """Partition atoms into fragments. Returns (owner[atom]->frag_id, frag_id->identity).

    Priority: minimal FGs (specific first) -> ring systems -> remaining atoms (chains /
    linkers).

    ``split_fused_aliphatic`` splits saturated fused ring systems at their junctions
    (rBRICS L18/L19; cage-guarded). ``subcut_chains`` sub-fragments long alkyl linkers
    into reusable ~C4 bricks (reBRICS) instead of leaving a whole rare `chain:C<n>`.
    Both require rBRICS; without it they no-op. Every atom still lands in exactly one
    fragment (partition invariant preserved).
    """
    n = mol.GetNumAtoms()
    owner = [-1] * n
    ident: Dict[int, str] = {}

    def claim(atoms, identity, require_whole=False):
        # require_whole: a multi-atom FG must be present in its ENTIRETY in the free atoms. If ANY of
        # its match atoms is already claimed (e.g. the carbonyl C is a ring atom, so [CX3]=[OX1] would
        # otherwise claim only the lone =O), the group is not actually present -> skip. The orphaned
        # atoms then fall to the leftover tier and are keyed by their real structure (a bare =O becomes
        # an oxo, never a lone-O 'carbonyl'). Subsumes the old connected-guard: a cyclic sulfone (S in
        # ring) leaves two disconnected =O, and a cyclic carbonyl one =O — require_whole skips both.
        if require_whole and any(owner[a] != -1 for a in atoms):
            return
        free = [a for a in atoms if owner[a] == -1]
        if not free:
            return
        fid = len(ident)
        for a in free:
            owner[a] = fid
        ident[fid] = identity

    # Ring bodies are claimed BEFORE functional groups so a ring HETEROATOM (the O in
    # THF/morpholine, the N in piperidine/piperazine) stays part of its ring instead of
    # being stolen by the ether/amine SMARTS and shattering the ring. FGs then claim only
    # the remaining (exocyclic) atoms, which still separates ring SUBSTITUENTS (Ar-OH,
    # Ar-COOH, Ar-Cl) since those atoms are not ring atoms. For a ring-embedded FG whose
    # match includes a ring atom (cyclic carbonyl/sulfone: the C/S is in the ring), the FG
    # is NOT present in the free atoms, so require_whole skips it; its exocyclic atom(s)
    # (a bare =O) fall to the leftover tier and are keyed by their real structure (an oxo),
    # never a lone-atom 'carbonyl'. The ring stays whole, the right attribution unit.
    for rs in _ring_systems(mol, split_fused_aliphatic, split_fused_aromatic,
                            break_nonarom_min, whole_ring_systems):
        # Key by the atoms actually CLAIMABLE here (free of earlier claims): under a fused-system split
        # the shared atoms already went to an earlier ring, so this piece's free atoms may be an open
        # path. INVARIANT: only a full (closed) ring is keyed ring:; a broken remnant is skipped so it
        # falls to the leftover tier and is keyed as a fragment. (Whole-systems mode: free==rs, closed.)
        free_rs = {a for a in rs if owner[a] == -1}
        if free_rs and _is_ring_set(mol, free_rs):
            claim(free_rs, _ring_identity(mol, free_rs))

    for name, patt in _MIN:
        for m in mol.GetSubstructMatches(patt):
            claim(m, f'fg:{name}', require_whole=True)

    # remaining atoms -> connected components (chains, linkers)
    rem = [i for i in range(n) if owner[i] == -1]
    # reBRICS long-alkyl cuts internal to the leftover -> reusable ~C4 bricks
    chain_cut: Set[frozenset] = set()
    if subcut_chains and _RBRICS_OK and rem:
        remset = set(rem)
        chain_cut = {frozenset((a, b)) for a, b in _BR.rbrics_full_bonds(mol)
                     if a in remset and b in remset}
    for comp in _components(mol, rem, chain_cut):
        claim(comp, _leftover_identity(mol, comp))

    # FG-completeness pass: if a whole leftover fragment IS a functional group in the ORIGINAL
    # molecule, promote it to the FG tier (never let a real FG hide as filler). Match on the
    # original structure — never on the H-capped leftover SMILES, which could invent a phantom FG.
    frag_atoms: Dict[int, Set[int]] = {}
    for a in range(n):
        f = owner[a]
        if f >= 0 and ident[f].startswith(('chain:', 'frag:')):
            frag_atoms.setdefault(f, set()).add(a)
    for name, patt in _MIN:
        for match in mol.GetSubstructMatches(patt):
            ms = set(match)
            for fid, ats in frag_atoms.items():
                if ats == ms:                    # a whole leftover fragment is exactly this FG
                    ident[fid] = f'fg:{name}'

    return owner, ident


def fragment(mol: Chem.Mol) -> List[Tuple[str, Set[int]]]:
    """[(identity, atom_set)] for the partition. Convenience over :func:`partition`."""
    owner, ident = partition(mol)
    groups: Dict[int, Set[int]] = {}
    for a, f in enumerate(owner):
        groups.setdefault(f, set()).add(a)
    return [(ident[f], atoms) for f, atoms in groups.items()]


def rekey_structural(mol: Chem.Mol,
                     fragments: List[Tuple[str, Set[int]]]) -> List[Tuple[str, Set[int]]]:
    """FINAL motif keying (structural, no name tables): rings KEEP their substituent-agnostic
    ring-canonical key (the one exception — so benzene stays one motif); EVERY other motif (FGs,
    linkers, MDL-merged units) is re-keyed by ``chemfrag.frag_key`` — canonical SMILES with ``[*]``
    attachment dummies. This is the uniform structural mapping: halogens split (``*F``/``*Cl``/
    ``*Br``/``*I``), hydroxyl ``*O`` != ether ``*O*``, a bare oxo is ``*=O`` (never a lone 'carbonyl'),
    ketone ``*C(*)=O`` != aldehyde ``*C=O``. Apply this to the tier-tagged output of :func:`fragment`
    (or cascade_bpe_linker.apply_rules), which keeps ``fg:``/``chain:``/``frag:`` tags internally so
    the MDL merge logic still works. Rings are detected by the ``ring:`` prefix.

    ``fragments`` = ``[(identity, atomset)]``. Returns the same list with non-ring identities replaced
    by their frag_key (falling back to the original identity only if frag_key fails)."""
    import chemfrag as _cf
    out: List[Tuple[str, Set[int]]] = []
    for idt, atoms in fragments:
        if idt.startswith('ring:'):
            out.append((idt, atoms))
        else:
            k = _cf.frag_key(mol, set(atoms))
            out.append((k if k else idt, atoms))
    return out


def _frag_skeleton(key: str) -> Optional[str]:
    """H-capped canonical skeleton of a non-ring frag key: drop its ``[*]`` attachment dummies and
    canonicalise (bond neighbours pick up implicit H). A bare whole-molecule key (no ``*``) is already
    its own skeleton. Rings return None (never folded). Used to fold a whole-molecule motif into a
    matching attachment-bearing fragment (same skeleton)."""
    if key.startswith('ring:'):
        return None
    m = Chem.MolFromSmiles(key)
    if m is None:
        return None
    if not any(a.GetAtomicNum() == 0 for a in m.GetAtoms()):     # already bare (whole molecule)
        return Chem.MolToSmiles(m, isomericSmiles=False)
    rw = Chem.RWMol(m)
    for a in sorted((a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0), reverse=True):
        rw.RemoveAtom(a)
    m2 = rw.GetMol()
    try:
        Chem.SanitizeMol(m2)
        return Chem.MolToSmiles(m2, isomericSmiles=False)
    except Exception:
        return None


def fold_whole_molecule_keys(mol_frags: List[List[Tuple[str, Set[int]]]]
                             ) -> List[List[Tuple[str, Set[int]]]]:
    """Vocabulary-level pass: a motif that spans a COMPLETE molecule has no attachment, so frag_key
    emits a bare SMILES (e.g. ``C=C`` for an ethylene record). If that same chemical skeleton also
    occurs as an attachment-bearing fragment elsewhere (``*C=C*`` in larger molecules), the whole
    molecule is the SAME chemical unit and is folded onto that fragment key — picking the highest-
    support match when several exist (``*C=C*`` over ``*C=C``). A bare key with no attachment-bearing
    match (e.g. carbon disulfide ``S=C=S`` — never a fragment of anything) is kept as-is. Rings are
    never touched. Returns a new list with the bare keys remapped in place."""
    from collections import Counter
    support: Counter = Counter()
    for frags in mol_frags:
        for idt, _ in frags:
            support[idt] += 1
    bare = [k for k in support if not k.startswith('ring:') and '*' not in k]
    if not bare:
        return mol_frags
    # skeleton -> best attachment-bearing key (max support)
    star_by_skel: Dict[str, Tuple[str, int]] = {}
    for k, c in support.items():
        if '*' in k and not k.startswith('ring:'):
            sk = _frag_skeleton(k)
            if sk is not None and (sk not in star_by_skel or c > star_by_skel[sk][1]):
                star_by_skel[sk] = (k, c)
    remap: Dict[str, str] = {}
    for b in bare:
        sk = _frag_skeleton(b)
        if sk is not None and sk in star_by_skel:
            remap[b] = star_by_skel[sk][0]
    if not remap:
        return mol_frags
    return [[(remap.get(idt, idt), atoms) for idt, atoms in frags] for frags in mol_frags]


def presence_motifs(mol: Chem.Mol, include_composite: bool = True) -> Set[str]:
    """Set of motif identities present in the molecule (multi-resolution).

    Partition identities PLUS composite FG identities when present. Used to build
    the molecule x motif indicator matrix for rule candidates and the LR probe.
    """
    ids = {idt for idt, _ in fragment(mol)}
    if include_composite:
        for name, patt in _COMP:
            if patt is not None and mol.HasSubstructMatch(patt):
                ids.add(f'fg:{name}')
    return ids


def _is_leftover(idt: str) -> bool:
    return idt.startswith('chain:') or idt.startswith('frag:')


def _pool(idt: str, keep: bool) -> str:
    """Rare leftover -> coarse bucket. Heads/bodies (fg:/ring:) are never pooled."""
    if keep or not _is_leftover(idt):
        return idt
    return 'chain:pooled' if idt.startswith('chain:') else 'frag:pooled'


def build_presence(smiles: Sequence[str], tau: float = 0.01,
                   include_composite: bool = True):
    """Corpus-level pooled motif vocabulary + indicator matrix.

    Frequency-pools LEFTOVER (chain:/frag:) identities occurring in fewer than
    ``tau`` of molecules into ``chain:pooled`` / ``frag:pooled`` — the data-driven
    linker tier. Heads (fg:) and bodies (ring:) are never pooled. ``tau=0`` disables
    pooling.

    Returns ``(vocab, M, mol_sets)``: sorted vocab list, boolean [n_mols x n_vocab]
    indicator matrix, and the per-molecule pooled motif sets. Molecules that fail to
    parse contribute an all-False row (kept so row indices align with ``smiles``).
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    raw = [presence_motifs(m, include_composite) if m is not None else set()
           for m in mols]
    # leftover frequencies (unpooled) drive the keep/pool decision
    freq: Counter = Counter()
    for s in raw:
        for idt in s:
            if _is_leftover(idt):
                freq[idt] += 1
    thr = tau * len(smiles)
    keep = {idt for idt, c in freq.items() if c >= thr}
    mol_sets = [{_pool(idt, idt in keep or not _is_leftover(idt)) for idt in s}
                for s in raw]
    vocab = sorted({idt for s in mol_sets for idt in s})
    vi = {s: i for i, s in enumerate(vocab)}
    import numpy as _np
    M = _np.zeros((len(smiles), len(vocab)), dtype=bool)
    for r, s in enumerate(mol_sets):
        for idt in s:
            M[r, vi[idt]] = True
    return vocab, M, mol_sets


def _selftest() -> None:
    """Partition invariants + a few known fragmentations. Run: python fg_first_frag.py"""
    # heads/bodies are asserted; leftover identity is not (reBRICS may sub-cut chains)
    cases = {
        'CCOC(=O)c1ccc(F)cc1': {'fg:carbonyl', 'fg:halogen', 'ring:aromatic:C6'},
        'O=[N+]([O-])c1ccccc1': {'fg:nitro_ion', 'ring:aromatic:C6'},
        'CCCCCC(=O)O': {'fg:carbonyl', 'fg:hydroxyl'},
        'c1ccc2ccccc2c1': {'ring:aromatic:C10'},          # naphthalene stays one aromatic unit
    }
    for smi, expect in cases.items():
        m = Chem.MolFromSmiles(smi)
        owner, ident = partition(m)
        assert -1 not in owner, f'{smi}: unpartitioned atom'
        assert len(owner) == m.GetNumAtoms(), f'{smi}: size mismatch'
        # every atom in exactly one fragment (partition invariant)
        seen = [0] * m.GetNumAtoms()
        for a in range(m.GetNumAtoms()):
            seen[a] += 1
        assert all(c == 1 for c in seen)
        present = presence_motifs(m)
        missing = expect - present
        assert not missing, f'{smi}: expected {expect}, missing {missing} (got {sorted(present)})'
    print('fg_first_frag selftest: OK')


if __name__ == '__main__':
    _selftest()
