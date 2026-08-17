"""entropy_bde_frag.py — objective-independent, threshold-free fragmentation.

Cut molecular bonds where structure articulates, decided by three parameter-light gates that each
veto a distinct over-cutting failure mode. No tuned numeric threshold anywhere.

    Gate 0 (chemistry, ring-protection): candidate bonds = CRUSH-valid acyclic cuts, rings never cut.
        Reused verbatim from crush_direct._meaningful_bonds (RECAP+BRICS+CRUSH SMIRKS, fused-protected).
    Gate 1  (data signal): each bond gets a corpus BRANCHING ENTROPY — how variable its partner is
        across the corpus given one side's radius-1 environment. High = interchangeable joint
        (substituent/linker); low = cohesive/bound (ring-internal, rigid unit). Unsupervised: never
        touches the label, so the vocabulary stays objective-independent.
    Gate 2  (threshold-free selector): cut a candidate bond iff its entropy is a STRICT LOCAL MAXIMUM
        among adjacent bonds (Harris / Tanaka-Ishii articulation, adapted to the molecular graph).
        Relative, not absolute — a uniform chain has no peak, so it is NOT shattered; only bonds that
        stand out from their neighbourhood are cut. This is what replaces a tuned entropy cutoff.
    Gate 3  (physics, categorical): cut only if the bond's dissociation energy is below the carbon
        skeleton line (< ~82 kcal/mol, i.e. weaker than C-C). Protects the C-C backbone AND the strong
        C-O / C-F functional-group bonds that Gate 2 alone would peel. Corpus-free, dataset-independent;
        any value in (81, 83) gives the identical partition, so it is a bond-class category, not a fit.
    Gate 4  (optional): terminal single-heteroatom exception. Allow peeling a degree-1 heteroatom
        (F/Cl/Br/I/O/N/S with one heavy neighbour) whose bond is a local max, EVEN if Gate 3 would
        protect it — re-enables isolating variable substituents. Internal heteroatoms (e.g. an ester O,
        degree 2) are untouched, so functional groups stay whole. Off by default.

Methods:
    'lm_strict'  = Gate 0 + Gate 2                       (entropy local-maxima only; shatters C-O groups)
    'lm_bde'     = Gate 0 + Gate 2 + Gate 3              (RECOMMENDED; keeps skeleton + C-O groups whole)
    'lm_bde_g4'  = Gate 0 + Gate 2 + Gate 3 + Gate 4     (as lm_bde but peels terminal heteroatoms)

Entry point build(smiles_all, method=...) matches the crush_direct.build / ring_mdl.build contract:
returns per-molecule ``[(key, set(atoms))]`` aligned to smiles_all (empty list for unparseable SMILES).
Branching-entropy statistics are estimated once from smiles_all (the whole corpus passed in), then each
molecule is fragmented independently.

Empirical basis (Aug 2026, fold-0 x 5 datasets BBBP/hERG/Mutagenicity/esol/Lipophilicity): lm_bde gives
the lowest functional-group over-fragmentation (6-17%) and highest cross-molecule cut-consistency
(89-96%) of every method tried — beating rBRICS (23-33% / 62-75%) and CRUSH-direct (12-18% / 68-76%),
with rings 100% intact and ester/ether/carboxylic-acid kept whole (0% shattered). It buys that
consistency conservatively (high under-fragmentation / larger vocab); see the completeness-R2 / GT-ROC
evaluation for whether that trade helps downstream attribution.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from rdkit import Chem

import chemfrag as cf
from crush_direct import _meaningful_bonds, _components, _fused_protected  # Gate 0 pool + helpers
import ertl_frag   # Ertl-IFG functional-group detection (algorithmic, complete over heteroatom FGs)

# ─────────────────────────── Gate 3: physical bond cost (BDE) ──────────────────────────
# Textbook average single-bond dissociation energies (kcal/mol). Discrete lookup, corpus-free.
BDE = {('C', 'C'): 83, ('C', 'N'): 73, ('C', 'O'): 85, ('C', 'F'): 115, ('C', 'Cl'): 81,
       ('C', 'Br'): 68, ('C', 'I'): 51, ('C', 'S'): 65, ('C', 'P'): 65, ('N', 'N'): 39,
       ('N', 'O'): 48, ('O', 'O'): 35, ('N', 'S'): 45, ('O', 'S'): 52, ('S', 'S'): 54}
SKELETON_LINE = 82   # cut only bonds weaker than the C-C skeleton (categorical; any 81<x<83 identical)


def _bde(bond):
    """Dissociation energy (kcal/mol) of an acyclic bond, or None for rings/aromatic (protected)."""
    if bond.IsInRing() or bond.GetBondType() == Chem.BondType.AROMATIC:
        return None
    a, b = bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()
    v = BDE.get(tuple(sorted((a, b))), 80)   # 80 = neutral default for unlisted pairs
    if bond.GetBondType() == Chem.BondType.DOUBLE:
        v += 60
    elif bond.GetBondType() == Chem.BondType.TRIPLE:
        v += 110
    return v


# ────────────────────────── Gate 1: corpus branching entropy ───────────────────────────
def _atom_desc(a):
    """radius-1 atom environment descriptor: element + aromatic + in-ring + heavy-degree."""
    return f"{a.GetSymbol()}{'a' if a.GetIsAromatic() else ''}{'R' if a.IsInRing() else ''}{a.GetDegree()}"


def _side_context(atom, other_idx):
    """One side of a bond: (this atom's desc, sorted descs of its OTHER heavy neighbours)."""
    return (_atom_desc(atom),
            tuple(sorted(_atom_desc(w) for w in atom.GetNeighbors() if w.GetIdx() != other_idx)))


def _shannon(counter):
    n = sum(counter.values())
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in counter.values() if c > 0)


def build_corpus_stats(mols):
    """One corpus pass → CTX[side_context] = Counter(partner atom-desc). Estimates, for every local
    environment, the distribution of what it bonds to; branching entropy is read off this."""
    ctx = defaultdict(Counter)
    for m in mols:
        if m is None:
            continue
        for b in m.GetBonds():
            u, v = b.GetBeginAtom(), b.GetEndAtom()
            ctx[_side_context(u, v.GetIdx())][_atom_desc(v)] += 1
            ctx[_side_context(v, u.GetIdx())][_atom_desc(u)] += 1
    return ctx


def _bond_entropy(bond, ctx):
    """Branching entropy of a bond = max variability seen from either side across the corpus."""
    u, v = bond.GetBeginAtom(), bond.GetEndAtom()
    return max(_shannon(ctx[_side_context(u, v.GetIdx())]),
               _shannon(ctx[_side_context(v, u.GetIdx())]))


# ─────────────────────────── Gate 2: local-maxima cut selector ─────────────────────────
def _entropy_cache(mol, ctx):
    return {b.GetIdx(): _bond_entropy(b, ctx) for b in mol.GetBonds()}


def _neighbour_entropies(mol, bond_idx, ecache):
    """Entropies of every bond sharing an atom with bond_idx (ring bonds included as low references)."""
    b = mol.GetBondWithIdx(bond_idx)
    out = []
    for at in (b.GetBeginAtom(), b.GetEndAtom()):
        for nb in at.GetBonds():
            if nb.GetIdx() != bond_idx:
                out.append(ecache[nb.GetIdx()])
    return out


def _is_terminal_heteroatom_bond(bond):
    """True if one endpoint is a degree-1 (single heavy neighbour) non-carbon atom — Gate 4."""
    for at in (bond.GetBeginAtom(), bond.GetEndAtom()):
        if at.GetDegree() == 1 and at.GetSymbol() != 'C':
            return True
    return False


# ───────────── Gate 0+ : freeze functional groups (measured causal units) ──────────────
# Functional groups are detected by ERTL-IFG (algorithmic, complete over heteroatom functionality),
# NOT a hand-SMARTS list. A size CAP drops over-merged Ertl units (contiguous-heteroatom blobs) so a
# frozen unit never spans several groups: Ertl gives coverage, the cap gives the "valid single group"
# property RDKit's atomic FG list couldn't. Bonds INTERNAL to a kept group are frozen (never cut), the
# same way ring bonds are. With freeze_fg on, Gate 0 = "CRUSH-valid bond that is not ring / fused /
# FG-internal", and the LM (+BDE) gates decide only those remaining linker/scaffold bonds.
_FG_CAP = 8   # Ertl units larger than this are treated as over-merged and NOT frozen (measured: kills
              #  the 26-atom merge tail while keeping over-merge ~0%; the entropy gate then cuts them).


def _fg_internal_bonds(mol):
    """Bond indices internal to an Ertl functional group of size <= _FG_CAP — frozen."""
    frozen = set()
    for grp in ertl_frag.ertl_fgs(mol):
        if len(grp) > _FG_CAP:                       # cap: skip over-merged Ertl blobs
            continue
        g = set(grp)
        for b in mol.GetBonds():
            if b.GetBeginAtomIdx() in g and b.GetEndAtomIdx() in g:
                frozen.add(b.GetIdx())
    return frozen


def _candidate_bonds_frozen(mol):
    """Gate 0 with FG-freezing: CRUSH-valid bonds MINUS ring / fused / FG-internal. CRUSH still
    constrains cuts to chemically-valid disconnections; rings + Ertl FGs are frozen; LM/BDE decide
    the rest."""
    prot = _fused_protected(mol)
    fg = _fg_internal_bonds(mol)
    cands = set()
    for bi in _meaningful_bonds(mol):                            # CRUSH-valid disconnections only
        if bi in fg:                                             # minus frozen functional groups
            continue
        b = mol.GetBondWithIdx(bi)
        if b.GetBeginAtomIdx() in prot or b.GetEndAtomIdx() in prot:  # minus fused systems
            continue
        cands.add(bi)                                            # rings already excluded by CRUSH
    return cands


def _select_cuts(mol, ctx, use_bde, gate4, freeze_fg=False):
    """Return the set of candidate-bond indices to cut under the chosen gates."""
    ecache = _entropy_cache(mol, ctx)
    candidates = _candidate_bonds_frozen(mol) if freeze_fg else _meaningful_bonds(mol)  # Gate 0
    cuts = set()
    for bi in candidates:
        neigh = _neighbour_entropies(mol, bi, ecache)
        if not (neigh and ecache[bi] > max(neigh)):         # Gate 2: strict local maximum
            continue
        if not use_bde:                                     # method 'lm_strict'
            cuts.add(bi)
            continue
        bond = mol.GetBondWithIdx(bi)
        strength = _bde(bond)                               # Gate 3: physical veto
        if strength is not None and strength < SKELETON_LINE:
            cuts.add(bi)
        elif gate4 and _is_terminal_heteroatom_bond(bond):  # Gate 4: terminal-heteroatom exception
            cuts.add(bi)
    return cuts


_METHODS = {
    'lm_strict':  dict(use_bde=False, gate4=False),
    'lm_bde':     dict(use_bde=True,  gate4=False),
    'lm_bde_g4':  dict(use_bde=True,  gate4=True),
    # rings + functional groups frozen at Gate 0; LM/BDE apply only to linker/scaffold bonds:
    'lm_fg':       dict(use_bde=False, gate4=False, freeze_fg=True),
    'lm_bde_fg':   dict(use_bde=True,  gate4=False, freeze_fg=True),
    'lm_bde_g4_fg': dict(use_bde=True, gate4=True,  freeze_fg=True),
}


def fragment_mol(mol, ctx, method='lm_bde'):
    """Fragment one molecule into a PARTITION of atom-sets (complete, non-overlapping; rings whole)."""
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(_METHODS)}")
    cuts = _select_cuts(mol, ctx, **_METHODS[method])
    return _components(mol, cuts)


# ───────────────────────────────────── entry point ────────────────────────────────────
def build(smiles_all, method='lm_bde', head_source='none', linker_method='mdl',
          break_fused_rings=False, verbose=True):
    """Per-molecule ``[(key, set(atoms))]`` aligned to smiles_all (empty for unparseable SMILES).

    ``method`` selects the gate combination ('lm_strict' | 'lm_bde' | 'lm_bde_g4'). The other kwargs
    are accepted for call-compatibility with crush_direct.build / ring_mdl.build and are IGNORED
    (this fragmenter is deterministic given the corpus; no head-freeze / linker selection)."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_all]
    ctx = build_corpus_stats(mols)                          # Gate 1: one corpus pass
    out, n_ok = [], 0
    for m in mols:
        if m is None:
            out.append([])
            continue
        n_ok += 1
        units = fragment_mol(m, ctx, method)
        out.append([(cf.frag_key(m, frozenset(u)), set(u)) for u in units])
    if verbose:
        print(f"    [entropy_bde_frag] method={method} threshold-free local-maxima"
              f"{' + BDE veto' if _METHODS[method]['use_bde'] else ''}"
              f"{' + G4' if _METHODS[method]['gate4'] else ''} on {n_ok}/{len(smiles_all)} mols",
              flush=True)
    return out


# ───────────────────────── local validation harness (python entropy_bde_frag.py ...) ──
def _validate(csv_path, method='lm_bde'):
    """Fragment a FOLDS CSV and print structural health stats — for local review before pushing."""
    import csv
    import statistics as st
    smiles = [r.get('smiles') or r.get('SMILES') for r in csv.DictReader(open(csv_path))]
    per_mol = build(smiles, method=method, verbose=False)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    n_atom = n_cov = n_frag = n_sing = 0
    sizes, vocab, sing_elem = [], set(), Counter()
    ring_ok = ring_tot = 0
    for m, frags in zip(mols, per_mol):
        if m is None or not frags:
            continue
        a2u = {a: i for i, (_, atoms) in enumerate(frags) for a in atoms}
        for ring in m.GetRingInfo().AtomRings():
            ring_tot += 1
            if len({a2u[a] for a in ring}) == 1:
                ring_ok += 1
        for key, atoms in frags:
            sz = len(atoms); sizes.append(sz); vocab.add(key); n_frag += 1; n_atom += sz
            if sz >= 2:
                n_cov += sz
            else:
                n_sing += 1
                sing_elem[m.GetAtomWithIdx(next(iter(atoms))).GetSymbol()] += 1
    print(f"{csv_path.split('/')[-1]}  method={method}")
    print(f"  coverage={n_cov / max(1, n_atom):.0%}  singleton_frags={n_sing / max(1, n_frag):.0%}"
          f"  vocab={len(vocab)}  median_size={sorted(sizes)[len(sizes) // 2] if sizes else 0}"
          f"  mean_size={st.mean(sizes):.1f}")
    print(f"  rings_intact={ring_ok}/{ring_tot}  singleton_elements={dict(sing_elem.most_common(8))}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("usage: python entropy_bde_frag.py <FOLDS_csv> [lm_strict|lm_bde|lm_bde_g4]")
        sys.exit(1)
    _validate(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'lm_bde')
