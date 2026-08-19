#!/usr/bin/env python3
"""
exp1_candidate_inventory.py -- EXPERIMENT 1: candidate inventory + corpus statistics.

Standalone. Reads fold CSVs, writes JSON/CSV statistics. Touches nothing in the
pipeline; no model, no training, no vocabulary artefacts.

PIPELINE (Experiment 1 covers stages 1-4 only; no partition is selected here):
  1  four off-the-shelf fragmenters propose acyclic cut bonds, atom-indexed
  2  UNION of all cut sets -> common finest segmentation (blocks), iterated to fixpoint
  3  candidates = every connected union of blocks (capped)
  4  corpus statistics over candidates, computed ONCE, before any selection
  5  (NOT HERE) exact-cover ILP

SETTLED DESIGN DECISIONS encoded below, with the reason each is here:
  * RECAP's urea/amine/ether rules delete their central atom and therefore cannot be
    compiled to a bond cut by anyone (RDKit included).  They are re-expressed as
    proposing BOTH flanking bonds, so the central atom becomes its own block and the
    optimizer later decides whether it joins left, joins right, or stands alone.
  * Ring systems keep a substituent-AGNOSTIC key ('ring:c1ccccc1').  Everything else
    gets the L2 key (isomeric SMILES with [*] dummies + external boundary context).
  * S_methods is NOT in W(F).  All three variants are computed and reported as
    diagnostics only -- measured degenerate on this candidate pool.
  * W(F) = alpha*S_structure + beta*S_support  (+ eta*S_stability - delta*S_degeneracy,
    both computed and reported, both weighted 0 by default until Experiment 2).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from collections import Counter, defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 -- configuration
# ─────────────────────────────────────────────────────────────────────────────
# MotifBreakdown holds the two vendored rule tables we need (CRUSH SMIRKS, rBRICS).
# Only rule TABLES are imported -- never recur_frag / boundary_frag, so nothing in
# this script can perturb, or be perturbed by, the frozen bf_recur arm.
DEFAULT_MB = os.path.join(os.path.dirname(os.path.abspath(__file__)))

FRAGMENTERS = ('BRICS', 'RECAP', 'rBRICS', 'CRUSH')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 -- SMIRKS -> cut-bond compilation, and the three repaired RECAP rules
# ─────────────────────────────────────────────────────────────────────────────
def _split_top(s):
    """Split a SMIRKS product side on top-level '.' (ignoring bracket nesting)."""
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch in '[(':
            depth += 1
        elif ch in '])':
            depth -= 1
        if ch == '.' and depth == 0:
            out.append(cur); cur = ''
        else:
            cur += ch
    out.append(cur)
    return out


def _compile(smirks):
    """Compile one SMIRKS into (reactant_query, mapnum->idx, [(mapA, mapB), ...]).

    A bond is a CUT when its two mapped endpoints land in DIFFERENT product groups.
    Returns None when no such pair exists -- which is exactly what happens to RECAP's
    urea / amine / ether rules, whose central atom is unmapped and therefore appears
    in NO product group.  Those three are handled explicitly in REPAIRED_RECAP below.
    Fails closed: an unparseable product means we emit nothing rather than guess.
    """
    from rdkit import Chem
    react, prod = smirks.split('>>')
    rmol = Chem.MolFromSmarts(react)
    if rmol is None:
        return None
    m2i = {a.GetAtomMapNum(): a.GetIdx() for a in rmol.GetAtoms() if a.GetAtomMapNum()}
    groups, bad = [], False
    for p in _split_top(prod):
        pm = Chem.MolFromSmarts(p)
        if pm is None:
            bad = True
            continue
        groups.append({a.GetAtomMapNum() for a in pm.GetAtoms() if a.GetAtomMapNum()})
    if bad:
        return None
    pairs = []
    for b in rmol.GetBonds():
        ma, mb = b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()
        if not ma or not mb:
            continue
        ga = next((i for i, g in enumerate(groups) if ma in g), -1)
        gb = next((i for i, g in enumerate(groups) if mb in g), -1)
        if ga != gb and ga >= 0 and gb >= 0:
            pairs.append((ma, mb))
    return (rmol, m2i, pairs) if pairs else None


# The three RECAP rules that cannot be a bond cut, re-expressed as PAIRS of flanking
# bonds.  Reactant constraints are copied verbatim from rdkit.Chem.Recap.reactionDefs;
# only the mapping is changed so the central atom is addressable.
#   (SMARTS, [(match_idx_a, match_idx_b), ...])  -- indices into the substructure match
REPAIRED_RECAP_DEFS = (
    ('ether', '[#6:1]-!@[O;+0:2]-!@[#6:3]', [(0, 1), (1, 2)]),
    ('amine', '[*:1]-!@[N;!D1;+0;!$(N-C=[#7,#8,#15,#16]):2]-!@[*:3]', [(0, 1), (1, 2)]),
    ('urea',  '[#7;+0;D2,D3:1]!@[C:2](!@=[O:4])!@[#7;+0;D2,D3:3]', [(0, 1), (1, 2)]),
)

_RULES = {}          # lazily built, per process (multiprocessing-safe)


def _ensure_rules(motifbreakdown_dir):
    """Build every compiled rule set once per process.

    Lazy + global because multiprocessing on macOS uses 'spawn': compiled RDKit query
    objects do not survive pickling, so each worker rebuilds them on first use.
    """
    global _RULES
    if _RULES:
        return _RULES
    if motifbreakdown_dir not in sys.path:
        sys.path.insert(0, motifbreakdown_dir)
    from rdkit import Chem
    from rdkit.Chem import Recap
    from crush_smirks_rules import CRUSH_smirks          # rule TABLE only

    recap_ok = [c for c in (_compile(d) for d in Recap.reactionDefs) if c]
    crush_ok = [c for c in (_compile(sk) for _, sk in CRUSH_smirks) if c]
    repaired = [(nm, Chem.MolFromSmarts(sma), pairs) for nm, sma, pairs in REPAIRED_RECAP_DEFS]

    # Fail loudly on a silently empty rule set -- a stubbed scheme would corrupt every
    # downstream count without raising anything.
    if not recap_ok or not crush_ok or any(q is None for _, q, _ in repaired):
        raise RuntimeError('rule compilation produced an empty/invalid set')
    _RULES = {'recap': recap_ok, 'crush': crush_ok, 'repaired_recap': repaired,
              'n_recap_src': len(Recap.reactionDefs), 'n_recap_ok': len(recap_ok)}
    return _RULES


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 -- the four fragmenter adapters: mol -> set of acyclic cut-bond indices
# ─────────────────────────────────────────────────────────────────────────────
# Every adapter returns bond INDICES on the molecule it was handed.  Ring bonds are
# always excluded, which is what makes "ring systems are never split" a structural
# guarantee rather than a rule we have to enforce later.
def _smirks_cuts(mol, compiled):
    out = set()
    for rmol, m2i, pairs in compiled:
        for match in mol.GetSubstructMatches(rmol, uniquify=False, maxMatches=5000):
            for ma, mb in pairs:
                if ma in m2i and mb in m2i:
                    bd = mol.GetBondBetweenAtoms(match[m2i[ma]], match[m2i[mb]])
                    if bd is not None and not bd.IsInRing():
                        out.add(bd.GetIdx())
    return out


def _repaired_recap_cuts(mol, repaired):
    """The ether/amine/urea rules, proposing BOTH flanking bonds (see SECTION 1)."""
    out = set()
    for _nm, query, pairs in repaired:
        for match in mol.GetSubstructMatches(query, uniquify=False, maxMatches=5000):
            for i, j in pairs:
                if i < len(match) and j < len(match):
                    bd = mol.GetBondBetweenAtoms(match[i], match[j])
                    if bd is not None and not bd.IsInRing():
                        out.add(bd.GetIdx())
    return out


def cut_sets(mol, rules):
    """All four fragmenters on ONE mol object -> {name: set(bond_idx)}."""
    from rdkit.Chem import BRICS
    import rBRICS_public

    brics = set()
    for (a, b), _ in BRICS.FindBRICSBonds(mol):
        bd = mol.GetBondBetweenAtoms(a, b)
        if bd is not None and not bd.IsInRing():
            brics.add(bd.GetIdx())

    rbr = set()
    try:
        for pair in rBRICS_public.FindrBRICSBonds(mol):
            a, b = pair[0]
            bd = mol.GetBondBetweenAtoms(a, b)
            if bd is not None and not bd.IsInRing():
                rbr.add(bd.GetIdx())
    except Exception:
        pass                       # rBRICS raises on exotic valences; treat as no proposal

    recap = _smirks_cuts(mol, rules['recap']) | _repaired_recap_cuts(mol, rules['repaired_recap'])
    crush = _smirks_cuts(mol, rules['crush'])
    return {'BRICS': brics, 'RECAP': recap, 'rBRICS': rbr, 'CRUSH': crush}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 -- union refinement, iterated to a fixpoint
# ─────────────────────────────────────────────────────────────────────────────
def _blocks(mol, cuts):
    """Connected components after removing `cuts` -- a true partition of the atoms."""
    n = mol.GetNumAtoms()
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for b in mol.GetBonds():
        if b.GetIdx() not in cuts:
            ra, rb = find(b.GetBeginAtomIdx()), find(b.GetEndAtomIdx())
            if ra != rb:
                parent[ra] = rb
    groups = defaultdict(set)
    for a in range(n):
        groups[find(a)].add(a)
    return sorted(groups.values(), key=lambda s: min(s))


def segment_to_fixpoint(mol, rules, max_rounds=6):
    """UNION of all four cut sets, then re-pass each block until nothing new appears.

    Re-passing matters because some rules are NON-MONOTONE: a rule blocked by context
    on the whole molecule can fire once that context is cut away.  Measured on BBBP,
    only CRUSH is non-monotone, on ~1% of molecules, always converging in one round --
    but the loop is what turns "the cut set is probably closed" into a guarantee.
    """
    from rdkit import Chem
    base = cut_sets(mol, rules)
    cuts = set().union(*base.values())
    n_atoms = mol.GetNumAtoms()
    rounds = 0
    for _ in range(max_rounds):
        found = set()
        for blk in _blocks(mol, cuts):
            if len(blk) < 3:                        # nothing to find inside 1-2 atoms
                continue
            bd = [b.GetIdx() for b in mol.GetBonds()
                  if (b.GetBeginAtomIdx() in blk) != (b.GetEndAtomIdx() in blk)]
            if not bd:
                continue                            # the block IS the molecule; already passed
            frag = Chem.FragmentOnBonds(mol, bd, addDummies=True)
            mapping = []
            try:
                pieces = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=True,
                                          fragsMolAtomMapping=mapping)
            except Exception:
                continue
            for fm, mp in zip(pieces, mapping):
                back = {i: mp[i] for i in range(len(mp)) if mp[i] < n_atoms}
                if set(back.values()) != set(blk):
                    continue
                try:
                    sub = cut_sets(fm, rules)
                except Exception:
                    continue
                for bidx in set().union(*sub.values()):
                    fb = fm.GetBondWithIdx(bidx)
                    u, v = fb.GetBeginAtomIdx(), fb.GetEndAtomIdx()
                    if u in back and v in back:     # both ends are real atoms, not dummies
                        ob = mol.GetBondBetweenAtoms(back[u], back[v])
                        if ob is not None and not ob.IsInRing() and ob.GetIdx() not in cuts:
                            found.add(ob.GetIdx())
        if not found:
            break
        cuts |= found
        rounds += 1
    return cuts, base, rounds


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 -- candidate enumeration: every connected union of blocks
# ─────────────────────────────────────────────────────────────────────────────
def connected_unions(adj, sizes, max_blocks, max_heavy, max_cands):
    """BFS over connected block-subsets, deduplicated by frozenset.

    This reproduces RDKit's own `BRICSDecompose(keepNonLeafNodes=True)` semantics
    (verified set-equal on test molecules) but atom-indexed, which is what makes
    overlap, coverage and reconstruction testable.

    EVERY SINGLE BLOCK is emitted unconditionally, ignoring max_heavy -- the all-blocks
    partition is the guaranteed-feasible fallback for the exact cover, so dropping a
    block (e.g. an oversized fused ring system) would make some molecule INFEASIBLE.
    Caps apply only to EXTENSIONS.

    Returns (candidates, truncated, cap_hits).  cap_hits counts EACH cap separately,
    so the report can say WHICH cap bound and how hard -- a cap that silently removes
    good candidates would make the inventory wrong while looking complete.
    """
    hits = {'max_blocks': 0, 'max_heavy': 0, 'max_cands': 0}
    out, seen, frontier = [], set(), []
    for i in range(len(sizes)):
        fs = frozenset((i,))
        seen.add(fs); out.append(fs); frontier.append(fs)
    n_single = len(out)                      # layer 0: one candidate per block
    while frontier:
        nxt = []
        for s_ in frontier:
            if len(s_) >= max_blocks:
                hits['max_blocks'] += 1      # this subset could have grown, but may not
                continue
            heavy = sum(sizes[i] for i in s_)
            nbrs = set()
            for i in s_:
                nbrs |= adj[i]
            for j in nbrs - s_:
                if heavy + sizes[j] > max_heavy:
                    hits['max_heavy'] += 1
                    continue
                t = frozenset(s_ | {j})
                if t in seen:
                    continue
                if len(out) >= max_cands:
                    hits['max_cands'] += 1
                    return out, True, hits, n_single
                seen.add(t); out.append(t); nxt.append(t)
        frontier = nxt
    return out, False, hits, n_single


# SECTION 5 -- motif keying
# ─────────────────────────────────────────────────────────────────────────────
# THREE levels are computed so vocabulary shattering is MEASURED, not assumed:
#   L0  legacy non-isomeric dummy SMILES        (comparability with every prior result)
#   L1  isomeric dummy SMILES                   (stereo, charge, boundary bond order)
#   L2  L1 + external boundary context          (the production key for non-rings)
# Rings are the ONE exception: a whole ring system keeps a substituent-agnostic
# 'ring:' key, because the optimizer can already express "this ring's context matters"
# by selecting a LARGER motif that includes the context -- whereas putting context in
# the ring's NAME shatters its corpus counts (benzene: 1 motif at support 2677 becomes
# 255 motifs, largest 428) and frequency is a PRIMARY term in W(F).
def _frag_smiles(mol, atoms, isomeric):
    """Canonical SMILES of `atoms` with [*] dummies at every boundary bond.

    Dummy isotopes MUST be zeroed: FragmentOnBonds stamps each dummy with the index of
    the atom it replaced, which would make every key unique to its molecule.
    """
    from rdkit import Chem
    aset = set(atoms)
    n = mol.GetNumAtoms()
    cross = [b.GetIdx() for b in mol.GetBonds()
             if (b.GetBeginAtomIdx() in aset) != (b.GetEndAtomIdx() in aset)]
    if not cross:
        return Chem.MolToSmiles(mol, isomericSmiles=isomeric)
    frag = Chem.FragmentOnBonds(mol, cross, addDummies=True)
    for pa, pm in zip(Chem.GetMolFrags(frag, asMols=False),
                      Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)):
        if {a for a in pa if a < n} == aset:
            try:
                Chem.SanitizeMol(pm)
            except Exception:
                pass
            for a in pm.GetAtoms():
                if a.GetAtomicNum() == 0:
                    a.SetIsotope(0); a.SetAtomMapNum(0)
            try:
                return Chem.MolToSmiles(pm, isomericSmiles=isomeric)
            except Exception:
                return None
    return None


def _boundary_ctx(mol, atoms):
    """Sorted descriptor of what sits on the OUTSIDE of each boundary bond.

    This is the part that makes two superficially identical '*CC' fragments distinct
    when their surroundings differ (spec section 6).  Sorted, so it is invariant.
    """
    aset = set(atoms)
    desc = []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if (u in aset) != (v in aset):
            out = v if u in aset else u
            a = mol.GetAtomWithIdx(out)
            desc.append((str(b.GetBondType()), a.GetSymbol(), int(a.GetIsAromatic()),
                         int(a.IsInRing()), a.GetFormalCharge()))
    return ';'.join(str(d) for d in sorted(desc))


def _is_whole_ring_system(mol, atoms):
    """True when the candidate is ring atoms only AND every atom is still in a ring
    once extracted.  Judged on the EXTRACTED piece, not the parent: a fragment can be
    a perfectly good ring system even if the parent had it fused to something else."""
    from rdkit import Chem
    if not all(mol.GetAtomWithIdx(a).IsInRing() for a in atoms):
        return False
    sub = Chem.MolFromSmiles(_ring_smiles(mol, atoms))
    return sub is not None and sub.GetNumAtoms() > 0 and all(a.IsInRing() for a in sub.GetAtoms())


def _ring_smiles(mol, atoms):
    from rdkit import Chem
    km = Chem.Mol(mol)
    Chem.RemoveStereochemistry(km)
    return Chem.MolFragmentToSmiles(km, atomsToUse=sorted(atoms), canonical=True)


def keys_for(mol, atoms):
    """Return (L0, L1, L2_production, L2_allctx, is_ring).

    L2_production  -- rings agnostic, everything else L2      <- what the method uses
    L2_allctx      -- L2 for EVERYTHING including rings       <- the ablation arm
    """
    is_ring = _is_whole_ring_system(mol, atoms)
    l0 = _frag_smiles(mol, atoms, isomeric=False)
    l1 = _frag_smiles(mol, atoms, isomeric=True)
    ctx = _boundary_ctx(mol, atoms)
    l2 = (l1 + '|ctx=' + ctx) if l1 else None
    if is_ring:
        ring_key = 'ring:' + _ring_smiles(mol, atoms)
        return l0, l1, ring_key, (l2 or ring_key), True
    return l0, l1, l2, l2, False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 -- per-candidate structural score and the three S_methods diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def s_structure(n_atoms, n_attach):
    """S_structure = (1 - 1/n) * n/(n + |dF|).  ZERO free parameters, by design.

    Its ONE job is to counteract the size-frequency bias: small fragments have inflated
    corpus support, so frequency alone would prefer a lone '*O*' over benzene.
      (1 - 1/n)      annihilates single atoms  -> enforces spec section 1.8 by SCORING
      n/(n + |dF|)   penalises attachment points RELATIVE to size -- 6 attachments on
                     20 atoms is a motif, 2 attachments on 1 atom is a hub
    Ring preservation is deliberately NOT a term: no block splits a ring, so it is
    already guaranteed and scoring it would pay for something free.
    KNOWN LIMITATION, stated rather than papered over: this is MONOTONE in size, so the
    whole molecule scores highest.  It does NOT prevent the degenerate whole-molecule
    cover; only S_support does.  Hence the root guard in the caller.
    """
    if n_atoms <= 0:
        return 0.0
    return (1.0 - 1.0 / n_atoms) * (n_atoms / (n_atoms + n_attach))


def methods_diagnostics(bd_bonds, cuts_by_method):
    """The three variants of 'how much chemistry endorses this fragment'.

    NONE of these enters W(F) -- all three measured degenerate on this pool.  They are
    reported so Experiment 2 can re-examine the decision on real numbers.
      frag  spec section 9 as written: subset test per method.  5 values; scores the
            undivided molecule at 1.0 vacuously; correlates -0.55 with attachment count
      bnd   mean per-bond vote share.  Well spread, BUT sums over an exact cover to a
            separable per-bond total -- i.e. it IS bond voting, which collapses the
            joint optimizer back into bf_recur's min_votes gate
      best  max over methods of boundary coverage.  Unbiased but 75% saturate at 1.0
    """
    m = len(FRAGMENTERS)
    if not bd_bonds:
        return 0.0, 0.0, 0.0                      # root: vacuous 1.0 deliberately refused
    frag = sum(1 for X in FRAGMENTERS if bd_bonds <= cuts_by_method[X]) / m
    bnd = sum(sum(1 for X in FRAGMENTERS if b in cuts_by_method[X]) / m
              for b in bd_bonds) / len(bd_bonds)
    best = max(len(bd_bonds & cuts_by_method[X]) / len(bd_bonds) for X in FRAGMENTERS)
    return frag, bnd, best


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 -- per-molecule worker
# ─────────────────────────────────────────────────────────────────────────────
def process_molecule(args):
    """One molecule -> everything the corpus aggregation needs.

    Returns per-key data ONCE per molecule (sets, not lists), because Support(F) is
    defined as DISTINCT-MOLECULE count: a molecule with five benzenes must not cast
    five votes for benzene.
    """
    smi, mb_dir, max_blocks, max_heavy, max_cands, do_fixpoint = args
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    rules = _ensure_rules(mb_dir)

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {'ok': False, 'smiles': smi}

    if do_fixpoint:
        cuts, per_method, rounds = segment_to_fixpoint(mol, rules)
    else:
        per_method = cut_sets(mol, rules)
        cuts, rounds = set().union(*per_method.values()), 0
    n_new = len(cuts) - len(set().union(*per_method.values()))

    blocks = _blocks(mol, cuts)
    # VALIDATION: the blocks must be an exact partition of the atoms.
    cover = set().union(*blocks) if blocks else set()
    assert cover == set(range(mol.GetNumAtoms())), 'blocks do not cover all atoms'
    assert sum(len(b) for b in blocks) == mol.GetNumAtoms(), 'blocks overlap'

    idx = {a: i for i, b in enumerate(blocks) for a in b}
    adj = defaultdict(set)
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if idx[u] != idx[v]:
            adj[idx[u]].add(idx[v]); adj[idx[v]].add(idx[u])
    sizes = [len(b) for b in blocks]

    unions, truncated, cap_hits, n_single_cands = connected_unions(
        adj, sizes, max_blocks, max_heavy, max_cands)
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]

    seen0, seen1, seen2, seen2all = set(), set(), set(), set()
    stats = {}                       # key_L2_production -> structural/method record
    multiplicity = Counter()         # key -> disjoint occurrences WITHIN this molecule
    cand_list = []                   # (block-index tuple, production key) for the SOLVER
    n_root_skipped = 0
    n_single_kept = 0
    n_unkeyable = 0
    for u in unions:
        atoms = set()
        for i in u:
            atoms |= blocks[i]
        # ROOT GUARD: a "partition into one part" is not a fragmentation.  Excluded
        # unless the molecule genuinely has a single block (then it is the only option).
        if len(atoms) == mol.GetNumAtoms() and len(blocks) > 1:
            n_root_skipped += 1
            continue
        # VALIDATION: no candidate may split a ring.
        for r in rings:
            assert r <= atoms or not (r & atoms), 'candidate splits a ring'
        bd = {b.GetIdx() for b in mol.GetBonds()
              if (b.GetBeginAtomIdx() in atoms) != (b.GetEndAtomIdx() in atoms)}
        k0, k1, k2, k2all, is_ring = keys_for(mol, atoms)
        if k2 is None:
            # FEASIBILITY GUARD: dropping a SINGLE BLOCK would leave its atoms
            # coverable by nothing at all -> infeasible ILP, not a worse answer.
            # Keep it under a synthetic key (it will score near 0 and lose on merit).
            if len(u) != 1:
                continue
            k2 = k2all = 'UNKEYABLE:%d' % len(atoms)
            k0 = k1 = k2
            n_unkeyable += 1
        if k0: seen0.add(k0)
        if k1: seen1.add(k1)
        seen2.add(k2); seen2all.add(k2all)
        multiplicity[k2] += 1
        cand_list.append((tuple(sorted(u)), k2))
        if len(u) == 1:
            n_single_kept += 1
        if k2 not in stats:
            mf, mb, mbest = methods_diagnostics(bd, per_method)
            stats[k2] = {'n_atoms': len(atoms), 'n_attach': len(bd), 'is_ring': int(is_ring),
                         'm_frag': mf, 'm_bnd': mb, 'm_best': mbest,
                         'ctx': _boundary_ctx(mol, atoms),
                         'l1': k1 or ''}
    pj = {}
    for i, a in enumerate(FRAGMENTERS):
        for b in FRAGMENTERS[i + 1:]:
            pj[f'{a}|{b}'] = (len(per_method[a] & per_method[b]),
                              len(per_method[a] | per_method[b]))
    return {'ok': True, 'smiles': smi, 'n_atoms': mol.GetNumAtoms(),
            'n_blocks': len(blocks), 'n_cands': len(stats), 'truncated': truncated,
            'cap_hits': cap_hits,
            'n_single_block_cands': n_single_cands,      # layer-0 candidates offered
            'n_single_block_kept': n_single_kept,        # ... that survived the guards
            'n_unkeyable_blocks': n_unkeyable,
            'rounds': rounds, 'n_new_cuts': n_new, 'n_root_skipped': n_root_skipped,
            'cuts_per_method': {k: len(v) for k, v in per_method.items()},
            'pair_jaccard': pj, 'block_sizes': sizes,
            'blocks': [tuple(sorted(b)) for b in blocks],   # for the SOLVER (pass 2)
            'cand_list': cand_list,                          # for the SOLVER (pass 2)
            'keys0': seen0, 'keys1': seen1, 'keys2': seen2, 'keys2all': seen2all,
            'stats': stats, 'multiplicity': dict(multiplicity)}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 -- corpus aggregation and W(F)
# ─────────────────────────────────────────────────────────────────────────────
def aggregate(results, alpha, beta, eta, delta, w_form='multiplicative'):
    """Corpus statistics computed ONCE over candidates, BEFORE any partition exists.

    This is the anti-circularity requirement (spec section 2): support here counts
    CANDIDATE occurrences, never selected ones, so no statistic can be contaminated by
    a selection that has not happened yet.
    """
    sup0, sup1, sup2, sup2all = Counter(), Counter(), Counter(), Counter()
    rec = {}
    mult = defaultdict(list)
    ctx_var = defaultdict(Counter)
    for r in results:
        if not r['ok']:
            continue
        for k in r['keys0']: sup0[k] += 1
        for k in r['keys1']: sup1[k] += 1
        for k in r['keys2']: sup2[k] += 1
        for k in r['keys2all']: sup2all[k] += 1
        for k, s in r['stats'].items():
            rec.setdefault(k, s)
            ctx_var[s['l1']][s['ctx']] += 1
        for k, n in r['multiplicity'].items():
            mult[k].append(n)

    max_sup = max(sup2.values()) if sup2 else 1
    denom = math.log1p(max_sup) or 1.0
    rows = []
    for k, s in rec.items():
        support = sup2[k]
        s_sup = math.log1p(support) / denom
        s_str = s_structure(s['n_atoms'], s['n_attach'])
        # S_stability: how concentrated is this motif's boundary context across the
        # corpus?  1 = always wired the same way, 0 = wired every possible way.  This
        # is where boundary context we kept OUT of the ring key comes back as EVIDENCE.
        cv = ctx_var.get(s['l1'], Counter())
        tot = sum(cv.values())
        if tot > 1 and len(cv) > 1:
            ent = -sum((c / tot) * math.log(c / tot) for c in cv.values())
            s_stab = 1.0 - ent / math.log(len(cv))
        else:
            s_stab = 1.0
        # S_degeneracy: mean disjoint occurrences of this key WITHIN one molecule.
        # If benzene appears 3x in a molecule, "benzene mattered" does not say WHICH --
        # a poor explanatory unit.  Normalised to [0,1] by a soft 1 - 1/x.
        mm = mult.get(k, [1])
        mean_mult = sum(mm) / len(mm)
        s_deg = 1.0 - 1.0 / mean_mult if mean_mult > 0 else 0.0
        # ADDITIVE is spec section 10 as written.  MEASURED to be unable to suppress
        # trivial fragments: S_structure=0 zeroes only the alpha term, while
        # beta*S_support still pays a bare '*=O' the maximum 1.000 -- so 66% of
        # selected fragments came out as single atoms at lambda=0.2.
        # MULTIPLICATIVE treats structure as a GATE on whether corpus evidence counts
        # at all, which sends every single-atom candidate to exactly 0 with no new
        # constant.  alpha is unused in this form.
        evidence = beta * s_sup + eta * s_stab - delta * s_deg
        if w_form == 'multiplicative':
            W = s_str * evidence
        else:
            W = alpha * s_str + evidence
        rows.append({'key': k, 'support': support, 'n_atoms': s['n_atoms'],
                     'n_attach': s['n_attach'], 'is_ring': s['is_ring'],
                     's_structure': round(s_str, 6), 's_support': round(s_sup, 6),
                     's_stability': round(s_stab, 6), 's_degeneracy': round(s_deg, 6),
                     'm_frag': round(s['m_frag'], 6), 'm_bnd': round(s['m_bnd'], 6),
                     'm_best': round(s['m_best'], 6), 'W': round(W, 6)})
    return rows, {'L0': sup0, 'L1': sup1, 'L2': sup2, 'L2_allctx': sup2all}



# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8b -- JOINT PARTITION OPTIMISATION (the exact-cover ILP)
# ─────────────────────────────────────────────────────────────────────────────
# This runs as a SECOND PASS, after aggregate().  That ordering is not incidental:
# W(F) is a CORPUS statistic and must be finished before any molecule is solved.
# Solving during pass 1 would make each partition depend on statistics that were
# themselves derived from partitions -- the circularity spec section 2 forbids.
def solve_partition(blocks, cand_list, W_by_key, n_atoms, lam):
    """max  sum_j (W(F_j) - lambda) * x_j   s.t.  sum_{j: a in F_j} x_j == 1  for all a

    lb=1, ub=1 IS the whole requirement: every atom covered at least once (nothing
    lost) and at most once (no overlap).  lambda is folded into each coefficient as
    the flat per-fragment fee, so a fragment must "pay its way" to be selected.
    scipy's milp MINIMISES, hence the negation.
    """
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint
    if not cand_list:
        return None
    ncand = len(cand_list)
    A = np.zeros((n_atoms, ncand))
    w = np.empty(ncand)
    for j, (bidx, key) in enumerate(cand_list):
        for i in bidx:
            for a in blocks[i]:
                A[a, j] = 1.0
        w[j] = W_by_key.get(key, 0.0)
    res = milp(c=-(w - lam),
               constraints=LinearConstraint(A, lb=np.ones(n_atoms), ub=np.ones(n_atoms)),
               integrality=np.ones(ncand),
               bounds=Bounds(np.zeros(ncand), np.ones(ncand)))
    if not res.success or res.x is None:
        return None
    sel = [j for j, v in enumerate(res.x) if v > 0.5]
    return sel, float(-res.fun)


def verify_partition(n_atoms, blocks, cand_list, sel, rings):
    """Spec section 16, asserted on EVERY solved molecule -- not a separate test pass
    that can drift out of sync with the code it is meant to check."""
    parts = []
    for j in sel:
        atoms = set()
        for i in cand_list[j][0]:
            atoms |= set(blocks[i])
        parts.append(atoms)
    union = set().union(*parts) if parts else set()
    assert union == set(range(n_atoms)), 'partition does not cover every atom'
    assert sum(len(p) for p in parts) == n_atoms, 'partition fragments overlap'
    for r in rings:
        assert any(r <= p for p in parts), 'a ring was split across fragments'
    return parts


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 -- fidelity check against RDKit's own hierarchy
# ─────────────────────────────────────────────────────────────────────────────
def brics_fidelity(smiles, mb_dir, n_check):
    """Assert that 'connected unions of blocks' really is RDKit's own notion of a
    hierarchy node.  Runs on BRICS-ONLY blocks, where RDKit gives us a ground truth
    via BRICSDecompose(keepNonLeafNodes=True).  A drifting match rate here means the
    candidate space has stopped being 'what off-the-shelf chemistry proposed'."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import BRICS
    RDLogger.DisableLog('rdApp.*')
    _ensure_rules(mb_dir)

    def strip(s):
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        for a in m.GetAtoms():
            if a.GetAtomicNum() == 0:
                a.SetIsotope(0); a.SetAtomMapNum(0)
        return Chem.MolToSmiles(m)

    ok = bad = skipped = 0
    for smi in smiles[:n_check]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        cuts = set()
        for (a, b), _ in BRICS.FindBRICSBonds(mol):
            bd = mol.GetBondBetweenAtoms(a, b)
            if bd is not None and not bd.IsInRing():
                cuts.add(bd.GetIdx())
        blocks = _blocks(mol, cuts)
        if len(blocks) > 10:
            skipped += 1                 # RDKit's own call is exponential; keep it cheap
            continue
        idx = {a: i for i, b in enumerate(blocks) for a in b}
        adj = defaultdict(set)
        for b in mol.GetBonds():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if idx[u] != idx[v]:
                adj[idx[u]].add(idx[v]); adj[idx[v]].add(idx[u])
        unions, _, _, _ = connected_unions(adj, [len(b) for b in blocks], 99, 999, 100000)
        mine = set()
        for u in unions:
            atoms = set()
            for i in u:
                atoms |= blocks[i]
            # isomeric=True: BRICSDecompose PRESERVES stereochemistry, so comparing
            # against a stereo-stripped key reports false mismatches on every
            # stereo-dense molecule (steroids, beta-lactams).  Measured: 6/20 false
            # mismatches on BBBP before this was corrected.
            k = _frag_smiles(mol, atoms, isomeric=True)
            if k:
                mine.add(strip(k))
        rd = {strip(s) for s in BRICS.BRICSDecompose(mol, keepNonLeafNodes=True)}
        if mine - {None} == rd - {None}:
            ok += 1
        else:
            bad += 1
    return {'match': ok, 'mismatch': bad, 'skipped_large': skipped}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 -- driver, report, outputs
# ─────────────────────────────────────────────────────────────────────────────
def pct(x, n):
    return round(100.0 * x / n, 2) if n else 0.0


def run_corpus(name, path, args):
    import pandas as pd
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')

    df = pd.read_csv(path)
    smiles = df['smiles'].astype(str).tolist()
    if args.limit:
        smiles = smiles[:args.limit]
    print(f'\n══ {name}: {len(smiles)} rows from {path}', flush=True)

    work = [(s, args.motifbreakdown, args.max_blocks, args.max_heavy,
             args.max_cands, not args.no_fixpoint) for s in smiles]
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            results = pool.map(process_molecule, work, chunksize=16)
    else:
        results = [process_molecule(w) for w in work]

    import statistics as st
    okr = [r for r in results if r['ok']]
    rows, sups = aggregate(okr, args.alpha, args.beta, args.eta, args.delta,
                           w_form=args.w_form)

    n = len(okr)

    # ══ PASS 2: JOINT PARTITION OPTIMISATION ═════════════════════════════════
    # W(F) is now final for the whole corpus, so every molecule can be solved
    # against statistics that no selection influenced.
    W_by_key = {r['key']: r['W'] for r in rows}
    lam_grid = [float(x) for x in args.lam_sweep.split(',') if x.strip()] or [args.lam]
    if args.lam not in lam_grid:
        lam_grid.append(args.lam)
    sweep = {}
    partitions = None
    for lam in sorted(lam_grid):
        n_frag = []; n_single = []; n_fail = 0; obj = []; sizes_sel = []
        keep = [] if lam == args.lam else None
        for r in okr:
            from rdkit import Chem as _C
            mol = _C.MolFromSmiles(r['smiles'])
            rings = [set(x) for x in mol.GetRingInfo().AtomRings()]
            got = solve_partition(r['blocks'], r['cand_list'], W_by_key,
                                  r['n_atoms'], lam)
            if got is None:
                n_fail += 1
                continue
            sel, value = got
            parts = verify_partition(r['n_atoms'], r['blocks'], r['cand_list'], sel, rings)
            n_frag.append(len(sel))
            n_single.append(sum(1 for j in sel if len(r['cand_list'][j][0]) == 1))
            sizes_sel.extend(len(p) for p in parts)
            obj.append(value)
            if keep is not None:
                keep.append({'smiles': r['smiles'],
                             'fragments': [{'key': r['cand_list'][j][1],
                                            'atoms': sorted(set().union(
                                                *[set(r['blocks'][i]) for i in r['cand_list'][j][0]])),
                                            'n_blocks': len(r['cand_list'][j][0])} for j in sel],
                             'objective': round(value, 6)})
        sweep[f'{lam:g}'] = {
            'solved': len(n_frag), 'infeasible_or_failed': n_fail,
            'fragments_per_molecule_mean': round(st.mean(n_frag), 3) if n_frag else 0,
            'fragments_per_molecule_median': st.median(n_frag) if n_frag else 0,
            'single_block_fragments_pct': pct(sum(n_single), sum(n_frag)) if n_frag else 0,
            'heavy_atoms_per_fragment_mean': round(st.mean(sizes_sel), 3) if sizes_sel else 0,
            'trivial_fragments_pct': pct(sum(1 for x in sizes_sel if x <= 1), len(sizes_sel)) if sizes_sel else 0,
            'objective_mean': round(st.mean(obj), 4) if obj else 0}
        if keep is not None:
            partitions = keep
        print(f'    lambda={lam:<5g} solved={len(n_frag):5d} fail={n_fail:3d} '
              f'frags/mol={sweep[f"{lam:g}"]["fragments_per_molecule_mean"]:.2f} '
              f'single-block={sweep[f"{lam:g}"]["single_block_fragments_pct"]:.1f}% '
              f'trivial={sweep[f"{lam:g}"]["trivial_fragments_pct"]:.1f}%', flush=True)

    tot_pair = defaultdict(lambda: [0, 0])
    for r in okr:
        for k, (i, u) in r['pair_jaccard'].items():
            tot_pair[k][0] += i; tot_pair[k][1] += u
    jac = {k: round(i / u, 4) if u else None for k, (i, u) in tot_pair.items()}

    blocks_per = [r['n_blocks'] for r in okr]
    cands_per = [r['n_cands'] for r in okr]
    sup2 = sups['L2']
    supvals = list(sup2.values())

    summary = {
        'corpus': name, 'path': path,
        'n_rows': len(smiles), 'n_parsed': n, 'n_unparsed': len(smiles) - n,
        'config': {k: getattr(args, k) for k in
                   ('max_blocks', 'max_heavy', 'max_cands', 'no_fixpoint',
                    'w_form', 'alpha', 'beta', 'eta', 'delta')},
        'fixpoint': {
            'molecules_gaining_cuts': sum(1 for r in okr if r['n_new_cuts'] > 0),
            'pct_gaining': pct(sum(1 for r in okr if r['n_new_cuts'] > 0), n),
            'rounds_histogram': dict(Counter(r['rounds'] for r in okr)),
            'total_new_cuts': sum(r['n_new_cuts'] for r in okr)},
        'cuts_per_method': {m: sum(r['cuts_per_method'][m] for r in okr) for m in FRAGMENTERS},
        'pairwise_cut_jaccard': jac,
        'blocks_per_molecule': {'mean': round(st.mean(blocks_per), 2) if blocks_per else 0,
                                'median': st.median(blocks_per) if blocks_per else 0,
                                'max': max(blocks_per) if blocks_per else 0},
        'candidates_per_molecule': {'mean': round(st.mean(cands_per), 2) if cands_per else 0,
                                    'median': st.median(cands_per) if cands_per else 0,
                                    'max': max(cands_per) if cands_per else 0},
        'truncated_molecules': sum(1 for r in okr if r['truncated']),
        'cap_hits': {k: sum(r['cap_hits'][k] for r in okr)
                     for k in ('max_blocks', 'max_heavy', 'max_cands')},
        'molecules_hitting_cap': {k: sum(1 for r in okr if r['cap_hits'][k] > 0)
                                  for k in ('max_blocks', 'max_heavy', 'max_cands')},
        'single_blocks': {
            'total_blocks': sum(r['n_blocks'] for r in okr),
            'blocks_of_one_atom': sum(sum(1 for x in r['block_sizes'] if x == 1) for r in okr),
            'layer0_candidates_offered': sum(r['n_single_block_cands'] for r in okr),
            'layer0_candidates_kept': sum(r['n_single_block_kept'] for r in okr),
            'unkeyable_blocks_rescued': sum(r['n_unkeyable_blocks'] for r in okr)},
        'partition_optimisation': sweep,
        'lambda_used_for_output': args.lam,
        'root_candidates_skipped': sum(r['n_root_skipped'] for r in okr),
        'unique_keys': {k: len(v) for k, v in sups.items()},
        'shattering': {
            'L0_to_L2': round(len(sups['L2']) / max(len(sups['L0']), 1), 3),
            'ring_agnostic_vs_allctx': round(len(sups['L2_allctx']) / max(len(sups['L2']), 1), 3)},
        'support': {
            'support_1_pct': pct(sum(1 for v in supvals if v == 1), len(supvals)),
            'median': st.median(supvals) if supvals else 0,
            'max': max(supvals) if supvals else 0},
        'trivial_candidates_pct': pct(sum(1 for r in rows if r['n_atoms'] <= 1), len(rows)),
        'ring_candidates_pct': pct(sum(1 for r in rows if r['is_ring']), len(rows)),
        's_methods_diagnostics': {
            v: {'distinct_values': len({round(r[v], 6) for r in rows}),
                'mean': round(sum(r[v] for r in rows) / max(len(rows), 1), 4)}
            for v in ('m_frag', 'm_bnd', 'm_best')},
        'brics_fidelity_vs_rdkit': brics_fidelity(smiles, args.motifbreakdown, args.fidelity_n),
    }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f'exp1_{name}_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)
    cpath = os.path.join(args.out, f'exp1_{name}_candidates.csv.gz')
    cols = ['key', 'support', 'n_atoms', 'n_attach', 'is_ring', 's_structure',
            's_support', 's_stability', 's_degeneracy', 'm_frag', 'm_bnd', 'm_best', 'W']
    with gzip.open(cpath, 'wt') as fh:
        fh.write(','.join(cols) + '\n')
        for r in sorted(rows, key=lambda x: -x['W']):
            fh.write(','.join('"' + str(r[c]).replace('"', '""') + '"'
                              if c == 'key' else str(r[c]) for c in cols) + '\n')

    if partitions is not None:
        ppath = os.path.join(args.out, f'exp1_{name}_partitions.jsonl.gz')
        with gzip.open(ppath, 'wt') as fh:
            for rec in partitions:
                fh.write(json.dumps(rec) + '\n')
        print(f'  wrote {ppath}')

    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(f'\n  top 15 candidates by W:')
    for r in sorted(rows, key=lambda x: -x['W'])[:15]:
        print(f'    W={r["W"]:.3f} sup={r["support"]:6d} n={r["n_atoms"]:3d} '
              f'att={r["n_attach"]:2d} ring={r["is_ring"]}  {r["key"][:70]}')
    print(f'\n  wrote {cpath}')
    return summary


def main():
    ap = argparse.ArgumentParser(description='Experiment 1: candidate inventory')
    ap.add_argument('--data_root', required=True, help='directory holding the fold CSVs')
    ap.add_argument('--datasets', default='BBBP_0,Mutagenicity_0,Alkane_Carbonyl_Verified_GT_0')
    ap.add_argument('--out', required=True)
    ap.add_argument('--motifbreakdown', default=DEFAULT_MB,
                    help='dir containing crush_smirks_rules.py and rBRICS_public.py')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0, help='first N molecules (smoke test)')
    # caps -- every one of these is REPORTED in the summary, never silent
    ap.add_argument('--max_blocks', type=int, default=12)
    ap.add_argument('--max_heavy', type=int, default=24)
    ap.add_argument('--max_cands', type=int, default=20000)
    ap.add_argument('--no_fixpoint', action='store_true')
    ap.add_argument('--fidelity_n', type=int, default=200)
    # W(F) weights -- gamma is absent BY DESIGN; eta/delta default 0 until Experiment 2
    ap.add_argument('--w_form', choices=['multiplicative', 'additive'],
                    default='multiplicative',
                    help='multiplicative = structural GATE (default); '
                         'additive = spec section 10 as literally written')
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--eta', type=float, default=0.0)
    ap.add_argument('--delta', type=float, default=0.0)
    # lambda = the flat per-fragment fee in the objective.  UNMEASURED: the sweep
    # exists precisely because we have no evidence for a value yet.
    ap.add_argument('--lam', type=float, default=0.2,
                    help='per-fragment penalty used for the WRITTEN partitions')
    ap.add_argument('--lam_sweep', default='0.0,0.1,0.2,0.4,0.8',
                    help='lambda values to report fragmentation statistics for')
    args = ap.parse_args()

    allsum = {}
    for name in [d.strip() for d in args.datasets.split(',') if d.strip()]:
        path = os.path.join(args.data_root, f'{name}.csv')
        if not os.path.isfile(path):
            print(f'  !! missing {path}, skipping', flush=True)
            continue
        allsum[name] = run_corpus(name, path, args)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'exp1_all_summaries.json'), 'w') as fh:
        json.dump(allsum, fh, indent=2, default=str)
    print(f'\nwrote {os.path.join(args.out, "exp1_all_summaries.json")}')


if __name__ == '__main__':
    main()
