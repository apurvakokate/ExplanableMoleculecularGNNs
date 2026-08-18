"""recur_frag.py -- bf_recur fragmentation arm (frozen; self-contained).

╔══════════════════════════════════════════════════════════════════════════════╗
║  TO REPLICATE THE BEST (rBRICS-BEATING) PERFORMANCE: USE  --method bf_recur_peel  ║
║                                                                              ║
║  The PEEL POLICY is the lever. It is the single largest factor on every       ║
║  downstream metric measured so far. Two settings, same code path:             ║
║                                                                              ║
║    bf_recur_peel   AGGRESSIVE peel  peels=1257  vocab=1025  <- BEATS rBRICS   ║
║    bf_recur        CONSERVATIVE     peels=12    vocab=988   <- default        ║
║                                                                              ║
║  pearson_motif_all (evaluate.py headline metric), BBBP fold 0, MOSE, seed 0:  ║
║        backbone   rBRICS   bf_recur_peel   bf_recur                          ║
║        GIN         0.359      0.350         0.257                            ║
║        GCN         0.714      0.755         0.738                            ║
║        SAGE        0.543      0.642         0.540                            ║
║        GAT         0.670      0.724         0.557                            ║
║                                                                              ║
║  bf_recur_peel beats bf_recur on ALL FOUR backbones and beats rBRICS on       ║
║  GCN / SAGE / GAT. If you are trying to reproduce the good numbers and are    ║
║  getting worse ones, you are almost certainly on the conservative peel.       ║
║                                                                              ║
║  See the PEEL POLICY block below for the exact commands, the expected         ║
║  counts, and why the default is nevertheless the conservative setting.        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Self-contained: imports nothing from boundary_frag, so work on other methods
cannot perturb this one. Deterministic (verified: identical tilings across runs).
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter, defaultdict

import rdkit
from rdkit import Chem

from crush_smirks_rules import recap_smirks, brics_smirks, CRUSH_smirks

_EFGS_DIR = os.path.join(os.path.dirname(rdkit.__file__), 'Contrib', 'efgs')


def _get_fgs(mol):
    global _get_fgs
    if os.path.isdir(_EFGS_DIR) and _EFGS_DIR not in sys.path:
        sys.path.insert(0, _EFGS_DIR)
    from efgs import get_fgs as _real          # no fallback: a missing detector would
    _get_fgs = lambda m: [set(g) for g in _real(m)]   # silently disable the FG freeze
    return _get_fgs(mol)


# ══════════════════════════════════════════════════════════════════════════════
# PEEL POLICY -- READ BEFORE CHANGING ANYTHING IN THE RING TIER
#
# The ring tier peels a FUSED ring system whose whole-system key falls below the
# support floor, splitting it into a core plus a remnant. Two policies exist and the
# choice is NOT cosmetic: it is the single largest lever on every downstream metric
# measured so far, and the two options disagree about what "better" means.
#
#   bf_recur       peel_lookahead=True   CONSERVATIVE   peels=12    vocab=988
#   bf_recur_peel  peel_lookahead=False  AGGRESSIVE     peels=1257  vocab=1025
#
# MEASURED, BBBP fold 0, MOSE, --w_feat --w_readout, 100 epochs, seed 0,
# pearson_motif_all (= analysis/evaluate.py's headline `grouped_pearson_u`:
# grouped per motif, UNWEIGHTED, UNK excluded, pooled over all splits):
#
#     backbone     rBRICS   AGGRESSIVE   CONSERVATIVE
#     GIN           0.359      0.350         0.257
#     GCN           0.714      0.755         0.738
#     SAGE          0.543      0.642         0.540
#     GAT           0.670      0.724         0.557
#
# => AGGRESSIVE beats CONSERVATIVE on all four backbones, and beats rBRICS on
#    GCN / SAGE / GAT. CONSERVATIVE beats rBRICS on GCN only.
#
# TO REPRODUCE THE rBRICS-BEATING RESULT:
#     VOCAB_NO_RULES=1 python3 generate_vocab_rules.py --datasets BBBP --fold 0 \
#         --data_root ../sweep/FOLDS --out_dir <out> --method bf_recur_peel
#     python3 ../MOSE-GNN/run.py --dataset BBBP --fold 0 --backbone <BB> \
#         --data_root ../sweep/FOLDS --vocab_root <out> --vocab_variant bf_recur_peel \
#         --w_feat --w_readout --epochs 100 --patience 30 --seed 0
#   Expect vocab=1025, peels=1257, splits=433, 100% node coverage. Verified to
#   reproduce the original 1034-motif run to within 0.003 AUC and 0.022 correlation
#   on GCN/SAGE/GAT (GIN loosest: +0.017 AUC, -0.038 Pearson).
#
# WHY CONSERVATIVE IS NEVERTHELESS THE DEFAULT -- the trade is chemical, not numeric:
#   AGGRESSIVE shatters canonical fused systems whose whole-key is rare, including
#   indole (support 15), benzimidazole (11) and a steroid nucleus (19), into a common
#   ring plus a remnant. Corpus-wide it created 337 new keys of which only 7 cleared
#   the floor, destroyed 294 fused-ring keys, and moved the aggregate below-floor rate
#   by 0.1pp (12.6% -> 12.7%). It buys metric score by manufacturing rare motifs.
#   Independent evidence that this is a metric artifact: restricting the correlation
#   to estimable motifs REVERSES it -- at support>=20 a vocabulary WITHOUT the rare
#   tail scored 0.441 vs 0.090 for one with it. pearson_motif_all is unweighted, so a
#   motif seen twice counts as much as benzene, and rare motifs inflate it.
#
# A CAUTION FOR THE RING TIER GENERALLY: the peel loop triggers on
# `k.startswith('ring:')`, so the KEY SPELLING is acting as control flow. Removing a
# `ringfrag:` prefix for unrelated key-validity reasons once silently cut peels from
# 1257 to 12 with no error. Tier membership should be an explicit attribute, never
# parsed out of a key string. Do not add prefix-dependent logic here.
# ══════════════════════════════════════════════════════════════════════════════

R_MAX = 3
SUP_FRAC = 0.005
SUP_FLOOR = 20
FG_CAP = 8

# 'vote' = CONSENSUS gate: the independent fragmentation schemes ARE the evidence for
# where a cut is chemically real; the corpus data decides GRANULARITY in Stage 3, not
# which bonds are boundaries. 'npmi'/'ti' are retained as measured-and-refuted arms.
METHODS = {'bf_recur': 'recur',        # conservative peel (default)
           'bf_recur_peel': 'recur'}   # aggressive peel -- the rBRICS-beating arm
RECUR_MIN_OCC = 5     # an environment must recur this often before its rate is trusted


# ── SMIRKS -> cut-bond compilation (inlined from crush_direct) ────────────────
def _split_top(s):
    out, d, cur = [], 0, ''
    for ch in s:
        if ch in '[(':
            d += 1
        elif ch in '])':
            d -= 1
        if ch == '.' and d == 0:
            out.append(cur); cur = ''
        else:
            cur += ch
    out.append(cur)
    return out


def _compile(smirks):
    react, prod = smirks.split('>>')
    rmol = Chem.MolFromSmarts(react)
    if rmol is None:
        return None
    m2i = {a.GetAtomMapNum(): a.GetIdx() for a in rmol.GetAtoms() if a.GetAtomMapNum()}
    groups, bad = [], False
    for p in _split_top(prod):
        pm = Chem.MolFromSmarts(p)
        if pm is None:
            bad = True                       # fail closed: a product we cannot parse
            continue                          # must not manufacture a cut-pair
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


# ── Stage 1: SEVERAL INDEPENDENT fragmentation schemes ────────────────────────
# The point of this stage is NOT one rule set's opinion. Each scheme proposes its own
# cut set; a bond's VOTE COUNT (how many independent schemes propose it) is recorded and
# reported, and the union is the candidate space the data statistic then chooses within.
_SMIRKS_SETS = {
    'recap': [c for c in (_compile(x) for _, x in recap_smirks) if c],
    'brics_smirks': [c for c in (_compile(x) for _, x in brics_smirks) if c],
    'crush': [c for c in (_compile(x) for _, x in CRUSH_smirks) if c],
}


def _smirks_bonds(mol, compiled, prot):
    bonds = set()
    for rmol, m2i, pairs in compiled:
        for match in mol.GetSubstructMatches(rmol, uniquify=False, maxMatches=5000):
            for ma, mb in pairs:
                if ma in m2i and mb in m2i:
                    a, b = match[m2i[ma]], match[m2i[mb]]
                    bd = mol.GetBondBetweenAtoms(a, b)
                    if bd and not bd.IsInRing() and a not in prot and b not in prot:
                        bonds.add(bd.GetIdx())
    return bonds


def _brics_rdkit(mol, prot):
    from rdkit.Chem import BRICS
    out = set()
    for (a, b), _ in BRICS.FindBRICSBonds(mol):
        bd = mol.GetBondBetweenAtoms(a, b)
        if bd and not bd.IsInRing() and a not in prot and b not in prot:
            out.add(bd.GetIdx())
    return out


def _rbrics_bonds(mol, prot):
    global _rbrics_bonds
    from rBRICS_public import FindrBRICSBonds as _f     # no fallback: a stubbed scheme

    def impl(m, p):                                      # would change every vote count
        out = set()
        for pair in _f(m):
            (a, b) = pair[0]
            bd = m.GetBondBetweenAtoms(a, b)
            if bd and not bd.IsInRing() and a not in p and b not in p:
                out.add(bd.GetIdx())
        return out
    _rbrics_bonds = impl
    return impl(mol, prot)


def _fg_boundary_bonds(mol, prot):
    # Ertl/EFGs boundary: a bond joining a functional group to the rest.
    inside = set()
    for grp in _get_fgs(mol):
        if len(grp) <= FG_CAP:
            inside |= set(grp)
    out = set()
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if b.IsInRing() or u in prot or v in prot:
            continue
        if (u in inside) != (v in inside):
            out.add(b.GetIdx())
    return out


for _nm, _src in (('recap', recap_smirks), ('brics_smirks', brics_smirks),
                  ('crush', CRUSH_smirks)):
    if not _SMIRKS_SETS[_nm]:
        raise RuntimeError(f'SMIRKS set {_nm!r} compiled to zero usable rules')
COMPILED_YIELD = {k: (len(v), len(src)) for k, v, src in
                  (('recap', _SMIRKS_SETS['recap'], recap_smirks),
                   ('brics_smirks', _SMIRKS_SETS['brics_smirks'], brics_smirks),
                   ('crush', _SMIRKS_SETS['crush'], CRUSH_smirks))}

SCHEMES = ('recap', 'brics_smirks', 'crush', 'brics_rdkit', 'rbrics', 'fg_boundary')


_VOTE_CACHE: dict = {}


def scheme_votes(mol):
    """{bond_idx: n_schemes_proposing} plus the per-scheme sets, for one molecule."""
    ck = mol.GetProp('_bfkey') if mol.HasProp('_bfkey') else None
    if ck is not None and ck in _VOTE_CACHE:
        return _VOTE_CACHE[ck]
    prot = _fused_protected(mol)
    per = {name: _smirks_bonds(mol, comp, prot) for name, comp in _SMIRKS_SETS.items()}
    per['brics_rdkit'] = _brics_rdkit(mol, prot)
    per['rbrics'] = _rbrics_bonds(mol, prot)
    per['fg_boundary'] = _fg_boundary_bonds(mol, prot)
    votes = Counter()
    for name in SCHEMES:
        for bi in per.get(name, ()):
            votes[bi] += 1
    if ck is not None:
        _VOTE_CACHE[ck] = (votes, per)
    return votes, per


def _fused_protected(mol):
    cnt = Counter()
    for ring in mol.GetRingInfo().AtomRings():
        for a in ring:
            cnt[a] += 1
    return {a for a, c in cnt.items() if c >= 2}


def _fg_internal(mol):
    frozen = set()
    for grp in _get_fgs(mol):
        if len(grp) > FG_CAP:
            continue
        g = set(grp)
        for b in mol.GetBonds():
            if b.GetBeginAtomIdx() in g and b.GetEndAtomIdx() in g:
                frozen.add(b.GetIdx())
    return frozen


def _components(mol, cut):
    n = mol.GetNumAtoms()
    p = list(range(n))

    def f(x):
        while p[x] != x:
            p[x] = p[p[x]]; x = p[x]
        return x

    for b in mol.GetBonds():
        if b.GetIdx() not in cut:
            ra, rb = f(b.GetBeginAtomIdx()), f(b.GetEndAtomIdx())
            if ra != rb:
                p[ra] = rb
    d = defaultdict(set)
    for a in range(n):
        d[f(a)].add(a)
    return list(d.values())


def _atom_components(mol, atoms):
    atoms = set(atoms)
    adj = defaultdict(set)
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if u in atoms and v in atoms:
            adj[u].add(v); adj[v].add(u)
    seen, out = set(), []
    for a in atoms:
        if a in seen:
            continue
        st, comp = [a], set()
        while st:
            x = st.pop()
            if x in comp:
                continue
            comp.add(x)
            st.extend(adj[x] - comp)
        seen |= comp
        out.append(comp)
    return out


# ── keys ──────────────────────────────────────────────────────────────────────
def _sub_smiles(mol, atoms):
    km = Chem.Mol(mol)
    Chem.RemoveStereochemistry(km)
    for b in km.GetBonds():
        b.SetBondDir(Chem.BondDir.NONE)
    return Chem.MolFragmentToSmiles(km, atomsToUse=sorted(atoms), canonical=True)


def _ring_key(mol, atoms, complete):
    return ('ring:' if complete else 'ringfrag:') + _sub_smiles(mol, atoms)


def _frag_key(mol, atoms):
    aset = set(atoms)
    n = mol.GetNumAtoms()
    cross = [b.GetIdx() for b in mol.GetBonds()
             if (b.GetBeginAtomIdx() in aset) != (b.GetEndAtomIdx() in aset)]
    if not cross:
        return Chem.MolToSmiles(mol, isomericSmiles=False)
    fr = Chem.FragmentOnBonds(mol, cross, addDummies=True)
    for pa, pm in zip(Chem.GetMolFrags(fr, asMols=False),
                      Chem.GetMolFrags(fr, asMols=True, sanitizeFrags=False)):
        if {a for a in pa if a < n} == aset:
            Chem.SanitizeMol(pm)
            return Chem.MolToSmiles(pm, isomericSmiles=False)
    return None


def _kekulized(mol):
    km = Chem.Mol(mol)
    Chem.Kekulize(km, clearAromaticFlags=True)
    return km


def _valid_frag_key(mol, atoms):
    """Key that is guaranteed to round-trip through MolFromSmiles, else None.

    An aromatic atom whose ring is only PARTLY inside `atoms` (a peeled fused-ring
    remnant) cannot be written as aromatic SMILES -- RDKit raises KekulizeException,
    and the old `try/except: pass` swallowed it and emitted a key from an unsanitized
    molecule, which is where the invalid 'cccc' keys came from. Detect that condition
    STRUCTURALLY and key off a kekulized copy; do not catch the exception."""
    aset = set(atoms)
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    broken_aromatic = any(
        mol.GetAtomWithIdx(a).GetIsAromatic()
        and any(a in r and not r <= aset for r in rings)
        for a in aset)
    k = _frag_key(_kekulized(mol) if broken_aromatic else mol, aset)
    if k and Chem.MolFromSmiles(k) is not None:
        return k
    return None


def _unit_key(mol, atoms, ring_atoms):
    atoms = set(atoms)
    if atoms <= ring_atoms:
        # COMPLETE is a property of the EXTRACTED piece, not of the parent's SSSR: a peeled
        # core is a perfectly good ring system even though the ring it was peeled from is
        # only partly present. Judging it on the parent labelled complete benzenes as
        # remnants (and collided ring:C1CNCCN1 with ringfrag:C1CNCCN1).
        sub = _sub_smiles(mol, atoms)
        sm = Chem.MolFromSmiles(sub)
        if sm is not None and sm.GetNumAtoms() and all(a.IsInRing() for a in sm.GetAtoms()):
            return 'ring:' + sub
        # genuine remnant: aromatic atoms outside a ring do not round-trip ('cccc' is not
        # parseable), so key it off a kekulized copy. Anything that cannot parse is dropped
        # silently downstream (see baselines/mage.py), so an invalid key is a real defect.
        k = _valid_frag_key(mol, atoms)
        if k is None:
            raise RuntimeError(f'unkeyable ring remnant {sorted(atoms)} in '
                               f'{Chem.MolToSmiles(mol)}')
        return k
    k = _valid_frag_key(mol, atoms)
    if k is None:
        raise RuntimeError(f'unkeyable unit {sorted(atoms)} in {Chem.MolToSmiles(mol)}')
    return k


# ── side-restricted WL environments ───────────────────────────────────────────
def _cuts_for(mol, tab, mode, inv, min_votes=1):
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    votes, per_scheme = scheme_votes(mol)
    frozen = _fg_internal(mol)
    cuts, scored = set(), {}
    openable = {}
    for b in mol.GetBonds():
        bi = b.GetIdx()
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if b.IsInRing():
            continue
        if not (u in ring_atoms or v in ring_atoms) and bi not in frozen:
            openable[bi] = (votes.get(bi, 0),)
        # Ring tier: ANY acyclic bond touching a ring atom is cut, so each ring
        # SYSTEM is its own unit and gets a substituent-agnostic key. This also
        # separates biaryls (both endpoints are ring atoms of different systems),
        # and cleanly cleaves a ring from an exocyclic FG. Outranks the FG freeze.
        if u in ring_atoms or v in ring_atoms:
            cuts.add(bi); continue
        if votes.get(bi, 0) < min_votes or bi in frozen:
            continue
        scored[bi] = [float(votes.get(bi, 0))]
        cuts.add(bi)          # vote gate above is the whole criterion
    return cuts, scored, ring_atoms, votes, per_scheme, openable


# ── ring tier: support-driven peeling of fused systems ────────────────────────
def _peel_options(mol, atoms):
    rings = [set(r) for r in mol.GetRingInfo().AtomRings() if set(r) <= atoms]
    out = []
    for r in rings:
        shared = set()
        for other in rings:
            if other is not r:
                shared |= (r & other)
        own = r - shared
        rest = atoms - own
        if not own or not rest:
            continue
        if len(_atom_components(mol, rest)) != 1:
            continue
        out.append((own, rest))
    return out


# ── entry point ───────────────────────────────────────────────────────────────
def build(smiles_all, method='bf_recur', groups=None, head_source='none',
          linker_method='mdl', break_fused_rings=False, verbose=True,
          sup_frac=SUP_FRAC, sup_floor=SUP_FLOOR, repair=True, min_votes=4,
          peel_lookahead=None):
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
    mode = METHODS[method]
    if peel_lookahead is None:                 # method name selects the peel policy
        peel_lookahead = (method != 'bf_recur_peel')
    mols = [Chem.MolFromSmiles(s) for s in smiles_all]
    for _s, _m in zip(smiles_all, mols):
        if _m is not None:
            _m.SetProp('_bfkey', _s)     # cache key: same input SMILES -> same bond indices

    # ── SPLIT SCOPE OF EVERY CORPUS STATISTIC -- declare it, do not let these drift ──
    #   build_breakability   : ALL splits   (unsupervised bond-environment counts)
    #   support() / floor    : ALL splits   (unsupervised fragment counts)
    #   kept-motif threshold : TRAIN+VAL    (downstream, in generate_vocab_rules)
    # The support floor counts FRAGMENTS, never labels, so counting test molecules is
    # not leakage -- and it is the scope build_breakability already used. Keeping the
    # two different was an inconsistency, not a safeguard.
    # MEASURED, BBBP fold 0 (1672 train+val vs 1859 all): floor is pinned at
    # sup_floor=20 since 0.005*1859=9.3, so only 2 of 1001 fragments change (furan,
    # a steroid core). On a larger corpus the fraction term dominates
    # (0.005*7670=38) and this scope genuinely matters.
    # `groups` stays REQUIRED so the caller states splits explicitly for the
    # downstream train+val kept-motif threshold.
    if groups is None:
        raise ValueError('recur_frag.build requires `groups` (split labels): the '
                         'downstream kept-motif threshold is defined on train+val, so '
                         'the caller must state the splits explicitly.')
    n_all = max(sum(1 for m in mols if m is not None), 1)
    floor = max(sup_floor, int(round(sup_frac * n_all)))

    per_mol = []
    for m in mols:
        if m is None:
            per_mol.append(None); continue
        cuts, scored, ring_atoms, votes, per_scheme, openable = _cuts_for(
            m, None, mode, None, min_votes=min_votes)
        per_mol.append({'mol': m, 'cuts': cuts, 'scored': scored,
                        'ring': ring_atoms, 'votes': votes, 'schemes': per_scheme,
                        'openable': openable})

    def emit(d):
        return [(set(u), _unit_key(d['mol'], u, d['ring']))
                for u in _components(d['mol'], d['cuts'])]

    def support():
        c = Counter()                      # ALL splits -- see SPLIT SCOPE note above
        for d in per_mol:
            if d is None:
                continue
            for _, k in emit(d):
                c[k] += 1
        return c

    n_peel = n_split = 0
    if repair:
        for _ in range(8):                                   # ring tier: peel
            sup = support()
            changed = False
            for d in per_mol:
                if d is None:
                    continue
                for atoms, k in emit(d):
                    if not k.startswith('ring:') or sup.get(k, 0) >= floor:
                        continue
                    opts = _peel_options(d['mol'], atoms)
                    if not opts:
                        continue          # nothing peelable in this fused system
                    # ── PEEL POLICY (see PEEL POLICY block at top of file) ──────────
                    if peel_lookahead:
                        # CONSERVATIVE: peel only if BOTH the core and the remnant clear
                        # the floor. Protects canonical fused cores (indole,
                        # benzimidazole, steroid nuclei) from being shattered into a
                        # common ring plus an unusable remnant. peels=12 on BBBP fold 0.
                        viable = []
                        for own_, rest_ in opts:
                            k_core = _unit_key(d['mol'], rest_, d['ring'])
                            k_rem = _unit_key(d['mol'], own_, d['ring'])
                            n_occ = sup.get(k, 0)
                            if (sup.get(k_core, 0) + n_occ >= floor
                                    and sup.get(k_rem, 0) + n_occ >= floor):
                                viable.append((own_, rest_, sup.get(k_core, 0)))
                        if not viable:
                            continue
                        best = max(viable, key=lambda o: (o[2], _sub_smiles(d['mol'], o[1])))
                        own, rest = best[0], best[1]
                    else:
                        # AGGRESSIVE: peel whenever the whole fused system is below the
                        # floor, regardless of what the pieces look like. Chemically
                        # worse, but this is the setting that beats rBRICS -- see the
                        # PEEL POLICY block. peels=1257 on BBBP fold 0.
                        best = max(opts, key=lambda o: (
                            sup.get(_unit_key(d['mol'], o[1], d['ring']), 0),
                            _sub_smiles(d['mol'], o[1])))
                        own, rest = best
                    for b in d['mol'].GetBonds():
                        x, y = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                        if (x in own) != (y in own) and x in atoms and y in atoms:
                            d['cuts'].add(b.GetIdx())
                    changed = True; n_peel += 1
            if not changed:
                break

        for _ in range(8):                                   # linker tier: split
            sup = support()
            plan = {}
            for d in per_mol:
                if d is None:
                    continue
                for atoms, k in emit(d):
                    if k.startswith('ring:') or sup.get(k, 0) >= floor:
                        continue
                    opts = [(bi, sc) for bi, sc in d['openable'].items()
                            if bi not in d['cuts']
                            and d['mol'].GetBondWithIdx(bi).GetBeginAtomIdx() in atoms
                            and d['mol'].GetBondWithIdx(bi).GetEndAtomIdx() in atoms]
                    if not opts:
                        continue
                    bi = max(opts, key=lambda o: o[1])[0]
                    sub = _components(d['mol'], d['cuts'] | {bi})
                    parts = [p for p in sub if p & atoms]
                    if len(parts) != 2:
                        continue
                    ks = tuple(sorted(_unit_key(d['mol'], p, d['ring']) for p in parts))
                    plan.setdefault(k, Counter())[ks] += 1
            if not plan:
                break
            accept = set()
            for k, cand in plan.items():
                ks, n_occ = cand.most_common(1)[0]
                # both halves must clear the floor once every occurrence of k splits
                if all(sup.get(x, 0) + n_occ >= floor for x in ks):
                    accept.add(k)
            if not accept:
                break
            changed = False
            for d in per_mol:
                if d is None:
                    continue
                for atoms, k in emit(d):
                    if k not in accept:
                        continue
                    opts = [(bi, sc) for bi, sc in d['openable'].items()
                            if bi not in d['cuts']
                            and d['mol'].GetBondWithIdx(bi).GetBeginAtomIdx() in atoms
                            and d['mol'].GetBondWithIdx(bi).GetEndAtomIdx() in atoms]
                    if not opts:
                        continue
                    d['cuts'].add(max(opts, key=lambda o: o[1])[0])
                    changed = True; n_split += 1
            if not changed:
                break

    out = []
    for d in per_mol:
        if d is None:
            out.append([]); continue
        out.append([(k, atoms) for atoms, k in emit(d)])
    if verbose:
        nv = sum(1 for m in mols if m is not None)
        sc_tot = Counter(); agree = Counter(); n_lb = 0
        for d in per_mol:
            if d is None:
                continue
            for name in SCHEMES:
                sc_tot[name] += len(d['schemes'].get(name, ()))
            for bi, v in d['votes'].items():
                if bi in d['scored'] or bi in d['cuts']:
                    agree[v] += 1; n_lb += 1
        print(f"    [boundary_frag] method={method} r_max={R_MAX} floor={floor} "
              f"(n_all={n_all}) peels={n_peel} splits={n_split} on {nv}/{len(mols)} mols",
              flush=True)
        print("      scheme proposals: "
              + "  ".join(f"{k}={sc_tot[k]}" for k in SCHEMES), flush=True)
        print("      linker-bond scheme agreement: "
              + "  ".join(f"{v}vote={100*agree[v]/max(n_lb,1):.0f}%"
                          for v in sorted(agree)), flush=True)
    return out
