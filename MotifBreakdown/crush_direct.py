"""crush_direct.py — CRUSH-direct fragmentation (chemically-meaningful cuts, NO MDL).

Recursively cut a molecule at CRUSH/BRICS/RECAP SMIRKS-matched ACYCLIC bonds while keeping every
fragment >= min_frag heavy atoms and never breaking rings / fused ring systems. The resulting units
ARE the fragmentation — there is NO corpus MDL/BPE selection.

Why no MDL (empirical, Aug 2026, one-fold-per-dataset study): unsupervised MDL/BPE over any pool
collapses to singletons (frequency favors ubiquitous 1-atom fragments — 91% singletons even at
model-weight 0), OR is inert once singletons are banned (<=20 merges, vocab unchanged). CRUSH's rule
cuts alone give 100% atom coverage, 0 singletons, cause-bearing mid-size units (BBBP median 6 atoms).
Data-driven granularity that preserves cause-bearing fragments requires SUPERVISION, not frequency.

Top-level entry:
    build(smiles_all, head_source='none', linker_method='mdl', break_fused_rings=False, verbose=True)
returns per-molecule ``[(key, set(atoms))]`` aligned to ``smiles_all`` (empty list for unparseable
SMILES) — the exact format generate_vocab_rules consumes (identical to ring_mdl.build). Keys are
re-derived by generate_vocab_rules (rekey_structural, [*] attachment dummies); the cf.frag_key here
is a sensible default. head_source / linker_method / break_fused_rings are accepted for
call-compatibility but IGNORED (deterministic rule cuts; no head-freeze / linker选择).

Rules are vendored verbatim from CRUSH (https://github.com/EdgL2/CRUSH) in crush_smirks_rules.py.
"""
from __future__ import annotations

from collections import defaultdict, Counter

from rdkit import Chem

import chemfrag as cf
from crush_smirks_rules import RULES

MIN_FRAG = 3   # no fragment below this many heavy atoms (this is what kills singletons)


# ───────────────────────── SMIRKS -> cut-bond compilation ──────────────────────
def _split_top(s):
    """split a SMARTS product side on top-level '.' (respecting [] and ())."""
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
    """Return (reactant_query, mapnum->tmpl_idx, [(mapA,mapB) cut-pairs]) for one SMIRKS.
    A cut-pair = a bond in the reactant template whose two mapped atoms end up in DIFFERENT products
    (i.e., the bond the rule breaks)."""
    react, prod = smirks.split('>>')
    rmol = Chem.MolFromSmarts(react)
    if rmol is None:
        return None
    m2i = {a.GetAtomMapNum(): a.GetIdx() for a in rmol.GetAtoms() if a.GetAtomMapNum()}
    groups = []
    for p in _split_top(prod):
        pm = Chem.MolFromSmarts(p)
        if pm is not None:
            groups.append({a.GetAtomMapNum() for a in pm.GetAtoms() if a.GetAtomMapNum()})
    pairs = []
    for b in rmol.GetBonds():
        ma, mb = b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()
        if not ma or not mb:
            continue
        ga = next((i for i, g in enumerate(groups) if ma in g), -1)
        gb = next((i for i, g in enumerate(groups) if mb in g), -1)
        if ga != gb:
            pairs.append((ma, mb))
    return (rmol, m2i, pairs)


_COMPILED = [c for c in (_compile(s) for _, s in RULES) if c]


# ───────────────────────────── recursive cutting ──────────────────────────────
def _fused_protected(mol):
    """atoms in >=2 SSSR rings (fused/spiro) — never cut a bond touching these (mirrors CRUSH)."""
    cnt = Counter()
    for ring in mol.GetRingInfo().AtomRings():
        for a in ring:
            cnt[a] += 1
    return {a for a, c in cnt.items() if c >= 2}


def _meaningful_bonds(mol):
    """acyclic bonds matched (as cut-pairs) by any CRUSH/BRICS/RECAP rule, not touching fused atoms."""
    prot = _fused_protected(mol)
    bonds = set()
    for rmol, m2i, pairs in _COMPILED:
        for match in mol.GetSubstructMatches(rmol):
            for ma, mb in pairs:
                if ma in m2i and mb in m2i:
                    a, b = match[m2i[ma]], match[m2i[mb]]
                    bd = mol.GetBondBetweenAtoms(a, b)
                    if bd and not bd.IsInRing() and a not in prot and b not in prot:
                        bonds.add(bd.GetIdx())
    return bonds


class _UF:
    def __init__(s, n):
        s.p = list(range(n))

    def f(s, x):
        while s.p[x] != x:
            s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x

    def u(s, a, b):
        s.p[s.f(a)] = s.f(b)


def _components(mol, cut):
    n = mol.GetNumAtoms(); uf = _UF(n)
    for b in mol.GetBonds():
        if b.GetIdx() not in cut:
            uf.u(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    d = defaultdict(set)
    for a in range(n):
        d[uf.f(a)].add(a)
    return list(d.values())


def crush_units(mol, min_frag=MIN_FRAG):
    """CRUSH recursive cut, bond-wise (atom-tracked): cut a meaningful bond only if BOTH resulting
    pieces (within the current component) have >= min_frag heavy atoms. Returns a PARTITION of the
    atoms into units (list of atom-sets); rings/fused systems stay whole (never cut)."""
    cand = _meaningful_bonds(mol)
    S = set()
    changed = True
    while changed:
        changed = False
        cur = _components(mol, S)
        a2c = {a: i for i, c in enumerate(cur) for a in c}
        for b in sorted(cand - S):
            bd = mol.GetBondWithIdx(b)
            a1, a2 = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
            if a2c[a1] != a2c[a2]:
                continue
            parts = [p for p in _components(mol, S | {b}) if a1 in p or a2 in p]
            if len(parts) == 2 and all(len(p) >= min_frag for p in parts):
                S.add(b); changed = True; break
    return _components(mol, S)


# ───────────────────────────────── entry point ────────────────────────────────
def build(smiles_all, head_source='none', linker_method='mdl', break_fused_rings=False,
          verbose=True, min_frag=MIN_FRAG):
    """Per-molecule [(key, set(atoms))] aligned to smiles_all (empty for unparseable). No MDL."""
    out = []
    n_ok = 0
    for s in smiles_all:
        m = Chem.MolFromSmiles(s)
        if m is None:
            out.append([]); continue
        n_ok += 1
        units = crush_units(m, min_frag)
        out.append([(cf.frag_key(m, frozenset(u)), set(u)) for u in units])
    if verbose:
        print(f"    [crush_direct] CRUSH rule cuts (min_frag={min_frag}, NO MDL) on "
              f"{n_ok}/{len(smiles_all)} mols", flush=True)
    return out
