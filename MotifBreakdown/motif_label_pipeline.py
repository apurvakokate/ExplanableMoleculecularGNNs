#!/usr/bin/env python3
"""
motif_label_pipeline.py  —  v5
================================
Motif-based synthetic label generation pipeline.

Design
------
Labels are generated purely from molecular structure — original labels are
never used during rule construction. They are used ONLY in step 6 for
post-hoc evaluation (SNR against original labels).

Step 1  select_top_motifs()         label-free, scored by support × n_atoms
Step 2  build_metadata_structural() pairwise Jaccard, subsuming families
Step 3  build_clauses_structural()  k=1 singletons + k=2 AND clauses
                                    AND constraint: structural witnesses required
Step 4  find_best_rule()            score = balance × separation × (1-spurious)
                                    separation = structural SNR (label-free)
Step 5  apply_synthetic_labels()    rule fires → 1, no-fire → 0
Step 6  compute_snr()               post-hoc: precision/SNR vs original labels

Motif selection scoring
-----------------------
  selection_score = support_pct × n_heavy_atoms

High support means the motif appears frequently (useful rule anchor).
High atom count means the motif is structurally specific (not a trivial linker).
A 9% motif with 6 atoms scores the same as an 18% motif with 3 atoms.
Single-atom fragments (n_heavy < 2) are always excluded.

Legacy public API (used by generate_vocab_rules.extract_rules — unchanged)
--------------------------------------------------------------------------
  build_catalog / compute_alert_families / compute_subsuming_families
  cooc_profile / build_proxy_lookup / build_clauses / build_dnf_rules
  label_dist / clause_mask / atom_count / jaccard / get_core
  too_generic / check_sub
"""

import os, warnings
import numpy as np
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Set, Tuple, Optional

warnings.filterwarnings('ignore')

from rdkit import Chem, RDLogger
from rdkit.Chem import RWMol
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog('rdApp.*')

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

TRIVIAL: Set[str] = {
    '*O*', '*N*', '*C*', '*CC*', '*CC', '*OC',
    '*C(*)*', '*S*', '*N(*)*',
}
MIN_SUP      = 0.01     # legacy (extract_rules)
J_COOC       = 0.15     # legacy (cooc_profile)
J_HIGH       = 0.70     # AND clause skip: always co-occurring pairs
P_AMBIGUOUS  = 0.70     # legacy (proxy lookup)
TOP_N        = 10       # legacy (extract_rules)
MIN_COV      = 5.0      # legacy (build_dnf_rules)
AND_MIN_ONLY = 0.05     # each singleton-only group must cover >= 5% of N
SNR_SAMPLE   = 60       # molecules sampled per group for structural SNR
CATALOG_NAMES = ['BRENK', 'CHEMBL_Dundee', 'CHEMBL_LINT']


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def atom_count(smarts: str) -> int:
    """Count heavy (non-wildcard) atoms only.
    [*]O[*] → 1,  [*]c1ccccc1 → 6,  [*][N+](=O)[O-] → 3.
    Matches molfragbpe5.atom_count — wildcards are NOT counted.
    """
    try:
        m = Chem.MolFromSmarts(smarts)
        if m is None:
            return 0
        return sum(1 for a in m.GetAtoms() if a.GetAtomicNum() != 0)
    except Exception:
        return 0


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    i = int((a & b).sum())
    u = int((a | b).sum())
    return round(i / u, 3) if u else 0.0


def label_dist(mask: np.ndarray, n: int) -> dict:
    n1 = int(mask.astype(bool).sum())
    return {'n1': n1, 'n0': n - n1,
            'pct1': round(n1 / n * 100, 1),
            'pct0': round((n - n1) / n * 100, 1)}


def clause_mask(c: dict, masks: Dict[str, np.ndarray]) -> np.ndarray:
    m = masks[c['motifs'][0]].astype(bool).copy()
    for s in c['motifs'][1:]:
        m &= masks[s].astype(bool)
    return m


def get_core(smarts: str) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmarts(smarts)
    if mol is None:
        return None
    rw = RWMol(mol)
    wildcards = sorted([a.GetIdx() for a in rw.GetAtoms()
                        if a.GetAtomicNum() == 0], reverse=True)
    for idx in wildcards:
        rw.RemoveAtom(idx)
    try:
        core = rw.GetMol()
        if core.GetNumAtoms() == 0:
            return None
        Chem.FastFindRings(core)
        try:
            Chem.SanitizeMol(core,
                Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                Chem.SanitizeFlags.SANITIZE_SYMMRINGS)
        except Exception:
            pass
        return core
    except Exception:
        return None


def heteroatoms(core: Chem.Mol) -> Set[int]:
    return {a.GetAtomicNum() for a in core.GetAtoms()
            if a.GetAtomicNum() not in (0, 6)}


def too_generic(core: Chem.Mol) -> bool:
    if core.GetRingInfo().NumRings() > 0:
        return False
    if core.GetNumAtoms() >= 3:
        return False
    for bond in core.GetBonds():
        if bond.GetBondTypeAsDouble() >= 2.0:
            return False
    return True


def aliphatic_pure_C(core: Chem.Mol, het: Set[int]) -> bool:
    if het or core.GetRingInfo().NumRings() > 0:
        return False
    return not any(a.GetIsAromatic() for a in core.GetAtoms())


def check_sub(sa: str, sb: str) -> bool:
    ca, cb = get_core(sa), get_core(sb)
    if ca is None or cb is None:
        return False
    ha, hb = heteroatoms(ca), heteroatoms(cb)
    ab, ba = False, False
    try:    ab = cb.HasSubstructMatch(ca)
    except Exception: pass
    if ab:
        if too_generic(ca): ab = False
        elif aliphatic_pure_C(ca, ha) and hb: ab = False
        elif ha and not ha <= hb: ab = False
    try:    ba = ca.HasSubstructMatch(cb)
    except Exception: pass
    if ba:
        if too_generic(cb): ba = False
        elif aliphatic_pure_C(cb, hb) and ha: ba = False
        elif hb and not hb <= ha: ba = False
    return ab or ba


def motif_intrinsic_alerts(smarts: str, catalog: FilterCatalog) -> frozenset:
    mol = Chem.MolFromSmiles(smarts.replace('[*]', 'C').replace('*', 'C'))
    if mol is None:
        return frozenset()
    return frozenset(e.GetDescription() for e in catalog.GetMatches(mol))


def build_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    for name in CATALOG_NAMES:
        params.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, name))
    return FilterCatalog(params)


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY API (generate_vocab_rules.extract_rules — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def compute_alert_families(top_motifs, all_cands, catalog):
    top_set    = set(top_motifs)
    top_alerts = {t: motif_intrinsic_alerts(t, catalog) for t in top_motifs}
    a2t: Dict[str, Set[str]] = defaultdict(set)
    for t, als in top_alerts.items():
        for al in als: a2t[al].add(t)
    ag: Dict[str, Dict] = defaultdict(lambda: defaultdict(list))
    for _, s, sv in all_cands:
        for al in motif_intrinsic_alerts(s, catalog):
            if al not in a2t: continue
            for top in a2t[al]:
                ag[top][al].append({'motif': s, 'support': round(sv*100,1),
                                    'in_support': sv >= MIN_SUP,
                                    'in_top': s in top_set})
    return top_alerts, {t: dict(d) for t, d in ag.items()}


def compute_subsuming_families(top_motifs, all_cands):
    top_set  = set(top_motifs)
    tc, th   = {}, {}
    for t in top_motifs:
        c = get_core(t)
        if c is not None:
            tc[t] = c; th[t] = heteroatoms(c)
    sub: Dict[str, List] = defaultdict(list)
    for _, sb, sv_b in all_cands:
        cb = get_core(sb)
        if cb is None: continue
        hb = heteroatoms(cb); nb = cb.GetNumAtoms()
        for t in top_motifs:
            ca = tc.get(t)
            if ca is None: continue
            ha = th.get(t, set()); na = ca.GetNumAtoms()
            ab = ba = False
            if na <= nb and (not ha or ha <= hb):
                try: ab = cb.HasSubstructMatch(ca)
                except Exception: pass
                if ab and (too_generic(ca) or (aliphatic_pure_C(ca,ha) and hb)
                           or (ha and not ha <= hb)): ab = False
            if nb <= na and (not hb or hb <= ha):
                try: ba = ca.HasSubstructMatch(cb)
                except Exception: pass
                if ba and (too_generic(cb) or (aliphatic_pure_C(cb,hb) and ha)
                           or (hb and not hb <= ha)): ba = False
            if ab and ba:
                if t.count('*') <= sb.count('*'): ba = False
                else: ab = False
            if ab or ba:
                sub[t].append({'motif': sb, 'support': round(sv_b*100,1),
                               'in_support': sv_b >= MIN_SUP,
                               'in_top': sb in top_set,
                               'direction': 'specific' if ab else 'general'})
    return dict(sub)


def cooc_profile(top_motifs, all_cands, all_masks):
    top_set = set(top_motifs)
    prof: Dict[Tuple, dict] = {}
    cg:   Dict[str, List]   = defaultdict(list)
    for t in top_motifs:
        ma = all_masks.get(t)
        if ma is None: continue
        ma = ma.astype(bool); na = int(ma.sum())
        if na == 0: continue
        for _, sb, sv_b in all_cands:
            if sb == t: continue
            mb = all_masks.get(sb)
            if mb is None: continue
            mb = mb.astype(bool); nb = int(mb.sum())
            if nb == 0: continue
            inter = int((ma & mb).sum())
            u = na + nb - inter
            J   = round(inter/u,3) if u else 0.0
            pba = round(inter/na,3) if na else 0.0
            pab = round(inter/nb,3) if nb else 0.0
            prof[(t,sb)] = {'J':J,'p_b_given_a':pba,'p_a_given_b':pab,'inter':inter}
            if sb in top_set:
                prof[(sb,t)] = {'J':J,'p_b_given_a':pab,'p_a_given_b':pba,'inter':inter}
            if J >= J_COOC:
                cg[t].append({'motif':sb,'support':round(sv_b*100,1),
                              'in_support':sv_b>=MIN_SUP,'in_top':sb in top_set,
                              'J':J,'p_b_given_a':pba,'p_a_given_b':pab})
    for t in cg: cg[t].sort(key=lambda x:-x['J'])
    return prof, dict(cg)


def build_proxy_lookup(profile):
    lup: Dict[str, List] = {}
    for (sa,sb), st in profile.items():
        if st['p_b_given_a'] >= P_AMBIGUOUS:
            lup.setdefault(sa,[]).append((sb,st['J'],st['p_b_given_a']))
    for sa in lup: lup[sa].sort(key=lambda x:-x[2])
    return lup


def build_clauses(top_motifs, masks, profile, n):
    cl = []
    tl = list(top_motifs)
    for s in tl:
        cl.append({'motifs':[s],'k':1,'pair_stats':{},
                   **label_dist(masks[s].astype(bool),n)})
    for k in [2,3]:
        for idx in combinations(range(len(tl)),k):
            sel = [tl[i] for i in idx]
            if any(profile.get((sel[ia],sel[ib]),{}).get('J',0) >= J_HIGH
                   for ia,ib in combinations(range(k),2)): continue
            inter = masks[sel[0]].astype(bool).copy()
            for s in sel[1:]: inter &= masks[s].astype(bool)
            if not inter.any(): continue
            if int(inter.sum()) >= min(int(masks[s].sum()) for s in sel): continue
            dist = label_dist(inter,n)
            ps = {}
            for ia,ib in combinations(range(k),2):
                p = profile.get((sel[ia],sel[ib]),{})
                ps[f'{sel[ia]}|||{sel[ib]}'] = {
                    'J':p.get('J',0),'p_b_given_a':p.get('p_b_given_a',0),
                    'p_a_given_b':p.get('p_a_given_b',0)}
            cl.append({'motifs':sel,'k':k,'pair_stats':ps,**dist})
    return cl


def ambiguity(rule_motifs, proxy_lookup):
    rs = set(rule_motifs)
    fl = []
    for m in rule_motifs:
        for proxy,J,p in proxy_lookup.get(m,[]):
            if proxy not in rs:
                fl.append({'rule_motif':m,'proxy':proxy,'J':J,'p_proxy_given_rule':p})
    fl.sort(key=lambda x:-x['p_proxy_given_rule'])
    return fl


def build_dnf_rules(clauses, masks, profile, n, proxy_lookup=None):
    cm = [clause_mask(c,masks) for c in clauses]
    if proxy_lookup is None: proxy_lookup = build_proxy_lookup(profile)
    lc: List[frozenset] = []
    rules: List[dict]   = []
    for nc in range(4,0,-1):
        tier = []
        for idx in combinations(range(len(clauses)),nc):
            is_ = frozenset(idx)
            if any(is_ < low for low in lc): continue
            final = cm[idx[0]].copy()
            for i in idx[1:]: final |= cm[i]
            if not final.any(): lc.append(is_); continue
            dist = label_dist(final,n)
            if dist['pct1'] < MIN_COV: lc.append(is_); continue
            sc = [clauses[i] for i in idx]
            am = list(dict.fromkeys(m for c in sc for m in c['motifs']))
            tier.append({'n_clauses':nc,
                         'clauses':[{'motifs':c['motifs'],'k':c['k'],
                                     'pair_stats':c['pair_stats']} for c in sc],
                         'ambiguity':ambiguity(am,proxy_lookup),**dist})
        tier.sort(key=lambda x:-x['n1'])
        rules.extend(tier[:50])
    return rules


def score_dnf_rules(rules: List[dict],
                    masks: Dict[str, np.ndarray],
                    tv_frags: List[List[str]],
                    pairwise: Dict[Tuple, dict],
                    sub_fams: Dict[str, List],
                    n: int,
                    snr_top_k: int = 200) -> List[dict]:
    """Attach the structural balance-aware score to each DNF rule in place.

    Reuses the existing components from the label-free structural pipeline:

        balance    = 1 - |pct_match - 50| / 50      (synthetic split quality)
        separation = 1 - mean_J(match, no-match)    (structural SNR, label-free)
        spurious   = mean pairwise J + subsuming    (motif redundancy)
        score      = balance * separation * (1 - spurious)

    ``pct_match`` here is the rule's coverage over ALL molecules — exactly the
    ``pct1`` field that ``label_dist`` already stores on each rule (n1 = #matched,
    n0 = n - n1). The rule mask is the OR of its clause masks (each clause = AND
    of its motifs).

    For efficiency, structural separation (the expensive part) is computed only
    for the ``snr_top_k`` rules with the highest balance*(1-spurious); the rest
    get ``separation=None`` and ``score=0.0`` so they sort to the bottom but
    remain inspectable. Returns the same list, sorted by score descending.
    """
    def _rule_mask(rule: dict) -> np.ndarray:
        combined = None
        for c in rule['clauses']:
            cm = clause_mask({'motifs': c['motifs']}, masks)
            combined = cm if combined is None else (combined | cm)
        if combined is None:
            return np.zeros(n, dtype=bool)
        return combined.astype(bool)

    # Phase 1: fast components (balance, spurious) for every rule
    for r in rules:
        mset = list(dict.fromkeys(m for c in r['clauses'] for m in c['motifs']))
        # The DNF mask's coverage over ALL molecules is exactly pct1 here:
        # label_dist sets n1 = #matched, n0 = n - n1, pct1 = n1/n*100.
        pct_match = float(r['pct1'])
        balance   = 1 - abs(pct_match - 50) / 50
        spurious  = _spurious_score(mset, pairwise, sub_fams)
        r['rule_pct_match'] = round(pct_match, 1)
        r['balance']        = round(balance, 3)
        r['spurious']       = round(spurious, 4)
        r['_fast']          = balance * (1 - spurious)

    # Phase 2: structural separation only for the most promising rules
    order = sorted(rules, key=lambda x: -x['_fast'])
    for r in order[:snr_top_k]:
        sep = compute_structural_snr(_rule_mask(r), tv_frags)
        r['separation'] = sep
        r['score']      = round(r['balance'] * sep * (1 - r['spurious']), 5)
    for r in order[snr_top_k:]:
        r['separation'] = None
        r['score']      = 0.0
    for r in rules:
        r.pop('_fast', None)

    return sorted(rules, key=lambda x: -x['score'])


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — LABEL-FREE RULE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def select_top_motifs(raw_stats: dict,
                      tv_frags: List[List[str]],
                      N_tv: int,
                      min_sup_pct: float = 1.0,
                      max_n: int = 25) -> Tuple[List[dict], Dict[str, np.ndarray]]:
    """
    Step 1 — LABEL-FREE.

    Select top N motifs scored by:   support_pct × n_heavy_atoms

    High support = frequently-seen anchor.
    High atom count = structurally specific, not a trivial linker.

    Hard filters:
      - support >= min_sup_pct  (default 1% of train+val)
      - n_heavy >= 2            (excludes [*]O[*], [*]N[*], [*]C, etc.)

    Returns top max_n sorted by selection_score descending.
    """
    min_n = max(1, int(min_sup_pct / 100 * N_tv))

    presence: Dict[str, np.ndarray] = {}
    for s, d in raw_stats.items():
        if d['n'] >= min_n:
            presence[s] = np.array([s in set(f) for f in tv_frags])

    candidates = []
    for s, d in raw_stats.items():
        if s not in presence:
            continue
        sup = d['n'] / N_tv * 100
        if sup < min_sup_pct:
            continue
        n_heavy = atom_count(s)   # wildcards excluded
        if n_heavy < 2:
            continue               # hard filter: no single-atom trivials
        sel_score = round(sup * n_heavy, 3)
        candidates.append({
            's':         s,
            'sup':       round(sup, 2),
            'n':         d['n'],
            'n_atoms':   n_heavy,
            'sel_score': sel_score,
        })

    candidates.sort(key=lambda x: -x['sel_score'])
    return candidates[:max_n], presence


def build_metadata_structural(selected: List[dict],
                               presence: Dict[str, np.ndarray]) -> Tuple[dict, dict]:
    """
    Step 2 — LABEL-FREE.
    Pairwise Jaccard between all selected motifs.
    Subsuming families: structural subgraph relationships.
    """
    pairwise: Dict[Tuple, dict] = {}
    for i, j in combinations(range(len(selected)), 2):
        si, sj = selected[i]['s'], selected[j]['s']
        vi = presence[si].astype(bool)
        vj = presence[sj].astype(bool)
        inter = int((vi & vj).sum())
        ni, nj = int(vi.sum()), int(vj.sum())
        u    = ni + nj - inter
        J    = round(inter/u, 3) if u else 0.0
        pij  = round(inter/ni, 3) if ni else 0.0
        pji  = round(inter/nj, 3) if nj else 0.0
        pairwise[(si,sj)] = {'J':J,'p_j_given_i':pij,'p_i_given_j':pji}
        pairwise[(sj,si)] = {'J':J,'p_j_given_i':pji,'p_i_given_j':pij}

    sub_fams: Dict[str, List] = defaultdict(list)
    for i, ci in enumerate(selected):
        si  = ci['s']; cai = get_core(si)
        if cai is None: continue
        hai = heteroatoms(cai)
        for j, cj in enumerate(selected):
            if i == j: continue
            sj  = cj['s']; caj = get_core(sj)
            if caj is None: continue
            haj = heteroatoms(caj)
            na, nb = cai.GetNumAtoms(), caj.GetNumAtoms()
            ab = ba = False
            if na <= nb and (not hai or hai <= haj):
                try: ab = caj.HasSubstructMatch(cai)
                except Exception: pass
                if ab and (too_generic(cai) or
                           (aliphatic_pure_C(cai,hai) and haj) or
                           (hai and not hai <= haj)): ab = False
            if nb <= na and (not haj or haj <= hai):
                try: ba = cai.HasSubstructMatch(caj)
                except Exception: pass
                if ba and (too_generic(caj) or
                           (aliphatic_pure_C(caj,haj) and hai) or
                           (haj and not haj <= hai)): ba = False
            if ab and ba:
                if si.count('*') <= sj.count('*'): ba = False
                else: ab = False
            if ab:
                sub_fams[si].append({'motif':sj,'direction':'specific',
                                     'sup':cj['sup']})
            elif ba:
                sub_fams[si].append({'motif':sj,'direction':'general',
                                     'sup':cj['sup']})

    return pairwise, dict(sub_fams)


def build_clauses_structural(selected: List[dict],
                              presence: Dict[str, np.ndarray],
                              pairwise: Dict[Tuple, dict],
                              N_tv: int) -> List[dict]:
    """
    Step 3 — LABEL-FREE.

    k=1 singletons — always valid.

    k=2 AND clauses — valid when all hold:
      1. Non-empty intersection.
      2. J < J_HIGH — not always co-occurring (redundant signal).
      3. Intersection < both singletons — genuinely more specific.
      4. |only-A| >= AND_MIN_ONLY×N and |only-B| >= AND_MIN_ONLY×N
         Structural negative witnesses exist for both components.
         This ensures the GNN must learn BOTH parts of the conjunction.
    """
    clauses: List[dict] = []

    for c in selected:
        s = c['s']
        clauses.append({
            'motifs':    [s],
            'k':         1,
            'n_match':   int(presence[s].sum()),
            'pct_match': round(presence[s].sum()/N_tv*100, 1),
            'and_valid': True,
        })

    min_only = AND_MIN_ONLY * N_tv

    for ci, cj in combinations(selected, 2):
        si, sj = ci['s'], cj['s']
        vi     = presence[si].astype(bool)
        vj     = presence[sj].astype(bool)
        inter  = vi & vj
        if not inter.any():
            continue
        J = pairwise.get((si,sj), {}).get('J', 0.0)
        if J >= J_HIGH:
            continue
        if inter.sum() >= min(vi.sum(), vj.sum()):
            continue
        only_i = vi & ~vj
        only_j = vj & ~vi
        clauses.append({
            'motifs':    [si, sj],
            'k':         2,
            'n_match':   int(inter.sum()),
            'pct_match': round(inter.sum()/N_tv*100, 1),
            'and_valid': (only_i.sum() >= min_only and only_j.sum() >= min_only),
            'J':         J,
            'n_only_i':  int(only_i.sum()),
            'n_only_j':  int(only_j.sum()),
        })

    return clauses


def compute_structural_snr(combined_mask: np.ndarray,
                            tv_frags: List[List[str]],
                            n_sample: int = SNR_SAMPLE,
                            seed: int = 42) -> float:
    """
    Structural SNR (label-free).

    separation = 1 - mean_J(matching_mols, non_matching_mols)

    Samples n_sample molecules from each group and computes mean pairwise
    Jaccard of their fragment sets. Higher = more structurally distinct
    groups = cleaner rule partition.
    """
    match_idx   = np.where(combined_mask)[0]
    nomatch_idx = np.where(~combined_mask)[0]
    if len(match_idx) == 0 or len(nomatch_idx) == 0:
        return 0.0
    rng    = np.random.RandomState(seed)
    m_idx  = rng.choice(match_idx,   min(n_sample, len(match_idx)),   replace=False)
    nm_idx = rng.choice(nomatch_idx, min(n_sample, len(nomatch_idx)), replace=False)
    js = []
    for mi in m_idx:
        fa = set(tv_frags[mi])
        if not fa: continue
        for nmi in nm_idx:
            fb = set(tv_frags[nmi])
            if not fb: continue
            inter = len(fa & fb)
            union = len(fa | fb)
            js.append(inter/union if union else 0.0)
    return round(1 - float(np.mean(js)), 4) if js else 0.0


def _spurious_score(motifs: List[str],
                    pairwise: Dict[Tuple, dict],
                    sub_fams: Dict[str, List]) -> float:
    """
    Structural spuriousness. Lower is better (0 = clean).

    mean pairwise J among rule motifs  →  redundant motifs inflate this
    subsuming penalty                  →  general variant in rule inflates this
    """
    if len(motifs) <= 1:
        return 0.0
    js = [pairwise.get((a,b), {}).get('J', 0.0)
          for a, b in combinations(motifs, 2)]
    mean_j = float(np.mean(js)) if js else 0.0
    sub_pen = min(0.2 * sum(
        1 for m in motifs
        for e in sub_fams.get(m, [])
        if e['direction'] == 'general' and e['motif'] in set(motifs)
    ), 1.0)
    return round(0.6 * mean_j + 0.4 * sub_pen, 4)


def _is_related(motif: str,
                existing: Set[str],
                pairwise: Dict[Tuple, dict],
                sub_fams: Dict[str, List],
                min_j: float = 0.10) -> bool:
    """True if motif co-occurs (J >= min_j) with any existing rule motif,
    or is a subsuming (general/specific) variant of any existing rule motif.
    Used to keep greedy extension structurally coherent.
    """
    for em in existing:
        if pairwise.get((em, motif), {}).get('J', 0) >= min_j:
            return True
        if pairwise.get((motif, em), {}).get('J', 0) >= min_j:
            return True
        for entry in sub_fams.get(em, []):
            if entry['motif'] == motif:
                return True
        for entry in sub_fams.get(motif, []):
            if entry['motif'] == em:
                return True
    return False


def _greedy_extend(combined: np.ndarray,
                   current_mset: List[str],
                   clauses: List[dict],
                   cl_vecs: List[np.ndarray],
                   tv_frags: List[List[str]],
                   pairwise: Dict[Tuple, dict],
                   sub_fams: Dict[str, List],
                   N_tv: int,
                   target_pct: float = 50.0,
                   tol: float = 10.0,
                   max_extra: int = 8) -> dict:
    """
    Greedy extension of an existing rule toward target_pct coverage.

    Tolerance default 10.0% — coverage within [40%, 60%] is accepted.

    Extension order (structurally coherent first):
      Pass 1 — consider only motifs that are co-occurring (J >= 0.10 with
                any existing rule motif) or subsuming (general/specific
                variant in sub_fams). These keep the rule on a coherent
                chemical theme and produce interpretable GNN ground truth.
      Pass 2 — if pass 1 cannot reach the tolerance, add any remaining
                k=1 singleton by maximum incremental coverage (fallback).

    At each step picks the candidate with maximum incremental coverage.
    Stops when |pct_match - target_pct| <= tol OR max_extra added.
    """
    used_mset = set(current_mset)
    combined  = combined.copy()
    added_clauses: List[dict] = []

    def _pick_best(allow_unrelated: bool) -> Optional[int]:
        best_delta, best_idx = -1, None
        for i, (c, vec) in enumerate(zip(clauses, cl_vecs)):
            if c['k'] != 1:
                continue
            motif = c['motifs'][0]
            if motif in used_mset:
                continue
            if not allow_unrelated and not _is_related(
                    motif, used_mset, pairwise, sub_fams):
                continue
            delta = int((combined | vec).sum()) - int(combined.sum())
            if delta > best_delta:
                best_delta = delta
                best_idx   = i
        return best_idx if best_delta > 0 else None

    for _ in range(max_extra):
        pct = combined.sum() / N_tv * 100
        if abs(pct - target_pct) <= tol:
            break

        # Pass 1: prefer structurally related motifs
        best_idx = _pick_best(allow_unrelated=False)
        # Pass 2: fall back to any motif if no related candidate available
        if best_idx is None:
            best_idx = _pick_best(allow_unrelated=True)
        if best_idx is None:
            break

        c   = clauses[best_idx]
        vec = cl_vecs[best_idx]
        combined |= vec
        used_mset.add(c['motifs'][0])
        added_clauses.append({'motifs':    c['motifs'],
                              'k':         1,
                              'and_valid': True})

    n_match   = int(combined.sum())
    pct_match = n_match / N_tv * 100
    balance   = 1 - abs(pct_match - 50) / 50
    spurious  = _spurious_score(list(used_mset), pairwise, sub_fams)
    sep       = compute_structural_snr(combined, tv_frags)
    score     = round(balance * sep * (1 - spurious), 5)

    return {
        'combined':      combined,
        'mset':          tuple(sorted(used_mset)),
        'added_clauses': added_clauses,
        'n_match':       n_match,
        'pct_match':     round(pct_match, 1),
        'balance':       round(balance, 3),
        'spurious':      round(spurious, 4),
        'separation':    sep,
        'score':         score,
    }


def find_best_rule(clauses: List[dict],
                   presence: Dict[str, np.ndarray],
                   tv_frags: List[List[str]],
                   pairwise: Dict[Tuple, dict],
                   sub_fams: Dict[str, List],
                   N_tv: int,
                   max_clauses: int = 4,
                   top_k: int = 30,
                   snr_top_k: int = 200) -> Optional[dict]:
    """
    Step 4 — LABEL-FREE.

    Enumerate OR combinations of up to max_clauses conjunctive clauses.

    Scoring:
      balance    = 1 - |pct_match - 50| / 50    (synthetic split quality)
      separation = 1 - mean_J(match, no-match)  (structural SNR, label-free)
      spurious   = mean pairwise J + subsuming   (motif redundancy)

      score = balance × separation × (1 - spurious)

    Two-phase for efficiency:
      Phase 1: all combinations scored by balance × (1-spurious) — fast.
      Phase 2: compute structural separation for top snr_top_k only.
    """
    # Seed clauses: AND-valid first, then by n_match
    top_cl = sorted(clauses,
                    key=lambda c: -(c['n_match'] * (1.2 if c['and_valid'] else 1.0)))[:top_k]

    cl_vecs = []
    for c in top_cl:
        vec = presence[c['motifs'][0]].astype(bool).copy()
        for s in c['motifs'][1:]: vec &= presence[s].astype(bool)
        cl_vecs.append(vec)

    # Phase 1: fast scoring
    phase1: List[dict] = []
    seen:   Set[tuple] = set()

    for n_cl in range(1, max_clauses + 1):
        for idx in combinations(range(len(top_cl)), n_cl):
            combined = cl_vecs[idx[0]].copy()
            for i in idx[1:]: combined |= cl_vecs[i]

            n_match   = int(combined.sum())
            pct_match = n_match / N_tv * 100
            if pct_match < 5 or pct_match > 95:
                continue

            mset = tuple(sorted(set(m for i in idx for m in top_cl[i]['motifs'])))
            if mset in seen: continue
            seen.add(mset)

            balance  = 1 - abs(pct_match - 50) / 50
            spurious = _spurious_score(list(mset), pairwise, sub_fams)
            phase1.append({
                'idx':       idx,
                'mset':      mset,
                'combined':  combined,
                'n_match':   n_match,
                'pct_match': round(pct_match, 1),
                'balance':   round(balance, 3),
                'spurious':  round(spurious, 4),
                'fast_score':balance * (1 - spurious),
                'and_valid': all(top_cl[i]['and_valid'] for i in idx),
                'n_clauses': n_cl,
                'clauses':   [{'motifs':    top_cl[i]['motifs'],
                               'k':         top_cl[i]['k'],
                               'and_valid': top_cl[i]['and_valid']}
                              for i in idx],
            })

    if not phase1:
        return None

    # Phase 2: structural SNR for top candidates
    phase1.sort(key=lambda x: -x['fast_score'])
    for entry in phase1[:snr_top_k]:
        sep = compute_structural_snr(entry['combined'], tv_frags)
        entry['separation'] = sep
        entry['score']      = round(entry['balance'] * sep * (1 - entry['spurious']), 5)
    for entry in phase1[snr_top_k:]:
        entry['separation'] = None
        entry['score']      = 0.0

    phase1.sort(key=lambda x: -x['score'])
    best = phase1[0]

    # Greedy extension: if best rule is imbalanced (balance < 0.9),
    # try adding singleton clauses one at a time toward 50% coverage.
    if best['balance'] < 0.8:  # trigger extension when outside 10% tolerance
        ext = _greedy_extend(
            combined   = best['combined'],
            current_mset = list(best['mset']),
            clauses    = top_cl,
            cl_vecs    = cl_vecs,
            tv_frags   = tv_frags,
            pairwise   = pairwise,
            sub_fams   = sub_fams,
            N_tv       = N_tv,
        )
        if ext['score'] > best['score']:
            # Build full clause list for extended rule
            orig_clauses = best['clauses'][:]
            best = {
                'idx':        best['idx'],
                'mset':       ext['mset'],
                'combined':   ext['combined'],
                'n_match':    ext['n_match'],
                'pct_match':  ext['pct_match'],
                'balance':    ext['balance'],
                'spurious':   ext['spurious'],
                'separation': ext['separation'],
                'score':      ext['score'],
                'and_valid':  best['and_valid'],
                'n_clauses':  best['n_clauses'],
                'clauses':    orig_clauses + ext['added_clauses'],
                'fast_score': best['fast_score'],
            }

    all_rules = [
        {'motifs':     list(e['mset']),
         'n_clauses':  e['n_clauses'],
         'n_match':    e['n_match'],
         'pct_match':  e['pct_match'],
         'balance':    e['balance'],
         'spurious':   e['spurious'],
         'separation': e.get('separation'),
         'score':      e['score'],
         'and_valid':  e['and_valid']}
        for e in phase1[:25]
    ]

    return {
        'motifs':       list(best['mset']),
        'n_clauses':    len(best['clauses']),
        'clauses':      best['clauses'],
        'n_match':      best['n_match'],
        'pct_match':    best['pct_match'],
        'pct_no_match': round(100 - best['pct_match'], 1),
        'balance':      best['balance'],
        'spurious':     best['spurious'],
        'separation':   best['separation'],
        'score':        best['score'],
        'and_valid':    best['and_valid'],
        'all_rules':    all_rules,
    }


def apply_synthetic_labels(best_rule: dict,
                            presence: Dict[str, np.ndarray],
                            N_tv: int) -> Tuple[np.ndarray, np.ndarray]:
    """Step 5 — LABEL-FREE. Rule fires → 1, no-fire → 0."""
    combined = np.zeros(N_tv, dtype=bool)
    for clause in best_rule['clauses']:
        vec = presence[clause['motifs'][0]].astype(bool).copy()
        for s in clause['motifs'][1:]: vec &= presence[s].astype(bool)
        combined |= vec
    return combined.astype(int), combined


def compute_snr(synth_labels: np.ndarray,
                original_labels: np.ndarray,
                combined_mask: np.ndarray,
                N_tv: int) -> dict:
    """
    Step 6 — Post-hoc evaluation using original labels.
    Called AFTER synthetic labels are applied.

    precision = fraction of rule-matching mols whose original label = 1
    snr       = precision / (1 - precision)
    """
    n_match = int(combined_mask.sum())
    if n_match > 0:
        tp   = int((combined_mask & (original_labels == 1)).sum())
        prec = tp / n_match
        snr  = prec / max(1 - prec, 1e-6)
    else:
        tp, prec, snr = 0, 0.0, 0.0

    consistency = int((synth_labels == original_labels).sum()) / N_tv

    return {
        'n_synth_1':   int((synth_labels==1).sum()),
        'n_synth_0':   int((synth_labels==0).sum()),
        'pct_synth_1': round((synth_labels==1).mean()*100, 1),
        'pct_synth_0': round((synth_labels==0).mean()*100, 1),
        'precision':   round(prec, 3),
        'snr':         round(min(snr, 1e6), 2),
        'consistency': round(consistency, 3),
        's1_o1': int(((synth_labels==1)&(original_labels==1)).sum()),
        's1_o0': int(((synth_labels==1)&(original_labels==0)).sum()),
        's0_o1': int(((synth_labels==0)&(original_labels==1)).sum()),
        's0_o0': int(((synth_labels==0)&(original_labels==0)).sum()),
    }


def compute_motif_comparison(selected: List[dict],
                              presence: Dict[str, np.ndarray],
                              tv_frags: List[List[str]],
                              original_labels: np.ndarray,
                              synth_labels: np.ndarray) -> List[dict]:
    """Per-motif: structural SNR + label-based comparison (post-hoc)."""
    rows = []
    for c in selected:
        s   = c['s']
        if s not in presence: continue
        vec = presence[s].astype(bool)
        n   = int(vec.sum())
        if n == 0: continue

        struct_snr = compute_structural_snr(vec, tv_frags, n_sample=40)

        row = {'s': s, 'sup': c['sup'], 'n_atoms': c['n_atoms'],
               'sel_score': c['sel_score'], 'struct_snr': struct_snr}
        for label_arr, pfx in [(original_labels,'orig'),(synth_labels,'synth')]:
            n1  = int((vec & (label_arr==1)).sum())
            c1p = n1/n*100
            prec = max(c1p, 100-c1p)/100
            lsnr = prec/max(1-prec,1e-6)
            row[f'{pfx}_c1']   = round(c1p, 1)
            row[f'{pfx}_snr']  = round(lsnr, 2)
        row['snr_delta'] = round(row['synth_snr'] - row['orig_snr'], 2)
        rows.append(row)
    rows.sort(key=lambda x: -x['sel_score'])
    return rows


def run_phase3(raw_stats: dict,
               tv_frags: List,
               tv_labels: np.ndarray,
               N_tv: int,
               method: str,
               min_sup_pct: float = 1.0,
               max_n: int = 25,
               max_clauses: int = 4) -> dict:
    """
    Full Phase 3 pipeline.
    Steps 1-5: completely label-free.
    Step 6: tv_labels used only for post-hoc SNR evaluation.
    """
    selected, presence = select_top_motifs(
        raw_stats, tv_frags, N_tv,
        min_sup_pct=min_sup_pct, max_n=max_n)

    if not selected:
        return {'error': 'no candidates above threshold'}

    pairwise, sub_fams = build_metadata_structural(selected, presence)
    clauses = build_clauses_structural(selected, presence, pairwise, N_tv)

    best_rule = find_best_rule(
        clauses, presence, tv_frags, pairwise, sub_fams, N_tv,
        max_clauses=max_clauses)

    if best_rule is None:
        return {'selected': selected, 'error': 'no valid rule found',
                'n_clauses': len(clauses)}

    synth_labels, combined_mask = apply_synthetic_labels(best_rule, presence, N_tv)
    snr_metrics  = compute_snr(synth_labels, tv_labels, combined_mask, N_tv)
    motif_comp   = compute_motif_comparison(
        selected, presence, tv_frags, tv_labels, synth_labels)

    return {
        'selected':          [{'s':c['s'],'sup':c['sup'],'n':c['n'],
                               'n_atoms':c['n_atoms'],'sel_score':c['sel_score']}
                              for c in selected],
        'n_clauses':         len(clauses),
        'n_and_valid':       sum(1 for c in clauses if c['and_valid']),
        'best_rule':         {k:v for k,v in best_rule.items()
                              if k not in ('clauses','all_rules')},
        'best_rule_clauses': best_rule.get('clauses', []),
        'all_rules':         best_rule.get('all_rules', []),
        'snr_metrics':       snr_metrics,
        'motif_comp':        motif_comp,
        'synth_labels':      synth_labels.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH BRIDGE API
# Used by SharedModules/data/ground_truth.py
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_rulebook(
    data_root: str,
    dataset_name: str,
    fold: int = 0,
) -> dict:
    """Load the motif-presence matrix and row metadata from the vocab output.

    Reads:
        {data_root}/{dataset_name}_fold{fold}/graph_motif_matrix.npz
        {data_root}/{dataset_name}_fold{fold}/graph_motif_matrix_columns.csv
        {data_root}/{dataset_name}_fold{fold}/graph_motif_matrix_rows.csv

    Returns a rulebook dict with keys:
        row_smiles       list[str]          SMILES string per row
        row_motif_sets   list[set[str]]     set of motif SMARTS for each row
        columns          list[str]          motif_identity per column
        matrix           scipy.sparse       binary presence matrix
    """
    import scipy.sparse as _sp
    import pandas as _pd
    from pathlib import Path as _P

    base = _P(data_root) / f'{dataset_name}_fold{fold}'
    npz  = _sp.load_npz(str(base / 'graph_motif_matrix.npz'))
    cols = _pd.read_csv(base / 'graph_motif_matrix_columns.csv')
    rows = _pd.read_csv(base / 'graph_motif_matrix_rows.csv')

    motif_ids = cols['motif_identity'].tolist()
    smiles    = rows['smiles'].tolist()

    # Build per-row motif sets from the sparse matrix
    npz_csr = npz.tocsr()
    row_motif_sets = []
    for i in range(npz_csr.shape[0]):
        col_idxs = npz_csr.getrow(i).nonzero()[1]
        row_motif_sets.append({motif_ids[j] for j in col_idxs})

    return {
        'row_smiles':    smiles,
        'row_motif_sets': row_motif_sets,
        'columns':       motif_ids,
        'matrix':        npz,
    }


def choose_rule_interactive(
    rulebook: dict,
    selected_index: Optional[int] = None,
    interactive: bool = False,
    rules: Optional[List[dict]] = None,
) -> dict:
    """Select a rule for GT annotation.

    If ``selected_index`` is provided, returns a dummy rule that fires when
    a molecule contains the motif at that column index.

    For the full rule-based pipeline, pass ``rules`` (from rules.json) and
    an index.  The returned dict must have:
        motifs      list[str]  SMARTS strings that must ALL be present
    """
    if rules is not None and selected_index is not None:
        if 0 <= selected_index < len(rules):
            r = rules[selected_index]
            # Handle both legacy DNF format (clauses list) and Phase3 format
            if 'clauses' in r:
                motifs = list({m for cl in r['clauses'] for m in cl.get('motifs', [])})
            elif 'motifs' in r:
                motifs = list(r['motifs'])
            else:
                motifs = []
            return {'motifs': motifs, 'rule_index': selected_index}

    if selected_index is not None:
        cols = rulebook.get('columns', [])
        if 0 <= selected_index < len(cols):
            return {'motifs': [cols[selected_index]], 'rule_index': selected_index}

    if interactive:
        # Print first few columns for inspection
        cols = rulebook.get('columns', [])
        print('Available motifs (first 20):')
        for i, m in enumerate(cols[:20]):
            cnt = int(rulebook['matrix'].getcol(i).nnz)
            print(f'  [{i:3d}] {m}  ({cnt} graphs)')
        idx = int(input('Enter motif index for GT rule: '))
        return {'motifs': [cols[idx]], 'rule_index': idx}

    # Default: return first column motif
    cols = rulebook.get('columns', [])
    return {'motifs': [cols[0]] if cols else [], 'rule_index': 0}


def evaluate_rule_on_motifs(
    present: set,
    rule: dict,
) -> tuple:
    """Evaluate whether a set of motif SMARTS fires the rule.

    Rule fires (→ class 1) when ALL motifs in rule['motifs'] are present.
    Returns (rule_positive: bool, active_motifs: set[str]).
    """
    motifs = rule.get('motifs', [])
    if not motifs:
        return False, set()
    active = set(motifs) & present
    rule_positive = all(m in present for m in motifs)
    return rule_positive, active


def save_rulebook_json(
    rulebook: dict,
    selected_rule: dict,
    path,
) -> None:
    """Save the selected rule to a JSON file for reproducibility."""
    import json as _json
    from pathlib import Path as _P
    out = {
        'motifs':     selected_rule.get('motifs', []),
        'rule_index': selected_rule.get('rule_index', -1),
    }
    _P(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as _f:
        _json.dump(out, _f, indent=2)
