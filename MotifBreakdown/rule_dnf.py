"""rule_dnf.py — DNF planted-rule engine via frequent-itemset mining.

** UPDATED 2026-08 — the engine is now `sample_dnf` (UNIFORM sampling from the frequent-itemset
   pool). The ONLY generation constraints are the three agreed ones: support floor (estimability),
   coverage band (class balance, at the rule level), and per-disjunct non-redundancy. Difficulty
   (overlap o, spread s, correlate r, LR-learnability auc_lr, annotation size, clause connectivity)
   is RECORDED per rule and stratified downstream, never enforced. The greedy anchor-growth engine
   (Psi overlap penalty, comparable_cov arm, structural single-atom gate, per-clause prevalence cap)
   is RETIRED as `_select_dnf_greedy_retired`; `select_dnf` is a back-compat shim → `sample_dnf`. **

The minimal, defensible rule engine agreed with the advisor. Governing principle
(same as rule_grid): plant a SMALL, TRANSPARENT set of causes that are EXPRESSIBLE and
LEARNABLE; DIFFICULTY is MEASURED downstream (phase-5 GT-ROC), never built in.

What changed vs rule_grid.py: the AND clauses are DISCOVERED by frequent-itemset mining
(Apriori) instead of enumerated, and the rules are genuine DNF, drawn by UNIFORM sampling.

Procedure. There are TWO acceptance tests (balance, non-redundancy). The support floor tau is
NOT an acceptance test — it is the MINING parameter that defines the Apriori pool, i.e. the
population that "uniform sampling" is uniform over; it is enforced once at pool construction and
is logically implied by non-redundancy anyway (new-positives(S) <= supp(S)). Identifiability is
COMPUTED and RECORDED but NOT gated (see step 5).

  1. TRANSACTIONS — each molecule = its set of motif keys (market-basket). No atom sets
     needed for mining (apply_gt marks GT atoms downstream from the keys).
  2. FREQUENT ITEMSETS — Apriori at support >= tau (absolute count = estimability).
     Gives conjunctions of ANY arity that actually occur; support self-limits the depth.
  3. ADMISSIBLE CLAUSE POOL — frequent itemsets of arity 1..a_max with support <= cov_hi*N
     (a clause too prevalent to sit in an in-band rule is pruned). Sorted deterministically so
     sampling is reproducible from `seed` alone. THIS is where tau acts.
  4. SAMPLE — draw k ~ Uniform{1..k_max} DISJUNCTS, then k distinct clauses uniformly from the
     pool; the OR of them is a candidate DNF. Rejection-sample until n_rules pass BOTH tests:
       (1) BALANCE          — coverage cov = |F|/N in [cov_lo, cov_hi] (non-degenerate classes).
       (2) NON-REDUNDANCY   — each disjunct adds >= tau NEW positives (a real added cause).
  5. RECORD (never select on): the acceptance VALUES, the per-clause support, and — as of the
     2026-08 revision — IDENTIFIABILITY. Every non-rule motif with |corr| >= spur_corr is a
     candidate confounder, split by sign into spurious_pos (class-1) / spurious_neg (class-0),
     and identifiability_sep = min over class-1 correlates of |F \ supp(m)| is stored. sep == 0
     means the cause is perfectly shadowed: an UNIDENTIFIABLE rule is now ADMITTED ON PURPOSE,
     because which of the two observationally-identical motifs a model attends to is an empirical
     fact about inductive bias and is exactly what the GT-ROC vs spurious-ROC contrast measures.
     Read GT-ROC in the low-sep stratum as PREFERENCE, not accuracy, and stratify tables by
     identifiability_sep. Also recorded: difficulty descriptors (overlap o, spread s, correlate r,
     LR-learnability auc_lr, annotation size, clause connectivity), structural look-alikes
     (family_motifs, size-gated), and foolability.

Output is apply_gt-compatible, keyed by 'dnf_k<k>_r<i>' where k is the number of DISJUNCTS
(conjunct width is capped separately by a_max and does NOT appear in the key) and i indexes the
rules sampled at that k. Tunables: min_sup_frac (2% of N, the estimability floor), [cov_lo,
cov_hi] (the coverage band), a_max (max clause arity), MAX_K (declared form axis), spur_corr
(the correlate-reporting threshold — reporting only, no longer a gate). Apriori depth is
uncapped — deeper conjunctions die off below tau on their own, so there is no MAX_LEN knob.

The greedy anchor-growth engine (Psi overlap penalty, comparable_cov arm, structural single-atom
gate, per-clause prevalence cap) is RETIRED as `_select_dnf_greedy_retired`; `select_dnf` is a
back-compat shim -> `sample_dnf`.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from rdkit import Chem

from rule_tiers import _foolability            # shared shortcut-report (single source of truth)


def _strip(key: str) -> str:
    """Bare SMILES/SMARTS of a motif key (drop the ring:/chain:/frag: prefix)."""
    return key.split(':', 1)[1] if key.startswith(('ring:', 'chain:', 'frag:')) else key


def _is_functional(key: str, mol) -> bool:
    """Whether a motif is worth a FAMILY: a ring, or bearing a heteroatom / multiple bond. Excludes
    saturated-carbon skeletons (*C, *CC, *CCC*) whose containment 'family' is degenerate (a bare
    carbon is a substructure of ~everything: measured *C -> 483/574 motifs)."""
    if key.startswith('ring:'):
        return True
    if mol is None:
        return False
    return (any(a.GetAtomicNum() not in (0, 1, 6) for a in mol.GetAtoms())        # heteroatom
            or any(b.GetBondType() != Chem.BondType.SINGLE for b in mol.GetBonds()))  # unsaturation

MIN_SUP_FRAC = 0.02     # THE knob: an itemset is frequent if it is in >= this FRACTION of molecules
                        # (scales with dataset size). support(itemset) = # transactions containing it.
MIN_SUP_FLOOR = 20      # estimability safety rail: never require fewer than this many molecules, so a
                        # small dataset (where 2% is a handful) still has enough per class to train.
                        # effective min_support = max(MIN_SUP_FLOOR, round(MIN_SUP_FRAC * N)).
MAX_K = 3               # declared largest DNF arity (k=1..MAX_K). A FORM choice, not a sweep.


def _apriori(txns: List[frozenset], min_sup: int) -> Dict[frozenset, Set[int]]:
    """Frequent itemsets (>= min_sup transactions). Returns {itemset: support-set}.
    No depth cap: downward closure guarantees support(size k+1) <= support(size k), so deeper
    conjunctions die off below min_sup on their own and the level loop terminates naturally
    (bounded by the largest transaction). This is why no MAX_LEN knob is needed."""
    n = len(txns)
    items: Dict[str, Set[int]] = {}
    for i, t in enumerate(txns):
        for k in t:
            items.setdefault(k, set()).add(i)
    freq: Dict[frozenset, Set[int]] = {frozenset([k]): s for k, s in items.items()
                                       if len(s) >= min_sup}
    level = dict(freq)
    size = 2
    while level:                                     # stops when no k-itemset reaches min_sup
        prev = set(level)
        cand: Set[frozenset] = set()
        for a, b in combinations(prev, 2):           # join step
            u = a | b
            if len(u) == size and all(frozenset(s) in prev for s in combinations(u, size - 1)):
                cand.add(u)
        nxt: Dict[frozenset, Set[int]] = {}
        for c in cand:                               # count via bitset intersection
            sup = set.intersection(*[items[x] for x in c])
            if len(sup) >= min_sup:
                nxt[c] = sup
        freq.update(nxt)
        level = nxt
        size += 1
    return freq


def _maximal(freq: Dict[frozenset, Set[int]]) -> List[frozenset]:
    """Maximal frequent itemsets: not a strict subset of any other frequent itemset."""
    keys = list(freq)
    return [s for s in keys if not any(s < o for o in keys)]


def sample_dnf(smiles: List[str],
               mol_frags: List[List[Tuple[str, Set[int]]]],
               *,
               min_sup_frac: float = MIN_SUP_FRAC,
               min_sup_floor: int = MIN_SUP_FLOOR,
               cov_lo: float = 0.10,
               cov_hi: float = 0.50,
               spur_corr: float = 0.5,
               a_max: int = 2,
               k_max: int = MAX_K,
               n_rules: int = 50,
               seed: int = 0,
               max_tries: int = 200_000,
               fam_atom_tol: int = 2,
               report_foolability: bool = True,
               report_family: bool = True,
               log: Callable[[str], None] = print) -> Dict[str, dict]:
    """Sample planted DNF rules by UNIFORM sampling from the frequent-itemset pool.

    Supersedes the greedy anchor-growth of select_dnf. There are TWO acceptance tests:
      (1) BALANCE       — rule coverage cov = |F|/N in [cov_lo, cov_hi] (non-degenerate classes).
      (2) NON-REDUNDANCY— each disjunct adds >= tau NEW positives (a real added cause).
    SUPPORT (each clause supp(S) >= tau) is guaranteed upstream by the Apriori pool and is implied
    by non-redundancy; it is recorded (support_min_clause), never re-tested here.
    IDENTIFIABILITY is COMPUTED AND RECORDED BUT NOT GATED (changed 2026-08). EVERY non-rule motif
    with |corr| >= spur_corr is a candidate confounder, split by sign: class-1 markers (present in
    positives) compete with the cause for attribution; class-0 markers (present in negatives) are
    prediction shortcuts. identifiability_sep = min_{m in spurious_pos} |F \ supp(m)| is stored.
    A rule with sep == 0 (cause perfectly shadowed) is ADMITTED ON PURPOSE: which of two
    observationally-identical motifs a model attends to is a fact about inductive bias, and the
    GT-ROC vs spurious-ROC contrast is precisely the instrument that measures it. Downstream,
    stratify by identifiability_sep and read low-sep GT-ROC as PREFERENCE, not accuracy.
    No Psi overlap penalty, no anchor selection, no structural single-atom gate, no per-clause
    prevalence cap (the coverage band subsumes it: supp(S_c) <= |F| <= cov_hi*N). Difficulty is
    MEASURED per rule (overlap o, spread s, correlate r, LR-learnability auc_lr, annotation size,
    clause connectivity) and stratified downstream, never enforced.

    Returns {'dnf_k<k>_r<i>': apply_gt-rule}, contract-compatible with select_dnf.
    """
    import random
    n = len(smiles)
    if any(not mf for mf in mol_frags):
        raise ValueError("sample_dnf: motif-less molecule present — drop unparseable SMILES upstream "
                         "so smiles/mol_frags stay aligned (an empty row inflates the negatives).")
    motset = [{k for k, _ in mf} for mf in mol_frags]
    txns = [frozenset(s) for s in motset]
    natoms: Dict[str, int] = {}
    atomsets: List[Dict[str, Set[int]]] = []                       # per-molecule motif-key -> atom set
    for mf in mol_frags:
        d: Dict[str, Set[int]] = {}
        for k, ats in mf:
            natoms.setdefault(k, len(ats)); d.setdefault(k, set()).update(ats)
        atomsets.append(d)
    min_sup = max(min_sup_floor, round(min_sup_frac * n))
    freq = _apriori(txns, min_sup)
    # admissible clause pool: frequent itemsets of arity 1..a_max, support <= cov_hi*N. The upper
    # cap only PRUNES clauses that could never sit in an in-band rule (balance is enforced once, at
    # the rule level, by the coverage band) — it is not a separate discriminativeness gate.
    pool = [(s, freq[s]) for s in freq if 1 <= len(s) <= a_max and len(freq[s]) <= cov_hi * n]
    # deterministic order: freq is a dict of frozenset keys whose iteration order varies with the
    # per-process string-hash seed; sort so rng.sample() draws reproducibly across runs (seed alone
    # is not enough without this). Key = (arity, sorted motif keys) — a total order on clauses.
    pool.sort(key=lambda sf: (len(sf[0]), tuple(sorted(sf[0]))))
    if not pool:
        log("    [sample_dnf] empty admissible pool at this support"); return {}
    cand = sorted(set().union(*motset)) if motset else []
    pres = {k: np.array([k in motset[i] for i in range(n)]) for k in cand}
    mols = [Chem.MolFromSmiles(s) for s in smiles]                 # for connectivity
    _vmol = {k: Chem.MolFromSmiles(_strip(k)) for k in cand}
    _vq = {k: Chem.MolFromSmarts(_strip(k)) for k in cand}
    _fam_cache: Dict[str, list] = {}

    def _family(M: str) -> list:
        """Structural look-alikes of motif M: motifs in a substructure relationship (either
        direction) AND of COMPARABLE size (|atoms| within fam_atom_tol of M). The size gate keeps a
        family to genuine confusables and drops arbitrary supersets — e.g. without it a carbonyl
        pulls every large fragment that merely CONTAINS a C=O (measured: 500+ on BBBP rbrics)."""
        if M in _fam_cache:
            return _fam_cache[M]
        qM, mM = _vq.get(M), _vmol.get(M); tgt = natoms.get(M, 0); fam = []
        for Mp in cand:
            if Mp == M:
                continue
            if abs(natoms.get(Mp, 0) - tgt) > fam_atom_tol:    # comparable size only (genuine look-alike)
                continue
            mp, qp = _vmol.get(Mp), _vq.get(Mp)
            try:
                if (qM is not None and mp is not None and mp.HasSubstructMatch(qM)) or \
                   (mM is not None and qp is not None and mM.HasSubstructMatch(qp)):
                    fam.append(Mp)
            except Exception:
                pass
        _fam_cache[M] = fam; return fam

    def _connected(gi: int, clause) -> Optional[bool]:
        """Is the induced subgraph on a clause's atoms connected in molecule gi? (|clause|<2 -> True)."""
        if len(clause) < 2:
            return True
        A = set().union(*[atomsets[gi].get(m, set()) for m in clause])
        if len(A) <= 1:
            return True
        mol = mols[gi]
        if mol is None or any(a >= mol.GetNumAtoms() for a in A):
            return None                                            # can't align atoms -> undefined
        adj = {a: set() for a in A}
        for b in mol.GetBonds():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if u in A and v in A:
                adj[u].add(v); adj[v].add(u)
        start = next(iter(A)); seen = {start}; stack = [start]
        while stack:
            x = stack.pop()
            for y2 in adj[x]:
                if y2 not in seen:
                    seen.add(y2); stack.append(y2)
        return len(seen) == len(A)

    def _auc_lr(rule_motifs: Set[str], y: np.ndarray) -> float:
        """Data-side learnability: 3-fold CV AUC of LR on NON-rule motif indicators -> y_rho
        (the multivariate shortcut analogue of r; rule motifs excluded so it isn't trivially 1)."""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            feats = [m for m in cand if m not in rule_motifs]
            if not feats or int(y.sum()) < 3 or int((~y).sum()) < 3:
                return float('nan')
            X = np.stack([pres[m] for m in feats], axis=1).astype(float)
            return float(np.mean(cross_val_score(
                LogisticRegression(max_iter=500), X, y.astype(int), cv=3, scoring='roc_auc')))
        except Exception:
            return float('nan')

    rng = random.Random(seed)
    jc = lambda a, b: len(a & b) / max(len(a | b), 1)
    out: Dict[str, dict] = {}
    seen_sig: Set[frozenset] = set(); per_k: Dict[int, int] = {}; tries = 0
    while len(out) < n_rules and tries < max_tries:
        tries += 1
        k = rng.randint(1, k_max)
        if len(pool) < k:
            continue
        picks = rng.sample(pool, k)
        clauses = [s for s, _ in picks]; sets = [ss for _, ss in picks]
        sig = frozenset(clauses)
        if sig in seen_sig:
            continue
        F = set().union(*sets); cov = len(F) / n
        # ── ACCEPTANCE: two tests, plus values COMPUTED here and RECORDED below ──
        # (1) BALANCE: rule coverage in band.
        if not (cov_lo <= cov <= cov_hi):
            continue
        # SUPPORT: every clause cleared the floor tau when the pool was built; not re-tested (it is
        # also implied by non-redundancy, since new-positives(S) <= supp(S)). Kept for the record.
        min_clause_support = min(len(ss) for ss in sets)
        # (2) NON-REDUNDANCY: every disjunct adds >= tau positives no other disjunct covers.
        min_new_positives = min(
            len(sets[i] - (set().union(*[sets[j] for j in range(k) if j != i]) if k > 1 else set()))
            for i in range(k))
        if min_new_positives < min_sup:
            continue
        # IDENTIFIABILITY (RECORDED, NOT GATED — 2026-08): collect ALL non-rule motifs with
        # |corr(x_m, y)| >= spur_corr, split by sign. class-1 (corr>0, present in positives) compete
        # with the cause for attribution and become the per-motif spurious-ROC targets downstream;
        # class-0 (corr<0, present in negatives) are prediction shortcuts. NO REJECTION happens on
        # this — a perfectly-shadowed cause (sep == 0) is an admissible and deliberately interesting
        # rule; stratify on identifiability_sep at reporting time instead.
        rule_motifs = {m for c in clauses for m in c}
        y = np.zeros(n, bool)
        for idx in F:
            y[idx] = True
        fidx = np.fromiter(F, int, len(F))
        yf = y.astype(float); rr = 0.0
        spur_pos: List[tuple] = []; spur_neg: List[tuple] = []
        best_pos: Optional[tuple] = None      # strongest POSITIVE correlate, whatever its value
        for m in cand:
            if m in rule_motifs:
                continue
            xm = pres[m].astype(float)
            if xm.std() <= 0 or yf.std() <= 0:
                continue
            cc = float(np.corrcoef(xm, yf)[0, 1])
            rr = max(rr, abs(cc))
            if cc > 0 and (best_pos is None or cc > best_pos[1]):
                best_pos = (m, round(cc, 4), int((~pres[m][fidx]).sum()), int(pres[m].sum()))
            if cc >= spur_corr:
                # (motif, corr, sep=|F\supp(m)|, supp=corpus support). supp is recorded so a
                # low-support correlate — whose per-motif spurious-ROC would rest on a handful of
                # graphs — can be filtered at REPORTING time rather than gated at generation time.
                spur_pos.append((m, round(cc, 4), int((~pres[m][fidx]).sum()), int(pres[m].sum())))
            elif cc <= -spur_corr:
                spur_neg.append((m, round(cc, 4)))
        # separability = worst class-1 correlate's |F \ supp(m)|; no strong class-1 correlate =>
        # identifiable. RECORDED ONLY — there is deliberately no `if separability < min_sup` gate.
        # COMPUTED BEFORE the audit fallback below, and therefore over the >= spur_corr entries
        # ONLY: identifiability is a claim about STRONG confounders, so a weak top-1 correlate
        # must not be allowed to drag sep down and corrupt the stratification axis.
        separability = min((s for _, _, s, _ in spur_pos), default=len(F))
        # AUDIT FALLBACK: keep every correlate at >= spur_corr (all of them, not just the top),
        # but when NONE reaches the threshold fall back to the single strongest positive correlate
        # so the shortcut audit is never empty. Measured: at spur_corr=0.5, 43/50 BBBP rules and
        # 50/50 Mutagenicity rules have no qualifying correlate, so without this the spurious axis
        # simply vanishes on most rules. Fallback entries are FLAGGED — their corr is below
        # threshold, so a high spurious-ROC against one is not evidence of shortcut-following.
        if not spur_pos and best_pos is not None:
            spur_pos = [best_pos]
            _fallback = {best_pos[0]}
        else:
            _fallback = set()
        seen_sig.add(sig)
        spur_pos.sort(key=lambda t: -t[1]); spur_neg.sort(key=lambda t: t[1])
        # ---- difficulty descriptors (recorded, never gated) ----
        o = max((jc(sets[i], sets[j]) for i, j in combinations(range(k), 2)), default=0.0)
        supp = [len(ss) for ss in sets]; s_spread = max(supp) / max(min(supp), 1)
        auc_lr = _auc_lr(rule_motifs, y)
        ann = []                                                   # median #GT atoms per positive graph
        for gi in F:
            gt: Set[int] = set()
            for c, ss in zip(clauses, sets):
                if gi in ss:
                    for m in c:
                        gt |= atomsets[gi].get(m, set())
            ann.append(len(gt))
        ann_size = float(np.median(ann)) if ann else 0.0
        conn_vals = []                                             # connectivity over multi-motif clauses
        for c, ss in zip(clauses, sets):
            if len(c) < 2:
                continue
            got = [g for g in (_connected(gi, c) for gi in ss) if g is not None]
            if got:
                conn_vals.append(float(np.mean(got)))
        connectivity = float(np.mean(conn_vals)) if conn_vals else float('nan')
        kk = len(clauses); ri = per_k.get(kk, 0) + 1; per_k[kk] = ri; key = f'dnf_k{kk}_r{ri}'
        rec = dict(
            clauses=[{'motifs': list(c)} for c in clauses],
            rule_str=' ∨ '.join('(' + ' ∧ '.join(sorted(c)) + ')' for c in clauses),
            form=key, k=kk, cov=round(cov, 4), n_pos=int(y.sum()), n_neg=int((~y).sum()),
            n_atoms=sum(natoms.get(m, 1) for c in clauses for m in c),
            # ── the four acceptance-constraint VALUES (each rule records why it was admitted) ──
            #   (1) BALANCE = cov (= |F|/N, above)
            support_min_clause=int(min_clause_support),          # pool-guaranteed: min clause support (>= tau)
            nonredundancy_min_new_pos=int(min_new_positives),    # (2) NON-REDUNDANCY: min new positives (>= tau)
            identifiability_sep=int(separability),               # RECORDED, NOT GATED: worst class-1 |F\supp(m)|
            # class-1 correlates: the per-motif spurious-ROC targets (apply_gt marks one
            # node_label_spurious column per entry, IN THIS ORDER — corr-descending).
            spurious_pos=[{'motif': m, 'corr': c, 'sep': s, 'supp': n_m,
                           'fallback': m in _fallback}
                          for m, c, s, n_m in spur_pos],
            spurious_neg=[{'motif': m, 'corr': c} for m, c in spur_neg],               # class-0 correlates (neg SpurROC)
            # (no singular `spurious_motif` field: with the audit fallback, spurious_pos[0] can
            #  be a SUB-THRESHOLD fallback correlate, so a field named "the spurious motif" would
            #  hand out a <spur_corr motif to any later reader. Use spurious_pos[0], which carries
            #  its corr and fallback flag.)
            # ── difficulty descriptors (recorded, never gated) ──
            overlap_o=round(float(o), 4), spread_s=round(float(s_spread), 4),
            correlate_r=round(float(rr), 4),
            auc_lr=(round(float(auc_lr), 4) if auc_lr == auc_lr else None),
            ann_size=round(ann_size, 2),
            connectivity=(round(float(connectivity), 4) if connectivity == connectivity else None),
            grader='dnf_sampled',
            params=dict(min_support=min_sup, cov_lo=cov_lo, cov_hi=cov_hi, spur_corr=spur_corr,
                        a_max=a_max, k_max=k_max, n_rules=n_rules, seed=seed, fam_atom_tol=fam_atom_tol),
        )
        if report_foolability:
            rec['foolability'] = _foolability([tuple(c) for c in clauses], cand, pres, motset, n)
        if report_family:
            fam: Set[str] = set()
            for m in rule_motifs:
                if _is_functional(m, _vmol.get(m)):
                    fam.update(_family(m))
            rec['family_motifs'] = sorted(fam - rule_motifs)
        out[key] = rec
        log(f"    [sample_dnf] {key}: cov={cov:.2f} k={kk} o={o:.2f} s={s_spread:.1f} r={rr:.2f} "
            f"annsz={ann_size:.0f} conn={rec['connectivity']} :: {rec['rule_str'][:44]}")
    log(f"    [sample_dnf] sampled {len(out)}/{n_rules} rules in {tries} tries; "
        f"by k={ {kk: per_k[kk] for kk in sorted(per_k)} }")
    return out


def select_dnf(smiles: List[str],
               mol_frags: List[List[Tuple[str, Set[int]]]],
               *,
               n_rules: Optional[int] = 50,
               max_k: int = MAX_K,
               min_sup_frac: float = MIN_SUP_FRAC,
               min_sup_floor: int = MIN_SUP_FLOOR,
               report_foolability: bool = True,
               report_family: bool = True,
               log: Callable[[str], None] = print,
               **_retired) -> Dict[str, dict]:
    """Back-compat shim → delegates to sample_dnf (UNIFORM sampling). The greedy anchor-growth
    engine (_select_dnf_greedy_retired) is RETIRED: its Psi overlap penalty, comparable_cov arm,
    structural single-atom gate and per-clause prevalence cap were 'nice-rule' heuristics, not
    anti-confound constraints. Retired kwargs (comparable_cov, gamma, max_clause_frac, min_k) are
    accepted and ignored. Same (smiles, mol_frags) contract; returns {'dnf_k<k>_r<i>': apply_gt-rule}."""
    return sample_dnf(smiles, mol_frags, min_sup_frac=min_sup_frac, min_sup_floor=min_sup_floor,
                      k_max=max_k, n_rules=(n_rules if n_rules else 50),
                      report_foolability=report_foolability, report_family=report_family, log=log)


def _select_dnf_greedy_retired(smiles: List[str],
               mol_frags: List[List[Tuple[str, Set[int]]]],
               *,
               min_sup_frac: float = MIN_SUP_FRAC,
               min_sup_floor: int = MIN_SUP_FLOOR,
               max_clause_frac: float = 0.30,   # RETIRED-engine literals (constants removed; fn raises anyway)
               max_k: int = MAX_K,
               min_k: int = 2,
               n_rules: Optional[int] = 20,
               comparable_cov: bool = True,   # True: near-disjoint + comparable-coverage complements
               gamma: float = 1.0,            #   -> DISSIMILAR/varied rules. False: min-overlap only ->
                                              #   SIMILAR/correlated rules (shared rare complement). Both
                                              #   valid: the flag controls rule-set diversity vs stability.
               report_foolability: bool = True,
               report_family: bool = True,
               log: Callable[[str], None] = print) -> Dict[str, dict]:
    """RETIRED (2026-08) — superseded by sample_dnf (uniform sampling; difficulty MEASURED, not
    enforced). NOT called by the pipeline anymore (select_dnf now delegates to sample_dnf); kept
    intact for reference. Greedy anchor-growth: mine maximal frequent itemsets, gate by the
    structural/discriminative admissibility, pick the top-N anchors by coverage, grow each by
    Score = MarginalCoverage - gamma*Overlap (+comparable_cov size-match), snapshot k=min_k..max_k."""
    raise NotImplementedError(
        "RETIRED: greedy anchor-growth DNF selection is disabled — rules are now UNIFORMLY sampled "
        "by rule_dnf.sample_dnf (difficulty measured downstream, not enforced). The dead body below "
        "is kept only for reference; do not call this function.")
    # ── everything below is UNREACHABLE (RETIRED) ─────────────────────────────────────────────
    # CONTRACT: unparseable molecules must be dropped by the caller (data-load), never passed in
    # as empty rows. An empty row has no motifs, so keeping it would silently inflate the negative
    # class and dilute every coverage denominator (n) with noise. Enforce it loudly rather than
    # limp: a molecule with no motifs is a data-cleaning error upstream, not a valid negative.
#     n_empty = sum(1 for mf in mol_frags if not mf)
#     if n_empty:
#         raise ValueError(
#             f"select_dnf received {n_empty} empty (motif-less) molecules — drop unparseable "
#             f"SMILES at data load so smiles/labels/mol_frags stay aligned, then re-call.")
#     n = len(smiles)
#     motset = [{k for k, _ in mf} for mf in mol_frags]
#     txns = [frozenset(s) for s in motset]                # no empties now; n == len(txns)
#     natoms: Dict[str, int] = {}
#     for mf in mol_frags:
#         for k, ats in mf:
#             natoms.setdefault(k, len(ats))

    # min_support as a FRACTION of the dataset (scales with N), floored for estimability
#     min_sup = max(min_sup_floor, round(min_sup_frac * n))
#     freq = _apriori(txns, min_sup)
#     maximal = _maximal(freq)
    # foolability distractor set = ALL motif keys in the corpus (widened from frequent-singles-only
    # for a STRICTER shortcut audit). _foolability excludes the rule's own motifs internally, so the
    # probe asks "can the cause be predicted from EVERY other motif, incl. rare ones". Trade-off:
    # rare motifs are sparse columns (mild overfit/noise risk) but this no longer UNDER-estimates
    # foolability by missing rare-motif shortcuts.
#     cand = sorted(set().union(*motset)) if motset else []
#     pres = {k: np.array([k in motset[i] for i in range(n)]) for k in cand}

    # FAMILY motifs: for a causal (clause) motif M, the vocab motifs structurally CONTAINING M or
    # CONTAINED IN M (bidirectional, size-agnostic) — the minimal-FG <-> maximal-FG family the MDL
    # merge grows (carbonyl <-> acid/ester/amide). Marks the "right chemistry, wrong granularity"
    # attribution class, distinct from spurious (unrelated). Computed only for FUNCTIONAL clause
    # motifs (rings / heteroatom / unsaturation); saturated-carbon seeds are degenerate. Cached.
#     _vmol = {k: Chem.MolFromSmiles(_strip(k)) for k in cand}
#     _vq = {k: Chem.MolFromSmarts(_strip(k)) for k in cand}
#     _fam_cache: Dict[str, list] = {}

#     def _family(M: str) -> list:
#         if M in _fam_cache:
#             return _fam_cache[M]
#         qM, mM = _vq.get(M), _vmol.get(M)
#         fam = []
#         for Mp in cand:
#             if Mp == M:
#                 continue
#             mp, qp = _vmol.get(Mp), _vq.get(Mp)
#             try:
#                 if (qM is not None and mp is not None and mp.HasSubstructMatch(qM)) or \
#                    (mM is not None and qp is not None and mM.HasSubstructMatch(qp)):
#                     fam.append(Mp)
#             except Exception:
#                 pass
#         _fam_cache[M] = fam
#         return fam
#     _maxsz = max((len(s) for s in freq), default=0)
#     log(f"    [dnf] transactions={len(txns)}  min_support={min_sup} "
#         f"(={min_sup_frac:.1%} of N, floor {min_sup_floor})  frequent itemsets={len(freq)} (by size "
#         f"{ {sz: sum(1 for s in freq if len(s)==sz) for sz in range(1, _maxsz + 1)} })  "
#         f"maximal={len(maximal)}  distractors={len(cand)} (all motifs)")

    # support-set (as bool mask over ALL n molecules incl. empty-frag rows) per itemset
#     def mask_of(itemset) -> np.ndarray:
#         m = np.ones(n, bool)
#         for k in itemset:
#             m &= np.array([k in motset[i] for i in range(n)])
#         return m
    # DISJUNCT POOL. A disjunct (one OR-clause) is admissible if it is either:
    #   (a) a frequent CONJUNCTION (itemset of size >= 2), OR
    #   (b) a frequent SINGLE motif that is STRUCTURAL (>= 2 heavy atoms) — a ring or a multi-atom
    #       FG/linker. A single-ATOM motif (*C, *O, *Cl, *=O, *F: 1 heavy atom) is NOT admissible as
    #       a standalone disjunct (it is a trivial shortcut); it may still appear INSIDE a size>=2
    #       conjunction. This lets a real single-motif cause (benzene 44%, a lone nitro) be planted
    #       while excluding bare-atom shortcuts.
    #   (c) DISCRIMINATIVE: clause support <= max_clause_frac * N (a clause in >30% of molecules is
    #       too prevalent to discriminate). This is what removes benzene@44% as a lone clause while
    #       keeping it available inside a lower-support conjunction.
    # NOT restricted to maximal itemsets (deepest -> rarest -> pins DNF at low coverage); the greedy
    # marginal-coverage step below drops redundant subsets for free, so no subsumption knob is needed.
#     _cap = max_clause_frac * n
#     def _structural(s):
#         return len(s) >= 2 or (len(s) == 1 and natoms.get(next(iter(s)), 1) >= 2)
#     maxi = []
#     for s in freq:
#         m = mask_of(s)
#         if _structural(s) and m.sum() <= _cap:        # (a)/(b) structural AND (c) discriminative
#             maxi.append((s, m))
#     maxi.sort(key=lambda sm: -sm[1].sum())            # highest coverage first (within the cap)

#     jc = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)

#     def rule_str(clauses):
#         return ' ∨ '.join('(' + ' ∧ '.join(sorted(cl)) + ')' for cl in clauses)

#     out: Dict[str, dict] = {}
#     if not maxi:
#         log("    [dnf] no maximal frequent itemsets at this support")
#         return out

#     def _sup1(mo):
#         return pres[mo].sum() if mo in pres else mask_of(frozenset([mo])).sum()

#     def _emit(clauses, fmask, key):
#         """Build the full apply_gt-compatible record for one DNF (its diagnostics) into out[key]."""
#         cls = [list(c) for c in clauses]
#         masks = [mask_of(c) for c in clauses]
#         npos, nneg = int(fmask.sum()), int((~fmask).sum())
        # DISJUNCTION effectiveness: max pairwise Jaccard of disjunct firing-sets (low = clean OR).
#         max_ov = max((jc(masks[i], masks[j]) for i, j in combinations(range(len(masks)), 2)),
#                      default=0.0)
        # CONJUNCTION effectiveness: per multi-motif clause, supp(clause)/min(supp motif) =
        # max_i P(rest | motif_i). Near 1 => a conjunct is implied => trivial AND. Report worst clause.
#         confs = [max(mc.sum() / max(_sup1(mo), 1) for mo in c)
#                  for c, mc in zip(clauses, masks) if len(c) >= 2]
#         max_conj_conf = round(float(max(confs)), 4) if confs else None
#         rec = dict(
#             clauses=[{'motifs': c} for c in cls],
#             rule_str=rule_str(clauses), form=key, k=len(clauses),
#             cov=round(float(fmask.mean()), 4), n_pos=npos, n_neg=nneg,
#             n_atoms=sum(natoms.get(m, 1) for c in clauses for m in c),
#             max_disjunct_jaccard=round(float(max_ov), 4),      # OR effectiveness (low = clean)
#             max_conjunct_confidence=max_conj_conf,             # AND effectiveness (low = non-trivial)
#             gamma=gamma, grader='dnf',
#             params=dict(min_support=min_sup, min_sup_frac=min_sup_frac, min_sup_floor=min_sup_floor,
#                         max_clause_frac=max_clause_frac, max_k=max_k, n_rules=n_rules, gamma=gamma),
#         )
#         if report_foolability:
#             fool = _foolability([tuple(c) for c in cls], cand, pres, motset, n)
#             rec['foolability'] = fool
#             rec['foolability_auc'] = round(float(np.mean(
#                 [f['shortcut_auc'] for f in fool])) if fool else float('nan'), 4)
#         if report_family:
#             rm = {m for c in clauses for m in c}
#             fam: Set[str] = set()
#             for m in rm:
#                 if _is_functional(m, _vmol.get(m)):
#                     fam.update(_family(m))
#             fam -= rm
#             rec['family_motifs'] = sorted(fam)
#         out[key] = rec
#         log(f"    [dnf] {key}: cov={rec['cov']:.2f} pos/neg={npos}/{nneg} "
#             f"fool={rec.get('foolability_auc', float('nan')):.3f} maxJ={max_ov:.2f} "
#             f"conjC={rec['max_conjunct_confidence']} family={len(rec.get('family_motifs', []))} "
#             f"{rec['rule_str'][:44]}")

    # conjunction-quality (lower = more genuine AND), used as the within-chain growth tiebreak below.
#     def _anchor_quality(clause, mask):
#         if len(clause) < 2:
#             return 0.0
#         return max(mask.sum() / max(_sup1(mo), 1) for mo in clause)
    # ── ANCHORS (the k=1 layer): the N most-prevalent discriminative causes = top-N by COVERAGE.
    # maxi is already sorted by coverage descending, so the head is the top-N. k=1 clauses are SEEDS
    # only; not emitted for GT-ROC (min_k>=2). n_rules=None uses the full population.
#     anchors = maxi if n_rules is None else maxi[:max(1, n_rules)]
#     log(f"    [dnf] anchors (k=1 clauses, seeds only): {len(anchors)} of {len(maxi)} "
#         f"(top by coverage); emitting k in [{min_k},{max_k}]")

    # ── grow each anchor into its own nested chain; emit only k in [min_k, max_k]. Dedup DNFs that
    # coincide across anchors (A∨B from anchor A == B∨A from anchor B) per arity by canonical clauses.
#     seen = {k: set() for k in range(min_k, max_k + 1)}
#     for ri, (a_clause, a_mask) in enumerate(anchors, 1):
#         anchor_cov = int(a_mask.sum())                         # target scale for BALANCED disjuncts
#         chosen: List[frozenset] = [a_clause]
#         fires = a_mask.copy()
#         snapshots = {1: (list(chosen), fires.copy())}
#         while len(chosen) < max_k:
#             best, best_score = None, None
#             for s, m in maxi:
#                 if s in chosen:
#                     continue
#                 new = int((m & ~fires).sum())
#                 if new == 0:
#                     continue                                   # a disjunct must add real new coverage
#                 overlap = int((m & fires).sum())
                # comparable_cov=True: NEAR-DISJOINT + COMPARABLE-COVERAGE complements — a balanced
                # disjunction ("25% anchor OR 22%", not "OR 2% rare ring"), varied across anchors ->
                # DISSIMILAR rules. False: min-overlap only -> the rarest disjoint clause -> SIMILAR
                # (shared) rules. Quality tiebreak either way.
#                 cov_gap = abs(int(m.sum()) - anchor_cov) if comparable_cov else 0
#                 score = (-(gamma * overlap + cov_gap), -_anchor_quality(s, m))
#                 if best is None or score > best_score:
#                     best, best_score = (s, m), score
#             if best is None:
#                 break
#             chosen.append(best[0]); fires |= best[1]
#             snapshots[len(chosen)] = (list(chosen), fires.copy())
#         for k in range(min_k, max_k + 1):
#             if k not in snapshots:
#                 continue
#             clauses, fmask = snapshots[k]
#             sig = frozenset(frozenset(c) for c in clauses)
#             if sig in seen[k]:
#                 continue                                       # dedup A∨B == B∨A
#             npos, nneg = int(fmask.sum()), int((~fmask).sum())
#             if min(npos, nneg) < min_sup:                      # learnability gate
#                 continue
#             seen[k].add(sig)
#             _emit(clauses, fmask, f'dnf_k{k}_r{ri}')
#     log(f"    [dnf] emitted { {k: len(seen[k]) for k in seen} } distinct rules by arity")
#     return out


# ── CLI smoke ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser(description='rule_dnf smoke on a corpus + detector')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--method', default='rdkit_fg_first',
                    choices=['fg_first', 'ertl_first', 'rdkit_fg_first'])
    ap.add_argument('--min_sup_frac', type=float, default=MIN_SUP_FRAC,
                    help='min support as a FRACTION of dataset size (default 0.02 = 2%%)')
    ap.add_argument('--max_k', type=int, default=MAX_K,
                    help='largest number of DISJUNCTS sampled, k ~ U{1..max_k} (default 3). '
                         'Conjunct width is capped separately by a_max and is NOT this knob.')
    ap.add_argument('--n_rules', type=int, default=50,
                    help='number of rules to sample (default 50 — the settled campaign size)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    import pandas as pd
    from rdkit import Chem
    import fg_first_frag as fgf; fgf.set_ring_identity('canonical')
    import cascade_bpe_linker as cb
    part = fgf.partition
    if args.method == 'ertl_first':
        import ertl_frag as _e; part = _e.partition
    elif args.method == 'rdkit_fg_first':
        import rdkit_fg_frag as _r; part = _r.partition

    raw = pd.read_csv(args.csv)['smiles'].tolist()
    smi = [s for s in raw if Chem.MolFromSmiles(s) is not None]   # DROP unparseable up front
    if len(smi) < len(raw):
        print(f"[load] dropped {len(raw) - len(smi)} unparseable SMILES ({len(smi)} kept)")
    rules, _, _ = cb.learn(smi, finest='all_bonds', beta=4.0, max_atoms=8,
                           freeze='rings', fg_partition=part)
    mol_frags = []
    for s in smi:                                                # smi is now parseable-only
        m = Chem.MolFromSmiles(s)
        tr = cb.apply_rules(m, rules, finest='all_bonds', freeze='rings', fg_partition=part)
        mol_frags.append([(k, set(a)) for k, a in fgf.rekey_structural(m, tr)])
    dnf = select_dnf(smi, mol_frags, min_sup_frac=args.min_sup_frac,
                     max_k=args.max_k, n_rules=args.n_rules)
    if args.out:
        json.dump(dnf, open(args.out, 'w'), indent=1)
        print('wrote', args.out)
