"""rule_dnf.py — DNF planted-rule engine via frequent-itemset mining.

The minimal, defensible rule engine agreed with the advisor. Governing principle
(same as rule_grid): plant a SMALL, TRANSPARENT set of causes that are EXPRESSIBLE and
LEARNABLE; DIFFICULTY is MEASURED downstream (phase-5 GT-ROC), never built in.

What changed vs rule_grid.py: the AND clauses are DISCOVERED by frequent-itemset mining
(Apriori) instead of enumerated, and the rules are genuine DNF. This nets out FEWER knobs.

Procedure (each step has a one-line reason; the ONLY tunable is min_support):

  1. TRANSACTIONS — each molecule = its set of motif keys (market-basket). No atom sets
     needed for mining (apply_gt marks GT atoms downstream from the keys).
  2. FREQUENT ITEMSETS — Apriori at support >= MIN_SUP (absolute count = estimability).
     Gives conjunctions of ANY arity that actually occur; support self-limits the depth.
     This REPLACES pairwise AND-enumeration and SUBSUMES a Jaccard/subsumption knob
     (we keep only MAXIMAL itemsets — those not contained in a larger frequent one).
  3. AND CLAUSES = frequent itemsets of size >= 2 (genuine conjunctions). We do NOT restrict
     to MAXIMAL (deepest) itemsets: those are the rarest, which pins the DNF at low coverage
     (measured — maximal-only gave cov 0.05->0.13 on BBBP). Subsumption is handled for free by
     the greedy marginal-coverage step below (a subset of a chosen clause adds no new coverage,
     so it is never picked). Maximal count is still reported as context.
  4. DNF at declared arity k = OR of k conjunctions chosen GREEDILY, highest-coverage clause
     as the anchor, then each next disjunct by
         Score = MarginalCoverage - GAMMA * Overlap
     where MarginalCoverage = |candidate ∩ not-yet-covered| and Overlap = |candidate ∩
     already-covered| (both in molecule units). GAMMA=1.0 (SHIPPED DEFAULT) applies UNIT-weighted
     penalty; GAMMA=0 recovers the parameter-free max-marginal rule. The penalty is a SOFT nudge
     toward CLEAN disjuncts
     (new coverage without dragging in redundant background vocabulary) rather than dense-but-
     redundant ones. It is a preference over WHICH clause fills a slot — it never changes k, and
     never changes the stop condition (still "no candidate adds new coverage"). k in {1..MAX_K}
     is a DECLARED FORM CHOICE; k=1 is a pure conjunction, k>=2 is a genuine DNF.
  5. GATES — exactly two, each with a hard reason:
       expressibility: every motif key maps to a real atom set (downstream, keys are valid);
       learnability : min(n_pos, n_neg) >= MIN_SUP (else no trainable model to explain).
  6. REPORT (never select on): coverage, INTER-DISJUNCT overlap (max pairwise Jaccard —
     the correlated-vocabulary caveat), and foolability (LR non-rule motifs -> label).

Output is apply_gt-compatible, keyed by 'dnf_k1', 'dnf_k2', ... Knobs: min_sup_frac (2% of N,
the estimability floor), max_clause_frac (30%, the discriminativeness ceiling), MAX_K (declared
form axis), and GAMMA (1.0, unit-weighted soft overlap penalty = the saturation point; GAMMA=0
recovers pure max-marginal). Apriori depth is uncapped — deeper conjunctions die off below
min_support on their own, so there is no MAX_LEN knob.
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
MAX_CLAUSE_FRAC = 0.30  # DISCRIMINATIVENESS ceiling: a clause (single OR conjunction) present in more
                        # than this fraction of molecules is too prevalent to discriminate, so it is
                        # NOT admissible as a disjunct. Symmetric to MIN_SUP_FRAC (the estimability
                        # floor): each clause has support in [min_support, MAX_CLAUSE_FRAC*N]. Excludes
                        # benzene@44% as a lone clause but keeps benzene inside a low-support
                        # conjunction (benzene ∧ *Cl). Conjunctions narrow support (AND is
                        # discriminative), so the cap mainly removes over-prevalent SINGLES.
MAX_K = 3               # declared largest DNF arity (k=1..MAX_K). A FORM choice, not a sweep.
N_RULES = 20            # CAP on anchors per run = the N most-prevalent discriminative causes (top-N by
                        # COVERAGE). Default 20: error bars are essentially as good as the full
                        # population (SE ~ std/sqrt(N) flattens past ~20) at ~10x less compute — the
                        # full ~183-anchor population is intractable for MAGE (300-epoch distill) +
                        # GNNExplainer (per-graph) per rule. Set to None for the full population.
MIN_K = 2               # smallest DNF arity EMITTED for GT-ROC. k=1 (single conjunctions) are anchors /
                        # seeds ONLY, never evaluated — the benchmark scores DISJUNCTIVE (k>=2) causes.
GAMMA = 1.0             # soft overlap penalty in the greedy pick (Step 4). SHIPPED DEFAULT = 1.0:
                        # UNIT weighting (one re-covered molecule cancels one new molecule) — the
                        # canonical, untuned choice, and the SATURATION point (measured: gamma>=1 all
                        # give the cleanest available disjunct, so 1.0 is "fully on" without tuning).
                        # gamma=0 recovers pure max-marginal coverage. Never a gate (only reorders).


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


def select_dnf(smiles: List[str],
               mol_frags: List[List[Tuple[str, Set[int]]]],
               *,
               min_sup_frac: float = MIN_SUP_FRAC,
               min_sup_floor: int = MIN_SUP_FLOOR,
               max_clause_frac: float = MAX_CLAUSE_FRAC,
               max_k: int = MAX_K,
               min_k: int = MIN_K,
               n_rules: Optional[int] = N_RULES,
               comparable_cov: bool = True,   # True: near-disjoint + comparable-coverage complements
               gamma: float = GAMMA,          #   -> DISSIMILAR/varied rules. False: min-overlap only ->
                                              #   SIMILAR/correlated rules (shared rare complement). Both
                                              #   valid: the flag controls rule-set diversity vs stability.
               report_foolability: bool = True,
               report_family: bool = True,
               log: Callable[[str], None] = print) -> Dict[str, dict]:
    """Mine AND clauses (maximal frequent itemsets) and build one DNF per declared arity
    k=1..max_k. Same (smiles, mol_frags) contract as rule_tiers.select_tiers, so it is a
    drop-in for the pipeline. Returns {'dnf_k<k>': apply_gt-rule}; empty if no itemsets."""
    # CONTRACT: unparseable molecules must be dropped by the caller (data-load), never passed in
    # as empty rows. An empty row has no motifs, so keeping it would silently inflate the negative
    # class and dilute every coverage denominator (n) with noise. Enforce it loudly rather than
    # limp: a molecule with no motifs is a data-cleaning error upstream, not a valid negative.
    n_empty = sum(1 for mf in mol_frags if not mf)
    if n_empty:
        raise ValueError(
            f"select_dnf received {n_empty} empty (motif-less) molecules — drop unparseable "
            f"SMILES at data load so smiles/labels/mol_frags stay aligned, then re-call.")
    n = len(smiles)
    motset = [{k for k, _ in mf} for mf in mol_frags]
    txns = [frozenset(s) for s in motset]                # no empties now; n == len(txns)
    natoms: Dict[str, int] = {}
    for mf in mol_frags:
        for k, ats in mf:
            natoms.setdefault(k, len(ats))

    # min_support as a FRACTION of the dataset (scales with N), floored for estimability
    min_sup = max(min_sup_floor, round(min_sup_frac * n))
    freq = _apriori(txns, min_sup)
    maximal = _maximal(freq)
    # foolability distractor set = ALL motif keys in the corpus (widened from frequent-singles-only
    # for a STRICTER shortcut audit). _foolability excludes the rule's own motifs internally, so the
    # probe asks "can the cause be predicted from EVERY other motif, incl. rare ones". Trade-off:
    # rare motifs are sparse columns (mild overfit/noise risk) but this no longer UNDER-estimates
    # foolability by missing rare-motif shortcuts.
    cand = sorted(set().union(*motset)) if motset else []
    pres = {k: np.array([k in motset[i] for i in range(n)]) for k in cand}

    # FAMILY motifs: for a causal (clause) motif M, the vocab motifs structurally CONTAINING M or
    # CONTAINED IN M (bidirectional, size-agnostic) — the minimal-FG <-> maximal-FG family the MDL
    # merge grows (carbonyl <-> acid/ester/amide). Marks the "right chemistry, wrong granularity"
    # attribution class, distinct from spurious (unrelated). Computed only for FUNCTIONAL clause
    # motifs (rings / heteroatom / unsaturation); saturated-carbon seeds are degenerate. Cached.
    _vmol = {k: Chem.MolFromSmiles(_strip(k)) for k in cand}
    _vq = {k: Chem.MolFromSmarts(_strip(k)) for k in cand}
    _fam_cache: Dict[str, list] = {}

    def _family(M: str) -> list:
        if M in _fam_cache:
            return _fam_cache[M]
        qM, mM = _vq.get(M), _vmol.get(M)
        fam = []
        for Mp in cand:
            if Mp == M:
                continue
            mp, qp = _vmol.get(Mp), _vq.get(Mp)
            try:
                if (qM is not None and mp is not None and mp.HasSubstructMatch(qM)) or \
                   (mM is not None and qp is not None and mM.HasSubstructMatch(qp)):
                    fam.append(Mp)
            except Exception:
                pass
        _fam_cache[M] = fam
        return fam
    _maxsz = max((len(s) for s in freq), default=0)
    log(f"    [dnf] transactions={len(txns)}  min_support={min_sup} "
        f"(={min_sup_frac:.1%} of N, floor {min_sup_floor})  frequent itemsets={len(freq)} (by size "
        f"{ {sz: sum(1 for s in freq if len(s)==sz) for sz in range(1, _maxsz + 1)} })  "
        f"maximal={len(maximal)}  distractors={len(cand)} (all motifs)")

    # support-set (as bool mask over ALL n molecules incl. empty-frag rows) per itemset
    def mask_of(itemset) -> np.ndarray:
        m = np.ones(n, bool)
        for k in itemset:
            m &= np.array([k in motset[i] for i in range(n)])
        return m
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
    _cap = max_clause_frac * n
    def _structural(s):
        return len(s) >= 2 or (len(s) == 1 and natoms.get(next(iter(s)), 1) >= 2)
    maxi = []
    for s in freq:
        m = mask_of(s)
        if _structural(s) and m.sum() <= _cap:        # (a)/(b) structural AND (c) discriminative
            maxi.append((s, m))
    maxi.sort(key=lambda sm: -sm[1].sum())            # highest coverage first (within the cap)

    jc = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)

    def rule_str(clauses):
        return ' ∨ '.join('(' + ' ∧ '.join(sorted(cl)) + ')' for cl in clauses)

    out: Dict[str, dict] = {}
    if not maxi:
        log("    [dnf] no maximal frequent itemsets at this support")
        return out

    def _sup1(mo):
        return pres[mo].sum() if mo in pres else mask_of(frozenset([mo])).sum()

    def _emit(clauses, fmask, key):
        """Build the full apply_gt-compatible record for one DNF (its diagnostics) into out[key]."""
        cls = [list(c) for c in clauses]
        masks = [mask_of(c) for c in clauses]
        npos, nneg = int(fmask.sum()), int((~fmask).sum())
        # DISJUNCTION effectiveness: max pairwise Jaccard of disjunct firing-sets (low = clean OR).
        max_ov = max((jc(masks[i], masks[j]) for i, j in combinations(range(len(masks)), 2)),
                     default=0.0)
        # CONJUNCTION effectiveness: per multi-motif clause, supp(clause)/min(supp motif) =
        # max_i P(rest | motif_i). Near 1 => a conjunct is implied => trivial AND. Report worst clause.
        confs = [max(mc.sum() / max(_sup1(mo), 1) for mo in c)
                 for c, mc in zip(clauses, masks) if len(c) >= 2]
        max_conj_conf = round(float(max(confs)), 4) if confs else None
        rec = dict(
            clauses=[{'motifs': c} for c in cls],
            rule_str=rule_str(clauses), form=key, k=len(clauses),
            cov=round(float(fmask.mean()), 4), n_pos=npos, n_neg=nneg,
            n_atoms=sum(natoms.get(m, 1) for c in clauses for m in c),
            max_disjunct_jaccard=round(float(max_ov), 4),      # OR effectiveness (low = clean)
            max_conjunct_confidence=max_conj_conf,             # AND effectiveness (low = non-trivial)
            gamma=gamma, grader='dnf',
            params=dict(min_support=min_sup, min_sup_frac=min_sup_frac, min_sup_floor=min_sup_floor,
                        max_clause_frac=max_clause_frac, max_k=max_k, n_rules=n_rules, gamma=gamma),
        )
        if report_foolability:
            fool = _foolability([tuple(c) for c in cls], cand, pres, motset, n)
            rec['foolability'] = fool
            rec['foolability_auc'] = round(float(np.mean(
                [f['shortcut_auc'] for f in fool])) if fool else float('nan'), 4)
        if report_family:
            rm = {m for c in clauses for m in c}
            fam: Set[str] = set()
            for m in rm:
                if _is_functional(m, _vmol.get(m)):
                    fam.update(_family(m))
            fam -= rm
            rec['family_motifs'] = sorted(fam)
        out[key] = rec
        log(f"    [dnf] {key}: cov={rec['cov']:.2f} pos/neg={npos}/{nneg} "
            f"fool={rec.get('foolability_auc', float('nan')):.3f} maxJ={max_ov:.2f} "
            f"conjC={rec['max_conjunct_confidence']} family={len(rec.get('family_motifs', []))} "
            f"{rec['rule_str'][:44]}")

    # conjunction-quality (lower = more genuine AND), used as the within-chain growth tiebreak below.
    def _anchor_quality(clause, mask):
        if len(clause) < 2:
            return 0.0
        return max(mask.sum() / max(_sup1(mo), 1) for mo in clause)
    # ── ANCHORS (the k=1 layer): the N most-prevalent discriminative causes = top-N by COVERAGE.
    # maxi is already sorted by coverage descending, so the head is the top-N. k=1 clauses are SEEDS
    # only; not emitted for GT-ROC (min_k>=2). n_rules=None uses the full population.
    anchors = maxi if n_rules is None else maxi[:max(1, n_rules)]
    log(f"    [dnf] anchors (k=1 clauses, seeds only): {len(anchors)} of {len(maxi)} "
        f"(top by coverage); emitting k in [{min_k},{max_k}]")

    # ── grow each anchor into its own nested chain; emit only k in [min_k, max_k]. Dedup DNFs that
    # coincide across anchors (A∨B from anchor A == B∨A from anchor B) per arity by canonical clauses.
    seen = {k: set() for k in range(min_k, max_k + 1)}
    for ri, (a_clause, a_mask) in enumerate(anchors, 1):
        anchor_cov = int(a_mask.sum())                         # target scale for BALANCED disjuncts
        chosen: List[frozenset] = [a_clause]
        fires = a_mask.copy()
        snapshots = {1: (list(chosen), fires.copy())}
        while len(chosen) < max_k:
            best, best_score = None, None
            for s, m in maxi:
                if s in chosen:
                    continue
                new = int((m & ~fires).sum())
                if new == 0:
                    continue                                   # a disjunct must add real new coverage
                overlap = int((m & fires).sum())
                # comparable_cov=True: NEAR-DISJOINT + COMPARABLE-COVERAGE complements — a balanced
                # disjunction ("25% anchor OR 22%", not "OR 2% rare ring"), varied across anchors ->
                # DISSIMILAR rules. False: min-overlap only -> the rarest disjoint clause -> SIMILAR
                # (shared) rules. Quality tiebreak either way.
                cov_gap = abs(int(m.sum()) - anchor_cov) if comparable_cov else 0
                score = (-(gamma * overlap + cov_gap), -_anchor_quality(s, m))
                if best is None or score > best_score:
                    best, best_score = (s, m), score
            if best is None:
                break
            chosen.append(best[0]); fires |= best[1]
            snapshots[len(chosen)] = (list(chosen), fires.copy())
        for k in range(min_k, max_k + 1):
            if k not in snapshots:
                continue
            clauses, fmask = snapshots[k]
            sig = frozenset(frozenset(c) for c in clauses)
            if sig in seen[k]:
                continue                                       # dedup A∨B == B∨A
            npos, nneg = int(fmask.sum()), int((~fmask).sum())
            if min(npos, nneg) < min_sup:                      # learnability gate
                continue
            seen[k].add(sig)
            _emit(clauses, fmask, f'dnf_k{k}_r{ri}')
    log(f"    [dnf] emitted { {k: len(seen[k]) for k in seen} } distinct rules by arity")
    return out


# ── CLI smoke ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser(description='rule_dnf smoke on a corpus + detector')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--method', default='rdkit_fg_first',
                    choices=['fg_first', 'ertl_first', 'rdkit_fg_first'])
    ap.add_argument('--min_sup_frac', type=float, default=MIN_SUP_FRAC,
                    help='min support as a FRACTION of dataset size (default 0.02 = 2%%)')
    ap.add_argument('--max_clause_frac', type=float, default=MAX_CLAUSE_FRAC,
                    help='discriminativeness ceiling: max clause support fraction (default 0.30)')
    ap.add_argument('--max_k', type=int, default=MAX_K)
    ap.add_argument('--min_k', type=int, default=MIN_K,
                    help='smallest DNF arity emitted (default 2; k=1 clauses are seeds only)')
    ap.add_argument('--n_rules', type=int, default=N_RULES,
                    help='cap on anchors (default None = ALL admissible k=1 clauses; int = top by quality)')
    ap.add_argument('--gamma', type=float, default=GAMMA,
                    help='soft overlap penalty (0 = knob-free default)')
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
                     max_clause_frac=args.max_clause_frac, max_k=args.max_k,
                     min_k=args.min_k, n_rules=args.n_rules, gamma=args.gamma)
    if args.out:
        json.dump(dnf, open(args.out, 'w'), indent=1)
        print('wrote', args.out)
