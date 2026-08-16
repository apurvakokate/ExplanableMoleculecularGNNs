"""rule_grid.py — minimal, defensible planted-rule engine.

Replaces the difficulty-tier machinery in rule_tiers.py. Governing principle:

    The rule engine's only job is to plant a SMALL, TRANSPARENT set of causes that
    are EXPRESSIBLE and LEARNABLE. Difficulty, faithfulness and spuriousness are
    MEASURED downstream (phase-5 GT-ROC / spurious-ROC), never built in.

Design (every step has a one-line reason; no step has a tuned threshold without one):

  0. GRID  — a DECLARED set of cells over (form x prevalence). Not a ranking, not a
     tuned parameter: a stated design choice ("we span form x prevalence"). We fill one
     representative rule per cell; empty cells are reported empty, never fabricated.
  1. CANDIDATES — every motif key in the tracked fragmentation that occurs in >= N_MIN
     molecules (absolute count = statistical estimability). NO coverage band, NO upper
     bound: a ubiquitous cause is allowed.
  2. FORMS — single / AND(A,B) / OR(A,B), enumerated deterministically, each with ONE
     integrity check: AND clauses must not subsume each other (P(A|B)<1 and P(B|A)<1) and
     share >= N_MIN positives; OR disjuncts must be near-disjoint (Jaccard <= JACCARD_MAX)
     so the OR is not a disguised single.
  3. FILL — for each cell, pick the rule whose coverage is CLOSEST to the cell's target
     prevalence and that passes the learnability gate. Tie-break: more atoms referenced
     (interpretability), a tiebreak — never a ranking axis. A rule fills at most one cell.
  4. GATES — exactly two, each with a hard reason:
       expressibility: every motif key maps to a real atom set (else no GT atoms);
       learnability : min(n_pos, n_neg) >= N_MIN (else no trainable model to explain).
  5. (downstream) apply_gt plants labels + marks GT atoms (Mode-1 / Mode-2) unchanged.
  6. (downstream) difficulty = MEASURED GT-ROC. foolability (LR of non-rule motifs ->
     label) is attached here only as a DESCRIPTIVE covariate, never a selector.

Output is apply_gt-compatible and keyed by CELL name (e.g. 'single_mid', 'and_low'), so
it drops into the pipeline wherever rule_tiers.json is read. Three surviving knobs:
N_MIN, JACCARD_MAX, and the GRID itself.
"""
from __future__ import annotations

import itertools
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# reuse the shared LR utilities (single source of truth for the shortcut measures)
from rule_tiers import _lr_cv_auc, _foolability, _atom_hist_matrix, _featurizer

# ── the three surviving knobs ────────────────────────────────────────────────────
N_MIN = 30                                  # min molecules per class / per motif (estimability)
JACCARD_MAX = 0.25                          # OR disjuncts near-disjoint (not a disguised single)
PREV_TOL = 0.075                            # a cell is filled only if a rule lands within this of
                                            # its target prevalence (= half the target spacing, so
                                            # the bands tile). Beyond it the cell is EMPTY, not a
                                            # mislabelled far-off rule ("and_high" @ cov 0.17).
# the GRID: declared cells. Prevalence targets are where we AIM; the fill step snaps the
# nearest valid rule to each. Add a 'size' axis by extending FORMS/PREVS if wanted.
PREVALENCES = [('low', 0.15), ('mid', 0.30), ('high', 0.45)]
FORMS = ['single', 'and', 'or']


def _presence(mol_frags: List[List[Tuple[str, Set[int]]]]):
    """(candidate keys, presence dict key->bool[n], motset list, per-key atom count)."""
    n = len(mol_frags)
    motset = [{k for k, _ in mf} for mf in mol_frags]
    from collections import Counter
    cov = Counter()
    for s in motset:
        cov.update(s)
    cand = [k for k, c in cov.items() if c >= N_MIN]                 # step 1: absolute floor only
    pres = {k: np.array([k in motset[i] for i in range(n)]) for k in cand}
    natoms: Dict[str, int] = {}
    for mf in mol_frags:
        for k, ats in mf:
            natoms.setdefault(k, len(ats))
    return cand, pres, motset, natoms, cov


def _enumerate(cand, pres, n):
    """Deterministic candidate rules per form. Each entry: (form, clauses, fires_bool)."""
    jc = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)
    out = []
    # singles
    for k in cand:
        out.append(('single', [[k]], pres[k]))
    # AND(A,B): genuine 2-condition, enough positives to train
    for a, b in itertools.combinations(cand, 2):
        both = pres[a] & pres[b]
        npos = int(both.sum())
        if npos < N_MIN:
            continue
        pa, pb = pres[a].sum(), pres[b].sum()
        if both.sum() >= pa or both.sum() >= pb:                    # subsumption: A=>B or B=>A
            continue
        out.append(('and', [[a, b]], both))
    # OR(A,B): near-disjoint disjuncts
    for a, b in itertools.combinations(cand, 2):
        if jc(pres[a], pres[b]) > JACCARD_MAX:
            continue
        fires = pres[a] | pres[b]
        out.append(('or', [[a], [b]], fires))
    return out


def select_grid(smiles: List[str],
                mol_frags: List[List[Tuple[str, Set[int]]]],
                *,
                n_min: int = N_MIN,
                jaccard_max: float = JACCARD_MAX,
                report_foolability: bool = True,
                log: Callable[[str], None] = print) -> Dict[str, dict]:
    raise NotImplementedError(
        "select_grid is RETIRED (form x prevalence grid engine). Planted rules are now UNIFORMLY "
        "SAMPLED by rule_dnf.sample_dnf; difficulty is measured downstream, not gridded here.")
#     """Plant one rule per (form x prevalence) cell. Returns {cell: apply_gt-rule}.
#     Empty cells are simply absent. Same (smiles, mol_frags) contract as
#     rule_tiers.select_tiers, so it is a drop-in for the pipeline."""
#     global N_MIN, JACCARD_MAX
#     N_MIN, JACCARD_MAX = n_min, jaccard_max
#     n = len(smiles)
#     cand, pres, motset, natoms, cov = _presence(mol_frags)
#     log(f"    [grid] candidates: {len(cand)} motifs with >= {n_min} molecules "
#         f"(of {len(cov)} in vocab)")
#     pool = _enumerate(cand, pres, n)
#     by_form: Dict[str, list] = {'single': [], 'and': [], 'or': []}
#     for form, cls, fires in pool:
#         by_form[form].append((cls, fires))
#     log(f"    [grid] pool: {len(by_form['single'])} single / "
#         f"{len(by_form['and'])} and / {len(by_form['or'])} or")

#     def rule_str(cls):
#         return ' ∨ '.join('(' + ' ∧ '.join(cl) + ')' for cl in cls)

#     def key_of(cls):
#         return tuple(sorted(tuple(sorted(cl)) for cl in cls))

#     used: Set[tuple] = set()
#     out: Dict[str, dict] = {}
    # deterministic cell order: prevalence-major, form-minor
#     for pv_name, pv in PREVALENCES:
#         for form in FORMS:
#             cell = f"{form}_{pv_name}"
#             best, best_score = None, None
#             for cls, fires in by_form[form]:
#                 if key_of(cls) in used:
#                     continue
#                 p = fires.mean()
#                 npos, nneg = int(fires.sum()), int((~fires).sum())
#                 if min(npos, nneg) < N_MIN:                         # gate: learnability
#                     continue
                # rank ONLY by distance to the cell's target; tiebreak = more atoms (interpretable)
#                 natom = sum(natoms.get(m, 1) for cl in cls for m in cl)
#                 score = (abs(p - pv), -natom)
#                 if best is None or score < best_score:
#                     best, best_score = (cls, fires, p, npos, nneg, natom), score
#             if best is None or best_score[0] > PREV_TOL:            # empty if nothing WITHIN the band
#                 why = "no valid rule" if best is None else \
#                       f"nearest is cov={best[2]:.2f}, > {PREV_TOL:.3f} from target"
#                 log(f"    [grid] {cell:12}: (empty — {why} near prevalence {pv:.2f})")
#                 continue
#             cls, fires, p, npos, nneg, natom = best
#             used.add(key_of(cls))
#             rec = dict(
#                 clauses=[{'motifs': list(cl)} for cl in cls],
#                 rule_str=rule_str(cls), form=form, cell=cell,
#                 cov=round(float(p), 4), n_pos=npos, n_neg=nneg,
#                 n_atoms=natom, grader='grid',
#             )
#             if report_foolability:
#                 fool = _foolability([tuple(cl) for cl in cls], cand, pres, motset, n)
#                 rec['foolability'] = fool                          # DESCRIPTIVE covariate only
#                 rec['foolability_auc'] = round(float(np.mean(
#                     [f['shortcut_auc'] for f in fool])) if fool else float('nan'), 4)
#             out[cell] = rec
#             fa = rec.get('foolability_auc', float('nan'))
#             log(f"    [grid] {cell:12}: cov={p:.2f} pos/neg={npos}/{nneg} "
#                 f"fool={fa:.3f} atoms={natom}  {rule_str(cls)[:52]}")
#     log(f"    [grid] filled {len(out)}/{len(PREVALENCES)*len(FORMS)} cells")
#     return out


# ── CLI smoke: run on a saved tracked fragmentation ──────────────────────────────
if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser(description='rule_grid smoke on a corpus + detector')
    ap.add_argument('--csv', required=True, help='fold CSV with a smiles column')
    ap.add_argument('--method', default='rdkit_fg_first',
                    choices=['fg_first', 'ertl_first', 'rdkit_fg_first'])
    ap.add_argument('--n_min', type=int, default=N_MIN)
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

    smi = [s for s in pd.read_csv(args.csv)['smiles'].tolist()]
    valid = [s for s in smi if Chem.MolFromSmiles(s) is not None]
    rules, _, info = cb.learn(valid, finest='all_bonds', beta=4.0, max_atoms=8,
                              freeze='rings', fg_partition=part)
    mol_frags = []
    for s in smi:
        m = Chem.MolFromSmiles(s)
        if m is None:
            mol_frags.append([]); continue
        tr = cb.apply_rules(m, rules, finest='all_bonds', freeze='rings', fg_partition=part)
        mol_frags.append([(k, set(a)) for k, a in fgf.rekey_structural(m, tr)])
    grid = select_grid(smi, mol_frags, n_min=args.n_min)
    if args.out:
        json.dump(grid, open(args.out, 'w'), indent=1)
        print('wrote', args.out)
