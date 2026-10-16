#!/usr/bin/env python3
"""fm_bpe_lambda_star.py — choose lambda* for FM-BPE (full-MDL cost) once per dataset.

lambda* = the smallest lambda at which merging never creates UNK motifs under the dataset's own
support filter. Two ways a merge can create UNK, both checked:
  (1) NEW: a merge-created type whose train+val count ends below the global cut;
  (2) DEMOTED: a finest type that was above the cut becomes below it because merges consumed its
      occurrences (its leftover atoms would become UNK).
The cut mirrors the MoSE loader exactly (threshold_config.select_threshold_motifs, global part):
    global_cut = int(threshold_pct * n_trainval),  counts = occurrences in train+val molecules.
The label-dependent minority-class rescue is deliberately NOT used, so lambda* stays label-free;
at MoSE time the rescue can only keep MORE motifs.

lambda* is found by GRID-THEN-REFINE (larger lambda -> fewer merges -> fewer violations):
  1. coarse half-decade grid 0.1, 0.3, 1, 3, ..., 10000 -> the full curve (merges, violations,
     coverage, vocab) and a direct monotonicity check;
  2. bisection in log-lambda ONLY inside the bracket [last failing grid point, first passing one].
Every probe is logged; non_monotone=true flags a pass/fail pattern that is not fail..fail,pass..pass.
The merge list at lambda* (pair, f12, molecule support, size) is saved for inspection.
MoSE-blind: uses SMILES structure and split membership only — no labels, no model.

Note: counts here use FM-BPE's internal fragment keys. The vocab pipeline later re-keys them
structurally (rekey_structural), which can only pool equivalent keys -> counts can only rise, so
this check is conservative. The loader's actual kept/dropped counts are logged per fold at MoSE time.

Usage:
  python fm_bpe_lambda_star.py --dataset BBBP --data_root <FOLDS> --threshold_pct 0.006 \
      --out <dir>/BBBP.json [--fold 0] [--refine_iters 8]
"""
import argparse, json, math, os, sys, time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from rdkit import Chem
import fm_bpe

_TRAINVAL = ('training', 'valid')


def _counts_and_atoms(graphs):
    cnt, size, tot_atoms = Counter(), {}, 0
    for g in graphs:
        for k, atoms in g.fragments():
            cnt[k] += 1
            size[k] = len(atoms)
            tot_atoms += len(atoms)
    return cnt, size, tot_atoms


def _coverage(cnt, size, tot_atoms, cut):
    kept_atoms = sum(cnt[k] * size[k] for k in cnt if cnt[k] >= cut)
    return 100.0 * kept_atoms / max(tot_atoms, 1)


GRID = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]


def probe(learn_smi, lam, cut, finest_cnt, start, freeze_rings, start_graphs):
    t0 = time.time()
    # start_graphs = the finest graphs built ONCE in main(); learn_fm_bpe deep-copies them, so every
    # probe starts from the identical state without re-running the union-start segmentation.
    rules, graphs = fm_bpe.learn_fm_bpe(learn_smi, lam=lam, max_merges=None, verbose=False,
                                        freeze_rings=freeze_rings, start=start, cost='full_mdl',
                                        start_graphs=start_graphs)
    cnt, size, tot = _counts_and_atoms(graphs)
    new_bad = [k for k, c in cnt.items() if c > 0 and k not in finest_cnt and c < cut]
    demoted = [k for k, c0 in finest_cnt.items() if c0 >= cut and 0 < cnt.get(k, 0) < cut]
    rec = dict(lam=lam, ok=(not new_bad and not demoted), n_merges=len(rules),
               n_new_below_cut=len(new_bad), n_demoted=len(demoted),
               min_merge_mols=(min(r[4] for r in rules) if rules else None),
               vocab=sum(1 for c in cnt.values() if c > 0),
               coverage_pct=round(_coverage(cnt, size, tot, cut), 2),
               seconds=round(time.time() - t0, 1))
    merges = [dict(step=i, m1=r[0][0], m2=r[0][1], f12=r[2], n_mols=r[4], merged_size=r[3])
              for i, r in enumerate(rules)]
    return rec, merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--threshold_pct', type=float, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fold', type=int, default=0, help='mining fold (rBRICS vocab_meta uses 0)')
    ap.add_argument('--refine_iters', type=int, default=8,
                    help='log-lambda bisection steps inside the failing/passing grid bracket')
    ap.add_argument('--start', default='union', choices=['union', 'atomic'])
    args = ap.parse_args()
    freeze_rings = True

    df = pd.read_csv(os.path.join(args.data_root, f'{args.dataset}_{args.fold}.csv'))
    tv = df[df['group'].isin(_TRAINVAL)]
    n_tv = len(tv)                                   # loader uses len(trainval rows)
    cut = int(args.threshold_pct * n_tv)
    learn_smi = tv['smiles'].astype(str).tolist()
    # Parseability is RDKit-version dependent: HPC rdkit 2023.09.6 parses all 8 datasets; local rdkit
    # 2026.03.3 rejects 2 Mutagenicity SMILES (see the RDKit-VERSION NOTE in generate_vocab_rules.py).
    # Authoritative lambda* = HPC env. Never skip or leniently parse them here.
    _bad = [s for s in learn_smi if Chem.MolFromSmiles(s) is None]
    if _bad:
        raise ValueError(f'{len(_bad)} unparseable train+val SMILES (no silent skip), e.g. {_bad[:3]}')
    if cut < 1:
        raise ValueError(f'global_cut={cut} < 1 (pct={args.threshold_pct}, n_trainval={n_tv})')
    print(f'[lambda*] {args.dataset} fold{args.fold}: n_trainval={n_tv} pct={args.threshold_pct} '
          f'global_cut={cut} learn_mols={len(learn_smi)}', flush=True)

    finest = [fm_bpe._finest_graph(Chem.MolFromSmiles(s), freeze_rings=freeze_rings, start=args.start)
              for s in learn_smi]
    finest_cnt, fsize, ftot = _counts_and_atoms(finest)
    finest_cov = _coverage(finest_cnt, fsize, ftot, cut)

    probes, merges_at = [], {}
    def P(lam):
        rec, merges = probe(learn_smi, lam, cut, finest_cnt, args.start, freeze_rings, finest)
        probes.append(rec); merges_at[lam] = merges
        print(f'  probe {rec}', flush=True)
        return rec['ok']

    # 1) coarse grid -> full curve + monotonicity check
    grid_ok = [(lam, P(lam)) for lam in GRID]
    oks = [ok for _, ok in grid_ok]
    first_pass = oks.index(True) if True in oks else None
    grid_non_monotone = first_pass is not None and not all(oks[first_pass:])

    # 2) refine by log-lambda bisection inside [last failing, first passing] grid points
    if first_pass is None:
        status, lam_star, bracket = 'no_lambda_in_range', None, [GRID[-1], None]
    elif first_pass == 0:
        status, lam_star, bracket = 'holds_at_grid_min', GRID[0], [None, GRID[0]]
    else:
        lo, hi = GRID[first_pass - 1], GRID[first_pass]
        for _ in range(args.refine_iters):
            mid = math.sqrt(lo * hi)
            if P(mid):
                hi = mid
            else:
                lo = mid
        status, lam_star, bracket = 'refined', hi, [lo, hi]

    srt = sorted(probes, key=lambda r: r['lam'])
    all_oks = [r['ok'] for r in srt]
    ft = all_oks.index(True) if True in all_oks else len(all_oks)
    non_monotone = grid_non_monotone or not all(all_oks[ft:])   # monotone == fail..fail, pass..pass

    accepted = (status == 'refined') and not non_monotone
    out = dict(dataset=args.dataset, fold=args.fold, threshold_pct=args.threshold_pct,
               n_trainval=n_tv, global_cut=cut, cost='full_mdl', start=args.start,
               f12_count='realizable',          # node-disjoint merges (fm_bpe._local_pairs), not raw adjacencies
               rdkit_version=Chem.rdBase.rdkitVersion,
               freeze_rings=freeze_rings,
               lambda_star=(lam_star if accepted else None),   # None => vocab stage refuses
               lambda_candidate=lam_star, accepted=accepted, status=status,
               bracket=bracket, grid=GRID, refine_iters=args.refine_iters,
               finest_coverage_pct=round(finest_cov, 2), finest_vocab=len(finest_cnt),
               non_monotone=non_monotone, probes=srt,
               merges_at_lambda_star=(merges_at.get(lam_star) if lam_star is not None else None))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[lambda*] {args.dataset}: lambda*={lam_star} status={status} '
          f'non_monotone={non_monotone} -> {args.out}', flush=True)
    # FAIL LOUD (after the diagnostics are written): no lambda is handed to the vocab stage unless
    # it is an interior, monotone, refined crossover.
    if not accepted:
        sys.exit(f'[lambda*] FAIL {args.dataset}: status={status} non_monotone={non_monotone} '
                 f'(see {args.out}); refusing to emit a lambda*')


if __name__ == '__main__':
    main()
