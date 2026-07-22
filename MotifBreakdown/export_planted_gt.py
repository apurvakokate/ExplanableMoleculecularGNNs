#!/usr/bin/env python3
"""export_planted_gt.py — export the Sanchez-Lengeling planted-ground-truth
molecular tasks (benzene / logic7 / logic8) into this project's fold-CSV layout,
plus the atom-level ground-truth attributions that ship with them.

WHY THIS EXISTS
---------------
These three tasks are the only datasets we have whose causal substructure is
planted INDEPENDENTLY of our motif vocabularies. Everything else in the project
derives its "ground truth" from a rule over the same vocabulary being evaluated,
which cannot measure explanation quality (see the entanglement analysis).

Source: https://github.com/google-research/graph-attribution  (data/{task}/)
  Sanchez-Lengeling et al., "Evaluating Attribution for Graph Neural Networks",
  NeurIPS 2020.  Rules defined verbatim in graph_attribution/tasks.py:284-297.

THREE THINGS THE UPSTREAM DATA GETS WRONG (all verified, all handled here)
-------------------------------------------------------------------------
1. NAMES. The source ships numbered dirs (logic7/logic8). GraphXAI named them
   logic7->AlkaneCarbonyl, logic8->FluorideCarbonyl -- BACKWARDS. Per tasks.py,
   logic7 is fluoride AND carbonyl; logic8 is unbranched-alkane AND carbonyl.
   We use the correct names here.

2. THE CSV `label` COLUMN IS JUNK. scripts/generate_mol_tasks_data.py reads only
   df['smiles'] and RECOMPUTES y from the rule; the csv `label` column is never
   read. For logic7/8/10 it disagrees with y_true (0.735 / 0.733 / 0.501 -- the
   last is chance). We take y from y_true.npz and ignore the csv label.
   (benzene's csv label happens to agree at 1.0000, but we ignore it there too,
   for uniformity.)

3. logic8's "unbranched alkane" SMARTS is ELEMENT-AGNOSTIC:
       [R0;D2,D1][R0;D2][R0;D2,D1]
   R0=acyclic, D=degree, no element constraint -- so C-C-O, C-N-C, C-S-C all
   match. Only 24.4% of its "alkane" ground truth is pure carbon. The data
   faithfully implements this pattern; the pattern itself is chemically wrong.
   Report this caveat whenever Alkane_Carbonyl results are shown.

ATOM ORDERING -- IMPORTANT
--------------------------
The attribution masks index atoms by RDKit's atom order for the EXACT smiles
string in the source csv. We therefore write that string through verbatim and
never canonicalize it. Any consumer must parse the same string with
Chem.MolFromSmiles() to keep mask indices aligned.

OUTPUT
------
  {out_root}/{Dataset}_{fold}.csv        smiles,label,group   (project convention)
  {gt_out}/{Dataset}_planted_gt.npz      smiles -> node_imp [n_atoms x J]

`node_imp` keeps ALL J explanation columns. J = 2^(n_fragments) - 1: the columns
enumerate every non-empty subset of matching fragments, so they include
non-minimal supersets alongside the minimal sufficient reasons (prime
implicants). Grade with max-over-columns, never against their union.

Usage:
    python3 export_planted_gt.py --out_root ./FOLDS --gt_out ./planted_gt
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    sys.exit('rdkit is required: conda install -c conda-forge rdkit')

RAW = ('https://raw.githubusercontent.com/google-research/'
       'graph-attribution/main/data')

# task dir -> (our dataset name, rule SMARTS conjuncts)
# Rules copied verbatim from graph_attribution/tasks.py:284-297.
#
# NAMES: the `_Verified_GT` suffix is deliberate and load-bearing. Our legacy
# `Alkane_Carbonyl` is logic7 (4,326 mols) and legacy `Fluoride_Carbonyl` is logic8
# (8,671 mols) -- i.e. the legacy names are SWAPPED relative to the rules. Reusing the
# bare names would collide with legacy artifacts that are keyed by dataset name
# (vocab_output/, results/, gt_cache/, processed_root/, and run_experiments.py's
# skip_existing out_dir). The worst case is silent: skip_existing would find a stale
# summary.json under the same config tag and report months-old logic7 numbers as new
# logic8 results. Fresh names make every one of those collisions impossible.
# These ARE new datasets: labels come from y_true (not the junk csv label) and they
# carry atom-level attributions the legacy exports never had.
TASKS: Dict[str, Tuple[str, List[str]]] = {
    'benzene': ('Benzene_Verified_GT', ['c1ccccc1']),
    'logic7':  ('Fluoride_Carbonyl_Verified_GT', ['[FX1]', '[CX3]=O']),
    'logic8':  ('Alkane_Carbonyl_Verified_GT', ['[R0;D2,D1][R0;D2][R0;D2,D1]', '[CX3]=O']),
}


def _fetch(url: str, cache: Path) -> bytes:
    if cache.exists():
        return cache.read_bytes()
    with urllib.request.urlopen(url) as r:
        blob = r.read()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(blob)
    return blob


def load_task(task: str, cache_dir: Path):
    """Return (smiles list, y [n], attribution datadicts [n])."""
    csv = pd.read_csv(io.BytesIO(
        _fetch(f'{RAW}/{task}/{task}_smiles.csv', cache_dir / f'{task}.csv')))
    y = np.load(io.BytesIO(
        _fetch(f'{RAW}/{task}/y_true.npz', cache_dir / f'{task}_y.npz')),
        allow_pickle=True)['y'].ravel()
    att = np.load(io.BytesIO(
        _fetch(f'{RAW}/{task}/true_raw_attribution_datadicts.npz',
               cache_dir / f'{task}_att.npz')),
        allow_pickle=True)['datadict_list']
    return csv['smiles'].tolist(), y, att


def verify(name: str, smiles: List[str], y: np.ndarray, att, patterns: List[str]):
    """Assert labels and attributions reproduce the source rule exactly.

    Fails loudly rather than exporting data we have not re-derived. This is the
    check that caught the upstream naming swap; keep it in the pipeline.
    """
    pats = [Chem.MolFromSmarts(p) for p in patterns]
    yb = y.astype(bool)
    rule, bad_neg, bad_pos = [], 0, 0
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(smi)
        hit = bool(m) and all(m.HasSubstructMatch(p) for p in pats)
        rule.append(hit)
        if m is None:
            continue
        nodes = np.asarray(att[i][0]['nodes'])
        marked = {int(k) for k in np.where(nodes.max(1) > 0)[0]
                  if k < m.GetNumAtoms()}
        if not yb[i]:
            if marked:
                bad_neg += 1
            continue
        expect = set()
        for p in pats:
            for mt in m.GetSubstructMatches(p):
                expect.update(mt)
        if marked != expect:
            bad_pos += 1
    rule = np.array(rule)
    n_fp = int((yb & ~rule).sum())
    n_fn = int((~yb & rule).sum())
    print(f'  verify {name}: n={len(yb)} pos={int(yb.sum())} '
          f'| y==rule {float((yb == rule).mean()):.6f} '
          f'| false_pos={n_fp} false_neg={n_fn} '
          f'| neg_with_attr={bad_neg} pos_attr_mismatch={bad_pos}')
    assert n_fp == 0 and n_fn == 0, f'{name}: labels disagree with the rule'
    assert bad_neg == 0, f'{name}: negatives carry attributions'
    assert bad_pos == 0, f'{name}: attributions != rule matches'


def make_folds(y: np.ndarray, n_folds: int, seed: int) -> List[np.ndarray]:
    """Stratified 80/10/10 training/valid/test groups, one array per fold."""
    out = []
    for f in range(n_folds):
        rng = np.random.RandomState(seed + f)
        grp = np.empty(len(y), dtype=object)
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            idx = idx[rng.permutation(len(idx))]
            n_te = int(round(0.10 * len(idx)))
            n_va = int(round(0.10 * len(idx)))
            grp[idx[:n_te]] = 'test'
            grp[idx[n_te:n_te + n_va]] = 'valid'
            grp[idx[n_te + n_va:]] = 'training'
        out.append(grp)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out_root', required=True, help='where to write {Dataset}_{fold}.csv')
    ap.add_argument('--gt_out', required=True, help='where to write {Dataset}_planted_gt.npz')
    ap.add_argument('--cache', default='./.planted_gt_cache', help='download cache dir')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    out_root, gt_out = Path(a.out_root), Path(a.gt_out)
    out_root.mkdir(parents=True, exist_ok=True)
    gt_out.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache)

    for task, (name, patterns) in TASKS.items():
        print(f'\n{task} -> {name}   rule: {" AND ".join(patterns)}')
        smiles, y, att = load_task(task, cache)
        verify(name, smiles, y, att, patterns)

        # Fold CSVs. label comes from y_true, NEVER the source csv `label`.
        for f, grp in enumerate(make_folds(y, a.folds, a.seed)):
            df = pd.DataFrame({'smiles': smiles,
                               'label': y.astype(int),
                               'group': grp})
            p = out_root / f'{name}_{f}.csv'
            df.to_csv(p, index=False)
        print(f'  wrote {a.folds} fold csv(s) -> {out_root}/{name}_*.csv')

        # Planted GT: smiles -> [n_atoms x J] float mask, all columns kept.
        gt = {smi: np.asarray(att[i][0]['nodes'], dtype=np.float32)
              for i, smi in enumerate(smiles)}
        gp = gt_out / f'{name}_planted_gt.npz'
        np.savez_compressed(gp, **{f'{i}': v for i, v in enumerate(gt.values())},
                            smiles=np.array(list(gt.keys()), dtype=object))
        js = [v.shape[1] for v in gt.values()]
        print(f'  wrote planted GT -> {gp}  (J: min={min(js)} max={max(js)} '
              f'mean={np.mean(js):.2f})')

    print('\nDone. NOTE: smiles strings are verbatim from source — do not '
          'canonicalize, or attribution atom indices will desync.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
