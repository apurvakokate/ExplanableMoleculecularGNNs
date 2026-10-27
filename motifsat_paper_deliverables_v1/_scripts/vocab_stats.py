"""vocab_stats.py — READ-ONLY vocabulary statistics for the write-up (S1, approved 2026-10-26).

Run on the HPC (l2xgnn env):  python3 vocab_stats.py OUT_JSON
For rbrics_v3 (vocab_final_v3/<ds>/rbrics, one vocabulary for all folds) and fmbpe_v4
(fmbpe_v4/vocab/fold0/<ds>/fm_bpe_full_union, the fold-0 vocabulary) on the 8 main + 6 GSAT datasets.
Source: <ds>_<variant>_mol_fragment_copies.pickle = {smiles: [(key, motif_id, [atom idx]) per occurrence]}
(every molecule of the dataset; atom indices are heavy atoms in RemoveHs numbering).
Per (vocab, dataset):
  molecules, vocabulary size (motif_list), types that occur, share of types seen in exactly one molecule,
  occurrences per molecule (mean / median), heavy atoms per occurrence (mean / median, histogram
  1 | 2-3 | 4-6 | 7-10 | >10), share of occurrences that contain a ring, share of single-atom occurrences,
  number of types covering 50% / 90% of all occurrences, share of molecules with a repeated type,
  and the 10 most frequent types (key, molecules containing it).
"""
import json
import pickle
import sys
from collections import Counter

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
R = '/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor'
DATASETS = ['Benzene_Verified_GT', 'Alkane_Carbonyl_Verified_GT', 'Fluoride_Carbonyl_Verified_GT', 'BBBP', 'hERG',
            'Mutagenicity', 'esol', 'Lipophilicity', 'mutag', 'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp',
            'ogbg-moltox21', 'ogbg-molclintox']
VOCABS = {'rbrics_v3': (f'{R}/vocab_final_v3', 'rbrics'),
          'fmbpe_v4': (f'{R}/fmbpe_v4/vocab/fold0', 'fm_bpe_full_union')}
BINS = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 10 ** 9)]
_ring = {}


def has_ring(key):
    if key not in _ring:
        m = Chem.MolFromSmiles(key, sanitize=False) or Chem.MolFromSmarts(key)
        if m is None:
            _ring[key] = None
        else:
            m.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(m)
            _ring[key] = m.GetRingInfo().NumRings() > 0
    return _ring[key]


def stats(root, var, ds):
    base = f'{root}/{ds}/{var}/{ds}_{var}'
    vocab = list(pickle.load(open(f'{base}_motif_list.pickle', 'rb')))
    copies = pickle.load(open(f'{base}_mol_fragment_copies.pickle', 'rb'))
    occ_per_mol, sizes, ring, rep = [], [], [], 0
    occ_count, mol_count = Counter(), Counter()
    for smi, occ in copies.items():
        occ_per_mol.append(len(occ))
        ids = [k for k, _, _ in occ]
        if len(set(ids)) < len(ids):
            rep += 1
        for k, _, atoms in occ:
            sizes.append(len(atoms)); occ_count[k] += 1
            r = has_ring(k)
            if r is not None:
                ring.append(r)
        for k in set(ids):
            mol_count[k] += 1
    sizes = np.array(sizes); tot = sizes.size
    cum = np.cumsum(sorted(occ_count.values(), reverse=True)) / tot
    return {
        'molecules': len(copies), 'vocab_size': len(vocab), 'types_occurring': len(occ_count),
        'singleton_type_share': float(np.mean([c == 1 for c in mol_count.values()])),
        'occ_per_mol_mean': float(np.mean(occ_per_mol)), 'occ_per_mol_median': float(np.median(occ_per_mol)),
        'atoms_per_occ_mean': float(sizes.mean()), 'atoms_per_occ_median': float(np.median(sizes)),
        'atoms_hist': {f'{a}-{b}' if b < 10 ** 9 else f'>{a - 1}': float(((sizes >= a) & (sizes <= b)).mean())
                       for a, b in BINS},
        'ring_occ_share': float(np.mean(ring)) if ring else None,
        'single_atom_occ_share': float((sizes == 1).mean()),
        'types_for_50pct_occ': int(np.searchsorted(cum, 0.5) + 1),
        'types_for_90pct_occ': int(np.searchsorted(cum, 0.9) + 1),
        'mol_with_repeated_type_share': rep / len(copies),
        'top10_by_molecules': [(k, c) for k, c in mol_count.most_common(10)],
    }


def main(out):
    rep = {}
    for v, (root, var) in VOCABS.items():
        for ds in DATASETS:
            try:
                rep[f'{v}::{ds}'] = stats(root, var, ds)
            except Exception as e:                     # report, never hide
                rep[f'{v}::{ds}'] = {'error': repr(e)}
            print(v, ds, json.dumps({k: x for k, x in rep[f'{v}::{ds}'].items() if k != 'top10_by_molecules'}),
                  flush=True)
    json.dump(rep, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
