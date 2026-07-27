"""Rule Set 1 — Stage 1 (Load) unit tests.

Tiny synthetic fixtures using the REAL registered dataset names (so the schema/routing
registries apply), plus direct build_graph checks. The headline test is the source-GT
hook driven through get_loaders(<name>) — the seam that today's bug bypassed.
"""
import csv
import os
import tempfile
import unittest

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from SharedModules.data.dataset import build_graph, MolDataset, NUM_ATOM_TYPES
from SharedModules.data.loader import get_loaders
from SharedModules.data.dataset_schema import DATASET_COLUMN, TASK_TYPE, SOURCE_GT_SMARTS
from SharedModules.data.dataset_routing import validate_use_gt, assert_vocab_rule_mining_allowed
from SharedModules.data.ground_truth import GT_SUPPORTED_DATASETS
import torch


def _write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


class Stage1Registry(unittest.TestCase):
    def test_classification_label_and_task(self):
        self.assertEqual(DATASET_COLUMN['BBBP'], 'BBBP')
        self.assertEqual(TASK_TYPE['BBBP'], 'BinaryClass')

    def test_regression_task_type(self):
        self.assertEqual(TASK_TYPE['esol'], 'Regression')

    def test_regression_rule_mining_refused(self):
        with self.assertRaises(Exception):
            assert_vocab_rule_mining_allowed('esol')


class Stage1BuildGraph(unittest.TestCase):
    def test_feature_dim_onehot(self):
        d = build_graph('c1ccccc1', torch.tensor([1.0]), None)
        self.assertEqual(d.x.shape[1], NUM_ATOM_TYPES)

    def test_smiles_atom_order_matches_x(self):
        for smi in ('CC(=O)Oc1ccccc1', 'CCN', 'O=C(N)c1ccccc1'):
            d = build_graph(smi, torch.tensor([1.0]), None)
            self.assertEqual(d.smiles, smi, 'smiles not stored verbatim')
            self.assertEqual(d.x.shape[0], Chem.MolFromSmiles(smi).GetNumAtoms(),
                             f'{smi}: x rows != atom count')
            if d.edge_index.numel():
                self.assertLess(int(d.edge_index.max()), d.x.shape[0],
                                f'{smi}: edge index out of range')

    def test_invalid_smiles_returns_none(self):
        self.assertIsNone(build_graph('not_a_valid_smiles', torch.tensor([1.0]), None))


class Stage1SplitsAndDrop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # tiny classification fixture under the real name 'BBBP' (label col 'BBBP')
        rows = [('c1ccccc1', 1, 'training'), ('CCO', 0, 'training'), ('CCN', 1, 'training'),
                ('c1ccncc1', 0, 'valid'), ('CCCC', 0, 'valid'),
                ('c1ccccc1O', 1, 'test'), ('not_a_smiles', 0, 'test')]   # 1 invalid
        _write_csv(os.path.join(self.tmp, 'BBBP_0.csv'), ['smiles', 'BBBP', 'group'], rows)

    def _ds(self, split):
        return MolDataset(root=os.path.join(self.tmp, 'proc', split),
                          csv_file=os.path.join(self.tmp, 'BBBP_0.csv'),
                          split=split, label_col='BBBP', force_reprocess=True)

    def test_splits_disjoint_and_cover(self):
        tr = {d.smiles for d in self._ds('training')}
        va = {d.smiles for d in self._ds('valid')}
        te = {d.smiles for d in self._ds('test')}
        self.assertEqual(tr & va, set()); self.assertEqual(tr & te, set()); self.assertEqual(va & te, set())
        # test split had one invalid SMILES → dropped (c1ccccc1O valid, not_a_smiles gone)
        self.assertEqual(te, {'c1ccccc1O'}, 'invalid SMILES not dropped from test split')


class Stage1SourceGTHook(unittest.TestCase):
    """The headline: driving get_loaders BY NAME must attach source GT (today's bug)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.proc = tempfile.mkdtemp()
        rows0 = [('c1ccccc1', 1, 'test'), ('c1ccccc1CC', 1, 'test'), ('CCO', 0, 'test'),
                 ('c1ccccc1O', 1, 'training'), ('CCCCO', 0, 'training'), ('CCN', 0, 'training'),
                 ('c1ccccc1N', 1, 'valid'), ('CCC', 0, 'valid')]
        _write_csv(os.path.join(self.tmp, 'Benzene_Verified_GT_0.csv'),
                   ['smiles', 'label', 'group'], rows0)
        # fold 1: same benzene molecule, different split → fold-consistency check
        rows1 = [('c1ccccc1', 1, 'training'), ('CCO', 0, 'test'), ('c1ccccc1CC', 1, 'test'),
                 ('CCCCO', 0, 'training'), ('CCN', 0, 'valid'), ('c1ccccc1N', 1, 'valid')]
        _write_csv(os.path.join(self.tmp, 'Benzene_Verified_GT_1.csv'),
                   ['smiles', 'label', 'group'], rows1)

    def _load(self, fold):
        loaders, test_ds, meta = get_loaders(
            'Benzene_Verified_GT', data_root=self.tmp, fold=fold, vocab=None,
            processed_root=os.path.join(self.proc, f'f{fold}'), force_reprocess=True)
        return loaders, test_ds

    def test_hook_attaches_node_label_via_get_loaders(self):
        _, test_ds = self._load(0)
        graphs = list(test_ds)
        self.assertTrue(all(getattr(g, 'node_label', None) is not None for g in graphs),
                        'get_loaders did not attach node_label for a _Verified_GT dataset')
        benz = next(g for g in graphs if g.smiles == 'c1ccccc1')
        self.assertEqual(int(benz.node_label.sum()), 6, 'benzene GT should mark all 6 ring atoms')

    def test_node_label_matches_smarts(self):
        _, test_ds = self._load(0)
        q = Chem.MolFromSmarts('c1ccccc1')
        for g in test_ds:
            m = Chem.MolFromSmiles(g.smiles)
            expected = set()
            for match in m.GetSubstructMatches(q):
                expected.update(match)
            got = {i for i, v in enumerate(g.node_label.tolist()) if v > 0}
            self.assertEqual(got, expected, f'{g.smiles}: node_label != SMARTS match')

    def test_fold_consistent(self):
        _, ds0 = self._load(0)
        _, ds1 = self._load(1)
        # benzene is in fold-0 test and fold-1 training; find it in each split's datasets
        def label_of(fold):
            loaders, _ = self._load(fold)
            for split in ('train', 'valid', 'test'):
                for g in loaders[split].dataset:
                    if g.smiles == 'c1ccccc1':
                        return g.node_label.tolist()
            return None
        self.assertEqual(label_of(0), label_of(1),
                         'benzene GT differs across folds — GT is not fold-consistent')

    def test_not_in_gt_supported(self):
        self.assertNotIn('Benzene_Verified_GT', GT_SUPPORTED_DATASETS)

    def test_use_gt_raises(self):
        with self.assertRaises(Exception):
            validate_use_gt('Benzene_Verified_GT', use_gt=True, gt_cache='/tmp/x')


class Stage1MutagDropArtifact(unittest.TestCase):
    """The mutag drop is a deliberate selection rule → recall-by-construction. Assert the
    construction property (flag), so a change in its magnitude trips."""

    def setUp(self):
        import glob
        hits = glob.glob(os.path.join(os.path.dirname(__file__), '..', '..', '**',
                                      'Mutagenicity_edge_gt.txt'), recursive=True)
        if not hits:
            self.skipTest('mutag raw not present locally')
        self.raw = os.path.dirname(hits[0])

    def test_toxicophore_selection_and_recall_by_construction(self):
        import numpy as np
        from collections import defaultdict
        L = lambda n: np.loadtxt(f'{self.raw}/Mutagenicity_{n}.txt', delimiter=',').astype(int)
        A, gi, gl, egt = L('A'), L('graph_indicator'), L('graph_labels'), L('edge_gt')
        has = defaultdict(bool)
        for (s, _), g in zip(A, egt):
            if g == 1:
                has[gi[s - 1]] = True
        graphs = sorted(set(gi)); cls = {g: int(gl[g - 1]) for g in graphs}   # 0=mutagen
        mut = [g for g in graphs if cls[g] == 0]
        with_motif = sum(1 for g in mut if has[g])
        frac = with_motif / len(mut)
        # raw ~42% of mutagens carry the toxicophore; the drop rule removes the rest → 100%
        self.assertAlmostEqual(frac, 0.423, delta=0.03,
                               msg=f'mutagen toxicophore coverage {frac:.3f} changed from ~0.423 '
                                   f'(loader/data changed — recall-by-construction assumption invalid)')


if __name__ == '__main__':
    unittest.main(verbosity=2)
