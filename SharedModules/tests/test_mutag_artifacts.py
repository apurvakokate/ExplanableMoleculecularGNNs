"""Integration tests for mutag export artifacts across pipeline stages."""

from __future__ import annotations

import csv
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:
    from SharedModules.data.mutag_artifacts import (
        MutagArtifactError,
        validate_mutag_artifacts,
    )
    from SharedModules.data.mutag_splits import (
        exclude_graph_ids_from_splits,
        save_mutag_splits,
    )
    _IMPORT_ERROR = None
except ImportError as exc:
    MutagArtifactError = None  # type: ignore
    validate_mutag_artifacts = None  # type: ignore
    exclude_graph_ids_from_splits = None  # type: ignore
    save_mutag_splits = None  # type: ignore
    _IMPORT_ERROR = exc


def _write_export_bundle(tmp: Path, graph_ids, splits, smiles='[C:1][C:2]'):
    csv_path = tmp / 'mutag_0.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(
            f, fieldnames=['smiles', 'label', 'group', 'graph_id',
                           'conversion_ok', 'verify_ok'])
        w.writeheader()
        for gid in graph_ids:
            grp = 'training'
            if gid in splits['test']:
                grp = 'test'
            elif gid in splits['valid']:
                grp = 'valid'
            w.writerow({
                'smiles': smiles,
                'label': 1.0,
                'group': grp,
                'graph_id': gid,
                'conversion_ok': True,
                'verify_ok': True,
            })
    maps = {smiles: {0: 0, 1: 1}}
    maps_path = tmp / 'mutag_0_index_maps.pkl'
    with open(maps_path, 'wb') as f:
        pickle.dump(maps, f)
    splits_path = tmp / 'mutag_0_splits.pkl'
    save_mutag_splits(splits_path, splits, seed=42)
    return csv_path, splits_path, maps_path


@unittest.skipIf(_IMPORT_ERROR is not None, f'deps missing: {_IMPORT_ERROR}')
class TestValidateMutagArtifacts(unittest.TestCase):

    def test_consistent_bundle_passes(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            splits = {'train': [0, 1], 'valid': [2], 'test': [3]}
            paths = _write_export_bundle(tmp, [0, 1, 2, 3], splits)
            info = validate_mutag_artifacts(*paths, dataset_size=10)
            self.assertEqual(info['n_graphs'], 4)
            self.assertEqual(info['n_train'], 2)

    def test_split_csv_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            splits = {'train': [0, 1], 'valid': [2], 'test': [3, 99]}
            paths = _write_export_bundle(tmp, [0, 1, 2, 3], splits)
            with self.assertRaises(MutagArtifactError):
                validate_mutag_artifacts(*paths)

    def test_empty_smiles_raises(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            csv_path = tmp / 'mutag_0.csv'
            with open(csv_path, 'w', newline='') as f:
                w = csv.DictWriter(
                    f, fieldnames=['smiles', 'label', 'group', 'graph_id'])
                w.writeheader()
                w.writerow({'smiles': '', 'label': 1, 'group': 'training',
                            'graph_id': 0})
            splits_path = tmp / 'mutag_0_splits.pkl'
            save_mutag_splits(splits_path, {'train': [0], 'valid': [], 'test': []},
                              seed=42)
            with self.assertRaises(MutagArtifactError):
                validate_mutag_artifacts(csv_path, splits_path)

    def test_exclude_failed_ids_from_splits(self):
        splits = {'train': [0, 1, 99], 'valid': [2], 'test': [3, 100]}
        out = exclude_graph_ids_from_splits(splits, [99, 100])
        self.assertEqual(out, {'train': [0, 1], 'valid': [2], 'test': [3]})


@unittest.skipIf(_IMPORT_ERROR is not None, f'deps missing: {_IMPORT_ERROR}')
class TestGenerateVocabCsvLoad(unittest.TestCase):
    """Stage: generate_vocab_rules._load_csv for mutag."""

    def test_load_mutag_csv(self):
        if str(REPO / 'MotifBreakdown') not in sys.path:
            sys.path.insert(0, str(REPO / 'MotifBreakdown'))
        from generate_vocab_rules import _load_csv

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            splits = {'train': [0], 'valid': [], 'test': []}
            csv_path, _, _ = _write_export_bundle(tmp, [0], splits)
            df = _load_csv(str(tmp), 'mutag', 0)
            self.assertEqual(len(df), 1)
            self.assertIn('label', df.columns)
            self.assertTrue(bool(df.iloc[0]['smiles']))

    def test_load_mutag_csv_rejects_empty_smiles(self):
        if str(REPO / 'MotifBreakdown') not in sys.path:
            sys.path.insert(0, str(REPO / 'MotifBreakdown'))
        from generate_vocab_rules import _load_csv

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            csv_path = tmp / 'mutag_0.csv'
            with open(csv_path, 'w', newline='') as f:
                w = csv.DictWriter(
                    f, fieldnames=['smiles', 'label', 'group', 'graph_id'])
                w.writeheader()
                w.writerow({'smiles': '', 'label': 1, 'group': 'training',
                            'graph_id': 0})
            with self.assertRaises(ValueError):
                _load_csv(str(tmp), 'mutag', 0)


@unittest.skipIf(_IMPORT_ERROR is not None, f'deps missing: {_IMPORT_ERROR}')
class TestHpcExportArtifacts(unittest.TestCase):
    """Validate mutag_0.* under MUTAG_DATA_ROOT when present."""

    def test_export_on_disk_if_present(self):
        import os
        data_root = Path(os.environ.get(
            'MUTAG_DATA_ROOT', REPO / 'data'))
        csv_p = data_root / 'mutag_0.csv'
        splits_p = data_root / 'mutag_0_splits.pkl'
        maps_p = data_root / 'mutag_0_index_maps.pkl'
        if not csv_p.is_file():
            self.skipTest(f'no export at {csv_p}')

        info = validate_mutag_artifacts(csv_p, splits_p, maps_p, dataset_size=2951)
        self.assertEqual(
            info['n_train'] + info['n_valid'] + info['n_test'],
            info['n_graphs'],
        )
        self.assertGreaterEqual(info['n_graphs'], 2937)


if __name__ == '__main__':
    unittest.main(verbosity=2)
