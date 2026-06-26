"""Integration tests for mutag export artifacts across pipeline stages."""

from __future__ import annotations

import csv
import importlib.util
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
SM = REPO / 'SharedModules'


def _load_module(name: str, path: Path, package=None):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    if package is not None:
        mod.__package__ = package
    spec.loader.exec_module(mod)
    return mod


_mutag_splits = _load_module('mutag_splits', SM / 'data' / 'mutag_splits.py')
import types
_pkg = types.ModuleType('SharedModules')
_pkg_data = types.ModuleType('SharedModules.data')
sys.modules.setdefault('SharedModules', _pkg)
sys.modules.setdefault('SharedModules.data', _pkg_data)
sys.modules['SharedModules.data.mutag_splits'] = _mutag_splits
_mutag_artifacts = _load_module(
    'mutag_artifacts', SM / 'data' / 'mutag_artifacts.py',
    package='SharedModules.data')

MutagArtifactError = _mutag_artifacts.MutagArtifactError
validate_mutag_artifacts = _mutag_artifacts.validate_mutag_artifacts
exclude_graph_ids_from_splits = _mutag_splits.exclude_graph_ids_from_splits
save_mutag_splits = _mutag_splits.save_mutag_splits


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


class TestGenerateVocabCsvLoad(unittest.TestCase):
    """Stage: generate_vocab_rules._load_csv for mutag."""

    def _import_load_csv(self):
        rdkit = mock.MagicMock()
        rdkit.Chem = mock.MagicMock()
        rdkit.RDLogger = mock.MagicMock()
        stubs = {
            'torch': mock.MagicMock(),
            'rdkit': rdkit,
            'rdkit.Chem': rdkit.Chem,
            'rdkit.RDLogger': rdkit.RDLogger,
        }
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        if str(REPO / 'MotifBreakdown') not in sys.path:
            sys.path.insert(0, str(REPO / 'MotifBreakdown'))
        with mock.patch.dict(sys.modules, stubs):
            import generate_vocab_rules as gvr
            return gvr._load_csv

    def test_load_mutag_csv(self):
        try:
            _load_csv = self._import_load_csv()
        except ModuleNotFoundError as e:
            self.skipTest(f'generate_vocab_rules deps missing: {e}')

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            splits = {'train': [0], 'valid': [], 'test': []}
            csv_path, _, _ = _write_export_bundle(tmp, [0], splits)
            df = _load_csv(str(tmp), 'mutag', 0)
            self.assertEqual(len(df), 1)
            self.assertIn('label', df.columns)
            self.assertTrue(bool(df.iloc[0]['smiles']))

    def test_load_mutag_csv_rejects_empty_smiles(self):
        try:
            _load_csv = self._import_load_csv()
        except ModuleNotFoundError as e:
            self.skipTest(f'generate_vocab_rules deps missing: {e}')

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


class TestUserDownloadedArtifacts(unittest.TestCase):
    """Optional: validate artifacts the user exported on HPC."""

    DOWNLOADS = Path('/Users/apurvakokate/Downloads')

    def test_user_export_if_present(self):
        csv_p = self.DOWNLOADS / 'mutag_0 (1).csv'
        splits_p = self.DOWNLOADS / 'mutag_0_splits (1).pkl'
        maps_p = self.DOWNLOADS / 'mutag_0_index_maps (1).pkl'
        if not csv_p.is_file():
            self.skipTest('user export CSV not in Downloads')

        import pandas as pd
        df = pd.read_csv(csv_p)
        if len(df) == 2951 and 'conversion_ok' in df.columns:
            with self.assertRaises(MutagArtifactError):
                validate_mutag_artifacts(csv_p, splits_p, maps_p,
                                         dataset_size=2951)
        elif len(df) == 2937:
            info = validate_mutag_artifacts(csv_p, splits_p, maps_p,
                                            dataset_size=2951)
            self.assertEqual(info['n_graphs'], 2937)
            self.assertEqual(
                info['n_train'] + info['n_valid'] + info['n_test'], 2937)


if __name__ == '__main__':
    unittest.main(verbosity=2)
