"""Lightweight CI tests for analysis aggregation (no torch required)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.aggregate_experiments import normalize, resolve_family
from analysis.make_results_table import build


def _collapse_redundant_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror SharedModules.data.dataset_routing.collapse_redundant_folds."""
    single = {'mutag'} | {f'ogbg-{n}' for n in (
        'molhiv', 'molbace', 'molbbbp', 'molclintox', 'moltox21',
        'molsider', 'molesol', 'molfreesolv', 'mollipo',
    )}
    ds = df['dataset'].astype(str)
    fold = pd.to_numeric(df.get('fold'), errors='coerce')
    return df.loc[~(ds.isin(single) & (fold > 0))].copy()


class TestAnalysisInvariants(unittest.TestCase):
    def test_collapse_redundant_folds(self):
        df = pd.DataFrame([
            {'dataset': 'mutag', 'fold': 0, 'auc': 0.9},
            {'dataset': 'mutag', 'fold': 1, 'auc': 0.9},
            {'dataset': 'ogbg-molhiv', 'fold': 2, 'auc': 0.7},
            {'dataset': 'BBBP', 'fold': 1, 'auc': 0.8},
        ])
        out = _collapse_redundant_folds(df)
        self.assertEqual(len(out), 2)
        self.assertEqual(set(out['dataset']), {'mutag', 'BBBP'})

    def test_normalize_preserves_family_from_path(self):
        df = pd.DataFrame([{
            'exp_dir': 'mose/BBBP/fold0/all_fallback_bpe/enc-onehot_norm-l2_real',
            'dataset': 'BBBP',
            'backbone': 'GIN',
            'vocab_variant': 'all_fallback_bpe',
            'conv_normalize': 'l2',
            'motif_method': 'mose',
        }])
        out = normalize(df)
        self.assertEqual(out.iloc[0]['family'], 'mose')

    def test_resolve_family_mose(self):
        meta = {'model_type': 'MOSE-GNN', 'motif_method': 'mose'}
        self.assertEqual(resolve_family(meta, 'foo/mose/bar'), 'mose')

    def test_mutagenicity_vs_mutag_paths_differ(self):
        df = pd.DataFrame([
            {'exp_dir': 'mose/Mutagenicity/fold0/x', 'dataset': 'Mutagenicity',
             'vocab_variant': 'all_fallback_bpe', 'conv_normalize': 'l2',
             'motif_method': 'mose', 'loader_kind': 'csv'},
            {'exp_dir': 'mose/mutag/fold0/x', 'dataset': 'mutag',
             'vocab_variant': 'all_fallback_bpe', 'conv_normalize': 'l2',
             'motif_method': 'mose', 'loader_kind': 'tudataset_mutag'},
        ])
        out = normalize(df)
        self.assertEqual(set(out['dataset']), {'Mutagenicity', 'mutag'})

    def test_results_table_separates_real_and_gt(self):
        df = pd.DataFrame([
            {'dataset': 'BBBP', 'family': 'mose', 'backbone': 'GIN',
             'vocab_variant': 'rbrics', 'fold': 0, 'auc': 0.55,
             'exp_dir': 'mose/BBBP/fold0/rbrics/GIN_onehot_norm-l2_real_rbrics'},
            {'dataset': 'BBBP', 'family': 'mose', 'backbone': 'GIN',
             'vocab_variant': 'rbrics_relabelled', 'fold': 0, 'auc': 0.98,
             'exp_dir': 'mose/BBBP/fold0/rbrics_relabelled/GIN_onehot_norm-l2_gt_rbrics',
             'use_gt': True},
        ])
        tbl = build(df, 'auc')
        self.assertEqual(len(tbl), 2)
        idx = tbl.index
        self.assertIn(('BBBP', 'mose', 'real', 'rbrics'), idx)
        self.assertIn(('BBBP', 'mose', 'gt', 'rbrics_relabelled'), idx)
        self.assertEqual(tbl.loc[('BBBP', 'mose', 'real', 'rbrics'), 'GIN'], '0.550')
        self.assertEqual(tbl.loc[('BBBP', 'mose', 'gt', 'rbrics_relabelled'), 'GIN'], '0.980')


if __name__ == '__main__':
    unittest.main()
