"""Lightweight CI tests for analysis aggregation (no torch required)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.aggregate_experiments import (
    enrich_from_exp_dir,
    expand_posthoc_explainer_rows,
    filter_prediction_rows,
    normalize,
    parse_vocab_variant,
    resolve_family,
)
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

    def test_resolve_family_gsat_not_motifsat_via_model_type(self):
        """E3: model_type MotifSAT must not collapse base_gsat into motifsat."""
        meta = {'model_type': 'MotifSAT', 'motif_method': 'none'}
        self.assertEqual(
            resolve_family(meta, 'gsat/BBBP/fold0/rbrics/bb-GIN_enc-onehot'),
            'gsat')

    def test_enrich_vanilla_layout_from_path(self):
        """E4: vanilla/baselines layout fills dataset/backbone/fold from exp_dir."""
        df = pd.DataFrame([{
            'exp_dir': 'vanilla/BBBP/fold1/rbrics_old_filter/bb-GIN_enc-onehot_norm-l2',
            'vocab_variant': 'rbrics_old_filter',
            'auc': 0.72,
        }])
        out = enrich_from_exp_dir(df)
        self.assertEqual(out.iloc[0]['dataset'], 'BBBP')
        self.assertEqual(out.iloc[0]['backbone'], 'GIN')
        self.assertEqual(out.iloc[0]['fold'], 1)

    def test_synthetic_from_use_gt_not_path(self):
        """E2: vanilla GT runs use summary use_gt when path has no gt token."""
        df = pd.DataFrame([{
            'exp_dir': 'baselines/BBBP/fold0/rbrics/bb-GIN_enc-onehot',
            'vocab_variant': 'rbrics',
            'use_gt': True,
            'auc': 0.99,
        }])
        out = normalize(df)
        self.assertEqual(out.iloc[0]['synthetic'], 'gt')

    def test_parse_vocab_variant_splits_filter_and_relabel(self):
        """E10: bundled vocab_variant splits into base + flags."""
        self.assertEqual(parse_vocab_variant('rbrics_old_filter_relabelled'),
                         ('rbrics_old', True, True))
        self.assertEqual(parse_vocab_variant('all_fallback_bpe'), ('all_fallback_bpe', False, False))

    def test_filter_prediction_drops_baselines(self):
        """E8: baselines duplicate vanilla predictive metrics."""
        df = pd.DataFrame([
            {'family': 'vanilla', 'auc': 0.8},
            {'family': 'baselines', 'auc': 0.8},
            {'family': 'mose', 'auc': 0.7},
        ])
        out = filter_prediction_rows(df)
        self.assertEqual(set(out['family']), {'vanilla', 'mose'})

    def test_expand_posthoc_explainer_rows(self):
        """E5: GNNExplainer metrics become explainer-family rows."""
        df = pd.DataFrame([{
            'family': 'baselines',
            'dataset': 'BBBP',
            'backbone': 'GIN',
            'fold': 0,
            'vocab_variant': 'rbrics',
            'exp_dir': 'baselines/BBBP/fold0/rbrics/bb-GIN_enc-onehot',
            'pearson': float('nan'),
            'gnnexplainer_mean_pearson': 0.41,
            'gnnexplainer_mean_gt_roc_node_auc_mean': 0.55,
            'pgexplainer_mean_pearson': 0.33,
        }])
        out = expand_posthoc_explainer_rows(normalize(df))
        fams = set(out['family'])
        self.assertIn('gnnexplainer', fams)
        self.assertIn('pgexplainer', fams)
        gnn = out[out['family'] == 'gnnexplainer'].iloc[0]
        self.assertAlmostEqual(gnn['pearson'], 0.41)
        self.assertAlmostEqual(gnn['gt_roc_node_auc_mean'], 0.55)

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
        tbl = build(df, 'auc', mode='prediction')
        self.assertEqual(len(tbl), 2)
        idx = tbl.index
        self.assertIn(('BBBP', 'mose', 'real', 'rbrics', False), idx)
        self.assertIn(('BBBP', 'mose', 'gt', 'rbrics', False), idx)
        self.assertEqual(tbl.loc[('BBBP', 'mose', 'real', 'rbrics', False), 'GIN'], '0.550')
        self.assertEqual(tbl.loc[('BBBP', 'mose', 'gt', 'rbrics', False), 'GIN'], '0.980')

    def test_explanation_table_includes_posthoc(self):
        df = pd.DataFrame([
            {'dataset': 'BBBP', 'family': 'vanilla', 'backbone': 'GIN',
             'vocab_variant': 'rbrics', 'fold': 0, 'pearson': 0.2,
             'exp_dir': 'vanilla/BBBP/fold0/rbrics/bb-GIN_enc-onehot'},
            {'dataset': 'BBBP', 'family': 'baselines', 'backbone': 'GIN',
             'vocab_variant': 'rbrics', 'fold': 0,
             'exp_dir': 'baselines/BBBP/fold0/rbrics/bb-GIN_enc-onehot',
             'gnnexplainer_mean_pearson': 0.45},
        ])
        tbl = build(df, 'pearson', mode='explanation')
        fams = {idx[1] for idx in tbl.index}
        self.assertIn('vanilla', fams)
        self.assertIn('gnnexplainer', fams)
        self.assertNotIn('baselines', fams)


if __name__ == '__main__':
    unittest.main()
