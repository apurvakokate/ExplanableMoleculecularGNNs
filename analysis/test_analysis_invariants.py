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
    expand_posthoc_explainer_rows,
    family_of,
    filter_prediction_rows,
    normalize,
    parse_vocab_variant,
)
from analysis.make_results_table import build


def _run(**kw) -> dict:
    """A complete run summary with every field normalize() now requires
    (analysis reads axes from fields, no path fallback). Override per test."""
    base = dict(
        family='mose', dataset='BBBP', backbone='GIN', fold=0,
        vocab_variant='rbrics', node_encoder='onehot', conv_normalize='none',
        use_gt=False, epochs=500, w_feat=True, w_message=False, w_readout=True,
        exp_dir='mose/BBBP/fold0/rbrics/GIN_tag',
    )
    base.update(kw)
    return base


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

    def test_axes_read_from_fields(self):
        """Every axis is read directly from the run's summary fields."""
        out = normalize(pd.DataFrame([_run(
            family='mose', node_encoder='onehot', conv_normalize='l2',
            use_gt=False, epochs=500, w_feat=True, w_message=False, w_readout=True)]))
        r = out.iloc[0]
        self.assertEqual(r['family'], 'mose')
        self.assertEqual(r['features'], 'onehot')
        self.assertEqual(r['norm'], 'l2')
        self.assertEqual(r['synthetic'], 'real')
        self.assertEqual(r['injection'], '101')   # w_feat/w_readout on, w_message off

    def test_features_from_node_encoder_not_path(self):
        """features = authoritative node_encoder, even if the path token differs."""
        out = normalize(pd.DataFrame([_run(
            dataset='ogbg-molhiv', node_encoder='atom_encoder',
            exp_dir='mose/ogbg-molhiv/fold0/rbrics/bb-GIN_enc-onehot')]))
        self.assertEqual(out.iloc[0]['features'], 'atom_encoder')

    def test_injection_na_for_vanilla(self):
        out = normalize(pd.DataFrame([_run(family='vanilla',
                                           w_feat=False, w_message=False, w_readout=False)]))
        self.assertEqual(out.iloc[0]['injection'], 'na')

    def test_normalize_fails_fast_without_node_encoder(self):
        """No path fallback: a summary missing node_encoder must raise."""
        row = _run()
        del row['node_encoder']
        with self.assertRaises(ValueError):
            normalize(pd.DataFrame([row]))

    def test_normalize_fails_fast_without_family(self):
        row = _run()
        del row['family']
        with self.assertRaises(ValueError):
            normalize(pd.DataFrame([row]))

    def test_vanilla_null_injection_flags_do_not_abort_collect(self):
        """vanilla/baselines are injection-agnostic and legitimately record
        w_feat/w_message/w_readout as null; that must NOT fail-fast (else the
        whole collect aborts, since vanilla always runs)."""
        row = _run(family='vanilla', w_feat=None, w_message=None, w_readout=None,
                   exp_dir='vanilla/BBBP/fold0/rbrics/bb-GIN_enc-onehot')
        out = normalize(pd.DataFrame([row]))
        self.assertEqual(out.iloc[0]['injection'], 'na')

    def test_antehoc_null_injection_flags_still_fail_fast(self):
        """Injection-bearing families (mose/motifsat/gsat) MUST record the
        injection flags; a null there is still a broken summary."""
        row = _run(family='mose', w_feat=None)
        with self.assertRaises(ValueError):
            normalize(pd.DataFrame([row]))

    def test_family_of_reads_field(self):
        self.assertEqual(family_of({'family': 'mose'}), 'mose')
        self.assertEqual(family_of({'family': 'base_gsat'}), 'gsat')  # normalised

    def test_family_of_fails_fast_when_missing(self):
        with self.assertRaises(ValueError):
            family_of({'model_type': 'MOSE-GNN', 'motif_method': 'mose'})

    def test_synthetic_from_use_gt_field(self):
        """synthetic comes from the recorded use_gt field, not the path."""
        out = normalize(pd.DataFrame([_run(family='baselines', use_gt=True,
                        exp_dir='baselines/BBBP/fold0/rbrics/bb-GIN_enc-onehot')]))
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
        df = pd.DataFrame([_run(
            family='baselines',
            exp_dir='baselines/BBBP/fold0/rbrics/bb-GIN_enc-onehot',
            pearson=float('nan'),
            gnnexplainer_mean_pearson=0.41,
            gnnexplainer_mean_gt_roc_node_auc_mean=0.55,
            pgexplainer_mean_pearson=0.33,
        )])
        out = expand_posthoc_explainer_rows(normalize(df))
        fams = set(out['family'])
        self.assertIn('gnnexplainer', fams)
        self.assertIn('pgexplainer', fams)
        gnn = out[out['family'] == 'gnnexplainer'].iloc[0]
        self.assertAlmostEqual(gnn['pearson'], 0.41)
        self.assertAlmostEqual(gnn['gt_roc_node_auc_mean'], 0.55)

    def test_mutagenicity_vs_mutag_paths_differ(self):
        df = pd.DataFrame([
            _run(dataset='Mutagenicity', vocab_variant='all_fallback_bpe',
                 exp_dir='mose/Mutagenicity/fold0/x'),
            _run(dataset='mutag', vocab_variant='all_fallback_bpe',
                 exp_dir='mose/mutag/fold0/x'),
        ])
        out = normalize(df)
        self.assertEqual(set(out['dataset']), {'Mutagenicity', 'mutag'})

    def test_pivot_collapses_folds_only(self):
        """Five folds with distinct values → mean ± std; vocab axes stay separate."""
        rows = []
        for fold in range(5):
            rows.append(_run(vocab_variant='rbrics', fold=fold,
                             exp_dir=f'mose/BBBP/fold{fold}/rbrics/GIN',
                             auc=0.50 + fold * 0.01))
            rows.append(_run(vocab_variant='rbrics_filter', fold=fold,
                             exp_dir=f'mose/BBBP/fold{fold}/rbrics_filter/GIN',
                             auc=0.60 + fold * 0.01))
        tbl = build(pd.DataFrame(rows), 'auc', mode='prediction')
        self.assertEqual(len(tbl), 2)
        self.assertIn('0.520 ± 0.016', tbl.loc[('BBBP', 'mose', 'real', 'rbrics', False), 'GIN'])
        self.assertIn('0.620 ± 0.016', tbl.loc[('BBBP', 'mose', 'real', 'rbrics', True), 'GIN'])

    def test_build_tidy_collapses_only_folds_not_configs(self):
        """Runs in the SAME coarse cell that differ in hyperparameter config
        (noise/info_loss_level + hp-hash in the run tag) must stay SEPARATE tidy
        rows; only same-config runs across folds may collapse."""
        from analysis.aggregate_experiments import (
            build_tidy, DEFAULT_EXPERIMENT_AXES, PERF)
        rows = []
        # config A (IB off) across two folds
        for f in (0, 1):
            rows.append(_run(
                family='motifsat', motif_method='readout',
                exp_dir=f'motifsat/rbrics/BBBP/fold{f}/GIN_readout_onehot_'
                        'norm-l2_wf+wm+wr_noise-none_il-none_real_ep500_'
                        'rbrics_L3_h64_lr0.001_hp-aaaa',
                fold=f, auc=0.9, gt_roc_node_auc_mean=0.50))
        # config B (IB on), fold 0, SAME coarse cell (different run dir → separate)
        rows.append(_run(
            family='motifsat', motif_method='readout',
            exp_dir='motifsat/rbrics/BBBP/fold0/GIN_readout_onehot_norm-l2_'
                    'wf+wm+wr_noise-motif_il-motif_real_ep500_rbrics_'
                    'L3_h64_lr0.001_hp-bbbb',
            fold=0, auc=0.9, gt_roc_node_auc_mean=0.87))
        tidy = build_tidy(normalize(pd.DataFrame(rows)),
                          [PERF, 'gt_roc_node_auc_mean'], DEFAULT_EXPERIMENT_AXES)
        ms = tidy[(tidy.family == 'motifsat') & (tidy.dataset == 'BBBP')
                  & (tidy.backbone == 'GIN')]
        # two distinct configs → two rows, NOT one averaged 0.685 row
        self.assertEqual(len(ms), 2)
        gts = sorted(round(float(v), 3)
                     for v in ms['gt_roc_node_auc_mean__mean'])
        self.assertEqual(gts, [0.50, 0.87])
        # config A collapsed its two folds into one row
        a = ms[ms['gt_roc_node_auc_mean__mean'].round(3) == 0.50].iloc[0]
        self.assertEqual(int(a['n_folds']), 2)

    def test_pivot_does_not_collapse_families_or_synthetic(self):
        df = pd.DataFrame([
            _run(vocab_variant='rbrics', auc=0.55,
                 exp_dir='mose/BBBP/fold0/rbrics/GIN_onehot_norm-l2_real_rbrics'),
            _run(vocab_variant='rbrics_relabelled', auc=0.98, use_gt=True,
                 exp_dir='mose/BBBP/fold0/rbrics_relabelled/GIN_onehot_norm-l2_gt_rbrics'),
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
            _run(family='vanilla', pearson=0.2,
                 exp_dir='vanilla/BBBP/fold0/rbrics/bb-GIN_enc-onehot'),
            _run(family='baselines', gnnexplainer_mean_pearson=0.45,
                 exp_dir='baselines/BBBP/fold0/rbrics/bb-GIN_enc-onehot'),
        ])
        tbl = build(df, 'pearson', mode='explanation')
        fams = {idx[1] for idx in tbl.index}
        self.assertIn('vanilla', fams)
        self.assertIn('gnnexplainer', fams)
        self.assertNotIn('baselines', fams)

    def test_expand_posthoc_includes_pooled_metrics(self):
        df = pd.DataFrame([
            _run(family='baselines',
                 gnnexplainer_mean_pearson=0.4,
                 gnnexplainer_mean_pearson_all=0.5,
                 gnnexplainer_mean_gt_roc_node_auc_mean_all=0.65,
                 exp_dir='baselines/BBBP/fold0/rbrics/bb-GIN_enc-onehot'),
        ])
        out = expand_posthoc_explainer_rows(df)
        row = out[out['family'] == 'gnnexplainer'].iloc[0]
        self.assertEqual(row['pearson_all'], 0.5)
        self.assertEqual(row['gt_roc_node_auc_mean_all'], 0.65)


if __name__ == '__main__':
    unittest.main()
