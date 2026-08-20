"""Rule Set 1 — Stage 5 (Save + Aggregate) unit tests.

Covers the aggregator logic that a correct run depends on to reach the tables:
  * parse_vocab_variant — tier-aware suffix split;
  * tier axis admits dnf_* keys (the dnf-collapse regression) and coerces junk to 'none';
  * real/gt axis derived from use_gt;
  * metric_map carries the three-way + Instance/Global GT-ROC (no silent metric drop).

NOTE (surfaced in chat): the single **dated, provenance-stamped, versioned** results CSV
and the per-paper **scope filters** are the RULED design (#1 commit-SHA+timestamp+hash, #2
keep-both-versioned) that is NOT yet implemented in aggregate_experiments.py — it emits
per-experiment tables only. Those tests are deliberately absent until that harvester exists.
"""
import unittest
import pandas as pd

from analysis.aggregate_experiments import (
    parse_vocab_variant, normalize, expand_posthoc_explainer_rows)

_BASE = dict(vocab_variant='fg_first', use_gt=True, family='baselines', fold=0,
             dataset='BBBP', backbone='GIN', threshold='off', model_type='VanillaGNN',
             synthetic='gt', node_encoder='onehot', conv_normalize='none',
             apply_layer_norm=False, epochs=10)


class Stage5ParseVocabVariant(unittest.TestCase):
    def test_base(self):
        self.assertEqual(parse_vocab_variant('fg_first'), ('fg_first', False, False, 'none'))

    def test_filter_and_relabel(self):
        self.assertEqual(parse_vocab_variant('rbrics_filter_relabelled'),
                         ('rbrics', True, True, 'none'))

    def test_legacy_tier_suffix(self):
        self.assertEqual(parse_vocab_variant('fg_first_relabelled_easy'),
                         ('fg_first', False, True, 'easy'))


class Stage5TierAxis(unittest.TestCase):
    def _tier(self, gt_tier):
        df = pd.DataFrame([{**_BASE, 'gt_tier': gt_tier}])
        return normalize(df.copy())['tier'].iloc[0]

    def test_dnf_key_admitted_not_collapsed(self):
        self.assertEqual(self._tier('dnf_k2_r1'), 'dnf_k2_r1')
        self.assertEqual(self._tier('dnf_k3_r2_rare'), 'dnf_k3_r2_rare')

    def test_legacy_tier_admitted(self):
        self.assertEqual(self._tier('easy'), 'easy')

    def test_junk_coerced_to_none(self):
        self.assertEqual(self._tier('garbage_tier'), 'none')


class Stage5RealGtAxis(unittest.TestCase):
    def test_use_gt_sets_synthetic(self):
        df = pd.DataFrame([{**_BASE, 'gt_tier': 'dnf_k2_r1', 'use_gt': True},
                           {**_BASE, 'gt_tier': 'none', 'use_gt': False}])
        out = normalize(df.copy())
        self.assertEqual(out['synthetic'].tolist(), ['gt', 'real'])


class Stage5MetricMapCompleteness(unittest.TestCase):
    """expand_posthoc_explainer_rows must map the three-way + Instance/Global keys —
    a metric computed and saved but absent from the map is silently dropped."""

    def test_three_way_and_dnf_metrics_mapped(self):
        cols = {f'gnnexplainer_mean_{m}_auc_mean': 0.8 for m in
                ('gt_roc_node', 'spurious_roc_node',
                 'family_roc_node', 'instance_gt_roc_node', 'global_gt_roc_node')}
        df = pd.DataFrame([{'family': 'baselines', 'dataset': 'BBBP', **cols}])
        out = expand_posthoc_explainer_rows(df)
        row = out[(out['family'] == 'gnnexplainer') & (out['explainer_agg'] == 'mean')]
        self.assertEqual(len(row), 1)
        for m in ('spurious_roc_node_auc_mean', 'family_roc_node_auc_mean',
                  'instance_gt_roc_node_auc_mean', 'global_gt_roc_node_auc_mean'):
            self.assertAlmostEqual(row[m].iloc[0], 0.8, msg=f'{m} not mapped (silent drop)')


if __name__ == '__main__':
    unittest.main(verbosity=2)
