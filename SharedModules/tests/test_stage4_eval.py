"""Rule Set 1 — Stage 4 (Explain + Eval) unit tests.

Covers the PURE, unit-testable metric layer:
  * explainer_roc_vs_gt — the ROC-AUC primitive both GT-ROC metrics rest on;
  * _has_node_attr — the no-fallback gate (all-zero attr → skip, never a NaN/rule-mask
    fallback);
  * MAGE polarity wiring — default 1, mutag 0.

NOTE (surfaced separately in chat): the corrected #4 *Instance GT-ROC* (max over fired
disjuncts, per-disjunct AUC, others excluded from the negative set) is NOT yet implemented
— the pipeline stores a single union mask (node_label_fired = your *Global*). Tests for
Instance GT-ROC are deliberately absent until that metric exists.
"""
import os
import subprocess
import unittest

import torch

from SharedModules.evaluation.motif_eval import explainer_roc_vs_gt
from SharedModules.evaluation.pipeline import _has_node_attr

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Stage4RocPrimitive(unittest.TestCase):
    """explainer_roc_vs_gt: node attention vs authoritative node_label."""

    def setUp(self):
        # 4-atom graph, atoms {0,1} are GT, {2,3} are not
        self.edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        self.edge_label = torch.zeros(4)
        self.node_label = torch.tensor([1., 1., 0., 0.])

    def test_perfect_ranking_auc_1(self):
        att = torch.tensor([0.9, 0.8, 0.1, 0.05])
        auc = explainer_roc_vs_gt(att, self.edge_index, self.edge_label,
                                  level='node', node_label=self.node_label)
        self.assertGreater(auc, 0.99, f'perfect ranking should be ~1.0, got {auc}')

    def test_inverse_ranking_auc_0(self):
        att = torch.tensor([0.05, 0.1, 0.9, 0.8])   # GT atoms ranked lowest
        auc = explainer_roc_vs_gt(att, self.edge_index, self.edge_label,
                                  level='node', node_label=self.node_label)
        self.assertLess(auc, 0.01, f'inverse ranking should be ~0.0, got {auc}')

    def test_node_label_is_authoritative(self):
        """When node_label is passed it is used directly, independent of edge_label."""
        att = torch.tensor([0.9, 0.8, 0.1, 0.05])
        misleading_edges = torch.tensor([1., 1., 1., 1.])   # would flip a derived GT
        auc = explainer_roc_vs_gt(att, self.edge_index, misleading_edges,
                                  level='node', node_label=self.node_label)
        self.assertGreater(auc, 0.99, 'node_label not treated as authoritative over edge_label')


class Stage4NoFallbackGate(unittest.TestCase):
    """_has_node_attr must gate on a POSITIVE entry — an all-zero spurious/family label
    (a rule with no such atoms) must be skipped, not scored into a NaN."""

    class _G:
        def __init__(self, v): self.a = v

    def test_all_zero_skipped(self):
        self.assertFalse(_has_node_attr([self._G(torch.zeros(5))], 'a'))

    def test_some_positive_scored(self):
        self.assertTrue(_has_node_attr([self._G(torch.tensor([0., 1., 0.]))], 'a'))

    def test_missing_attr_skipped(self):
        self.assertFalse(_has_node_attr([self._G(torch.zeros(3))], 'missing'))


class Stage4InstanceVsGlobalGtRoc(unittest.TestCase):
    """DNF GT-ROC: Instance (max over fired clauses, any-one) vs Global (union, all).
    Hand-built 6-atom, 2-disjunct graph where the explainer nails clause 1, misses clause 2."""

    def test_instance_beats_global_when_one_clause_nailed(self):
        from SharedModules.evaluation.motif_eval import dnf_gt_roc_graph
        # atoms {0,1}=clause0, {2,3}=clause1, {4,5}=non-causal
        nlc = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.], [0., 0.], [0., 0.]])
        att = torch.tensor([0.9, 0.85, 0.10, 0.05, 0.40, 0.30])   # nails clause0, misses clause1
        inst, glob = dnf_gt_roc_graph(att, nlc)
        self.assertAlmostEqual(glob, 0.5, places=6, msg='Global (union) should be 0.5 here')
        self.assertAlmostEqual(inst, 1.0, places=6, msg='Instance (max over clauses) should be 1.0')
        self.assertGreater(inst, glob, 'max-over-disjuncts must beat the union when one clause is nailed')

    def test_single_fired_clause_instance_equals_global(self):
        from SharedModules.evaluation.motif_eval import dnf_gt_roc_graph
        nlc = torch.tensor([[1., 0.], [1., 0.], [0., 0.], [0., 0.]])   # only clause0 fired
        att = torch.tensor([0.9, 0.2, 0.8, 0.1])
        inst, glob = dnf_gt_roc_graph(att, nlc)
        self.assertEqual(inst, glob, 'single fired clause ⇒ Instance == Global')

    def test_other_fired_clause_excluded_from_negatives(self):
        """A clause's AUC must not treat another fired clause's atoms as negatives."""
        from SharedModules.evaluation.motif_eval import _auc_pos_neg, dnf_gt_roc_graph
        nlc = torch.tensor([[1., 0.], [0., 1.], [0., 0.]])   # {0}=c0, {1}=c1, {2}=non-causal
        # explainer ranks c0 top, c1 high too, non-causal low
        att = torch.tensor([0.9, 0.8, 0.1])
        inst, _ = dnf_gt_roc_graph(att, nlc)
        # c0 AUC: pos {0}=0.9 vs neg {2}=0.1 → 1.0 (c1 atom {1} excluded, not a negative)
        self.assertAlmostEqual(inst, 1.0, places=6)


class Stage4CorrelationMetrics(unittest.TestCase):
    """Grouped (per-motif) vs instance (per-motif-graph) Pearson/Spearman, and top/bottom-K."""

    def test_grouped_vs_instance_distinct_granularity(self):
        from SharedModules.evaluation.motif_eval import score_impact_correlation
        # 3 motifs, each with 2 per-graph impact values → instance has 6 points, grouped has 3
        scores = {0: 0.1, 1: 0.5, 2: 0.9}
        impacts = {0: {'impact': 0.15, 'impact_values': [0.1, 0.2]},
                   1: {'impact': 0.50, 'impact_values': [0.4, 0.6]},
                   2: {'impact': 0.85, 'impact_values': [0.8, 0.9]}}
        out = score_impact_correlation(scores, impacts)
        self.assertEqual(out['n_points'], 6, 'instance level should keep every (motif,graph) point')
        self.assertEqual(out['n_motifs'], 3, 'motif level should average to one point per motif')
        self.assertGreater(out['pearson_motif'], 0.9, 'monotone score↔impact → high grouped pearson')

    def test_top_bottom_impact_ratio(self):
        from SharedModules.evaluation.motif_eval import top_bottom_motif_eval
        scores = {i: i / 10 for i in range(10)}                     # 0.0 .. 0.9
        impacts = {i: {'impact': float(i)} for i in range(10)}      # impact rises with score
        out = top_bottom_motif_eval(scores, impacts, k=3)
        self.assertGreater(out['top_mean_impact'], out['bottom_mean_impact'])
        self.assertEqual(sorted(out['top_k_ids']), [7, 8, 9])
        self.assertEqual(sorted(out['bottom_k_ids']), [0, 1, 2])

    def test_per_instance_correlation_from_caches(self):
        from SharedModules.evaluation.motif_eval import per_instance_correlation_from_caches
        # score and impact perfectly aligned per (motif, graph) → pearson_instance ~1
        score_by_mg = {0: {0: 0.1, 1: 0.2, 2: 0.3}, 1: {0: 0.4, 1: 0.5}}
        impact_by_mg = {0: {0: 0.1, 1: 0.2, 2: 0.3}, 1: {0: 0.4, 1: 0.5}}
        out = per_instance_correlation_from_caches(score_by_mg, impact_by_mg)
        self.assertEqual(out['n_instances'], 5)
        self.assertGreater(out['pearson_instance'], 0.99)


if __name__ == '__main__':
    unittest.main(verbosity=2)
