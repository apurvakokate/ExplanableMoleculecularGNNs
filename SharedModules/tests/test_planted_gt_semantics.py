"""Regression tests for the 2026-08 planted-GT semantics change.

Three things changed and each has a way of silently regressing:

  1. node_label is FIRED-CLAUSE ONLY. The old Mode-1 'whole-rule' mask marked a motif's
     atoms even when its clause never fired, which penalised an explainer for correctly
     ignoring an atom that caused nothing. A regression here looks like a small, plausible
     GT-ROC drop rather than an error, so it is asserted directly.
  2. node_label_fired no longer exists (node_label carries those semantics).
  3. node_label_spurious is [N, S] — one column per class-1 correlate from the rule
     record's spurious_pos, in that order, each audited separately.
"""
import unittest

import torch
from torch_geometric.data import Data

from SharedModules.data.apply_gt import annotate_split


def _graph(smi, n, edges):
    ei = (torch.tensor(edges, dtype=torch.long).t() if edges
          else torch.zeros((2, 0), dtype=torch.long))
    d = Data(x=torch.ones(n, 1), edge_index=ei, y=torch.tensor([0.]))
    d.smiles = smi
    return d


class TestPlantedNodeGT(unittest.TestCase):
    """Rule: (*C=O ^ ring:benzene) v (*Cl ^ ring:pip)."""

    RULE = [{'*C=O', 'ring:benzene'}, {'*Cl', 'ring:pip'}]
    # A: benzene + C=O + Cl but NO piperidine -> clause 1 fires, clause 2 does NOT.
    #    The Cl is present yet caused nothing here: it must not be ground truth.
    LOOKUP_A = {0: ('ring:benzene', 1), 1: ('ring:benzene', 1), 2: ('*C=O', 2),
                3: ('*Cl', 3), 4: ('*CC*', 4)}
    # B: both clauses fire.
    LOOKUP_B = {0: ('ring:benzene', 1), 1: ('*C=O', 2), 2: ('*Cl', 3),
                3: ('ring:pip', 5), 4: ('*CC*', 4)}
    EDGES = [[0, 1], [1, 2], [2, 3], [3, 4]]

    def _run(self, spurious_motifs=('*CC*',)):
        gl = [_graph('A', 5, self.EDGES), _graph('B', 5, self.EDGES)]
        return annotate_split(gl, self.RULE, {'A': self.LOOKUP_A, 'B': self.LOOKUP_B},
                              relabel=True, spurious_motifs=list(spurious_motifs),
                              family_motifs=set())

    def test_unfired_clause_atoms_are_not_ground_truth(self):
        a, _ = self._run()
        self.assertEqual(a[0].node_label.tolist(), [1., 1., 1., 0., 0.],
                         'atom 3 (*Cl) belongs to a clause that never fired on A — '
                         'marking it is the removed Mode-1 behaviour')

    def test_node_label_is_max_over_fired_clause_columns(self):
        a, _ = self._run()
        for d in a:
            self.assertTrue(torch.equal(d.node_label, d.node_label_clauses.max(dim=1).values))

    def test_unfired_clause_column_is_all_zero(self):
        a, _ = self._run()
        self.assertEqual(a[0].node_label_clauses[:, 1].sum().item(), 0.0)
        self.assertGreater(a[1].node_label_clauses[:, 1].sum().item(), 0.0)

    def test_node_label_fired_attribute_is_gone(self):
        a, _ = self._run()
        for d in a:
            self.assertIsNone(getattr(d, 'node_label_fired', None))

    def test_edge_label_derives_from_fired_atoms(self):
        a, _ = self._run()
        for d in a:
            for e in range(d.edge_index.size(1)):
                if d.edge_label[e] > 0:
                    u, v = d.edge_index[0, e], d.edge_index[1, e]
                    self.assertTrue(d.node_label[u] > 0 and d.node_label[v] > 0)

    def test_spurious_is_one_column_per_motif_in_record_order(self):
        a, stats = self._run(spurious_motifs=('*CC*', '*Cl'))
        for d in a:
            self.assertEqual(tuple(d.node_label_spurious.shape), (5, 2))
            self.assertEqual(d.spurious_motif_keys.split('\t'), ['*CC*', '*Cl'])
        # column 0 == *CC* (atom 4), column 1 == *Cl (atom 3 on A, atom 2 on B)
        self.assertEqual(a[0].node_label_spurious[:, 0].nonzero().view(-1).tolist(), [4])
        self.assertEqual(a[0].node_label_spurious[:, 1].nonzero().view(-1).tolist(), [3])
        self.assertEqual(a[1].node_label_spurious[:, 1].nonzero().view(-1).tolist(), [2])
        self.assertEqual(stats['spurious_motifs'], ['*CC*', '*Cl'])
        self.assertEqual(stats['n_graphs_with_spurious'], [2, 2])

    def test_no_spurious_motifs_yields_empty_audit(self):
        a, stats = self._run(spurious_motifs=())
        self.assertEqual(stats['spurious_motifs'], [])
        for d in a:
            self.assertEqual(d.node_label_spurious.sum().item(), 0.0)


class TestGtRocRejectsFlattening(unittest.TestCase):
    """A 2-D mask must never be silently flattened against an N-long attribution."""

    def test_explainer_roc_rejects_2d_node_label(self):
        from SharedModules.evaluation.motif_eval import explainer_roc_vs_gt
        with self.assertRaises(ValueError):
            explainer_roc_vs_gt(
                node_att=torch.rand(5),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_label=torch.zeros(0),
                level='node',
                node_label=torch.zeros(5, 2))


# NOTE: TestSpuriousScopeIsolation was removed 2026-08-21. It guarded a scope-leak in
# explainability_summary_fields' prefix-matched per-motif spurious block, which no longer
# exists: pipeline.py stopped emitting SpurROC/FamROC entirely once they became contrasts
# against the cause (analysis/evaluate.py::_contrast_vs_cause is the sole producer). The
# leak it tested is unreachable, so the test asserted behaviour that is now absent by design.


if __name__ == '__main__':
    unittest.main()
