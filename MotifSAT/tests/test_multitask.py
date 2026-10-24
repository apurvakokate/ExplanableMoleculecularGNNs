"""Multi-task (MultiLabel) correctness of the evaluation and data paths.

Covers: impact = mean over task outputs of |Δp| (motif_eval._single_pred/_pred_delta,
loo_impact, compute_motif_impact, occurrence_eval per-task columns) with single-output
results unchanged; compute_pos_weights one weight per task for OGB's [1, T] labels;
AttachCopies through a mutag index map with explicit-H propagation; the transform hook on the
mutag / OGB dataset wrappers.

Run:  pytest MotifSAT/tests/test_multitask.py -v
"""
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from SharedModules.tests.pin_pkg_imports import MOTIFSAT_TOPLEVEL, pin_trainer_imports  # noqa: E402

pin_trainer_imports(ROOT / 'MotifSAT', ROOT, MOTIFSAT_TOPLEVEL)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from SharedModules.data.dataset import NUM_ATOM_TYPES, EDGE_FEAT_DIM  # noqa: E402
from model import GSAT  # noqa: E402

H = 16
DEG = torch.tensor([0., 4., 10., 3.])
CPU = torch.device('cpu')


def _graph(i, gen, n_tasks):
    x = torch.randn(6, NUM_ATOM_TYPES, generator=gen)
    ei = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    ei = torch.cat([ei, ei.flip(0)], 1)
    d = Data(x=x, edge_index=ei, edge_attr=torch.randn(12, EDGE_FEAT_DIM, generator=gen),
             nodes_to_motifs=torch.tensor([0, 0, 1, 0, 0, 1]),
             y=torch.full((1, n_tasks), float(i % 2)))
    d.nodes_to_copy = torch.tensor([0, 0, 0, 1, 1, 0])
    d.smiles = f'G{i}'
    return d


def _model(n_tasks):
    torch.manual_seed(0)
    return GSAT(x_dim=NUM_ATOM_TYPES, hidden_dim=H, num_layers=2, backbone_name='GIN',
                num_classes=n_tasks, deg=DEG, arch_style='gsat', init_r=0.9, final_r=0.7,
                decay_interval=10, decay_r=0.1, pna_scalers=False).eval()


class TestMultiTaskImpact(unittest.TestCase):
    def test_single_output_unchanged(self):
        from SharedModules.evaluation.motif_eval import _single_pred, _single_prob, _pred_delta
        g = torch.Generator().manual_seed(0)
        m = _model(1); d = _graph(0, g, 1)
        for tt in ('BinaryClass', 'Regression'):
            w = torch.rand(6, generator=g)
            v = _single_pred(m, d, CPU, tt, node_weights=w)
            self.assertEqual(v.shape, (1,))
            self.assertEqual(float(v[0]), _single_prob(m, d, CPU, tt, node_weights=w))
            w2 = w.clone(); w2[:2] = 0
            self.assertEqual(_pred_delta(v, _single_pred(m, d, CPU, tt, node_weights=w2)),
                             abs(_single_prob(m, d, CPU, tt, node_weights=w)
                                 - _single_prob(m, d, CPU, tt, node_weights=w2)))

    def test_multilabel_impact_is_mean_over_tasks(self):
        from SharedModules.evaluation.motif_eval import compute_motif_impact, model_node_att_fn
        from SharedModules.evaluation.metrics import _model_forward
        from occurrence_eval import occurrence_tables
        g = torch.Generator().manual_seed(1)
        T = 3
        m = _model(T); gs = [_graph(i, g, T) for i in range(4)]
        voc = types.SimpleNamespace(motif_list=['A', 'B'], motif_lengths=[2, 1])
        ref = compute_motif_impact(m, gs, voc, CPU, task_type='MultiLabel')
        att = model_node_att_fn(m, CPU)
        d = gs[2]; W = att(d).view(-1).float()
        Wm = W.clone(); Wm[d.nodes_to_motifs == 0] = 0
        p0 = torch.sigmoid(_model_forward(m, d.clone(), node_weights=W).view(-1)).detach().numpy()
        p1 = torch.sigmoid(_model_forward(m, d.clone(), node_weights=Wm).view(-1)).detach().numpy()
        self.assertEqual(p0.shape, (T,))
        want = float(np.mean(np.abs(p0 - p1)))
        got = ref[0]['impact_values'][ref[0]['graph_indices'].index(2)]
        self.assertAlmostEqual(got, want, places=6)
        self.assertNotAlmostEqual(got, float(abs(p0[0] - p1[0])), places=6)  # not task 0 only
        occ, allm = occurrence_tables(m, gs, CPU, 'MultiLabel')
        for col in ('impact_t0', 'impact_t1', 'impact_t2'):
            self.assertIn(col, allm.columns); self.assertIn(col, occ.columns)
        row = allm[(allm.graph_idx == 2) & (allm.motif_id == 0)].iloc[0]
        self.assertAlmostEqual(row.impact, want, places=5)
        self.assertAlmostEqual(row.impact, np.mean([row.impact_t0, row.impact_t1, row.impact_t2]),
                               places=6)
        for col, j in (('impact_t0', 0), ('impact_t2', 2)):
            self.assertAlmostEqual(row[col], abs(p0[j] - p1[j]), places=5)

    def test_binary_occurrence_has_no_task_columns(self):
        from occurrence_eval import occurrence_tables
        g = torch.Generator().manual_seed(2)
        occ, allm = occurrence_tables(_model(1), [_graph(i, g, 1) for i in range(2)],
                                      CPU, 'BinaryClass')
        self.assertFalse(any(c.startswith('impact_t') for c in allm.columns))


class TestPosWeightsMultiTask(unittest.TestCase):
    def test_one_weight_per_task_with_missing(self):
        from SharedModules.data.loader import compute_pos_weights
        nan = float('nan')
        ys = [[1, 0, nan], [0, 0, 1], [0, 1, 1], [0, nan, 0]]
        ds = [Data(y=torch.tensor([y])) for y in ys]          # OGB layout: y is [1, T]
        w = compute_pos_weights(ds)
        self.assertEqual(w.tolist(), [3.0, 2.0, 0.5])         # neg/pos per task, NaN ignored
        single = [Data(y=torch.tensor([[v]])) for v in (1., 0., 0.)]
        self.assertEqual(compute_pos_weights(single).tolist(), [2.0])


class TestAttachCopiesMutag(unittest.TestCase):
    # mapped SMILES with explicit H: atoms 0 C, 1-3 H, 4 C, 5-7 H (two copies of motif 'C')
    SMI = '[C:1]([H:3])([H:4])([H:5])[C:2]([H:6])([H:7])[H:8]'

    def _graph(self):
        # graph node order differs from SMILES order: node 0 = SMILES C4, node 1 = SMILES C0;
        # H nodes 2-4 bonded to node 1, 5-7 bonded to node 0
        pairs = [(0, 1), (1, 2), (1, 3), (1, 4), (0, 5), (0, 6), (0, 7)]
        ei = torch.tensor(pairs + [(b, a) for a, b in pairs]).t()
        return Data(x=torch.zeros(8, 2), edge_index=ei, smiles=self.SMI,
                    nodes_to_motifs=torch.zeros(8, dtype=torch.long))

    def test_index_map_and_h_propagation(self):
        from SharedModules.data.dataset import AttachCopies
        rec = {self.SMI: [('C', 0, [0, 1, 2, 3]), ('C', 0, [4, 5, 6, 7])]}
        out = AttachCopies(rec, index_maps={self.SMI: {0: 4, 1: 0}})(self._graph())
        self.assertEqual(out.nodes_to_copy.tolist(), [1, 0, 0, 0, 0, 1, 1, 1])

    def test_unmapped_heavy_atom_raises(self):
        from SharedModules.data.dataset import AttachCopies
        rec = {self.SMI: [('C', 0, [0, 1, 2, 3]), ('C', 0, [4, 5, 6, 7])]}
        with self.assertRaises(ValueError):
            AttachCopies(rec, index_maps={self.SMI: {1: 0}})(self._graph())


class TestDatasetTransformHook(unittest.TestCase):
    def _tf(self, d):
        d.tf_applied = torch.tensor([1])
        return d

    def test_ogb_and_mutag_wrappers_apply_transform(self):
        from SharedModules.data.loader import OGBMotifDataset, MutagTUDataset
        gs = [Data(x=torch.zeros(3, 2), edge_index=torch.zeros(2, 0, dtype=torch.long),
                   y=torch.tensor([[1., 0.]])) for _ in range(2)]
        ogb = OGBMotifDataset(gs, [0, 1], vocab=None, transform=self._tf)
        self.assertEqual(int(ogb[1].tf_applied), 1)
        mut = MutagTUDataset(gs, vocab=None, transform=self._tf)
        self.assertEqual(int(mut[0].tf_applied), 1)
        self.assertFalse(hasattr(OGBMotifDataset(gs, [0], vocab=None)[0], 'tf_applied'))


if __name__ == '__main__':
    unittest.main()
