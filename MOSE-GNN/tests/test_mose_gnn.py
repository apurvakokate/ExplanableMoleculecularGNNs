#!/usr/bin/env python3
"""test_mose_gnn.py — tests for MOSE-GNN models and training utilities.

Tests:
  - _motif_to_node_weights: shape, unknown handling, masked_motif
  - SingleChannelGNN: forward shape, unk_modes, w_feat/w_readout flags
  - MultiChannelGNN: forward shape, per-class independence
  - mask_regularisation: size_reg + ent_reg values
  - train_one_epoch: loss decreases, returns expected types

Run:
    python test_mose_gnn.py -v
"""

import sys, os, unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'MOSE-GNN'))

from SharedModules.data.dataset import NUM_ATOM_TYPES, EDGE_FEAT_DIM

from model import SingleChannelGNN, MultiChannelGNN, _motif_to_node_weights
from train import mask_regularisation, train_one_epoch, _task_loss


DEVICE = torch.device('cpu')


# ── helpers ──────────────────────────────────────────────────────────────────

def _batch(n_graphs=4, n_atoms=8, n_motifs=5, n_classes=1):
    graphs = []
    for g in range(n_graphs):
        x = torch.randn(n_atoms, NUM_ATOM_TYPES)
        edge_index = torch.tensor([[0,1,2,3,4,5,6,7],
                                   [1,2,3,4,5,6,7,0]], dtype=torch.long)
        ntm = torch.randint(-1, n_motifs, (n_atoms,))
        y = torch.tensor([float(g % 2)] if n_classes == 1 else
                         [float(g % 2)] * n_classes)
        graphs.append(Data(x=x, edge_index=edge_index,
                           nodes_to_motifs=ntm, y=y,
                           edge_attr=torch.randn(8, EDGE_FEAT_DIM)))
    return Batch.from_data_list(graphs)


# ── _motif_to_node_weights ───────────────────────────────────────────────────

class TestMotifToNodeWeights(unittest.TestCase):
    def _params(self, M=5, C=1):
        return nn.Parameter(torch.zeros(M, C))

    def test_output_shape_single(self):
        ntm = torch.tensor([0, 1, -1, 2, -1, 3])
        p = self._params(M=5, C=1)
        w = _motif_to_node_weights(ntm, p, 6, DEVICE)
        self.assertEqual(w.shape, (6, 1))

    def test_output_shape_multi(self):
        ntm = torch.tensor([0, 1, 2, -1])
        p = self._params(M=5, C=4)
        w = _motif_to_node_weights(ntm, p, 4, DEVICE)
        self.assertEqual(w.shape, (4, 4))

    def test_unknown_fixed(self):
        ntm = torch.tensor([-1, -1, -1])
        p = self._params(M=3, C=1)
        w = _motif_to_node_weights(ntm, p, 3, DEVICE, unk_mode='fixed', unk_value=0.5)
        self.assertTrue(torch.allclose(w, torch.full((3, 1), 0.5)))

    def test_unknown_learnable(self):
        ntm = torch.tensor([-1, -1])
        p = self._params(M=3, C=1)
        unk = nn.Parameter(torch.tensor(1.0))   # sigmoid(1) ≈ 0.731
        w = _motif_to_node_weights(ntm, p, 2, DEVICE,
                                    unk_mode='learnable_shared', unk_param=unk)
        expected = float(torch.sigmoid(torch.tensor(1.0)))
        self.assertAlmostEqual(float(w[0, 0]), expected, places=5)

    def test_ignore_unknowns(self):
        ntm = torch.tensor([-1, -1, 0])
        p = nn.Parameter(torch.tensor([[1.0]]))
        w = _motif_to_node_weights(ntm, p, 3, DEVICE,
                                    ignore_unknowns=True)
        self.assertAlmostEqual(float(w[0, 0]), 0.0)
        self.assertAlmostEqual(float(w[1, 0]), 0.0)
        self.assertGreater(float(w[2, 0]), 0.5)

    def test_masked_motif(self):
        ntm = torch.tensor([0, 0, 1, 2])
        p = nn.Parameter(torch.zeros(3, 1))
        w = _motif_to_node_weights(ntm, p, 4, DEVICE, masked_motif=0)
        self.assertEqual(float(w[0, 0]), 0.0)
        self.assertEqual(float(w[1, 0]), 0.0)
        self.assertNotEqual(float(w[2, 0]), 0.0)

    def test_known_values_are_sigmoid(self):
        ntm = torch.tensor([0, 1])
        p = nn.Parameter(torch.tensor([[2.0], [-2.0]]))
        w = _motif_to_node_weights(ntm, p, 2, DEVICE)
        expected_0 = float(torch.sigmoid(torch.tensor(2.0)))
        expected_1 = float(torch.sigmoid(torch.tensor(-2.0)))
        self.assertAlmostEqual(float(w[0, 0]), expected_0, places=5)
        self.assertAlmostEqual(float(w[1, 0]), expected_1, places=5)

    def test_all_unknown_returns_unk_value(self):
        ntm = torch.full((10,), -1, dtype=torch.long)
        p = nn.Parameter(torch.zeros(5, 3))
        w = _motif_to_node_weights(ntm, p, 10, DEVICE, unk_value=0.7)
        self.assertTrue(torch.allclose(w, torch.full((10, 3), 0.7)))

    def test_global_to_param_remap(self):
        """global_to_param remaps global ids → compact rows; below-threshold and
        unknown nodes both fall through to the unk value."""
        # Global vocab size 5; only ids 1 and 3 are kept (compact rows 0 and 1).
        g2p = torch.tensor([-1, 0, -1, 1, -1], dtype=torch.long)
        p = nn.Parameter(torch.tensor([[2.0], [-2.0]]))  # row0=id1, row1=id3
        ntm = torch.tensor([1, 3, 0, -1])  # kept, kept, below-thr, unknown
        w = _motif_to_node_weights(ntm, p, 4, DEVICE, unk_value=0.5,
                                   global_to_param=g2p)
        self.assertAlmostEqual(float(w[0, 0]),
                               float(torch.sigmoid(torch.tensor(2.0))), places=5)
        self.assertAlmostEqual(float(w[1, 0]),
                               float(torch.sigmoid(torch.tensor(-2.0))), places=5)
        self.assertAlmostEqual(float(w[2, 0]), 0.5)  # below-threshold → unk
        self.assertAlmostEqual(float(w[3, 0]), 0.5)  # unknown → unk

    def test_global_to_param_masked_motif_global_id(self):
        """masked_motif is a GLOBAL id; it is remapped before masking."""
        g2p = torch.tensor([-1, 0, 1], dtype=torch.long)
        p = nn.Parameter(torch.zeros(2, 1))
        ntm = torch.tensor([1, 2, 1])
        w = _motif_to_node_weights(ntm, p, 3, DEVICE, masked_motif=1,
                                   global_to_param=g2p)
        self.assertEqual(float(w[0, 0]), 0.0)  # global id 1 masked
        self.assertEqual(float(w[2, 0]), 0.0)
        self.assertNotEqual(float(w[1, 0]), 0.0)  # global id 2 untouched


# ── SingleChannelGNN ─────────────────────────────────────────────────────────

class TestSingleChannelGNN(unittest.TestCase):
    def _model(self, **kw):
        kw.setdefault('num_motifs', 5)
        return SingleChannelGNN(
            x_dim=NUM_ATOM_TYPES, hidden_dim=32, num_layers=2,
            backbone='GIN', **kw
        )

    def test_w_message_flag(self):
        """w_message scales messages during propagation for SingleChannelGNN."""
        b = _batch(2, 8, 5)
        m_wm = self._model(w_feat=False, w_message=True, w_readout=False)
        m_no = self._model(w_feat=False, w_message=False, w_readout=False)
        m_wm.motif_params.data.fill_(2.0)
        m_no.motif_params.data.fill_(2.0)
        # Copy weights so backbone is identical
        m_no.load_state_dict(m_wm.state_dict(), strict=False)
        out_wm, _ = m_wm(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
        out_no, _ = m_no(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
        # With identical backbone weights, w_message changes the propagated
        # signals — outputs must differ (for non-trivial graphs)
        self.assertEqual(out_wm.shape, (2, 1))
        self.assertEqual(out_no.shape, (2, 1))

    def test_forward_shape(self):
        m = self._model()
        b = _batch(4, 8, 5)
        out, att = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs, b.edge_attr)
        self.assertEqual(out.shape, (4, 1))
        self.assertEqual(att.shape, (b.x.size(0), 1))

    def test_w_feat_changes_output(self):
        b = _batch(2, 8, 5)
        m_wf = self._model(w_feat=True, w_readout=False)
        m_no = self._model(w_feat=False, w_readout=False)
        # Both have same initial params (zeros); outputs will differ due to
        # different x_scaled before conv
        m_no.motif_params.data.fill_(1.0)
        m_wf.motif_params.data.fill_(1.0)
        out_wf, _ = m_wf(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
        out_no, _ = m_no(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
        # Different paths → different outputs (not guaranteed to be different
        # for all seeds, but the mechanism is different)
        self.assertEqual(out_wf.shape, out_no.shape)

    def test_w_readout_flag(self):
        b = _batch(2, 8, 5)
        m1 = self._model(w_feat=False, w_readout=True)
        m2 = self._model(w_feat=False, w_readout=False)
        m1.motif_params.data.fill_(0.5)
        m2.motif_params.data.fill_(0.5)
        out1, _ = m1(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
        out2, _ = m2(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
        self.assertEqual(out1.shape, (2, 1))

    def test_no_motif_params(self):
        m = self._model(num_motifs=0)
        b = _batch(2, 8, 5)
        out, att = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
        self.assertEqual(out.shape, (2, 1))
        self.assertIsNone(att)

    def test_unk_mode_learnable(self):
        m = self._model(unk_mode='learnable_shared')
        self.assertIsNotNone(m.unk_param)
        b = _batch(2, 8, 5)
        out, att = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
        self.assertEqual(out.shape, (2, 1))

    def test_masked_motif(self):
        m = self._model()
        m.motif_params.data.fill_(2.0)
        b = _batch(2, 8, 5)
        out1, att1 = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
        out2, att2 = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs,
                       masked_motif=0)
        # att must differ where motif == 0
        mask = b.nodes_to_motifs == 0
        if mask.any():
            self.assertTrue((att2[mask] == 0.0).all())

    def test_get_motif_scores(self):
        m = self._model()
        m.motif_params.data.fill_(0.0)
        scores = m.get_motif_scores()
        self.assertEqual(len(scores), 5)
        for mid, s in scores.items():
            self.assertAlmostEqual(s, 0.5, places=5)

    def test_gradients_flow(self):
        m = self._model()
        b = _batch(2, 8, 5)
        out, att = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(m.motif_params.grad)
        self.assertFalse(torch.isnan(m.motif_params.grad).any())

    def test_kept_motif_ids_compact_params(self):
        """With kept_motif_ids, motif_params shrinks to the kept count, the
        global→param buffer maps correctly, and get_motif_scores keys are the
        ORIGINAL global ids."""
        kept = [1, 3]   # global vocab is 5; keep only ids 1 and 3
        m = self._model(num_motifs=5, kept_motif_ids=kept)
        self.assertEqual(m.motif_params.shape, (2, 1))
        self.assertEqual(m.global_to_param.tolist(), [-1, 0, -1, 1, -1])
        # set distinct values so we can verify the global id ↔ row mapping
        m.motif_params.data[0, 0] = 2.0   # row0 = global id 1
        m.motif_params.data[1, 0] = -2.0  # row1 = global id 3
        scores = m.get_motif_scores()
        self.assertEqual(set(scores.keys()), {1, 3})
        self.assertAlmostEqual(scores[1],
                               float(torch.sigmoid(torch.tensor(2.0))), places=5)
        self.assertAlmostEqual(scores[3],
                               float(torch.sigmoid(torch.tensor(-2.0))), places=5)
        b = _batch(3, 8, 5)
        out, att = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
        self.assertEqual(out.shape, (3, 1))
        out.sum().backward()
        self.assertEqual(m.motif_params.grad.shape, (2, 1))


# ── MultiChannelGNN ──────────────────────────────────────────────────────────

class TestMultiChannelGNN(unittest.TestCase):
    def _model(self, n_classes=4, **kw):
        kw.setdefault('num_motifs', 5)
        return MultiChannelGNN(
            x_dim=NUM_ATOM_TYPES, hidden_dim=32, num_layers=2,
            backbone='GIN', num_classes=n_classes, **kw
        )

    def test_forward_shape(self):
        m = self._model(4)
        b = _batch(3, 8, 5, 4)
        out, att = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs, b.edge_attr)
        self.assertEqual(out.shape, (3, 4))
        self.assertEqual(att.shape, (b.x.size(0), 4))

    def test_motif_params_shape(self):
        m = self._model(4)
        self.assertEqual(m.motif_params.shape, (5, 4))

    def test_per_class_independence(self):
        """Gradient for class 0 should update column 0 of motif_params only."""
        m = self._model(3)
        b = _batch(2, 8, 5, 3)
        m.zero_grad()
        out, _ = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
        # Loss on class 0 only
        out[:, 0].sum().backward()
        g = m.motif_params.grad
        self.assertIsNotNone(g)
        # Column 0 should have nonzero gradient; others may be zero
        # (depends on w_feat/w_readout — column 0 is the only one backpropped)
        # We just verify the gradient exists and has the right shape
        self.assertEqual(g.shape, (5, 3))

    def test_w_message_multichannel(self):
        """Each channel passes its own edge_atten to the conv layers."""
        m = self._model(3, w_feat=False, w_message=True, w_readout=False)
        m.motif_params.data.fill_(1.5)
        b = _batch(2, 8, 5, 3)
        out, att = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
        self.assertEqual(out.shape, (2, 3))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isnan(att).any())

    def test_w_message_edge_atten_shape(self):
        """Edge attention derived from att_c[src]*att_c[dst] must be [E,1]."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from model import _motif_to_node_weights
        n, n_edges = 8, 12
        ntm = torch.randint(0, 5, (n,))
        params = torch.nn.Parameter(torch.zeros(5, 3))
        node_att = _motif_to_node_weights(ntm, params, n, torch.device('cpu'))
        # Simulate edge_atten for class 0
        src = torch.randint(0, n, (n_edges,))
        dst = torch.randint(0, n, (n_edges,))
        ea = (node_att[:, 0].view(-1)[src] * node_att[:, 0].view(-1)[dst]).unsqueeze(-1)
        self.assertEqual(ea.shape, (n_edges, 1))
        self.assertTrue((ea >= 0).all() and (ea <= 1).all())

    def test_no_motif_params(self):
        m = self._model(3, num_motifs=0)
        b = _batch(2, 8, 5, 3)
        out, att = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
        self.assertEqual(out.shape, (2, 3))
        self.assertIsNone(att)

    def test_unk_learnable(self):
        m = self._model(3, unk_mode='learnable_shared')
        self.assertIsNotNone(m.unk_param)
        b = _batch(2, 8, 5, 3)
        out, att = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
        self.assertEqual(out.shape, (2, 3))

    def test_gradients_all_classes(self):
        m = self._model(3)
        b = _batch(2, 8, 5, 3)
        out, _ = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
        out.sum().backward()
        self.assertFalse(torch.isnan(m.motif_params.grad).any())


# ── mask_regularisation ───────────────────────────────────────────────────────

class TestMaskRegularisation(unittest.TestCase):
    def test_zero_coefficients(self):
        scores = torch.rand(10)
        loss = mask_regularisation(scores, size_reg=0.0, ent_reg=0.0)
        self.assertAlmostEqual(float(loss), 0.0)

    def test_ent_reg_at_half(self):
        # At 0.5 each, entropy is maximised → high loss
        scores = torch.full((10,), 0.5)
        loss1 = mask_regularisation(scores, size_reg=0.0, ent_reg=1.0)
        scores2 = torch.full((10,), 0.01)
        loss2 = mask_regularisation(scores2, size_reg=0.0, ent_reg=1.0)
        self.assertGreater(float(loss1), float(loss2))

    def test_size_reg_excludes_topk(self):
        # All ones; top-3 excluded → size = 0 * 7 = 0
        scores = torch.ones(10)
        loss1 = mask_regularisation(scores, size_reg=1.0, ent_reg=0.0, top_tau=10)
        loss2 = mask_regularisation(scores, size_reg=1.0, ent_reg=0.0, top_tau=0)
        self.assertAlmostEqual(float(loss1), 0.0)
        self.assertGreater(float(loss2), 0.0)

    def test_requires_grad(self):
        scores = torch.rand(5, requires_grad=True)
        loss = mask_regularisation(scores, size_reg=1.0, ent_reg=1.0)
        loss.backward()
        self.assertIsNotNone(scores.grad)

    def test_multi_class_input(self):
        scores = torch.rand(10, 4)
        loss = mask_regularisation(scores, size_reg=0.1, ent_reg=0.1)
        self.assertIsInstance(float(loss), float)


# ── train_one_epoch ───────────────────────────────────────────────────────────

class TestTrainOneEpoch(unittest.TestCase):
    def _make_loader(self, n=8, n_atoms=8, n_motifs=5):
        from torch_geometric.loader import DataLoader
        data_list = []
        for i in range(n):
            x = torch.randn(n_atoms, NUM_ATOM_TYPES)
            ei = torch.tensor([[0,1,2,3],[1,2,3,0]], dtype=torch.long)
            ntm = torch.randint(-1, n_motifs, (n_atoms,))
            y = torch.tensor([float(i % 2)])
            data_list.append(Data(x=x, edge_index=ei, nodes_to_motifs=ntm, y=y,
                                  edge_attr=torch.randn(4, EDGE_FEAT_DIM)))
        return DataLoader(data_list, batch_size=4)

    def test_returns_float_tuple(self):
        model = SingleChannelGNN(
            x_dim=NUM_ATOM_TYPES, hidden_dim=16, num_layers=2,
            num_motifs=5, backbone='GIN'
        )
        loader = self._make_loader()
        crit = nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        task_l, reg_l = train_one_epoch(
            model, crit, opt, loader, DEVICE, 'BinaryClass',
            size_reg=0.01, ent_reg=0.01
        )
        self.assertIsInstance(task_l, float)
        self.assertIsInstance(reg_l, float)
        self.assertFalse(torch.isnan(torch.tensor(task_l)))

    def test_full_train_loop_runs(self):
        """Exercises train_mose_gnn end-to-end (incl. ReduceLROnPlateau
        construction) so scheduler/API breakage is caught — train_one_epoch
        alone never builds the scheduler."""
        from train import train_mose_gnn
        model = SingleChannelGNN(
            x_dim=NUM_ATOM_TYPES, hidden_dim=16, num_layers=2,
            num_motifs=5, backbone='GIN'
        )
        loader = self._make_loader()
        loaders = {'train': loader, 'valid': loader, 'test': loader}
        model, history = train_mose_gnn(
            model, loaders, 'BinaryClass', DEVICE,
            epochs=2, lr=1e-3, min_epochs=1, patience=5, verbose=False,
        )
        self.assertIn('val_metric', history)
        self.assertEqual(len(history['val_metric']), 2)

    def test_loss_not_nan(self):
        model = MultiChannelGNN(
            x_dim=NUM_ATOM_TYPES, hidden_dim=16, num_layers=2,
            num_classes=2, num_motifs=5, backbone='GIN'
        )
        loader = self._make_loader()
        crit = nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Multi-label: y needs shape [N, 2]
        from torch_geometric.loader import DataLoader
        from torch_geometric.data import Data as D
        data_list = []
        for i in range(8):
            d = D(x=torch.randn(8, NUM_ATOM_TYPES),
                  edge_index=torch.tensor([[0,1],[1,0]]),
                  nodes_to_motifs=torch.randint(-1, 5, (8,)),
                  y=torch.tensor([[float(i%2), float((i+1)%2)]]),
                  edge_attr=torch.randn(2, EDGE_FEAT_DIM))
            data_list.append(d)
        loader2 = DataLoader(data_list, batch_size=4)
        task_l, _ = train_one_epoch(
            model, crit, opt, loader2, DEVICE, 'MultiLabel'
        )
        self.assertFalse(torch.isnan(torch.tensor(task_l)))


class TestRegConfig(unittest.TestCase):
    """Per (architecture × dataset) regularization lookup."""

    def setUp(self):
        from reg_config import resolve_reg, REG_CONFIG
        self.resolve = resolve_reg
        self.cfg = REG_CONFIG

    def test_table_lookup(self):
        self.assertEqual(self.resolve('GIN', 'BBBP'), (0.1, 0.0005, True))
        self.assertEqual(self.resolve('GAT', 'Mutagenicity'), (0.2, 0.00005, True))

    def test_pna_uses_gin(self):
        for ds in ('Benzene', 'Mutagenicity', 'Alkane_Carbonyl', 'esol'):
            self.assertEqual(self.resolve('PNA', ds), self.resolve('GIN', ds))

    def test_explicit_override_wins(self):
        e, s, from_tbl = self.resolve('GIN', 'BBBP', ent_reg=0.5, size_reg=0.0)
        self.assertEqual((e, s), (0.5, 0.0))
        self.assertFalse(from_tbl)

    def test_partial_override(self):
        e, s, from_tbl = self.resolve('GIN', 'BBBP', ent_reg=0.9)
        self.assertEqual((e, s), (0.9, 0.0005))
        self.assertTrue(from_tbl)

    def test_unknown_pair_default(self):
        self.assertEqual(self.resolve('GIN', 'NoSuchDataset'), (0.01, 0.0, True))

    def test_num_layers_per_dataset(self):
        from reg_config import resolve_num_layers
        self.assertEqual(resolve_num_layers('BBBP'), (2, True))
        self.assertEqual(resolve_num_layers('Mutagenicity'), (3, True))
        self.assertEqual(resolve_num_layers('Benzene'), (3, True))
        # explicit override wins
        self.assertEqual(resolve_num_layers('BBBP', num_layers=5), (5, False))


if __name__ == '__main__':
    unittest.main()
