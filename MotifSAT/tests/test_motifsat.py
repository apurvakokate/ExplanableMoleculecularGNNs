#!/usr/bin/env python3
"""test_motifsat.py — tests for MotifSAT models, losses, and motif modules.

Tests:
  - losses: info_loss, motif_consistency_loss, motif_size_weights
  - motif_modules: compute_inverse_idx, lift_motif_to_node, MotifPooling,
                   ExtractorMLP, MotifReadoutScorer
  - model.GSAT: all motif_method × noise × info_loss_level combinations
    forward shape, loss keys, gradient flow
  - _concrete_sample / _add_logistic_noise

Run:
    python test_motifsat.py -v
"""

import sys, os, unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'MotifSAT'))

from SharedModules.data.dataset import NUM_ATOM_TYPES, EDGE_FEAT_DIM

from losses import info_loss, motif_consistency_loss, motif_size_weights
from motif_modules import (
    compute_inverse_idx, lift_motif_to_node,
    MotifPooling, ExtractorMLP, MotifReadoutScorer,
)
from model import GSAT, _concrete_sample, _add_logistic_noise

DEVICE = torch.device('cpu')


# ── helpers ──────────────────────────────────────────────────────────────────

def _batch(n_graphs=4, n_atoms=6, n_motifs=3, n_classes=1):
    graphs = []
    for g in range(n_graphs):
        x = torch.randn(n_atoms, NUM_ATOM_TYPES)
        ei = torch.tensor([[0,1,2,3,4,5],[1,2,3,4,5,0]], dtype=torch.long)
        ntm = torch.randint(0, n_motifs, (n_atoms,))
        y = torch.tensor([float(g % 2)] if n_classes == 1
                         else [float(g % 2)] * n_classes)
        graphs.append(Data(x=x, edge_index=ei, nodes_to_motifs=ntm, y=y,
                           edge_attr=torch.randn(6, EDGE_FEAT_DIM)))
    return Batch.from_data_list(graphs)


def _make_gsat(**kwargs):
    defaults = dict(
        x_dim=NUM_ATOM_TYPES, hidden_dim=32, num_layers=2,
        backbone_name='GIN', num_classes=1,
    )
    defaults.update(kwargs)
    return GSAT(**defaults)


# ── losses ────────────────────────────────────────────────────────────────────

class TestInfoLoss(unittest.TestCase):
    def test_output_is_scalar(self):
        att = torch.rand(20)
        loss = info_loss(att, r=0.5)
        self.assertEqual(loss.shape, ())

    def test_loss_at_r_is_lower(self):
        # att = r should give lower KL than att far from r
        att_good = torch.full((10,), 0.5)
        att_bad = torch.full((10,), 0.01)
        self.assertLess(float(info_loss(att_good, r=0.5)),
                        float(info_loss(att_bad, r=0.5)))

    def test_size_weights_applied(self):
        att = torch.rand(10)
        l1 = info_loss(att, r=0.5, size_weights=torch.ones(10))
        l2 = info_loss(att, r=0.5, size_weights=torch.full((10,), 0.1))
        # Smaller weights → smaller loss
        self.assertGreater(float(l1), float(l2))

    def test_requires_grad(self):
        att = torch.rand(10, requires_grad=True)
        loss = info_loss(att, r=0.5)
        loss.backward()
        self.assertIsNotNone(att.grad)

    def test_r_clamped(self):
        att = torch.rand(5)
        # Should not raise even for extreme r values
        _ = info_loss(att, r=0.0)
        _ = info_loss(att, r=1.0)


class TestMotifSizeWeights(unittest.TestCase):
    def test_shape(self):
        ntm = torch.tensor([0, 1, 2, -1])
        lengths = [6, 1, 3, 1]
        w = motif_size_weights(ntm, lengths)
        self.assertEqual(w.shape, (4,))

    def test_known_values(self):
        ntm = torch.tensor([0, 1])
        lengths = [4, 2]
        w = motif_size_weights(ntm, lengths)
        self.assertAlmostEqual(float(w[0]), 1.0 / 4)
        self.assertAlmostEqual(float(w[1]), 1.0 / 2)

    def test_unknown_gets_one(self):
        ntm = torch.tensor([-1])
        w = motif_size_weights(ntm, [])
        self.assertAlmostEqual(float(w[0]), 1.0)


class TestMotifConsistencyLoss(unittest.TestCase):
    def test_output_is_two_scalars(self):
        att = torch.rand(12)
        ntm = torch.tensor([0,0,1,1,0,0,1,1,2,2,2,2])
        batch = torch.tensor([0,0,0,0,1,1,1,1,2,2,2,2])
        within, between = motif_consistency_loss(att, ntm, batch)
        self.assertEqual(within.shape, ())
        self.assertEqual(between.shape, ())

    def test_zero_within_for_single_node_motifs(self):
        # Each motif has only 1 node → within variance = 0
        att = torch.rand(4)
        ntm = torch.tensor([0, 1, 2, 3])
        batch = torch.zeros(4, dtype=torch.long)
        within, _ = motif_consistency_loss(att, ntm, batch)
        self.assertAlmostEqual(float(within), 0.0, places=4)

    def test_requires_grad(self):
        att = torch.rand(8, requires_grad=True)
        ntm = torch.tensor([0,0,1,1,0,0,1,1])
        batch = torch.zeros(8, dtype=torch.long)
        within, between = motif_consistency_loss(att, ntm, batch)
        (within + between).backward()
        self.assertIsNotNone(att.grad)

    def test_uniform_within_zero(self):
        # Constant att → within-motif variance = 0
        att = torch.ones(6)
        ntm = torch.tensor([0,0,1,1,0,0])
        batch = torch.zeros(6, dtype=torch.long)
        within, _ = motif_consistency_loss(att, ntm, batch)
        self.assertAlmostEqual(float(within), 0.0, places=5)


# ── motif_modules ─────────────────────────────────────────────────────────────

class TestComputeInverseIdx(unittest.TestCase):
    def test_basic(self):
        ntm = torch.tensor([0, 0, 1, 1, 2])
        batch = torch.tensor([0, 0, 0, 0, 0])
        inv, mb, mv = compute_inverse_idx(ntm, batch)
        self.assertEqual(inv.shape, (5,))
        # Nodes 0,1 same motif → same inv index
        self.assertEqual(int(inv[0]), int(inv[1]))
        self.assertEqual(int(inv[2]), int(inv[3]))

    def test_multi_graph(self):
        ntm = torch.tensor([0, 1, 0, 1])
        batch = torch.tensor([0, 0, 1, 1])
        inv, mb, mv = compute_inverse_idx(ntm, batch)
        # (graph0, motif0), (graph0, motif1), (graph1, motif0), (graph1, motif1)
        # → 4 unique motif instances
        self.assertEqual(len(mb), 4)

    def test_inverse_within_range(self):
        ntm = torch.randint(0, 5, (20,))
        batch = torch.randint(0, 3, (20,))
        inv, mb, _ = compute_inverse_idx(ntm, batch)
        self.assertTrue((inv >= 0).all())
        self.assertTrue((inv < len(mb)).all())

    def test_unknown_nodes_isolated(self):
        """Bug 2 regression: unknown nodes (-1) must NOT share a motif row
        with any real motif, and must not collide across graphs."""
        # graph0: motifs 0,0,2,-1 ; graph1: motifs 1,2,-1
        ntm   = torch.tensor([0, 0, 2, -1, 1, 2, -1])
        batch = torch.tensor([0, 0, 0,  0, 1, 1,  1])
        inv, mb, mv = compute_inverse_idx(ntm, batch)

        known = ntm >= 0
        unk_rows   = set(inv[~known].tolist())
        known_rows = set(inv[known].tolist())
        # No unknown node shares a row with a known node
        self.assertEqual(unk_rows & known_rows, set(),
                         msg='unknown node contaminated a real motif row')
        # The two unknown nodes are in different graphs → different rows
        self.assertNotEqual(int(inv[3]), int(inv[6]),
                            msg='unknown nodes collided across graphs')
        # Rows belonging to unknown nodes carry vocab id -1
        for r in unk_rows:
            self.assertEqual(int(mv[r]), -1)
        # Rows belonging to known nodes carry a real (>=0) vocab id
        for r in known_rows:
            self.assertGreaterEqual(int(mv[r]), 0)

    def test_known_vocab_ids_preserved(self):
        """Real motif ids survive the +1/-1 shift round-trip."""
        ntm   = torch.tensor([0, 3, 2, 3])
        batch = torch.tensor([0, 0, 0, 0])
        inv, mb, mv = compute_inverse_idx(ntm, batch)
        # Each node's recovered vocab id must equal its original motif id
        for i in range(ntm.size(0)):
            self.assertEqual(int(mv[inv[i]]), int(ntm[i]))

    def test_all_unknown(self):
        """A batch where every node is unknown must not crash and must
        produce only vocab id -1 rows."""
        ntm   = torch.tensor([-1, -1, -1])
        batch = torch.tensor([0, 0, 1])
        inv, mb, mv = compute_inverse_idx(ntm, batch)
        self.assertEqual(inv.shape, (3,))
        self.assertTrue((mv == -1).all())


class TestLiftMotifToNode(unittest.TestCase):
    def test_shape(self):
        motif_vals = torch.tensor([1.0, 2.0, 3.0])
        inv = torch.tensor([0, 0, 1, 2, 2])
        out = lift_motif_to_node(motif_vals, inv)
        self.assertEqual(out.shape, (5,))

    def test_values(self):
        motif_vals = torch.tensor([10.0, 20.0])
        inv = torch.tensor([0, 0, 1, 0])
        out = lift_motif_to_node(motif_vals, inv)
        expected = torch.tensor([10., 10., 20., 10.])
        self.assertTrue(torch.allclose(out, expected))

    def test_2d(self):
        motif_vals = torch.randn(3, 8)
        inv = torch.tensor([0, 1, 2, 0, 1])
        out = lift_motif_to_node(motif_vals, inv)
        self.assertEqual(out.shape, (5, 8))


class TestMotifPooling(unittest.TestCase):
    def _run(self, mode):
        pool = MotifPooling(mode)
        emb = torch.randn(12, 16)
        inv = torch.tensor([0,0,0,1,1,1,2,2,2,3,3,3])
        out = pool(emb, inv, num_motifs=4)
        expected_d = {'mean': 16, 'max': 16, 'max_mean': 32, 'multi': 48}[mode]
        self.assertEqual(out.shape, (4, expected_d))

    def test_mean(self):   self._run('mean')
    def test_max(self):    self._run('max')
    def test_max_mean(self): self._run('max_mean')
    def test_multi(self):  self._run('multi')

    def test_out_mult(self):
        self.assertEqual(MotifPooling('mean').out_mult, 1)
        self.assertEqual(MotifPooling('multi').out_mult, 3)


class TestExtractorMLP(unittest.TestCase):
    def test_output_shape(self):
        mlp = ExtractorMLP(in_dim=32, hidden_mult=2)
        x = torch.randn(15, 32)
        out = mlp(x)
        self.assertEqual(out.shape, (15, 1))

    def test_gradient_flows(self):
        mlp = ExtractorMLP(in_dim=16)
        x = torch.randn(5, 16, requires_grad=True)
        out = mlp(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)


class TestMotifReadoutScorer(unittest.TestCase):
    def test_output_shapes(self):
        scorer = MotifReadoutScorer(in_dim=32, pool_mode='mean')
        emb = torch.randn(12, 32)
        inv = torch.tensor([0,0,0,1,1,1,2,2,2,3,3,3])
        m_logits, n_logits = scorer(emb, inv, num_motifs=4)
        self.assertEqual(m_logits.shape, (4, 1))
        self.assertEqual(n_logits.shape, (12, 1))

    def test_node_logits_broadcast_correctly(self):
        scorer = MotifReadoutScorer(in_dim=16, pool_mode='mean')
        emb = torch.randn(6, 16)
        inv = torch.tensor([0, 0, 1, 1, 2, 2])
        _, n_logits = scorer(emb, inv, num_motifs=3)
        # Nodes 0,1 same motif → same logit
        self.assertAlmostEqual(float(n_logits[0]), float(n_logits[1]), places=5)


# ── sampling helpers ──────────────────────────────────────────────────────────

class TestConcreteSample(unittest.TestCase):
    def test_train_output_in_01(self):
        logits = torch.randn(20, 1)
        att = _concrete_sample(logits, r=0.5, training=True)
        self.assertTrue((att >= 0).all() and (att <= 1).all())

    def test_eval_soft_sigmoid(self):
        # Eval returns the soft sigmoid of the logits, NOT a hard 0/1 threshold.
        logits = torch.tensor([[5.0], [-5.0], [0.0]])
        att = _concrete_sample(logits, r=0.5, training=False)
        # Clamped at |3|, so sigmoid(3)=0.9526, sigmoid(-3)=0.0474, sigmoid(0)=0.5
        self.assertAlmostEqual(float(att[0]), 0.9526, places=3)
        self.assertAlmostEqual(float(att[1]), 0.0474, places=3)
        self.assertAlmostEqual(float(att[2]), 0.5, places=4)
        # Crucially: values are continuous, not collapsed to {0,1}
        self.assertNotIn(float(att[0]), (0.0, 1.0))

    def test_shape_preserved(self):
        logits = torch.randn(10, 1)
        att = _concrete_sample(logits, r=0.3, training=True)
        self.assertEqual(att.shape, (10, 1))


class TestAddLogisticNoise(unittest.TestCase):
    def test_output_shapes(self):
        logits = torch.randn(8, 1)
        noisy, noise = _add_logistic_noise(logits)
        self.assertEqual(noisy.shape, logits.shape)
        self.assertEqual(noise.shape, logits.shape)

    def test_noise_is_not_zero(self):
        logits = torch.zeros(100, 1)
        _, noise = _add_logistic_noise(logits)
        # Extremely unlikely all noise samples are zero
        self.assertGreater(float(noise.abs().sum()), 0.01)


# ── GSAT model ────────────────────────────────────────────────────────────────

class TestGSAT(unittest.TestCase):
    def _fwd(self, **kwargs):
        m = _make_gsat(**kwargs)
        m.eval()
        b = _batch(4, 6, 3)
        return m(b.x, b.edge_index, b.batch, b.nodes_to_motifs, b.edge_attr)

    def test_none_method_shape(self):
        logits, att, aux = self._fwd(motif_method='none', w_message=True)
        self.assertEqual(logits.shape, (4, 1))
        self.assertIsNotNone(att)

    def test_loss_method_shape(self):
        logits, att, aux = self._fwd(motif_method='loss', w_message=True)
        self.assertEqual(logits.shape, (4, 1))

    def test_node_emb_removed(self):
        # node_emb was removed entirely; it is no longer a valid motif_method.
        with self.assertRaises(ValueError):
            _make_gsat(motif_method='node_emb')

    def test_motif_emb_not_implemented(self):
        # motif_emb is a reserved-but-unimplemented method.
        with self.assertRaises(NotImplementedError):
            _make_gsat(motif_method='motif_emb')

    def test_readout_method_shape(self):
        logits, att, aux = self._fwd(motif_method='readout')
        self.assertEqual(logits.shape, (4, 1))
        self.assertIsNotNone(aux['motif_logits'])

    def test_node_noise(self):
        m = _make_gsat(motif_method='none', noise='node')
        m.train()
        b = _batch(2, 6, 3)
        logits, att, _ = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
        self.assertEqual(logits.shape, (2, 1))

    def test_motif_noise(self):
        m = _make_gsat(motif_method='readout', noise='motif')
        m.train()
        b = _batch(2, 6, 3)
        logits, att, _ = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
        self.assertEqual(logits.shape, (2, 1))

    def test_edge_att_path(self):
        m = _make_gsat(learn_edge_att=True, motif_method='none')
        b = _batch(2, 6, 3)
        logits, att, aux = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
        self.assertEqual(logits.shape, (2, 1))
        self.assertIsNotNone(aux['edge_att'])
        self.assertIsNone(att)  # node_att is None for edge path

    def test_eval_node_att_is_soft(self):
        """At eval, node_att is a soft gate in (0,1) — never a hard 0/1 mask."""
        _, att, _ = self._fwd(motif_method='none', w_message=True)
        v = att.view(-1)
        self.assertTrue((v > 0).all() and (v < 1).all(),
                        msg=f'eval node_att should be soft in (0,1), got {v.tolist()}')
        uniq = set(v.unique().tolist())
        self.assertFalse(uniq.issubset({0.0, 1.0}),
                         msg='eval node_att collapsed to a hard mask')

    def test_train_soft_att_differs_from_sampled(self):
        """At train time the sampled att carries logistic noise, so the clean
        node_att_soft differs from the sampled node_att. (At eval they coincide
        — both are the soft sigmoid.)"""
        m = _make_gsat(motif_method='none', w_message=True, noise='node')
        m.train()
        b = _batch(4, 6, 3)
        _, att, aux = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs, b.edge_attr)
        self.assertIn('node_att_soft', aux)
        soft = aux['node_att_soft'].view(-1)
        self.assertTrue((soft > 0).all() and (soft < 1).all())
        # Sampled (noisy) att should not exactly equal the clean sigmoid
        self.assertFalse(torch.equal(soft, att.view(-1)))

    def test_eval_soft_att_matches_node_att(self):
        """At eval, node_att and node_att_soft are both the soft sigmoid."""
        _, att, aux = self._fwd(motif_method='none', w_message=True)
        self.assertTrue(torch.allclose(aux['node_att_soft'].view(-1),
                                       att.view(-1), atol=1e-6))

    def test_eval_edge_soft_att_present(self):
        """Edge path exposes edge_att_soft as continuous per-edge scores."""
        m = _make_gsat(learn_edge_att=True, motif_method='none')
        m.eval()
        b = _batch(2, 6, 3)
        _, _, aux = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs, b.edge_attr)
        self.assertIsNotNone(aux['edge_att_soft'])
        soft = aux['edge_att_soft'].view(-1)
        self.assertTrue((soft > 0).all() and (soft < 1).all())

    def test_w_feat_flag(self):
        m = _make_gsat(w_feat=True, w_message=False, w_readout=False)
        b = _batch(2, 6, 3)
        logits, _, _ = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
        self.assertEqual(logits.shape, (2, 1))

    def test_w_readout_flag(self):
        m = _make_gsat(w_feat=False, w_message=False, w_readout=True)
        b = _batch(2, 6, 3)
        logits, _, _ = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
        self.assertEqual(logits.shape, (2, 1))

    def test_compute_loss_keys(self):
        m = _make_gsat(motif_method='loss', info_loss_coef=1.0,
                       motif_loss_coef=1.0, between_motif_coef=0.5,
                       within_node_coef=0.5)
        m.train()
        b = _batch(2, 6, 3)
        _, _, aux = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
        task = torch.tensor(0.5, requires_grad=True)
        total, breakdown = m.compute_loss(task, aux, b.nodes_to_motifs, b.batch)
        self.assertIn('total', breakdown)
        self.assertIn('task', breakdown)
        self.assertIn('info_loss', breakdown)

    def test_gradients_flow_all_methods(self):
        for method in ('none', 'loss', 'readout'):
            m = _make_gsat(motif_method=method, info_loss_coef=1.0)
            m.train()
            b = _batch(2, 6, 3)
            logits, att, aux = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
            task = logits.mean()
            total, _ = m.compute_loss(task, aux, b.nodes_to_motifs, b.batch)
            total.backward()
            for name, p in m.named_parameters():
                if p.requires_grad and p.grad is None:
                    # Some params may not be used in a given method
                    pass
                elif p.requires_grad and p.grad is not None:
                    self.assertFalse(
                        torch.isnan(p.grad).any(),
                        f'NaN gradient in {name} for method={method}'
                    )

    def test_anneal_r(self):
        m = _make_gsat(init_r=0.9, final_r=0.1,
                       decay_interval=10, decay_r=0.1)
        m.anneal_r(0)
        self.assertAlmostEqual(float(m.r), 0.9, places=4)
        m.anneal_r(20)
        # After 20 epochs, decayed by 2 steps of 0.1 → 0.9 - 0.2 = 0.7
        self.assertAlmostEqual(float(m.r), 0.7, places=4)
        m.anneal_r(100)
        self.assertGreaterEqual(float(m.r), 0.1)

    def test_no_nan_in_output(self):
        for method in ('none', 'readout'):
            m = _make_gsat(motif_method=method)
            b = _batch(4, 6, 3)
            logits, att, _ = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs)
            self.assertFalse(torch.isnan(logits).any(),
                             f'NaN in logits for method={method}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
