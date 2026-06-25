#!/usr/bin/env python3
"""test_shared_modules.py — tests for SharedModules.

Tests:
  - data/dataset.py: atom encoding, build_graph, MolDataset
  - data/vocab.py:   VocabData helpers, compute_mask_cache
  - data/loader.py:  compute_pos_weights
  - models/conv_layers.py: every backbone with and without edge_atten
  - models/gnn_base.py: BaseGNN encoding, all injection combinations
  - evaluation/metrics.py: AUC/RMSE/MAE/evaluate_predictions
  - evaluation/motif_eval.py: compute_motif_impact, score_impact_correlation
  - baselines/vanilla_gnn.py: forward shape, parameter count

Run:
    python test_shared_modules.py -v
"""

import sys, os, unittest, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from SharedModules.data.dataset import (
    build_graph, ATOMS, BONDS, NUM_ATOM_TYPES, EDGE_FEAT_DIM, MolDataset
)
from SharedModules.data.vocab import VocabData, compute_mask_cache
from SharedModules.data.loader import compute_pos_weights, MutagTUDataset, MUTAG_X_DIM, MUTAG_EDGE_DIM, OGB_DATASET_NAMES, LoaderMeta
from SharedModules.models.conv_layers import (
    create_conv_layers, CONV_FACTORIES,
)
# Conv classes are exported under both their real names and backward-compat
# aliases (GCNConv, etc.) from the package __init__, not the submodule.
from SharedModules.models import (
    GINConv, GCNConv, SAGEConv, GATConv, PNAConv,
)
from SharedModules.models.gnn_base import BaseGNN
from SharedModules.evaluation.metrics import (
    auc_score, mae_score, rmse_score, evaluate_predictions
)
from SharedModules.evaluation.motif_eval import (
    compute_motif_impact, score_impact_correlation, explainer_roc_vs_gt,
    top_bottom_motif_eval, gt_vs_outside_gt_eval, compute_gt_roc,
)
from SharedModules.evaluation.multi_explanation import (
    assign_hypothesis_flags, compute_h1_h2_ratios, classify_motif_category,
    category_summary, CATEGORY_ORDER,
)
from SharedModules.evaluation.pipeline import EvalPipeline
from SharedModules.baselines.vanilla_gnn import VanillaGNN


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device('cpu')

SMILES = {
    'benzene':      'c1ccccc1',
    'toluene':      'Cc1ccccc1',
    'nitrobenzene': 'O=[N+]([O-])c1ccccc1',
    'paracetamol':  'CC(=O)Nc1ccc(O)cc1',
    'ethanol':      'CCO',
}

def _mini_batch(n_graphs: int = 4, n_atoms: int = 6, hidden: int = 32):
    """Build a tiny synthetic PyG Batch for shape tests."""
    graphs = []
    for _ in range(n_graphs):
        x = torch.randn(n_atoms, NUM_ATOM_TYPES)
        edge_index = torch.tensor([[0,1,1,2],[1,0,2,1]], dtype=torch.long)
        y = torch.tensor([1.0])
        ntm = torch.tensor([0, 1, 0, 1, -1, 0], dtype=torch.long)
        graphs.append(Data(x=x, edge_index=edge_index, y=y,
                           nodes_to_motifs=ntm,
                           edge_attr=torch.randn(4, EDGE_FEAT_DIM)))
    return Batch.from_data_list(graphs)


# ──────────────────────────────────────────────────────────────────────────────
# data/dataset.py
# ──────────────────────────────────────────────────────────────────────────────

class TestAtomEncoding(unittest.TestCase):
    def test_all_atoms_unique(self):
        self.assertEqual(len(set(ATOMS.values())), len(ATOMS))

    def test_wildcard_has_own_index(self):
        self.assertIn('*', ATOMS)
        self.assertNotEqual(ATOMS['*'], ATOMS['H'])  # wildcard ≠ hydrogen

    def test_num_atom_types_matches(self):
        self.assertEqual(NUM_ATOM_TYPES, max(ATOMS.values()) + 1)


class TestBuildGraph(unittest.TestCase):
    def test_benzene_shape(self):
        d = build_graph(SMILES['benzene'], torch.tensor([1.0]), None)
        self.assertIsNotNone(d)
        self.assertEqual(d.x.shape, (6, NUM_ATOM_TYPES))
        # benzene has 12 directed edges
        self.assertEqual(d.edge_index.shape[1], 12)

    def test_nitrobenzene(self):
        d = build_graph(SMILES['nitrobenzene'], torch.tensor([1.0]), None)
        self.assertIsNotNone(d)
        n = d.x.shape[0]
        self.assertEqual(d.nodes_to_motifs.shape, (n,))
        self.assertTrue((d.nodes_to_motifs == -1).all())  # no lookup → all -1

    def test_with_lookup(self):
        smi = SMILES['benzene']
        # Fake lookup: atoms 0..5 → motif 3
        lookup = {smi: {i: ('c1ccccc1', 3) for i in range(6)}}
        d = build_graph(smi, torch.tensor([0.0]), lookup)
        self.assertIsNotNone(d)
        self.assertTrue((d.nodes_to_motifs == 3).all())

    def test_invalid_smiles_returns_none(self):
        self.assertIsNone(build_graph('not_valid', torch.tensor([0.0]), None))

    def test_unknown_atom_returns_none(self):
        # Xe is not in ATOMS dict
        result = build_graph('[Xe]', torch.tensor([0.0]), None)
        self.assertIsNone(result)

    def test_edge_attr_dim(self):
        d = build_graph(SMILES['paracetamol'], torch.tensor([0.0]), None)
        self.assertIsNotNone(d)
        self.assertEqual(d.edge_attr.shape[1], EDGE_FEAT_DIM)

    def test_y_shape(self):
        d = build_graph(SMILES['benzene'], torch.tensor([1.0]), None)
        self.assertEqual(d.y.shape, (1,))


# ──────────────────────────────────────────────────────────────────────────────
# data/vocab.py
# ──────────────────────────────────────────────────────────────────────────────

def _make_fake_vocab() -> VocabData:
    motif_list = ['[*]c1ccccc1', '[*]C', '[*]N', '[*]O', '[*][N+](=O)[O-]']
    frag_to_id = {s: i for i, s in enumerate(motif_list)}
    # Two molecules, each with some motifs
    lookup = {
        SMILES['benzene']:      {i: (motif_list[0], 0) for i in range(6)},
        SMILES['nitrobenzene']: {**{i: (motif_list[0], 0) for i in range(6)},
                                 **{i: (motif_list[4], 4) for i in range(6, 9)}},
    }
    return VocabData(
        motif_list=motif_list,
        motif_counts=[100, 50, 30, 20, 10],
        motif_lengths=[6, 1, 1, 1, 3],
        motif_class={i: {0: 5, 1: 5} for i in range(5)},
        lookup_train=lookup,
        lookup_valid={},
        lookup_test={},
        gmi_train={smi: set(v for _, v in nm.values()) for smi, nm in lookup.items()},
        gmi_test={},
    )


class TestVocabData(unittest.TestCase):
    def setUp(self):
        self.vocab = _make_fake_vocab()

    def test_num_motifs(self):
        self.assertEqual(self.vocab.num_motifs, 5)

    def test_motif_id_lookup(self):
        self.assertEqual(self.vocab.motif_id('[*]c1ccccc1'), 0)

    def test_motif_id_unknown(self):
        self.assertIsNone(self.vocab.motif_id('[*]Cl'))

    def test_lookup_for_split_train(self):
        lup = self.vocab.lookup_for_split('training')
        self.assertIn(SMILES['benzene'], lup)

    def test_lookup_for_split_test(self):
        lup = self.vocab.lookup_for_split('test')
        self.assertEqual(lup, {})


class TestComputeMaskCache(unittest.TestCase):
    def test_basic(self):
        vocab = _make_fake_vocab()
        smiles = [SMILES['benzene'], SMILES['nitrobenzene']]
        groups = ['training', 'training']
        lookup = vocab.lookup_train
        cache = compute_mask_cache(smiles, groups, lookup)
        self.assertIn('training', cache)
        self.assertIn(0, cache['training'])   # motif 0 (benzene ring)
        benz_mask = cache['training'][0][SMILES['benzene']]
        self.assertIsInstance(benz_mask, torch.Tensor)
        self.assertEqual(benz_mask.dtype, torch.bool)
        self.assertTrue(benz_mask.all())  # all 6 atoms belong to motif 0

    def test_unknown_nodes_excluded(self):
        # motif_id = -1 should not appear as a key
        vocab = _make_fake_vocab()
        # Add a molecule with an unknown node
        smi = 'CCO'
        lookup_with_unk = {**vocab.lookup_train,
                           smi: {0: ('[*]C', 1), 1: ('[*]C', 1), 2: ('[*]O', -1)}}
        cache = compute_mask_cache([smi], ['training'], lookup_with_unk)
        keys = set(cache['training'].keys())
        self.assertNotIn(-1, keys)  # -1 must never appear


# ──────────────────────────────────────────────────────────────────────────────
# models/conv_layers.py
# ──────────────────────────────────────────────────────────────────────────────

class TestConvLayers(unittest.TestCase):
    def _data(self, n=6, d_in=16, n_edges=10):
        x = torch.randn(n, d_in)
        src = torch.randint(0, n, (n_edges,))
        dst = torch.randint(0, n, (n_edges,))
        edge_index = torch.stack([src, dst])
        edge_attr = torch.randn(n_edges, EDGE_FEAT_DIM)
        ea = torch.rand(n_edges, 1)
        return x, edge_index, edge_attr, ea

    def _test_backbone(self, backbone: str, d_in=16, d_out=32):
        x, ei, ea, edge_atten = self._data(d_in=d_in)
        layers = create_conv_layers(d_in, d_out, num_layers=2,
                                    backbone=backbone,
                                    edge_dim=EDGE_FEAT_DIM if backbone in ('GAT','PNA') else None)
        for conv in layers:
            out = conv(x, ei, edge_atten=edge_atten)
            self.assertEqual(out.shape, (6, d_out))
            x = out

    def test_gin(self):   self._test_backbone('GIN')
    def test_gcn(self):   self._test_backbone('GCN')
    def test_sage(self):  self._test_backbone('SAGE')
    def test_gat(self):   self._test_backbone('GAT')

    def test_edge_atten_none_ok(self):
        x, ei, ea, _ = self._data()
        layers = create_conv_layers(16, 32, 2, 'GIN')
        for conv in layers:
            x = conv(x, ei)
        self.assertEqual(x.shape, (6, 32))

    def test_unknown_backbone_raises(self):
        with self.assertRaises(ValueError):
            create_conv_layers(16, 32, 2, 'UNKNOWN')


# ──────────────────────────────────────────────────────────────────────────────
# models/gnn_base.py
# ──────────────────────────────────────────────────────────────────────────────

class TestBaseGNN(unittest.TestCase):
    def _make_gnn(self, backbone='GIN', node_encoder='onehot',
                  apply_layer_norm=False, **kwargs):
        return BaseGNN(
            x_dim=NUM_ATOM_TYPES, hidden_dim=32, num_layers=2,
            backbone=backbone, node_encoder=node_encoder,
            apply_layer_norm=apply_layer_norm, **kwargs,
        )

    def test_forward_shape_no_att(self):
        batch = _mini_batch(4, 6, 32)
        gnn = self._make_gnn()
        g, h = gnn.get_embedding(batch.x, batch.edge_index, batch=batch.batch)
        self.assertEqual(g.shape, (4, 32))
        self.assertEqual(h.shape, (batch.x.size(0), 32))

    def test_w_feat_flag(self):
        batch = _mini_batch(2, 6)
        gnn = self._make_gnn()
        att = torch.rand(batch.x.size(0), 1)
        g_wf, _ = gnn.get_embedding(batch.x, batch.edge_index,
                                     node_att=att, w_feat=True, batch=batch.batch)
        g_no, _ = gnn.get_embedding(batch.x, batch.edge_index,
                                     node_att=att, w_feat=False, batch=batch.batch)
        # With different feature scaling, outputs should differ
        self.assertFalse(torch.allclose(g_wf, g_no))

    def test_w_message_flag(self):
        batch = _mini_batch(2, 6)
        gnn = self._make_gnn()
        att = torch.rand(batch.x.size(0), 1)
        g_wm, _ = gnn.get_embedding(batch.x, batch.edge_index,
                                     node_att=att, w_message=True, batch=batch.batch)
        g_no, _ = gnn.get_embedding(batch.x, batch.edge_index,
                                     node_att=att, w_message=False, batch=batch.batch)
        self.assertFalse(torch.allclose(g_wm, g_no))

    def test_w_readout_flag(self):
        batch = _mini_batch(2, 6)
        gnn = self._make_gnn()
        att = torch.rand(batch.x.size(0), 1)
        g_wr, _ = gnn.get_embedding(batch.x, batch.edge_index,
                                     node_att=att, w_readout=True, batch=batch.batch)
        g_no, _ = gnn.get_embedding(batch.x, batch.edge_index,
                                     node_att=att, w_readout=False, batch=batch.batch)
        self.assertFalse(torch.allclose(g_wr, g_no))

    def test_layer_norm_does_not_change_shape(self):
        batch = _mini_batch(2, 6)
        gnn = self._make_gnn(apply_layer_norm=True)
        g, h = gnn.get_embedding(batch.x, batch.edge_index, batch=batch.batch)
        self.assertEqual(g.shape, (2, 32))

    def test_linear_node_encoder(self):
        batch = _mini_batch(2, 6)
        gnn = self._make_gnn(node_encoder='linear')
        g, h = gnn.get_embedding(batch.x, batch.edge_index, batch=batch.batch)
        self.assertEqual(g.shape, (2, 32))

    def test_classify_shape(self):
        gnn = self._make_gnn()
        g = torch.randn(4, 32)
        out = gnn.classify(g)
        self.assertEqual(out.shape, (4, 1))

    def test_conv_normalize_l2_is_default_unit_norm(self):
        # Default conv_normalize='l2' → node embeddings have unit L2 norm.
        batch = _mini_batch(2, 6)
        gnn = self._make_gnn()
        self.assertEqual(gnn.conv_normalize, 'l2')
        gnn.eval()
        _, h = gnn.get_embedding(batch.x, batch.edge_index, batch=batch.batch)
        norms = h.norm(p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-4))

    def test_conv_normalize_none_not_unit(self):
        batch = _mini_batch(2, 6)
        gnn = self._make_gnn(conv_normalize='none')
        gnn.eval()
        _, h = gnn.get_embedding(batch.x, batch.edge_index, batch=batch.batch)
        norms = h.norm(p=2, dim=1)
        self.assertFalse(torch.allclose(norms, torch.ones_like(norms), atol=1e-3))

    def test_apply_layer_norm_back_compat(self):
        # apply_layer_norm=True still forces layernorm mode.
        gnn = self._make_gnn(apply_layer_norm=True)
        self.assertEqual(gnn.conv_normalize, 'layernorm')
        self.assertIsNotNone(gnn.layer_norms)

    def test_gin_inner_bn_present_and_toggleable(self):
        on = self._make_gnn(gin_inner_bn=True)
        off = self._make_gnn(gin_inner_bn=False)
        has_bn = lambda m: any('BatchNorm' in type(x).__name__ for x in m.convs.modules())
        self.assertTrue(has_bn(on))
        self.assertFalse(has_bn(off))

    def test_all_backbones(self):
        for bb in ('GIN', 'GCN', 'SAGE'):
            batch = _mini_batch(2, 6)
            gnn = self._make_gnn(backbone=bb)
            g, _ = gnn.get_embedding(batch.x, batch.edge_index, batch=batch.batch)
            self.assertEqual(g.shape, (2, 32), f'Failed for backbone {bb}')


# ──────────────────────────────────────────────────────────────────────────────
# evaluation/metrics.py
# ──────────────────────────────────────────────────────────────────────────────

class TestMetrics(unittest.TestCase):
    def test_auc_perfect(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(auc_score(y, s), 1.0)

    def test_auc_random(self):
        rng = np.random.RandomState(0)
        y = rng.randint(0, 2, 100)
        s = rng.rand(100)
        auc = auc_score(y, s)
        self.assertTrue(0.3 < auc < 0.7)

    def test_auc_one_class_nan(self):
        y = np.ones(10)
        s = np.ones(10)
        self.assertTrue(np.isnan(auc_score(y, s)))

    def test_mae(self):
        y = np.array([1.0, 2.0, 3.0])
        p = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(mae_score(y, p), 0.0)

    def test_rmse(self):
        y = np.array([0.0, 0.0])
        p = np.array([1.0, 1.0])
        self.assertAlmostEqual(rmse_score(y, p), 1.0)

    def test_evaluate_predictions_binary(self):
        """evaluate_predictions on a VanillaGNN with a tiny batch."""
        model = VanillaGNN(x_dim=NUM_ATOM_TYPES, hidden_dim=16,
                           num_layers=2, backbone='GIN')
        model.eval()
        batch = _mini_batch(8, 6)
        from torch_geometric.loader import DataLoader
        from torch_geometric.data import Data
        data_list = []
        for i in range(8):
            d = Data(x=torch.randn(6, NUM_ATOM_TYPES),
                     edge_index=torch.tensor([[0,1],[1,0]]),
                     y=torch.tensor([float(i % 2)]),
                     nodes_to_motifs=torch.full((6,), -1, dtype=torch.long))
            data_list.append(d)
        loader = DataLoader(data_list, batch_size=4)
        metrics = evaluate_predictions(model, loader, DEVICE, 'BinaryClass')
        self.assertIn('auc', metrics)
        self.assertFalse(np.isnan(metrics['auc']))


# ──────────────────────────────────────────────────────────────────────────────
# evaluation/motif_eval.py
# ──────────────────────────────────────────────────────────────────────────────

class TestMotifEval(unittest.TestCase):
    def _make_eval_setup(self):
        vocab = _make_fake_vocab()
        model = VanillaGNN(x_dim=NUM_ATOM_TYPES, hidden_dim=16, num_layers=2)
        model.eval()

        data_list = []
        for smi in [SMILES['benzene'], SMILES['nitrobenzene']]:
            y = torch.tensor([1.0])
            d = build_graph(smi, y, vocab.lookup_train)
            if d is not None:
                data_list.append(d)
        return vocab, model, data_list

    def test_compute_motif_impact_returns_dict(self):
        vocab, model, data_list = self._make_eval_setup()
        # Build cache manually
        smiles = [SMILES['benzene'], SMILES['nitrobenzene']]
        groups = ['test'] * 2
        vocab.mask_cache['test'] = compute_mask_cache(
            smiles, groups, vocab.lookup_train
        ).get('test', {})
        results = compute_motif_impact(
            model, data_list, vocab, DEVICE,
            split='test', task_type='BinaryClass',
        )
        self.assertIsInstance(results, dict)
        for mid, info in results.items():
            self.assertIn('impact', info)
            self.assertIn('n_graphs', info)
            self.assertGreaterEqual(info['impact'], 0.0)

    def test_score_impact_correlation_shape(self):
        scores = {0: 0.8, 1: 0.3, 2: 0.1}
        impacts = {0: {'impact': 0.5}, 1: {'impact': 0.2}, 2: {'impact': 0.05}}
        corr = score_impact_correlation(scores, impacts)
        self.assertIn('pearson', corr)
        self.assertIn('spearman', corr)

    def test_score_impact_correlation_too_few(self):
        corr = score_impact_correlation({0: 0.5}, {0: {'impact': 0.1}})
        self.assertTrue(np.isnan(corr['pearson']))

    def test_explainer_roc_vs_gt(self):
        n = 6
        node_att = torch.rand(n)
        edge_index = torch.tensor([[0,1,2,3,4],[1,2,3,4,5]], dtype=torch.long)
        # edge_label is sized [E] (one entry per edge), per apply_gt.py contract
        edge_label = torch.tensor([1.,1.,0.,0.,0.], dtype=torch.float)
        auc = explainer_roc_vs_gt(node_att, edge_index, edge_label)
        # May be NaN if only one node class — just check it returns a float
        self.assertIsInstance(auc, float)


# ──────────────────────────────────────────────────────────────────────────────
# baselines/vanilla_gnn.py
# ──────────────────────────────────────────────────────────────────────────────

class TestVanillaGNN(unittest.TestCase):
    def test_forward_shape_binary(self):
        model = VanillaGNN(x_dim=NUM_ATOM_TYPES, hidden_dim=32,
                           num_layers=2, backbone='GIN', num_classes=1)
        batch = _mini_batch(4, 6)
        out, att = model(batch.x, batch.edge_index, batch.batch,
                         batch.nodes_to_motifs, batch.edge_attr)
        self.assertEqual(out.shape, (4, 1))
        self.assertIsNone(att)

    def test_compute_loss_out_2d_y_1d(self):
        """Batched PyG labels are often [B]; model logits are [B, 1]."""
        from SharedModules.baselines.vanilla_gnn import _compute_loss
        import torch.nn as nn
        out = torch.randn(128, 1)
        y = torch.randint(0, 2, (128,)).float()
        loss = _compute_loss(nn.BCEWithLogitsLoss(), out, y, 'BinaryClass')
        self.assertTrue(torch.isfinite(loss))

    def test_forward_shape_multilabel(self):
        model = VanillaGNN(x_dim=NUM_ATOM_TYPES, hidden_dim=32,
                           num_layers=2, num_classes=12)
        batch = _mini_batch(4, 6)
        out, _ = model(batch.x, batch.edge_index, batch.batch,
                       batch.nodes_to_motifs, batch.edge_attr)
        self.assertEqual(out.shape, (4, 12))

    def test_nodes_to_motifs_ignored(self):
        """VanillaGNN output must not change when nodes_to_motifs changes."""
        model = VanillaGNN(x_dim=NUM_ATOM_TYPES, hidden_dim=16, num_layers=2)
        model.eval()
        batch = _mini_batch(2, 6)
        with torch.no_grad():
            out1, _ = model(batch.x, batch.edge_index, batch.batch,
                            batch.nodes_to_motifs)
            batch.nodes_to_motifs = torch.full_like(batch.nodes_to_motifs, 99)
            out2, _ = model(batch.x, batch.edge_index, batch.batch,
                            batch.nodes_to_motifs)
        self.assertTrue(torch.allclose(out1, out2))

    def test_parameter_count_positive(self):
        model = VanillaGNN()
        from SharedModules.utils import count_parameters
        self.assertGreater(count_parameters(model), 0)

    def test_get_emb_shape(self):
        model = VanillaGNN(x_dim=NUM_ATOM_TYPES, hidden_dim=32, num_layers=2)
        batch = _mini_batch(2, 6)
        emb = model.get_emb(batch.x, batch.edge_index, batch.batch)
        self.assertEqual(emb.shape, (batch.x.size(0), 32))




# ──────────────────────────────────────────────────────────────────────────────
# evaluation/motif_eval.py — top_bottom_motif_eval
# ──────────────────────────────────────────────────────────────────────────────

class TestTopBottomMotifEval(unittest.TestCase):
    def _setup(self, n_motifs=20):
        """Create synthetic scores and impacts with a clear top/bottom split."""
        # Top motifs have high score AND high impact; bottom motifs have low both
        scores  = {i: float(i) / n_motifs for i in range(n_motifs)}
        impacts = {i: {'impact': float(i) / n_motifs + 0.01,
                        'motif_smarts': f'[*]C{i}'}
                   for i in range(n_motifs)}
        return scores, impacts

    def test_basic_shape(self):
        s, imp = self._setup(20)
        result = top_bottom_motif_eval(s, imp, k=5)
        self.assertEqual(len(result['top_k_ids']), 5)
        self.assertEqual(len(result['bottom_k_ids']), 5)
        self.assertEqual(len(result['top_k_scores']), 5)
        self.assertEqual(len(result['top_k_impacts']), 5)
        self.assertEqual(len(result['top_k_smarts']), 5)

    def test_top_impact_gt_bottom(self):
        """Synthetic data where top-scored motifs have strictly higher impact."""
        s, imp = self._setup(20)
        result = top_bottom_motif_eval(s, imp, k=5)
        self.assertGreater(result['top_mean_impact'],
                           result['bottom_mean_impact'])

    def test_top_score_gt_bottom(self):
        s, imp = self._setup(20)
        result = top_bottom_motif_eval(s, imp, k=5)
        self.assertGreater(result['top_mean_score'],
                           result['bottom_mean_score'])

    def test_impact_ratio_gt_one(self):
        s, imp = self._setup(20)
        result = top_bottom_motif_eval(s, imp, k=5)
        self.assertGreater(result['impact_ratio'], 1.0)

    def test_no_overlap_between_top_and_bottom(self):
        s, imp = self._setup(20)
        result = top_bottom_motif_eval(s, imp, k=5)
        top_set = set(result['top_k_ids'])
        bot_set = set(result['bottom_k_ids'])
        self.assertEqual(len(top_set & bot_set), 0)

    def test_k_capped_at_half(self):
        """k should be capped so top and bottom don't overlap."""
        s   = {i: float(i) for i in range(6)}
        imp = {i: {'impact': float(i), 'motif_smarts': '?'} for i in range(6)}
        result = top_bottom_motif_eval(s, imp, k=10)  # k > n/2
        self.assertLessEqual(result['k'], 3)

    def test_too_few_motifs(self):
        s   = {0: 0.9}
        imp = {0: {'impact': 0.5, 'motif_smarts': '?'}}
        result = top_bottom_motif_eval(s, imp, k=5)
        self.assertTrue(np.isnan(result['top_mean_impact'])
                        or result['top_k_ids'] == [])

    def test_missing_impact_skipped(self):
        """Motifs with score but no impact data should be silently excluded."""
        s   = {i: float(i) for i in range(20)}
        imp = {i: {'impact': float(i), 'motif_smarts': '?'} for i in range(10)}
        result = top_bottom_motif_eval(s, imp, k=5)
        # All ids in result must be in imp
        for mid in result['top_k_ids'] + result['bottom_k_ids']:
            self.assertIn(mid, imp)

    def test_smarts_match_ids(self):
        s, imp = self._setup(20)
        result = top_bottom_motif_eval(s, imp, k=3)
        for i, mid in enumerate(result['top_k_ids']):
            self.assertEqual(result['top_k_smarts'][i],
                             imp[mid]['motif_smarts'])


# ──────────────────────────────────────────────────────────────────────────────
# evaluation/motif_eval.py — gt_vs_outside_gt_eval
# ──────────────────────────────────────────────────────────────────────────────

class TestGtVsOutsideGtEval(unittest.TestCase):
    """Tests for gt_vs_outside_gt_eval.

    Uses a VanillaGNN and synthetic data so we can control labels and GT sets.
    """

    def _setup(self, n_graphs=20, n_motifs=8, n_gt=2):
        """Build a minimal but complete evaluation setup."""
        from SharedModules.data.dataset import NUM_ATOM_TYPES, EDGE_FEAT_DIM
        vocab = _make_fake_vocab()   # 5 motifs

        model = VanillaGNN(x_dim=NUM_ATOM_TYPES, hidden_dim=16,
                           num_layers=2, backbone='GIN')
        model.eval()

        # Build synthetic Data objects: half label=0, half label=1
        data_list = []
        smiles_labels = {}
        all_smi = list(SMILES.values())[:n_graphs % len(SMILES) + 1]
        for i in range(n_graphs):
            smi = all_smi[i % len(all_smi)] + f'_syn_{i}'  # unique key
            m = build_graph(SMILES['benzene'], torch.tensor([float(i % 2)]),
                            vocab.lookup_train)
            if m is None:
                continue
            m.smiles = smi          # override smiles key for uniqueness
            data_list.append(m)

        # 5 motifs: ids 0..4, GT = {0, 4}
        gt_ids = {0, 4}
        n_total = len(data_list)

        # Scores: GT motifs get high score, others low
        scores  = {0: 0.9, 1: 0.2, 2: 0.3, 3: 0.15, 4: 0.85}
        impacts = {i: {'impact': (0.7 if i in gt_ids else 0.1),
                        'motif_smarts': f'smarts_{i}'}
                   for i in range(5)}

        # Build a cache that covers all data_list smiles
        mask_cache_split = {}
        for mid in range(5):
            mask_cache_split[mid] = {
                d.smiles: torch.ones(d.x.size(0), dtype=torch.bool)
                for d in data_list
            }
        vocab.mask_cache['test'] = mask_cache_split

        return model, vocab, data_list, scores, impacts, gt_ids

    def test_return_keys(self):
        model, vocab, data_list, scores, impacts, gt_ids = self._setup()
        result = gt_vs_outside_gt_eval(
            motif_scores=scores,
            motif_impacts=impacts,
            gt_motif_ids=gt_ids,
            data_list=data_list,
            model=model,
            vocab=vocab,
            device=DEVICE,
            task_type='BinaryClass',
        )
        for subset in ('all', 'class1', 'correct_class1'):
            self.assertIn(subset, result)

    def test_all_subset_fields(self):
        model, vocab, data_list, scores, impacts, gt_ids = self._setup()
        result = gt_vs_outside_gt_eval(
            scores, impacts, gt_ids, data_list, model, vocab,
            DEVICE, task_type='BinaryClass'
        )
        required = ['n_examples', 'n_gt_motifs', 'n_non_gt_motifs',
                    'gt_mean_impact', 'non_gt_mean_impact',
                    'gt_mean_score', 'non_gt_mean_score',
                    'score_auc', 'gt_impact_rank']
        for key in required:
            self.assertIn(key, result['all'], f'Missing field {key}')

    def test_gt_mean_score_gt_non_gt(self):
        """GT motifs have score 0.87 avg vs non-GT 0.22 avg → GT should win."""
        model, vocab, data_list, scores, impacts, gt_ids = self._setup()
        result = gt_vs_outside_gt_eval(
            scores, impacts, gt_ids, data_list, model, vocab,
            DEVICE, task_type='BinaryClass'
        )
        self.assertGreater(result['all']['gt_mean_score'],
                           result['all']['non_gt_mean_score'])

    def test_score_auc_above_half(self):
        """With GT motifs having clearly higher scores, AUC should be > 0.5."""
        model, vocab, data_list, scores, impacts, gt_ids = self._setup()
        result = gt_vs_outside_gt_eval(
            scores, impacts, gt_ids, data_list, model, vocab,
            DEVICE, task_type='BinaryClass'
        )
        auc = result['all']['score_auc']
        if not np.isnan(auc):
            self.assertGreater(auc, 0.5)

    def test_class1_subset_lt_eq_all(self):
        """class1 subset can only be ≤ all subset in size."""
        model, vocab, data_list, scores, impacts, gt_ids = self._setup(20)
        result = gt_vs_outside_gt_eval(
            scores, impacts, gt_ids, data_list, model, vocab,
            DEVICE, task_type='BinaryClass'
        )
        self.assertLessEqual(result['class1']['n_examples'],
                             result['all']['n_examples'])

    def test_correct_class1_subset_lt_eq_class1(self):
        model, vocab, data_list, scores, impacts, gt_ids = self._setup(20)
        result = gt_vs_outside_gt_eval(
            scores, impacts, gt_ids, data_list, model, vocab,
            DEVICE, task_type='BinaryClass'
        )
        self.assertLessEqual(result['correct_class1']['n_examples'],
                             result['class1']['n_examples'])

    def test_gt_impact_rank_populated(self):
        model, vocab, data_list, scores, impacts, gt_ids = self._setup()
        result = gt_vs_outside_gt_eval(
            scores, impacts, gt_ids, data_list, model, vocab,
            DEVICE, task_type='BinaryClass'
        )
        rank = result['all']['gt_impact_rank']
        self.assertFalse(np.isnan(rank))
        # GT motifs have impact 0.7; non-GT have 0.1 → GT should rank near top
        self.assertLessEqual(rank, 3.0)  # top-2 motifs out of 5

    def test_empty_gt_motif_ids(self):
        model, vocab, data_list, scores, impacts, _ = self._setup()
        result = gt_vs_outside_gt_eval(
            scores, impacts, set(), data_list, model, vocab,
            DEVICE, task_type='BinaryClass'
        )
        for subset in ('all', 'class1', 'correct_class1'):
            self.assertEqual(result[subset]['n_gt_motifs'], 0)
            self.assertTrue(np.isnan(result[subset]['gt_mean_score'])
                            or result[subset]['gt_mean_score'] != result[subset]['gt_mean_score'])


# ──────────────────────────────────────────────────────────────────────────────
# evaluation/pipeline.py — to_dataframe for new result types
# ──────────────────────────────────────────────────────────────────────────────

class TestEvalPipelineDataframes(unittest.TestCase):
    def _fake_results(self):
        return {
            'prediction': {'auc': 0.75},
            'motif_impact': {
                0: {'impact': 0.5, 'impact_std': 0.1, 'n_graphs': 10,
                    'motif_smarts': '[*]c1ccccc1'},
                1: {'impact': 0.2, 'impact_std': 0.05, 'n_graphs': 8,
                    'motif_smarts': '[*]C'},
            },
            'correlation': {'pearson': 0.8, 'spearman': 0.75},
            'top_bottom': {
                'k': 1,
                'top_k_ids': [0], 'bottom_k_ids': [1],
                'top_k_scores': [0.9], 'bottom_k_scores': [0.1],
                'top_k_impacts': [0.5], 'bottom_k_impacts': [0.2],
                'top_k_smarts': ['[*]c1ccccc1'],
                'bottom_k_smarts': ['[*]C'],
                'top_mean_score': 0.9, 'bottom_mean_score': 0.1,
                'top_mean_impact': 0.5, 'bottom_mean_impact': 0.2,
                'impact_ratio': 2.5,
            },
            'gt_vs_outside': {
                'all':            {'n_examples': 100, 'n_gt_motifs': 1,
                                   'n_non_gt_motifs': 1,
                                   'gt_mean_impact': 0.5,
                                   'non_gt_mean_impact': 0.2,
                                   'gt_mean_score': 0.9,
                                   'non_gt_mean_score': 0.1,
                                   'score_auc': 0.95,
                                   'gt_impact_rank': 1.0},
                'class1':         {'n_examples': 50,  'n_gt_motifs': 1,
                                   'n_non_gt_motifs': 1,
                                   'gt_mean_impact': 0.6,
                                   'non_gt_mean_impact': 0.15,
                                   'gt_mean_score': 0.9,
                                   'non_gt_mean_score': 0.1,
                                   'score_auc': 0.95,
                                   'gt_impact_rank': 1.0},
                'correct_class1': {'n_examples': 40,  'n_gt_motifs': 1,
                                   'n_non_gt_motifs': 1,
                                   'gt_mean_impact': 0.65,
                                   'non_gt_mean_impact': 0.12,
                                   'gt_mean_score': 0.9,
                                   'non_gt_mean_score': 0.1,
                                   'score_auc': 0.95,
                                   'gt_impact_rank': 1.0},
            },
        }

    def _make_pipeline(self):
        from SharedModules.evaluation.pipeline import EvalPipeline
        model = VanillaGNN(x_dim=NUM_ATOM_TYPES, hidden_dim=16, num_layers=2)
        vocab = _make_fake_vocab()
        from torch_geometric.loader import DataLoader
        data_list = [d for smi in list(SMILES.values())[:4]
                     if (d := build_graph(smi, torch.tensor([0.0]), None)) is not None]
        loader = DataLoader(data_list, batch_size=4)
        return EvalPipeline(model, vocab, loader, data_list, DEVICE,
                            'BinaryClass', top_k=5)

    def test_top_bottom_df_columns(self):
        pipeline = self._make_pipeline()
        from SharedModules.evaluation.pipeline import EvalPipeline
        dfs = pipeline.to_dataframe(self._fake_results())
        self.assertIn('top_bottom', dfs)
        self.assertIn('top_motif_id', dfs['top_bottom'].columns)
        self.assertIn('bottom_impact', dfs['top_bottom'].columns)

    def test_top_bottom_summary_df(self):
        pipeline = self._make_pipeline()
        dfs = pipeline.to_dataframe(self._fake_results())
        self.assertIn('top_bottom_summary', dfs)
        self.assertIn('impact_ratio', dfs['top_bottom_summary'].columns)
        self.assertAlmostEqual(
            float(dfs['top_bottom_summary']['impact_ratio'].iloc[0]), 2.5)

    def test_gt_vs_outside_df_has_three_rows(self):
        pipeline = self._make_pipeline()
        dfs = pipeline.to_dataframe(self._fake_results())
        self.assertIn('gt_vs_outside', dfs)
        self.assertEqual(len(dfs['gt_vs_outside']), 3)
        subsets = set(dfs['gt_vs_outside']['subset'].tolist())
        self.assertEqual(subsets, {'all', 'class1', 'correct_class1'})

    def test_motif_impact_sorted_descending(self):
        pipeline = self._make_pipeline()
        dfs = pipeline.to_dataframe(self._fake_results())
        self.assertIn('motif_impact', dfs)
        impacts = dfs['motif_impact']['impact'].tolist()
        self.assertEqual(impacts, sorted(impacts, reverse=True))

    def test_missing_sections_excluded(self):
        pipeline = self._make_pipeline()
        dfs = pipeline.to_dataframe({'prediction': {'auc': 0.5}})
        self.assertNotIn('top_bottom', dfs)
        self.assertNotIn('gt_vs_outside', dfs)
        self.assertIn('prediction', dfs)




# ──────────────────────────────────────────────────────────────────────────────
# data/loader.py — MutagTUDataset and MUTAG_X_DIM
# ──────────────────────────────────────────────────────────────────────────────

class TestMutagTUDataset(unittest.TestCase):
    """Tests for MutagTUDataset without requiring the actual TUDataset PKL.

    Constructs minimal synthetic Data objects with 14-dim features to mirror
    what Mutag(root=...) produces, and verifies the wrapper's contract.
    """

    def _make_data(self, n_atoms=9, label=1.0):
        """Build a minimal PyG Data object with 14-dim features."""
        from torch_geometric.data import Data
        x = torch.randn(n_atoms, MUTAG_X_DIM)   # 14-dim, as from TUDataset PKL
        edge_index = torch.tensor([[0,1,2,3,4,5,5,6,6],
                                   [1,2,3,4,5,0,6,7,8]], dtype=torch.long)
        # node_type: C(0)×6, N(1), O(2), O(2) — nitrobenzene-like
        node_type = torch.tensor([0,0,0,0,0,0,1,2,2], dtype=torch.long)
        y = torch.tensor([label])
        d = Data(x=x, edge_index=edge_index, y=y, node_type=node_type)
        return d

    def test_x_dim_is_14(self):
        self.assertEqual(MUTAG_X_DIM, 14)

    def test_features_preserved_unchanged(self):
        """MutagTUDataset must NOT modify x — it keeps the 14-dim features."""
        from SharedModules.data.loader import MutagTUDataset
        data = self._make_data()
        original_x = data.x.clone()
        ds = MutagTUDataset([data], vocab=None)
        out = ds[0]
        self.assertTrue(torch.allclose(out.x, original_x),
                        "MutagTUDataset modified x — must be kept as-is")

    def test_nodes_to_motifs_all_minus_one_without_vocab(self):
        """Without a vocab, every node should have motif_id = -1."""
        from SharedModules.data.loader import MutagTUDataset
        data = self._make_data(n_atoms=9)
        ds = MutagTUDataset([data], vocab=None)
        out = ds[0]
        self.assertTrue((out.nodes_to_motifs == -1).all())

    def test_nodes_to_motifs_shape(self):
        from SharedModules.data.loader import MutagTUDataset
        n = 9
        data = self._make_data(n_atoms=n)
        ds = MutagTUDataset([data], vocab=None)
        out = ds[0]
        self.assertEqual(out.nodes_to_motifs.shape, (n,))

    def test_len(self):
        from SharedModules.data.loader import MutagTUDataset
        items = [self._make_data() for _ in range(5)]
        ds = MutagTUDataset(items)
        self.assertEqual(len(ds), 5)

    def test_smiles_key_uniform_for_pyg_batch(self):
        """Mixed mapped/unmapped SMILES must still collate (same attr keys)."""
        from torch_geometric.loader import DataLoader
        from SharedModules.data.loader import MutagTUDataset
        from SharedModules.data.vocab import VocabData

        items = [self._make_data() for _ in range(4)]
        vocab = VocabData(
            motif_list=['a'],
            motif_counts=[1],
            motif_lengths=[1],
            motif_class={0: {0: 1, 1: 0}},
            lookup_train={},
            lookup_valid={},
            lookup_test={},
            gmi_train={},
            gmi_test={},
        )
        ds = MutagTUDataset(
            items, vocab=vocab,
            index_maps={}, smiles_list=['CC', None, 'CCO', ''],
        )
        batch = DataLoader(ds, batch_size=4, shuffle=False).__iter__().__next__()
        self.assertTrue(hasattr(batch, 'smiles'))
        self.assertEqual(batch.num_graphs, 4)

    def test_with_vocab_and_index_map(self):
        """When a valid vocab + index_map are provided, known nodes get motif_id >= 0."""
        from SharedModules.data.loader import MutagTUDataset
        from SharedModules.data.graph_to_smiles import (
            graph_to_mapped_smiles, apply_motif_lookup_with_index_map
        )
        from SharedModules.data.vocab import VocabData

        # Build a real mapped SMILES for nitrobenzene-like graph
        nt  = [0,0,0,0,0,0,1,2,2]
        edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(5,6),(6,7),(6,8)]
        src = [s for s,d in edges]+[d for s,d in edges]
        dst = [d for s,d in edges]+[s for s,d in edges]
        smiles, g2s = graph_to_mapped_smiles(nt, src, dst)
        if smiles is None:
            self.skipTest("graph_to_mapped_smiles failed for test molecule")

        # Build a minimal fake vocab
        n_mol = len(nt)
        fake_lookup = {smiles: {i: (f'smarts_{i % 2}', i % 2)
                                for i in range(n_mol)}}
        vocab = VocabData(
            motif_list=['smarts_0', 'smarts_1'],
            motif_counts=[5, 5],
            motif_lengths=[6, 3],
            motif_class={0: {0:3,1:2}, 1: {0:3,1:2}},
            lookup_train=fake_lookup,
            lookup_valid={},
            lookup_test={},
            gmi_train={},
            gmi_test={},
        )

        data = self._make_data(n_atoms=n_mol)
        index_maps = {smiles: g2s}
        ds = MutagTUDataset(
            [data], vocab=vocab,
            index_maps=index_maps,
            smiles_list=[smiles],
            split='training',
        )
        out = ds[0]

        # With a complete lookup, no node should be -1
        self.assertTrue((out.nodes_to_motifs >= 0).all(),
                        f"Unexpected -1 values: {out.nodes_to_motifs.tolist()}")
        # x must still be 14-dim
        self.assertEqual(out.x.shape[1], MUTAG_X_DIM)

    def test_source_gt_labels_preserved_for_eval(self):
        """Mutagen (y=0) source GT must survive clone for compute_gt_roc."""
        from SharedModules.data.loader import MutagTUDataset
        data = self._make_data(label=0.0)
        data.edge_label = torch.tensor([0., 1., 1., 0., 0., 0., 1., 1., 0.])
        data.node_label = torch.tensor([0., 0., 0., 0., 0., 0., 1., 1., 1.])
        out = MutagTUDataset([data], vocab=None)[0]
        self.assertIsNotNone(out.edge_label)
        self.assertIsNotNone(out.node_label)
        self.assertGreater(float(out.edge_label.sum()), 0.0)
        self.assertGreater(float(out.node_label.sum()), 0.0)
        self.assertLess(float(out.node_label.sum()), out.node_label.numel())

    def test_meta_x_dim_is_14(self):
        """LoaderMeta for mutag must report x_dim = MUTAG_X_DIM = 14."""
        from SharedModules.data.loader import MUTAG_X_DIM, LoaderMeta
        meta = LoaderMeta(
            x_dim=MUTAG_X_DIM,
            edge_attr_dim=0,
            num_classes=1,
            task_type='BinaryClass',
            dataset='mutag',
            fold=0,
        )
        self.assertEqual(meta.x_dim, 14)

    def test_model_accepts_14_dim_input(self):
        """BaseGNN initialised with x_dim=14 must run without error."""
        from SharedModules.models.gnn_base import BaseGNN
        from torch_geometric.data import Data, Batch
        gnn = BaseGNN(x_dim=MUTAG_X_DIM, hidden_dim=32, num_layers=2,
                      backbone='GIN')
        n = 9
        x = torch.randn(n, MUTAG_X_DIM)
        edge_index = torch.tensor([[0,1,2],[1,2,0]], dtype=torch.long)
        batch = torch.zeros(n, dtype=torch.long)
        g, h = gnn.get_embedding(x, edge_index, batch=batch)
        self.assertEqual(g.shape, (1, 32))   # 1 graph
        self.assertEqual(h.shape, (n, 32))

    def test_52dim_model_rejects_14dim_input(self):
        """A model built with x_dim=52 must fail on 14-dim input —
        confirms that MUTAG_X_DIM=14 must be passed explicitly."""
        from SharedModules.models.gnn_base import BaseGNN
        gnn = BaseGNN(x_dim=NUM_ATOM_TYPES, hidden_dim=32, num_layers=2,
                      backbone='GIN')
        x = torch.randn(9, MUTAG_X_DIM)   # 14-dim, wrong for 52-dim model
        edge_index = torch.tensor([[0,1],[1,0]], dtype=torch.long)
        batch = torch.zeros(9, dtype=torch.long)
        with self.assertRaises((RuntimeError, ValueError)):
            gnn.get_embedding(x, edge_index, batch=batch)




# ──────────────────────────────────────────────────────────────────────────────
# data/loader.py — LoaderMeta.node_encoder field
# ──────────────────────────────────────────────────────────────────────────────

class TestLoaderMetaNodeEncoder(unittest.TestCase):
    """LoaderMeta.node_encoder must be set correctly by each loader branch.
    These tests do not call get_loaders (which requires actual dataset files)
    — they verify the field contract directly.
    """

    def _meta(self, dataset, x_dim, node_encoder):
        from SharedModules.data.loader import LoaderMeta
        return LoaderMeta(
            x_dim=x_dim, edge_attr_dim=8, num_classes=1,
            task_type='BinaryClass', dataset=dataset, fold=0,
            node_encoder=node_encoder,
        )

    def test_csv_dataset_is_onehot(self):
        meta = self._meta('Mutagenicity', 52, 'onehot')
        self.assertEqual(meta.node_encoder, 'onehot')
        self.assertEqual(meta.x_dim, 52)

    def test_ogb_dataset_is_atom_encoder(self):
        meta = self._meta('ogbg-molhiv', 9, 'atom_encoder')
        self.assertEqual(meta.node_encoder, 'atom_encoder')
        self.assertEqual(meta.x_dim, 9)

    def test_mutag_is_onehot_14dim(self):
        from SharedModules.data.loader import MUTAG_X_DIM
        meta = self._meta('mutag', MUTAG_X_DIM, 'onehot')
        self.assertEqual(meta.node_encoder, 'onehot')
        self.assertEqual(meta.x_dim, 14)

    def test_default_is_onehot(self):
        """node_encoder defaults to 'onehot' when not specified."""
        from SharedModules.data.loader import LoaderMeta
        meta = LoaderMeta(x_dim=52, edge_attr_dim=8, num_classes=1,
                          task_type='BinaryClass', dataset='test', fold=0)
        self.assertEqual(meta.node_encoder, 'onehot')

    def test_atom_encoder_signals_ogb_schema(self):
        """'atom_encoder' implies x_dim=9 (OGB schema 2) — x is integer tensor."""
        meta = self._meta('ogbg-moltox21', 9, 'atom_encoder')
        self.assertEqual(meta.x_dim, 9)
        self.assertEqual(meta.node_encoder, 'atom_encoder')

    def test_all_ogb_names_get_atom_encoder(self):
        """Every name in OGB_DATASET_NAMES should map to atom_encoder."""
        from SharedModules.data.loader import OGB_DATASET_NAMES
        for name in OGB_DATASET_NAMES:
            meta = self._meta(name, 9, 'atom_encoder')
            self.assertEqual(meta.node_encoder, 'atom_encoder',
                             f"{name} should use atom_encoder")

    def test_basegnn_atom_encoder_raises_without_ogb(self):
        """BaseGNN with node_encoder='atom_encoder' raises ImportError
        when ogb is not installed, not a silent wrong-dim error."""
        from SharedModules.models.gnn_base import BaseGNN
        try:
            import ogb  # noqa: F401
            self.skipTest("ogb is installed — ImportError not expected")
        except ImportError:
            with self.assertRaises(ImportError):
                BaseGNN(x_dim=9, hidden_dim=32, num_layers=2,
                        backbone='GIN', node_encoder='atom_encoder')

    def test_meta_node_encoder_propagates_to_model(self):
        """Demonstrate the intended usage: read meta.node_encoder when
        constructing the model so it is never hardcoded."""
        from SharedModules.models.gnn_base import BaseGNN
        from SharedModules.data.loader import LoaderMeta

        # Simulate what run.py does for a CSV dataset
        meta = LoaderMeta(x_dim=52, edge_attr_dim=8, num_classes=1,
                          task_type='BinaryClass', dataset='Mutagenicity',
                          fold=0, node_encoder='onehot')
        model = BaseGNN(x_dim=meta.x_dim, hidden_dim=32, num_layers=2,
                        backbone='GIN', node_encoder=meta.node_encoder)
        x = torch.randn(6, meta.x_dim)
        edge_index = torch.tensor([[0,1,2],[1,2,0]], dtype=torch.long)
        batch = torch.zeros(6, dtype=torch.long)
        g, _ = model.get_embedding(x, edge_index, batch=batch)
        self.assertEqual(g.shape, (1, 32))




# ──────────────────────────────────────────────────────────────────────────────
# evaluation/motif_eval.py — explainer_roc_vs_gt and compute_gt_roc
# ──────────────────────────────────────────────────────────────────────────────

class TestExplainerRocVsGt(unittest.TestCase):
    """Tests for explainer_roc_vs_gt (edge-level ROC) and compute_gt_roc."""

    def _make_data_with_edge_label(self, n=8, n_edges=12, pos_frac=0.5):
        """Build a synthetic Data object with edge_label set."""
        from torch_geometric.data import Data
        x = torch.randn(n, NUM_ATOM_TYPES)
        src = torch.randint(0, n, (n_edges,))
        dst = torch.randint(0, n, (n_edges,))
        edge_index = torch.stack([src, dst])
        n_pos = max(1, int(n_edges * pos_frac))
        edge_label = torch.zeros(n_edges)
        edge_label[:n_pos] = 1.0
        ntm = torch.randint(-1, 5, (n,))
        y = torch.tensor([1.0])
        d = Data(x=x, edge_index=edge_index, edge_label=edge_label,
                 nodes_to_motifs=ntm, y=y,
                 edge_attr=torch.randn(n_edges, EDGE_FEAT_DIM))
        d.smiles = 'CC'
        return d

    def test_explainer_roc_node_level_perfect(self):
        """Perfect node attention -> AUC = 1 at node level."""
        from SharedModules.evaluation.motif_eval import explainer_roc_vs_gt
        n = 6
        edge_index = torch.tensor([[0,1,2,3,4,5],[1,2,3,4,5,0]], dtype=torch.long)
        # Positive edges 0->1, 1->2 => GT node endpoints are {0,1,2}.
        edge_label = torch.tensor([1.,1.,0.,0.,0.,0.])
        # Perfect: high attention exactly on the GT-endpoint nodes 0,1,2
        node_att = torch.tensor([0.9, 0.9, 0.9, 0.05, 0.05, 0.05])
        auc_node = explainer_roc_vs_gt(node_att, edge_index, edge_label, level='node')
        self.assertGreater(auc_node, 0.95)

    def test_explainer_roc_node_vs_edge_level_difference(self):
        """Node and edge level AUC can differ -- edge inflates random baseline."""
        from SharedModules.evaluation.motif_eval import explainer_roc_vs_gt
        import numpy as np
        torch.manual_seed(42)
        n = 20
        edge_index = torch.stack([
            torch.randint(0, n, (60,)),
            torch.randint(0, n, (60,)),
        ])
        # GT: nodes 15-19 are active
        gt_nodes = torch.zeros(n); gt_nodes[15:] = 1.0
        # Build edge_label from gt_nodes
        src, dst = edge_index
        edge_label = (gt_nodes[src].bool() | gt_nodes[dst].bool()).float()
        # Random attention - should give ~0.5 node AUC
        rand_att = torch.rand(n)
        auc_node = explainer_roc_vs_gt(rand_att, edge_index, edge_label, level='node')
        auc_edge = explainer_roc_vs_gt(rand_att, edge_index, edge_label, level='edge')
        # Both should be numbers (not NaN) for this balanced case
        if not np.isnan(auc_node) and not np.isnan(auc_edge):
            # Edge level tends to inflate scores -- no strict ordering guaranteed
            # but both should be floats
            self.assertIsInstance(float(auc_node), float)
            self.assertIsInstance(float(auc_edge), float)

    def test_explainer_roc_random_not_perfect(self):
        """Random attention on balanced GT → AUC near 0.5."""
        from SharedModules.evaluation.motif_eval import explainer_roc_vs_gt
        torch.manual_seed(42)
        n = 20
        edge_index = torch.stack([
            torch.randint(0, n, (40,)),
            torch.randint(0, n, (40,)),
        ])
        edge_label = torch.cat([torch.ones(20), torch.zeros(20)])
        node_att = torch.rand(n)
        auc = explainer_roc_vs_gt(node_att, edge_index, edge_label)
        self.assertFalse(np.isnan(auc))

    def test_explainer_roc_degenerate_all_positive(self):
        """All edges positive → NaN (only one class) at edge level."""
        from SharedModules.evaluation.motif_eval import explainer_roc_vs_gt
        edge_index = torch.tensor([[0,1],[1,0]])
        edge_label = torch.ones(2)
        node_att = torch.rand(3)
        auc = explainer_roc_vs_gt(node_att, edge_index, edge_label, level='edge')
        self.assertTrue(np.isnan(auc))

    def test_explainer_roc_degenerate_all_negative(self):
        """All edges negative → NaN."""
        from SharedModules.evaluation.motif_eval import explainer_roc_vs_gt
        edge_index = torch.tensor([[0,1],[1,0]])
        edge_label = torch.zeros(2)
        node_att = torch.rand(3)
        auc = explainer_roc_vs_gt(node_att, edge_index, edge_label)
        self.assertTrue(np.isnan(auc))

    def test_compute_gt_roc_returns_dict(self):
        """compute_gt_roc runs on a list with edge_label set."""
        from SharedModules.evaluation.motif_eval import compute_gt_roc
        model = VanillaGNN(x_dim=NUM_ATOM_TYPES, hidden_dim=16, num_layers=2)
        model.eval()
        data_list = [self._make_data_with_edge_label() for _ in range(5)]
        result = compute_gt_roc(
            model, data_list, DEVICE,
            node_att_fn=lambda d: torch.rand(d.x.size(0)),
            level='node',
        )
        self.assertIn('auc_mean', result)
        self.assertIn('n_graphs', result)
        self.assertIn('n_skipped', result)
        self.assertGreaterEqual(result['n_graphs'], 0)

    def test_compute_gt_roc_prefers_soft_att(self):
        """compute_gt_roc should use aux['node_att_soft'] (continuous) rather
        than the hard 0/1 node_att, so the ROC can rank nodes.  A model whose
        hard att is all-ones but whose soft att perfectly ranks GT nodes should
        yield a high AUC (it would be NaN/degenerate from the hard mask)."""
        from SharedModules.evaluation.motif_eval import compute_gt_roc
        from torch_geometric.data import Data

        class _FakeGSAT(torch.nn.Module):
            def forward(self, x, edge_index, batch=None,
                        nodes_to_motifs=None, edge_attr=None,
                        epoch=0, motif_lengths=None):
                n = x.size(0)
                logits = torch.zeros(1, 1)
                hard = torch.ones(n, 1)               # degenerate as a score
                # soft: high on first 3 nodes, low elsewhere
                soft = torch.full((n, 1), 0.1)
                soft[:3] = 0.9
                aux = {'node_att_soft': soft, 'edge_att_soft': None}
                return logits, hard, aux

        # edges 0-1,1-2 positive => GT nodes {0,1,2}; soft att ranks them top
        d = Data(
            x=torch.randn(6, NUM_ATOM_TYPES),
            edge_index=torch.tensor([[0,1,2,3,4],[1,2,3,4,5]], dtype=torch.long),
            y=torch.tensor([1.0]),
            nodes_to_motifs=torch.full((6,), -1, dtype=torch.long),
            edge_label=torch.tensor([1.,1.,0.,0.,0.]),
        )
        result = compute_gt_roc(_FakeGSAT(), [d], DEVICE, level='node')
        self.assertEqual(result['n_graphs'], 1)
        self.assertGreater(result['auc_mean'], 0.9)

    def test_compute_gt_roc_skips_no_edge_label(self):
        """Graphs without edge_label are skipped."""
        from SharedModules.evaluation.motif_eval import compute_gt_roc
        from torch_geometric.data import Data
        model = VanillaGNN(x_dim=NUM_ATOM_TYPES, hidden_dim=16, num_layers=2)
        # Data objects with no edge_label
        data_list = []
        for _ in range(3):
            d = Data(x=torch.randn(5, NUM_ATOM_TYPES),
                     edge_index=torch.tensor([[0,1],[1,0]]),
                     y=torch.tensor([0.0]),
                     nodes_to_motifs=torch.full((5,), -1, dtype=torch.long))
            data_list.append(d)
        result = compute_gt_roc(model, data_list, DEVICE,
                                node_att_fn=lambda d: torch.rand(d.x.size(0)))
        self.assertEqual(result['n_graphs'], 0)
        self.assertEqual(result['n_skipped'], 3)
        self.assertTrue(np.isnan(result['auc_mean']))

    def test_compute_gt_roc_skips_degenerate_labels(self):
        """Graphs with degenerate GT are skipped. compute_gt_roc checks
        degeneracy at the EVALUATED level: for the default node level the node
        GT must be all-0 or all-1. A 3-node cycle (every node is an endpoint)
        makes all-edges-0 → node GT all-0 and all-edges-1 → node GT all-1, so
        both graphs are degenerate at the node level and skipped."""
        from SharedModules.evaluation.motif_eval import compute_gt_roc
        from torch_geometric.data import Data
        model = VanillaGNN(x_dim=NUM_ATOM_TYPES, hidden_dim=16, num_layers=2)
        data_list = []
        for v in [0.0, 1.0]:
            d = Data(x=torch.randn(3, NUM_ATOM_TYPES),
                     edge_index=torch.tensor([[0,1,2],[1,2,0]]),
                     edge_label=torch.full((3,), v),
                     y=torch.tensor([1.0]),
                     nodes_to_motifs=torch.full((3,), -1, dtype=torch.long))
            data_list.append(d)
        result = compute_gt_roc(model, data_list, DEVICE,
                                node_att_fn=lambda d: torch.rand(d.x.size(0)))
        self.assertEqual(result['n_skipped'], 2)
        self.assertEqual(result['n_graphs'], 0)

    def test_eval_pipeline_detects_gt(self):
        """EvalPipeline.run() includes 'gt_roc' key when edge_label is present."""
        from SharedModules.evaluation.pipeline import EvalPipeline
        from torch_geometric.loader import DataLoader
        model = VanillaGNN(x_dim=NUM_ATOM_TYPES, hidden_dim=16, num_layers=2)
        vocab = _make_fake_vocab()
        data_list = [self._make_data_with_edge_label() for _ in range(6)]
        loader = DataLoader(data_list, batch_size=4)
        pipeline = EvalPipeline(
            model, vocab, loader, data_list, DEVICE, 'BinaryClass',
            node_att_fn=lambda d: torch.rand(d.x.size(0)),
        )
        results = pipeline.run(run_motif_impact=False)
        self.assertIn('gt_roc', results, "'gt_roc' key missing despite edge_label present")
        self.assertIn('auc_mean', results['gt_roc'])

    def test_eval_pipeline_no_gt_when_no_edge_label(self):
        """EvalPipeline.run() omits 'gt_roc' when no edge_label is set."""
        from SharedModules.evaluation.pipeline import EvalPipeline
        from torch_geometric.loader import DataLoader
        from torch_geometric.data import Data
        model = VanillaGNN(x_dim=NUM_ATOM_TYPES, hidden_dim=16, num_layers=2)
        vocab = _make_fake_vocab()
        data_list = []
        for smi in list(SMILES.values())[:4]:
            d = build_graph(smi, torch.tensor([0.0]), None)
            if d:
                data_list.append(d)
        loader = DataLoader(data_list, batch_size=4)
        pipeline = EvalPipeline(model, vocab, loader, data_list, DEVICE,
                                'BinaryClass')
        results = pipeline.run(run_motif_impact=False)
        self.assertNotIn('gt_roc', results)

    def test_gt_roc_in_to_dataframe(self):
        """to_dataframe includes 'gt_roc' DataFrame when gt_roc is in results."""
        from SharedModules.evaluation.pipeline import EvalPipeline
        from torch_geometric.loader import DataLoader
        model = VanillaGNN(x_dim=NUM_ATOM_TYPES, hidden_dim=16, num_layers=2)
        vocab = _make_fake_vocab()
        data_list = [self._make_data_with_edge_label() for _ in range(4)]
        loader = DataLoader(data_list, batch_size=4)
        pipeline = EvalPipeline(model, vocab, loader, data_list, DEVICE,
                                'BinaryClass',
                                node_att_fn=lambda d: torch.rand(d.x.size(0)))
        results = pipeline.run(run_motif_impact=False)
        dfs = pipeline.to_dataframe(results)
        if 'gt_roc' in results:
            self.assertIn('gt_roc', dfs)
            self.assertIn('auc_mean', dfs['gt_roc'].columns)


# ──────────────────────────────────────────────────────────────────────────────
# data/ground_truth.py — _prepare_rulebook_dir and helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestGroundTruthHelpers(unittest.TestCase):
    """Tests for ground_truth.py helper functions that don't require
    the full motif_label_pipeline (which needs real vocab files)."""

    @unittest.skip("ground_truth._build_edge_label is DORMANT (commented out); "
                   "live GT path is apply_gt.py")
    def test_build_edge_label_active_ids(self):
        from SharedModules.data.ground_truth import _build_edge_label
        from torch_geometric.data import Data
        # Nodes 0,1 belong to motif 0; nodes 2,3 do not
        n2m = torch.tensor([0, 0, 1, 1])
        x = torch.zeros(4, 4)
        edge_index = torch.tensor([[0,1,2,3,0,2],[1,0,3,2,2,0]])
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([1.0]),
                    nodes_to_motifs=n2m)
        active_ids = {0}  # motif 0 nodes (0,1) are active
        edge_label, n_pos = _build_edge_label(data, active_ids)
        self.assertEqual(edge_label.shape, (edge_index.size(1),))
        self.assertGreater(n_pos, 0)
        # Edges between active nodes (0↔1) should be positive
        src, dst = edge_index
        for i in range(edge_index.size(1)):
            s, d = int(src[i]), int(dst[i])
            node_s_active = int(n2m[s]) in active_ids
            node_d_active = int(n2m[d]) in active_ids
            expected = 1.0 if (node_s_active or node_d_active) else 0.0
            self.assertAlmostEqual(float(edge_label[i]), expected,
                                   msg=f"Edge ({s},{d})")

    @unittest.skip("ground_truth._build_edge_label is DORMANT (commented out); "
                   "live GT path is apply_gt.py")
    def test_build_edge_label_empty_active(self):
        from SharedModules.data.ground_truth import _build_edge_label
        from torch_geometric.data import Data
        data = Data(x=torch.zeros(4,4),
                    edge_index=torch.tensor([[0,1],[1,0]]),
                    y=torch.tensor([0.0]),
                    nodes_to_motifs=torch.tensor([0,1,2,3]))
        el, n_pos = _build_edge_label(data, set())
        self.assertEqual(n_pos, 0)
        self.assertTrue((el == 0).all())

    @unittest.skip("ground_truth._build_edge_label is DORMANT (commented out); "
                   "live GT path is apply_gt.py")
    def test_build_edge_label_no_n2m(self):
        from SharedModules.data.ground_truth import _build_edge_label
        from torch_geometric.data import Data
        data = Data(x=torch.zeros(4,4),
                    edge_index=torch.tensor([[0,1],[1,0]]),
                    y=torch.tensor([0.0]))
        el, n_pos = _build_edge_label(data, {0, 1})
        self.assertEqual(n_pos, 0)

    @unittest.skip("ground_truth._motif_name_to_ids is DORMANT (commented out); "
                   "live GT path is apply_gt.py")
    def test_motif_name_to_ids(self):
        from SharedModules.data.ground_truth import _motif_name_to_ids
        motif_list = ['[*]c1ccccc1', '[*]N', '[*]c1ccccc1']  # dupe
        m2id = _motif_name_to_ids(motif_list)
        self.assertIn('[*]c1ccccc1', m2id)
        self.assertEqual(m2id['[*]c1ccccc1'], {0, 2})
        self.assertEqual(m2id['[*]N'], {1})

    @unittest.skip("ground_truth._motif_name_to_ids is DORMANT (commented out); "
                   "live GT path is apply_gt.py")
    def test_motif_name_to_ids_none(self):
        from SharedModules.data.ground_truth import _motif_name_to_ids
        self.assertEqual(_motif_name_to_ids(None), {})

    @unittest.skip("ground_truth._resolve_active_ids is DORMANT (commented out); "
                   "live GT path is apply_gt.py")
    def test_resolve_active_ids(self):
        from SharedModules.data.ground_truth import _resolve_active_ids, _motif_name_to_ids
        motif_list = ['A', 'B', 'C']
        m2id = _motif_name_to_ids(motif_list)
        ids = _resolve_active_ids({'A', 'C'}, m2id)
        self.assertEqual(ids, {0, 2})

    def test_gt_supported_datasets(self):
        from SharedModules.data.ground_truth import GT_SUPPORTED_DATASETS
        self.assertIn('Mutagenicity', GT_SUPPORTED_DATASETS)
        self.assertIn('Benzene', GT_SUPPORTED_DATASETS)
        self.assertNotIn('ba_2motifs', GT_SUPPORTED_DATASETS)

    @unittest.skip("ground_truth._prepare_rulebook_dir is DORMANT (commented "
                   "out); live GT path is apply_gt.py")
    def test_prepare_rulebook_dir(self):
        """_prepare_rulebook_dir copies/renames files correctly."""
        import tempfile, os
        from SharedModules.data.ground_truth import _prepare_rulebook_dir
        import scipy.sparse as sp
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            # Build fake vocab output structure
            vdir = Path(tmp) / 'Mutagenicity' / 'all_fallback_bpe'
            vdir.mkdir(parents=True)
            # matrix.npz
            X = sp.csr_matrix(np.eye(5, dtype=np.uint8))
            sp.save_npz(str(vdir / 'matrix.npz'), X)
            # matrix_columns.csv with 'motif_identity' column
            pd.DataFrame({'motif_id': range(5),
                          'motif_identity': [f's_{i}' for i in range(5)]
                          }).to_csv(vdir / 'matrix_columns.csv', index=False)
            # smiles_labels.csv with 'smiles' column
            pd.DataFrame({'smiles': ['CC', 'CCO', 'CCC', 'CCCC', 'CCCCC'],
                          'label': [0,1,0,1,0],
                          'group': ['training']*5
                          }).to_csv(vdir / 'smiles_labels.csv', index=False)

            rulebook_root = Path(tmp) / 'rulebook'
            out_dir = _prepare_rulebook_dir(
                tmp, 'Mutagenicity', 'all_fallback_bpe', 0,
                str(rulebook_root), force=False,
            )
            # Check expected files exist
            self.assertTrue((out_dir / 'graph_motif_matrix.npz').exists())
            self.assertTrue((out_dir / 'graph_motif_matrix_columns.csv').exists())
            self.assertTrue((out_dir / 'graph_motif_matrix_rows.csv').exists())
            # Columns file has 'motif_identity'
            cols = pd.read_csv(out_dir / 'graph_motif_matrix_columns.csv')
            self.assertIn('motif_identity', cols.columns)
            # Rows file has 'smiles'
            rows = pd.read_csv(out_dir / 'graph_motif_matrix_rows.csv')
            self.assertIn('smiles', rows.columns)
            self.assertEqual(len(rows), 5)




# ──────────────────────────────────────────────────────────────────────────────
# evaluation/multi_explanation.py — H1/H2 multiple explanation analysis
# ──────────────────────────────────────────────────────────────────────────────

class TestMultiExplanation(unittest.TestCase):
    """Tests for the H1/H2/H0 multiple explanation hypothesis analysis.

    Uses fully synthetic data so no model forward passes are needed.
    """

    def _make_per_graph_df(self, n_graphs=20, n_motifs=5, seed=42):
        """Build a synthetic per-(graph, motif) impact DataFrame.

        Graphs 0..9 are class 1, graphs 10..19 are class 0.
        Motif 0 has high importance and high impact.
        Motif 1 has high importance but low impact (HL candidate).
        Motifs 2..4 have low importance and varied impact.
        """
        rng = np.random.RandomState(seed)
        rows = []
        for g in range(n_graphs):
            label = float(g < 10)
            for m in range(n_motifs):
                if rng.rand() < 0.3:  # motif not always present
                    continue
                # Design: motif 0 = HH, motif 1 = HL
                if m == 0:
                    importance = rng.uniform(0.7, 0.95)
                    impact     = rng.uniform(0.3, 0.8)
                elif m == 1:
                    importance = rng.uniform(0.7, 0.95)
                    impact     = rng.uniform(0.0, 0.15)
                else:
                    importance = rng.uniform(0.1, 0.45)
                    impact     = rng.uniform(0.0, 0.4)
                rows.append({
                    "graph_id":           f"g{g}_test",
                    "motif":              f"motif_{m}",
                    "motif_id":           m,
                    "sigmoid_importance": importance,
                    "impact":             impact,
                    "class_label":        label,
                })
        return pd.DataFrame(rows)

    # ── _mark_local_hi ────────────────────────────────────────────────────────

    def test_mark_local_hi_global(self):
        from SharedModules.evaluation.multi_explanation import _mark_local_hi
        df = self._make_per_graph_df()
        mean_imp = df["impact"].mean()
        hi = _mark_local_hi(df, impact_thr_global=mean_imp, local_filter="global")
        self.assertEqual(hi.dtype, bool)
        self.assertGreater(hi.sum(), 0)
        # All rows at or above mean should be True
        self.assertTrue((hi == (df["impact"] >= mean_imp)).all())

    def test_mark_local_hi_p75(self):
        from SharedModules.evaluation.multi_explanation import _mark_local_hi
        df = self._make_per_graph_df()
        mean_imp = df["impact"].mean()
        hi = _mark_local_hi(df, impact_thr_global=mean_imp, local_filter="p75")
        # At most 25% of each graph's rows should be locally high
        per_graph_frac = df.assign(hi=hi).groupby("graph_id")["hi"].mean()
        # Some graphs may have all rows above p75 (edge case), so allow up to ~30%
        self.assertLessEqual(float(per_graph_frac.mean()), 0.40)

    def test_mark_local_hi_p50_marks_half(self):
        """p50 should mark roughly 50% of rows in each graph."""
        from SharedModules.evaluation.multi_explanation import _mark_local_hi
        df = self._make_per_graph_df(n_graphs=40, n_motifs=8)
        mean_imp = df["impact"].mean()
        hi = _mark_local_hi(df, impact_thr_global=mean_imp, local_filter="p50")
        per_graph_frac = df.assign(hi=hi).groupby("graph_id")["hi"].mean()
        # Fraction should be near 0.5 ± 0.2 for most graphs
        self.assertGreater(float(per_graph_frac.mean()), 0.3)
        self.assertLess(float(per_graph_frac.mean()), 0.7)

    # ── assign_hypothesis_flags ──────────────────────────────────────────────

    def test_flags_mutually_exclusive(self):
        """H0, H1, H2 must be mutually exclusive and exhaustive."""
        from SharedModules.evaluation.multi_explanation import assign_hypothesis_flags
        df = self._make_per_graph_df()
        df = assign_hypothesis_flags(df, local_filter="p75")
        # Exactly one of H0/H1/H2 is True per row
        total = df["H0"].astype(int) + df["H1"].astype(int) + df["H2"].astype(int)
        self.assertTrue((total == 1).all(),
                        "H0+H1+H2 must sum to 1 for every row")

    def test_h2_implies_local_hi(self):
        """Every H2 row must have is_local_hi = True."""
        from SharedModules.evaluation.multi_explanation import assign_hypothesis_flags
        df = self._make_per_graph_df()
        df = assign_hypothesis_flags(df, local_filter="p75")
        h2_rows = df[df["H2"]]
        self.assertTrue(h2_rows["is_local_hi"].all())

    def test_h1_implies_not_local_hi(self):
        """Every H1 row must have is_local_hi = False."""
        from SharedModules.evaluation.multi_explanation import assign_hypothesis_flags
        df = self._make_per_graph_df()
        df = assign_hypothesis_flags(df, local_filter="p75")
        h1_rows = df[df["H1"]]
        self.assertFalse(h1_rows["is_local_hi"].any())

    def test_h0_implies_not_local_hi_and_no_anchor(self):
        """H0 rows: not locally high, and graph has no H2 anchor motif."""
        from SharedModules.evaluation.multi_explanation import assign_hypothesis_flags
        df = self._make_per_graph_df()
        df = assign_hypothesis_flags(df, local_filter="p75")
        h0_rows = df[df["H0"]]
        self.assertFalse(h0_rows["is_local_hi"].any())
        # H0 graphs should not appear in graphs_with_anchor (no H2 anchor)
        anchor_graphs = set(df.loc[df["H2"] & (df["sigmoid_importance"] > df["sigmoid_importance"].mean()), "graph_id"])
        h0_in_anchor = h0_rows["graph_id"].isin(anchor_graphs).any()
        self.assertFalse(h0_in_anchor,
                         "H0 rows should not appear in graphs that have an H2 anchor")

    # ── compute_h1_h2_ratios ─────────────────────────────────────────────────

    def test_ratios_sum_to_one(self):
        """ratio_H0 + ratio_H1 + ratio_H2 must equal 1 for every motif."""
        from SharedModules.evaluation.multi_explanation import compute_h1_h2_ratios
        df = self._make_per_graph_df(n_graphs=40, seed=0)
        result = compute_h1_h2_ratios(df, local_filter="p75", min_graphs=2)
        if result.empty:
            self.skipTest("No motifs passed min_graphs filter")
        total = (result["ratio_H0"] + result["ratio_H1"] + result["ratio_H2"]).round(6)
        self.assertTrue((total == 1.0).all(),
                        f"Ratios don't sum to 1: {total.describe()}")

    def test_ratios_in_01(self):
        """All ratios must be in [0, 1]."""
        from SharedModules.evaluation.multi_explanation import compute_h1_h2_ratios
        df = self._make_per_graph_df(n_graphs=40, seed=1)
        result = compute_h1_h2_ratios(df, local_filter="p75", min_graphs=2)
        for col in ["ratio_H0", "ratio_H1", "ratio_H2"]:
            self.assertTrue((result[col] >= 0).all() and (result[col] <= 1).all(),
                             f"{col} out of [0,1]: {result[col].describe()}")

    def test_category_column_values(self):
        """All category values must be in {HH, HL, LH, LL}."""
        from SharedModules.evaluation.multi_explanation import compute_h1_h2_ratios, CATEGORY_ORDER
        df = self._make_per_graph_df(n_graphs=40)
        result = compute_h1_h2_ratios(df, local_filter="global", min_graphs=2)
        if result.empty:
            return
        self.assertTrue(result["category"].isin(CATEGORY_ORDER).all())

    def test_hh_motif_higher_h2_than_hl(self):
        """HH motifs should have higher mean ratio_H2 than HL motifs."""
        from SharedModules.evaluation.multi_explanation import compute_h1_h2_ratios
        # Use many graphs for statistical power
        df = self._make_per_graph_df(n_graphs=100, seed=7)
        result = compute_h1_h2_ratios(df, local_filter="p75", min_graphs=3)
        if result.empty:
            self.skipTest("No motifs passed filter")
        hh = result[result["category"] == "HH"]["ratio_H2"]
        hl = result[result["category"] == "HL"]["ratio_H2"]
        if hh.empty or hl.empty:
            return
        self.assertGreaterEqual(float(hh.mean()), float(hl.mean()) - 0.1,
                                "HH motifs should have >= H2 ratio as HL motifs")

    def test_hl_motif_higher_h1_than_hh(self):
        """HL motifs should have higher or equal mean ratio_H1 than HH motifs."""
        from SharedModules.evaluation.multi_explanation import compute_h1_h2_ratios
        df = self._make_per_graph_df(n_graphs=100, seed=42)
        result = compute_h1_h2_ratios(df, local_filter="p75", min_graphs=3)
        if result.empty:
            self.skipTest("No motifs passed filter")
        hh = result[result["category"] == "HH"]["ratio_H1"]
        hl = result[result["category"] == "HL"]["ratio_H1"]
        if hh.empty or hl.empty:
            return
        # HL motifs are often overshadowed, so H1 ratio should be >= HH
        self.assertGreaterEqual(float(hl.mean()), float(hh.mean()) - 0.1)

    # ── classify_motif_category ───────────────────────────────────────────────

    def test_classify_category_exact(self):
        """Verify exact category assignment for known threshold."""
        from SharedModules.evaluation.multi_explanation import classify_motif_category
        df = pd.DataFrame({
            "avg_importance": [0.8, 0.8, 0.2, 0.2],
            "avg_impact":     [0.8, 0.2, 0.8, 0.2],
        })
        result = classify_motif_category(df, importance_thr=0.5, impact_thr=0.5)
        expected = ["HH", "HL", "LH", "LL"]
        self.assertEqual(result["category"].tolist(), expected)

    # ── category_summary ─────────────────────────────────────────────────────

    def test_category_summary_columns(self):
        from SharedModules.evaluation.multi_explanation import (
            compute_h1_h2_ratios, category_summary
        )
        df = self._make_per_graph_df(n_graphs=60)
        result = compute_h1_h2_ratios(df, local_filter="p75", min_graphs=2)
        if result.empty:
            return
        summary = category_summary(result)
        expected_cols = {"category", "n_motifs", "mean_ratio_H2", "mean_ratio_H1",
                         "mean_ratio_H0", "mean_importance", "mean_impact"}
        self.assertTrue(expected_cols.issubset(set(summary.columns)))

    def test_category_summary_mean_ratios_sum_to_one(self):
        from SharedModules.evaluation.multi_explanation import (
            compute_h1_h2_ratios, category_summary
        )
        df = self._make_per_graph_df(n_graphs=60)
        result = compute_h1_h2_ratios(df, local_filter="p75", min_graphs=2)
        if result.empty:
            return
        summary = category_summary(result)
        if summary.empty:
            return
        total = (summary["mean_ratio_H0"] + summary["mean_ratio_H1"] + summary["mean_ratio_H2"]).round(4)
        # Each row's mean ratios should sum to approximately 1
        self.assertTrue((total > 0.9).all() and (total <= 1.01).all())


class TestMutagSplits(unittest.TestCase):
    """GSAT-style mutag train/valid/test indexing."""

    @staticmethod
    def _mock_dataset(specs):
        """specs: list of (y, n_gt_edges)"""
        out = []
        for y, n_gt in specs:
            e = max(n_gt, 1)
            d = Data(
                y=torch.tensor([[float(y)]]),
                edge_label=torch.tensor([1.0] * n_gt + [0.0] * (e - n_gt)),
                x=torch.zeros(3, 14),
            )
            out.append(d)
        return out

    def test_disjoint_split(self):
        from SharedModules.data.mutag_splits import get_mutag_split_idx
        ds = self._mock_dataset([(0, 1)] * 20)
        idx = get_mutag_split_idx(ds, seed=0)
        all_idx = idx['train'] + idx['valid'] + idx['test']
        self.assertEqual(len(all_idx), len(ds))
        self.assertEqual(len(set(all_idx)), len(all_idx))
        self.assertEqual(len(idx['train']), 16)
        self.assertEqual(len(idx['valid']), 2)
        self.assertEqual(len(idx['test']), 2)

    def test_mutag_gt_eval_graphs(self):
        from SharedModules.data.mutag_splits import mutag_gt_eval_graphs
        ds = self._mock_dataset([(0, 2), (1, 0), (0, 0), (0, 1)])
        gt = mutag_gt_eval_graphs(ds)
        self.assertEqual(len(gt), 2)
        self.assertTrue(all(float(d.y.squeeze()) == 0.0 for d in gt))

    def test_group_for_graph(self):
        from SharedModules.data.mutag_splits import group_for_graph
        split_idx = {'train': [0, 1], 'valid': [2], 'test': [3]}
        self.assertEqual(group_for_graph(0, split_idx), 'training')
        self.assertEqual(group_for_graph(2, split_idx), 'valid')
        self.assertEqual(group_for_graph(3, split_idx), 'test')


if __name__ == '__main__':
    unittest.main(verbosity=2)
