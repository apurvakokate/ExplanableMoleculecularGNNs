#!/usr/bin/env python3
"""test_gsat_parity.py — the GSAT-parity knobs (official GSAT reproduction).

Covers: arch_style='gsat' GIN/PNA (SharedModules/models/gsat_arch.py), lr_scheduler,
ckpt_metric='acc', the 'acc' prediction metric, the GSAT mutag split, the pooled
edge-AUC explanation metric, and build_model wiring. Defaults stay unchanged
(the rest of the suite covers arch_style='ours').

Run:  pytest MotifSAT/tests/test_gsat_parity.py -v
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
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch_geometric.data import Batch, Data  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from SharedModules.data.dataset import NUM_ATOM_TYPES, EDGE_FEAT_DIM  # noqa: E402
from SharedModules.data.mutag_splits import gsat_mutag_split_idx  # noqa: E402
from SharedModules.evaluation.metrics import evaluate_predictions  # noqa: E402
from SharedModules.models.gnn_base import BaseGNN  # noqa: E402
from SharedModules.models.gsat_arch import GSATArchGNN, GSATPNAConv, GSATGINConv  # noqa: E402
from model import GSAT  # noqa: E402
import train as T  # noqa: E402
from gsat_metrics import gsat_explanation_metrics, pooled_edge_auc  # noqa: E402

H = 16
DEG = torch.tensor([0., 4., 10., 3.])


def _graph(i, n=6, gen=None):
    x = torch.randn(n, NUM_ATOM_TYPES, generator=gen)
    x[:, 0] += 2.0 * (i % 2)
    ei = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    ei = torch.cat([ei, ei.flip(0)], 1)
    return Data(x=x, edge_index=ei, edge_attr=torch.randn(12, EDGE_FEAT_DIM, generator=gen),
                nodes_to_motifs=torch.tensor([0, 0, 1, 1, 2, 2]),
                y=torch.tensor([[float(i % 2)]]))


def _loaders(n=48):
    g = torch.Generator().manual_seed(0)
    gs = [_graph(i, gen=g) for i in range(n)]
    return {'train': DataLoader(gs[:32], batch_size=8, shuffle=True),
            'valid': DataLoader(gs[32:], batch_size=8)}


def _gsat(backbone='GIN', **kw):
    torch.manual_seed(0)
    return GSAT(x_dim=NUM_ATOM_TYPES, hidden_dim=H, num_layers=2, backbone_name=backbone,
                num_classes=1, deg=DEG, arch_style='gsat', init_r=0.9, final_r=0.7,
                decay_interval=10, decay_r=0.1, pna_scalers=False, **kw)


class TestGSATArch(unittest.TestCase):
    def test_gin_param_count_matches_official(self):
        m = GSATArchGNN(NUM_ATOM_TYPES, H, 2, 'GIN', num_out=1)
        enc = NUM_ATOM_TYPES * H + H
        mlp = 2 * (H * H + H) + 2 * H          # Lin + BN(weight,bias) + Lin
        head = H + 1
        self.assertEqual(sum(p.numel() for p in m.parameters()), enc + 2 * mlp + head)
        self.assertFalse(any(isinstance(p, torch.nn.Parameter) and p.numel() == 1
                             for n_, p in m.named_parameters() if n_.endswith('eps')))

    def test_gin_message_is_gated_self_term_is_not(self):
        conv = GSATGINConv(torch.nn.Identity())
        x = torch.randn(3, 4)
        ei = torch.tensor([[0, 1], [1, 2]])
        out = conv(x, ei, edge_atten=torch.zeros(2, 1))
        self.assertTrue(torch.allclose(out, x))             # (1+0)*x, no messages
        out1 = conv(x, ei, edge_atten=torch.ones(2, 1))
        self.assertTrue(torch.allclose(out1[1], x[1] + x[0]))

    def test_pna_message_concat_and_input_width(self):
        for aggs, scal in ((['mean', 'min', 'max', 'std'], ['identity']),
                           (['mean', 'min', 'max', 'std', 'sum'], ['identity']),
                           (['mean'], ['identity', 'amplification', 'attenuation'])):
            conv = GSATPNAConv(2 * H, H, aggs, scal, deg=DEG)
            self.assertEqual(conv.post_nn[0].in_features, len(aggs) * len(scal) * 2 * H)
        conv = GSATPNAConv(2 * 3, 3, ['sum'], ['identity'], deg=DEG)
        m = conv.message(torch.ones(2, 3), 2 * torch.ones(2, 3), edge_atten=None)
        self.assertEqual(m.tolist(), [[1, 1, 1, 2, 2, 2]] * 2)   # cat[x_i, x_j]

    def test_pna_rejects_unknown_aggregator(self):
        with self.assertRaises(ValueError):
            GSATPNAConv(4, 2, ['median'], ['identity'], deg=DEG)

    def test_pna_head_and_pooling(self):
        m = GSATArchGNN(NUM_ATOM_TYPES, H, 2, 'PNA', num_out=3, deg=DEG)
        self.assertEqual(m.graph_pool, 'mean')
        self.assertEqual([l.out_features for l in m.fc_out if hasattr(l, 'out_features')],
                         [H // 2, H // 4, 3])
        self.assertEqual(GSATArchGNN(NUM_ATOM_TYPES, H, 2, 'GIN', num_out=1).graph_pool, 'add')

    def test_only_gin_and_pna(self):
        with self.assertRaises(ValueError):
            GSATArchGNN(NUM_ATOM_TYPES, H, 2, 'GAT', num_out=1)

    def test_gsat_model_forward_backward(self):
        b = Batch.from_data_list([_graph(i) for i in range(4)])
        for bb in ('GIN', 'PNA'):
            m = _gsat(bb)
            self.assertIsInstance(m.clf, GSATArchGNN)
            m.train()
            logits, node_att, aux = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs,
                                      b.edge_attr, epoch=1)
            self.assertEqual(tuple(logits.shape), (4, 1))
            logits.sum().backward()
            self.assertTrue(all(p.grad is not None for p in m.extractor.parameters()))

    def test_default_arch_unchanged(self):
        m = GSAT(x_dim=NUM_ATOM_TYPES, hidden_dim=H, num_layers=2, backbone_name='GIN',
                 num_classes=1)
        self.assertIs(type(m.clf), BaseGNN)
        with self.assertRaises(ValueError):
            GSAT(x_dim=NUM_ATOM_TYPES, hidden_dim=H, num_layers=2, backbone_name='GIN',
                 num_classes=1, arch_style='official')


class TestOurPNAPieces(unittest.TestCase):
    """GSAT pieces switched onto OUR PNA (arch_style=ours): defaults unchanged."""
    def _m(self, pieces=()):
        torch.manual_seed(0)
        return GSAT(x_dim=NUM_ATOM_TYPES, hidden_dim=H, num_layers=2, backbone_name='PNA',
                    num_classes=1, deg=DEG, pna_pieces=pieces)

    def test_default_unchanged(self):
        m = self._m()
        self.assertEqual(m.clf.convs[0].post_nn[0].in_features, 4 * 3 * NUM_ATOM_TYPES)
        self.assertFalse(m.clf.pna_bn_residual)
        self.assertIsNone(m.clf.pna_batch_norms)

    def test_noscalers_drops_two_scalers(self):
        m = self._m(('noscalers',))
        self.assertEqual(len(m.clf.convs[1]._scalers), 1)
        self.assertEqual(m.clf.convs[1].post_nn[0].in_features, 4 * 1 * H)

    def test_bnres_structure_and_residual(self):
        m = self._m(('bnres',))
        self.assertTrue(m.clf.pna_bn_residual)
        self.assertEqual(len(m.clf.pna_batch_norms), 2)
        m.eval()
        x = torch.randn(6, H); ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
        conv, bn = m.clf.convs[1], m.clf.pna_batch_norms[1]
        h = torch.relu(bn(conv(x, ei)))
        # one-layer check of x + h through the second (width-matching) layer
        m.clf.convs = torch.nn.ModuleList([conv]); m.clf.pna_batch_norms = torch.nn.ModuleList([bn])
        self.assertTrue(torch.allclose(m.clf.convolve(x, ei), x + h, atol=1e-6))

    def test_bad_piece_rejected(self):
        with self.assertRaises(ValueError):
            self._m(('residual',))

    def test_forward_backward_with_all_pieces(self):
        b = Batch.from_data_list([_graph(i) for i in range(4)])
        m = self._m(('noscalers', 'bnres')); m.train()
        lg, _, _ = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs, b.edge_attr, epoch=1)
        lg.sum().backward()
        self.assertTrue(all(p.grad is not None for p in m.clf.pna_batch_norms.parameters()))


class TestTrainingKnobs(unittest.TestCase):
    def _fit(self, **kw):
        m = _gsat('GIN')
        return m, T.train_gsat(m, _loaders(), 'BinaryClass', torch.device('cpu'),
                               epochs=kw.pop('epochs', 15), patience=1000, min_epochs=1,
                               verbose=False, **kw)[1]

    def test_lr_scheduler_none_keeps_lr(self):
        captured = {}
        orig = torch.optim.Adam

        class SpyAdam(orig):
            def __init__(s, *a, **k):
                super().__init__(*a, **k); captured['opt'] = s
        torch.optim.Adam = SpyAdam
        try:
            self._fit(lr_scheduler='none', lr=1e-3)
        finally:
            torch.optim.Adam = orig
        self.assertEqual([g['lr'] for g in captured['opt'].param_groups], [1e-3])

    def test_bad_values_rejected(self):
        for kw in ({'lr_scheduler': 'cosine'}, {'ckpt_metric': 'f1'}):
            with self.assertRaises(ValueError):
                self._fit(epochs=1, **kw)

    def test_ckpt_metric_acc_uses_accuracy(self):
        seen = []
        orig = T._val_score

        def spy(metrics, task_type, ckpt_metric='auc'):
            seen.append(ckpt_metric)
            return orig(metrics, task_type, ckpt_metric)
        T._val_score = spy
        try:
            _, h = self._fit(ckpt_metric='acc', epochs=3)
        finally:
            T._val_score = orig
        self.assertEqual(set(seen), {'acc'})
        self.assertTrue(all(0.0 <= v <= 1.0 for v in h['val_metric']))

    def test_acc_metric_value(self):
        m = _gsat('GIN')
        loader = _loaders()['valid']
        out = evaluate_predictions(m, loader, torch.device('cpu'), 'BinaryClass')
        m.eval()
        preds, ys = [], []
        with torch.no_grad():
            for d in loader:
                lg, _, _ = m(d.x, d.edge_index, d.batch, d.nodes_to_motifs, d.edge_attr, epoch=1)
                preds.append(torch.sigmoid(lg).view(-1)); ys.append(d.y.view(-1))
        acc = float(((torch.cat(preds) > 0.5) == (torch.cat(ys) > 0.5)).float().mean())
        self.assertAlmostEqual(out['acc'], acc, places=6)


class _FakeG:
    def __init__(self, y, gt):
        self.y = torch.tensor([[float(y)]])
        self.edge_label = torch.tensor([1., 0.] if gt else [0., 0.])


class TestGSATMutagSplit(unittest.TestCase):
    def setUp(self):
        self.ds = [_FakeG(i % 2, gt=(i % 2 == 0 and i % 4 != 2)) for i in range(50)]

    def test_protocol(self):
        sp = gsat_mutag_split_idx(self.ds, seed=3)
        self.assertEqual(len(sp['train']), 40)
        self.assertEqual(len(sp['valid']), 10)
        self.assertEqual(sorted(sp['train'] + sp['valid']), list(range(50)))
        expect = [i for i in range(50) if i % 2 == 0 and i % 4 != 2]
        self.assertEqual(sp['test'], expect)        # all annotated mutagens (y==0)
        self.assertTrue(set(sp['test']) & set(sp['train']))   # overlap by design

    def test_seeded(self):
        a, b = gsat_mutag_split_idx(self.ds, 1), gsat_mutag_split_idx(self.ds, 1)
        c = gsat_mutag_split_idx(self.ds, 2)
        self.assertEqual(a, b)
        self.assertNotEqual(a['train'], c['train'])
        self.assertEqual(a['test'], c['test'])


class _StubModel(torch.nn.Module):
    """Returns fixed node attention = x[:, 0] (sigmoid-free) for metric checks."""
    def forward(self, x, edge_index, batch, n2m, edge_attr, epoch=1):
        att = x[:, 0].clone()
        return torch.zeros(int(batch.max()) + 1, 1), att, {'edge_att': None}


class TestGSATMetric(unittest.TestCase):
    def test_pooled_edge_auc_matches_lift_and_sklearn(self):
        gen = torch.Generator().manual_seed(1)
        gs = []
        for i in range(5):
            x = torch.rand(5, 3, generator=gen)
            ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
            el = torch.tensor([1., 1., 0., 0.]) if i % 2 == 0 else torch.zeros(4)
            nl = torch.tensor([1., 1., 1., 0., 0.]) if i % 2 == 0 else torch.zeros(5)
            gs.append(Data(x=x, edge_index=ei, edge_label=el, node_label=nl,
                           y=torch.tensor([[0.]])))
        out = gsat_explanation_metrics(_StubModel(), DataLoader(gs, batch_size=2), 'cpu')
        sc = np.concatenate([(g.x[:, 0][g.edge_index[0]] * g.x[:, 0][g.edge_index[1]]).numpy()
                             for g in gs])
        lb = np.concatenate([g.edge_label.numpy() for g in gs])
        self.assertAlmostEqual(out['gsat_edge_auc'], roc_auc_score(lb, sc), places=6)
        pg = [roc_auc_score(g.node_label.numpy(), g.x[:, 0].numpy()) for g in gs[::2]]
        self.assertAlmostEqual(out['node_gtroc_pergraph'], float(np.mean(pg)), places=6)
        self.assertEqual(out['n_gt_graphs'], 3)

    def test_single_class_is_nan_and_no_labels_is_nan(self):
        self.assertTrue(np.isnan(pooled_edge_auc(np.array([.1, .2]), np.array([0, 0]))))
        g = Data(x=torch.rand(3, 2), edge_index=torch.tensor([[0], [1]]), y=torch.tensor([[0.]]))
        out = gsat_explanation_metrics(_StubModel(), DataLoader([g]), 'cpu')
        self.assertTrue(np.isnan(out['gsat_edge_auc']))


class TestFixR(unittest.TestCase):
    def test_noanneal_preset_fixes_r_at_final(self):
        from config import MotifSATConfig
        import run as R
        cfg = MotifSATConfig.from_yaml(str(ROOT / 'MotifSAT/configs/base_gsat_noanneal.yaml'))
        self.assertTrue(cfg.fix_r)
        cfg.init_r, cfg.final_r, cfg.decay_r, cfg.decay_interval = 0.9, 0.5, 0.1, 10
        R._apply_fix_r(cfg)
        self.assertEqual((cfg.init_r, cfg.final_r, cfg.decay_r), (0.5, 0.5, 0.0))
        m = GSAT(x_dim=NUM_ATOM_TYPES, hidden_dim=H, num_layers=2, backbone_name='GIN',
                 num_classes=1, init_r=cfg.init_r, final_r=cfg.final_r,
                 decay_interval=cfg.decay_interval, decay_r=cfg.decay_r)
        rs = []
        for ep in range(60):
            m.anneal_r(ep); rs.append(round(float(m.r.item()), 6))
        self.assertEqual(set(rs), {0.5})
        self.assertEqual([T._gsat_ckpt_eligible(m, e) for e in (10, 11)], [False, True])

    def test_base_gsat_unaffected(self):
        from config import MotifSATConfig
        import run as R
        cfg = MotifSATConfig.from_yaml(str(ROOT / 'MotifSAT/configs/base_gsat.yaml'))
        self.assertFalse(cfg.fix_r)
        cfg.init_r, cfg.final_r, cfg.decay_r = 0.9, 0.5, 0.1
        R._apply_fix_r(cfg)
        self.assertEqual((cfg.init_r, cfg.decay_r), (0.9, 0.1))


class TestBuildModelWiring(unittest.TestCase):
    def test_build_model_passes_gsat_knobs(self):
        from config import MotifSATConfig
        import run as R
        cfg = MotifSATConfig(dataset='mutag', backbone='PNA', hidden_dim=H, num_layers=2,
                             arch_style='gsat', pna_aggregators='mean,min,max,std,sum',
                             pna_scalers=False, info_loss_coef=1.0, init_r=0.9,
                             final_r=0.5, decay_interval=5, decay_r=0.1)
        meta = types.SimpleNamespace(x_dim=NUM_ATOM_TYPES, deg=DEG, edge_attr_dim=0)
        m = R.build_model(cfg, 'BinaryClass', meta)
        self.assertIsInstance(m.clf, GSATArchGNN)
        self.assertEqual(len(m.clf.convs[0].aggregators), 5)
        self.assertEqual(len(m.clf.convs[0].scalers), 1)


if __name__ == '__main__':
    unittest.main()
