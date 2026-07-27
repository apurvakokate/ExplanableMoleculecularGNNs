"""Rule Set 1 — Stage 3 (Train) unit tests for the loss assembly.

Targets the real failure mode "losses not correctly executed", focused on the
motif-consistency loss gating (the trap: within/between coefs do nothing unless
motif_loss_coef > 0). The constructor GUARDS the inert config (raises), and when
active the term must move the total and flow gradients.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from SharedModules.tests.pin_pkg_imports import MOTIFSAT_TOPLEVEL, pin_trainer_imports  # noqa: E402
pin_trainer_imports(ROOT / 'MotifSAT', ROOT, MOTIFSAT_TOPLEVEL)

import torch  # noqa: E402
from model import GSAT  # noqa: E402


def _mk(**kw):
    return GSAT(x_dim=8, hidden_dim=8, num_layers=1, info_loss_coef=0.0, **kw)


def _aux(node_att):
    return {'node_att': node_att, 'edge_att': None, 'motif_att': None,
            'motif_logits': None, 'inv_idx': None, 'r': torch.tensor(0.5)}


class Stage3MotifLossGating(unittest.TestCase):
    def test_ctor_rejects_inert_within(self):
        # within/between > 0 with motif_loss_coef == 0 is inert → constructor must refuse
        with self.assertRaises(ValueError):
            _mk(within_node_coef=0.5)          # motif_loss_coef defaults 0
        with self.assertRaises(ValueError):
            _mk(between_motif_coef=0.5)

    def test_active_term_moves_total(self):
        n2m = torch.tensor([0, 0, 1, 1])       # 2 motifs, 2 nodes each
        batch = torch.tensor([0, 0, 0, 0])
        att = torch.tensor([[0.9], [0.1], [0.5], [0.5]])   # motif 0 has within-variance > 0
        task = torch.tensor(1.0)

        active = _mk(motif_loss_coef=1.0, within_node_coef=1.0, between_motif_coef=0.0)
        tot_a, bd = active.compute_loss(task, _aux(att), n2m, batch)
        self.assertGreater(float(tot_a), 1.0, 'active consistency loss did not move the total')
        self.assertGreater(bd.get('within_var', 0.0), 0.0)

        off = _mk(motif_loss_coef=0.0, within_node_coef=0.0, between_motif_coef=0.0)
        tot_o, _ = off.compute_loss(task, _aux(att), n2m, batch)
        self.assertAlmostEqual(float(tot_o), 1.0, places=6,
                               msg='with all coefs 0 the total must equal the task loss')

    def test_gradient_flows_through_consistency(self):
        n2m = torch.tensor([0, 0, 1, 1]); batch = torch.tensor([0, 0, 0, 0])
        att = torch.tensor([[0.9], [0.1], [0.5], [0.5]], requires_grad=True)
        m = _mk(motif_loss_coef=1.0, within_node_coef=1.0)
        total, _ = m.compute_loss(torch.tensor(1.0), _aux(att), n2m, batch)
        total.backward()
        self.assertIsNotNone(att.grad)
        self.assertGreater(float(att.grad.abs().sum()), 0.0,
                           'consistency loss is not differentiable w.r.t. node attention')


if __name__ == '__main__':
    unittest.main(verbosity=2)
