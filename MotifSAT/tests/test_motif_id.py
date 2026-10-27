"""Motif-identity control (2026-10-26): base GSAT whose predictor also sees each atom's motif-id
embedding ⊕ descriptors (motif_id_features=True)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from SharedModules.tests.pin_pkg_imports import MOTIFSAT_TOPLEVEL, pin_trainer_imports  # noqa: E402,F401

import torch  # noqa: E402
from torch_geometric.data import Batch, Data  # noqa: E402

from SharedModules.data.dataset import NUM_ATOM_TYPES, EDGE_FEAT_DIM  # noqa: E402
from model import GSAT  # noqa: E402
from motif_modules import MOTIF_DESC_DIM  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _batch(n2m):
    g = torch.Generator().manual_seed(0)
    ei = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]])
    d = Data(x=torch.eye(NUM_ATOM_TYPES)[[1, 1, 3, 2, 1]], edge_index=ei,
             edge_attr=torch.randn(ei.size(1), EDGE_FEAT_DIM, generator=g),
             nodes_to_motifs=torch.tensor(n2m), y=torch.tensor([[1.0]]))
    return Batch.from_data_list([d])


def _model(flag=True, **kw):
    torch.manual_seed(0)
    extra = dict(num_motifs=4, motif_desc_table=torch.randn(4, MOTIF_DESC_DIM)) if flag else {}
    return GSAT(x_dim=NUM_ATOM_TYPES, hidden_dim=16, num_layers=2, backbone_name='GIN', num_classes=1,
                motif_id_features=flag, **extra, **kw).eval()


class TestMotifIdFeatures(unittest.TestCase):
    def test_forward_and_identity_reaches_the_predictor(self):
        m = _model()
        b1, b2 = _batch([0, 0, 1, 1, 2]), _batch([0, 0, 3, 3, 2])     # same atoms, different motif ids
        o1, att, _ = m(b1.x, b1.edge_index, b1.batch, b1.nodes_to_motifs, b1.edge_attr)
        o2, _, _ = m(b2.x, b2.edge_index, b2.batch, b2.nodes_to_motifs, b2.edge_attr)
        self.assertEqual(tuple(att.shape), (5, 1))                    # gates stay per atom
        self.assertFalse(torch.allclose(o1, o2))                      # motif identity changes the prediction

    def test_node_weights_path_also_uses_identity(self):
        m = _model(); b = _batch([0, 0, 1, 1, 2])
        o, _, _ = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs, b.edge_attr, node_weights=torch.ones(5, 1))
        self.assertEqual(tuple(o.shape), (1, 1))

    def test_fails_loud(self):
        m = _model(); b = _batch([0, 0, -1, 1, 2])
        with self.assertRaises(ValueError):                           # UNK atom
            m(b.x, b.edge_index, b.batch, b.nodes_to_motifs, b.edge_attr)
        with self.assertRaises(ValueError):                           # no annotations
            m(b.x, b.edge_index, b.batch, None, b.edge_attr)
        with self.assertRaises(ValueError):                           # no descriptor table
            GSAT(x_dim=NUM_ATOM_TYPES, hidden_dim=16, num_layers=2, backbone_name='GIN', motif_id_features=True)
        with self.assertRaises(ValueError):                           # not a control for the motif graph
            _model(motif_method='motif_emb', noise='motif', info_loss_level='motif', motif_feat='multihot')

    def test_off_by_default_is_unchanged(self):
        torch.manual_seed(0)
        a = GSAT(x_dim=NUM_ATOM_TYPES, hidden_dim=16, num_layers=2, backbone_name='GIN')
        torch.manual_seed(0)
        c = GSAT(x_dim=NUM_ATOM_TYPES, hidden_dim=16, num_layers=2, backbone_name='GIN', motif_id_features=False)
        self.assertEqual(a.state_dict().keys(), c.state_dict().keys())
        self.assertFalse(any(k.startswith('motif_id') for k in c.state_dict()))

    def test_preset_loads(self):
        from config import MotifSATConfig
        c = MotifSATConfig.from_yaml(str(ROOT / 'MotifSAT/configs/base_gsat_motifid.yaml'))
        self.assertEqual((c.motif_method, c.noise, c.info_loss_level, c.motif_id_features), ('none', 'none', 'node', True))


if __name__ == '__main__':
    unittest.main()
