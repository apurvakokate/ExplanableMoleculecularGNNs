"""Mechanism ③ fragment graph: one node per motif occurrence under motif_grouping='copy'.

Regression (2026-10-25): fragment nodes were one per (molecule, motif TYPE), so bonded copies
of a motif lost their junction and separated copies merged into one hub node.

Run:  pytest MotifSAT/tests/test_motif_graph.py -v
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from SharedModules.tests.pin_pkg_imports import MOTIFSAT_TOPLEVEL, pin_trainer_imports  # noqa: E402

pin_trainer_imports(ROOT / 'MotifSAT', ROOT, MOTIFSAT_TOPLEVEL)

import torch  # noqa: E402
from torch_geometric.data import Batch, Data  # noqa: E402

from SharedModules.data.dataset import NUM_ATOM_TYPES, EDGE_FEAT_DIM  # noqa: E402
from model import GSAT, motif_emb_rows  # noqa: E402
from motif_modules import build_motif_graph, MOTIF_DESC_DIM  # noqa: E402


def _chain(n, extra=()):
    pairs = [(i, i + 1) for i in range(n - 1)] + list(extra)
    return torch.tensor(pairs + [(b, a) for a, b in pairs]).t()


def _junctions(n2m, cp, ei, grouping):
    inv, mb, voc = motif_emb_rows(n2m, cp, torch.zeros(n2m.numel(), dtype=torch.long), grouping)
    m_ei, _ = build_motif_graph(inv, ei, int(inv.max()) + 1, edge_mode='none')
    return int(inv.max()) + 1, m_ei.size(1), voc


class TestFragmentNodes(unittest.TestCase):
    def test_bonded_copies_keep_their_junction(self):
        # biphenyl-like: two 6-atom rings of motif 0, bonded by 5-6
        ei = _chain(12, extra=[(0, 5), (6, 11)])
        n2m = torch.zeros(12, dtype=torch.long)
        cp = torch.tensor([0] * 6 + [1] * 6)
        n, e, _ = _junctions(n2m, cp, ei, 'pooled')
        self.assertEqual((n, e), (1, 0))               # old: one node, junction lost
        n, e, voc = _junctions(n2m, cp, ei, 'copy')
        self.assertEqual((n, e), (2, 2))               # two nodes, one junction (both directions)
        self.assertEqual(voc.tolist(), [0, 0])         # both keep vocabulary id 0

    def test_separated_copies_are_not_a_hub(self):
        # chain A A B B A A: motif 0 copies {0,1} and {4,5}, motif 1 = {2,3}
        ei = _chain(6)
        n2m = torch.tensor([0, 0, 1, 1, 0, 0]); cp = torch.tensor([0, 0, 0, 0, 1, 1])
        n, e, _ = _junctions(n2m, cp, ei, 'pooled')
        self.assertEqual((n, e), (2, 2))               # old: A-B only (both copies merged)
        n, e, voc = _junctions(n2m, cp, ei, 'copy')
        self.assertEqual((n, e), (3, 4))               # A0-B and B-A1, no A-A edge
        self.assertEqual(sorted(voc.tolist()), [0, 0, 1])

    def test_copy_without_ids_raises(self):
        with self.assertRaises(ValueError):
            motif_emb_rows(torch.zeros(3, dtype=torch.long), None, None, 'copy')


class TestMotifEmbForwardCopy(unittest.TestCase):
    def _graph(self, i, g):
        ei = _chain(6, extra=[(0, 5)])
        return Data(x=torch.randn(6, NUM_ATOM_TYPES, generator=g), edge_index=ei,
                    edge_attr=torch.randn(ei.size(1), EDGE_FEAT_DIM, generator=g),
                    nodes_to_motifs=torch.tensor([0, 0, 1, 1, 0, 0]),
                    nodes_to_copy=torch.tensor([0, 0, 0, 0, 1, 1]), y=torch.tensor([[float(i % 2)]]))

    def _model(self, grouping, feat='multihot'):
        torch.manual_seed(0)
        kw = dict(num_motifs=2, motif_desc_table=torch.randn(2, MOTIF_DESC_DIM)) if feat == 'id_desc' else {}
        return GSAT(x_dim=NUM_ATOM_TYPES, hidden_dim=16, num_layers=2, backbone_name='GIN', num_classes=1,
                    motif_method='motif_emb', noise='motif', info_loss_level='motif', motif_feat=feat,
                    motif_edge_feat=True, motif_edge_chem=True, motif_gnn_norm='layer',
                    motif_gnn_residual=True, motif_grouping=grouping, **kw).eval()

    def test_rows_per_occurrence_and_constant_gate(self):
        g = torch.Generator().manual_seed(0)
        b = Batch.from_data_list([self._graph(i, g) for i in range(3)])
        for feat in ('multihot', 'id_desc'):
            for grouping, rows in (('pooled', 2), ('copy', 3)):
                _, na, aux = self._model(grouping, feat)(b.x, b.edge_index, b.batch, b.nodes_to_motifs,
                                                         b.edge_attr, nodes_to_copy=b.nodes_to_copy)
                self.assertEqual(aux['motif_logits'].numel(), rows * 3, (feat, grouping))
                att = na.view(-1)[:6]
                self.assertAlmostEqual(float(att[0]), float(att[1]), places=6)   # same copy, one gate
            # copy: the two copies of motif 0 get their own gates
            _, na, _ = self._model('copy', feat)(b.x, b.edge_index, b.batch, b.nodes_to_motifs,
                                                 b.edge_attr, nodes_to_copy=b.nodes_to_copy)
            self.assertNotAlmostEqual(float(na.view(-1)[0]), float(na.view(-1)[4]), places=6)

    def test_copy_forward_without_ids_raises(self):
        g = torch.Generator().manual_seed(1)
        b = Batch.from_data_list([self._graph(0, g)])
        with self.assertRaises(ValueError):
            self._model('copy')(b.x, b.edge_index, b.batch, b.nodes_to_motifs, b.edge_attr)

    def test_presets_load(self):
        from config import MotifSATConfig
        for f, (feat, ef, ec) in {'motif_emb_multihot_gin_res': ('multihot', False, False),
                                  'motif_emb_multihot_chem_res': ('multihot', True, True),
                                  'motif_emb_iddesc_gin_res': ('id_desc', False, False),
                                  'motif_emb_iddesc_chem_res': ('id_desc', True, True)}.items():
            c = MotifSATConfig.from_yaml(str(ROOT / f'MotifSAT/configs/{f}.yaml'))
            self.assertEqual((c.motif_method, c.motif_feat, c.motif_edge_feat, c.motif_edge_chem,
                              c.motif_gnn_norm, c.motif_gnn_residual, c.motif_gnn_layers),
                             ('motif_emb', feat, ef, ec, 'layer', True, 2), f)


if __name__ == '__main__':
    unittest.main()
