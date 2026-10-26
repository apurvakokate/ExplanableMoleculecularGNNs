"""KL-level control (2026-10-26): motif gates with the KL at motif level vs node level.

info_loss_level='motif' : KL averaged over motif rows (one term per sampled motif gate).
info_loss_level='node'  : KL averaged over ATOMS of the broadcast motif gates, i.e. each motif's KL
                          weighted by its atom count — sum_m n_m KL(a_m) / sum_m n_m.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from SharedModules.tests.pin_pkg_imports import MOTIFSAT_TOPLEVEL, pin_trainer_imports  # noqa: E402,F401

import torch  # noqa: E402
from torch_geometric.data import Batch, Data  # noqa: E402

from SharedModules.data.dataset import NUM_ATOM_TYPES, EDGE_FEAT_DIM  # noqa: E402
from model import GSAT  # noqa: E402
from losses import info_loss  # noqa: E402


def _graph(g):
    # 7 atoms: motif 0 = 4 atoms, motif 1 = 2 atoms, motif 2 = 1 atom
    ei = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5]])
    return Data(x=torch.randn(7, NUM_ATOM_TYPES, generator=g), edge_index=ei,
                edge_attr=torch.randn(ei.size(1), EDGE_FEAT_DIM, generator=g),
                nodes_to_motifs=torch.tensor([0, 0, 0, 0, 1, 1, 2]), y=torch.tensor([[1.0]]))


class TestKLLevel(unittest.TestCase):
    def _run(self, level, method='readout', **kw):
        torch.manual_seed(0)
        m = GSAT(x_dim=NUM_ATOM_TYPES, hidden_dim=16, num_layers=2, backbone_name='GIN', num_classes=1,
                 motif_method=method, noise='motif', info_loss_level=level, motif_scorer_norm='layer', **kw)
        m.train()
        b = Batch.from_data_list([_graph(torch.Generator().manual_seed(1))])
        logits, node_att, aux = m(b.x, b.edge_index, b.batch, b.nodes_to_motifs, b.edge_attr)
        _, br = m.compute_loss(torch.tensor(0.0), aux, b.nodes_to_motifs, b.batch)
        return br['info_loss'], aux, float(m.r.item())

    def test_motif_level_is_mean_over_motif_rows(self):
        ib, aux, r = self._run('motif')
        self.assertAlmostEqual(ib, float(info_loss(aux['motif_att'].view(-1), r)), places=6)
        self.assertEqual(aux['motif_att'].numel(), 3)

    def test_node_level_is_atom_weighted_motif_kl(self):
        ib, aux, r = self._run('node')
        ma = aux['motif_att'].view(-1)
        per = torch.stack([info_loss(ma[i:i + 1], r) for i in range(3)])
        sizes = torch.tensor([4.0, 2.0, 1.0])
        self.assertAlmostEqual(ib, float((per * sizes).sum() / sizes.sum()), places=5)
        self.assertAlmostEqual(ib, float(info_loss(aux['node_att'].view(-1), r)), places=6)

    def test_presets_load(self):
        from config import MotifSATConfig
        root = Path(__file__).resolve().parents[1] / 'configs'
        for name, method in (('readout_layernorm_nodekl', 'readout'), ('readout_nonorm_nodekl', 'readout'),
                             ('motif_emb_iddesc_gin_res_nodekl', 'motif_emb')):
            c = MotifSATConfig.from_yaml(str(root / f'{name}.yaml'))
            self.assertEqual((c.motif_method, c.noise, c.info_loss_level), (method, 'motif', 'node'), name)


if __name__ == '__main__':
    unittest.main()
