#!/usr/bin/env python3
"""test_ckpt_rule.py — ckpt_rule='gsat' must reproduce the official GSAT checkpoint rule.

Upstream (Graph-COM/GSAT src/run_gsat.py, GSAT.train):
    r = self.fix_r if self.fix_r else self.get_r(...)
    if (r == self.final_r or self.fix_r) and epoch > 10 and (valid > best or
            (valid == best and valid_loss < best_loss)): <checkpoint>

1. Eligibility is checked against upstream's OWN get_r source (extracted with ast from
   $GSAT_UPSTREAM_SRC/run_gsat.py), across schedules and epochs 0..200.
2. A short synthetic training run checks that the selected epoch is eligible, that
   ckpt_rule='any' is unchanged, and that a bad ckpt_rule is rejected.

Run:  GSAT_UPSTREAM_SRC=<GSAT clone>/src pytest MotifSAT/tests/test_ckpt_rule.py -v
"""
import ast
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from SharedModules.tests.pin_pkg_imports import MOTIFSAT_TOPLEVEL, pin_trainer_imports  # noqa: E402

pin_trainer_imports(ROOT / 'MotifSAT', ROOT, MOTIFSAT_TOPLEVEL)

import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from SharedModules.data.dataset import NUM_ATOM_TYPES, EDGE_FEAT_DIM  # noqa: E402
from model import GSAT  # noqa: E402
import train as T  # noqa: E402

UP = os.environ.get('GSAT_UPSTREAM_SRC')


def _upstream_get_r():
    """Exec upstream GSAT.get_r verbatim (method -> plain function)."""
    src = (Path(UP) / 'run_gsat.py').read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == 'get_r':
            ns = {}
            exec(compile(ast.Module([node], []), 'upstream_get_r', 'exec'), ns)
            return lambda di, dr, ep, init_r, final_r: ns['get_r'](None, di, dr, ep, init_r, final_r)
    raise RuntimeError('get_r not found in upstream run_gsat.py')


class _Sched:
    def __init__(self, init_r, final_r, di, dr):
        self.init_r, self.final_r, self.decay_interval, self.decay_r = init_r, final_r, di, dr


@unittest.skipUnless(UP and (Path(UP) / 'run_gsat.py').exists(), 'GSAT_UPSTREAM_SRC not set')
class TestEligibilityMatchesUpstream(unittest.TestCase):
    def test_all_schedules(self):
        get_r = _upstream_get_r()
        n = 0
        for final_r in (0.5, 0.6, 0.7, 0.8, 0.9):
            for di in (5, 10, 20, 30, 60):
                m = _Sched(0.9, final_r, di, 0.1)
                for ep in range(201):
                    r = get_r(di, 0.1, ep, 0.9, final_r)
                    up = (r == final_r) and ep > 10
                    self.assertEqual(T._gsat_ckpt_eligible(m, ep), up,
                                     f'final_r={final_r} di={di} ep={ep} r={r!r}')
                    n += 1
        self.assertEqual(n, 5 * 5 * 201)

    def test_fixed_r_is_upstream_fix_r(self):
        # no decay == upstream fix_r branch: eligible iff epoch > 10
        for m in (_Sched(0.9, 0.5, 10, 0.0), _Sched(0.9, 0.5, None, None)):
            self.assertEqual([T._gsat_ckpt_eligible(m, e) for e in range(13)],
                             [False] * 11 + [True, True])


def _loaders(n=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        x = torch.randn(6, NUM_ATOM_TYPES, generator=g)
        x[:, 0] += 2.0 * (i % 2)  # learnable signal
        ei = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
        ei = torch.cat([ei, ei.flip(0)], 1)
        out.append(Data(x=x, edge_index=ei, edge_attr=torch.randn(12, EDGE_FEAT_DIM, generator=g),
                        nodes_to_motifs=torch.tensor([0, 0, 1, 1, 2, 2]),
                        y=torch.tensor([[float(i % 2)]])))
    return {'train': DataLoader(out[:32], batch_size=8, shuffle=True),
            'valid': DataLoader(out[32:], batch_size=8)}


def _fit(rule, epochs=45, final_r=0.7):
    torch.manual_seed(0)
    m = GSAT(x_dim=NUM_ATOM_TYPES, hidden_dim=16, num_layers=2, backbone_name='GIN',
             num_classes=1, init_r=0.9, final_r=final_r, decay_interval=10, decay_r=0.1)
    return T.train_gsat(m, _loaders(), 'BinaryClass', torch.device('cpu'), epochs=epochs,
                        patience=1000, min_epochs=1, verbose=False, ckpt_rule=rule)


class TestTrainLoop(unittest.TestCase):
    def test_gsat_rule_selects_only_eligible(self):
        m, h = _fit('gsat')
        self.assertEqual(len(h['ckpt_eligible']), 45)
        # final_r 0.7, decay 0.1 / 10 epochs: r reaches 0.7 at 0-indexed epoch 20.
        self.assertEqual(h['ckpt_eligible'].index(True), 20)
        self.assertIsNotNone(h['best_epoch'])
        self.assertTrue(h['ckpt_eligible'][h['best_epoch'] - 1])
        self.assertTrue(T._gsat_ckpt_eligible(m, h['best_epoch'] - 1))

    def test_fallback_when_never_eligible(self):
        _, h = _fit('gsat', epochs=12)
        self.assertIsNone(h['best_epoch'])
        self.assertEqual(h.get('ckpt_fallback'), 'last_epoch')

    def test_any_rule_unchanged(self):
        _, h = _fit('any')
        self.assertNotIn('ckpt_eligible', h)
        self.assertIsNotNone(h['best_epoch'])
        self.assertEqual(len(h['r']), 45)

    def test_bad_rule_rejected(self):
        with self.assertRaises(ValueError):
            _fit('best', epochs=1)


if __name__ == '__main__':
    unittest.main()
