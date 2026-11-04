"""fixed_norm.py — fixed-norm reparameterisation for the weight-scaling check (default OFF everywhere).

Weight normalisation (Salimans & Kingma, 2016) writes each output unit's weight vector as
    w_i = g_i * v_i / ||v_i||.
Here g_i is FROZEN at the row norm the layer has at initialisation (a buffer, never seen by the optimiser), so
only the direction v trains and no layer can grow or shrink its norm. This tests whether a model's explanation
depends on the network rescaling its own weights to cancel the motif weights / gates.

`apply_fixed_norm(gnn)` constrains, on a BaseGNN predictor (call it AFTER the model is fully built, i.e. after
any `lin2` replacement):
  * every Linear weight in the node encoder (if any), the message-passing layers and the head (lin1, lin2);
  * GAT attention vectors (att_src / att_dst / att_edge), norm over the feature dimension;
  * BatchNorm / LayerNorm scales (gamma): frozen at their initial value (1); shifts stay trainable;
  * GIN's eps: frozen (it scales the self term).
Biases stay trainable (they shift, they do not scale). Nothing outside the predictor is touched (MOSE motif
parameters, GSAT/MotifSAT extractors).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

try:                                    # PyG's own Linear (used inside GCNConv / GATConv / SAGEConv / PNAConv)
    from torch_geometric.nn.dense.linear import Linear as _PygLinear
    _LINEAR_TYPES = (nn.Linear, _PygLinear)
except Exception:                       # pragma: no cover
    _LINEAR_TYPES = (nn.Linear,)

_ATT_NAMES = ('att_src', 'att_dst', 'att_edge')


class FixedRowNorm(nn.Module):
    """w = g * v / ||v|| per row (dim 0), g frozen at the initial row norms of the tensor it is built from."""

    def __init__(self, w0: torch.Tensor):
        super().__init__()
        self.register_buffer('g', w0.detach().flatten(1).norm(dim=1).clone())

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        n = v.flatten(1).norm(dim=1).clamp_min(1e-12)
        return v * (self.g / n).view(-1, *([1] * (v.dim() - 1)))


def _constrain(module: nn.Module, name: str) -> bool:
    p = getattr(module, name, None)
    if not isinstance(p, nn.Parameter) or isinstance(p, nn.parameter.UninitializedParameter) or p.dim() < 2:
        return False
    if parametrize.is_parametrized(module, name):
        return False
    parametrize.register_parametrization(module, name, FixedRowNorm(p))
    return True


def apply_fixed_norm(gnn: nn.Module) -> dict:
    """Constrain a BaseGNN predictor in place. Returns counts of what was constrained (for logging / tests)."""
    counts = {'linear': 0, 'attention': 0, 'norm_scale_frozen': 0, 'gin_eps_frozen': 0}
    parts = [getattr(gnn, 'node_encoder', None), getattr(gnn, 'convs', None),
             getattr(gnn, 'lin1', None), getattr(gnn, 'lin2', None),
             getattr(gnn, 'pna_batch_norms', None), getattr(gnn, 'layer_norms', None),
             getattr(gnn, 'node_encoder_norm', None)]
    for part in parts:
        if part is None:
            continue
        for mod in part.modules():
            if isinstance(mod, _LINEAR_TYPES):
                counts['linear'] += _constrain(mod, 'weight')
            for a in _ATT_NAMES:
                counts['attention'] += _constrain(mod, a)
            if isinstance(mod, (nn.BatchNorm1d, nn.LayerNorm)) and getattr(mod, 'weight', None) is not None:
                mod.weight.requires_grad_(False)
                counts['norm_scale_frozen'] += 1
            eps = getattr(mod, 'eps', None)
            if isinstance(eps, nn.Parameter):          # GINConv(train_eps=True)
                eps.requires_grad_(False)
                counts['gin_eps_frozen'] += 1
    gnn.fixed_norm = True
    return counts


def fixed_norm_report(gnn: nn.Module) -> float:
    """Largest |row norm - frozen g| over all constrained tensors (should be ~0 at any point of training)."""
    worst = 0.0
    for mod in gnn.modules():
        if not parametrize.is_parametrized(mod):
            continue
        for name, plist in mod.parametrizations.items():
            for f in plist:
                if isinstance(f, FixedRowNorm):
                    w = getattr(mod, name)
                    worst = max(worst, float((w.detach().flatten(1).norm(dim=1) - f.g).abs().max()))
    return worst
