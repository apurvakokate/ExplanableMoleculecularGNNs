"""gsat_arch.py — the GIN and PNA backbones of the official GSAT code (arch_style='gsat').

Re-implements Graph-COM/GSAT @ 6812f5c ``src/models/{gin,pna,conv_layers}.py`` so a
MotifSAT ``base_gsat`` run can reproduce the paper's architecture exactly, while
keeping the BaseGNN interface (``encode`` / ``convolve`` / ``get_embedding`` /
``classify``) that MotifSAT's two-pass attention injection relies on.

Differences from the project default (``BaseGNN``, arch_style='ours'):

  GIN  node encoder   Linear(x_dim, H) (AtomEncoder for OGB)   [ours: identity / LN]
       conv MLP       Linear -> BatchNorm -> ReLU -> Linear, eps=0 fixed
                                                   [ours: Lin-ReLU-Lin-ReLU-BN, train_eps]
       per layer      conv -> ReLU -> dropout      [ours: conv -> ReLU, no dropout]
       readout/head   sum pool -> Linear(H, out)   [ours: 2-layer MLP head]
  PNA  message        cat[x_i, x_j] * edge_atten   [ours: x_j * edge_atten]
       aggregators    configurable (paper: mean,min,max,std[,sum])  [ours: sum,mean,max,std]
       scalers        identity only when scalers=False (paper)      [ours: always 3]
       per layer      h = ReLU(BN(conv(x))); x = h + x; dropout     [ours: conv -> ReLU]
       readout/head   mean pool -> Linear(H,H/2)-ReLU-Linear(H/2,H/4)-ReLU-Linear(H/4,out)

Attention injection is unchanged: w_message passes edge_atten = att[src]*att[dst]
(BaseGNN.get_embedding), exactly GSAT's lift_node_att_to_edge_att.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import BatchNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.utils import degree

from .gnn_base import BaseGNN

try:                                   # same op the official code uses
    from torch_scatter import scatter as _scatter
except ImportError:                    # PyG >= 2.3 ships an equivalent
    from torch_geometric.utils import scatter as _scatter


# ── aggregators / scalers (verbatim semantics of GSAT conv_layers.py) ─────────

def _agg_sum(src, index, dim_size):
    return _scatter(src, index, dim=0, dim_size=dim_size, reduce='sum')


def _agg_mean(src, index, dim_size):
    return _scatter(src, index, dim=0, dim_size=dim_size, reduce='mean')


def _agg_min(src, index, dim_size):
    return _scatter(src, index, dim=0, dim_size=dim_size, reduce='min')


def _agg_max(src, index, dim_size):
    return _scatter(src, index, dim=0, dim_size=dim_size, reduce='max')


def _agg_var(src, index, dim_size):
    mean = _agg_mean(src, index, dim_size)
    mean_squares = _agg_mean(src * src, index, dim_size)
    return mean_squares - mean * mean


def _agg_std(src, index, dim_size):
    return torch.sqrt(torch.relu(_agg_var(src, index, dim_size)) + 1e-5)


AGGREGATORS = {'sum': _agg_sum, 'mean': _agg_mean, 'min': _agg_min,
               'max': _agg_max, 'var': _agg_var, 'std': _agg_std}


def _scale_identity(src, deg, avg_deg):
    return src


def _scale_amplification(src, deg, avg_deg):
    return src * (torch.log(deg + 1) / avg_deg['log'])


def _scale_attenuation(src, deg, avg_deg):
    scale = avg_deg['log'] / torch.log(deg + 1)
    scale[deg == 0] = 1
    return src * scale


SCALERS = {'identity': _scale_identity, 'amplification': _scale_amplification,
           'attenuation': _scale_attenuation}

#: PNA aggregators of the official configs (PNA-mutag adds 'sum').
PAPER_PNA_AGGREGATORS = ('mean', 'min', 'max', 'std')


# ── conv layers ───────────────────────────────────────────────────────────────

class GSATGINConv(MessagePassing):
    """GSAT GINConv: sum of x_j * edge_atten, plus ungated self term (1+eps)·x; eps=0
    fixed (PyG GINConv default train_eps=False). Written on MessagePassing directly:
    newer PyG generates GINConv.propagate from its own message signature, which
    would drop the edge_atten argument in a subclass."""

    def __init__(self, nn_: nn.Module, eps: float = 0.0):
        super().__init__(aggr='add')
        self.nn = nn_
        self.register_buffer('eps', torch.tensor([eps]))
        reset(self.nn)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr=None,
                edge_atten: Optional[Tensor] = None, node_self_gate=None) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_atten=edge_atten, size=None)
        out = out + (1 + self.eps) * x
        return self.nn(out)

    def message(self, x_j: Tensor, edge_atten: Optional[Tensor] = None) -> Tensor:
        return x_j * edge_atten if edge_atten is not None else x_j


def gsat_gin_mlp(in_dim: int, out_dim: int) -> Sequential:
    """GSAT GIN.MLP: Linear -> BatchNorm1d -> ReLU -> Linear."""
    return Sequential(Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim),
                      ReLU(inplace=True), Linear(out_dim, out_dim))


class GSATPNAConv(MessagePassing):
    """GSAT PNAConvSimple: message = cat[x_i, x_j] (·edge_atten), configurable
    aggregators x scalers, post-aggregation Linear(len(agg)*len(scal)*in -> out)."""

    def __init__(self, in_channels: int, out_channels: int,
                 aggregators: Sequence[str], scalers: Sequence[str],
                 deg: Tensor, post_layers: int = 1):
        super().__init__(aggr=None, node_dim=0)
        unknown = [a for a in aggregators if a not in AGGREGATORS]
        unknown += [s for s in scalers if s not in SCALERS]
        if unknown:
            raise ValueError(f"GSATPNAConv: unknown aggregator/scaler {unknown}")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.aggregators = [AGGREGATORS[a] for a in aggregators]
        self.scalers = [SCALERS[s] for s in scalers]
        deg = deg.to(torch.float)
        self.avg_deg: Dict[str, float] = {
            'lin': deg.mean().item(),
            'log': (deg + 1).log().mean().item(),
        }
        modules: List[nn.Module] = [
            Linear(len(aggregators) * len(scalers) * in_channels, out_channels)]
        for _ in range(post_layers - 1):
            modules += [ReLU(), Linear(out_channels, out_channels)]
        self.post_nn = Sequential(*modules)
        reset(self.post_nn)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr=None,
                edge_atten: Optional[Tensor] = None, node_self_gate=None) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_atten=edge_atten, size=None)
        return self.post_nn(out)

    def message(self, x_i: Tensor, x_j: Tensor,
                edge_atten: Optional[Tensor] = None) -> Tensor:
        m = torch.cat([x_i, x_j], dim=-1)
        return m * edge_atten if edge_atten is not None else m

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:
        out = torch.cat([a(inputs, index, dim_size) for a in self.aggregators], dim=-1)
        deg = degree(index, dim_size, dtype=inputs.dtype).view(-1, 1)
        return torch.cat([s(out, deg, self.avg_deg) for s in self.scalers], dim=-1)


# ── backbone ──────────────────────────────────────────────────────────────────

class GSATArchGNN(BaseGNN):
    """BaseGNN-compatible backbone with the official GSAT GIN / PNA architecture.

    BaseGNN.__init__ is deliberately NOT called (it would build the project's own
    conv stack); the attributes BaseGNN.get_embedding reads are set here.
    """

    def __init__(self, x_dim: int, hidden_dim: int, num_layers: int,
                 backbone: str, num_out: int, node_encoder: str = 'onehot',
                 dropout: float = 0.3, deg: Optional[Tensor] = None,
                 pna_aggregators: Optional[Sequence[str]] = None,
                 pna_scalers: bool = False):
        nn.Module.__init__(self)
        self.backbone = backbone.upper()
        if self.backbone not in ('GIN', 'PNA'):
            raise ValueError(f"arch_style='gsat' exists for GIN and PNA only "
                             f"(the official GSAT backbones), got {backbone!r}")
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.self_gate = False                       # read by get_embedding
        self.node_encoder_type = node_encoder
        self.graph_pool = 'add' if self.backbone == 'GIN' else 'mean'

        if node_encoder == 'atom_encoder':
            from ogb.graphproppred.mol_encoder import AtomEncoder
            self.node_encoder = AtomEncoder(emb_dim=hidden_dim)
        else:
            self.node_encoder = Linear(x_dim, hidden_dim)
        self.node_encoder_norm = nn.Identity()

        self.convs = nn.ModuleList()
        self.batch_norms = None
        if self.backbone == 'GIN':
            for _ in range(num_layers):
                self.convs.append(GSATGINConv(gsat_gin_mlp(hidden_dim, hidden_dim)))
            self.fc_out = Linear(hidden_dim, num_out)
        else:
            if deg is None:
                raise ValueError("arch_style='gsat' PNA needs the training degree histogram")
            aggs = list(pna_aggregators or PAPER_PNA_AGGREGATORS)
            scal = (['identity', 'amplification', 'attenuation'] if pna_scalers
                    else ['identity'])
            self.batch_norms = nn.ModuleList()
            for _ in range(num_layers):
                self.convs.append(GSATPNAConv(2 * hidden_dim, hidden_dim, aggs, scal,
                                              deg=deg, post_layers=1))
                self.batch_norms.append(BatchNorm(hidden_dim))
            self.fc_out = Sequential(
                Linear(hidden_dim, hidden_dim // 2), ReLU(),
                Linear(hidden_dim // 2, hidden_dim // 4), ReLU(),
                Linear(hidden_dim // 4, num_out))

    def convolve(self, x: Tensor, edge_index: Tensor, edge_attr=None,
                 edge_atten: Optional[Tensor] = None,
                 node_self_gate: Optional[Tensor] = None) -> Tensor:
        for i, conv in enumerate(self.convs):
            if self.backbone == 'GIN':
                x = F.relu(conv(x, edge_index, edge_atten=edge_atten))
            else:
                h = F.relu(self.batch_norms[i](conv(x, edge_index, edge_atten=edge_atten)))
                x = h + x
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def classify(self, graph_emb: Tensor) -> Tensor:
        return self.fc_out(graph_emb)
