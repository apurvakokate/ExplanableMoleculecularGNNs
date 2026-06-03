"""conv_layers.py — Custom GNN message-passing layers with edge_atten support.

All layers accept an ``edge_atten`` argument: a float tensor [E, 1] that
scales messages during propagation (the w_message injection point).
When ``edge_atten=None`` the layer behaves like its standard counterpart.

Edge features (``edge_attr``) are globally disabled in this project
(``edge_attr = None`` is enforced in ``BaseGNN.get_embedding()`` and
``MultiChannelGNN.forward()``).  The ``edge_attr`` parameter is accepted
for API compatibility but ignored.

Reference: GSAT (Miao et al., 2022) custom conv implementations.

Classes
-------
GINConv          — GIN with edge_atten message scaling
GCNConvWithAtten — Custom GCN (no self-loops) with edge_atten × norm scaling
GATConvWithAtten — Custom GAT with edge_atten applied to softmax alpha
SAGEConvWithAtten— Custom GraphSAGE with edge_atten message scaling
PNAConvSimple    — Custom PNA with 4 aggregators × 3 scalers, edge_atten
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINConv as BaseGINConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptTensor, Size
from torch_geometric.utils import degree, softmax

try:
    from torch_scatter import scatter
    def _scatter(src, index, dim, dim_size, reduce):
        return scatter(src, index, dim, None, dim_size, reduce=reduce)
except ImportError:
    from torch_geometric.utils import scatter as _tg_scatter
    def _scatter(src, index, dim, dim_size, reduce):
        return _tg_scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)


# ─────────────────────────────────────────────────────────────────────────────
# GINConv
# ─────────────────────────────────────────────────────────────────────────────

class GINConv(BaseGINConv):
    """GIN with optional edge-attention scaling in the message step."""

    def forward(self, x, edge_index, edge_attr=None,
                edge_atten: OptTensor = None) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)
        out = self.propagate(edge_index, x=x, edge_atten=edge_atten)
        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r
        return self.nn(out)

    def message(self, x_j: Tensor, edge_atten: OptTensor = None) -> Tensor:
        if edge_atten is not None:
            return x_j * edge_atten
        return x_j


# ─────────────────────────────────────────────────────────────────────────────
# GCNConvWithAtten
# ─────────────────────────────────────────────────────────────────────────────

class GCNConvWithAtten(MessagePassing):
    """Custom GCN with edge_atten support and no self-loops.

    Self-loops are omitted so edge_atten (which covers the original edges
    only) never has a size mismatch with ``edge_index``.

    Normalization is D^{-1/2} A D^{-1/2} (normalize=True, default) or
    raw adjacency (normalize=False). Without normalization, the model still
    distinguishes node degrees even with constant input features.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True,
                 normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.normalize    = normalize

        self.lin = Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None,
                edge_atten: OptTensor = None) -> Tensor:
        x = self.lin(x)

        row, col = edge_index
        if self.normalize:
            deg     = degree(col, x.size(0), dtype=x.dtype)
            deg_inv = deg.pow(-0.5)
            deg_inv[deg_inv == float('inf')] = 0.0
            norm = deg_inv[row] * deg_inv[col]
            edge_weight = norm * edge_atten.squeeze(-1) if edge_atten is not None else norm
        else:
            if edge_atten is not None:
                edge_weight = edge_atten.squeeze(-1)
            else:
                edge_weight = torch.ones(edge_index.size(1),
                                         dtype=x.dtype, device=x.device)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return x_j * edge_weight.view(-1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# GATConvWithAtten
# ─────────────────────────────────────────────────────────────────────────────

class GATConvWithAtten(MessagePassing):
    """Custom multi-head GAT with edge_atten applied to softmax alpha.

    edge_atten is multiplied onto the (already-softmaxed) per-head
    attention coefficients before the weighted message sum:

        alpha_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
        msg_ij   = alpha_ij * edge_atten_ij * Wh_j

    This means the motif bottleneck (edge_atten from GSAT) gates each
    neighbour's contribution on top of the learned graph-structure attention.

    Parameters
    ----------
    concat : bool
        True  → output [N, heads * out_channels]
        False → output [N, out_channels]  (mean over heads; default in ref)
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 4,
                 concat: bool = False, negative_slope: float = 0.2,
                 dropout: float = 0.0, add_self_loops: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels     = in_channels
        self.out_channels    = out_channels
        self.heads           = heads
        self.concat          = concat
        self.negative_slope  = negative_slope
        self.dropout         = dropout
        self.add_self_loops  = add_self_loops

        self.lin = Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = torch.nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = torch.nn.Parameter(torch.empty(1, heads, out_channels))
        out_bias = heads * out_channels if concat else out_channels
        self.bias = torch.nn.Parameter(torch.zeros(out_bias))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.att_src)
        torch.nn.init.xavier_uniform_(self.att_dst)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None,
                edge_atten: OptTensor = None) -> Tensor:
        H, C = self.heads, self.out_channels

        # Linear transform: [N, in] → [N, H, C]
        x = self.lin(x).view(-1, H, C)

        alpha_src = (x * self.att_src).sum(dim=-1)   # [N, H]
        alpha_dst = (x * self.att_dst).sum(dim=-1)   # [N, H]

        out = self.propagate(edge_index, x=x,
                             alpha=(alpha_src, alpha_dst),
                             edge_atten=edge_atten)

        out = out.view(-1, H * C) if self.concat else out.mean(dim=1)
        return out + self.bias

    def message(self, x_j: Tensor,
                alpha_j: Tensor, alpha_i: Tensor,
                edge_atten: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        """Compute alpha-weighted message, then scale by edge_atten.

        alpha_j, alpha_i : [E, H]  — attention scores for source / dest
        x_j              : [E, H, C] — transformed source features
        edge_atten       : [E, 1]  or None
        """
        alpha = alpha_j + alpha_i                        # [E, H]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)       # [E, H]
        alpha = F.dropout(alpha, p=self.dropout,
                          training=self.training)
        if edge_atten is not None:
            alpha = alpha * edge_atten                   # [E, H] × [E, 1]
        return x_j * alpha.unsqueeze(-1)                 # [E, H, C]


# ─────────────────────────────────────────────────────────────────────────────
# SAGEConvWithAtten
# ─────────────────────────────────────────────────────────────────────────────

class SAGEConvWithAtten(MessagePassing):
    """Custom GraphSAGE with edge_atten message scaling.

    x_i = W1 * mean_j(edge_atten_ij * x_j)  +  W2 * x_i
    """

    def __init__(self, in_channels: int, out_channels: int,
                 normalize: bool = False, bias: bool = True,
                 aggr: str = 'mean', **kwargs):
        kwargs['aggr'] = aggr
        super().__init__(**kwargs)

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.normalize    = normalize

        self.lin_l = Linear(in_channels, out_channels, bias=bias)
        self.lin_r = Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index: Adj,
                edge_attr: OptTensor = None,
                edge_atten: OptTensor = None) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)
        out = self.propagate(edge_index, x=x, edge_atten=edge_atten)
        out = self.lin_l(out)
        x_r = x[1]
        if x_r is not None:
            out = out + self.lin_r(x_r)
        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)
        return out

    def message(self, x_j: Tensor, edge_atten: OptTensor = None) -> Tensor:
        if edge_atten is not None:
            return x_j * edge_atten
        return x_j


# ─────────────────────────────────────────────────────────────────────────────
# PNAConvSimple
# ─────────────────────────────────────────────────────────────────────────────

class PNAConvSimple(MessagePassing):
    """Custom PNA with 4 aggregators (sum, mean, max, std) ×
    3 scalers (identity, amplification, attenuation).

    Since edge_attr is disabled globally, messages are plain ``x_j``
    scaled by ``edge_atten`` when provided.  The post-aggregation MLP maps:

        [12 × F_in] → F_out

    Parameters
    ----------
    deg : Tensor [max_degree + 1]
        In-degree histogram of the TRAINING set.  Required — computed by
        ``loader.compute_deg_histogram()``.
    post_layers : int
        Number of linear layers in the post-aggregation MLP (default 1).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 deg: Tensor, post_layers: int = 1, **kwargs):
        super().__init__(aggr=None, node_dim=0, **kwargs)

        self.in_channels  = in_channels
        self.out_channels = out_channels

        self._aggregators = [
            _agg_sum, _agg_mean, _agg_max, _agg_std,
        ]
        self._scalers = [
            _scale_identity, _scale_amplification, _scale_attenuation,
        ]

        deg = deg.float()
        self.avg_deg: Dict[str, float] = {
            'lin': deg.mean().item(),
            'log': (deg + 1).log().mean().item(),
            'exp': deg.exp().mean().item(),
        }

        # Post-aggregation MLP: 4 agg × 3 scalers × F_in → F_out
        post_in = len(self._aggregators) * len(self._scalers) * in_channels
        layers: list = [Linear(post_in, out_channels)]
        for _ in range(post_layers - 1):
            layers += [ReLU(), Linear(out_channels, out_channels)]
        self.post_nn = Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.post_nn)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None,
                edge_atten: OptTensor = None) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_atten=edge_atten)
        return self.post_nn(out)

    def message(self, x_j: Tensor, edge_atten: OptTensor = None) -> Tensor:
        """Plain neighbour feature, scaled by edge_atten if provided."""
        if edge_atten is not None:
            return x_j * edge_atten
        return x_j

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:
        outs = [agg(inputs, index, dim_size) for agg in self._aggregators]
        out  = torch.cat(outs, dim=-1)                      # [N, 4*F_in]
        deg  = degree(index, dim_size, dtype=inputs.dtype).view(-1, 1)
        outs = [scaler(out, deg, self.avg_deg) for scaler in self._scalers]
        return torch.cat(outs, dim=-1)                      # [N, 12*F_in]


# ── PNA aggregators ───────────────────────────────────────────────────────────

def _agg_sum(src, index, dim_size):
    return _scatter(src, index, 0, dim_size, 'sum')

def _agg_mean(src, index, dim_size):
    return _scatter(src, index, 0, dim_size, 'mean')

def _agg_max(src, index, dim_size):
    return _scatter(src, index, 0, dim_size, 'max')

def _agg_var(src, index, dim_size):
    mean  = _agg_mean(src, index, dim_size)
    mean2 = _agg_mean(src * src, index, dim_size)
    return mean2 - mean * mean

def _agg_std(src, index, dim_size):
    return torch.sqrt(torch.relu(_agg_var(src, index, dim_size)) + 1e-5)


# ── PNA scalers ───────────────────────────────────────────────────────────────

def _scale_identity(src, deg, avg_deg):
    return src

def _scale_amplification(src, deg, avg_deg):
    return src * (torch.log(deg + 1) / avg_deg['log'])

def _scale_attenuation(src, deg, avg_deg):
    scale = avg_deg['log'] / torch.log(deg + 1)
    scale[deg == 0] = 1.0
    return src * scale


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def make_gin_conv(in_dim: int, out_dim: int, inner_bn: bool = True) -> GINConv:
    # Reference GIN MLP: Linear -> ReLU -> Linear -> ReLU -> BatchNorm.
    # inner_bn=True matches the original DomainDrivenGlobalExpl design (Xu et al.
    # GIN); inner_bn=False is the leaner Linear -> ReLU -> Linear.
    if inner_bn:
        from torch.nn import BatchNorm1d
        nn = Sequential(Linear(in_dim, out_dim), ReLU(),
                        Linear(out_dim, out_dim), ReLU(),
                        BatchNorm1d(out_dim))
    else:
        nn = Sequential(Linear(in_dim, out_dim), ReLU(), Linear(out_dim, out_dim))
    return GINConv(nn, train_eps=True)

def make_gcn_conv(in_dim: int, out_dim: int) -> GCNConvWithAtten:
    return GCNConvWithAtten(in_dim, out_dim, normalize=True)

def make_sage_conv(in_dim: int, out_dim: int) -> SAGEConvWithAtten:
    return SAGEConvWithAtten(in_dim, out_dim, normalize=False, aggr='mean')

def make_gat_conv(in_dim: int, out_dim: int,
                  heads: int = 4, edge_dim: Optional[int] = None) -> GATConvWithAtten:
    """GAT with concat=False: output is out_dim (not heads × out_dim).

    No constraint on out_dim % heads since heads are averaged, not concatenated.
    """
    return GATConvWithAtten(in_dim, out_dim, heads=heads, concat=False)

def make_pna_conv(in_dim: int, out_dim: int,
                  deg: Optional[Tensor] = None,
                  edge_dim: Optional[int] = None) -> PNAConvSimple:
    if deg is None:
        raise ValueError(
            "PNA requires a degree histogram (deg). "
            "Pass meta.deg from get_loaders() to build_model(). "
            "See loader.py:compute_deg_histogram()."
        )
    return PNAConvSimple(in_dim, out_dim, deg=deg)


CONV_FACTORIES = {
    'GIN':  make_gin_conv,
    'GCN':  make_gcn_conv,
    'SAGE': make_sage_conv,
    'GAT':  make_gat_conv,
    'PNA':  make_pna_conv,
}


def create_conv_layers(
    in_dim: int,
    hidden_dim: int,
    num_layers: int,
    backbone: str,
    deg: Optional[Tensor] = None,
    edge_dim: Optional[int] = None,
    gin_inner_bn: bool = True,
) -> torch.nn.ModuleList:
    """Create a ModuleList of conv layers for the given backbone.

    All layers output ``hidden_dim`` features.

    GAT note: uses concat=False so output is hidden_dim per layer —
    no need for out_dim divisibility by heads.

    PNA note: requires ``deg`` (degree histogram from training set).
    Use ``loader.compute_deg_histogram(train_dataset)`` and pass via
    ``meta.deg`` through ``build_model()``.
    """
    backbone = backbone.upper()
    if backbone not in CONV_FACTORIES:
        raise ValueError(
            f"Unknown backbone {backbone!r}. Choose from {list(CONV_FACTORIES)}")

    factory = CONV_FACTORIES[backbone]
    layers  = torch.nn.ModuleList()

    for i in range(num_layers):
        d_in = in_dim if i == 0 else hidden_dim
        if backbone == 'GAT':
            layers.append(factory(d_in, hidden_dim, heads=4))
        elif backbone == 'PNA':
            layers.append(factory(d_in, hidden_dim, deg=deg))
        elif backbone == 'GIN':
            layers.append(factory(d_in, hidden_dim, inner_bn=gin_inner_bn))
        else:
            layers.append(factory(d_in, hidden_dim))

    return layers
