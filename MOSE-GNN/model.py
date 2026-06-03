"""model.py — MOSE-GNN learnable global motif importance models.

Two variants:
  SingleChannelGNN  — BinaryClass or Regression (one output)
  MultiChannelGNN   — MultiLabel (one conv stack per class)

Both accept motif_params [M] or [M, C] and map node i → σ(θ_{m(i),c}).
Unknown nodes (motif_id = -1) use unk_param (fixed or learnable scalar).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import global_add_pool

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from SharedModules.models.gnn_base import BaseGNN
from SharedModules.models.conv_layers import create_conv_layers
from SharedModules.data.dataset import NUM_ATOM_TYPES, EDGE_FEAT_DIM


# ─────────────────────────────────────────────────────────────────────────────
# Shared motif weight helper
# ─────────────────────────────────────────────────────────────────────────────

def _motif_to_node_weights(
    node_to_motifs: Tensor,
    motif_params: Tensor,           # [M, C] or [M, 1]
    num_nodes: int,
    device: torch.device,
    unk_mode: str = 'fixed',
    unk_param: Optional[nn.Parameter] = None,
    unk_value: float = 0.5,
    ignore_unknowns: bool = False,
    masked_motif: Optional[int] = None,
) -> Tensor:
    """Vectorised: map each node to its sigmoid weight per class.

    Parameters
    ----------
    node_to_motifs : Tensor [N]
        Motif index per node (-1 for unknown).
    motif_params : Tensor [M, C] or [M, 1]
    num_nodes : int
    unk_mode : 'fixed' | 'learnable_shared'
    unk_param : nn.Parameter (scalar) or None  — used when unk_mode='learnable_shared'
    unk_value : float  — used when unk_mode='fixed'
    ignore_unknowns : bool  — if True unknown nodes get weight 0
    masked_motif : int or None  — set this motif's weight to 0 (for impact eval)

    Returns
    -------
    Tensor [N, C]
    """
    n_classes = motif_params.size(1) if motif_params.dim() == 2 else 1

    # Determine unknown value
    if ignore_unknowns:
        unk_val = 0.0
    elif unk_mode == 'learnable_shared' and unk_param is not None:
        unk_val = float(unk_param.sigmoid())
    else:
        unk_val = unk_value

    weights = torch.full((num_nodes, n_classes), unk_val,
                         device=device, dtype=torch.float32)

    known_mask = node_to_motifs >= 0
    if known_mask.any():
        known_idx = node_to_motifs[known_mask]
        p = motif_params if motif_params.dim() == 2 else motif_params.unsqueeze(-1)
        weights[known_mask] = p[known_idx].sigmoid()

    if masked_motif is not None:
        weights[node_to_motifs == masked_motif] = 0.0

    return weights  # [N, C]


# ─────────────────────────────────────────────────────────────────────────────
# Single-channel model  (BinaryClass / Regression)
# ─────────────────────────────────────────────────────────────────────────────

class SingleChannelGNN(nn.Module):
    """GNN with a single conv stack and per-motif scalar weights.

    Parameters
    ----------
    x_dim, hidden_dim, num_layers, backbone, node_encoder, apply_layer_norm
        Passed to BaseGNN.
    num_motifs : int
        Vocabulary size (number of known motifs).
    unk_mode : 'fixed' | 'learnable_shared'
    unk_value : float
        Used when unk_mode='fixed' (default 0.5).
    w_feat, w_message, w_readout : bool
        Injection flags.
    dropout : float
    deg, edge_dim : for PNA / GAT.
    """

    def __init__(
        self,
        x_dim: int = NUM_ATOM_TYPES,
        hidden_dim: int = 64,
        num_layers: int = 3,
        backbone: str = 'GIN',
        node_encoder: str = 'onehot',
        apply_layer_norm: bool = False,
        num_motifs: int = 0,
        unk_mode: str = 'fixed',
        unk_value: float = 0.5,
        w_feat: bool = True,
        w_message: bool = False,
        w_readout: bool = True,
        dropout: float = 0.5,
        deg=None,
        edge_dim: Optional[int] = None,
        conv_normalize: str = 'l2',
        gin_inner_bn: bool = True,
    ):
        super().__init__()
        self.w_feat = w_feat
        self.w_message = w_message
        self.w_readout = w_readout
        self.unk_mode = unk_mode
        self.unk_value = unk_value

        self.backbone = BaseGNN(
            x_dim=x_dim, hidden_dim=hidden_dim, num_layers=num_layers,
            backbone=backbone, node_encoder=node_encoder,
            apply_layer_norm=apply_layer_norm, dropout=dropout,
            deg=deg, edge_dim=edge_dim,
            conv_normalize=conv_normalize, gin_inner_bn=gin_inner_bn,
        )
        self.backbone.lin2 = nn.Linear(hidden_dim, 1)

        if num_motifs > 0:
            self.motif_params = nn.Parameter(torch.zeros(num_motifs, 1))
        else:
            self.register_parameter('motif_params', None)

        if unk_mode == 'learnable_shared':
            self.unk_param = nn.Parameter(torch.tensor(0.0))  # sigmoid → 0.5 init
        else:
            self.register_parameter('unk_param', None)

    def get_motif_scores(self) -> Dict[int, float]:
        """Return {motif_id: sigmoid(θ_m)} for all known motifs."""
        if self.motif_params is None:
            return {}
        scores = self.motif_params.squeeze(-1).sigmoid().detach().cpu()
        return {i: float(s) for i, s in enumerate(scores)}

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
        nodes_to_motifs: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        ignore_unknowns: bool = False,
        masked_motif: Optional[int] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if self.motif_params is None or nodes_to_motifs is None:
            node_att = None
        else:
            node_att = _motif_to_node_weights(
                nodes_to_motifs.to(x.device),
                self.motif_params,
                x.size(0), x.device,
                unk_mode=self.unk_mode,
                unk_param=self.unk_param,
                unk_value=self.unk_value,
                ignore_unknowns=ignore_unknowns,
                masked_motif=masked_motif,
            )  # [N, 1]

        graph_emb, _ = self.backbone.get_embedding(
            x, edge_index,
            edge_attr=edge_attr,
            node_att=node_att,
            w_feat=self.w_feat and node_att is not None,
            w_message=self.w_message and node_att is not None,
            w_readout=self.w_readout and node_att is not None,
            batch=batch,
        )
        return self.backbone.classify(graph_emb), node_att


# ─────────────────────────────────────────────────────────────────────────────
# Multi-channel model  (MultiLabel)
# ─────────────────────────────────────────────────────────────────────────────

class MultiChannelGNN(nn.Module):
    """GNN with per-class conv stacks and per-motif-per-class weights.

    ``motif_params`` shape is [M, C] so motif m has an independent importance
    score for each class c.  This means gradients for class c only update
    column c of motif_params, implementing per-task information bottleneck.

    Parameters: same as SingleChannelGNN, plus:
    num_classes : int
        Number of output tasks.
    """

    def __init__(
        self,
        x_dim: int = NUM_ATOM_TYPES,
        hidden_dim: int = 64,
        num_layers: int = 3,
        backbone: str = 'GIN',
        node_encoder: str = 'onehot',
        apply_layer_norm: bool = False,
        num_classes: int = 12,
        num_motifs: int = 0,
        unk_mode: str = 'fixed',
        unk_value: float = 0.5,
        w_feat: bool = True,
        w_message: bool = False,
        w_readout: bool = True,
        dropout: float = 0.5,
        deg=None,
        edge_dim: Optional[int] = None,
        conv_normalize: str = 'l2',
        gin_inner_bn: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.w_feat = w_feat
        self.w_message = w_message
        self.w_readout = w_readout
        self.unk_mode = unk_mode
        self.unk_value = unk_value
        conv_normalize = (conv_normalize or 'none').lower()
        if apply_layer_norm:
            conv_normalize = 'layernorm'
        self.conv_normalize = conv_normalize

        # Per-class conv stacks + MLP heads
        self.convs = nn.ModuleDict()
        self.lin1  = nn.ModuleDict()
        self.lin2  = nn.ModuleDict()
        for c in range(num_classes):
            layers = create_conv_layers(
                x_dim if node_encoder == 'onehot' else hidden_dim,
                hidden_dim, num_layers, backbone, deg=deg, edge_dim=edge_dim,
                gin_inner_bn=gin_inner_bn,
            )
            self.convs[str(c)] = layers
            self.lin1[str(c)] = nn.Linear(hidden_dim, hidden_dim)
            self.lin2[str(c)] = nn.Linear(hidden_dim, 1)

        # Shared node encoder (one-hot = identity)
        if node_encoder == 'linear':
            self.node_encoder = nn.Linear(x_dim, hidden_dim)
            self.node_encoder_norm = nn.LayerNorm(hidden_dim)
        else:
            self.node_encoder = nn.Identity()
            self.node_encoder_norm = nn.Identity()

        if conv_normalize == 'layernorm':
            self.layer_norms = nn.ModuleDict({
                str(c): nn.ModuleList([nn.LayerNorm(hidden_dim)
                                       for _ in range(num_layers)])
                for c in range(num_classes)
            })
        else:
            self.layer_norms = None

        self.dropout = dropout

        if num_motifs > 0:
            self.motif_params = nn.Parameter(torch.zeros(num_motifs, num_classes))
        else:
            self.register_parameter('motif_params', None)

        if unk_mode == 'learnable_shared':
            self.unk_param = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter('unk_param', None)

    def get_motif_scores(self) -> Dict[str, Dict[int, float]]:
        """Return {class_idx: {motif_id: score}} for all classes."""
        if self.motif_params is None:
            return {}
        scores = self.motif_params.sigmoid().detach().cpu()
        return {
            c: {i: float(scores[i, c]) for i in range(scores.size(0))}
            for c in range(scores.size(1))
        }

    def _encode(self, x: Tensor) -> Tensor:
        return self.node_encoder_norm(self.node_encoder(x))

    def _conv_forward(self, x: Tensor, edge_index: Tensor,
                      class_id: int, edge_attr=None,
                      edge_atten=None) -> Tensor:
        layers = self.convs[str(class_id)]
        norms = (self.layer_norms[str(class_id)]
                 if self.layer_norms is not None else None)
        for i, conv in enumerate(layers):
            x = conv(x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            if self.conv_normalize == 'l2':
                x = F.normalize(x, p=2, dim=1)
            elif self.conv_normalize == 'layernorm' and norms is not None:
                x = norms[i](x)
            x = F.relu(x)
        return x

    def _classify(self, x: Tensor, class_id: int) -> Tensor:
        x = F.relu(self.lin1[str(class_id)](x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2[str(class_id)](x)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
        nodes_to_motifs: Optional[Tensor] = None,
        edge_attr=None,
        ignore_unknowns: bool = False,
        masked_motif: Optional[int] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Edge features disabled for all experiments
        # (MultiChannelGNN bypasses BaseGNN.get_embedding so must null here)
        edge_attr = None

        if self.motif_params is None or nodes_to_motifs is None:
            node_att = None
        else:
            node_att = _motif_to_node_weights(
                nodes_to_motifs.to(x.device),
                self.motif_params,
                x.size(0), x.device,
                unk_mode=self.unk_mode,
                unk_param=self.unk_param,
                unk_value=self.unk_value,
                ignore_unknowns=ignore_unknowns,
                masked_motif=masked_motif,
            )  # [N, C]

        h_base = self._encode(x)

        outputs = []
        for c in range(self.num_classes):
            att_c = node_att[:, c].unsqueeze(-1) if node_att is not None else None

            # w_feat: scale encoded features by class-c motif weight
            h_c = h_base * att_c if (self.w_feat and att_c is not None) else h_base

            # w_message: per-class edge attention = att_c[src] * att_c[dst]
            edge_atten_c = None
            if self.w_message and att_c is not None:
                src, dst = edge_index
                edge_atten_c = (att_c.view(-1)[src] * att_c.view(-1)[dst]).unsqueeze(-1)

            # Conv
            h_c = self._conv_forward(h_c, edge_index, c, edge_attr=edge_attr,
                                     edge_atten=edge_atten_c)

            # w_readout: scale before pooling
            h_pool = h_c * att_c if (self.w_readout and att_c is not None) else h_c
            g_c = global_add_pool(h_pool, batch)

            outputs.append(self._classify(g_c, c))

        return torch.cat(outputs, dim=1), node_att
