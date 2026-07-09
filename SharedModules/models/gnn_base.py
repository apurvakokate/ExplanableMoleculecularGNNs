"""gnn_base.py — BaseGNN backbone used by both MOSE-GNN and MotifSAT.

The backbone handles:
  1. Node encoding: one-hot passthrough OR Linear + LayerNorm
  2. L conv layers (GIN/GCN/SAGE/GAT/PNA) with optional per-layer LayerNorm
  3. Three attention injection points:
       w_feat    — scale node features BEFORE conv: x̃ = att * x
       w_message — pass ``edge_atten`` into conv layers (scales messages)
       w_readout — scale node embeddings BEFORE pooling: h̃ = att * h
  4. Global add pooling → classification / regression head

The model is intentionally agnostic to where the attention weights (``att``)
come from.  MOSE-GNN passes learned sigmoid(θ_m) weights; MotifSAT passes
sampled Concrete/Sigmoid attention.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import global_add_pool

from .conv_layers import create_conv_layers, CONV_FACTORIES


class BaseGNN(nn.Module):
    """GNN backbone with configurable injection points.

    Parameters
    ----------
    x_dim : int
        Input node feature dimension (``NUM_ATOM_TYPES`` = 51 by default).
    hidden_dim : int
        Hidden dimension for conv layers and MLP.
    num_layers : int
        Number of message-passing layers.
    backbone : str
        Convolution type: ``'GIN'``, ``'GCN'``, ``'SAGE'``, ``'GAT'``, ``'PNA'``.
    node_encoder : str
        ``'onehot'`` — identity (no projection); input fed directly to first conv.
        ``'linear'`` — project x_dim → hidden_dim with LayerNorm.
    apply_layer_norm : bool
        Apply LayerNorm after each conv layer if True.
    dropout : float
        Dropout rate in the classification MLP.
    deg : Tensor or None
        Degree histogram for PNA.
    edge_dim : int or None
        Edge attribute dimension (for GAT/PNA).
    """

    def __init__(
        self,
        x_dim: int,
        hidden_dim: int,
        num_layers: int,
        backbone: str = 'GIN',
        node_encoder: str = 'onehot',
        apply_layer_norm: bool = False,
        dropout: float = 0.5,
        deg: Optional[Tensor] = None,
        edge_dim: Optional[int] = None,
        conv_normalize: str = 'none',
        gin_inner_bn: bool = True,
        self_gate: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.backbone = backbone.upper()
        # self_gate (optional, default OFF): when True, GIN/SAGE scale their
        # ungated self/root term by node attention during w_message injection,
        # so the attention gate controls ALL of a node's signal (parity with the
        # self-loop-free GCN/GAT). No-op for GCN/GAT/PNA. See get_embedding.
        self.self_gate = bool(self_gate)
        self.node_encoder_type = node_encoder
        self.dropout = dropout
        # Per-conv normalization applied AFTER each conv, BEFORE ReLU:
   #   'l2'        — F.normalize(x, p=2, dim=1): unit-length node embeddings.
   #   'layernorm' — LayerNorm(hidden_dim) per layer (learned scale/shift).
   #   'none'      — no per-conv normalization (default).
        # Back-compat: apply_layer_norm=True forces 'layernorm'.
        conv_normalize = (conv_normalize or 'none').lower()
        if apply_layer_norm:
            conv_normalize = 'layernorm'
        if conv_normalize not in ('l2', 'layernorm', 'none'):
            raise ValueError(f"conv_normalize must be l2|layernorm|none, "
                             f"got {conv_normalize!r}")
        self.conv_normalize = conv_normalize
        self.apply_layer_norm = (conv_normalize == 'layernorm')

        # Node encoder:
        #   'onehot'       — identity passthrough (x is already one-hot, dim = x_dim)
        #   'linear'       — Linear(x_dim → hidden_dim) + LayerNorm
        #   'atom_encoder' — OGB AtomEncoder (x is [N, 9] integer tensor)
        #                    Requires ogb to be installed; emb_dim = hidden_dim
        if node_encoder == 'linear':
            self.node_encoder = nn.Linear(x_dim, hidden_dim)
            self.node_encoder_norm = nn.LayerNorm(hidden_dim)
            conv_in_dim = hidden_dim
        elif node_encoder == 'atom_encoder':
            try:
                from ogb.graphproppred.mol_encoder import AtomEncoder
                self.node_encoder = AtomEncoder(emb_dim=hidden_dim)
            except ImportError:
                raise ImportError(
                    "node_encoder='atom_encoder' requires ogb: pip install ogb")
            self.node_encoder_norm = nn.Identity()
            conv_in_dim = hidden_dim
        else:  # onehot
            self.node_encoder = nn.Identity()
            self.node_encoder_norm = nn.Identity()
            conv_in_dim = x_dim

        self.convs = create_conv_layers(
            conv_in_dim, hidden_dim, num_layers,
            backbone, deg=deg, edge_dim=edge_dim,
            gin_inner_bn=gin_inner_bn,
        )

        if self.conv_normalize == 'layernorm':
            self.layer_norms = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
            )
        else:
            self.layer_norms = None

        # Classification / regression head
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)   # output size overridden in subclasses

    def encode(self, x: Tensor) -> Tensor:
        """Apply node encoder (projection + LayerNorm, or identity)."""
        return self.node_encoder_norm(self.node_encoder(x))

    def convolve(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        edge_atten: Optional[Tensor] = None,
        node_self_gate: Optional[Tensor] = None,
    ) -> Tensor:
        """Run all conv layers.

        ``edge_atten`` is passed to each layer for ``w_message`` injection.
        ``node_self_gate`` (optional) is passed to each layer to gate the
        per-node self/root term (GIN/SAGE only); None ⇒ unchanged behaviour.
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr,
                     edge_atten=edge_atten, node_self_gate=node_self_gate)
            # Per-conv normalization (before ReLU), matching reference order.
            if self.conv_normalize == 'l2':
                x = F.normalize(x, p=2, dim=1)
            elif self.conv_normalize == 'layernorm' and self.layer_norms is not None:
                x = self.layer_norms[i](x)
            x = F.relu(x)
        return x

    def get_embedding(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_att: Optional[Tensor] = None,
        edge_atten: Optional[Tensor] = None,
        w_feat: bool = False,
        w_message: bool = True,
        w_readout: bool = False,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward through encoder → (optional w_feat) → conv → (optional w_readout) → pool.

        Parameters
        ----------
        x : Tensor [N, x_dim]
        edge_index : Tensor [2, E]
        edge_attr : Tensor [E, edge_dim] or None
        node_att : Tensor [N, 1] or None
            Attention weights.  Must be provided when any of w_feat/w_readout
            is True.  Used as edge_atten when w_message is True and
            edge_atten is None.
        edge_atten : Tensor [E, 1] or None
            Explicit edge-level attention.  If None and w_message=True,
            derived from node_att as ``att[src] * att[dst]``.
        w_feat, w_message, w_readout : bool
            Injection flags.
        batch : Tensor [N] or None

        Returns
        -------
        graph_emb : Tensor [B, hidden_dim]
            Pooled graph embedding (the w_readout weighting, when enabled, is
            applied to node embeddings before this pooling).
        node_emb  : Tensor [N, hidden_dim]
            Conv-stack output BEFORE any w_readout weighting (the raw node
            embeddings; w_readout only affects the pooled graph_emb).
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Edge features are explicitly disabled for all experiments.
        # Bond-type information is carried implicitly by node features;
        # enabling edge_attr would change model capacity across runs.
        edge_attr = None

        h = self.encode(x)

        # w_feat: scale encoded node features by attention weight
        if w_feat and node_att is not None:
            h = h * node_att.view(-1, 1)

        # Build edge_atten for w_message
        _edge_atten: Optional[Tensor] = None
        if w_message:
            if edge_atten is not None:
                _edge_atten = edge_atten
            elif node_att is not None:
                src, dst = edge_index
                _edge_atten = (node_att.view(-1)[src] * node_att.view(-1)[dst]).unsqueeze(-1)

        # self_gate: gate GIN/SAGE self-terms by node attention (only meaningful
        # alongside w_message, and only when attention is present). Default OFF.
        _self_gate = (node_att if (self.self_gate and w_message
                                   and node_att is not None) else None)
        h = self.convolve(h, edge_index, edge_attr=edge_attr,
                          edge_atten=_edge_atten, node_self_gate=_self_gate)

        # w_readout: scale node embeddings before pooling
        h_readout = h * node_att.view(-1, 1) if (w_readout and node_att is not None) else h

        graph_emb = global_add_pool(h_readout, batch)

        return graph_emb, h

    def classify(self, graph_emb: Tensor) -> Tensor:
        """Two-layer MLP head (override for multi-output)."""
        out = F.relu(self.lin1(graph_emb))
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.lin2(out)
