"""motif_modules.py — MotifSAT motif-level building blocks.

MotifPooling          — pool node embeddings to motif-instance level
MotifReadoutScorer    — score each motif instance with an MLP
compute_inverse_idx   — map nodes to dense motif-row indices
lift_motif_to_node    — broadcast motif-level values back to nodes
ExtractorMLP          — official GSAT extractor (InstanceNorm MLP)
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import InstanceNorm

logger = logging.getLogger("motifsat.motif_modules")

_VERIFY_FIXES = os.environ.get("MOTIFSAT_VERIFY_FIXES", "0") == "1"
_logged_inverse_idx_fix = False
try:
    from torch_scatter import scatter_mean, scatter_max, scatter_add
except ImportError:
    from torch_geometric.utils import scatter as _sc
    def scatter_mean(src, index, dim=0, dim_size=None):
        return _sc(src, index, dim=dim, dim_size=dim_size, reduce="mean")
    def scatter_add(src, index, dim=0, dim_size=None):
        return _sc(src, index, dim=dim, dim_size=dim_size, reduce="sum")
    def scatter_max(src, index, dim=0, dim_size=None):
        out = _sc(src, index, dim=dim, dim_size=dim_size, reduce="max")
        return out, None


# ─────────────────────────────────────────────────────────────────────────────
# Index helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_inverse_idx(
    nodes_to_motifs: Tensor,
    batch: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Map each node to a dense motif-row index for scatter operations."""
    if batch is None:
        batch = torch.zeros(nodes_to_motifs.size(0), dtype=torch.long,
                            device=nodes_to_motifs.device)
    batch = batch.long()
    offset = nodes_to_motifs.long() + 1
    max_mid = int(offset.max().item()) + 1
    gm_id = batch * max_mid + offset
    unique, inverse_indices = gm_id.unique(return_inverse=True)
    motif_batch = unique // max_mid
    motif_vocab_ids = (unique % max_mid) - 1

    global _logged_inverse_idx_fix
    if _VERIFY_FIXES and not _logged_inverse_idx_fix:
        n_unknown = int((nodes_to_motifs < 0).sum().item())
        n_unknown_rows = int((motif_vocab_ids < 0).sum().item())
        logger.info(
            "[FIX#2 active] compute_inverse_idx: unknown nodes isolated "
            "(this batch: %d unknown nodes -> %d dedicated -1 rows; "
            "%d real motif rows uncontaminated)",
            n_unknown, n_unknown_rows,
            int((motif_vocab_ids >= 0).sum().item()),
        )
        _logged_inverse_idx_fix = True

    return inverse_indices, motif_batch, motif_vocab_ids


def lift_motif_to_node(
    motif_vals: Tensor,
    inverse_indices: Tensor,
) -> Tensor:
    """Broadcast motif-level values back to node level."""
    return motif_vals[inverse_indices]


# ─────────────────────────────────────────────────────────────────────────────
# Motif pooling
# ─────────────────────────────────────────────────────────────────────────────

class MotifPooling(nn.Module):
    """Pool node embeddings to motif-instance level."""

    def __init__(self, mode: str = 'mean'):
        super().__init__()
        if mode not in ('mean', 'max', 'max_mean', 'multi'):
            raise ValueError(f"Unknown pool mode: {mode}")
        self.mode = mode

    @property
    def out_mult(self) -> int:
        return {'mean': 1, 'max': 1, 'max_mean': 2, 'multi': 3}[self.mode]

    def forward(
        self,
        emb: Tensor,
        inverse_indices: Tensor,
        num_motifs: Optional[int] = None,
    ) -> Tensor:
        M = (inverse_indices.max().item() + 1 if num_motifs is None
             else num_motifs)
        M = int(M)
        if self.mode == 'mean':
            return scatter_mean(emb, inverse_indices, dim=0, dim_size=M)
        elif self.mode == 'max':
            out, _ = scatter_max(emb, inverse_indices, dim=0, dim_size=M)
            return out
        elif self.mode == 'max_mean':
            mean = scatter_mean(emb, inverse_indices, dim=0, dim_size=M)
            mx, _ = scatter_max(emb, inverse_indices, dim=0, dim_size=M)
            return torch.cat([mx, mean], dim=1)
        else:
            mean = scatter_mean(emb, inverse_indices, dim=0, dim_size=M)
            mx, _ = scatter_max(emb, inverse_indices, dim=0, dim_size=M)
            s = scatter_add(emb, inverse_indices, dim=0, dim_size=M)
            return torch.cat([mean, mx, s], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Official GSAT Extractor MLP  (Graph-COM/GSAT run_gsat.py ExtractorMLP + MLP)
# ─────────────────────────────────────────────────────────────────────────────

class BatchSequential(nn.Sequential):
    """Sequential module that passes ``batch`` into InstanceNorm layers."""

    def forward(self, inputs: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        for module in self._modules.values():
            if isinstance(module, InstanceNorm):
                if batch is None:
                    raise ValueError("InstanceNorm in ExtractorMLP requires batch indices")
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


def _gsat_mlp_channels(in_dim: int, edge_mode: bool) -> List[int]:
    """Channel sizes matching official GSAT ExtractorMLP."""
    if edge_mode:
        # [2H, 4H, H, 1] when in_dim = 2H
        h = in_dim // 2
        return [in_dim, in_dim * 2, h, 1]
    # node: [H, 2H, H, 1]
    return [in_dim, in_dim * 2, in_dim, 1]


class ExtractorMLP(nn.Module):
    """Official GSAT extractor: graph-wise InstanceNorm MLP → scalar logit.

    Node path (``edge_mode=False``): ``[D, 2D, D, 1]``.
    Edge path (``edge_mode=True``): ``[2D, 4D, D, 1]`` with ``in_dim=2D``.

    ``batch`` indexes the graph id per row (nodes, motif instances, or edge
    source nodes for the edge extractor — same as official GSAT).
    """

    def __init__(
        self,
        in_dim: int,
        dropout_p: float = 0.5,
        edge_mode: bool = False,
        hidden_mult: int = 2,  # kept for API compat; official widths are fixed
    ):
        super().__init__()
        del hidden_mult  # official architecture uses fixed channel schedule
        channels = _gsat_mlp_channels(in_dim, edge_mode)
        layers: list[nn.Module] = []
        for i in range(1, len(channels)):
            layers.append(nn.Linear(channels[i - 1], channels[i], bias=True))
            if i < len(channels) - 1:
                layers.append(InstanceNorm(channels[i]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_p))
        self.net = BatchSequential(*layers)
        self.edge_mode = edge_mode

    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        return self.net(x, batch)


# ─────────────────────────────────────────────────────────────────────────────
# Motif Readout Scorer
# ─────────────────────────────────────────────────────────────────────────────

class MotifReadoutScorer(nn.Module):
    """Pool node embeddings → motif MLP → motif logits (+ optional node broadcast).

    Used when ``motif_method='readout'`` or ``noise in ('node', 'motif')``.
    """

    def __init__(
        self,
        in_dim: int,
        pool_mode: str = 'mean',
        dropout_p: float = 0.5,
        hidden_mult: int = 2,
    ):
        super().__init__()
        del hidden_mult
        self.pooling = MotifPooling(pool_mode)
        pooled_dim = in_dim * self.pooling.out_mult
        self.scorer = ExtractorMLP(pooled_dim, dropout_p=dropout_p)

    def forward(
        self,
        node_emb: Tensor,
        inverse_indices: Tensor,
        motif_batch: Tensor,
        num_motifs: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return (motif_logits [M, 1], node_logits_broadcast [N, 1])."""
        motif_emb = self.pooling(node_emb, inverse_indices, num_motifs)
        motif_logits = self.scorer(motif_emb, motif_batch)
        node_logits = lift_motif_to_node(motif_logits, inverse_indices)
        return motif_logits, node_logits
