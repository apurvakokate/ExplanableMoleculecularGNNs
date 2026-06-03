"""motif_modules.py — MotifSAT motif-level building blocks.

MotifPooling          — pool node embeddings to motif-instance level
MotifReadoutScorer    — score each motif instance with an MLP
compute_inverse_idx   — map nodes to dense motif-row indices
lift_motif_to_node    — broadcast motif-level values back to nodes
ExtractorMLP          — GSAT extractor: h_i → scalar logit
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("motifsat.motif_modules")

# Set MOTIFSAT_VERIFY_FIXES=1 to emit one-time confirmation logs proving the
# bug fixes are active at runtime (useful for sanity-checking a fresh run).
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
        return out, None   # torch_scatter returns (values, argmax); stub argmax as None


# ─────────────────────────────────────────────────────────────────────────────
# Index helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_inverse_idx(
    nodes_to_motifs: Tensor,
    batch: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Map each node to a dense motif-row index for scatter operations.

    Parameters
    ----------
    nodes_to_motifs : [N]  global motif vocab id per node
    batch : [N]            graph id per node

    Returns
    -------
    inverse_indices : [N]  dense index 0..M-1 for each node's motif instance
    motif_batch     : [M]  graph id for each motif instance
    motif_vocab_ids : [M]  global vocab motif_id for each motif instance
    """
    if batch is None:
        batch = torch.zeros(nodes_to_motifs.size(0), dtype=torch.long,
                            device=nodes_to_motifs.device)
    batch = batch.long()
    # Shift motif ids by +1 so unknown nodes (-1) become 0 and occupy their
    # own isolated per-graph buckets, distinct from any real motif. Without
    # the shift, -1 folds into a real motif row (and collides across graphs:
    # an unknown node in graph b lands on motif (max_mid-1) of graph b-1),
    # contaminating pooled motif embeddings in MotifReadoutScorer.
    offset = nodes_to_motifs.long() + 1            # unknown -1 -> 0, real k -> k+1
    max_mid = int(offset.max().item()) + 1
    gm_id = batch * max_mid + offset
    unique, inverse_indices = gm_id.unique(return_inverse=True)
    motif_batch = unique // max_mid
    motif_vocab_ids = (unique % max_mid) - 1       # unknown buckets -> -1, real -> k

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
    motif_vals: Tensor,     # [M] or [M, D]
    inverse_indices: Tensor,  # [N]
) -> Tensor:
    """Broadcast motif-level values back to node level."""
    return motif_vals[inverse_indices]


# ─────────────────────────────────────────────────────────────────────────────
# Motif pooling
# ─────────────────────────────────────────────────────────────────────────────

class MotifPooling(nn.Module):
    """Pool node embeddings to motif-instance level.

    mode : 'mean' | 'max' | 'max_mean' | 'multi'
      mean      → [M, D]
      max       → [M, D]
      max_mean  → [M, 2D]
      multi     → [M, 3D]  (mean + max + sum)
    """

    def __init__(self, mode: str = 'mean'):
        super().__init__()
        assert mode in ('mean', 'max', 'max_mean', 'multi'), \
            f"Unknown pool mode: {mode}"
        self.mode = mode

    @property
    def out_mult(self) -> int:
        return {'mean': 1, 'max': 1, 'max_mean': 2, 'multi': 3}[self.mode]

    def forward(
        self,
        emb: Tensor,              # [N, D]
        inverse_indices: Tensor,  # [N]
        num_motifs: Optional[int] = None,
    ) -> Tensor:
        """Return motif embeddings [M, out_mult * D]."""
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
        else:  # multi
            mean = scatter_mean(emb, inverse_indices, dim=0, dim_size=M)
            mx, _ = scatter_max(emb, inverse_indices, dim=0, dim_size=M)
            s = scatter_add(emb, inverse_indices, dim=0, dim_size=M)
            return torch.cat([mean, mx, s], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Extractor MLP  (GSAT-style: node/motif embedding → scalar logit)
# ─────────────────────────────────────────────────────────────────────────────

class ExtractorMLP(nn.Module):
    """MLP that maps node (or motif) embeddings to a scalar attention logit.

    Architecture: D → hidden → 1, with dropout.

    Parameters
    ----------
    in_dim : int
        Input dimension (hidden_dim from backbone, possibly multiplied by pool mult).
    hidden_mult : int
        hidden = in_dim * hidden_mult.
    dropout_p : float
    """

    def __init__(self, in_dim: int, hidden_mult: int = 2, dropout_p: float = 0.5):
        super().__init__()
        hidden = in_dim * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """Return log-logits [N, 1] or [M, 1]."""
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Motif Readout Scorer
# ─────────────────────────────────────────────────────────────────────────────

class MotifReadoutScorer(nn.Module):
    """Score motif instances: pool node embeddings, score with MLP.

    Used for motif_method='readout'.  Returns per-motif logits [M, 1]
    and corresponding per-node attention (broadcast back from motif level).

    Parameters
    ----------
    in_dim : int
        Node embedding dimension.
    pool_mode : str
        Aggregation mode for MotifPooling.
    hidden_mult, dropout_p : for ExtractorMLP.
    """

    def __init__(
        self,
        in_dim: int,
        pool_mode: str = 'mean',
        hidden_mult: int = 2,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.pooling = MotifPooling(pool_mode)
        pooled_dim = in_dim * self.pooling.out_mult
        self.scorer = ExtractorMLP(pooled_dim, hidden_mult, dropout_p)

    def forward(
        self,
        node_emb: Tensor,           # [N, D]
        inverse_indices: Tensor,    # [N]
        num_motifs: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return (motif_logits [M, 1], node_logits_broadcast [N, 1])."""
        motif_emb = self.pooling(node_emb, inverse_indices, num_motifs)
        motif_logits = self.scorer(motif_emb)                       # [M, 1]
        node_logits = lift_motif_to_node(motif_logits, inverse_indices)  # [N, 1]
        return motif_logits, node_logits
