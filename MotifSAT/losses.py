"""losses.py — MotifSAT training losses.

info_loss         — GSAT-style KL(Bernoulli(att) ‖ Beta(r)) per node/motif
motif_size_weight — normalise info loss by motif size
motif_consistency_loss — vectorised within/between motif variance (replaces O(M) Python loop)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
try:
    from torch_scatter import scatter_mean, scatter_var
except ImportError:
    from torch_geometric.utils import scatter as _sc
    def scatter_mean(src, index, dim=0, dim_size=None):
        return _sc(src, index, dim=dim, dim_size=dim_size, reduce="mean")
    def scatter_var(src, index, dim=0, dim_size=None):
        mean = scatter_mean(src, index, dim=dim, dim_size=dim_size)
        diff = (src - mean[index]) ** 2
        return scatter_mean(diff, index, dim=dim, dim_size=dim_size)


# ─────────────────────────────────────────────────────────────────────────────
# GSAT information loss
# ─────────────────────────────────────────────────────────────────────────────

def info_loss(
    att: Tensor,                # [N] or [M]  soft attention in (0,1)
    r: float,                   # prior Beta parameter (target retention rate)
    size_weights: Tensor = None,  # [N] or [M]  optional normalisation
) -> Tensor:
    """KL divergence between att and a Beta(r, 1-r) prior.

    Approximated as: KL = att * log(att/r) + (1-att) * log((1-att)/(1-r))

    Parameters
    ----------
    att : Tensor  soft attention values in (0, 1)
    r : float     prior mean (GSAT's decayed retention rate)
    size_weights : Tensor or None
        Per-element weights (e.g. 1/motif_size for size-normalised loss).

    Returns
    -------
    scalar loss
    """
    EPS = 1e-6
    att = att.clamp(EPS, 1 - EPS)
    r = max(EPS, min(1 - EPS, r))

    kl = att * (att.log() - torch.tensor(r).log().to(att.device)) \
       + (1 - att) * ((1 - att).log() - torch.tensor(1 - r).log().to(att.device))

    if size_weights is not None:
        kl = kl * size_weights.view(-1).to(att.device)

    return kl.mean()


def motif_size_weights(
    nodes_to_motifs: Tensor,
    motif_lengths: list,
) -> Tensor:
    """Compute 1/length per node based on the motif it belongs to.

    Returns a float tensor [N] where each entry is 1/length_of_motif_for_that_node.
    Unknown nodes (motif_id = -1) get weight 1.0.
    """
    weights = torch.ones(nodes_to_motifs.size(0), dtype=torch.float,
                         device=nodes_to_motifs.device)
    known = nodes_to_motifs >= 0
    if known.any():
        ids = nodes_to_motifs[known].cpu().tolist()
        lens = torch.tensor(
            [max(motif_lengths[i], 1) if i < len(motif_lengths) else 1
             for i in ids],
            dtype=torch.float, device=nodes_to_motifs.device
        )
        weights[known] = 1.0 / lens
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Motif consistency loss (vectorised via scatter)
# ─────────────────────────────────────────────────────────────────────────────

def motif_consistency_loss(
    att: Tensor,                  # [N, 1] or [N]  node attentions
    nodes_to_motifs: Tensor,      # [N]  global motif id per node
    batch: Tensor,                # [N]  graph id per node
) -> tuple[Tensor, Tensor]:
    """Compute within-motif and between-motif attention variance.

    Within-motif:   mean over (graph, motif) pairs of Var(att within motif)
                    → want LOW  (consistent attention within a motif)
    Between-motif:  mean over graphs of Var(mean_att across motifs in graph)
                    → want HIGH (discriminative across motifs)

    Fully vectorised — no Python loops over unique motifs.

    Returns
    -------
    within_var  : scalar Tensor
    between_var : scalar Tensor
    """
    att = att.view(-1)
    device = att.device

    nodes_to_motifs = nodes_to_motifs.to(device=device, dtype=torch.long)
    batch = batch.to(device=device, dtype=torch.long)

    # Build a unique (graph, motif) compound index for scatter operations
    n_graphs = int(batch.max().item()) + 1
    max_motif = int(nodes_to_motifs.max().item()) + 1
    gm_id = batch * max_motif + nodes_to_motifs   # [N]
    # Unknown nodes (nodes_to_motifs == -1) produce negative gm_id values
    # which crash torch.bincount.  Mask them out so they don't contribute
    # to within- or between-motif statistics.
    known = nodes_to_motifs >= 0                   # [N] bool
    att    = att[known]
    gm_id  = gm_id[known]

    # Within-motif variance: Var(att | gm_id) averaged over all (g,m) pairs
    # scatter_var computes unbiased variance (returns 0 for single-element groups)
    within = scatter_var(att, gm_id, dim=0, dim_size=n_graphs * max_motif)
    # Only include pairs with ≥ 2 nodes
    counts = torch.bincount(gm_id, minlength=n_graphs * max_motif).float()
    valid = counts >= 2
    within_var = within[valid].mean() if valid.any() else att.new_tensor(0.0)

    # Between-motif variance: for each graph, compute variance of per-motif means
    motif_means = scatter_mean(att, gm_id, dim=0, dim_size=n_graphs * max_motif)
    # motif_means shape: [n_graphs * max_motif]
    # Reshape to [n_graphs, max_motif] then compute variance per graph
    motif_means_2d = motif_means.view(n_graphs, max_motif)
    # Mask out (g, m) pairs that don't exist in this batch
    present = (counts.view(n_graphs, max_motif) > 0)
    between_vars = []
    for g in range(n_graphs):
        m_mask = present[g]
        if m_mask.sum() >= 2:
            between_vars.append(motif_means_2d[g][m_mask].var())
    between_var = (torch.stack(between_vars).mean() if between_vars
                   else att.new_tensor(0.0))

    return within_var, between_var
