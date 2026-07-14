"""mage.py — MAGE motif-level importance scores.

MAGE measures how much removing a motif type shifts the model's graph-level
representation, using cosine distance in embedding space.

Algorithm (per motif m, per graph g containing m):
  1. Forward pass on g → node embeddings h[N, D]
  2. Zero the input features of all nodes where nodes_to_motifs == m
  3. Forward pass on masked g → h_masked[N, D]
  4. Pool both: g_full = pool(h), g_masked = pool(h_masked)
  5. dist(m, g) = 1 - cosine_similarity(g_full, g_masked)
  score(m) = mean_{g containing m} dist(m, g)

Implementation notes
---------------------
- Uses nodes_to_motifs on each Data object, not the mask_cache pickle.
  This avoids a dependency on the MotifBreakdown output directory.
- Pooling uses the same pooling as the model (reads model.pool_type if
  available, otherwise defaults to global_mean_pool).
- Scores are raw cosine distances in [0, 1]. Not normalised.
- Node features are zeroed (not removed) to preserve graph topology.
  This matches how compute_motif_impact masks nodes in the evaluation pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, global_add_pool

# Return type shared with node-score baselines.
# For MAGE, 'mean' and 'max' are identical (score is already per-motif).
NodeScoreResult = Dict[str, Dict[int, float]]


@torch.no_grad()
def run_mage(
    model: torch.nn.Module,
    test_list: List[Data],
    vocab,
    device: torch.device,
    task_type: str = 'BinaryClass',
    max_graphs_per_motif: Optional[int] = 300,
    return_per_instance: bool = False,
) -> NodeScoreResult:
    """Per-motif importance scores from embedding-space cosine distances.

    Returns
    -------
    dict[motif_id -> float]
        Mean cosine distance when masking each motif type, in [0, 1].
        Not normalised — higher = more important to the model's representation.
    When ``return_per_instance`` is set, also returns a second dict
    ``{motif_id: {graph_idx: cosine_distance}}`` — MAGE's native per-(motif,
    graph) score, for the per-instance correlation. graph_idx indexes
    ``test_list``.
    """
    if not hasattr(model, 'get_emb'):
        print('  [warn] MAGE requires model.get_emb(x, edge_index, batch) -> node_emb')
        return ({}, {}) if return_per_instance else {}

    model.eval()
    model.to(device)

    # Determine pooling function from model if exposed, else default to mean.
    # NOTE: add vs mean is a NO-OP for MAGE's cosine-distance score (scale-invariant,
    # constant node count under feature-zeroing masks) — verified bit-identical.
    pool_type = getattr(model, 'pool_type', 'mean')
    pool_fn   = global_add_pool if pool_type == 'add' else global_mean_pool

    # Build smiles → data lookup
    smi_to_data: Dict[str, Data] = {}
    for d in test_list:
        smi = getattr(d, 'smiles', None)
        if smi:
            smi_to_data[str(smi)] = d

    # Collect all (motif_id, graph) pairs via nodes_to_motifs
    # Structure: motif_id -> list of (graph_idx, Data) containing it. graph_idx
    # indexes test_list so the per-instance dist aligns with the impact cache.
    motif_to_graphs: Dict[int, List] = {}
    for _gi, d in enumerate(test_list):
        n2m = getattr(d, 'nodes_to_motifs', None)
        if n2m is None:
            continue
        for mid in n2m[n2m >= 0].unique().tolist():
            motif_to_graphs.setdefault(int(mid), []).append((_gi, d))

    motif_scores: Dict[int, float] = {}
    per_instance: Dict[int, Dict[int, float]] = {}

    for mid, graphs in motif_to_graphs.items():
        cap     = max_graphs_per_motif or len(graphs)
        dists   = []

        for _gi, data in graphs[:cap]:
            data = data.to(device)
            n    = data.x.size(0)
            n2m  = data.nodes_to_motifs
            batch = (data.batch if data.batch is not None
                     else torch.zeros(n, dtype=torch.long, device=device))

            edge_attr = getattr(data, 'edge_attr', None)

            # Full embedding
            try:
                h_full = model.get_emb(data.x, data.edge_index, batch, edge_attr)
            except TypeError:
                h_full = model.get_emb(data.x, data.edge_index, batch)

            # Masked embedding: zero features for nodes belonging to motif mid
            motif_mask = (n2m == mid)          # [N] bool
            x_masked   = data.x.clone()
            x_masked[motif_mask] = 0.0

            try:
                h_masked = model.get_emb(x_masked, data.edge_index, batch, edge_attr)
            except TypeError:
                h_masked = model.get_emb(x_masked, data.edge_index, batch)

            g_full   = pool_fn(h_full,   batch)   # [1, D]
            g_masked = pool_fn(h_masked, batch)   # [1, D]

            cos_sim = F.cosine_similarity(g_full, g_masked, dim=-1)
            dist    = float((1.0 - cos_sim).clamp(min=0).item())
            dists.append(dist)
            if _gi is not None:
                per_instance.setdefault(mid, {})[_gi] = dist

        if dists:
            motif_scores[mid] = float(sum(dists) / len(dists))

    # MAGE operates at motif level (cosine distance per motif, not per node),
    # so mean and max are identical — both expose the same score.
    scores = {
        'mean': motif_scores,
        'max':  motif_scores,
    }
    return (scores, per_instance) if return_per_instance else scores
