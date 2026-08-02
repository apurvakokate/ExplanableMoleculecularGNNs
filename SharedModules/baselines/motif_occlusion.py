"""motif_occlusion.py — Motif-Occlusion motif-level importance scores.

NOTE: This is NOT the MAGE method (arXiv 2405.12519). It was previously
mislabelled "MAGE"; it is a homegrown cosine-embedding *occlusion* baseline.
The real MAGE Stage-2 attention scorer lives in ``mage.py``. This method
measures how much removing (zeroing) a motif type shifts the model's
graph-level representation, using cosine distance in embedding space.

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
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, global_add_pool

# Return type shared with node-score baselines.
# For Motif-Occlusion, 'mean' and 'max' are identical (score is already per-motif).
NodeScoreResult = Dict[str, Dict[int, float]]


@torch.no_grad()
def run_motif_occlusion(
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
    ``{motif_id: {graph_idx: cosine_distance}}`` — the native per-(motif,
    graph) score, for the per-instance correlation. graph_idx indexes
    ``test_list``.
    """
    if not hasattr(model, 'get_emb'):
        print('  [warn] Motif_Occlusion requires model.get_emb(x, edge_index, batch) -> node_emb')
        return ({}, {}) if return_per_instance else {}

    model.eval()
    model.to(device)

    # Determine pooling function from model if exposed, else default to mean.
    # NOTE: add vs mean is a NO-OP for the cosine-distance score (scale-invariant,
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

    # Motif-Occlusion operates at motif level (cosine distance per motif, not per node),
    # so mean and max are identical — both expose the same score.
    scores = {
        'mean': motif_scores,
        'max':  motif_scores,
    }
    return (scores, per_instance) if return_per_instance else scores


@torch.no_grad()
def run_motif_occlusion_batched(
    model: torch.nn.Module,
    test_list: List[Data],
    vocab,
    device: torch.device,
    task_type: str = 'BinaryClass',
    max_graphs_per_motif: Optional[int] = 300,
    batch_size: int = 256,
    return_per_instance: bool = False,
) -> NodeScoreResult:
    """GPU-batched Motif-Occlusion — bit-identical to ``run_motif_occlusion`` but
    batches the encoder forwards (the GPU-underutilized part).

    Two determinism-preserving optimizations over the per-graph loop, both exact:
      * the FULL-graph embedding depends only on the graph (not the motif), so it is
        computed ONCE per graph and cached — the sequential version recomputes it for
        every (motif, graph) pair, wastefully but to the same value;
      * the MASKED embeddings for a motif's graphs are computed in one batched forward.
    Add-pool over a disconnected batch is a per-graph segment sum, so results match
    the loop to float precision (verified by ab_occlusion_batched.py). The
    ``max_graphs_per_motif`` cap and graph order are preserved so the SAME graphs are
    scored as the sequential path."""
    if not hasattr(model, 'get_emb'):
        print('  [warn] Motif_Occlusion requires model.get_emb(...)')
        return ({}, {}) if return_per_instance else {}
    model.eval()
    model.to(device)
    pool_type = getattr(model, 'pool_type', 'mean')
    pool_fn = global_add_pool if pool_type == 'add' else global_mean_pool

    def _embed(datas: List[Data]) -> Optional[torch.Tensor]:
        """Pooled embeddings [len(datas), D] via batched forwards."""
        if not datas:
            return None
        out = []
        for s in range(0, len(datas), batch_size):
            b = Batch.from_data_list(datas[s:s + batch_size]).to(device)
            ea = getattr(b, 'edge_attr', None)
            try:
                h = model.get_emb(b.x, b.edge_index, b.batch, ea)
            except TypeError:
                h = model.get_emb(b.x, b.edge_index, b.batch)
            out.append(pool_fn(h, b.batch))
        return torch.cat(out, dim=0)

    # motif_id -> [(graph_idx, Data)] containing it (graph_idx indexes test_list)
    motif_to_graphs: Dict[int, List] = {}
    for gi, d in enumerate(test_list):
        n2m = getattr(d, 'nodes_to_motifs', None)
        if n2m is None:
            continue
        for mid in n2m[n2m >= 0].unique().tolist():
            motif_to_graphs.setdefault(int(mid), []).append((gi, d))

    # Apply the per-motif cap FIRST, then embed only the graphs actually scored
    # (matches which graphs the sequential path embeds).
    sel_by_mid = {mid: graphs[:(max_graphs_per_motif or len(graphs))]
                  for mid, graphs in motif_to_graphs.items()}
    needed = sorted({gi for sel in sel_by_mid.values() for gi, _ in sel})
    g_full_mat = _embed([test_list[gi] for gi in needed])
    g_full = {gi: g_full_mat[i] for i, gi in enumerate(needed)} if g_full_mat is not None else {}

    motif_scores: Dict[int, float] = {}
    per_instance: Dict[int, Dict[int, float]] = {}
    for mid, sel in sel_by_mid.items():
        masked = []
        for _gi, d in sel:
            n2m = d.nodes_to_motifs.view(-1)
            xm = d.x.clone()
            xm[n2m == mid] = 0.0
            md = Data(x=xm, edge_index=d.edge_index)
            ea = getattr(d, 'edge_attr', None)
            if ea is not None:
                md.edge_attr = ea
            masked.append(md)
        g_masked = _embed(masked)
        dists = []
        for i, (_gi, _d) in enumerate(sel):
            cos = F.cosine_similarity(g_full[_gi].unsqueeze(0),
                                      g_masked[i].unsqueeze(0), dim=-1)
            dist = float((1.0 - cos).clamp(min=0).item())
            dists.append(dist)
            per_instance.setdefault(mid, {})[_gi] = dist
        if dists:
            motif_scores[mid] = float(sum(dists) / len(dists))

    scores = {'mean': motif_scores, 'max': motif_scores}
    return (scores, per_instance) if return_per_instance else scores
