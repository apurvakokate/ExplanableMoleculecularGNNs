"""gnn_explainer_batched.py — GPU-batched GNNExplainer (graph classification).

The stock ``run_gnnexplainer`` optimizes a node mask ONE graph at a time (PyG's
Explainer is per-instance). For the tiny molecular graphs here the GPU is
launch-overhead-bound, so batching many graphs per optimization step is the main
speedup lever (see arXiv:2601.04807, "Parallelizing Node-Level Explainability").

CORRECTNESS — avoiding the "gradient-mixing" problem
----------------------------------------------------
K graphs are packed into one disconnected ``Batch`` with a SINGLE node mask of
size ``[N_total, 1]``. This is equivalent to K independent optimizations iff each
graph's mask entries receive gradients ONLY from that graph:

  * message passing never crosses disconnected components, and pooling is
    per-graph via the ``batch`` vector, so ``out_i`` depends only on graph i's
    nodes → the prediction loss is block-diagonal automatically;
  * the size / entropy penalties MUST be reduced PER GRAPH (segment-mean over the
    ``batch`` vector), NOT as one global mean — a global mean divides every node's
    penalty gradient by ``N_total`` instead of that graph's ``N_i``, which couples
    graphs (this is the gradient-mixing trap). We use ``scatter(..., reduce=mean)``.

With per-graph losses SUMMED (not averaged) over the batch, each graph's gradient
equals what it would get run alone, so Adam evolves each mask slice identically to
the sequential baseline. ``ab_gnnexplainer_batched.py`` verifies this numerically
(batched batch_size=K == batched batch_size=1, max|Δ| ~ 0).

Objective mirrors PyG's GNNExplainer (node_mask_type='object'): mask init
``randn(N,1)*0.1``, applied ``x * sigmoid(mask)``, coeffs node_feat_size=1.0,
node_feat_ent=0.1, Adam lr=0.01, epochs=100. Per-graph mask slices are seeded by
GLOBAL graph index so batched and sequential share identical initialisation.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter

NodeScoreResult = Dict[str, Dict[int, float]]
_EPS = 1e-15


def _forward_logits(model, x, edge_index, batch, n2m):
    """Model forward returning the logits tensor (mirrors run_gnnexplainer's
    _Wrapper: routes nodes_to_motifs through, unwraps a (logits, …) tuple)."""
    out = model(x, edge_index, batch, n2m)
    return out[0] if isinstance(out, (tuple, list)) else out


def _pred_loss_sum(out, full, task_type: str) -> torch.Tensor:
    """Sum (NOT mean) of per-graph model-explanation prediction losses, so each
    graph contributes exactly the gradient it would get run alone. Target is the
    frozen model's own full-graph prediction (explanation_type='model')."""
    if task_type == 'Regression':
        return F.mse_loss(out.view(-1), full.view(-1).detach(), reduction='sum')
    if task_type == 'MultiLabel':
        tgt = (torch.sigmoid(full) > 0.5).float().detach()
        return F.binary_cross_entropy_with_logits(out, tgt, reduction='sum')
    if out.dim() > 1 and out.size(1) > 1:                 # multiclass
        tgt = full.argmax(dim=1).detach()
        return F.cross_entropy(out, tgt, reduction='sum')
    tgt = (torch.sigmoid(full.view(-1)) > 0.5).float().detach()   # binary
    return F.binary_cross_entropy_with_logits(out.view(-1), tgt, reduction='sum')


def _explain_batch(model, chunk: List[Data], device, task_type, epochs, lr,
                   coeff_size, coeff_ent, seed, gi0) -> Dict[int, torch.Tensor]:
    """Optimize one disconnected batch; return {global_graph_idx: [N_i] node att}."""
    batch = Batch.from_data_list(chunk).to(device)
    x = batch.x.float()
    ei = batch.edge_index
    bvec = batch.batch
    n2m = getattr(batch, 'nodes_to_motifs', None)
    K = batch.num_graphs
    N = x.size(0)

    with torch.no_grad():                                  # frozen full-graph target
        full = _forward_logits(model, x, ei, bvec, n2m)

    counts = scatter(torch.ones(N, device=device), bvec, dim=0,
                     dim_size=K, reduce='sum').long()      # [K] nodes per graph
    # Per-graph mask init seeded by GLOBAL index → identical to the sequential run.
    mask = torch.empty(N, 1, device=device)
    off = 0
    for k in range(K):
        nk = int(counts[k].item())
        gen = torch.Generator().manual_seed(seed * 1000003 + (gi0 + k))
        mask[off:off + nk] = (torch.randn(nk, 1, generator=gen) * 0.1).to(device)
        off += nk
    mask.requires_grad_(True)
    opt = torch.optim.Adam([mask], lr=lr)

    for _ in range(epochs):
        opt.zero_grad()
        h = x * mask.sigmoid()
        out = _forward_logits(model, h, ei, bvec, n2m)
        ploss = _pred_loss_sum(out, full, task_type)
        m = mask.sigmoid().view(-1)
        # PER-GRAPH segment means (the anti-gradient-mixing step), then summed.
        size_per = scatter(m, bvec, dim=0, dim_size=K, reduce='mean')
        ent = -m * torch.log(m + _EPS) - (1 - m) * torch.log(1 - m + _EPS)
        ent_per = scatter(ent, bvec, dim=0, dim_size=K, reduce='mean')
        loss = ploss + coeff_size * size_per.sum() + coeff_ent * ent_per.sum()
        loss.backward()
        opt.step()

    att = mask.sigmoid().detach().view(-1).cpu()
    out_atts: Dict[int, torch.Tensor] = {}
    off = 0
    for k in range(K):
        nk = int(counts[k].item())
        out_atts[gi0 + k] = att[off:off + nk].clone()
        off += nk
    return out_atts


def _aggregate_motif_scores(node_atts: Dict[int, torch.Tensor],
                            graphs: List[Data]) -> NodeScoreResult:
    """Node→motif reduction identical to run_gnnexplainer: per graph, mean/max of
    the node att over each motif's atoms; then average across graphs."""
    mean_sum: Dict[int, float] = {}; mean_cnt: Dict[int, int] = {}
    max_sum: Dict[int, float] = {}; max_cnt: Dict[int, int] = {}
    for gi, att in node_atts.items():
        n2m = getattr(graphs[gi], 'nodes_to_motifs', None)
        if n2m is None:
            continue
        n2m = n2m.view(-1).cpu()
        att = att.view(-1)
        for mid in n2m[n2m >= 0].unique().tolist():
            vals = att[n2m == mid]
            if vals.numel() == 0:
                continue
            mid = int(mid)
            mean_sum[mid] = mean_sum.get(mid, 0.0) + float(vals.mean())
            mean_cnt[mid] = mean_cnt.get(mid, 0) + 1
            max_sum[mid] = max_sum.get(mid, 0.0) + float(vals.max())
            max_cnt[mid] = max_cnt.get(mid, 0) + 1
    return {
        'mean': {m: mean_sum[m] / mean_cnt[m] for m in mean_sum if mean_cnt[m]},
        'max':  {m: max_sum[m] / max_cnt[m] for m in max_sum if max_cnt[m]},
    }


def run_gnnexplainer_batched(
    model: torch.nn.Module,
    data_list: List[Data],
    vocab,
    device: torch.device,
    task_type: str = 'BinaryClass',
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 64,
    coeff_size: float = 1.0,
    coeff_ent: float = 0.1,
    seed: int = 0,
    max_graphs: Optional[int] = None,
    return_node_atts: bool = False,
    verbose: bool = True,
) -> NodeScoreResult:
    """GPU-batched GNNExplainer. Same return contract as ``run_gnnexplainer``
    (``{'mean':…, 'max':…}`` + optional per-graph node atts). ``batch_size=1``
    reproduces the sequential baseline exactly (given the same ``seed``)."""
    model.eval().to(device)
    graphs = list(data_list[:max_graphs] if max_graphs else data_list)
    # Same refusal contract as the sequential run_gnnexplainer: integer atom
    # features (OGB AtomEncoder → nn.Embedding indices) cannot be multiplied by a
    # soft mask, so we do NOT silently float-mask garbage — raise and let the caller
    # record GNNExplainer as unavailable (N/A on OGB, e.g. molbace), never substitute.
    if graphs and not graphs[0].x.is_floating_point():
        raise RuntimeError(
            "GNNExplainer(batched) unavailable: integer atom features (OGB "
            "AtomEncoder) cannot be float-masked; refusing to substitute.")
    node_atts: Dict[int, torch.Tensor] = {}
    for start in range(0, len(graphs), batch_size):
        chunk = graphs[start:start + batch_size]
        node_atts.update(_explain_batch(
            model, chunk, device, task_type, epochs, lr,
            coeff_size, coeff_ent, seed, gi0=start))
        if verbose and start and start % (batch_size * 10) == 0:
            print(f'    GNNExplainer(batched): {start}/{len(graphs)} graphs ...')
    scores = _aggregate_motif_scores(node_atts, graphs)
    return (scores, node_atts) if return_node_atts else scores
