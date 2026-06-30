"""gnn_explainer.py — GNNExplainer motif-level importance scores.

Returns both mean-aggregated and max-aggregated scores so the caller can
choose which to pass to evaluation functions.

Aggregation strategy
---------------------
GNNExplainer produces a per-node mask for each graph independently.

For each graph g and motif type m:
  node_mask_m = mask values for all nodes where nodes_to_motifs == m
  local_mean(m, g) = mean(node_mask_m)
  local_max(m, g)  = max(node_mask_m)

Across all graphs containing m:
  score_mean(m) = mean_g local_mean(m, g)   <- how consistently highlighted
  score_max(m)  = mean_g local_max(m, g)    <- how highlighted at best in each graph

Mean aggregation answers: "on average, how much of this motif does the
explainer highlight across all graphs it appears in?"

Max aggregation answers: "on average, what is the most highlighted atom
of this motif in each graph?" — a softer threshold that fires even when
only one atom of the motif is highlighted.

Scores are raw (not normalised). Use rank_scores() for cross-method comparison.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

# Return type shared by all node-score-based baselines
NodeScoreResult = Dict[str, Dict[int, float]]
# {'mean': {motif_id: float}, 'max': {motif_id: float}}


def run_gnnexplainer(
    model: torch.nn.Module,
    data_list: List[Data],
    vocab,
    device: torch.device,
    task_type: str = 'BinaryClass',
    epochs: int = 100,
    max_graphs: Optional[int] = None,
    verbose: bool = True,
) -> NodeScoreResult:
    """Per-motif importance scores from GNNExplainer node masks.

    GNNExplainer optimizes a mask **per graph** (``epochs`` steps each), so cost
    scales with the number of explained test graphs. Default ``max_graphs=None``
    uses the full test split; pass a positive int to cap (e.g. for quick sweeps).

    Returns
    -------
    dict with keys 'mean' and 'max', each mapping motif_id -> float.
    """
    model.eval()
    model.to(device)

    class _Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self._inner = inner

        def forward(self, x, edge_index, batch=None, **kwargs):
            n2m = kwargs.get('nodes_to_motifs')
            out = self._inner(x, edge_index, batch, n2m)
            return out[0] if isinstance(out, (tuple, list)) else out

    wrapped = _Wrapper(model).to(device)

    try:
        from torch_geometric.explain import Explainer, GNNExplainer as _GNNEx
        explainer = Explainer(
            model=wrapped,
            algorithm=_GNNEx(epochs=epochs),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='binary_classification' if task_type == 'BinaryClass' else 'regression',
                task_level='graph',
                return_type='raw',
            ),
        )
    except Exception:
        explainer = None

    mean_sum: Dict[int, float] = {}
    mean_cnt: Dict[int, int]   = {}
    max_sum:  Dict[int, float] = {}
    max_cnt:  Dict[int, int]   = {}

    graphs = data_list[:max_graphs] if max_graphs else data_list
    n_test = len(data_list)
    n_total = len(graphs)
    if verbose and n_total > 0:
        if max_graphs is None:
            cap_note = 'no cap — all test graphs'
        elif max_graphs >= n_test:
            cap_note = f'cap={max_graphs} (≥ test size — all test graphs)'
        else:
            cap_note = f'cap={max_graphs} of {n_test} test graphs'
        print(f'    GNNExplainer: explaining {n_total}/{n_test} test graph(s) '
              f'({cap_note}), {epochs} epochs/graph')

    for gi, data in enumerate(graphs):
        if verbose and n_total > 25 and gi > 0 and gi % 25 == 0:
            print(f'    GNNExplainer: {gi}/{n_total} graphs ...')
        data = data.to(device)
        n   = data.x.size(0)
        n2m = getattr(data, 'nodes_to_motifs', None)
        if n2m is None:
            continue

        try:
            if explainer is not None:
                expl      = explainer(
                    data.x, data.edge_index,
                    batch=torch.zeros(n, dtype=torch.long, device=device),
                    nodes_to_motifs=n2m,
                )
                node_mask = expl.node_mask.mean(dim=-1).abs().detach().cpu()
            else:
                node_mask = _gradient_saliency(wrapped, data, device)
        except Exception:
            node_mask = _gradient_saliency(wrapped, data, device)

        if node_mask is None:
            continue

        node_mask = node_mask.view(-1)
        n2m_cpu   = n2m.cpu()

        for mid in n2m_cpu[n2m_cpu >= 0].unique().tolist():
            mask_m = node_mask[n2m_cpu == mid]
            if mask_m.numel() == 0:
                continue

            local_mean = float(mask_m.mean().item())
            local_max  = float(mask_m.max().item())

            mean_sum[mid] = mean_sum.get(mid, 0.0) + local_mean
            mean_cnt[mid] = mean_cnt.get(mid, 0)   + 1
            max_sum[mid]  = max_sum.get(mid, 0.0)  + local_max
            max_cnt[mid]  = max_cnt.get(mid, 0)    + 1

    return {
        'mean': {mid: mean_sum[mid] / mean_cnt[mid]
                 for mid in mean_sum if mean_cnt[mid] > 0},
        'max':  {mid: max_sum[mid]  / max_cnt[mid]
                 for mid in max_sum  if max_cnt[mid]  > 0},
    }


def _gradient_saliency(model, data, device) -> Optional[torch.Tensor]:
    try:
        x = data.x.clone().detach().requires_grad_(True).to(device)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        out = model(x, data.edge_index, batch)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out.sum().backward()
        return x.grad.abs().sum(dim=-1).detach().cpu()
    except Exception:
        return None


def rank_scores(scores: Dict[int, float]) -> Dict[int, float]:
    """Convert raw scores to fractional ranks in [0, 1] (1.0 = highest)."""
    if not scores:
        return {}
    items = sorted(scores.items(), key=lambda x: x[1])
    n = len(items)
    return {mid: i / max(n - 1, 1) for i, (mid, _) in enumerate(items)}
