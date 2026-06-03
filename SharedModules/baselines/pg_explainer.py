"""pg_explainer.py — PGExplainer motif-level importance scores.

Returns both mean-aggregated and max-aggregated node scores per motif.

Edge -> node conversion
-----------------------
PGExplainer produces per-edge masks.
node_score[i] = mean of edge_mask values for all edges incident to node i.
Mean (not sum) avoids inflating high-degree hub nodes.

Node -> motif aggregation (per graph g, motif type m)
------------------------------------------------------
local_mean(m, g) = mean(node_score[nodes where n2m == m])
local_max(m, g)  = max(node_score[nodes where n2m == m])

Across all graphs:
score_mean(m) = mean_g local_mean(m, g)
score_max(m)  = mean_g local_max(m, g)

explanation_type='phenomenon' is used: PGExplainer explains ground-truth
labels. target=data.y is passed during both training and inference.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch_geometric.data import Data

NodeScoreResult = Dict[str, Dict[int, float]]


def run_pgexplainer(
    model: torch.nn.Module,
    loaders: Dict,
    test_list: List[Data],
    vocab,
    device: torch.device,
    task_type: str = 'BinaryClass',
    epochs: int = 30,
    max_graphs: Optional[int] = 200,
) -> NodeScoreResult:
    """Per-motif importance scores from PGExplainer edge masks.

    Returns
    -------
    dict with keys 'mean' and 'max', each mapping motif_id -> float.
    """
    model.eval()
    model.to(device)

    try:
        from torch_geometric.explain import Explainer, PGExplainer as _PGEx
    except ImportError:
        print('  [warn] PGExplainer requires PyG >= 2.3; falling back to gradient saliency')
        return _gradient_fallback(model, test_list, device)

    class _Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self._inner = inner

        def forward(self, x, edge_index, batch=None, **kwargs):
            out = self._inner(x, edge_index, batch)
            return out[0] if isinstance(out, (tuple, list)) else out

    wrapped = _Wrapper(model).to(device)

    try:
        # PGExplainer (PyG >= 2.3) only supports explanation_type='phenomenon',
        # meaning it explains ground-truth labels, not the model's own predictions.
        # We pass target=data.y during both training and inference.
        explainer = Explainer(
            model=wrapped,
            algorithm=_PGEx(epochs=epochs),
            explanation_type='phenomenon',
            edge_mask_type='object',
            node_mask_type=None,
            model_config=dict(
                mode='binary_classification' if task_type == 'BinaryClass' else 'regression',
                task_level='graph',
                return_type='raw',
            ),
        )

        # Use the DataLoader for batched training (much faster for large datasets
        # like Benzene with 10800 training molecules)
        print(f'    Training PGExplainer ({epochs} epochs) ...')
        for batch_data in loaders['train']:
            if batch_data.x is None:
                continue
            batch_data = batch_data.to(device)
            # target required for explanation_type='phenomenon'
            target = batch_data.y.view(-1).long()
            try:
                explainer(batch_data.x, batch_data.edge_index,
                          batch=batch_data.batch, target=target)
            except Exception:
                pass

        mean_sum: Dict[int, float] = {}
        mean_cnt: Dict[int, int]   = {}
        max_sum:  Dict[int, float] = {}
        max_cnt:  Dict[int, int]   = {}

        graphs = test_list[:max_graphs] if max_graphs else test_list

        for data in graphs:
            data = data.to(device)
            n    = data.x.size(0)
            n2m  = getattr(data, 'nodes_to_motifs', None)
            if n2m is None:
                continue

            batch  = torch.zeros(n, dtype=torch.long, device=device)
            target = data.y.view(-1).long()
            try:
                expl      = explainer(data.x, data.edge_index,
                                      batch=batch, target=target)
                edge_mask = expl.edge_mask.detach().cpu()
            except Exception:
                continue

            # Edge -> node: mean of incident edge mask values
            node_score = torch.zeros(n)
            node_cnt   = torch.zeros(n)
            src, dst   = data.edge_index.cpu()
            for i in range(edge_mask.size(0)):
                s, d = int(src[i]), int(dst[i])
                node_score[s] += float(edge_mask[i])
                node_score[d] += float(edge_mask[i])
                node_cnt[s]   += 1
                node_cnt[d]   += 1
            node_score = node_score / node_cnt.clamp(min=1)

            # Node -> motif: mean and max within each motif's nodes
            n2m_cpu = n2m.cpu()
            for mid in n2m_cpu[n2m_cpu >= 0].unique().tolist():
                scores_m = node_score[n2m_cpu == mid]
                if scores_m.numel() == 0:
                    continue

                local_mean = float(scores_m.mean().item())
                local_max  = float(scores_m.max().item())

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

    except Exception as e:
        print(f'  [warn] PGExplainer failed ({e}); using gradient saliency fallback')
        return _gradient_fallback(model, test_list, device)


def _gradient_fallback(model, test_list, device) -> NodeScoreResult:
    from .gnn_explainer import _gradient_saliency
    mean_sum: Dict[int, float] = {}
    mean_cnt: Dict[int, int]   = {}
    max_sum:  Dict[int, float] = {}
    max_cnt:  Dict[int, int]   = {}

    for data in test_list[:200]:
        data = data.to(device)
        n2m  = getattr(data, 'nodes_to_motifs', None)
        if n2m is None:
            continue
        sal = _gradient_saliency(model, data, device)
        if sal is None:
            continue
        n2m_cpu = n2m.cpu()
        for mid in n2m_cpu[n2m_cpu >= 0].unique().tolist():
            s = sal[n2m_cpu == mid]
            if s.numel() == 0:
                continue
            mean_sum[mid] = mean_sum.get(mid, 0.0) + float(s.mean())
            mean_cnt[mid] = mean_cnt.get(mid, 0)   + 1
            max_sum[mid]  = max_sum.get(mid, 0.0)  + float(s.max())
            max_cnt[mid]  = max_cnt.get(mid, 0)    + 1

    return {
        'mean': {mid: mean_sum[mid] / mean_cnt[mid]
                 for mid in mean_sum if mean_cnt[mid] > 0},
        'max':  {mid: max_sum[mid]  / max_cnt[mid]
                 for mid in max_sum  if max_cnt[mid]  > 0},
    }
