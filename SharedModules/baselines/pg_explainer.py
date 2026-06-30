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

explanation_type='phenomenon' is required by PyG's PGExplainer (model mode is
not supported). When explain_model=True (default), we pass the trained GNN's
predicted graph label as target — a standard workaround that aligns PGExplainer
with GNNExplainer's explanation_type='model' intent.

PyG >= 2.3 requires an explicit ``explainer.algorithm.train(epoch, ...)``
loop before ``explainer(...)`` can return edge masks; calling ``explainer``
during training does NOT train the parametric MLP.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

NodeScoreResult = Dict[str, Dict[int, float]]


def _pg_model_mode(task_type: str) -> str:
    if task_type == 'BinaryClass':
        return 'binary_classification'
    if task_type == 'MultiLabel':
        return 'multilabel_classification'
    return 'regression'


def _graph_target(data: Data) -> torch.Tensor:
    """Graph-level ground-truth label tensor (phenomenon / dataset label)."""
    y = data.y.view(-1)
    if y.dtype in (torch.float32, torch.float64):
        return y.long()
    return y


@torch.no_grad()
def _model_graph_target(
    wrapped: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    task_type: str,
    **model_kwargs,
) -> torch.Tensor:
    """Graph-level targets from the trained GNN (model-explanation workaround)."""
    wrapped.eval()
    out = wrapped(x, edge_index, batch, **model_kwargs)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if task_type == 'BinaryClass':
        return (torch.sigmoid(out.view(-1)) > 0.5).long()
    if task_type == 'MultiLabel':
        probs = torch.sigmoid(out)
        if probs.dim() == 1:
            return (probs > 0.5).long()
        return (probs > 0.5).long().view(probs.size(0), -1)
    return out.view(-1).float()


def _pg_target(
    wrapped: torch.nn.Module,
    data: Data,
    device: torch.device,
    task_type: str,
    explain_model: bool,
    batch: Optional[torch.Tensor] = None,
    **model_kwargs,
) -> torch.Tensor:
    """Target passed to PGExplainer train/explain (GT label or model prediction)."""
    if batch is None:
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    if explain_model:
        return _model_graph_target(
            wrapped, data.x, data.edge_index, batch, task_type, **model_kwargs)
    return _graph_target(data)


def _aggregate_motif_scores(
    test_list: List[Data],
    device: torch.device,
    edge_masks_fn,
    max_graphs: Optional[int],
) -> NodeScoreResult:
    """Shared node→motif reduction given a per-graph edge-mask callback."""
    mean_sum: Dict[int, float] = {}
    mean_cnt: Dict[int, int] = {}
    max_sum: Dict[int, float] = {}
    max_cnt: Dict[int, int] = {}

    graphs = test_list[:max_graphs] if max_graphs else test_list
    for data in graphs:
        data = data.to(device)
        n = data.x.size(0)
        n2m = getattr(data, 'nodes_to_motifs', None)
        if n2m is None:
            continue

        edge_mask = edge_masks_fn(data)
        if edge_mask is None:
            continue

        node_score = torch.zeros(n)
        node_cnt = torch.zeros(n)
        src, dst = data.edge_index.cpu()
        for i in range(edge_mask.size(0)):
            s, d = int(src[i]), int(dst[i])
            val = float(edge_mask[i])
            node_score[s] += val
            node_score[d] += val
            node_cnt[s] += 1
            node_cnt[d] += 1
        node_score = node_score / node_cnt.clamp(min=1)

        n2m_cpu = n2m.cpu()
        for mid in n2m_cpu[n2m_cpu >= 0].unique().tolist():
            scores_m = node_score[n2m_cpu == mid]
            if scores_m.numel() == 0:
                continue
            local_mean = float(scores_m.mean().item())
            local_max = float(scores_m.max().item())
            mean_sum[mid] = mean_sum.get(mid, 0.0) + local_mean
            mean_cnt[mid] = mean_cnt.get(mid, 0) + 1
            max_sum[mid] = max_sum.get(mid, 0.0) + local_max
            max_cnt[mid] = max_cnt.get(mid, 0) + 1

    return {
        'mean': {mid: mean_sum[mid] / mean_cnt[mid]
                 for mid in mean_sum if mean_cnt[mid] > 0},
        'max': {mid: max_sum[mid] / max_cnt[mid]
                for mid in max_sum if max_cnt[mid] > 0},
    }


def run_pgexplainer(
    model: torch.nn.Module,
    loaders: Dict,
    test_list: List[Data],
    vocab,
    device: torch.device,
    task_type: str = 'BinaryClass',
    epochs: int = 30,
    max_graphs: Optional[int] = None,
    explain_model: bool = True,
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
        return _gradient_fallback(model, test_list, device, max_graphs)

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
        explainer = Explainer(
            model=wrapped,
            algorithm=_PGEx(epochs=epochs, lr=0.003),
            explanation_type='phenomenon',
            edge_mask_type='object',
            node_mask_type=None,
            model_config=dict(
                mode=_pg_model_mode(task_type),
                task_level='graph',
                return_type='raw',
            ),
        )

        print(f'    Training PGExplainer ({epochs} epochs) ...')
        if explain_model:
            print('    PGExplainer target: model predictions '
                  '(phenomenon API workaround for model-level explanation)')
        else:
            print('    PGExplainer target: ground-truth graph labels (phenomenon)')
        train_ok, train_fail, last_err = _train_pgexplainer(
            explainer, wrapped, loaders, device, epochs, task_type, explain_model)
        if train_ok == 0:
            msg = last_err or 'no successful train steps'
            print(f'  [warn] PGExplainer training failed ({msg}); using gradient saliency fallback')
            return _gradient_fallback(model, test_list, device, max_graphs)
        if train_fail:
            print(f'    PGExplainer train: {train_ok} ok, {train_fail} skipped/failed')

        def _explain_graph(data: Data) -> Optional[torch.Tensor]:
            n = data.x.size(0)
            batch = torch.zeros(n, dtype=torch.long, device=device)
            kwargs = {}
            n2m = getattr(data, 'nodes_to_motifs', None)
            if n2m is not None:
                kwargs['nodes_to_motifs'] = n2m
            target = _pg_target(
                wrapped, data, device, task_type, explain_model,
                batch=batch, **kwargs)
            expl = explainer(
                data.x, data.edge_index,
                batch=batch,
                target=target,
                index=0,
                **kwargs,
            )
            return expl.edge_mask.detach().cpu()

        scores = _aggregate_motif_scores(
            test_list, device,
            edge_masks_fn=lambda data: _explain_graph(data),
            max_graphs=max_graphs,
        )
        if scores['mean'] or scores['max']:
            return scores
        print('  [warn] PGExplainer produced no motif scores after training; using gradient fallback')
        return _gradient_fallback(model, test_list, device, max_graphs)

    except Exception as e:
        print(f'  [warn] PGExplainer failed ({e}); using gradient saliency fallback')
        return _gradient_fallback(model, test_list, device, max_graphs)


def _train_pgexplainer(
    explainer,
    wrapped: torch.nn.Module,
    loaders: Dict,
    device: torch.device,
    epochs: int,
    task_type: str,
    explain_model: bool,
) -> Tuple[int, int, Optional[str]]:
    """Run PGExplainer.algorithm.train for every epoch (required by PyG)."""
    train_ok = 0
    train_fail = 0
    last_err: Optional[str] = None

    for epoch in range(epochs):
        for batch_data in loaders['train']:
            if batch_data.x is None:
                continue
            batch_data = batch_data.to(device)
            kwargs = {}
            n2m = getattr(batch_data, 'nodes_to_motifs', None)
            if n2m is not None:
                kwargs['nodes_to_motifs'] = n2m
            batch_vec = getattr(batch_data, 'batch', None)
            if batch_vec is not None:
                kwargs['batch'] = batch_vec

            target = _pg_target(
                wrapped, batch_data, device, task_type, explain_model,
                batch=batch_vec, **kwargs)

            n_graphs = int(getattr(batch_data, 'num_graphs', 1))
            indices = range(n_graphs) if n_graphs > 1 else [None]
            for g_idx in indices:
                try:
                    explainer.algorithm.train(
                        epoch, wrapped,
                        batch_data.x, batch_data.edge_index,
                        target=target,
                        index=g_idx,
                        **kwargs,
                    )
                    train_ok += 1
                except Exception as e:
                    train_fail += 1
                    last_err = str(e)
                    if train_fail <= 2:
                        print(f'      [warn] PGExplainer train epoch {epoch}: {e}')

    return train_ok, train_fail, last_err


def _gradient_fallback(
    model, test_list, device, max_graphs: Optional[int] = None,
) -> NodeScoreResult:
    from .gnn_explainer import _gradient_saliency

    mean_sum: Dict[int, float] = {}
    mean_cnt: Dict[int, int] = {}
    max_sum: Dict[int, float] = {}
    max_cnt: Dict[int, int] = {}

    graphs = test_list[:max_graphs] if max_graphs else test_list
    for data in graphs:
        data = data.to(device)
        n2m = getattr(data, 'nodes_to_motifs', None)
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
            mean_cnt[mid] = mean_cnt.get(mid, 0) + 1
            max_sum[mid] = max_sum.get(mid, 0.0) + float(s.max())
            max_cnt[mid] = max_cnt.get(mid, 0) + 1

    return {
        'mean': {mid: mean_sum[mid] / mean_cnt[mid]
                 for mid in mean_sum if mean_cnt[mid] > 0},
        'max': {mid: max_sum[mid] / max_cnt[mid]
                for mid in max_sum if max_cnt[mid] > 0},
    }
