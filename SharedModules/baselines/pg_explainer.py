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
loop before ``explainer(...)`` can return edge masks.  Training must use
**one graph at a time** — batched graphs make the edge-size penalty sum over
the whole batch while only one graph's loss is optimised, collapsing masks to 0.
"""

from __future__ import annotations

import copy
import statistics
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

NodeScoreResult = Dict[str, Dict[int, float]]

# PGExplainer's edge-mask is regularised by an ``edge_size`` sparsity penalty
# (loss += edge_size * sum(sigmoid(mask))). When the prediction-loss gradient is
# weak — e.g. the model is confident regardless of masking, the common case for
# the phenomenon/model-prediction target — this penalty dominates and drives the
# WHOLE mask to 0 (mask collapse → near-constant per-motif scores → NaN
# score-vs-impact). We retry with a progressively smaller ``edge_size`` so the
# prediction loss has room to keep informative edges, and keep the first
# non-collapsed solution. Default PyG edge_size is 0.05 (first entry).
_PGEX_EDGE_SIZE_SCHEDULE = (0.05, 0.01, 0.002, 0.0005)
# A mean-pooled per-motif score set with std below this is treated as collapsed.
_PGEX_COLLAPSE_STD = 1e-6


def _scores_collapsed(scores: NodeScoreResult,
                      thresh: float = _PGEX_COLLAPSE_STD) -> bool:
    """True if the (mean-pooled) per-motif scores are effectively constant."""
    vals = list((scores or {}).get('mean', {}).values())
    if len(vals) < 2:
        return False  # too few motifs to judge — not treated as collapse
    return statistics.pstdev(vals) < thresh


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
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    else:
        batch = batch.to(device)
    mk = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in model_kwargs.items()
    }
    if explain_model:
        tgt = _model_graph_target(
            wrapped, x, edge_index, batch, task_type, **mk)
    else:
        tgt = _graph_target(data).to(device)
    return tgt.to(device)


def _aggregate_motif_scores(
    test_list: List[Data],
    device: torch.device,
    edge_masks_fn,
    max_graphs: Optional[int],
    return_node_atts: bool = False,
) -> NodeScoreResult:
    """Shared node→motif reduction given a per-graph edge-mask callback."""
    mean_sum: Dict[int, float] = {}
    mean_cnt: Dict[int, int] = {}
    max_sum: Dict[int, float] = {}
    max_cnt: Dict[int, int] = {}
    # Per-graph node attributions (gi -> [N]) for the per-instance correlation.
    node_atts: Dict[int, torch.Tensor] = {}

    graphs = test_list[:max_graphs] if max_graphs else test_list
    for gi, data in enumerate(graphs):
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
        node_atts[gi] = node_score.clone()

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

    scores = {
        'mean': {mid: mean_sum[mid] / mean_cnt[mid]
                 for mid in mean_sum if mean_cnt[mid] > 0},
        'max': {mid: max_sum[mid] / max_cnt[mid]
                for mid in max_sum if max_cnt[mid] > 0},
    }
    return (scores, node_atts) if return_node_atts else scores


def run_pgexplainer(
    model: torch.nn.Module,
    loaders: Dict,
    test_list: List[Data],
    vocab,
    device: torch.device,
    task_type: str = 'BinaryClass',
    epochs: int = 30,
    max_graphs: Optional[int] = None,
    max_train_graphs: Optional[int] = None,
    explain_model: bool = True,
    return_node_atts: bool = False,
) -> NodeScoreResult:
    """Per-motif importance scores from PGExplainer edge masks.

    Returns
    -------
    dict with keys 'mean' and 'max', each mapping motif_id -> float.
    When ``return_node_atts`` is set, returns ``(scores, node_atts)`` where
    ``node_atts`` maps graph index -> per-node attribution (for the per-instance
    score-vs-impact correlation).
    """
    model.eval()
    model.to(device)

    # Per-graph node attributions captured from the winning attempt.
    _captured: Dict[str, Dict[int, torch.Tensor]] = {}
    def _ret(sc):
        return (sc, _captured.get('node_atts', {})) if return_node_atts else sc

    try:
        from torch_geometric.explain import Explainer, PGExplainer as _PGEx
    except ImportError:
        print('  [warn] PGExplainer requires PyG >= 2.3; skipping (no gradient '
              'fallback — PGExplainer results must be genuine PGExplainer).')
        return _ret({'mean': {}, 'max': {}})

    class _Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self._inner = inner

        def forward(self, x, edge_index, batch=None, **kwargs):
            n2m = kwargs.get('nodes_to_motifs')
            out = self._inner(x, edge_index, batch, n2m)
            return out[0] if isinstance(out, (tuple, list)) else out

    # One training+scoring attempt at a given edge_size sparsity coefficient.
    # Each attempt gets its OWN fresh model copy: PyG's explainer instruments the
    # model in place (MessagePassing.explain rewrites propagate/inspector), so a
    # retry on the same object would be contaminated by the previous attempt.
    def _attempt(edge_size: float):
        m = copy.deepcopy(model).to(device)
        m.eval()
        wrapped = _Wrapper(m).to(device)
        explainer = Explainer(
            model=wrapped,
            algorithm=_PGEx(epochs=epochs, lr=0.003, edge_size=edge_size),
            explanation_type='phenomenon',
            edge_mask_type='object',
            node_mask_type=None,
            model_config=dict(
                mode=_pg_model_mode(task_type),
                task_level='graph',
                return_type='raw',
            ),
        )
        explainer.algorithm.to(device)
        train_ok, train_fail, last_err = _train_pgexplainer(
            explainer, wrapped, loaders, device, epochs, task_type, explain_model,
            max_train_graphs=max_train_graphs)
        if train_ok == 0:
            return None, (last_err or 'no successful train steps'), train_fail

        def _explain_graph(data: Data) -> Optional[torch.Tensor]:
            data = data.to(device)
            n = data.x.size(0)
            batch = torch.zeros(n, dtype=torch.long, device=device)
            kwargs = {}
            n2m = getattr(data, 'nodes_to_motifs', None)
            if n2m is not None:
                kwargs['nodes_to_motifs'] = n2m.to(device)
            target = _pg_target(
                wrapped, data, device, task_type, explain_model,
                batch=batch, **kwargs)
            expl = explainer(
                data.x, data.edge_index, batch=batch, target=target,
                index=0, **kwargs)
            return expl.edge_mask.detach().cpu()

        scores, _na = _aggregate_motif_scores(
            test_list, device, edge_masks_fn=_explain_graph, max_graphs=max_graphs,
            return_node_atts=True)
        _captured['node_atts'] = _na
        return scores, None, train_fail

    print(f'    Training PGExplainer ({epochs} epochs) ...')
    print('    PGExplainer target: '
          + ('model predictions (phenomenon API workaround)' if explain_model
             else 'ground-truth graph labels (phenomenon)'))

    last_scores: NodeScoreResult = {'mean': {}, 'max': {}}
    schedule = _PGEX_EDGE_SIZE_SCHEDULE
    for i, edge_size in enumerate(schedule):
        try:
            scores, err, train_fail = _attempt(edge_size)
        except Exception as e:
            print(f'  [warn] PGExplainer attempt (edge_size={edge_size}) failed '
                  f'({e}); skipping — no gradient fallback.')
            return _ret({'mean': {}, 'max': {}})
        if scores is None:  # training produced no successful steps
            print(f'  [warn] PGExplainer training failed ({err}); skipping — no '
                  f'gradient fallback (would masquerade as PGExplainer).')
            return _ret({'mean': {}, 'max': {}})
        if train_fail:
            print(f'    PGExplainer (edge_size={edge_size}): {train_fail} train step(s) skipped/failed')
        if not (scores['mean'] or scores['max']):
            last_scores = scores
            continue
        last_scores = scores
        if not _scores_collapsed(scores):
            if i > 0:
                print(f'    PGExplainer recovered from mask collapse at '
                      f'edge_size={edge_size} (default {schedule[0]}).')
            return _ret(scores)
        # collapsed → retry with a smaller sparsity penalty if any remain
        if i < len(schedule) - 1:
            print(f'    [info] PGExplainer mask collapsed at edge_size={edge_size}; '
                  f'retrying with edge_size={schedule[i + 1]}.')

    # Every edge_size in the schedule collapsed. Return the genuine (collapsed)
    # PGExplainer scores so it is honestly recorded and flagged downstream
    # (_warn_if_collapsed in run_vanilla) — never a gradient-saliency substitute.
    print(f'  [warn] PGExplainer mask collapsed at every edge_size in '
          f'{schedule} — reporting the collapsed PGExplainer result (no gradient '
          f'fallback). Score-vs-impact will be NaN for this run.')
    return _ret(last_scores)


def _iter_train_graphs(loaders: Dict, max_graphs: Optional[int] = None):
    """Yield individual graphs from the train loader (never batched PyG Batch)."""
    ds = loaders['train'].dataset
    n = len(ds)
    if max_graphs is not None and max_graphs < n:
        idx = torch.randperm(n)[:max_graphs].tolist()
    else:
        idx = list(range(n))
    for i in idx:
        yield ds[int(i)]


def _train_pgexplainer(
    explainer,
    wrapped: torch.nn.Module,
    loaders: Dict,
    device: torch.device,
    epochs: int,
    task_type: str,
    explain_model: bool,
    max_train_graphs: Optional[int] = None,
) -> Tuple[int, int, Optional[str]]:
    """Run PGExplainer.algorithm.train for every epoch (required by PyG).

    PGExplainer must be trained on **single graphs**.  Batched training applies
    edge-size regularisation over every edge in the batch while only one graph's
    prediction loss is optimised, which drives all edge masks to ~0.
    """
    train_ok = 0
    train_fail = 0
    last_err: Optional[str] = None

    for epoch in range(epochs):
        for data in _iter_train_graphs(loaders, max_train_graphs):
            if data.x is None:
                continue
            data = data.to(device)
            model_kwargs = {}
            n2m = getattr(data, 'nodes_to_motifs', None)
            if n2m is not None:
                model_kwargs['nodes_to_motifs'] = n2m.to(device)

            n = data.x.size(0)
            batch_vec = torch.zeros(n, dtype=torch.long, device=device)
            target = _pg_target(
                wrapped, data, device, task_type, explain_model,
                batch=batch_vec, **model_kwargs)

            try:
                explainer.algorithm.train(
                    epoch, wrapped,
                    data.x, data.edge_index,
                    target=target,
                    index=None,
                    batch=batch_vec,
                    **model_kwargs,
                )
                train_ok += 1
            except Exception as e:
                train_fail += 1
                last_err = str(e)
                if train_fail <= 2:
                    print(f'      [warn] PGExplainer train epoch {epoch}: {e}')

    return train_ok, train_fail, last_err

# NOTE: the former ``_gradient_fallback`` (gradient-saliency substitute) was
# removed intentionally. PGExplainer must never silently return gradient-saliency
# scores under the "pgexplainer" label — on failure or a degenerate/collapsed
# mask it now returns empty/genuine PGExplainer results instead (see run_pgexplainer).
