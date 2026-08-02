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
    dict with keys 'mean' and 'max' (motif_id -> float); with ``return_node_atts``
    also returns ``(scores, node_atts)``.
    """
    empty = {'mean': {}, 'max': {}}
    fit = fit_pgexplainer(
        model, loaders, test_list, device, task_type, epochs=epochs,
        max_graphs=max_graphs, max_train_graphs=max_train_graphs,
        explain_model=explain_model, verbose=True)
    if fit is None:
        return (empty, {}) if return_node_atts else empty
    # fit_pgexplainer already scored the ref split (== test_list) for its collapse
    # check and stashed the result — reuse it instead of scoring test again.
    scores = fit.get('ref_scores') or {'mean': {}, 'max': {}}
    atts = fit.get('ref_atts', {})
    if not (scores['mean'] or scores['max']):
        scores, atts = score_pgexplainer(
            fit, test_list, device, max_graphs=max_graphs, return_node_atts=True)
    return (scores, atts) if return_node_atts else scores


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


# ─────────────────────────────────────────────────────────────────────────────
# Split-aware fit / score (train-on-train once, score-per-split)
#
# ``run_pgexplainer`` fits the explainer on the train loader and scores test in
# one shot. A train/val/test evaluation needs the SAME fitted explainer scored on
# each split (and saved as an artifact). These two functions expose exactly that:
# ``fit_pgexplainer`` runs the edge_size collapse-retry training loop once and
# returns the winning fitted explainer; ``score_pgexplainer`` scores any split.
# Training still uses ``loaders['train']`` (via ``_train_pgexplainer``), so the
# explainer is always trained on TRAIN regardless of which split is scored.
# ─────────────────────────────────────────────────────────────────────────────

class _PGWrapper(torch.nn.Module):
    """Module-level twin of run_pgexplainer's inner _Wrapper (so fit/score can
    share it). Routes nodes_to_motifs through and returns the logits tensor."""
    def __init__(self, inner):
        super().__init__()
        self._inner = inner

    def forward(self, x, edge_index, batch=None, **kwargs):
        n2m = kwargs.get('nodes_to_motifs')
        out = self._inner(x, edge_index, batch, n2m)
        return out[0] if isinstance(out, (tuple, list)) else out


def _pgex_build(model, edge_size, epochs, task_type, device):
    """Build a fresh Explainer+PGExplainer on a deepcopy of the model (PyG
    instruments the model in place, so every attempt needs its own copy)."""
    from torch_geometric.explain import Explainer, PGExplainer as _PGEx
    m = copy.deepcopy(model).to(device)
    m.eval()
    wrapped = _PGWrapper(m).to(device)
    explainer = Explainer(
        model=wrapped,
        algorithm=_PGEx(epochs=epochs, lr=0.003, edge_size=edge_size),
        explanation_type='phenomenon',
        edge_mask_type='object',
        node_mask_type=None,
        model_config=dict(
            mode=_pg_model_mode(task_type), task_level='graph', return_type='raw'),
    )
    explainer.algorithm.to(device)
    return explainer, wrapped


def _pgex_explain_graph(explainer, wrapped, data, device, task_type, explain_model):
    data = data.to(device)
    n = data.x.size(0)
    batch = torch.zeros(n, dtype=torch.long, device=device)
    kwargs = {}
    n2m = getattr(data, 'nodes_to_motifs', None)
    if n2m is not None:
        kwargs['nodes_to_motifs'] = n2m.to(device)
    target = _pg_target(
        wrapped, data, device, task_type, explain_model, batch=batch, **kwargs)
    expl = explainer(data.x, data.edge_index, batch=batch, target=target,
                     index=0, **kwargs)
    return expl.edge_mask.detach().cpu()


def fit_pgexplainer(
    model: torch.nn.Module,
    loaders: Dict,
    ref_list: List[Data],
    device: torch.device,
    task_type: str = 'BinaryClass',
    epochs: int = 30,
    max_graphs: Optional[int] = None,
    max_train_graphs: Optional[int] = None,
    explain_model: bool = True,
    verbose: bool = True,
) -> Optional[Dict]:
    """Train PGExplainer ONCE on ``loaders['train']``, selecting the edge_size that
    avoids mask collapse (judged on ``ref_list`` — pass the test split). Returns a
    fit dict ``{explainer, wrapped, edge_size, task_type, explain_model}`` (score
    it per split with ``score_pgexplainer``), or None if PyG is unavailable / no
    train step succeeded. On all-collapse it returns the (honest) collapsed fit."""
    try:
        from torch_geometric.explain import Explainer, PGExplainer as _PGEx  # noqa: F401
    except ImportError:
        if verbose:
            print('  [warn] PGExplainer requires PyG >= 2.3; skipping (no fallback).')
        return None

    # The retry loop distinguishes two failure classes on purpose:
    #   • BUILD/TRAIN failure (an exception, or zero successful train steps) is HARD
    #     — a broken model/data/PyG-API is not fixed by a different edge_size, so we
    #     abort with None rather than burn three more futile training passes.
    #   • SCORE/COLLAPSE is SOFT — empty *or* collapsed masks retry at a smaller
    #     edge_size if any remain, else we keep the last honest fit. A scoring
    #     *exception* must NOT discard an already-trained explainer, so it only skips
    #     to the next edge_size (never returns None mid-loop).
    schedule = _PGEX_EDGE_SIZE_SCHEDULE
    last_fit = None
    for i, edge_size in enumerate(schedule):
        try:                                          # build + train — HARD failures
            explainer, wrapped = _pgex_build(model, edge_size, epochs, task_type, device)
            train_ok, train_fail, last_err = _train_pgexplainer(
                explainer, wrapped, loaders, device, epochs, task_type,
                explain_model, max_train_graphs=max_train_graphs)
        except Exception as e:
            if verbose:
                print(f'  [warn] PGExplainer build/train failed (edge_size={edge_size}): '
                      f'{e} — aborting (no gradient fallback).')
            return None
        if train_ok == 0:
            if verbose:
                print(f'  [warn] PGExplainer training produced no successful steps '
                      f'({last_err}) — aborting (no gradient fallback).')
            return None
        if train_fail and verbose:
            print(f'    PGExplainer (edge_size={edge_size}): {train_fail} train step(s) skipped/failed')

        fit = {'explainer': explainer, 'wrapped': wrapped, 'edge_size': edge_size,
               'task_type': task_type, 'explain_model': explain_model}
        last_fit = fit                                # honest fallback if all collapse

        try:                                          # score ref_list — SOFT failures
            sc, ref_atts = score_pgexplainer(fit, ref_list, device,
                                             max_graphs=max_graphs, return_node_atts=True)
        except Exception as e:
            if verbose:
                print(f'  [warn] PGExplainer collapse-check scoring failed at '
                      f'edge_size={edge_size} ({e}); trying next edge_size.')
            continue
        fit['ref_scores'], fit['ref_atts'] = sc, ref_atts

        # Good masks → done. Empty or collapsed → retry at a smaller edge_size if any
        # remain (both are the same "no usable signal" outcome); else keep last_fit.
        if (sc['mean'] or sc['max']) and not _scores_collapsed(sc):
            if i > 0 and verbose:
                print(f'    PGExplainer recovered from mask collapse at '
                      f'edge_size={edge_size} (default {schedule[0]}).')
            return fit
        if i < len(schedule) - 1 and verbose:
            _why = 'empty' if not (sc['mean'] or sc['max']) else 'collapsed'
            print(f'    [info] PGExplainer masks {_why} at edge_size={edge_size}; '
                  f'retrying with edge_size={schedule[i + 1]}.')

    if verbose:
        print(f'  [warn] PGExplainer masks empty/collapsed at every edge_size in '
              f'{schedule} — returning the last fit (honest; score-vs-impact NaN).')
    return last_fit


def score_pgexplainer(
    fit: Optional[Dict],
    split_list: List[Data],
    device: torch.device,
    max_graphs: Optional[int] = None,
    return_node_atts: bool = False,
) -> NodeScoreResult:
    """Score a split with a PRE-FIT PGExplainer (from ``fit_pgexplainer``). Returns
    ``{'mean':…, 'max':…}`` (and per-graph node atts when requested), consistent
    with ``run_pgexplainer``."""
    empty = {'mean': {}, 'max': {}}
    if fit is None:
        return (empty, {}) if return_node_atts else empty

    def _eg(data):
        return _pgex_explain_graph(
            fit['explainer'], fit['wrapped'], data, device,
            fit['task_type'], fit['explain_model'])

    return _aggregate_motif_scores(
        split_list, device, edge_masks_fn=_eg, max_graphs=max_graphs,
        return_node_atts=return_node_atts)


# NOTE: the former ``_gradient_fallback`` (gradient-saliency substitute) was
# removed intentionally. PGExplainer must never silently return gradient-saliency
# scores under the "pgexplainer" label — on failure or a degenerate/collapsed
# mask it now returns empty/genuine PGExplainer results instead (see run_pgexplainer).
