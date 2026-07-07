"""metrics.py — prediction performance metrics.

All functions operate on numpy arrays or PyTorch tensors and return dicts.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC.  Returns NaN if only one class present."""
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float('nan')


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def motif_score_stats(motif_scores) -> Dict[str, float]:
    """Distributional stats of the learned per-motif scores.

    Accepts a dict[motif_id -> score] or a flat iterable of scores. Returns
    min/max/mean/std/median/mode (plus count). Mode is computed on values
    rounded to 3 decimals so near-ties collapse sensibly for continuous scores;
    NaN if there is no repeated value. All NaN if fewer than one finite score.
    """
    if isinstance(motif_scores, dict):
        vals = list(motif_scores.values())
    elif motif_scores is None:
        vals = []
    else:
        vals = list(motif_scores)
    arr = np.array([float(v) for v in vals if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    nan = float('nan')
    if arr.size == 0:
        return {'score_min': nan, 'score_max': nan, 'score_mean': nan,
                'score_std': nan, 'score_median': nan, 'score_mode': nan,
                'score_count': 0}
    rounded = np.round(arr, 3)
    uniq, counts = np.unique(rounded, return_counts=True)
    mode_val = float(uniq[int(np.argmax(counts))]) if counts.max() > 1 else nan
    return {
        'score_min':    float(arr.min()),
        'score_max':    float(arr.max()),
        'score_mean':   float(arr.mean()),
        'score_std':    float(arr.std()),
        'score_median': float(np.median(arr)),
        'score_mode':   mode_val,
        'score_count':  int(arr.size),
    }


@torch.no_grad()
def evaluate_predictions(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    task_type: str,
    denorm: Optional[tuple] = None,
) -> Dict[str, float]:
    """Run model on a DataLoader and compute prediction metrics.

    Parameters
    ----------
    model : nn.Module
        Must support forward(data) or forward(x, edge_index, batch, node_to_motifs).
    loader : DataLoader
    device : torch.device
    task_type : str
        ``'BinaryClass'``, ``'Regression'``, or ``'MultiLabel'``.

    Returns
    -------
    dict with keys depending on task_type:
      BinaryClass:  auc
      Regression:   mae, rmse
      MultiLabel:   auc_mean, auc_per_task (list)
    """
    model.eval()
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for data in loader:
        data = data.to(device)
        out = _model_forward(model, data)
        all_preds.append(out.cpu().numpy())
        all_labels.append(data.y.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    if task_type == 'BinaryClass':
        scores = torch.sigmoid(torch.tensor(preds)).numpy().ravel()
        return {'auc': auc_score(labels.ravel(), scores)}

    elif task_type == 'Regression':
        preds_r = preds.ravel()
        labels_r = labels.ravel()
        # NORMALISED metrics: normalised model outputs vs normalised targets —
        # both live on the training z-score scale the model was trained on, so
        # the two sides are matched.
        out = {
            'mae':  mae_score(labels_r, preds_r),
            'rmse': rmse_score(labels_r, preds_r),
        }
        # ORIGINAL-UNIT metrics: inverse-transform BOTH the model outputs and the
        # targets back to real units (y = y_norm * std + mean) and score there,
        # so unnormalised targets are compared against unnormalised outputs — the
        # two sides stay on the same (original) scale. For MAE/RMSE the additive
        # mean cancels in the error so this equals rmse_norm * std, but computing
        # it explicitly keeps output/target scales matched and stays correct if a
        # non-affine original-unit metric is ever added.
        if denorm is not None:
            _mean, _std = float(denorm[0]), float(denorm[1])
            preds_o  = preds_r  * _std + _mean
            labels_o = labels_r * _std + _mean
            out['mae_orig']  = mae_score(labels_o, preds_o)
            out['rmse_orig'] = rmse_score(labels_o, preds_o)
        return out

    else:  # MultiLabel
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        scores_per = torch.sigmoid(torch.tensor(preds)).numpy()
        aucs = []
        for c in range(labels.shape[1]):
            valid = ~np.isnan(labels[:, c])
            if valid.sum() < 2:
                aucs.append(float('nan'))
            else:
                aucs.append(auc_score(labels[valid, c], scores_per[valid, c]))
        return {
            'auc_mean':     float(np.nanmean(aucs)),
            'auc_per_task': aucs,
        }


def _node_att_from_weights(nw: torch.Tensor) -> torch.Tensor:
    """Eval-side conversion of a masking argument to a node-attention tensor
    ``[N, 1]``. This lives in the evaluator (NOT the model) so models stay pure
    — they simply apply whatever node attention they are handed.

    Semantics:
      * bool mask — ``True`` marks the motif's atoms → weight 0 (suppressed),
        everything else weight 1;
      * float weights — used as-is (already the per-node weight vector, e.g. the
        model's learned attention or an explainer's scores with the target
        motif zeroed).
    """
    nw = (~nw).float() if nw.dtype == torch.bool else nw.float()
    return nw.view(-1, 1)


def _model_forward(model, data, node_weights=None) -> torch.Tensor:
    """Dispatch to the model's forward and return the raw logit tensor.

    Selects the calling convention from the forward *signature* (rather than
    catching TypeError and retrying), so a genuine bug inside the model's
    forward propagates instead of being masked as a "could not call" error.

    Uses out[0] (not `out, _ = ...`) so models returning 2-tuples (MOSE-GNN,
    VanillaGNN) and 3-tuples (GSAT/MotifSAT) both work without raising.

    Parameters
    ----------
    node_weights : Tensor [N] or [N, 1] (bool or float) or None
        When provided, passed to models that accept ``node_weights`` in forward
        (motif masking without graph removal).
    """
    import inspect

    candidate = dict(
        x=data.x,
        edge_index=data.edge_index,
        batch=getattr(data, 'batch', None),
        nodes_to_motifs=getattr(data, 'nodes_to_motifs', None),
        edge_attr=getattr(data, 'edge_attr', None),
    )
    if node_weights is not None:
        # Convert mask/weights → node attention HERE, so the model receives a
        # ready-to-apply [N,1] float and needs no mask-interpretation logic.
        candidate['node_weights'] = _node_att_from_weights(node_weights)

    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        params = None

    if params is not None and 'x' in params:
        has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
        kwargs = {
            k: v for k, v in candidate.items()
            if v is not None and (has_var_kw or k in params)
        }
        out = model(**kwargs)
    else:
        # Forward takes a Data object (or its signature is unavailable).
        out = model(data)

    return out[0] if isinstance(out, (tuple, list)) else out
