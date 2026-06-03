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
    node_att_fn=None,
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
    node_att_fn : callable or None
        If provided, called as ``node_att_fn(data)`` and the result is passed
        as ``node_att`` to the model.  Used for GSAT-style models.

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
        return {
            'mae':  mae_score(labels_r, preds_r),
            'rmse': rmse_score(labels_r, preds_r),
        }

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


def _model_forward(model, data) -> torch.Tensor:
    """Try several calling conventions; return raw logit tensor.

    Uses out[0] pattern (not `out, _ = ...`) so that models returning
    2-tuples (MOSE-GNN, VanillaGNN) and 3-tuples (GSAT/MotifSAT) both
    work without raising ValueError.
    """
    # Pass edge_attr to match the training loop (critical for GAT / PNA)
    edge_attr = getattr(data, 'edge_attr', None)
    try:
        out = model(data.x, data.edge_index, data.batch,
                    data.nodes_to_motifs, edge_attr)
        return out[0] if isinstance(out, (tuple, list)) else out
    except TypeError:
        pass
    try:
        out = model(data.x, data.edge_index, data.batch, data.nodes_to_motifs)
        return out[0] if isinstance(out, (tuple, list)) else out
    except TypeError:
        pass
    try:
        out = model(data)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out
    except TypeError:
        pass
    return model(data.x, data.edge_index, data.batch)
