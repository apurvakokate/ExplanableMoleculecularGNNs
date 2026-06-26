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
        out = {
            'mae':  mae_score(labels_r, preds_r),
            'rmse': rmse_score(labels_r, preds_r),
        }
        # Optionally also report metrics in the ORIGINAL target units. With
        # z-score normalisation y_norm = (y - mean) / std, the error scales
        # exactly by std (the mean cancels in differences), so the
        # denormalised MAE/RMSE are the normalised values times std.
        if denorm is not None:
            _mean, _std = denorm
            out['mae_orig']  = out['mae']  * float(_std)
            out['rmse_orig'] = out['rmse'] * float(_std)
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


def _model_forward(model, data) -> torch.Tensor:
    """Try several calling conventions; return raw logit tensor.

    Uses out[0] pattern (not `out, _ = ...`) so that models returning
    2-tuples (MOSE-GNN, VanillaGNN) and 3-tuples (GSAT/MotifSAT) both
    work without raising ValueError.
    """
    edge_attr = getattr(data, 'edge_attr', None)
    nodes_to_motifs = getattr(data, 'nodes_to_motifs', None)
    errors: list = []

    attempts = (
        dict(x=data.x, edge_index=data.edge_index, batch=data.batch,
             nodes_to_motifs=nodes_to_motifs, edge_attr=edge_attr),
        dict(x=data.x, edge_index=data.edge_index, batch=data.batch,
             nodes_to_motifs=nodes_to_motifs),
    )
    for i, kwargs in enumerate(attempts, start=1):
        try:
            out = model(**{k: v for k, v in kwargs.items() if v is not None})
            return out[0] if isinstance(out, (tuple, list)) else out
        except TypeError as e:
            errors.append(f"attempt {i}: {e}")

    try:
        out = model(data)
        return out[0] if isinstance(out, (tuple, list)) else out
    except TypeError as e:
        errors.append(f"data object: {e}")

    raise TypeError(
        f"Could not call {model.__class__.__name__} forward. Tried:\n  "
        + "\n  ".join(errors)
    )
