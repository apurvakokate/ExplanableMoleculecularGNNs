"""train.py — MotifSAT training loop."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from SharedModules.evaluation.metrics import evaluate_predictions
from SharedModules.evaluation.embedding_viz import EmbeddingVizLogger
from SharedModules.evaluation.wandb_logger import WandbLogger


def _task_loss(criterion, out, y, task_type):
    if task_type == 'MultiLabel':
        # out, y are [B, C]. Mask NaN targets PER ELEMENT (a molecule may have
        # observed labels for only some tasks), then average BCE over the
        # observed entries only. Masking per-row (valid.any) would still feed
        # NaN targets from unobserved columns into the loss → NaN gradient.
        if out.shape != y.shape:
            out = out.view_as(y)
        valid = ~torch.isnan(y)
        y0 = torch.nan_to_num(y, nan=0.0).float()
        per = F.binary_cross_entropy_with_logits(
            out, y0, pos_weight=getattr(criterion, 'pos_weight', None),
            reduction='none')
        per = per[valid]
        return per.mean() if per.numel() > 0 else (out.sum() * 0.0)
    valid = ~torch.isnan(y.view(-1))
    return criterion(out.view(-1)[valid], y.view(-1)[valid].float())


def _val_score(metrics: Dict, task_type: str) -> float:
    if task_type == 'Regression':
        return metrics.get('rmse', float('inf'))
    elif task_type == 'MultiLabel':
        return metrics.get('auc_mean', 0.0)
    return metrics.get('auc', 0.0)


@torch.no_grad()
def _val_task_loss(model, criterion, loader, device, task_type,
                   motif_lengths=None) -> float:
    """Mean validation TASK (prediction) loss — no info/motif regularisers.
    Used as the early-stop / LR-scheduler signal: val AUC saturates long before
    the GSAT/MotifSAT attention converges, so stopping on AUC undertrains the
    explainer (val loss keeps improving after AUC plateaus)."""
    model.eval()
    tot, n = 0.0, 0
    for data in loader:
        data = data.to(device)
        logits, _, _ = model(
            data.x, data.edge_index, data.batch, data.nodes_to_motifs,
            getattr(data, 'edge_attr', None), epoch=1, motif_lengths=motif_lengths)
        tot += float(_task_loss(criterion, logits, data.y, task_type).item())
        n += 1
    return tot / max(n, 1)


def train_one_epoch(
    model,
    criterion,
    optimizer,
    loader,
    device,
    task_type: str,
    epoch: int,
    motif_lengths: Optional[list] = None,
    clip_grad: float = 0.0,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    n = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        _edge_attr = getattr(data, 'edge_attr', None)
        if _edge_attr is None and os.environ.get('MOTIFSAT_VERIFY_FIXES') == '1':
            print('  [FIX#3 active] batch had no edge_attr; '
                  'used getattr fallback (None) instead of crashing')
        logits, node_att, aux = model(
            data.x, data.edge_index, data.batch,
            data.nodes_to_motifs, _edge_attr,
            epoch=epoch, motif_lengths=motif_lengths,
        )

        task = _task_loss(criterion, logits, data.y, task_type)
        total, breakdown = model.compute_loss(
            task, aux, data.nodes_to_motifs, data.batch, motif_lengths
        )
        total.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        for k, v in breakdown.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1

    return {k: v / max(n, 1) for k, v in totals.items()}


def train_gsat(
    model,
    loaders: Dict,
    task_type: str,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    pos_weights: Optional[torch.Tensor] = None,
    motif_lengths: Optional[list] = None,
    patience: int = 20,
    min_epochs: int = 20,
    early_stop_metric: str = 'loss',
    clip_grad: float = 2.0,
    save_path: Optional[str] = None,
    verbose: bool = True,
    viz_logger: Optional['EmbeddingVizLogger'] = None,
    wandb_logger: Optional['WandbLogger'] = None,
    epoch_hook: Optional['Callable[[object, int], None]'] = None,
) -> Tuple[object, Dict]:
    """Full MotifSAT training loop.

    Returns (best_model, history).
    """
    model.to(device)

    if task_type in ('BinaryClass', 'MultiLabel'):
        pw = pos_weights.to(device) if pos_weights is not None else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    # Early-stop / LR-scheduler signal. 'loss' (default) = smoothed val task
    # loss: val AUC saturates well before the attention converges, so stopping
    # on AUC undertrains the explainer (memory: training length is the lever).
    _stop_on_loss = str(early_stop_metric).lower() == 'loss'
    scheduler = ReduceLROnPlateau(optimizer, patience=(5 if _stop_on_loss else 10),
                                  factor=0.5, min_lr=1e-5)

    best_val = float('inf') if task_type == 'Regression' else 0.0  # checkpoint metric
    best_stop = float('inf')                 # smoothed val loss (min is better)
    loss_queue: deque = deque(maxlen=5)
    no_improve = 0
    best_state = deepcopy(model.state_dict())
    history: Dict = {}

    for epoch in range(1, epochs + 1):
        # Anneal IB prior retention (info_loss only; Concrete temp stays fixed)
        model.anneal_r(epoch - 1)  # 0-indexed, matches official GSAT get_r(epoch)

        ep_losses = train_one_epoch(
            model, criterion, optimizer, loaders['train'], device,
            task_type, epoch, motif_lengths, clip_grad,
        )
        for k, v in ep_losses.items():
            history.setdefault(k, []).append(v)

        val_m = evaluate_predictions(model, loaders['valid'], device, task_type)
        val_score = _val_score(val_m, task_type)
        history.setdefault('val_metric', []).append(val_score)

        # Per-epoch explainability telemetry (auxiliary; never raises).
        if epoch_hook is not None:
            epoch_hook(model, epoch)

        # Smoothed validation task loss (early-stop / scheduler signal).
        vloss = _val_task_loss(model, criterion, loaders['valid'], device,
                               task_type, motif_lengths)
        loss_queue.append(vloss)
        smoothed_loss = sum(loss_queue) / len(loss_queue)
        history.setdefault('val_loss', []).append(smoothed_loss)

        # Checkpoint is ALWAYS the best val AUC (classification) / RMSE
        # (regression) snapshot, decoupled from the early-stop signal.
        is_better = (val_score < best_val if task_type == 'Regression'
                     else val_score > best_val)
        if is_better:
            best_val = val_score
            best_state = deepcopy(model.state_dict())

        if _stop_on_loss:
            scheduler.step(smoothed_loss)
            if smoothed_loss < best_stop - 1e-4:
                best_stop = smoothed_loss
                no_improve = 0
            else:
                no_improve += 1
        else:
            scheduler.step(val_score if task_type == 'Regression' else -val_score)
            no_improve = 0 if is_better else no_improve + 1

        # W&B logging
        if wandb_logger is not None:
            wandb_logger.log_epoch(
                epoch=epoch,
                train_losses=ep_losses,
                val_metrics=val_m,
                optimizer=optimizer,
                no_improve=no_improve,
                best_val=best_val,
                valid_loader=loaders['valid'],
                device=device,
            )

        if verbose and epoch % 10 == 0:
            loss_str = '  '.join(f'{k}={v:.4f}'
                                 for k, v in ep_losses.items() if k != 'total')
            print(f'  Epoch {epoch:4d}  r={model.r.item():.3f}  '
                  f'{loss_str}  val={val_score:.4f}  best={best_val:.4f}  '
                  f'patience={no_improve}/{patience}')

        # Motif embedding PCA visualisation
        if viz_logger is not None:
            viz_logger.log(loaders['valid'], epoch)

        if epoch >= min_epochs and no_improve >= patience:
            if verbose:
                print(f'  Early stopping at epoch {epoch}')
            break

    model.load_state_dict(best_state)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'epoch': epoch, 'model_state_dict': best_state,
                    'best_val': best_val}, save_path)
    return model, history
