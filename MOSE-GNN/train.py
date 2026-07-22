"""train.py — MOSE-GNN training loop with entropy + size regularisation.

Loss:
  L = task_loss(ŷ, y)
      + size_reg  · Σ_{m ∉ top-τ} σ(θ_m)     (sparsity)
      + ent_reg   · mean_m H(σ(θ_m))           (uncertainty push toward 0/1)

Where H(p) = -p log p - (1-p) log(1-p) is binary entropy.
"""

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


EPS = 1e-15


def mask_regularisation(
    motif_scores: torch.Tensor,    # sigmoid outputs, shape [M] or [M, C]
    size_reg: float,
    ent_reg: float,
    top_tau: int = 10,
) -> torch.Tensor:
    """Compute sparsity + entropy penalty on motif importance scores.

    Parameters
    ----------
    motif_scores : Tensor
        sigma(theta_m) values in [0, 1].  Flat or [M, C].
    size_reg : float
    ent_reg : float
    top_tau : int
        Top-τ motifs are excluded from the size penalty.

    Returns
    -------
    scalar loss tensor
    """
    s = motif_scores.view(-1)

    # Size loss: penalise all except the top-τ values
    tau = min(top_tau, s.numel())
    vals, _ = torch.sort(s, descending=True)
    penalized = vals[tau:]
    size_loss = size_reg * penalized.sum()

    # Entropy loss (binary): encourages confident 0/1 decisions
    ent = -s * torch.log(s + EPS) - (1 - s) * torch.log(1 - s + EPS)
    ent_loss = ent_reg * ent.mean()

    return size_loss + ent_loss


def _task_loss(
    criterion: nn.Module,
    out: torch.Tensor,
    y: torch.Tensor,
    task_type: str,
) -> torch.Tensor:
    if task_type == 'MultiLabel':
        # Mask NaN targets PER ELEMENT (not per row): a partially-labelled row
        # must keep its valid columns while ignoring the NaN ones. Selecting
        # whole rows with valid.any(dim=1) would still feed NaN targets to BCE
        # and produce a NaN loss. Compute the unreduced loss, zero out invalid
        # entries, and average over the valid count only.
        valid = ~torch.isnan(y)
        if not valid.any():
            # Keep the loss connected to the graph (out.sum()*0) so backward is a
            # no-op gradient rather than a disconnected leaf, which would have no
            # grad_fn and silently propagate nothing.
            return out.sum() * 0.0
        pw = getattr(criterion, 'pos_weight', None)
        per_elem = F.binary_cross_entropy_with_logits(
            out, torch.nan_to_num(y.float()), pos_weight=pw, reduction='none')
        return per_elem[valid].mean()
    valid = ~torch.isnan(y.view(-1))
    return criterion(out.view(-1)[valid], y.view(-1)[valid].float())


@torch.no_grad()
def _val_task_loss(
    model,
    criterion: nn.Module,
    loader,
    device: torch.device,
    task_type: str,
    ignore_unknowns: bool = False,
) -> float:
    """Mean validation TASK loss (no motif regularisation) — the paper's
    early-stop / LR-scheduler signal (smoothed over recent epochs)."""
    model.eval()
    tot, n = 0.0, 0
    for data in loader:
        data = data.to(device)
        out, _ = model(
            data.x, data.edge_index, data.batch,
            data.nodes_to_motifs, getattr(data, 'edge_attr', None),
            ignore_unknowns=ignore_unknowns,
        )
        tot += float(_task_loss(criterion, out, data.y, task_type).item())
        n += 1
    return tot / max(n, 1)


def train_one_epoch(
    model,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader,
    device: torch.device,
    task_type: str,
    size_reg: float = 0.0,
    ent_reg: float = 0.01,
    top_tau: int = 10,
    ignore_unknowns: bool = False,
    clip_grad: float = 0.0,
) -> Tuple[float, Optional[float]]:
    model.train()
    total_task = 0.0
    total_reg = 0.0
    n_batches = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        _edge_attr = getattr(data, 'edge_attr', None)
        if _edge_attr is None and os.environ.get('MOTIFSAT_VERIFY_FIXES') == '1':
            print('  [FIX#3 active] batch had no edge_attr; '
                  'used getattr fallback (None) instead of crashing')
        out, node_att = model(
            data.x, data.edge_index, data.batch,
            data.nodes_to_motifs, _edge_attr,
            ignore_unknowns=ignore_unknowns,
        )

        loss = _task_loss(criterion, out, data.y, task_type)

        # Regularisation on motif_params (if model has them)
        reg = torch.tensor(0.0, device=device)
        if hasattr(model, 'motif_params') and model.motif_params is not None:
            scores = model.motif_params.sigmoid()
            # Include unk_param in entropy regularisation
            if (hasattr(model, 'unk_param') and model.unk_param is not None):
                unk_s = model.unk_param.sigmoid().view(1, 1).expand(1, scores.size(-1))
                scores = torch.cat([scores, unk_s], dim=0)
            reg = mask_regularisation(scores, size_reg, ent_reg, top_tau)

        (loss + reg).backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_task += loss.item()
        total_reg += reg.item()
        n_batches += 1

    mean_task = total_task / max(n_batches, 1)
    mean_reg = total_reg / max(n_batches, 1)
    return mean_task, mean_reg


def train_mose_gnn(
    model,
    loaders: Dict,
    task_type: str,
    device: torch.device,
    epochs: int = 150,
    lr: float = 1e-3,
    explainer_lr: Optional[float] = None,
    gnn_lr: Optional[float] = None,
    weight_decay: float = 0.01,
    optimizer: str = 'adamw',
    early_stop_metric: str = 'loss',
    pos_weights: Optional[torch.Tensor] = None,
    size_reg: float = 0.0,
    ent_reg: float = 0.01,
    top_tau: int = 10,
    ignore_unknowns: bool = False,
    patience: int = 30,
    min_epochs: int = 20,
    clip_grad: float = 2.0,
    save_path: Optional[str] = None,
    verbose: bool = True,
    viz_logger: Optional['EmbeddingVizLogger'] = None,
    wandb_logger: Optional['WandbLogger'] = None,
    epoch_hook: Optional['Callable[[object, int], None]'] = None,
) -> Tuple[object, Dict]:
    """Full training loop with early stopping.

    Returns (best_model, history_dict).
    """
    model.to(device)

    if task_type in ('BinaryClass', 'MultiLabel'):
        pw = pos_weights.to(device) if pos_weights is not None else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        criterion = nn.MSELoss()

    # Optimizer choice: AdamW (decoupled weight decay, the paper default) or
    # Adam. Resolved from the ``optimizer`` string before the param-group build.
    _OptCls = torch.optim.AdamW if str(optimizer).lower() == 'adamw' else torch.optim.Adam

    # Two learning rates: the explainer (motif-importance params) and the GNN
    # backbone train at different speeds. When explainer_lr/gnn_lr are given we
    # build separate param groups; otherwise fall back to a single LR (`lr`).
    if explainer_lr is not None or gnn_lr is not None:
        _exp_lr = explainer_lr if explainer_lr is not None else lr
        _gnn_lr = gnn_lr if gnn_lr is not None else lr
        explainer_names = {'motif_params', 'unk_param'}
        exp_params, gnn_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # the explainer = top-level motif-importance params
            if name in explainer_names or name.split('.')[-1] in explainer_names:
                exp_params.append(p)
            else:
                gnn_params.append(p)
        groups = []
        if exp_params:
            groups.append({'params': exp_params, 'lr': _exp_lr})
        if gnn_params:
            groups.append({'params': gnn_params, 'lr': _gnn_lr})
        optimizer = _OptCls(groups, weight_decay=weight_decay)
        if verbose:
            print(f'  [opt] {_OptCls.__name__}(weight_decay={weight_decay})  '
                  f'explainer={_exp_lr} ({len(exp_params)} params)  '
                  f'gnn={_gnn_lr} ({len(gnn_params)} params)')
    else:
        optimizer = _OptCls(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    # LR scheduler & early-stop signal.
    #   'loss' (paper default): drive off SMOOTHED validation task loss
    #       (deque mean), scheduler patience 5.
    #   'auc'  (legacy): drive off the checkpoint metric (val AUC / -RMSE),
    #       scheduler patience 15.
    _stop_on_loss = str(early_stop_metric).lower() == 'loss'
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=(5 if _stop_on_loss else 15), min_lr=1e-5)

    best_val = float('inf') if task_type == 'Regression' else 0.0  # checkpoint metric
    best_stop = float('inf')                # smoothed val loss (min is better)
    loss_queue: deque = deque(maxlen=5)
    no_improve = 0
    best_state = deepcopy(model.state_dict())
    history: Dict = {'train_loss': [], 'reg_loss': [], 'val_metric': [],
                     'val_loss': []}

    for epoch in range(1, epochs + 1):
        t_loss, r_loss = train_one_epoch(
            model, criterion, optimizer, loaders['train'], device,
            task_type, size_reg, ent_reg, top_tau, ignore_unknowns, clip_grad,
        )
        history['train_loss'].append(t_loss)
        history['reg_loss'].append(r_loss)

        val_m = evaluate_predictions(model, loaders['valid'], device, task_type)
        val_score = _val_score(val_m, task_type)
        history['val_metric'].append(val_score)

        # Per-epoch explainability telemetry (auxiliary; never raises — see
        # TrainingExplTracker). Evaluated on the current (in-progress) model.
        if epoch_hook is not None:
            epoch_hook(model, epoch)

        # Smoothed validation task loss (paper early-stop / scheduler signal).
        vloss = _val_task_loss(model, criterion, loaders['valid'], device,
                               task_type, ignore_unknowns)
        loss_queue.append(vloss)
        smoothed_loss = sum(loss_queue) / len(loss_queue)
        history['val_loss'].append(smoothed_loss)

        # Checkpoint is ALWAYS the best validation AUC (classification) / RMSE
        # (regression) snapshot — matching the paper's best_model_acc used for
        # explanation — regardless of the early-stop signal.
        is_better = (val_score < best_val if task_type == 'Regression'
                     else val_score > best_val)
        if is_better:
            best_val = val_score
            best_state = deepcopy(model.state_dict())

        # Early-stop + LR schedule signal (decoupled from checkpoint selection).
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
                train_losses={'task': t_loss, 'reg': r_loss,
                              'total': t_loss + r_loss},
                val_metrics=val_m,
                optimizer=optimizer,
                no_improve=no_improve,
                best_val=best_val,
            )

        if verbose and epoch % 10 == 0:
            print(f'  Epoch {epoch:4d}  task={t_loss:.4f}  reg={r_loss:.4f}'
                  f'  val={val_score:.4f}  best={best_val:.4f}'
                  f'  patience={no_improve}/{patience}')

        # Motif embedding PCA visualisation
        if viz_logger is not None:
            if hasattr(model, 'get_motif_scores'):
                viz_logger.update_motif_scores(model.get_motif_scores())
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


def _val_score(metrics: Dict, task_type: str) -> float:
    if task_type == 'Regression':
        return metrics.get('rmse', float('inf'))
    elif task_type == 'MultiLabel':
        return metrics.get('auc_mean', 0.0)
    return metrics.get('auc', 0.0)
