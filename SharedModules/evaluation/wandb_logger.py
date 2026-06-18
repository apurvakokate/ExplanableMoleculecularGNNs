"""wandb_logger.py — structured W&B metric logging for MOSE-GNN and MotifSAT.

Logs every epoch:
  train/task_loss          — task loss (BCE or MSE)
  train/reg_loss           — regularisation loss (MOSE: size + entropy; MotifSAT: IB + consistency)
  train/total_loss         — sum of the above
  train/<k>                — any additional loss components from MotifSAT compute_loss breakdown
  val/auc  or val/rmse     — validation prediction performance
  val/auc_mean             — for multi-label tasks
  lr                       — current learning rate (after scheduler step)
  early_stop/no_improve    — patience counter
  early_stop/best_val      — best validation metric seen so far

  MOSE-GNN only (every log_scores_every epochs):
    motif_scores/mean        — mean σ(θ_m) across vocabulary
    motif_scores/max         — max σ(θ_m)
    motif_scores/min         — min σ(θ_m)
    motif_scores/std         — std of σ(θ_m)
    motif_scores/above_0.5   — fraction of motifs with σ(θ_m) > 0.5
    motif_scores/entropy     — mean binary entropy H(σ(θ_m)) — how uncertain the scores are
    motif_scores/top10_table — W&B Table of top-10 highest-importance motifs

  MotifSAT only (every log_scores_every epochs):
    attention/mean           — mean node attention across validation batch
    attention/std            — std of node attention
    attention/above_0.5      — fraction of nodes with attention > 0.5
    attention/temperature_r  — current concrete distribution temperature (model.r)
    attention/entropy        — mean H(att) — how peaked the attention is

Usage
-----
    logger = WandbLogger(
        model=model,
        vocab=vocab,
        task_type='BinaryClass',
        model_type='mose',      # 'mose' | 'motifsat'
        log_scores_every=5,     # how often to log motif score histograms
    )
    # inside epoch loop:
    logger.log_epoch(
        epoch=epoch,
        train_losses={'task': t_loss, 'reg': r_loss, 'total': t_loss + r_loss},
        val_metrics=val_m,
        optimizer=optimizer,
        no_improve=no_improve,
        best_val=best_val,
        valid_loader=loaders['valid'],   # for MotifSAT attention stats
        device=device,
    )
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


EPS = 1e-9


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


class WandbLogger:
    """Per-epoch metric logger for MOSE-GNN and MotifSAT training.

    Parameters
    ----------
    model : nn.Module
    vocab : VocabData or None
        Used to look up motif SMARTS names for the top-10 table.
    task_type : str  'BinaryClass' | 'Regression' | 'MultiLabel'
    model_type : str  'mose' | 'motifsat'
        Controls which model-specific stats are collected.
    log_scores_every : int
        Log motif importance / attention distribution stats every N epochs.
    wandb_run : wandb.Run or None
        Active run.  If None, wandb.run is used.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        vocab=None,
        task_type: str = 'BinaryClass',
        model_type: str = 'mose',
        log_scores_every: int = 5,
        wandb_run=None,
    ):
        self.model            = model
        self.vocab            = vocab
        self.task_type        = task_type
        self.model_type       = model_type.lower()
        self.log_scores_every = log_scores_every
        self._run             = wandb_run

    @property
    def run(self):
        return self._run or (wandb.run if HAS_WANDB else None)

    def active(self) -> bool:
        return HAS_WANDB and self.run is not None

    # ── main entry point ──────────────────────────────────────────────────────

    @torch.no_grad()
    def log_epoch(
        self,
        epoch: int,
        train_losses: Dict[str, float],
        val_metrics: Dict[str, float],
        optimizer: torch.optim.Optimizer,
        no_improve: int,
        best_val: float,
        valid_loader=None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Log all per-epoch metrics to W&B.

        Parameters
        ----------
        epoch : int
        train_losses : dict
            Must contain 'task' and 'total' keys at minimum.
            MotifSAT passes the full breakdown from compute_loss
            (task, info_loss, motif_loss, etc.).
        val_metrics : dict
            Output of evaluate_predictions().
        optimizer : Optimizer
            Used to read current learning rate after scheduler step.
        no_improve : int
            Current patience counter.
        best_val : float
        valid_loader : DataLoader or None
            Required for MotifSAT attention stats.
        device : torch.device or None
        """
        if not self.active():
            return

        payload: Dict[str, Any] = {}

        # ── training losses ───────────────────────────────────────────────────
        for k, v in train_losses.items():
            if k == 'total':
                payload['train/total_loss'] = v
            elif k == 'task':
                payload['train/task_loss'] = v
            elif k in ('reg', 'reg_loss'):
                payload['train/reg_loss'] = v
            else:
                payload[f'train/{k}'] = v

        # Ensure total_loss is always present
        if 'train/total_loss' not in payload:
            payload['train/total_loss'] = sum(train_losses.values())

        # ── validation metrics ────────────────────────────────────────────────
        if self.task_type == 'Regression':
            payload['val/rmse'] = val_metrics.get('rmse', float('nan'))
            payload['val/mae']  = val_metrics.get('mae',  float('nan'))
        elif self.task_type == 'MultiLabel':
            payload['val/auc_mean'] = val_metrics.get('auc_mean', float('nan'))
        else:
            payload['val/auc'] = val_metrics.get('auc', float('nan'))

        # ── learning rate ─────────────────────────────────────────────────────
        payload['lr'] = optimizer.param_groups[0]['lr']

        # ── early stopping ────────────────────────────────────────────────────
        payload['early_stop/no_improve'] = no_improve
        payload['early_stop/best_val']   = best_val

        # ── model-specific stats (every log_scores_every epochs) ───────────────
        if epoch % self.log_scores_every == 0:
            if self.model_type == 'mose':
                payload.update(self._mose_score_stats(epoch))
            elif self.model_type == 'motifsat':
                payload.update(self._motifsat_attention_stats(
                    valid_loader, device, epoch))

        self.run.log(payload, step=epoch)

    def log_final_results(
        self,
        split_metrics: dict,
        correlation: dict = None,
        gt_roc: dict = None,
        top_bottom: dict = None,
        extra: dict = None,
    ) -> None:
        """Log post-training evaluation results to W&B summary.

        Call once after training and evaluation are complete.

        Parameters
        ----------
        split_metrics : dict
            {'train': {auc: ...}, 'valid': {auc: ...}, 'test': {auc: ...}}
        correlation : dict or None
            {'pearson': float, 'spearman': float}
        gt_roc : dict or None
            {'auc_mean': float, 'auc_std': float, 'n_graphs': int}
        top_bottom : dict or None
            Output of top_bottom_motif_eval()
        extra : dict or None
            Any additional key-value pairs to log as summary metrics.
        """
        if not self.active():
            return

        payload: dict = {}

        for split_name, metrics in split_metrics.items():
            if self.task_type == 'Regression':
                payload[f'final/{split_name}/rmse'] = metrics.get('rmse', float('nan'))
                payload[f'final/{split_name}/mae']  = metrics.get('mae',  float('nan'))
            elif self.task_type == 'MultiLabel':
                payload[f'final/{split_name}/auc_mean'] = metrics.get('auc_mean', float('nan'))
            else:
                payload[f'final/{split_name}/auc'] = metrics.get('auc', float('nan'))

        if correlation:
            payload['final/pearson']  = correlation.get('pearson',  float('nan'))
            payload['final/spearman'] = correlation.get('spearman', float('nan'))

        if gt_roc:
            payload['final/gt_roc_auc_mean']  = gt_roc.get('auc_mean',  float('nan'))
            payload['final/gt_roc_auc_std']   = gt_roc.get('auc_std',   float('nan'))
            payload['final/gt_roc_n_graphs']  = gt_roc.get('n_graphs',  0)

        if top_bottom:
            payload['final/top_mean_impact']    = top_bottom.get('top_mean_impact',    float('nan'))
            payload['final/bottom_mean_impact'] = top_bottom.get('bottom_mean_impact', float('nan'))
            payload['final/impact_ratio']       = top_bottom.get('impact_ratio',       float('nan'))

        if extra:
            for k, v in extra.items():
                payload[f'final/{k}'] = v

        # Use wandb.summary for final metrics so they appear on the run overview
        for k, v in payload.items():
            self.run.summary[k] = v
        self.run.log(payload)

    # ── MOSE-GNN: motif importance scores ─────────────────────────────────────

    def _mose_score_stats(self, epoch: int) -> Dict[str, Any]:
        """Summarise the learned motif importance scores σ(θ_m)."""
        if not hasattr(self.model, 'motif_params') or self.model.motif_params is None:
            return {}

        scores = self.model.motif_params.sigmoid().detach().cpu()

        # Flatten to 1D: [M] for single-class, [M*C] for multi-class
        s = scores.view(-1).numpy()

        ent = _binary_entropy(s)
        stats = {
            'motif_scores/mean':       float(s.mean()),
            'motif_scores/max':        float(s.max()),
            'motif_scores/min':        float(s.min()),
            'motif_scores/std':        float(s.std()),
            'motif_scores/above_0.5':  float((s > 0.5).mean()),
            'motif_scores/entropy':    float(ent.mean()),
        }

        # UNK importance
        if hasattr(self.model, 'unk_param') and self.model.unk_param is not None:
            unk_s = float(self.model.unk_param.sigmoid().item())
            stats['motif_scores/unk'] = unk_s

        # Top-10 table (per-vocabulary motif, mean across classes).
        # motif_params rows are COMPACT (kept-only); map each row back to its
        # global motif id via kept_motif_ids before indexing motif_list.
        motif_list = getattr(self.vocab, 'motif_list', None) if self.vocab else None
        kept = getattr(self.model, 'kept_motif_ids', None)
        mean_scores = scores.mean(dim=-1) if scores.dim() > 1 else scores  # [K]
        top10_rows = mean_scores.argsort(descending=True)[:10].tolist()

        rows = []
        for row in top10_rows:
            gid = int(kept[row]) if kept is not None else int(row)
            name = (motif_list[gid][:40] if motif_list and gid < len(motif_list)
                    else f'motif_{gid}')
            rows.append([
                gid,
                name,
                round(float(mean_scores[row]), 4),
            ])
        stats['motif_scores/top10_table'] = wandb.Table(
            columns=['motif_id', 'smarts', 'importance'],
            data=rows,
        )

        return stats

    # ── MotifSAT: node attention distribution ─────────────────────────────────

    @torch.no_grad()
    def _motifsat_attention_stats(
        self,
        valid_loader,
        device: Optional[torch.device],
        epoch: int,
        max_batches: int = 8,
    ) -> Dict[str, Any]:
        """Collect node attention values from validation batches and summarise."""
        if valid_loader is None or device is None:
            return {}

        self.model.eval()
        all_att: List[np.ndarray] = []
        n_batches = 0

        for data in valid_loader:
            if n_batches >= max_batches:
                break
            data = data.to(device)
            n = data.x.size(0)
            batch = (data.batch if data.batch is not None
                     else torch.zeros(n, dtype=torch.long, device=device))
            try:
                out = self.model(
                    data.x, data.edge_index, batch,
                    getattr(data, 'nodes_to_motifs', None),
                    getattr(data, 'edge_attr', None),
                )
                node_att = out[1]  # (logits, node_att, aux)
                if node_att is not None:
                    all_att.append(node_att.view(-1).cpu().numpy())
            except Exception:
                pass
            n_batches += 1

        if not all_att:
            return {}

        att = np.concatenate(all_att)
        ent = _binary_entropy(np.clip(att, EPS, 1 - EPS))

        stats = {
            'attention/mean':       float(att.mean()),
            'attention/std':        float(att.std()),
            'attention/above_0.5':  float((att > 0.5).mean()),
            'attention/entropy':    float(ent.mean()),
        }

        # Concrete distribution temperature
        if hasattr(self.model, 'r'):
            stats['attention/temperature_r'] = float(self.model.r.item())

        return stats
