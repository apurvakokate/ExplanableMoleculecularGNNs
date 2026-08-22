"""training_tracker.py — periodic per-epoch explainability telemetry during training.

Logs, every ``every`` epochs, to CSV (via EpochLogger) for an ante-hoc model
(MOSE / MotifSAT / GSAT):
  epoch_scalars.csv : gt_roc_node_auc (GT-ROC vs the fired-clause planted cause)
                      (Mode 2), spurious_roc_node_auc — all from the model's OWN attention.
  epoch_motifs.csv  : per-motif importance (the model's learned motif score) and impact
                      (faithful leave-one-out |Δp|), reusing the exact eval-time helpers.

This is AUXILIARY telemetry: a failure is logged and swallowed so it can NEVER kill a
long training run (unlike the coherence metric, which is a correctness gate). Cost is
bounded by ``max_graphs`` (a sample of the eval split) and the ``every`` interval; the
GT-ROC snapshots are one forward pass per graph, the motif impact is LOO over that sample.
Evaluated on the VALIDATION split (no test peeking); the split's Data must carry
``node_label`` (and, for synthetic GT, ``node_label_spurious`` [N,S]).
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import torch
from torch_geometric.data import Data


def build_from_loaders(loaders, vocab, device, run_dir: str, task_type: str,
                       split: str = 'valid') -> Tuple[Optional['TrainingExplTracker'], object]:
    """Factory: build (tracker, EpochLogger) from a trainer's context, or (None, None)
    when disabled. Controlled by env:
      TRACK_EXPL_EVERY      snapshot interval in epochs (default 25; 0 disables).
      TRACK_EXPL_MAX_GRAPHS eval-split sample cap per snapshot (default 128).
    Tracks on the ``split`` loader (default 'valid' — no test peeking). The split's
    Data must carry node_label (the fired-clause cause); synthetic-GT splits also carry
    node_label_spurious [N,S] (source-GT tracks gt_roc only)."""
    every = int(os.environ.get('TRACK_EXPL_EVERY', '25'))
    if every <= 0:
        return None, None
    try:
        graphs = list(loaders[split].dataset)
    except Exception:
        return None, None
    if not graphs or getattr(graphs[0], 'node_label', None) is None:
        return None, None
    from .epoch_logger import EpochLogger
    logger = EpochLogger(run_dir, enabled=True, motif_every=1,
                         top_bottom=int(os.environ.get('TRACK_EXPL_TOP_BOTTOM', '15')))
    tracker = TrainingExplTracker(
        graphs, vocab, device, logger, every=every,
        max_graphs=int(os.environ.get('TRACK_EXPL_MAX_GRAPHS', '128')),
        task_type=task_type)
    return tracker, logger


class TrainingExplTracker:
    """Callable ``tracker(model, epoch)`` for the training loop's per-epoch hook."""

    def __init__(self,
                 gt_graphs: List[Data],
                 vocab,
                 device: torch.device,
                 logger,
                 *,
                 every: int = 25,
                 max_graphs: Optional[int] = 128,
                 max_motifs: Optional[int] = None,
                 task_type: str = 'BinaryClass'):
        self.graphs = (gt_graphs[:max_graphs] if max_graphs else gt_graphs) or []
        self.vocab = vocab
        self.device = device
        self.logger = logger
        self.every = int(every)
        self.max_motifs = max_motifs
        self.task_type = task_type
        self._log = logging.getLogger(__name__)

    def __call__(self, model, epoch: int,
                 val_auc: Optional[float] = None) -> None:
        if self.every <= 0 or not self.graphs or (epoch % self.every):
            return
        try:
            from .motif_eval import (compute_gt_roc, compute_motif_impact,
                                     model_node_att_fn)
            was_training = model.training
            model.eval()
            att = model_node_att_fn(model, self.device)

            scal = {}
            # Per-epoch validation task metric (AUC for classification), logged
            # alongside gt_roc so the score-vs-impact trajectory can be selected
            # under a val-AUC tolerance (faithfulness s.t. val_auc >= best-eps).
            if val_auc is not None:
                scal['val_auc'] = float(val_auc)
            g = compute_gt_roc(model, self.graphs, self.device,
                               node_att_fn=att, level='node')
            scal['gt_roc_node_auc'] = g['auc_mean']
            # node_label IS the fired-clause cause now — no separate 'fired' curve to track.
            # No spurious curve either: SpurROC is a contrast against the cause and
            # compute_gt_roc cannot express it (it scores one mask against its complement).
            # Logging a complement-based value under the name spurious_roc_node_auc would
            # mean something different from the phase-6 metric of that name.
            self.logger.log_scalars(epoch, **scal)

            # per-motif importance + impact — from compute_motif_impact, which works
            # for ALL ante-hoc families: 'impact' = faithful LOO |Δp|; importance =
            # mean 'score_values' = the model's OWN per-motif weighting (MOSE global
            # score, degenerate-constant; MotifSAT/GSAT = mean node attention). No
            # dependency on get_motif_scores (MotifSAT/GSAT don't have it).
            import numpy as _np
            mi = compute_motif_impact(model, self.graphs, self.vocab, self.device,
                                      task_type=self.task_type,
                                      max_motifs=self.max_motifs)
            importance, impact = {}, {}
            for mid, d in mi.items():
                sv = d.get('score_values') or []
                importance[str(mid)] = float(_np.nanmean(sv)) if len(sv) else float('nan')
                impact[str(mid)] = float(d.get('impact', float('nan')))
            if importance:
                self.logger.log_motifs(epoch, importance, impact)

            if was_training:
                model.train()
        except Exception as e:                                   # telemetry: never kill training
            self._log.warning("training expl tracker failed at epoch %d: %s", epoch, e)
