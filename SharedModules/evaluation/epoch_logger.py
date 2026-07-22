"""epoch_logger.py — tiny append-only per-epoch CSV logger. No server, no per-step IO.

Records exactly what wandb doesn't give you for this project: per epoch, each motif's IMPORTANCE
(what the explainer says) and IMPACT (δ = prob drop when occluded), alongside the train/val/test
metrics that epoch — so you can watch, as training proceeds, how motif attribution and generalisation
move together.

Two files per run:
    {run_dir}/epoch_scalars.csv   one row / epoch:  epoch, <scalar metrics...>          (always)
    {run_dir}/epoch_motifs.csv    long format:      epoch, motif, importance, impact    (every motif_every)

IO cost is per-EPOCH, not per-step: one line (scalars) + ~|V| lines (motif snapshot, throttled). At
500 epochs this is a few hundred KB and well under a second total — negligible beside epoch compute.
Opt-in via `enabled` (mirror your existing no-wandb default). Handles stay open and are flushed each
epoch so a killed job keeps everything written so far.
"""
from __future__ import annotations
import csv, os
from typing import Dict, Mapping, Optional


class EpochLogger:
    def __init__(self, run_dir: str, enabled: bool = True, motif_every: int = 10,
                 top_bottom: Optional[int] = None):
        """`motif_every`: snapshot motif importance/impact every N epochs (1 = every epoch).
        `top_bottom`: if set, log only the k highest- and k lowest-importance motifs each snapshot
        (keeps the motif file tiny for large vocabularies)."""
        self.enabled = enabled
        self.dir = run_dir
        self.motif_every = max(1, motif_every)
        self.top_bottom = top_bottom
        self._sf = self._sw = None      # scalars file / writer
        self._mf = self._mw = None      # motifs file / writer
        if enabled:
            os.makedirs(run_dir, exist_ok=True)

    def log_scalars(self, epoch: int, **metrics: float) -> None:
        if not self.enabled:
            return
        if self._sw is None:
            self._sf = open(os.path.join(self.dir, 'epoch_scalars.csv'), 'a', newline='')
            self._sw = csv.writer(self._sf)
            if self._sf.tell() == 0:
                self._sw.writerow(['epoch'] + list(metrics.keys()))
            self._cols = list(metrics.keys())
        self._sw.writerow([epoch] + [metrics.get(k, '') for k in self._cols])
        self._sf.flush()

    def log_motifs(self, epoch: int,
                   importance: Mapping[str, float],
                   impact: Mapping[str, float]) -> None:
        if not self.enabled or (epoch % self.motif_every):
            return
        motifs = list(importance.keys())
        if self.top_bottom and len(motifs) > 2 * self.top_bottom:
            motifs.sort(key=lambda m: importance.get(m, 0.0))
            motifs = motifs[:self.top_bottom] + motifs[-self.top_bottom:]
        if self._mw is None:
            self._mf = open(os.path.join(self.dir, 'epoch_motifs.csv'), 'a', newline='')
            self._mw = csv.writer(self._mf)
            if self._mf.tell() == 0:
                self._mw.writerow(['epoch', 'motif', 'importance', 'impact'])
        for m in motifs:
            self._mw.writerow([epoch, m, importance.get(m, ''), impact.get(m, '')])
        self._mf.flush()

    def close(self) -> None:
        for f in (self._sf, self._mf):
            if f is not None:
                f.close()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()
