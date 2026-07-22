"""motif_coherence.py — eval-time within/between-motif attention variance, as a paper-ready METRIC.

Unlike MotifSAT's loss-side within_var/between_var (only computed when a consistency-loss coefficient
is > 0, and only over TRAINING batches), this evaluates on any graph list for ANY attention model
(GSAT node-level, MotifSAT motif-level, MOSE). It is the direct, training-insensitive evidence for
MotifSAT's claim: motif-coherent attention has within-motif variance ≈ 0, GSAT's is > 0.

Definitions (per the loss, but as a standalone metric):
    within-motif variance  = mean over (graph, motif) groups of  Var({ a_n : n ∈ that motif instance })
    between-motif variance = mean over graphs of  Var({ mean_n a_n : per motif in the graph })
UNK nodes (nodes_to_motifs < 0) are excluded; motifs/graphs with a single node/motif contribute 0.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional
import numpy as np
import torch
from torch_geometric.data import Data


def _graph_within_between(att: np.ndarray, n2m: np.ndarray) -> (float, float, int):
    """within-var (mean over motifs of Var within motif), between-var (Var over per-motif means),
    and the number of motifs, for ONE graph. UNK (n2m < 0) excluded."""
    keep = n2m >= 0
    att, n2m = att[keep], n2m[keep]
    if att.size == 0:
        return np.nan, np.nan, 0
    within_per_motif = []
    motif_means = []
    for mid in np.unique(n2m):
        a = att[n2m == mid]
        motif_means.append(float(a.mean()))
        within_per_motif.append(float(a.var()) if a.size > 1 else 0.0)   # population var within motif
    within = float(np.mean(within_per_motif))
    between = float(np.var(motif_means)) if len(motif_means) > 1 else 0.0
    return within, between, len(motif_means)


def motif_attention_coherence(
    model: torch.nn.Module,
    data_list: List[Data],
    device: torch.device,
    node_att_fn: Callable[[Data], torch.Tensor],
    max_graphs: Optional[int] = None,
) -> Dict[str, float]:
    """Return {within_motif_var, between_motif_var, n_graphs} averaged over graphs that have a motif
    map. `node_att_fn(data) -> Tensor[N]` supplies the per-node attention (use model_node_att_fn)."""
    model.eval()
    wl, bl = [], []
    graphs = data_list if max_graphs is None else data_list[:max_graphs]
    for d in graphs:
        n2m = getattr(d, 'nodes_to_motifs', None)
        if n2m is None:
            continue
        with torch.no_grad():
            a = node_att_fn(d.to(device))
        if a is None:
            continue
        a = a.detach().view(-1).cpu().numpy()
        n2m_np = n2m.detach().view(-1).cpu().numpy()
        if a.shape[0] != n2m_np.shape[0]:
            continue
        w, b, nm = _graph_within_between(a, n2m_np)
        if nm >= 1 and np.isfinite(w):
            wl.append(w); bl.append(b)
    return {
        'within_motif_var':  float(np.mean(wl)) if wl else float('nan'),
        'between_motif_var': float(np.mean(bl)) if bl else float('nan'),
        'n_graphs':          len(wl),
    }
