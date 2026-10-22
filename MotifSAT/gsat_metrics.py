"""gsat_metrics.py — the official GSAT explanation metric, computed on a split.

GSAT (run_gsat.py ``get_eval_score``) reports explanation ROC-AUC POOLED over all
edges of a split: node attention is lifted to edges as att[src] * att[dst]
(``lift_node_att_to_edge_att``) and scored against ``edge_label`` with a single
``roc_auc_score``; attention comes from the eval-mode forward (deterministic
sigmoid, no Concrete noise). Graphs without edge GT contribute negatives, exactly
as in GSAT. Alongside it we report the project's per-graph node GT-ROC (mean over
graphs that have both node classes) so the two conventions sit side by side.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def pooled_edge_auc(edge_scores: np.ndarray, edge_labels: np.ndarray) -> float:
    """roc_auc_score over all edges; NaN when the labels have a single class."""
    if np.unique(edge_labels).size < 2:
        return float('nan')
    return float(roc_auc_score(edge_labels, edge_scores))


@torch.no_grad()
def gsat_explanation_metrics(model, loader, device) -> Dict[str, float]:
    """{'gsat_edge_auc', 'node_gtroc_pergraph', 'n_gt_graphs'} for one split.

    Returns NaNs when the split carries no ``edge_label`` / ``node_label``.
    """
    model.eval()
    e_scores, e_labels, per_graph = [], [], []
    for data in loader:
        data = data.to(device)
        if getattr(data, 'edge_label', None) is None:
            return {'gsat_edge_auc': float('nan'), 'node_gtroc_pergraph': float('nan'),
                    'n_gt_graphs': 0}
        _, node_att, aux = model(
            data.x, data.edge_index, data.batch,
            getattr(data, 'nodes_to_motifs', None), getattr(data, 'edge_attr', None),
            epoch=1)
        if aux.get('edge_att') is not None:          # learn_edge_att path
            ea = aux['edge_att'].view(-1)
        else:
            att = node_att.view(-1)
            src, dst = data.edge_index
            ea = att[src] * att[dst]
        e_scores.append(ea.cpu().numpy())
        e_labels.append(data.edge_label.view(-1).cpu().numpy())
        nl = getattr(data, 'node_label', None)
        if nl is not None and node_att is not None:
            att = node_att.view(-1).cpu().numpy()
            nl = nl.view(-1).cpu().numpy()
            b = data.batch.cpu().numpy()
            for g in np.unique(b):
                m = b == g
                if np.unique(nl[m]).size > 1:
                    per_graph.append(roc_auc_score(nl[m], att[m]))
    return {
        'gsat_edge_auc': pooled_edge_auc(np.concatenate(e_scores), np.concatenate(e_labels)),
        'node_gtroc_pergraph': float(np.mean(per_graph)) if per_graph else float('nan'),
        'n_gt_graphs': len(per_graph),
    }
