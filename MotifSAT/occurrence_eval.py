"""occurrence_eval.py — motif impact at two granularities: all occurrences vs one.

For every graph (one split) with ``nodes_to_motifs`` and ``nodes_to_copy``:

  W        = the model's own eval attention (``model_node_att_fn``: noise-free
             node_att_soft), the faithful-LOO baseline used everywhere else.
  mask-all = for each motif m in the graph: zero W on ALL copies of m
             score_all = mean W over those atoms;  impact_all = |p(W) - p(W \\ m)|
  mask-one = for each occurrence (m, c):     zero W on copy c only
             score = mean W over copy c's atoms; impact = |p(W) - p(W \\ (m,c))|

p = all model outputs: sigmoid for BinaryClass / MultiLabel, raw for Regression — the same
rule as ``motif_eval._single_pred``; impact = mean over outputs of |Δp| (``_pred_delta``), so
mask-all reproduces the existing ``compute_motif_impact``. Multi-task models also get
per-task impacts in columns ``impact_t<j>`` (for per-task / per-category analysis).
All masked variants of a graph are scored in ONE batched forward (node_weights path).

Metrics (Pearson of score vs impact):
  instance  — one point per (graph, motif) [mask-all] or (graph, motif, copy) [mask-one]
  grouped   — one point per motif: atom-weighted mean score vs mean impact over its
              instances (the existing grouped definition)
  *_multi   — the instance metrics restricted to (graph, motif) pairs with >= 2 copies,
              the only place the two maskings differ.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from SharedModules.evaluation.motif_eval import model_node_att_fn


def _probs(model, d, weights: List[torch.Tensor], device, task_type: str) -> np.ndarray:
    """[k, T] predictions for the k weightings of one graph (T = number of outputs)."""
    k = len(weights)
    b = Batch.from_data_list([d] * k).to(device)
    w = torch.cat(weights).view(-1, 1).to(device)
    out, _, _ = model(b.x, b.edge_index, b.batch, getattr(b, 'nodes_to_motifs', None),
                      getattr(b, 'edge_attr', None), node_weights=w)
    out = out.view(k, -1)
    if task_type in ('BinaryClass', 'MultiLabel'):
        out = torch.sigmoid(out)
    return out.cpu().numpy().astype(float)


@torch.no_grad()
def occurrence_tables(model, data_list, device, task_type: str
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(per-occurrence rows, per-(graph, motif) rows) for one split."""
    model.eval()
    att_fn = model_node_att_fn(model, device)
    occ, allm = [], []
    for gi, d in enumerate(data_list):
        n2m = getattr(d, 'nodes_to_motifs', None)
        n2c = getattr(d, 'nodes_to_copy', None)
        if n2m is None or n2c is None:
            raise ValueError("occurrence_eval needs nodes_to_motifs and nodes_to_copy on every "
                             "graph (vocabulary with _mol_fragment_copies.pickle)")
        W = att_fn(d)
        if W is None:
            raise ValueError("occurrence_eval: model exposes no node attention")
        W = W.view(-1).float().cpu()
        n2m, n2c = n2m.view(-1).cpu(), n2c.view(-1).cpu()
        motifs = [int(m) for m in torch.unique(n2m).tolist() if m >= 0]
        masks, labels = [], []                 # labels: ('all', m) | ('one', m, c)
        for m in motifs:
            in_m = n2m == m
            copies = sorted(int(c) for c in torch.unique(n2c[in_m]).tolist())
            masks.append(in_m); labels.append(('all', m, None, len(copies)))
            if len(copies) > 1:                # one copy only => mask-one == mask-all
                for c in copies:
                    masks.append(in_m & (n2c == c)); labels.append(('one', m, c, len(copies)))
        weights = [W] + [W.masked_fill(mk, 0.0) for mk in masks]
        p = _probs(model, d, weights, device, task_type)
        p0, smi = p[0], str(getattr(d, 'smiles', ''))
        multi = p.shape[1] > 1
        for (kind, m, c, ncp), mk, pk in zip(labels, masks, p[1:]):
            row = dict(graph_idx=gi, smiles=smi, motif_id=m, n_copies=ncp,
                       n_atoms=int(mk.sum()), score=float(W[mk].mean()),
                       impact=float(np.mean(np.abs(p0 - pk))))
            if multi:
                row.update({f'impact_t{j}': float(abs(p0[j] - pk[j])) for j in range(p.shape[1])})
            if kind == 'all':
                allm.append(row)
                if ncp == 1:                   # its single occurrence
                    occ.append(dict(row, copy=0))
            else:
                occ.append(dict(row, copy=c))
    return pd.DataFrame(occ), pd.DataFrame(allm)


def _pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def _grouped(df: pd.DataFrame) -> float:
    if df.empty:
        return float('nan')
    g = df.assign(wsc=df.score * df.n_atoms).groupby('motif_id').agg(
        wsc=('wsc', 'sum'), na=('n_atoms', 'sum'), imp=('impact', 'mean'))
    return _pearson(g.wsc / g.na, g.imp)


def occurrence_metrics(occ: pd.DataFrame, allm: pd.DataFrame) -> Dict[str, float]:
    multi_all = allm[allm.n_copies > 1] if not allm.empty else allm
    multi_one = occ[occ.n_copies > 1] if not occ.empty else occ
    return {
        'pearson_instance_maskall': _pearson(allm.score, allm.impact) if not allm.empty else float('nan'),
        'pearson_instance_maskone': _pearson(occ.score, occ.impact) if not occ.empty else float('nan'),
        'pearson_motif_maskall': _grouped(allm),
        'pearson_motif_maskone': _grouped(occ),
        'pearson_instance_maskall_multi': _pearson(multi_all.score, multi_all.impact) if not multi_all.empty else float('nan'),
        'pearson_instance_maskone_multi': _pearson(multi_one.score, multi_one.impact) if not multi_one.empty else float('nan'),
        'n_instances_maskall': int(len(allm)), 'n_instances_maskone': int(len(occ)),
        'n_multi_pairs': int(len(multi_all)),
    }


def run_occurrence_eval(model, split_lists: Dict[str, list], device, task_type: str,
                        out_dir) -> Dict[str, float]:
    """Per split + pooled 'all': write occurrence_<split>.csv / motif_allmask_<split>.csv
    and return flat summary fields ``occ_<metric>_<split>``."""
    from pathlib import Path
    out, occs, alls = {}, [], []
    for split, dl in split_lists.items():
        occ, allm = occurrence_tables(model, dl, device, task_type)
        occ.to_csv(Path(out_dir) / f'occurrence_{split}.csv', index=False)
        allm.to_csv(Path(out_dir) / f'motif_allmask_{split}.csv', index=False)
        for k, v in occurrence_metrics(occ, allm).items():
            out[f'occ_{k}_{split}'] = v
        occs.append(occ); alls.append(allm)
    for k, v in occurrence_metrics(pd.concat(occs, ignore_index=True),
                                   pd.concat(alls, ignore_index=True)).items():
        out[f'occ_{k}_all'] = v
    return out
