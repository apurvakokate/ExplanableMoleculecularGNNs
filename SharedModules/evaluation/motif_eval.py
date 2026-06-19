"""motif_eval.py — motif-level explainer evaluation.

Four evaluation modes
---------------------
1. compute_motif_impact
   For each motif, zero its atoms and measure |p(x) - p(x_masked)|.
   Uses the bool mask cache from generate_vocab_rules for efficiency.

2. score_impact_correlation
   Pearson + Spearman between learned scores and mask-based impacts.

3. top_bottom_motif_eval
   Compare the mean impact of the top-K highest-scored motifs against the
   bottom-K lowest-scored motifs.  A well-calibrated model should show
   top_impact >> bottom_impact.

4. gt_vs_outside_gt_eval
   For datasets with known ground-truth explanatory motifs, compare impact
   and learned scores of GT motifs vs non-GT motifs across three subsets:
     - all test examples
     - all class-1 test examples
     - correctly-predicted class-1 test examples
   Also reports motif-level AUC: using learned score as classifier, GT=1.

5. explainer_roc_vs_gt  (edge-level, for attention-based models)
   ROC-AUC between node-derived attention weights and GT node mask.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score

from ..data.vocab import VocabData


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _get_probs(
    model,
    data_list: List[Data],
    device: torch.device,
    task_type: str,
) -> Dict[str, float]:
    """smiles → predicted probability (sigmoid for binary, raw for regression)."""
    from .metrics import _model_forward
    probs = {}
    for d in data_list:
        out = _model_forward(model, d.to(device))
        if task_type == 'BinaryClass':
            probs[d.smiles] = float(torch.sigmoid(out.view(-1)[0]))
        else:
            probs[d.smiles] = float(out.view(-1)[0])
    return probs


@torch.no_grad()
def _single_prob(model, data: Data, device: torch.device, task_type: str) -> float:
    from .metrics import _model_forward
    out = _model_forward(model, data.to(device))
    if task_type == 'BinaryClass':
        return float(torch.sigmoid(out.view(-1)[0]))
    return float(out.view(-1)[0])


def _resolve_mask_cache(
    vocab: VocabData,
    data_list: List[Data],
    split: str,
) -> Dict[int, Dict[str, torch.BoolTensor]]:
    """Return mask cache for the given split, computing on-the-fly if absent."""
    cache = vocab.mask_cache.get(split, {})
    if cache:
        return cache
    from ..data.vocab import compute_mask_cache
    smiles = [d.smiles for d in data_list]
    groups = [split] * len(smiles)
    merged = compute_mask_cache(smiles, groups, vocab.lookup_for_split(split))
    return merged.get(split, {})


def _true_label(data: Data) -> Optional[int]:
    """Return the integer label of a Data object, or None if unavailable."""
    if data.y is None:
        return None
    y = data.y.view(-1)
    if y.numel() == 0:
        return None
    return int(y[0].item())


def _injection_modes(model) -> Tuple[bool, bool]:
    """Decide what to ablate when removing a motif, based on WHERE the model
    injects the motif attention.

    The ablation must mirror the injection point so that "impact" measures the
    contribution through the same channel the model actually uses:
      - node-feature injection (``w_feat``) or readout injection (``w_readout``)
        → mask NODES   (zero the motif's atom features)
      - edge / message injection (``w_message``)
        → mask EDGES   (drop every edge incident to the motif's atoms)
      - all three active → mask BOTH nodes and edges

    Falls back to node masking when the model does not expose injection flags
    (e.g. VanillaGNN / post-hoc explainers), preserving the previous behaviour.

    Returns
    -------
    (mask_nodes, mask_edges) : tuple[bool, bool]
    """
    w_feat    = bool(getattr(model, 'w_feat', False))
    w_message = bool(getattr(model, 'w_message', False))
    w_readout = bool(getattr(model, 'w_readout', False))
    if not (w_feat or w_message or w_readout):
        return True, False
    return (w_feat or w_readout), w_message


def _ablate_motif(
    data: Data,
    bool_mask: torch.Tensor,
    mask_nodes: bool,
    mask_edges: bool,
) -> Optional[Data]:
    """Return a clone of ``data`` with one motif removed.

    Parameters
    ----------
    bool_mask : Tensor [n_atoms] (bool)
        True at the motif's atom indices.
    mask_nodes : bool
        Zero the motif atoms' input features (node-level ablation).
    mask_edges : bool
        Drop every edge incident to a motif atom, and the matching edge_attr
        rows (edge-level ablation), fully disconnecting the motif so no message
        flows through it.

    Returns None when the mask length does not match the graph (atom-index
    mismatch) so the caller can SKIP rather than silently produce a wrong
    ablation — this is the node→motif index-consistency guard.
    """
    nm = bool_mask.view(-1)
    if nm.numel() != data.num_nodes:
        return None
    out = data.clone()
    nm = nm.to(out.x.device)
    if mask_nodes:
        out.x = out.x * (~nm).float().unsqueeze(-1)
    if mask_edges and out.edge_index.numel() > 0:
        src, dst = out.edge_index
        keep = ~(nm[src] | nm[dst])
        out.edge_index = out.edge_index[:, keep]
        ea = getattr(out, 'edge_attr', None)
        if ea is not None and ea.size(0) == keep.numel():
            out.edge_attr = ea[keep]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mask-based motif impact
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_motif_impact(
    model: torch.nn.Module,
    data_list: List[Data],
    vocab: VocabData,
    device: torch.device,
    split: str = 'test',
    task_type: str = 'BinaryClass',
    max_motifs: Optional[int] = None,
) -> Dict[int, Dict[str, float]]:
    """Per-motif marginal impact via node-feature zeroing.

    For each known motif m:
        impact(m) = mean_{g : m ∈ g} |p(g) - p(g with m zeroed)|

    Returns
    -------
    dict[motif_id → {
        'impact': float,
        'impact_std': float,
        'n_graphs': int,
        'motif_smarts': str,
    }]
    """
    model.eval()
    mask_cache = _resolve_mask_cache(vocab, data_list, split)
    smi_to_data = {d.smiles: d for d in data_list}
    orig_probs = _get_probs(model, data_list, device, task_type)
    mask_nodes, mask_edges = _injection_modes(model)

    motif_ids = sorted(mask_cache.keys())
    if max_motifs is not None:
        motif_ids = motif_ids[:max_motifs]

    results: Dict[int, Dict[str, float]] = {}
    for mid in motif_ids:
        impacts = []
        for smi, bool_mask in mask_cache[mid].items():
            d = smi_to_data.get(smi)
            orig_p = orig_probs.get(smi)
            if d is None or orig_p is None:
                continue
            masked_d = _ablate_motif(d, bool_mask, mask_nodes, mask_edges)
            if masked_d is None:
                continue
            impacts.append(abs(orig_p - _single_prob(model, masked_d, device, task_type)))

        if impacts:
            smarts = vocab.motif_list[mid] if mid < len(vocab.motif_list) else '?'
            results[mid] = {
                'impact':       float(np.mean(impacts)),
                'impact_std':   float(np.std(impacts)),
                'n_graphs':     len(impacts),
                'motif_smarts': smarts,
            }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Score vs impact correlation
# ─────────────────────────────────────────────────────────────────────────────

def score_impact_correlation(
    motif_scores: Dict[int, float],
    motif_impacts: Dict[int, Dict[str, float]],
) -> Dict[str, float]:
    """Pearson and Spearman correlation between learned scores and impacts."""
    from scipy.stats import pearsonr, spearmanr

    common = sorted(set(motif_scores) & set(motif_impacts))
    if len(common) < 3:
        return {'pearson': float('nan'), 'spearman': float('nan')}

    s   = np.array([motif_scores[m]          for m in common])
    imp = np.array([motif_impacts[m]['impact'] for m in common])

    def _safe(fn, a, b):
        return float(fn(a, b)[0]) if np.std(a) > 0 and np.std(b) > 0 else float('nan')

    return {
        'pearson':  _safe(pearsonr,  s, imp),
        'spearman': _safe(spearmanr, s, imp),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2b. Motif class-discriminativeness (label-aware)
# ─────────────────────────────────────────────────────────────────────────────

def motif_class_discriminativeness(
    data_list: List[Data],
    vocab: VocabData,
    split: str = 'test',
    max_motifs: Optional[int] = None,
) -> Dict[int, Dict[str, float]]:
    """How class-discriminative is each motif's *presence*, independent of the
    model.

    Motivation: an explainer can assign a high score to a motif that is not
    actually predictive of the label (a non-discriminative / shortcut motif).
    This measures, purely from data, whether a motif's presence separates the
    classes — so a learned score can be checked against it.

    For each motif m with presence indicator z_m ∈ {0,1} over graphs:
        prevalence     = P(z_m = 1)
        p1_given_m     = P(y = 1 | z_m = 1)
        p1_given_not_m = P(y = 1 | z_m = 0)
        delta_p1       = p1_given_m - p1_given_not_m      (signed separation)
        presence_auc   = AUC(y_true = y, y_score = z_m)   (0.5 = useless)
        abs_disc       = |presence_auc - 0.5| * 2         (0..1, 1 = perfect)

    Only defined for BinaryClass. Returns
        dict[motif_id → {prevalence, p1_given_m, p1_given_not_m, delta_p1,
                         presence_auc, abs_disc, n_present}]
    """
    mask_cache = _resolve_mask_cache(vocab, data_list, split)
    # Per-graph binary label keyed by smiles
    y_by_smi: Dict[str, float] = {}
    for d in data_list:
        y = d.y
        try:
            y_by_smi[d.smiles] = float(y.view(-1)[0].item())
        except Exception:
            continue
    smis = [d.smiles for d in data_list if d.smiles in y_by_smi]
    y_all = np.array([y_by_smi[s] for s in smis])
    n = len(smis)
    if n == 0 or len(set(y_all.tolist())) < 2:
        return {}
    base_rate = float(y_all.mean())

    motif_ids = sorted(mask_cache.keys())
    if max_motifs is not None:
        motif_ids = motif_ids[:max_motifs]

    out: Dict[int, Dict[str, float]] = {}
    for mid in motif_ids:
        present_smis = set(mask_cache[mid].keys())
        z = np.array([1.0 if s in present_smis else 0.0 for s in smis])
        n_present = int(z.sum())
        if n_present == 0 or n_present == n:
            continue  # motif gives no contrast on this split
        y1 = y_all[z == 1]
        y0 = y_all[z == 0]
        p1_m   = float(y1.mean()) if len(y1) else float('nan')
        p1_notm= float(y0.mean()) if len(y0) else float('nan')
        try:
            pauc = float(roc_auc_score(y_all, z))
        except ValueError:
            pauc = float('nan')
        smarts = vocab.motif_list[mid] if mid < len(vocab.motif_list) else '?'
        out[mid] = {
            'prevalence':     round(n_present / n, 4),
            'p1_given_m':     round(p1_m, 4),
            'p1_given_not_m': round(p1_notm, 4),
            'delta_p1':       round(p1_m - p1_notm, 4),
            'presence_auc':   round(pauc, 4) if pauc == pauc else pauc,
            'abs_disc':       round(abs(pauc - 0.5) * 2, 4) if pauc == pauc else pauc,
            'n_present':      n_present,
            'motif_smarts':   smarts,
            'base_rate':      round(base_rate, 4),
        }
    return out


def top_motifs_discriminative_check(
    motif_scores: Dict[int, float],
    discrim: Dict[int, Dict[str, float]],
    k: int = 10,
) -> Dict[str, float]:
    """Check whether the top-k highest-SCORED motifs are class-discriminative.

    Returns the mean |discriminativeness| of the top-k scored motifs vs all
    scored motifs, plus the Spearman correlation between score and
    discriminativeness. If top-scored motifs are genuinely discriminative,
    top_k_abs_disc should exceed mean_abs_disc and the correlation be positive.
    """
    from scipy.stats import spearmanr
    common = [m for m in motif_scores if m in discrim
              and discrim[m].get('abs_disc') == discrim[m].get('abs_disc')]
    if len(common) < 3:
        return {'top_k_abs_disc': float('nan'),
                'mean_abs_disc': float('nan'),
                'score_disc_spearman': float('nan'),
                'n_motifs': len(common)}
    ranked = sorted(common, key=lambda m: -motif_scores[m])
    topk = ranked[:k]
    s   = np.array([motif_scores[m]            for m in common])
    ad  = np.array([discrim[m]['abs_disc']     for m in common])
    sp  = (float(spearmanr(s, ad)[0])
           if np.std(s) > 0 and np.std(ad) > 0 else float('nan'))
    return {
        'top_k_abs_disc':   round(float(np.mean([discrim[m]['abs_disc'] for m in topk])), 4),
        'mean_abs_disc':    round(float(ad.mean()), 4),
        'score_disc_spearman': round(sp, 4) if sp == sp else sp,
        'n_motifs':         len(common),
        'k':                min(k, len(ranked)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Top-K vs Bottom-K motif evaluation
# ─────────────────────────────────────────────────────────────────────────────

def top_bottom_motif_eval(
    motif_scores: Dict[int, float],
    motif_impacts: Dict[int, Dict[str, float]],
    k: int = 10,
) -> Dict[str, object]:
    """Compare the mean impact of the top-K vs bottom-K scored motifs.

    A well-calibrated model should show top_impact >> bottom_impact.
    Only motifs with impact data are considered.

    Parameters
    ----------
    motif_scores : dict[motif_id → score]
        Learned importance scores (e.g. sigmoid of θ_m).
    motif_impacts : dict[motif_id → {'impact': float, ...}]
        Output of compute_motif_impact().
    k : int
        Number of motifs in each group.

    Returns
    -------
    dict with keys:
        top_k_ids       list[int]   motif ids in top-K
        bottom_k_ids    list[int]   motif ids in bottom-K
        top_k_scores    list[float]
        bottom_k_scores list[float]
        top_k_impacts   list[float]
        bottom_k_impacts list[float]
        top_mean_impact   float
        bottom_mean_impact float
        top_mean_score    float
        bottom_mean_score float
        impact_ratio      float     top_mean / bottom_mean (NaN if bottom=0)
        top_k_smarts      list[str]
        bottom_k_smarts   list[str]
    """
    # Only consider motifs that have both a score and an impact measurement
    common = sorted(set(motif_scores) & set(motif_impacts))
    if len(common) < 2:
        return {
            'top_k_ids': [], 'bottom_k_ids': [],
            'top_mean_impact': float('nan'),
            'bottom_mean_impact': float('nan'),
            'impact_ratio': float('nan'),
        }

    # Sort by score descending
    ranked = sorted(common, key=lambda m: motif_scores[m], reverse=True)
    actual_k = min(k, len(ranked) // 2)   # guard: can't overlap top and bottom

    top_ids    = ranked[:actual_k]
    bottom_ids = ranked[-actual_k:]

    def _stats(ids):
        scores  = [motif_scores[m]           for m in ids]
        impacts = [motif_impacts[m]['impact'] for m in ids]
        smarts  = [motif_impacts[m].get('motif_smarts', '?') for m in ids]
        return scores, impacts, smarts

    top_s,   top_imp,    top_smarts    = _stats(top_ids)
    bot_s,   bot_imp,    bot_smarts    = _stats(bottom_ids)

    top_mean    = float(np.mean(top_imp))
    bottom_mean = float(np.mean(bot_imp))
    ratio = (top_mean / bottom_mean) if bottom_mean > 1e-9 else float('nan')

    return {
        'k':                 actual_k,
        'top_k_ids':         top_ids,
        'bottom_k_ids':      bottom_ids,
        'top_k_scores':      top_s,
        'bottom_k_scores':   bot_s,
        'top_k_impacts':     top_imp,
        'bottom_k_impacts':  bot_imp,
        'top_mean_score':    float(np.mean(top_s)),
        'bottom_mean_score': float(np.mean(bot_s)),
        'top_mean_impact':   top_mean,
        'bottom_mean_impact': bottom_mean,
        'impact_ratio':      ratio,
        'top_k_smarts':      top_smarts,
        'bottom_k_smarts':   bot_smarts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. GT vs outside-GT motif evaluation
# ─────────────────────────────────────────────────────────────────────────────

def gt_vs_outside_gt_eval(
    motif_scores: Dict[int, float],
    motif_impacts: Dict[int, Dict[str, float]],
    gt_motif_ids: Set[int],
    data_list: List[Data],
    model: torch.nn.Module,
    vocab: VocabData,
    device: torch.device,
    split: str = 'test',
    task_type: str = 'BinaryClass',
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Compare GT vs non-GT motifs across three example subsets.

    GT motifs are the ground-truth explanatory motifs for the dataset
    (e.g. {benzene_ring_motif_id} for Benzene, {nitro_motif_id} for
    Mutagenicity).

    Three subsets
    -------------
    all              — all test examples
    class1           — all examples with true label = 1
    correct_class1   — examples with true label = 1 AND p̂ > threshold

    For each subset, reports:
      gt_mean_impact     mean impact of GT motifs
      non_gt_mean_impact mean impact of non-GT motifs
      gt_mean_score      mean learned score of GT motifs
      non_gt_mean_score  mean learned score of non-GT motifs
      score_auc          AUC using learned score to separate GT(1) vs non-GT(0)
                         across all motifs with both score and impact data
      gt_impact_rank     mean rank (1 = highest) of GT motifs by impact

    Parameters
    ----------
    gt_motif_ids : set[int]
        Motif vocab ids considered ground-truth explanatory motifs.
    threshold : float
        Probability threshold for "correctly predicted" (default 0.5).
    """
    model.eval()
    mask_cache = _resolve_mask_cache(vocab, data_list, split)
    smi_to_data = {d.smiles: d for d in data_list}
    orig_probs = _get_probs(model, data_list, device, task_type)
    mask_nodes, mask_edges = _injection_modes(model)

    # Partition data_list into the three subsets by smiles
    all_smiles        = {d.smiles for d in data_list}
    class1_smiles     = {d.smiles for d in data_list
                         if _true_label(d) == 1}
    correct1_smiles   = {smi for smi in class1_smiles
                         if orig_probs.get(smi, 0.0) > threshold}

    subsets = {
        'all':            all_smiles,
        'class1':         class1_smiles,
        'correct_class1': correct1_smiles,
    }

    common_motifs = sorted(set(motif_scores) & set(motif_impacts))
    gt_ids     = [m for m in common_motifs if m in gt_motif_ids]
    non_gt_ids = [m for m in common_motifs if m not in gt_motif_ids]

    def _subset_impacts(motif_id: int, allowed_smiles: Set[str]) -> List[float]:
        """Impacts of motif_id restricted to allowed_smiles."""
        d = smi_to_data
        mc = mask_cache.get(motif_id, {})
        probs_orig = orig_probs
        impacts = []
        for smi, bool_mask in mc.items():
            if smi not in allowed_smiles:
                continue
            data = d.get(smi)
            orig_p = probs_orig.get(smi)
            if data is None or orig_p is None:
                continue
            masked = _ablate_motif(data, bool_mask, mask_nodes, mask_edges)
            if masked is None:
                continue
            impacts.append(abs(orig_p - _single_prob(model, masked, device, task_type)))
        return impacts

    def _mean_safe(vals):
        return float(np.mean(vals)) if vals else float('nan')

    # Score-level AUC: GT motifs = 1, non-GT = 0 (independent of subset)
    score_auc = float('nan')
    if gt_ids and non_gt_ids:
        labels = [1] * len(gt_ids) + [0] * len(non_gt_ids)
        scores = ([motif_scores[m] for m in gt_ids]
                  + [motif_scores[m] for m in non_gt_ids])
        try:
            score_auc = float(roc_auc_score(labels, scores))
        except ValueError:
            pass

    # Impact rank of GT motifs among all motifs (lower = higher importance)
    all_impacts_sorted = sorted(
        common_motifs,
        key=lambda m: motif_impacts[m]['impact'],
        reverse=True,
    )
    rank_map = {m: i + 1 for i, m in enumerate(all_impacts_sorted)}
    gt_impact_rank = _mean_safe([rank_map[m] for m in gt_ids])

    results = {}
    for subset_name, allowed in subsets.items():
        gt_impacts     = [_mean_safe(_subset_impacts(m, allowed)) for m in gt_ids]
        non_gt_impacts = [_mean_safe(_subset_impacts(m, allowed)) for m in non_gt_ids]

        # Filter out NaN entries from per-motif means
        gt_imp_clean     = [v for v in gt_impacts     if not np.isnan(v)]
        non_gt_imp_clean = [v for v in non_gt_impacts if not np.isnan(v)]

        results[subset_name] = {
            'n_examples':          len(allowed),
            'n_gt_motifs':         len(gt_ids),
            'n_non_gt_motifs':     len(non_gt_ids),
            'gt_mean_impact':      _mean_safe(gt_imp_clean),
            'non_gt_mean_impact':  _mean_safe(non_gt_imp_clean),
            'gt_mean_score':       _mean_safe([motif_scores[m] for m in gt_ids]),
            'non_gt_mean_score':   _mean_safe([motif_scores[m] for m in non_gt_ids]),
            'score_auc':           score_auc,        # same for all subsets
            'gt_impact_rank':      gt_impact_rank,   # same for all subsets
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 5. Explainer ROC vs ground truth
#
# Two levels:
#   node-level (default) -- for node-attention models (GSAT motif variants)
#   edge-level           -- for base GSAT with learn_edge_att=True
#
# Node-level directly asks: did the model assign high attention to GT atoms?
# Edge-level (att[src]*att[dst]) inflates the random baseline because edges
# sharing only one endpoint with a GT node score positively even when the
# model assigned low attention to the GT atom itself.
# ─────────────────────────────────────────────────────────────────────────────


def _gt_node_mask(
    edge_label: torch.Tensor,
    edge_index: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """Derive a node-level GT bool mask from edge_label.

    A node is GT-positive if it is an endpoint of at least one GT edge.
    """
    gt = torch.zeros(n_nodes, dtype=torch.bool)
    pos = edge_label.view(-1).bool()
    src, dst = edge_index
    if pos.size(0) != src.size(0):
        raise ValueError(
            f"edge_label length ({pos.size(0)}) does not match number of edges "
            f"({src.size(0)}). edge_label must be sized [E], one entry per edge "
            f"(as produced by apply_gt.py)."
        )
    gt[src[pos]] = True
    gt[dst[pos]] = True
    return gt


@torch.no_grad()
def explainer_roc_vs_gt(
    node_att: torch.Tensor,
    edge_index: torch.Tensor,
    edge_label: torch.Tensor,
    level: str = 'node',
    edge_score: bool = False,
    node_label: torch.Tensor = None,
) -> float:
    """ROC-AUC between explainer attention and ground truth.

    Parameters
    ----------
    node_att : Tensor [N] or [E]
        Per-node attention values from the explainer.  When ``edge_score=True``
        this is interpreted as a per-EDGE score tensor [E] instead.
    edge_index : Tensor [2, E]
    edge_label : Tensor [E]
        Float ground truth -- 1.0 for GT edges, 0.0 otherwise.
        Produced by apply_gt.py (stored as data.edge_label, AND of endpoints).
    level : str
        'node' (default) -- compare node_att against the node-level GT.  When
        ``node_label`` is given it is used directly (authoritative); otherwise
        the node GT is derived from edge_label (legacy/back-compat).  Use for
        all node-attention models.

        'edge' -- compare edge-level attention against edge_label directly.
        Use only for base GSAT with learn_edge_att=True.
    edge_score : bool
        Only meaningful when level='edge'.  If True, ``node_att`` is already a
        per-edge score [E] (e.g. the model's soft edge attention) and is used
        directly.  If False (legacy), edge scores are derived from node
        attention as att[src]*att[dst].
    node_label : Tensor [N] or None
        Explicit node-level GT (1.0 = rule-motif atom), as produced by
        apply_gt.py (data.node_label).  Preferred for level='node'.

    Returns
    -------
    float  AUC, or NaN if only one class present in GT.
    """
    if level == 'node':
        y_score = node_att.view(-1).cpu().numpy()
        if node_label is not None:
            y_true = node_label.view(-1).cpu().numpy().astype(float)
        else:
            n = node_att.view(-1).size(0)
            gt_nodes = _gt_node_mask(edge_label, edge_index, n)
            y_true = gt_nodes.cpu().numpy().astype(float)
    else:
        if edge_score:
            y_score = node_att.view(-1).cpu().numpy()
        else:
            src, dst = edge_index
            y_score = (node_att.view(-1)[src] * node_att.view(-1)[dst]).cpu().numpy()
        y_true  = edge_label.view(-1).cpu().numpy()

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float('nan')
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float('nan')


def model_node_att_fn(model: torch.nn.Module, device: torch.device):
    """Return a callable ``data -> Tensor[N]`` giving the model's per-node
    attention, mirroring ``compute_gt_roc``'s internal extraction.

    Prefers the noise-free ``node_att_soft`` from the aux dict when present,
    falling back to ``out[1]``.  Returns ``None`` (inside the callable) when the
    model exposes no node attention (e.g. VanillaGNN), so callers can skip.
    """
    @torch.no_grad()
    def _fn(data: Data):
        data_dev = data.to(device)
        batch = getattr(data_dev, 'batch', None)
        if batch is None:
            batch = torch.zeros(data_dev.x.size(0), dtype=torch.long, device=device)
        out = model(data_dev.x, data_dev.edge_index, batch,
                    getattr(data_dev, 'nodes_to_motifs', None),
                    getattr(data_dev, 'edge_attr', None))
        node_att = None
        if len(out) >= 3 and isinstance(out[2], dict) \
                and out[2].get('node_att_soft') is not None:
            node_att = out[2]['node_att_soft']
        if node_att is None and len(out) >= 2:
            node_att = out[1]
        if node_att is None:
            return None
        return node_att.view(-1)
    return _fn


def motif_broadcast_att_fn(base_fn, agg: str = 'mean'):
    """Wrap a per-node attribution fn so node scores are aggregated to motif
    level (``mean`` or ``max``) over each motif's atoms and broadcast back.

    This is the node→motif reduction used everywhere else in the evaluation
    (score-vs-impact, plots): every atom of a motif instance receives that
    motif's aggregated attribution.  Because ``node_label`` is uniform within a
    motif, scoring the broadcast attribution at node level is a motif-granular
    GT-ROC in the requested aggregation flavour.

    Atoms with ``nodes_to_motifs < 0`` (no motif assignment) keep their raw
    per-node score.  Returns ``None`` when ``base_fn`` returns ``None`` so
    ``compute_gt_roc`` can skip.
    """
    if agg not in ('mean', 'max'):
        raise ValueError(f"agg must be 'mean' or 'max', got {agg!r}")

    def _fn(data: Data):
        att = base_fn(data)
        if att is None:
            return None
        att = att.view(-1)
        n2m = getattr(data, 'nodes_to_motifs', None)
        if n2m is None:
            return att
        n2m = n2m.view(-1).to(att.device)
        out = att.clone()
        for mid in torch.unique(n2m).tolist():
            if mid < 0:
                continue
            sel = n2m == mid
            vals = att[sel]
            if vals.numel() == 0:
                continue
            out[sel] = vals.mean() if agg == 'mean' else vals.max()
        return out
    return _fn


def compute_gt_roc(
    model: torch.nn.Module,
    data_list: List[Data],
    device: torch.device,
    node_att_fn=None,
    level: str = 'node',
) -> Dict[str, float]:
    """Compute mean explainer ROC-AUC vs ground truth across all test graphs.

    Each graph that has ``data.edge_label`` set (by ``apply_gt.py``)
    and at least one positive and one negative edge contributes one AUC value.
    Graphs without ``edge_label`` or with degenerate labels are skipped.

    Parameters
    ----------
    model : nn.Module
        Must return ``(logits, node_att, ...)`` or ``(logits, node_att)``.
        ``node_att`` should be ``[N, 1]`` or ``[N]``.
        If it returns ``None`` for node_att, ``node_att_fn`` is used instead.
    data_list : list of Data
        Test Data objects with ``data.edge_label`` already attached.
    device : torch.device
    node_att_fn : callable or None
        Alternative: ``node_att_fn(data) -> Tensor [N]``.  Used when the
        model's second return value is None (e.g. VanillaGNN + post-hoc).

    Returns
    -------
    dict with:
        auc_mean      float   mean per-graph AUC across valid graphs
        auc_std       float   standard deviation
        n_graphs      int     graphs with valid GT edge labels
        n_skipped     int     graphs skipped (no edge_label or degenerate)
    """
    model.eval()
    aucs = []
    n_skipped = 0

    for data in data_list:
        edge_label = getattr(data, 'edge_label', None)
        node_label = getattr(data, 'node_label', None)

        # GT vector for THIS level (used for the degeneracy/skip check).
        # node level prefers the authoritative node_label, falling back to the
        # legacy mask derived from edge_label for older caches.
        if level == 'node':
            if node_label is not None:
                gt_vec = node_label.view(-1)
            elif edge_label is not None:
                gt_vec = _gt_node_mask(
                    edge_label, data.edge_index, data.num_nodes).float().view(-1)
            else:
                n_skipped += 1
                continue
        else:  # edge level
            if edge_label is None:
                n_skipped += 1
                continue
            gt_vec = edge_label.view(-1)

        _s = float(gt_vec.sum().item())
        if _s == 0 or _s == gt_vec.numel():
            n_skipped += 1
            continue

        data_dev = data.to(device)

        # Get node attention
        _is_edge_score = False
        if node_att_fn is not None:
            _na = node_att_fn(data_dev)
            if _na is None:
                n_skipped += 1
                continue
            node_att = _na.view(-1)
        else:
            out = model(data_dev.x, data_dev.edge_index,
                        getattr(data_dev, 'batch',
                                torch.zeros(data_dev.x.size(0),
                                            dtype=torch.long, device=device)),
                        getattr(data_dev, 'nodes_to_motifs', None),
                        getattr(data_dev, 'edge_attr', None))
            # Handle (logits,), (logits, att), (logits, att, aux_dict).
            # Prefer the clean soft attention from aux when present: model att is
            # a soft sigmoid gate in (0,1), but at train time it carries injected
            # logistic noise. node_att_soft / edge_att_soft are the noise-free
            # probabilities, giving cleaner ROC ranking. Falls back to out[1] for
            # models (e.g. VanillaGNN) that don't expose the soft keys.
            node_att_raw = None
            if len(out) >= 3 and isinstance(out[2], dict):
                aux = out[2]
                if level == 'edge' and aux.get('edge_att_soft') is not None:
                    node_att_raw = aux['edge_att_soft']
                    _is_edge_score = True
                elif aux.get('node_att_soft') is not None:
                    node_att_raw = aux['node_att_soft']
            if node_att_raw is None:
                node_att_raw = out[1] if len(out) >= 2 else None
            if node_att_raw is None:
                n_skipped += 1
                continue
            node_att = node_att_raw.view(-1)

        auc = explainer_roc_vs_gt(
            node_att, data_dev.edge_index,
            edge_label.to(device) if edge_label is not None else None,
            level=level, edge_score=_is_edge_score,
            node_label=node_label.to(device) if node_label is not None else None,
        )
        if not (auc != auc):   # not NaN
            aucs.append(auc)
        else:
            n_skipped += 1

    if not aucs:
        return {
            'auc_mean': float('nan'),
            'auc_std':  float('nan'),
            'n_graphs': 0,
            'n_skipped': n_skipped,
        }
    return {
        'auc_mean': float(np.mean(aucs)),
        'auc_std':  float(np.std(aucs)),
        'n_graphs': len(aucs),
        'n_skipped': n_skipped,
    }

