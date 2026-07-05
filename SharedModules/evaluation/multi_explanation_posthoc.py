"""Post-hoc H0/H1/H2 multi-explanation analysis for trained ante-hoc models."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch

from SharedModules.evaluation.multi_explanation import MultiExplanationAnalysis


def flatten_mose_scores(motif_scores) -> Optional[Dict[int, float]]:
    """Average multi-class MOSE scores to a single float per motif id."""
    if not motif_scores:
        return None
    if isinstance(motif_scores, dict) and motif_scores and isinstance(
            next(iter(motif_scores.values())), dict):
        all_scores = list(motif_scores.values())
        common_ids = set(all_scores[0].keys())
        return {
            mid: float(np.mean([sc[mid] for sc in all_scores if mid in sc]))
            for mid in common_ids
        }
    return motif_scores


def resolve_motif_scores_from_model(
    model: torch.nn.Module,
    test_list: list,
    device: torch.device,
    *,
    learn_edge_att: bool = False,
    att_aggregate_fn: Optional[Callable] = None,
    max_graphs: int = 500,
) -> Optional[Dict[int, float]]:
    """Derive global motif importance scores for multi-explanation analysis."""
    if learn_edge_att:
        return None
    if hasattr(model, 'get_motif_scores'):
        return flatten_mose_scores(model.get_motif_scores())
    if att_aggregate_fn is not None:
        agg = att_aggregate_fn(model, test_list, device,
                               learn_edge_att=learn_edge_att,
                               max_graphs=max_graphs)
        scores = (agg or {}).get('mean') or {}
        return scores if scores else None
    return None


def run_multi_explanation_posthoc(
    model: torch.nn.Module,
    vocab,
    test_list: list,
    device: torch.device,
    task_type: str,
    out_dir: Path,
    *,
    motif_scores: Optional[Dict[int, float]] = None,
    learn_edge_att: bool = False,
    att_aggregate_fn: Optional[Callable] = None,
    max_motifs: Optional[int] = None,
    local_filter: str = 'p75',
) -> bool:
    """Run H0/H1/H2 analysis and write ``multi_explanation_*.csv`` into *out_dir*.

    Fails fast: any error in the analysis propagates (no broad swallow), so a
    real bug surfaces with its traceback rather than a silent ``False``.
    """
    if learn_edge_att:
        print('  [skip multi_explanation] learn_edge_att=True (edge scores, not motif-level)')
        return False

    scores = motif_scores or resolve_motif_scores_from_model(
        model, test_list, device,
        learn_edge_att=learn_edge_att,
        att_aggregate_fn=att_aggregate_fn,
        max_graphs=max_motifs or 500,
    )
    if not scores:
        print('  [skip multi_explanation] no motif scores available')
        return False

    print('\n  Running multi-explanation analysis (post-hoc) ...')
    analysis = MultiExplanationAnalysis(
        model, vocab, test_list, device,
        motif_scores=scores,
        task_type=task_type,
        max_motifs=max_motifs,
    )
    analysis.run(local_filter=local_filter)
    analysis.save(str(out_dir / 'multi_explanation'))
    return True
