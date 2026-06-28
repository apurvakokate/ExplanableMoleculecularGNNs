"""Per-fold motif threshold application for CSV CV datasets.

Contract (see project design notes):
  - SMILES → atom → motif_id mapping is mined once (fold-0 vocab, no threshold).
  - Threshold *percentage* comes from vocab_meta.json or CHOSEN_THRESHOLD
    (SharedModules/data/threshold_config.py — same source as phase-1 mining).
  - Support is re-counted on each fold's train+val; ``threshold_motifs`` and
    ``-1`` remapping can differ per fold.
  - Synthetic GT rules use pre-threshold fragmentation (``lookup_all``); only ``y``
    (and node/edge GT labels) change — ``nodes_to_motifs`` comes from the
    fold-specific threshold applied in ``get_loaders``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from .dataset_schema import TASK_TYPE
from .threshold_config import get_chosen_threshold, select_threshold_motifs


def resolve_threshold_pct(
    vocab_dir: Path,
    variant: str,
    dataset: str,
    *,
    apply_threshold: Optional[bool] = None,
    threshold_pct: Optional[float] = None,
) -> Optional[float]:
    """Return threshold fraction of N_trainval, or None if filtering is off."""
    if threshold_pct is not None:
        return float(threshold_pct)
    if apply_threshold is False:
        return None

    meta_path = vocab_dir / 'vocab_meta.json'
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if not meta.get('apply_threshold', False):
            return None
        if meta.get('threshold_pct') is not None:
            return float(meta['threshold_pct'])

    if apply_threshold is True or variant.endswith('_filter'):
        return get_chosen_threshold(variant, dataset)
    return None


def compute_threshold_motifs(
    csv_path: str,
    label_col: str,
    mol_fragment_smarts: Dict[str, List[str]],
    resolved_pct: float,
    *,
    is_regression: bool = False,
) -> Set[str]:
    """Re-apply fold-0 threshold policy on this fold's train+val support."""
    df = pd.read_csv(csv_path)
    trainval = df[df['group'].isin(('training', 'valid'))]
    n_tv = len(trainval)
    if n_tv == 0:
        return set()

    mol_counts: Counter = Counter()
    wt_counts_0: Counter = Counter()
    wt_counts_1: Counter = Counter()

    for _, row in trainval.iterrows():
        smi = str(row['smiles'])
        lbl = row[label_col]
        for smarts in mol_fragment_smarts.get(smi, []):
            mol_counts[smarts] += 1.0
            if not is_regression:
                if int(lbl) == 0:
                    wt_counts_0[smarts] += 1.0
                else:
                    wt_counts_1[smarts] += 1.0

    n0_tv = n1_tv = None
    if not is_regression:
        labels_tv = trainval[label_col].astype(int)
        n0_tv = int((labels_tv == 0).sum())
        n1_tv = n_tv - n0_tv

    return select_threshold_motifs(
        mol_counts, wt_counts_0, wt_counts_1,
        n_tv=n_tv, n0_tv=n0_tv, n1_tv=n1_tv,
        resolved_pct=resolved_pct,
        is_regression=is_regression,
    )


def apply_threshold_to_node_map(
    node_map: Dict[int, Tuple[str, int]],
    threshold_motifs: Optional[Set[str]],
) -> Dict[int, Tuple[str, int]]:
    if threshold_motifs is None:
        return node_map
    out: Dict[int, Tuple[str, int]] = {}
    for idx, (smarts, mid) in node_map.items():
        if mid < 0 or smarts not in threshold_motifs:
            out[idx] = (smarts, -1)
        else:
            out[idx] = (smarts, mid)
    return out


def apply_threshold_to_lookup(
    lookup_all: Dict[str, Dict[int, Tuple[str, int]]],
    threshold_motifs: Optional[Set[str]],
) -> Dict[str, Dict[int, Tuple[str, int]]]:
    if threshold_motifs is None:
        return lookup_all
    return {
        smi: apply_threshold_to_node_map(node_map, threshold_motifs)
        for smi, node_map in lookup_all.items()
    }


def kept_motif_ids_from_threshold(
    threshold_motifs: Optional[Set[str]],
    motif_list: List[str],
) -> Optional[List[int]]:
    if threshold_motifs is None:
        return None
    return [i for i, s in enumerate(motif_list) if s in threshold_motifs]


def build_fold_annotation(
    *,
    lookup_all: Dict[str, Dict[int, Tuple[str, int]]],
    motif_list: List[str],
    mol_fragment_smarts: Optional[Dict[str, List[str]]],
    csv_path: str,
    label_col: str,
    dataset: str,
    variant: str,
    vocab_dir: Path,
    apply_threshold: Optional[bool] = None,
    threshold_pct: Optional[float] = None,
) -> Tuple[
    Dict[str, Dict[int, Tuple[str, int]]],
    Optional[List[int]],
    Optional[Set[str]],
    Optional[float],
]:
    """Build fold-specific thresholded lookup + kept_motif_ids for training."""
    task_type = TASK_TYPE.get(dataset, 'BinaryClass')
    is_regression = task_type == 'Regression'
    resolved_pct = resolve_threshold_pct(
        vocab_dir, variant, dataset,
        apply_threshold=apply_threshold,
        threshold_pct=threshold_pct,
    )

    if resolved_pct is None:
        kept = kept_motif_ids_from_threshold(None, motif_list)
        return lookup_all, kept, None, None

    if mol_fragment_smarts is None:
        raise FileNotFoundError(
            f"{vocab_dir}: missing _mol_fragment_smarts.pickle — re-run "
            f"generate_vocab_rules.py to enable per-fold thresholding."
        )

    threshold_motifs = compute_threshold_motifs(
        csv_path, label_col, mol_fragment_smarts, resolved_pct,
        is_regression=is_regression,
    )
    kept = kept_motif_ids_from_threshold(threshold_motifs, motif_list)
    lookup = apply_threshold_to_lookup(lookup_all, threshold_motifs)
    return lookup, kept, threshold_motifs, resolved_pct
