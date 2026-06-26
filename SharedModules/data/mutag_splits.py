"""mutag_splits.py — train/valid/test indexing for the Mutag dataset.

Disjoint random split (default 80% / 10% / 10%), matching the CSV ``group``
column written by ``export_mutag_dataset_to_csv.py``.

Ground-truth ROC (GT-ROC) is **not** defined by the split itself: trainers pass
the held-out **test** loader to ``EvalPipeline``, and GT-ROC is computed only on
test mutagens with source motif labels (``y == 0``, ``edge_label.sum() > 0``).
See :func:`mutag_gt_eval_graphs`.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

DEFAULT_SPLITS = {'train': 0.8, 'valid': 0.1, 'test': 0.1}


def get_mutag_split_idx(
    dataset,
    splits: Optional[Dict[str, float]] = None,
    seed: int = 0,
) -> Dict[str, List[int]]:
    """Return disjoint ``{train, valid, test}`` index lists for mutag.

    Parameters
    ----------
    dataset
        ``datasets.mutag.Mutag`` (or any indexable dataset).
    splits
        Fractions for train / valid; test receives the remainder.
    seed
        RNG seed for the shuffle (use ``seed + fold`` for multi-fold exports).
    """
    if splits is None:
        splits = DEFAULT_SPLITS

    rng = np.random.RandomState(int(seed))
    idx = np.arange(len(dataset))
    rng.shuffle(idx)

    n_train = int(splits['train'] * len(idx))
    n_valid = int(splits['valid'] * len(idx))
    train_idx = idx[:n_train].tolist()
    valid_idx = idx[n_train:n_train + n_valid].tolist()
    test_idx = idx[n_train + n_valid:].tolist()

    return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}


def exclude_graph_ids_from_splits(
    split_idx: Dict[str, List[int]],
    exclude: Sequence[int],
) -> Dict[str, List[int]]:
    """Remove graph indices from every split (e.g. SMILES conversion failures)."""
    bad = {int(x) for x in exclude}
    return {k: [i for i in v if i not in bad] for k, v in split_idx.items()}


def group_for_graph(
    graph_id: int,
    split_idx: Dict[str, Sequence[int]],
) -> str:
    """Map a graph index → CSV ``group`` column (training | valid | test)."""
    if graph_id in split_idx['train']:
        return 'training'
    if graph_id in split_idx['valid']:
        return 'valid'
    if graph_id in split_idx['test']:
        return 'test'
    return 'training'


def mutag_gt_eval_graphs(data_list: Sequence) -> List:
    """Held-out test mutagens with annotated source motif GT.

    Used for GT-ROC on mutag: ``y == 0`` (mutagen) and ``edge_label.sum() > 0``.
    Non-mutagen test graphs are excluded (their labels are all zero).
    """
    out = []
    for d in data_list:
        if float(d.y.squeeze()) != 0.0:
            continue
        edge_label = getattr(d, 'edge_label', None)
        if edge_label is None or float(edge_label.sum()) <= 0.0:
            continue
        out.append(d)
    return out


def save_mutag_splits(path: Union[str, Path], split_idx: Dict[str, List[int]],
                      *, seed: int) -> None:
    """Persist split indices next to the fold CSV for reproducible loading."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'split_idx': split_idx, 'seed': seed}, f)


def load_mutag_splits(path: Union[str, Path]) -> Dict[str, List[int]]:
    """Load ``split_idx`` from a pickle written by :func:`save_mutag_splits`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mutag splits file not found: {path}")
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    if isinstance(payload, dict):
        if payload.get('mutag_x'):
            warnings.warn(
                f"{path} was saved with legacy mutag_x=True (GSAT overlapping "
                "splits). Re-run export_mutag_dataset_to_csv.py for disjoint "
                "80/10/10 splits.",
                stacklevel=2,
            )
        if 'split_idx' in payload:
            return payload['split_idx']
    return payload
