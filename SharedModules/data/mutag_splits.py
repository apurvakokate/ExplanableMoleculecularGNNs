"""mutag_splits.py — train/valid/test indexing for the Mutag dataset.

Mirrors Graph-COM/GSAT ``get_random_split_idx`` (src/utils/get_data_loaders.py).

Two modes
---------
**Standard** (``mutag_x=False``, default for property prediction):
    Random shuffle → 80% train / 10% valid / 10% test (disjoint).

**GSAT explanation** (``mutag_x=True``, GSAT default for Mutag):
    80% train / 20% valid (disjoint).
    Test = *all* graphs with ``y == 0`` (mutagen) and ``edge_label.sum() > 0``
    (annotated NO2/NH2 motif present).  Test may overlap train/valid — that is
    intentional for GSAT-style explanation evaluation on mutagenic molecules with
    source GT.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np


DEFAULT_SPLITS = {'train': 0.8, 'valid': 0.1, 'test': 0.1}
GSAT_MUTAG_X_SPLITS = {'train': 0.8, 'valid': 0.2}


def get_mutag_split_idx(
    dataset,
    splits: Optional[Dict[str, float]] = None,
    seed: int = 0,
    mutag_x: bool = False,
) -> Dict[str, List[int]]:
    """Return ``{train, valid, test}`` index lists for a Mutag InMemoryDataset.

    Parameters
    ----------
    dataset
        ``datasets.mutag.Mutag`` (or any indexable with ``.y`` and ``.edge_label``).
    splits
        Fractions for train / valid.  Remaining fraction is unused when
        ``mutag_x=True`` (test comes from GT-positive graphs).  When
        ``mutag_x=False``, test gets the remainder after train+valid.
    seed
        RNG seed for the shuffle (use ``seed + fold`` for multi-fold exports).
    mutag_x
        If True, use GSAT's explanation test set: mutagen (``y==0``) graphs
        with annotated NO2/NH2 motif edges (see module docstring).
    """
    if splits is None:
        splits = GSAT_MUTAG_X_SPLITS if mutag_x else DEFAULT_SPLITS

    rng = np.random.RandomState(int(seed))
    idx = np.arange(len(dataset))
    rng.shuffle(idx)

    if not mutag_x:
        n_train = int(splits['train'] * len(idx))
        n_valid = int(splits['valid'] * len(idx))
        train_idx = idx[:n_train].tolist()
        valid_idx = idx[n_train:n_train + n_valid].tolist()
        test_idx = idx[n_train + n_valid:].tolist()
    else:
        n_train = int(splits['train'] * len(idx))
        train_idx = idx[:n_train].tolist()
        valid_idx = idx[n_train:].tolist()
        test_idx = [
            i for i in range(len(dataset))
            if (float(dataset[i].y.squeeze()) == 0.0
                and float(dataset[i].edge_label.sum()) > 0.0)
        ]

    return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}


def group_for_graph(
    graph_id: int,
    split_idx: Dict[str, Sequence[int]],
    mutag_x: bool = False,
) -> str:
    """Map a graph index → CSV ``group`` column (training | valid | test).

    When ``mutag_x`` and a graph is in the explanation test set, it is labelled
    ``test`` even if it also appears in train/valid (vocab + GT eval priority).
    """
    if mutag_x and graph_id in split_idx['test']:
        return 'test'
    if graph_id in split_idx['train']:
        return 'training'
    if graph_id in split_idx['valid']:
        return 'valid'
    if graph_id in split_idx['test']:
        return 'test'
    return 'training'


def save_mutag_splits(path: Union[str, Path], split_idx: Dict[str, List[int]],
                      *, seed: int, mutag_x: bool) -> None:
    """Persist split indices next to the fold CSV for reproducible loading."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'split_idx': split_idx, 'seed': seed, 'mutag_x': mutag_x},
                    f)


def load_mutag_splits(path: Union[str, Path]) -> Dict[str, List[int]]:
    """Load ``split_idx`` from a pickle written by :func:`save_mutag_splits`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mutag splits file not found: {path}")
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and 'split_idx' in payload:
        return payload['split_idx']
    return payload
