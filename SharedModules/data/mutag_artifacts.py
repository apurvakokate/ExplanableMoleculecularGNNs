"""mutag_artifacts.py — validate exported mutag CSV / splits / index_maps."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import pandas as pd

try:
    from .mutag_splits import load_mutag_splits
except ImportError:  # standalone / test isolation
    from SharedModules.data.mutag_splits import load_mutag_splits


class MutagArtifactError(ValueError):
    """Raised when mutag export artifacts are inconsistent."""


def validate_mutag_artifacts(
    csv_path: Union[str, Path],
    splits_path: Union[str, Path],
    index_maps_path: Optional[Union[str, Path]] = None,
    *,
    dataset_size: Optional[int] = None,
) -> Dict[str, object]:
    """Check CSV, splits, and index_maps are mutually consistent.

    Expected export contract (post phase-0):
    - CSV contains only successfully converted graphs (non-empty ``smiles``).
    - ``split_idx`` lists are disjoint and their union equals CSV ``graph_id``s.
    - Every CSV ``smiles`` appears as a key in ``index_maps`` (when path given).

    Parameters
    ----------
    csv_path, splits_path, index_maps_path
        Paths written by ``export_mutag_dataset_to_csv.py``.
    dataset_size
        Optional ``len(Mutag)`` — ensures split indices are in range.

    Returns
    -------
    dict with n_graphs, n_train, n_valid, n_test, graph_ids.

    Raises
    ------
    MutagArtifactError
    """
    csv_path = Path(csv_path)
    splits_path = Path(splits_path)
    if not csv_path.is_file():
        raise MutagArtifactError(f"mutag CSV not found: {csv_path}")
    if not splits_path.is_file():
        raise MutagArtifactError(f"mutag splits not found: {splits_path}")

    df = pd.read_csv(csv_path)
    for col in ('smiles', 'label', 'group', 'graph_id'):
        if col not in df.columns:
            raise MutagArtifactError(
                f"{csv_path} missing column {col!r}; got {list(df.columns)}")

    if 'conversion_ok' in df.columns:
        bad = df[~df['conversion_ok'].astype(bool)]
        if len(bad):
            raise MutagArtifactError(
                f"{csv_path} contains {len(bad)} row(s) with conversion_ok=False "
                f"(graph_ids={bad['graph_id'].tolist()[:10]}…). Re-run export.")

    empty = df['smiles'].isna() | (df['smiles'].astype(str).str.strip() == '') \
        | (df['smiles'].astype(str).str.lower() == 'nan')
    if empty.any():
        raise MutagArtifactError(
            f"{csv_path} has {int(empty.sum())} row(s) with empty smiles "
            f"(graph_ids={df.loc[empty, 'graph_id'].tolist()[:10]}…). Re-run export.")

    graph_ids: Set[int] = {int(x) for x in df['graph_id'].tolist()}
    if len(graph_ids) != len(df):
        raise MutagArtifactError(
            f"{csv_path} has duplicate graph_id values "
            f"({len(df)} rows, {len(graph_ids)} unique ids).")

    split_idx = load_mutag_splits(splits_path)
    for key in ('train', 'valid', 'test'):
        if key not in split_idx:
            raise MutagArtifactError(
                f"{splits_path} split_idx missing {key!r} key.")

    train = [int(i) for i in split_idx['train']]
    valid = [int(i) for i in split_idx['valid']]
    test = [int(i) for i in split_idx['test']]
    all_split = train + valid + test
    if len(set(all_split)) != len(all_split):
        raise MutagArtifactError(f"{splits_path} split indices are not disjoint.")

    split_set = set(all_split)
    if split_set != graph_ids:
        only_split = sorted(split_set - graph_ids)[:10]
        only_csv = sorted(graph_ids - split_set)[:10]
        raise MutagArtifactError(
            f"CSV graph_ids and splits disagree "
            f"(csv={len(graph_ids)}, splits={len(split_set)}). "
            f"in splits not csv={only_split}; in csv not splits={only_csv}. "
            f"Re-run export_mutag_dataset_to_csv.py.")

    if dataset_size is not None:
        oob = [i for i in split_set if i < 0 or i >= dataset_size]
        if oob:
            raise MutagArtifactError(
                f"split index out of range for dataset size {dataset_size}: "
                f"{oob[:10]}")

    index_maps: Dict = {}
    if index_maps_path is not None:
        index_maps_path = Path(index_maps_path)
        if not index_maps_path.is_file():
            raise MutagArtifactError(f"index_maps not found: {index_maps_path}")
        with open(index_maps_path, 'rb') as f:
            index_maps = pickle.load(f)
        csv_smiles = set(df['smiles'].astype(str))
        map_keys = set(index_maps.keys())
        if csv_smiles != map_keys:
            missing_maps = sorted(csv_smiles - map_keys)[:3]
            extra_maps = sorted(map_keys - csv_smiles)[:3]
            raise MutagArtifactError(
                f"index_maps keys != CSV smiles "
                f"(csv={len(csv_smiles)}, maps={len(map_keys)}). "
                f"missing maps e.g. {missing_maps!r}; extra e.g. {extra_maps!r}.")

    return {
        'n_graphs': len(graph_ids),
        'n_train': len(train),
        'n_valid': len(valid),
        'n_test': len(test),
        'graph_ids': graph_ids,
    }
