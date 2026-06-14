"""loader.py — factory for train/val/test DataLoaders.

Usage
-----
    from SharedModules.data.loader import get_loaders
    loaders, meta = get_loaders(cfg, vocab)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.loader import DataLoader

from .dataset import MolDataset, NUM_ATOM_TYPES, EDGE_FEAT_DIM
from .vocab import VocabData

from .dataset_schema import DATASET_COLUMN  # unified schema (single source of truth)

from .dataset_schema import TASK_TYPE       # unified schema (single source of truth)

# OGB dataset metadata — node features come as [N, 9] integer tensor
# (OGB Schema 2), not one-hot.  x_dim is set to OGB_NODE_FEAT_DIM = 9.
OGB_DATASET_NAMES = set([
    'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp', 'ogbg-molclintox',
    'ogbg-moltox21', 'ogbg-molsider', 'ogbg-molesol', 'ogbg-molfreesolv',
    'ogbg-mollipo',
])

NUM_CLASSES: Dict[str, int] = {
    'tox21':            12,
    'ogbg-moltox21':    12,
    'ogbg-molsider':    27,
    'ogbg-molclintox':   2,
}

# mutag TUDataset node feature dimension.
# The pre-baked PKL stores a 14-dim one-hot over
# {C, N, O, F, I, Cl, Br, S, P, Na, K, Li, Ca, ?}.
# We accept this as-is and set x_dim=14 so models are initialised correctly.
MUTAG_X_DIM = 14
MUTAG_EDGE_DIM = 0   # TUDataset adjacency has no bond-type features


@dataclass
class LoaderMeta:
    x_dim: int
    edge_attr_dim: int
    num_classes: int
    task_type: str
    dataset: str
    fold: int
    node_encoder: str = 'onehot'
    # 'onehot'       — identity passthrough (x is already one-hot, x_dim dims)
    # 'atom_encoder' — OGB AtomEncoder  (x is [N,9] integer, ogbg-mol* datasets)
    # 'linear'       — Linear(x_dim → hidden) + LayerNorm  (explicit projection)
    deg: Optional[torch.Tensor] = None
    # Degree histogram [max_deg+1] computed from training set.
    # Required for PNA backbone; None for all others.


# ── mutag TUDataset ──────────────────────────────────────────────────────────

class MutagTUDataset(torch.utils.data.Dataset):
    """Wraps a list of mutag PyG Data objects, optionally attaching
    ``nodes_to_motifs`` from a pre-computed index_map + vocab lookup.

    The 14-dim node features (``data.x``) are kept exactly as loaded from
    the TUDataset PKL — no re-encoding to the 52-dim one-hot.

    Parameters
    ----------
    data_list : list of PyG Data
        Loaded from ``datasets.mutag.Mutag`` (must have .x, .y, .node_type,
        .edge_index).
    vocab : VocabData or None
        Vocabulary produced by MotifBreakdown.  If None, all nodes get -1.
    index_maps : dict or None
        ``{mapped_smiles: {graph_node_idx: smiles_atom_idx}}`` produced by
        ``build_mutag_smiles_df()``.  Required when vocab is not None.
    smiles_list : list[str] or None
        Mapped SMILES string for each graph (same order as data_list).
        Required when vocab is not None.
    split : str
        'training', 'valid', or 'test' — selects which lookup to use from vocab.
    """

    def __init__(
        self,
        data_list: List,
        vocab: Optional[VocabData] = None,
        index_maps: Optional[Dict] = None,
        smiles_list: Optional[List[str]] = None,
        split: str = 'training',
    ):
        self._data = data_list
        self._vocab = vocab
        self._index_maps = index_maps or {}
        self._smiles = smiles_list or [None] * len(data_list)
        self._split = split

        if vocab is not None:
            self._lookup = vocab.lookup_for_split(split)
        else:
            self._lookup = {}

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        from .graph_to_smiles import apply_motif_lookup_with_index_map
        data = self._data[idx].clone()
        n = data.x.size(0)

        if self._vocab is None or not self._smiles[idx]:
            data.nodes_to_motifs = torch.full((n,), -1, dtype=torch.long)
            return data

        mapped_smi = self._smiles[idx]
        data.nodes_to_motifs = apply_motif_lookup_with_index_map(
            n, mapped_smi, self._lookup, self._index_maps
        )
        return data


def _get_mutag_loaders(
    data_root: str,
    vocab: Optional[VocabData],
    batch_size: int,
    num_workers: int,
    index_maps_path: Optional[str] = None,
    smiles_csv_path: Optional[str] = None,
):
    """Build loaders for the mutag TUDataset.

    Uses the 14-dim pre-baked node features from the PKL directly.
    Attaches ``nodes_to_motifs`` via the atom-map index_map if a vocab
    and ``index_maps_path`` are provided.

    Parameters
    ----------
    data_root : str
        Directory containing the ``mutag/`` TUDataset folder.
    vocab : VocabData or None
    index_maps_path : str or None
        Path to ``mutag_0_index_maps.pkl`` produced by
        ``build_mutag_smiles_df()``.  Required to attach motif annotations.
    smiles_csv_path : str or None
        Path to the ``mutag_0.csv`` produced by ``export_mutag_dataset_to_csv.py``
        (columns: smiles, label, group, graph_id).  Used to recover the
        per-graph mapped SMILES and split assignments.
    """
    try:
        import sys
        from pathlib import Path as _P
        _src = str(_P(data_root).parent / 'src')
        if _src not in sys.path:
            sys.path.insert(0, _src)
        from datasets.mutag import Mutag
    except ImportError:
        raise ImportError(
            "Cannot import datasets.mutag.Mutag. "
            "Ensure the MotifSAT src directory is on PYTHONPATH or pass "
            "data_root pointing to a directory with datasets/ accessible.")

    dataset = Mutag(root=str(Path(data_root) / 'mutag'))

    # Load index_maps and smiles_csv if provided
    index_maps: Dict = {}
    smiles_by_graph: Dict[int, str] = {}  # graph_id → mapped_smiles
    split_by_graph: Dict[int, str] = {}   # graph_id → split name

    if index_maps_path and Path(index_maps_path).exists():
        with open(index_maps_path, 'rb') as f:
            index_maps = pickle.load(f)

    if smiles_csv_path and Path(smiles_csv_path).exists():
        import pandas as pd
        df_smi = pd.read_csv(smiles_csv_path)
        for _, row in df_smi.iterrows():
            gid = int(row['graph_id'])
            smiles_by_graph[gid] = str(row['smiles'])
            split_by_graph[gid]  = str(row.get('group', 'training'))

    # Partition by split
    train_items: List[Tuple[int, object]] = []
    val_items:   List[Tuple[int, object]] = []
    test_items:  List[Tuple[int, object]] = []

    for i in range(len(dataset)):
        grp = split_by_graph.get(i, 'training')
        if grp == 'valid':
            val_items.append(i)
        elif grp == 'test':
            test_items.append(i)
        else:
            train_items.append(i)

    # When no CSV is provided, use a fixed 80/10/10 split
    if not smiles_csv_path:
        n = len(dataset)
        n_train = int(0.8 * n)
        n_val   = int(0.1 * n)
        train_items = list(range(n_train))
        val_items   = list(range(n_train, n_train + n_val))
        test_items  = list(range(n_train + n_val, n))

    def _build_ds(indices, split_name):
        data_list   = [dataset[i] for i in indices]
        smiles_list = [smiles_by_graph.get(i) for i in indices]
        return MutagTUDataset(
            data_list, vocab, index_maps, smiles_list, split=split_name)

    train_ds = _build_ds(train_items, 'training')
    val_ds   = _build_ds(val_items,   'valid')
    test_ds  = _build_ds(test_items,  'test')

    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size,
                            shuffle=True,  num_workers=num_workers),
        'valid': DataLoader(val_ds,   batch_size=batch_size,
                            shuffle=False, num_workers=num_workers),
        'test':  DataLoader(test_ds,  batch_size=batch_size,
                            shuffle=False, num_workers=num_workers),
    }
    meta = LoaderMeta(
        x_dim=MUTAG_X_DIM,
        edge_attr_dim=MUTAG_EDGE_DIM,
        num_classes=1,
        task_type='BinaryClass',
        dataset='mutag',
        fold=0,
        node_encoder='onehot',   # 14-dim pre-baked features, identity passthrough
    )
    return loaders, test_ds, meta


def _get_ogb_loaders(
    dataset: str,
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 0,
):
    """Build loaders for an OGB molecular dataset.

    Node features are the raw OGB integer tensor [N, 9].
    The model is responsible for applying AtomEncoder or a Linear projection.
    Returns (loaders, test_dataset, meta) matching the same signature as get_loaders.
    """
    from .dataset import load_ogb_dataset, OGB_NODE_FEAT_DIM, OGB_EDGE_FEAT_DIM

    ogb_dataset, split_idx = load_ogb_dataset(data_root, dataset)
    task_type = TASK_TYPE.get(dataset, 'BinaryClass')
    num_classes = NUM_CLASSES.get(dataset, 1)

    train_ds = ogb_dataset[split_idx['train']]
    val_ds   = ogb_dataset[split_idx['valid']]
    test_ds  = ogb_dataset[split_idx['test']]

    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers),
        'valid': DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers),
        'test':  DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers),
    }
    meta = LoaderMeta(
        x_dim=OGB_NODE_FEAT_DIM,
        edge_attr_dim=OGB_EDGE_FEAT_DIM,
        num_classes=num_classes,
        task_type=task_type,
        dataset=dataset,
        fold=0,
        node_encoder='atom_encoder',
    )
    return loaders, test_ds, meta


def get_loaders(
    dataset: str,
    data_root: str,
    fold: int = 0,
    vocab: Optional[VocabData] = None,
    processed_root: str = '/tmp/motif_processed',
    batch_size: int = 128,
    num_workers: int = 0,
    normalize: bool = False,
    force_reprocess: bool = False,
    mutag_index_maps_path: Optional[str] = None,
    mutag_smiles_csv_path: Optional[str] = None,
) -> Tuple[Dict[str, DataLoader], object, LoaderMeta]:
    """Build train/val/test DataLoaders for a dataset fold.

    Parameters
    ----------
    dataset : str
    data_root : str
        For CSV datasets: directory containing ``{dataset}_{fold}.csv``.
        For mutag: directory containing the ``mutag/`` TUDataset folder.
        For OGB: root passed to ``PygGraphPropPredDataset``.
    fold : int
    vocab : VocabData or None
        If None, ``nodes_to_motifs`` will be all -1 (no motif annotations).
    processed_root : str
        Root for PyG processed ``.pt`` cache files (CSV datasets only).
    batch_size : int
    normalize : bool
        Normalise labels (regression only).
    force_reprocess : bool
        Rebuild cached ``.pt`` files (CSV datasets only).
    mutag_index_maps_path : str or None
        Path to ``mutag_0_index_maps.pkl`` (mutag only).
    mutag_smiles_csv_path : str or None
        Path to ``mutag_0.csv`` exported by ``export_mutag_dataset_to_csv.py``
        (mutag only). Provides split assignments and mapped SMILES per graph.

    Returns
    -------
    loaders : dict[str, DataLoader]   keys: 'train', 'valid', 'test'
    test_dataset                       raw test dataset for evaluation
    meta : LoaderMeta                  x_dim, edge_attr_dim, task_type, ...
    """
    # ── OGB datasets ──────────────────────────────────────────────────────
    if dataset in OGB_DATASET_NAMES:
        return _get_ogb_loaders(dataset, data_root, batch_size, num_workers)

    # ── mutag TUDataset (14-dim pre-baked features) ───────────────────────
    if dataset == 'mutag':
        return _get_mutag_loaders(
            data_root, vocab, batch_size, num_workers,
            index_maps_path=mutag_index_maps_path,
            smiles_csv_path=mutag_smiles_csv_path,
        )

    # ── CSV-based molecular datasets ──────────────────────────────────────
    csv = f'{data_root}/{dataset}_{fold}.csv'
    label_col = DATASET_COLUMN[dataset]
    task_type = TASK_TYPE.get(dataset, 'BinaryClass')
    num_classes = NUM_CLASSES.get(dataset, 1)

    lookup_train = vocab.lookup_train if vocab is not None else None
    lookup_valid = vocab.lookup_valid if vocab is not None else None
    lookup_test  = vocab.lookup_test  if vocab is not None else None

    # Training split — compute normalisation stats from training data
    train_ds = MolDataset(
        root=f'{processed_root}/{dataset}_fold{fold}/train',
        csv_file=csv,
        split='training',
        label_col=label_col,
        normalize=normalize,
        lookup=lookup_train,
        num_classes=num_classes if task_type == 'MultiLabel' else None,
        force_reprocess=force_reprocess,
    )

    val_ds = MolDataset(
        root=f'{processed_root}/{dataset}_fold{fold}/valid',
        csv_file=csv,
        split='valid',
        label_col=label_col,
        normalize=normalize,
        mean=train_ds.mean if normalize else None,
        std=train_ds.std if normalize else None,
        lookup=lookup_valid,
        num_classes=num_classes if task_type == 'MultiLabel' else None,
        force_reprocess=force_reprocess,
    )

    test_ds = MolDataset(
        root=f'{processed_root}/{dataset}_fold{fold}/test',
        csv_file=csv,
        split='test',
        label_col=label_col,
        normalize=normalize,
        mean=train_ds.mean if normalize else None,
        std=train_ds.std if normalize else None,
        lookup=lookup_test,
        num_classes=num_classes if task_type == 'MultiLabel' else None,
        force_reprocess=force_reprocess,
    )

    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers),
        'valid': DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers),
        'test':  DataLoader(test_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers),
    }
    # Compute degree histogram for PNA (cheap — train_ds is already in memory)
    _deg = compute_deg_histogram(train_ds)

    meta = LoaderMeta(
        x_dim=NUM_ATOM_TYPES,
        edge_attr_dim=EDGE_FEAT_DIM,
        num_classes=num_classes,
        task_type=task_type,
        dataset=dataset,
        fold=fold,
        node_encoder='onehot',   # 52-dim atom-type one-hot, identity passthrough
        deg=_deg,
    )
    return loaders, test_ds, meta


def compute_deg_histogram(dataset) -> torch.Tensor:
    """Compute node degree histogram from a dataset for use with PNA.

    Iterates over all graphs and counts per-node in-degrees (edge_index[1]).
    Returns a LongTensor of shape [max_degree + 1] where entry d is the
    number of nodes with in-degree d across the entire dataset.

    Parameters
    ----------
    dataset : MolDataset or any iterable of PyG Data
        Should be the TRAINING split only (not val/test).

    Returns
    -------
    torch.Tensor  [max_degree + 1]  dtype=torch.long
    """
    from torch_geometric.utils import degree
    max_degree = 0
    for data in dataset:
        if data.edge_index.numel() == 0:
            continue
        d = degree(data.edge_index[1], num_nodes=data.num_nodes)
        max_degree = max(max_degree, int(d.max().item()))

    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in dataset:
        if data.edge_index.numel() == 0:
            continue
        d = degree(data.edge_index[1],
                   num_nodes=data.num_nodes).long()
        deg += torch.bincount(d, minlength=deg.numel())
    return deg


def compute_pos_weights(dataset) -> torch.Tensor:
    """Compute BCEWithLogitsLoss positive class weights from a dataset.

    For single-label: returns Tensor([n_neg / n_pos]).
    For multi-label:  returns Tensor([n_neg_c / n_pos_c]) per task.
    Accepts MolDataset, MutagTUDataset, or any iterable of PyG Data.
    """
    ys = torch.tensor([d.y.item() if d.y.numel() == 1 else d.y.tolist()
                       for d in dataset], dtype=torch.float)
    if ys.dim() == 1:
        pos = (ys == 1).sum().clamp(min=1)
        neg = (ys == 0).sum().clamp(min=1)
        return (neg / pos).unsqueeze(0)
    weights = []
    for c in range(ys.shape[1]):
        col = ys[:, c]
        valid = col[~torch.isnan(col)]
        pos = (valid == 1).sum().clamp(min=1).float()
        neg = (valid == 0).sum().clamp(min=1).float()
        weights.append(neg / pos)
    return torch.stack(weights)
