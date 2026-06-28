"""dataset.py — PyG InMemoryDataset for molecular property prediction.

Builds Data objects with:
    x              float [N, 52]   atom one-hot features (51 elements + wildcard;
                                    or raw for linear encoder)
    edge_index     long  [2, E]
    edge_attr      float [E, 8]    bond type (4) + stereo (4)
    y              float           label(s)
    smiles         str             original CSV SMILES (lookup key)
    nodes_to_motifs long [N]       motif_id per atom, -1 = unknown

The dataset is keyed by the exact CSV SMILES string, never re-canonicalised,
so atom indices match the vocabulary lookup.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

# ─────────────────────────────────────────────────────────────────────────────
# Atom / bond vocabulary
# ─────────────────────────────────────────────────────────────────────────────

ATOMS: Dict[str, int] = {
    'H': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'F': 5, 'P': 6,
    'Cl': 7, 'Br': 8, 'I': 9, 'Cu': 10, 'Bi': 11, 'B': 12,
    'Zn': 13, 'Hg': 14, 'Ti': 15, 'Fe': 16, 'Au': 17, 'Mn': 18,
    'Tl': 19, 'As': 20, 'Ca': 21, 'Si': 22, 'Co': 23, 'Al': 24,
    'Na': 25, 'Ni': 26, 'K': 27, 'Sn': 28, 'Cr': 29, 'Dy': 30,
    'Zr': 31, 'Sb': 32, 'In': 33, 'Yb': 34, 'Nd': 35, 'Be': 36,
    'Se': 37, 'Cd': 38, 'Li': 39, 'Mg': 40, 'Pt': 41, 'Gd': 42,
    'V': 43, 'Ge': 44, 'Mo': 45, 'Ag': 46, 'Ba': 47, 'Pb': 48,
    'Sr': 49, 'Pd': 50,
    # wildcard/dummy atom mapped to a dedicated index
    '*': 51,
}
NUM_ATOM_TYPES = 52   # 51 real elements + 1 wildcard

BONDS = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}
STEREO = {
    'STEREOZ': 0, 'STEREOE': 1, 'STEREOANY': 2, 'STEREONONE': 3,
}
EDGE_FEAT_DIM = len(BONDS) + len(STEREO)   # 8


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def _atom_features(mol) -> Optional[torch.Tensor]:
    """Return float [N, NUM_ATOM_TYPES] one-hot tensor, or None if unknown atom."""
    indices = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym == '*':
            idx = ATOMS['*']
        else:
            idx = ATOMS.get(sym)
            if idx is None:
                return None
        indices.append(idx)
    return F.one_hot(torch.tensor(indices), num_classes=NUM_ATOM_TYPES).float()


def _edge_features(bond) -> Optional[torch.Tensor]:
    """Return float [8] edge feature tensor, or None if unknown bond/stereo."""
    bidx = BONDS.get(bond.GetBondType())
    if bidx is None:
        return None
    sidx = STEREO.get(str(bond.GetStereo()))
    if sidx is None:
        return None
    boh = F.one_hot(torch.tensor(bidx), num_classes=len(BONDS))
    soh = F.one_hot(torch.tensor(sidx), num_classes=len(STEREO))
    return torch.cat([boh, soh], dim=-1).float()


def build_graph(
    smiles: str,
    y: torch.Tensor,
    lookup: Optional[Dict[str, Dict[int, Tuple[str, int]]]],
) -> Optional[Data]:
    """Build a PyG Data object from a SMILES string.

    Parameters
    ----------
    smiles : str
        Exact CSV SMILES (never re-canonicalised).
    y : Tensor
        Label tensor.
    lookup : dict or None
        {smiles: {node_idx: (smarts, motif_id)}}.  None → all nodes get -1.

    Returns None if the molecule cannot be parsed or contains unknown atoms/bonds.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = _atom_features(mol)
    if x is None:
        return None

    rows, cols = np.nonzero(GetAdjacencyMatrix(mol))
    edge_index = torch.stack([
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(cols, dtype=torch.long),
    ], dim=0)

    edge_feats = []
    for i, j in zip(rows.tolist(), cols.tolist()):
        bond = mol.GetBondBetweenAtoms(i, j)
        ef = _edge_features(bond)
        if ef is None:
            return None
        edge_feats.append(ef)

    edge_attr = (
        torch.stack(edge_feats, dim=0) if edge_feats
        else torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float)
    )

    n = x.size(0)
    nodes_to_motifs = torch.full((n,), -1, dtype=torch.long)
    if lookup is not None:
        node_map = lookup.get(smiles, {})
        for node_idx, (_, mid) in node_map.items():
            if 0 <= node_idx < n:
                nodes_to_motifs[node_idx] = mid

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        smiles=smiles,
        nodes_to_motifs=nodes_to_motifs,
    )




# ─────────────────────────────────────────────────────────────────────────────
# OGB molecular graph dataset
# ─────────────────────────────────────────────────────────────────────────────

OGB_NODE_FEAT_DIM = 9    # raw integer columns per atom
OGB_EDGE_FEAT_DIM = 3    # raw integer columns per bond

# OGB datasets accessible via ogb.graphproppred.PygGraphPropPredDataset.
# Node features: [N, 9] integer tensor — see table in dataset.py docstring.
# AtomEncoder / BondEncoder (from ogb) project these to [N, hidden_dim].
OGB_DATASETS = {
    'ogbg-molhiv':      {'task': 'BinaryClass',  'num_classes': 1},
    'ogbg-molbace':     {'task': 'BinaryClass',  'num_classes': 1},
    'ogbg-molbbbp':     {'task': 'BinaryClass',  'num_classes': 1},
    'ogbg-molclintox':  {'task': 'MultiLabel',   'num_classes': 2},
    'ogbg-moltox21':    {'task': 'MultiLabel',   'num_classes': 12},
    'ogbg-molsider':    {'task': 'MultiLabel',   'num_classes': 27},
    'ogbg-molesol':     {'task': 'Regression',   'num_classes': 1},
    'ogbg-molfreesolv': {'task': 'Regression',   'num_classes': 1},
    'ogbg-mollipo':     {'task': 'Regression',   'num_classes': 1},
}


def load_ogb_dataset(data_dir: str, dataset_name: str):
    """Load an OGB molecular dataset with SMILES attached.

    Requires ``ogb`` to be installed:  pip install ogb

    Parameters
    ----------
    data_dir : str
        Root directory passed to PygGraphPropPredDataset.
    dataset_name : str
        OGB name, e.g. ``'ogbg-molhiv'``.  On-disk cache dir uses underscores
        (``ogbg_molhiv``); the PyG API name stays hyphenated.

    Returns
    -------
    dataset : PygGraphPropPredDataset (with .smiles on each Data)
    split_idx : dict with keys 'train', 'valid', 'test'
    """
    try:
        from ogb.graphproppred import PygGraphPropPredDataset
    except ImportError:
        raise ImportError(
            "OGB not installed.  Run: pip install ogb")

    # OGB expects hyphens; normalise underscores
    name_hyphen = dataset_name.replace('_', '-')
    dataset = PygGraphPropPredDataset(root=data_dir, name=name_hyphen)
    split_idx = dataset.get_idx_split()

    # Attach SMILES from mapping/mol.csv.gz (OGB guarantees row i = graph i)
    from .dataset_routing import resolve_ogb_mol_csv
    mol_csv = resolve_ogb_mol_csv(data_dir, dataset_name)
    if mol_csv is not None:
        smiles_df = pd.read_csv(str(mol_csv))
        if 'smiles' in smiles_df.columns and len(smiles_df) == len(dataset):
            for i, smi in enumerate(smiles_df['smiles']):
                dataset[i].smiles = str(smi)

    return dataset, split_idx

# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class MolDataset(InMemoryDataset):
    """Molecular property prediction dataset backed by a fold CSV.

    Parameters
    ----------
    root : str
        Directory for the processed `.pt` file.
    csv_file : str
        Path to ``{dataset}_{fold}.csv``.  Must have columns
        ``smiles``, ``group``, and a label column.
    split : str
        One of ``'training'``, ``'valid'``, ``'test'``.
    label_col : str
        Name of the label column (e.g. ``'Mutagenicity'``).
    normalize : bool
        Normalise labels to zero mean / unit variance (for regression).
    mean, std : float or None
        Pre-computed statistics.  If None and normalize=True, computed from
        the current split's labels.
    lookup : dict or None
        Vocabulary lookup dict; if None all nodes get motif_id = -1.
    num_classes : int or None
        Overrides the automatic ``num_classes`` property.
    force_reprocess : bool
        Delete and re-build the cached ``.pt`` file.
    """

    def __init__(
        self,
        root: str,
        csv_file: str,
        split: str,
        label_col: str,
        normalize: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        lookup: Optional[Dict] = None,
        num_classes: Optional[int] = None,
        force_reprocess: bool = False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.csv_file = csv_file
        self.split = split
        self.label_col = label_col
        self.lookup = lookup
        self._num_classes = num_classes
        self.normalize = normalize

        df = pd.read_csv(csv_file)
        df = df[df['group'] == split].reset_index(drop=True)
        self._smiles = df['smiles'].values
        self._labels = df[label_col].values

        # MolDataset reads a single label column and ``process`` builds one
        # scalar target per graph. A multi-label dataset (num_classes > 1) cannot
        # be represented this way without silently truncating to one task, so
        # fail loud rather than train/evaluate on a wrong target. Multi-label
        # datasets must go through the OGB loader path instead.
        if num_classes is not None and num_classes > 1:
            raise NotImplementedError(
                f"MolDataset reads a single label column ({label_col!r}) and "
                f"cannot represent a multi-label target (num_classes={num_classes}). "
                f"Use the OGB loader path for multi-label datasets, or export one "
                f"task per CSV."
            )

        if normalize:
            raw = self._labels.astype(float)
            if mean is None:
                self.mean = float(np.nanmean(raw))
            else:
                self.mean = float(mean)
            if std is None:
                self.std = float(np.nanstd(raw))
            else:
                self.std = float(std)
        else:
            self.mean = 0.0
            self.std = 1.0

        proc_dir = Path(root) / 'processed'
        pt_name = f'{split}_{Path(csv_file).stem}.pt'
        proc_file = proc_dir / pt_name

        if force_reprocess:
            proc_file.unlink(missing_ok=True)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False
        )

    @property
    def num_classes(self) -> int:
        if self._num_classes is not None:
            return self._num_classes
        return super().num_classes

    @property
    def processed_file_names(self):
        return f'{self.split}_{Path(self.csv_file).stem}.pt'

    def process(self):
        data_list = []
        skipped = 0
        for idx, smiles in enumerate(tqdm(self._smiles, desc=f'Building {self.split}')):
            raw_y = float(self._labels[idx]) if not isinstance(
                self._labels[idx], (list, np.ndarray)) else self._labels[idx]
            y = torch.tensor(raw_y, dtype=torch.float)
            if self.normalize:
                y = (y - self.mean) / (self.std + 1e-8)
            if y.dim() == 0:
                y = y.unsqueeze(0)

            data = build_graph(smiles, y, self.lookup)
            if data is None or data.num_nodes == 0:
                skipped += 1
                continue
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        if skipped:
            print(f'  [MolDataset] skipped {skipped} invalid molecules in {self.split}')
        torch.save(self.collate(data_list), self.processed_paths[0])

    @property
    def x_dim(self) -> int:
        return NUM_ATOM_TYPES

    @property
    def edge_attr_dim(self) -> int:
        return EDGE_FEAT_DIM
