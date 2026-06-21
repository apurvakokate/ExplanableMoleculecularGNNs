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

from .dataset_routing import OGB_DATASET_NAMES  # single source (was duplicated here)

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
    norm_mean: float = 0.0
    norm_std: float = 1.0
    # Regression target normalisation stats (z-score) computed from the TRAIN
    # split. Identity (0.0 / 1.0) when normalize=False. Used to denormalise
    # MAE/RMSE back to the original target units for reporting.


def resolve_node_encoder(cli_value: Optional[str], meta_value: str) -> str:
    """Resolve which node encoder the model should use.

    The CLI flag (--node_encoder) MUST be honored for CSV datasets so the
    feature sweep (onehot vs linear) actually takes effect — previously trainers
    built from meta.node_encoder and silently ignored the CLI value.

    OGB datasets are the exception: their node features are a [N,9] integer
    tensor that only the OGB AtomEncoder can consume, so 'atom_encoder' is forced
    regardless of the CLI value (with a warning if the user asked for something
    else). If cli_value is None/empty, fall back to the loader's recommendation.
    """
    if meta_value == 'atom_encoder':
        if cli_value and cli_value not in (None, 'atom_encoder'):
            print(f"  [node_encoder] OGB dataset requires 'atom_encoder'; "
                  f"ignoring --node_encoder={cli_value!r}.")
        return 'atom_encoder'
    if not cli_value:
        return meta_value
    if cli_value not in ('onehot', 'linear', 'atom_encoder'):
        raise ValueError(f"Unknown node_encoder {cli_value!r}; "
                         f"choose from onehot | linear | atom_encoder.")
    return cli_value


# ── Hyperparameter tagging (hybrid: readable axes + hash of the fine knobs) ──
import hashlib as _hashlib
import json as _json

# Knobs spelled out in the directory name (the ones swept most often).
# Format: short prefix -> cfg attribute name.
_HP_SPELLED = [
    ('L',    'num_layers'),
    ('h',    'hidden_dim'),
    ('glr',  'gnn_lr'),        # MOSE: GNN-backbone LR
    ('xlr',  'explainer_lr'),  # MOSE: motif-importance (explainer) LR
    ('lr',   'lr'),            # vanilla / MotifSAT single LR (used when gnn_lr absent)
]

# Fine knobs folded into the hash (regularization / fine-tuning). Anything here
# changing → a different hp-hash → a different directory. Extend freely; the
# hparams.json written alongside makes the hash fully decodable.
_HP_HASHED = [
    'weight_decay', 'dropout', 'clip_grad',
    'size_reg', 'ent_reg', 'top_tau',            # MOSE regularization
    'info_loss_coef', 'motif_loss_coef',         # MotifSAT regularization
    'between_motif_coef', 'within_node_coef',
    'pool_mode',                                 # MotifSAT readout pooling
    'init_r', 'final_r', 'logit_clamp',          # MotifSAT scheduling
    'unk_value', 'extractor_dropout_p',
]


def _fmt_num(v):
    """Compact, stable string for a numeric hyperparameter (no trailing zeros)."""
    if isinstance(v, float):
        return ('%g' % v)
    return str(v)


def hp_spelled(cfg) -> str:
    """Readable segment for the most-swept knobs, e.g. 'L3_h64_glr0.001_xlr0.01'.
    A knob is included only if the cfg actually has it AND (for the single 'lr')
    only when the model doesn't use the split gnn_lr/explainer_lr."""
    parts = []
    has_split_lr = getattr(cfg, 'gnn_lr', None) is not None
    for prefix, attr in _HP_SPELLED:
        if attr == 'lr' and has_split_lr:
            continue          # split LRs already captured by glr/xlr
        if attr in ('gnn_lr', 'explainer_lr') and not has_split_lr:
            continue
        v = getattr(cfg, attr, None)
        if v is None:
            continue
        parts.append(f'{prefix}{_fmt_num(v)}')
    return '_'.join(parts)


def hp_hash(cfg, length: int = 8) -> str:
    """Short deterministic hash of the fine hyperparameters present on cfg.
    Two configs differing in ANY hashed knob get different hashes."""
    items = {}
    for attr in _HP_HASHED:
        v = getattr(cfg, attr, None)
        if v is not None:
            items[attr] = _fmt_num(v)
    if not items:
        return 'hp-none'
    blob = _json.dumps(items, sort_keys=True)
    return 'hp-' + _hashlib.sha1(blob.encode()).hexdigest()[:length]


def hp_suffix(cfg) -> str:
    """Full hyperparameter path segment: '<spelled>_<hash>'."""
    spelled = hp_spelled(cfg)
    h = hp_hash(cfg)
    return f'{spelled}_{h}' if spelled else h


def write_hparams(out_dir, cfg) -> None:
    """Write hparams.json (full key→value of spelled+hashed knobs) into out_dir so
    the hp-hash is always decodable. Call once per run after out_dir is created."""
    from pathlib import Path as _P
    rec = {}
    for _, attr in _HP_SPELLED:
        if getattr(cfg, attr, None) is not None:
            rec[attr] = getattr(cfg, attr)
    for attr in _HP_HASHED:
        if getattr(cfg, attr, None) is not None:
            rec[attr] = getattr(cfg, attr)
    rec['hp_hash'] = hp_hash(cfg)
    p = _P(out_dir) / 'hparams.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        _json.dump(rec, f, indent=2, default=str)


# ── mutag TUDataset ──────────────────────────────────────────────────────────

class MutagTUDataset(torch.utils.data.Dataset):
    """Wraps mutag ``Data`` objects from ``datasets.mutag.Mutag``, attaching
    ChemIntuit ``nodes_to_motifs`` from the vocab lookup + index map.

    Source ground truth: each ``Data`` from ``Mutag.process()`` already carries
    ``node_label`` [N] and ``edge_label`` [E] (from ``Mutagenicity_edge_gt.txt``).
    ``clone()`` preserves them; do NOT use ``--use_gt`` / synthetic relabelling
    for mutag.

    Motif annotations: this wrapper sets ``nodes_to_motifs`` (plural) from the
    Phase-1 vocab lookup. That is independent of Mutag's optional built-in
    ``node_to_motifs`` (BRICS, only when ``Mutag(add_motifs=True)``).

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

        # Source ground-truth explanation labels. mutag ships GT in its source
        # (no synthetic relabelling). data.clone() preserves any node_label /
        # edge_label the external Mutag loader already attached; if it instead
        # uses common alias names (edge_mask / node_mask), normalise them to the
        # canonical node_label / edge_label the eval pipeline reads.
        _normalize_source_gt_labels(data)

        if self._vocab is None or not self._smiles[idx]:
            data.nodes_to_motifs = torch.full((n,), -1, dtype=torch.long)
            return data

        mapped_smi = self._smiles[idx]
        data.smiles = mapped_smi
        data.nodes_to_motifs = apply_motif_lookup_with_index_map(
            n, mapped_smi, self._lookup, self._index_maps
        )
        return data


class OGBMotifDataset(torch.utils.data.Dataset):
    """Wrap OGB graph indices, attaching ``nodes_to_motifs`` from vocab lookup.

    OGB SMILES (from ``mol.csv.gz``) use the same atom order as the PyG graph,
    so canonical-SMILES lookup applies without an index map.
    """

    def __init__(
        self,
        ogb_dataset,
        indices,
        vocab: Optional[VocabData] = None,
        split: str = 'training',
        *,
        label_mean: float = 0.0,
        label_std: float = 1.0,
        normalize_labels: bool = False,
        require_smiles: bool = False,
    ):
        self._ds = ogb_dataset
        self._indices = list(indices)
        self._lookup = vocab.lookup_for_split(split) if vocab else {}
        self._label_mean = float(label_mean)
        self._label_std = float(label_std) if float(label_std) != 0.0 else 1.0
        self._normalize_labels = normalize_labels
        self._require_smiles = require_smiles

        if require_smiles and vocab is not None:
            missing = [
                self._indices[i] for i in range(len(self._indices))
                if not getattr(self._ds[self._indices[i]], 'smiles', None)
            ]
            if missing:
                raise ValueError(
                    f"OGB dataset is missing SMILES on {len(missing)} graph(s) "
                    f"in split={split!r} (e.g. idx={missing[:3]}). Motif vocab "
                    "requires mol.csv.gz mapping. Re-download the OGB dataset or "
                    "run export_ogb_to_csv.py from a complete cache.")

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        from .graph_to_smiles import apply_motif_lookup_canonical
        i = self._indices[idx]
        data = self._ds[i].clone()
        n = data.num_nodes
        smiles = getattr(data, 'smiles', None)
        if self._require_smiles and self._lookup and not smiles:
            raise ValueError(
                f"OGB graph index {i} has no SMILES; cannot attach motif annotations.")
        if self._lookup and smiles:
            data.smiles = smiles
            data.nodes_to_motifs = apply_motif_lookup_canonical(
                n, smiles, self._lookup)
        else:
            data.nodes_to_motifs = torch.full((n,), -1, dtype=torch.long)
        if self._normalize_labels and data.y is not None:
            data.y = (data.y - self._label_mean) / self._label_std
        return data


def _normalize_source_gt_labels(data) -> None:
    """In-place: ensure ``node_label`` / ``edge_label`` (float) are present for
    source-GT datasets like mutag.

    ``datasets.mutag.Mutag`` already attaches canonical ``node_label`` and
    ``edge_label`` on each ``Data`` (from ``Mutagenicity_edge_gt.txt``). For
    mutag specifically (``Mutagenicity_label_readme.txt``: ``y=0`` mutagen,
    ``y=1`` nonmutagen):
      - ``edge_label`` [E]: 1 on NO2/NH2 motif edges (from edge_gt).
      - ``node_label`` [N]: 1 on nodes incident to those edges, but ONLY when
        ``y == 0`` (mutagen); non-mutagen graphs (``y == 1``) have both labels
        zeroed by the loader.
      - Mutagen graphs (``y == 0``) with no motif edges are dropped at process
        time.

    ``MutagTUDataset`` calls this after ``clone()`` as a fallback for loaders
    that use alias names (``edge_mask``, etc.). No-op when canonical labels exist.
    """
    if getattr(data, 'edge_label', None) is None:
        for _alias in ('edge_mask', 'edge_gt', 'edge_y', 'ground_truth_mask'):
            _v = getattr(data, _alias, None)
            if _v is not None:
                data.edge_label = _v.float().view(-1)
                break
    if getattr(data, 'node_label', None) is None:
        for _alias in ('node_mask', 'node_gt', 'node_y'):
            _v = getattr(data, _alias, None)
            if _v is not None:
                data.node_label = _v.float().view(-1)
                break


def _import_mutag_class(data_root: str):
    """Import ``datasets.mutag.Mutag`` from the repo or legacy HPC layout.

    Search order (first match wins):
      1. ``{repo_root}/``               — vendored ``datasets/mutag.py``
      2. ``{data_root.parent}/src/``    — legacy cluster layout
      3. ``{data_root}/``               — data dir on PYTHONPATH
    """
    import sys
    from pathlib import Path as _P
    _here = _P(__file__).resolve()
    _repo = _here.parents[2]   # SharedModules/data/loader.py → repo root
    for cand in (_repo, _P(data_root).parent / 'src', _P(data_root)):
        p = str(cand)
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from datasets.mutag import Mutag
        return Mutag
    except ImportError as e:
        raise ImportError(
            "Cannot import datasets.mutag.Mutag. Expected datasets/mutag.py "
            f"under the repo root ({_repo}) or on PYTHONPATH. Original: {e}"
        ) from e


def _get_mutag_loaders(
    data_root: str,
    vocab: Optional[VocabData],
    batch_size: int,
    num_workers: int,
    fold: int = 0,
    index_maps_path: Optional[str] = None,
    smiles_csv_path: Optional[str] = None,
    splits_path: Optional[str] = None,
    mutag_seed: int = 42,
):
    """Build loaders for the mutag TUDataset.

    Uses the 14-dim pre-baked node features from the PKL directly.
    Attaches ``nodes_to_motifs`` via the atom-map index_map if a vocab
    and ``index_maps_path`` are provided.

    Parameters
    ----------
    data_root : str
        Parent of the ``mutag/`` TUDataset folder (``…/data``) or the PyG
        dataset folder itself (``…/data/mutag``). See ``resolve_mutag_roots``.
    vocab : VocabData or None
    index_maps_path : str or None
        Path to ``mutag_0_index_maps.pkl`` produced by
        ``build_mutag_smiles_df()``.  Required to attach motif annotations.
    smiles_csv_path : str or None
        Path to the ``mutag_0.csv`` produced by ``export_mutag_dataset_to_csv.py``
        (columns: smiles, label, group, graph_id).  Used to recover the
        per-graph mapped SMILES and split assignments.
    splits_path : str or None
        Path to ``mutag_{fold}_splits.pkl`` (disjoint indices).  Preferred over
        the CSV ``group`` column when present.
    mutag_seed : int
        Base RNG seed; effective seed = ``mutag_seed + fold`` when splits are
        computed on the fly (fallback only).
    """
    Mutag = _import_mutag_class(data_root)
    from .dataset_routing import resolve_mutag_roots, mutag_artifact_paths
    tudataset_root, _artifact_dir = resolve_mutag_roots(data_root)
    dataset = Mutag(root=tudataset_root)
    _art = mutag_artifact_paths(data_root, fold)

    # Fail fast: a vocab was supplied (caller wants motif annotations) but the
    # index maps / mapped-SMILES CSV are absent. Without them every node would
    # silently become motif_id=-1 (a degenerate, all-unknown run that looks like
    # it succeeded). Require the export artifacts instead of degrading silently.
    if vocab is not None:
        _missing = []
        if not (index_maps_path and Path(index_maps_path).exists()):
            _missing.append(index_maps_path or _art['mutag_index_maps_path'])
        if not (smiles_csv_path and Path(smiles_csv_path).exists()):
            _missing.append(smiles_csv_path or _art['mutag_smiles_csv_path'])
        if _missing:
            raise FileNotFoundError(
                "mutag vocab was provided but its motif-annotation artifacts are "
                "missing:\n  " + "\n  ".join(_missing) +
                "\nRun MotifBreakdown/export_mutag_dataset_to_csv.py first to "
                "produce mutag_<fold>.csv + mutag_<fold>_index_maps.pkl, or pass "
                "vocab=None to run mutag without motif annotations.")

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

    # Resolve train/valid/test indices (splits pickle preferred)
    train_items: List[int] = []
    val_items:   List[int] = []
    test_items:  List[int] = []

    _splits_file = splits_path or _art['mutag_splits_path']
    if Path(_splits_file).exists():
        from .mutag_splits import load_mutag_splits
        split_idx = load_mutag_splits(_splits_file)
        train_items = list(split_idx['train'])
        val_items   = list(split_idx['valid'])
        test_items  = list(split_idx['test'])
    elif split_by_graph:
        for i in range(len(dataset)):
            grp = split_by_graph.get(i, 'training')
            if grp == 'valid':
                val_items.append(i)
            elif grp == 'test':
                test_items.append(i)
            else:
                train_items.append(i)
        if not val_items and not test_items:
            raise ValueError(
                f"mutag CSV {smiles_csv_path} has no valid/test groups. "
                f"Re-run export_mutag_dataset_to_csv.py (writes mutag_{fold}_splits.pkl).")
    else:
        from .mutag_splits import get_mutag_split_idx
        print(f"  [mutag] no splits file; computing on-the-fly "
              f"(seed={mutag_seed + fold}, 80/10/10 disjoint). "
              f"Run export_mutag_dataset_to_csv.py for reproducible splits.")
        split_idx = get_mutag_split_idx(dataset, seed=mutag_seed + fold)
        train_items = list(split_idx['train'])
        val_items   = list(split_idx['valid'])
        test_items  = list(split_idx['test'])

    def _build_ds(indices, split_name):
        data_list   = [dataset[i] for i in indices]
        smiles_list = [smiles_by_graph.get(i) for i in indices]
        return MutagTUDataset(
            data_list, vocab, index_maps, smiles_list, split=split_name)

    train_ds = _build_ds(train_items, 'training')
    val_ds   = _build_ds(val_items,   'valid')
    test_ds  = _build_ds(test_items,  'test')
    _deg = compute_deg_histogram(train_ds)

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
        fold=fold,
        node_encoder='onehot',   # 14-dim pre-baked features, identity passthrough
        deg=_deg,
    )
    return loaders, test_ds, meta


def _get_ogb_loaders(
    dataset: str,
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 0,
    vocab: Optional[VocabData] = None,
    normalize: bool = False,
):
    """Build loaders for an OGB molecular dataset.

    Node features are the raw OGB integer tensor [N, 9].
    The model is responsible for applying AtomEncoder or a Linear projection.
    Returns (loaders, test_dataset, meta) matching the same signature as get_loaders.
    """
    import numpy as np
    from .dataset import load_ogb_dataset, OGB_NODE_FEAT_DIM, OGB_EDGE_FEAT_DIM

    ogb_dataset, split_idx = load_ogb_dataset(data_root, dataset)
    task_type = TASK_TYPE.get(dataset, 'BinaryClass')
    num_classes = NUM_CLASSES.get(dataset, 1)

    label_mean, label_std = 0.0, 1.0
    if normalize and task_type == 'Regression':
        train_y = [
            float(ogb_dataset[i].y.view(-1)[0].item())
            for i in split_idx['train']
        ]
        label_mean = float(np.mean(train_y))
        label_std = float(np.std(train_y)) or 1.0

    _require_smiles = vocab is not None
    _norm = normalize and task_type == 'Regression'
    _ds_kw = dict(
        label_mean=label_mean, label_std=label_std,
        normalize_labels=_norm, require_smiles=_require_smiles,
    )

    train_ds = OGBMotifDataset(
        ogb_dataset, split_idx['train'], vocab, split='training', **_ds_kw)
    val_ds = OGBMotifDataset(
        ogb_dataset, split_idx['valid'], vocab, split='valid', **_ds_kw)
    test_ds = OGBMotifDataset(
        ogb_dataset, split_idx['test'], vocab, split='test', **_ds_kw)

    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers),
        'valid': DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers),
        'test':  DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers),
    }
    _deg = compute_deg_histogram(train_ds)
    meta = LoaderMeta(
        x_dim=OGB_NODE_FEAT_DIM,
        edge_attr_dim=OGB_EDGE_FEAT_DIM,
        num_classes=num_classes,
        task_type=task_type,
        dataset=dataset,
        fold=0,
        node_encoder='atom_encoder',
        deg=_deg,
        norm_mean=label_mean,
        norm_std=label_std,
    )
    return loaders, test_ds, meta


def get_loaders(
    dataset: str,
    data_root: str,
    fold: int = 0,
    vocab: Optional[VocabData] = None,
    processed_root: Optional[str] = None,
    batch_size: int = 128,
    num_workers: int = 0,
    normalize: bool = False,
    force_reprocess: bool = False,
    mutag_index_maps_path: Optional[str] = None,
    mutag_smiles_csv_path: Optional[str] = None,
    mutag_splits_path: Optional[str] = None,
    mutag_seed: int = 42,
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
    processed_root : str or None
        Root for PyG processed ``.pt`` cache files (CSV datasets only).
        When omitted, uses ``$PROCESSED_ROOT`` or ``{data_root}/../processed``.
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
    mutag_splits_path : str or None
        Path to ``mutag_{fold}_splits.pkl`` (mutag only).
    mutag_seed
        Fallback RNG seed when no splits pickle exists (mutag only).

    Returns
    -------
    loaders : dict[str, DataLoader]   keys: 'train', 'valid', 'test'
    test_dataset                       raw test dataset for evaluation
    meta : LoaderMeta                  x_dim, edge_attr_dim, task_type, ...
    """
    from .dataset_routing import default_processed_base

    if processed_root in (None, ''):
        processed_root = default_processed_base(data_root, None)

    # ── OGB datasets ──────────────────────────────────────────────────────
    if dataset in OGB_DATASET_NAMES:
        return _get_ogb_loaders(
            dataset, data_root, batch_size, num_workers, vocab=vocab,
            normalize=normalize)

    # ── mutag TUDataset (14-dim pre-baked features) ───────────────────────
    if dataset == 'mutag':
        from .dataset_routing import mutag_artifact_paths
        _map = mutag_artifact_paths(data_root, fold)
        _imp = mutag_index_maps_path or _map['mutag_index_maps_path']
        _scsv = mutag_smiles_csv_path or _map['mutag_smiles_csv_path']
        _splits = mutag_splits_path or _map['mutag_splits_path']
        return _get_mutag_loaders(
            data_root, vocab, batch_size, num_workers,
            fold=fold,
            index_maps_path=_imp,
            smiles_csv_path=_scsv,
            splits_path=_splits,
            mutag_seed=mutag_seed,
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
        norm_mean=(train_ds.mean if normalize else 0.0),
        norm_std=(train_ds.std if normalize else 1.0),
    )
    return loaders, test_ds, meta


def apply_gt_loaders(
    loaders: Dict[str, DataLoader],
    test_ds,
    *,
    gt_cache: str,
    dataset: str,
    fold: int,
    vocab_variant: str,
    batch_size: int,
    num_workers: int = 0,
    relabel: bool = True,
    verbose: bool = True,
) -> Tuple[Dict[str, DataLoader], object]:
    """Swap train/valid/test loaders for the GT-relabelled graphs cached by
    ``SharedModules/data/apply_gt.py`` (Phase 4).

    apply_gt.py writes ``{split}_with_gt.pt`` under::

        {gt_cache}/{dataset}/fold{fold}/{vocab_variant}/relabel1/

    where each cached Data object carries the rule-derived ``data.y`` plus
    ``data.node_label`` and ``data.edge_label``.  When GT training is requested
    ALL three splits are replaced, so the model trains on (and is evaluated
    against) the synthetic rule target rather than the original activity label.

    Fails fast (``FileNotFoundError``) if the cache is incomplete instead of
    silently mixing GT and original-label loaders — a partial swap would train
    on a different target than intended and corrupt the experiment.

    Parameters
    ----------
    loaders : dict[str, DataLoader]
        Existing loaders from :func:`get_loaders`; returned with GT-backed
        entries swapped in.
    test_ds
        Raw test dataset; replaced by the GT test list when present.
    gt_cache : str
        Root of the gt_cache directory written by phase 4.
    dataset, fold, vocab_variant : str / int / str
        Identify which cache subtree to load.
    batch_size, num_workers : int
        DataLoader settings for the replacement loaders.
    relabel : bool
        Load the ``relabel1`` (rule-relabelled ``y``) subtree when True,
        ``relabel0`` otherwise.
    verbose : bool
        Print per-split load lines and the replacement summary.

    Returns
    -------
    (loaders, test_ds) with GT-backed entries substituted.
    """
    gt_base = (Path(gt_cache) / dataset / f'fold{fold}' / vocab_variant
               / ('relabel1' if relabel else 'relabel0'))
    gt_loaded: Dict[str, list] = {}
    gt_missing: List[str] = []
    for split in ('train', 'valid', 'test'):
        gt_path = gt_base / f'{split}_with_gt.pt'
        if gt_path.exists():
            gt_loaded[split] = torch.load(gt_path, weights_only=False)
            if verbose:
                print(f'  GT {split}: {len(gt_loaded[split])} graphs '
                      f'← {gt_path.name}')
        else:
            gt_missing.append(str(gt_path))

    if gt_missing:
        raise FileNotFoundError(
            "use_gt=True but the ground-truth cache is incomplete. Missing:\n  "
            + "\n  ".join(gt_missing)
            + f"\nRun phase-4 relabelling (SharedModules/data/apply_gt.py) for "
              f"dataset={dataset} fold={fold} variant={vocab_variant} first, "
              f"or unset --use_gt.")

    for split, shuffle in (('train', True), ('valid', False), ('test', False)):
        if split in gt_loaded:
            loaders[split] = DataLoader(
                gt_loaded[split], batch_size=batch_size,
                shuffle=shuffle, num_workers=num_workers,
            )
    if 'test' in gt_loaded:
        test_ds = gt_loaded['test']

    if verbose and gt_loaded:
        print('  Training on GT-relabelled data '
              '(data.y = rule-based synthetic labels)')
        print(f'  GT loaders replaced: {sorted(gt_loaded.keys())} '
              f"(test loader now GT-backed: {'test' in gt_loaded})")

    return loaders, test_ds


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
