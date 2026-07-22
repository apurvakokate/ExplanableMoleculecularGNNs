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

from .dataset import MolDataset, NUM_ATOM_TYPES, EDGE_FEAT_DIM, reject_wildcard_smiles_in_csv
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
# The pre-baked PKL (or node_labels.txt one-hot) uses 14 types indexed 0–13
# per Mutagenicity_label_readme.txt / MUTAG_ATOM_TYPE_MAP in graph_to_smiles.py:
#   0:C  1:O  2:Cl  3:H  4:N  5:F  6:Br  7:S  8:P  9:I  10:Na  11:K  12:Li  13:Ca
# Index 3 is explicit H (separate graph nodes). Unknown type ints are dropped at export.
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
    # Per-fold motif threshold (CSV datasets with vocab). When set, MOSE/MotifSAT
    # should use this instead of vocab.kept_motif_ids from fold-0 mining.
    kept_motif_ids: Optional[List[int]] = None
    threshold_pct: Optional[float] = None
    # Fold-thresholded SMILES lookup from build_fold_annotation (CSV + mutag).
    motif_lookup: Optional[Dict] = None


# Fraction of all-UNK graphs above which thresholded training likely failed.
MAX_ALL_UNK_GRAPH_FRACTION = 0.9


def _fraction_all_unk_motif_graphs(graphs) -> Tuple[int, int]:
    """Return (n_all_unk, n_total) for graphs with every node motif_id=-1."""
    n_total = len(graphs)
    if n_total == 0:
        return 0, 0
    n_all_unk = 0
    for data in graphs:
        ntm = getattr(data, 'nodes_to_motifs', None)
        if ntm is None or ntm.numel() == 0 or int((ntm >= 0).sum()) == 0:
            n_all_unk += 1
    return n_all_unk, n_total


def guard_excessive_all_unk_motifs(
    graphs,
    *,
    apply_threshold: bool,
    label: str,
    max_fraction: float = MAX_ALL_UNK_GRAPH_FRACTION,
) -> None:
    """Fail fast when thresholding leaves almost every graph entirely UNK.

    Individual all-UNK graphs are valid (every fragment below cutoff). When
    > ``max_fraction`` of a split is all-UNK, the threshold is likely too harsh
    for this dataset/variant or there is a vocab/SMILES/index mismatch.
    """
    if not apply_threshold:
        return
    n_all_unk, n_total = _fraction_all_unk_motif_graphs(graphs)
    if n_total == 0:
        return
    frac = n_all_unk / n_total
    if frac > max_fraction:
        raise ValueError(
            f"{label}: {n_all_unk}/{n_total} ({frac:.1%}) graphs have all "
            f"motif_id=-1 under the thresholded vocab (>{max_fraction:.0%}). "
            f"This usually means CHOSEN_THRESHOLD is too aggressive for this "
            f"dataset/variant, or there is a vocab/SMILES/index mismatch — not "
            f"expected per-graph filtering. Lower the threshold or use an "
            f"unfiltered vocab."
        )
    if n_all_unk and frac >= 0.25:
        print(
            f"  [threshold] {label}: {n_all_unk}/{n_total} "
            f"({frac:.1%}) all-UNK graphs (within guard ≤{max_fraction:.0%})"
        )


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
    'init_r', 'final_r', 'logit_clamp', 'deterministic_att',  # MotifSAT scheduling
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
        'training', 'valid', or 'test' — legacy split slice when ``motif_lookup``
        is omitted (prefer passing ``motif_lookup`` from ``build_fold_annotation``).
    motif_lookup : dict or None
        Pre-threshold or fold-thresholded ``{smiles: {node_idx: (smarts, mid)}}``.
        When set, overrides ``vocab.lookup_for_split(split)``.
    apply_threshold : bool
        Whether ``motif_id=-1`` nodes are allowed (thresholded vocabs).
    """

    def __init__(
        self,
        data_list: List,
        vocab: Optional[VocabData] = None,
        index_maps: Optional[Dict] = None,
        smiles_list: Optional[List[str]] = None,
        split: str = 'training',
        *,
        motif_lookup: Optional[Dict] = None,
        apply_threshold: bool = False,
    ):
        self._data = data_list
        self._vocab = vocab
        self._index_maps = index_maps or {}
        self._smiles = smiles_list or [None] * len(data_list)
        self._split = split
        self._apply_threshold = apply_threshold

        if motif_lookup is not None:
            self._lookup = motif_lookup
        elif vocab is not None:
            self._lookup = vocab.lookup_for_split(split)
            self._apply_threshold = getattr(vocab, 'apply_threshold', False)
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

        mapped_smi = self._smiles[idx] if idx < len(self._smiles) else None
        # PyG DataLoader collation requires every graph in a batch to expose the
        # same attribute keys; always attach smiles (empty when unmapped).
        data.smiles = mapped_smi if mapped_smi else ''

        if self._lookup and mapped_smi:
            data.nodes_to_motifs = apply_motif_lookup_with_index_map(
                n, mapped_smi, self._lookup, self._index_maps,
                edge_index=getattr(data, 'edge_index', None),
            )
            from .graph_to_smiles import validate_nodes_to_motifs
            g2s = self._index_maps.get(mapped_smi, {})
            smi_lookup = self._lookup.get(mapped_smi, {})
            validate_nodes_to_motifs(
                data.nodes_to_motifs, smiles=mapped_smi,
                apply_threshold=self._apply_threshold,
                smi_lookup=smi_lookup,
                heavy_smiles_indices=set(g2s.values()) if g2s else None,
            )
        elif self._vocab is not None or self._apply_threshold:
            raise ValueError(
                f"mutag graph {idx}: vocab set but mapped SMILES is missing.")
        else:
            data.nodes_to_motifs = torch.full((n,), -1, dtype=torch.long)
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
        self._apply_threshold = getattr(vocab, 'apply_threshold', False) if vocab else False
        self._label_mean = float(label_mean)
        self._label_std = float(label_std) if float(label_std) != 0.0 else 1.0
        self._normalize_labels = normalize_labels
        self._require_smiles = require_smiles

        if require_smiles and vocab is not None:
            # SMILES are stored as ``ogb_dataset.smiles_list`` (indexed by graph
            # id); fall back to a per-graph ``.smiles`` attribute if present.
            _slist = getattr(self._ds, 'smiles_list', None)
            def _smi(gid):
                if _slist is not None:
                    return _slist[gid]
                return getattr(self._ds[gid], 'smiles', None)
            missing = [
                self._indices[i] for i in range(len(self._indices))
                if not _smi(self._indices[i])
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
        _slist = getattr(self._ds, 'smiles_list', None)
        smiles = _slist[i] if _slist is not None else getattr(data, 'smiles', None)
        if self._require_smiles and self._lookup and not smiles:
            raise ValueError(
                f"OGB graph index {i} has no SMILES; cannot attach motif annotations.")
        # Keep smiles on every graph so PyG batch collation keys are uniform.
        data.smiles = smiles if smiles else ''
        if self._lookup and smiles:
            data.nodes_to_motifs = apply_motif_lookup_canonical(
                n, smiles, self._lookup)
            from .graph_to_smiles import validate_nodes_to_motifs
            validate_nodes_to_motifs(
                data.nodes_to_motifs, smiles=str(smiles),
                apply_threshold=self._apply_threshold,
                smi_lookup=self._lookup.get(str(smiles), {}),
            )
        elif self._lookup and self._require_smiles:
            raise ValueError(
                f"OGB graph index {i} has no SMILES; cannot attach motif annotations.")
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
        reject_wildcard_smiles_in_csv(smiles_csv_path)
        import pandas as pd
        df_smi = pd.read_csv(smiles_csv_path)
        for _, row in df_smi.iterrows():
            gid = int(row['graph_id'])
            split_by_graph[gid] = str(row.get('group', 'training'))
            smi = row.get('smiles')
            if smi is None or (isinstance(smi, float) and pd.isna(smi)):
                continue
            smi = str(smi).strip()
            if not smi or smi.lower() == 'nan':
                continue
            smiles_by_graph[gid] = smi

    _splits_file = splits_path or _art['mutag_splits_path']
    _csv_file = smiles_csv_path or _art['mutag_smiles_csv_path']
    _maps_file = index_maps_path or _art['mutag_index_maps_path']
    if (Path(_csv_file).exists() and Path(_splits_file).exists()
            and Path(_maps_file).exists()):
        from .mutag_artifacts import validate_mutag_artifacts
        validate_mutag_artifacts(
            _csv_file, _splits_file, _maps_file, dataset_size=len(dataset))

    # Resolve train/valid/test indices (splits pickle preferred)
    train_items: List[int] = []
    val_items:   List[int] = []
    test_items:  List[int] = []

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

    motif_lookup = None
    kept_motif_ids = None
    threshold_pct = None
    apply_threshold = False

    if vocab is not None:
        from .fold_threshold import build_fold_annotation
        from .dataset_schema import DATASET_COLUMN
        from pathlib import Path as _Path

        label_col = DATASET_COLUMN['mutag']
        if not label_col:
            raise ValueError(
                "mutag has no DATASET_COLUMN entry — cannot apply fold threshold.")
        motif_lookup, kept_motif_ids, _thr_motifs, threshold_pct = build_fold_annotation(
            lookup_all=vocab.lookup_all,
            motif_list=vocab.motif_list,
            mol_fragment_smarts=vocab.mol_fragment_smarts,
            csv_path=str(_csv_file),
            label_col=label_col,
            dataset='mutag',
            variant=vocab.variant or '',
            vocab_dir=_Path(vocab.vocab_dir) if vocab.vocab_dir else _Path('.'),
            apply_threshold=vocab.apply_threshold,
            threshold_pct=vocab.threshold_pct,
        )
        apply_threshold = threshold_pct is not None
        if threshold_pct is not None:
            print(
                f'  [fold threshold] mutag fold={fold} pct={threshold_pct} '
                f'kept={len(kept_motif_ids)}/{vocab.num_motifs} motifs '
                f'(support from train+val in {_csv_file})')
        else:
            print(
                f'  [fold lookup] mutag fold={fold} no threshold filter '
                f'({vocab.num_motifs} motifs)')

    def _build_ds(indices, split_name):
        data_list   = [dataset[i] for i in indices]
        smiles_list = [smiles_by_graph.get(i) for i in indices]
        if vocab is not None:
            n_missing = sum(1 for s in smiles_list if not s)
            if n_missing:
                missing_ids = [i for i, s in zip(indices, smiles_list) if not s]
                raise ValueError(
                    f"mutag {split_name}: {n_missing}/{len(smiles_list)} graphs "
                    f"in splits lack mapped SMILES (graph_ids={missing_ids[:10]}…). "
                    f"Re-run export_mutag_dataset_to_csv.py or refresh artifacts.")
        return MutagTUDataset(
            data_list, vocab, index_maps, smiles_list, split=split_name,
            motif_lookup=motif_lookup, apply_threshold=apply_threshold,
        )

    train_ds = _build_ds(train_items, 'training')
    val_ds   = _build_ds(val_items,   'valid')
    test_ds  = _build_ds(test_items,  'test')
    _tag = f'mutag fold={fold}'
    guard_excessive_all_unk_motifs(
        train_ds, apply_threshold=apply_threshold, label=f'{_tag} train')
    guard_excessive_all_unk_motifs(
        val_ds, apply_threshold=apply_threshold, label=f'{_tag} valid')
    guard_excessive_all_unk_motifs(
        test_ds, apply_threshold=apply_threshold, label=f'{_tag} test')
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
        kept_motif_ids=kept_motif_ids,
        threshold_pct=threshold_pct,
        motif_lookup=motif_lookup,
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

    kept_motif_ids = None
    threshold_pct = None
    lookup = None
    proc_tag = ''

    if vocab is not None:
        from .fold_threshold import build_fold_annotation
        from pathlib import Path as _Path

        if vocab.lookup_all is None or vocab.mol_fragment_smarts is None:
            missing = []
            if vocab.lookup_all is None:
                missing.append('_lookup_all.pickle')
            if vocab.mol_fragment_smarts is None:
                missing.append('_mol_fragment_smarts.pickle')
            raise FileNotFoundError(
                f"Vocab {dataset}/{vocab.variant} missing required artifacts: "
                f"{', '.join(missing)}. Re-run phase 1 "
                f"(generate_vocab_rules.py). Legacy split-lookup fallback "
                f"is disabled."
            )

        lookup, kept_motif_ids, _thr_motifs, threshold_pct = build_fold_annotation(
            lookup_all=vocab.lookup_all,
            motif_list=vocab.motif_list,
            mol_fragment_smarts=vocab.mol_fragment_smarts,
            csv_path=csv,
            label_col=label_col,
            dataset=dataset,
            variant=vocab.variant or '',
            vocab_dir=_Path(vocab.vocab_dir) if vocab.vocab_dir else _Path('.'),
            apply_threshold=vocab.apply_threshold,
            threshold_pct=vocab.threshold_pct,
        )
        proc_tag = 'pfold_thr'
        _n_kept = len(kept_motif_ids) if kept_motif_ids is not None else vocab.num_motifs
        if threshold_pct is not None:
            print(f'  [fold threshold] fold={fold} pct={threshold_pct} '
                  f'kept={_n_kept}/{vocab.num_motifs} motifs '
                  f'(support from this fold train+val)')
        else:
            print(f'  [fold lookup] fold={fold} no threshold filter '
                  f'({vocab.num_motifs} motifs)')

    proc_base = f'{processed_root}/{dataset}_fold{fold}'
    if proc_tag:
        proc_base = f'{proc_base}/{proc_tag}'

    reject_wildcard_smiles_in_csv(csv)

    # Training split — compute normalisation stats from training data
    train_ds = MolDataset(
        root=f'{proc_base}/train',
        csv_file=csv,
        split='training',
        label_col=label_col,
        normalize=normalize,
        lookup=lookup,
        num_classes=num_classes if task_type == 'MultiLabel' else None,
        force_reprocess=force_reprocess,
    )

    val_ds = MolDataset(
        root=f'{proc_base}/valid',
        csv_file=csv,
        split='valid',
        label_col=label_col,
        normalize=normalize,
        mean=train_ds.mean if normalize else None,
        std=train_ds.std if normalize else None,
        lookup=lookup,
        num_classes=num_classes if task_type == 'MultiLabel' else None,
        force_reprocess=force_reprocess,
    )

    test_ds = MolDataset(
        root=f'{proc_base}/test',
        csv_file=csv,
        split='test',
        label_col=label_col,
        normalize=normalize,
        mean=train_ds.mean if normalize else None,
        std=train_ds.std if normalize else None,
        lookup=lookup,
        num_classes=num_classes if task_type == 'MultiLabel' else None,
        force_reprocess=force_reprocess,
    )

    _apply_thr = threshold_pct is not None
    _tag = f'{dataset} fold={fold}'
    guard_excessive_all_unk_motifs(
        train_ds, apply_threshold=_apply_thr, label=f'{_tag} train')
    guard_excessive_all_unk_motifs(
        val_ds, apply_threshold=_apply_thr, label=f'{_tag} valid')
    guard_excessive_all_unk_motifs(
        test_ds, apply_threshold=_apply_thr, label=f'{_tag} test')

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
        kept_motif_ids=kept_motif_ids,
        threshold_pct=threshold_pct,
        motif_lookup=lookup if vocab is not None else None,
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
    gt_vocab_variant: Optional[str] = None,
    gt_relabel_dir: Optional[str] = None,
    refresh_vocab: Optional[VocabData] = None,
    refresh_index_maps: Optional[Dict] = None,
    fold_motif_lookup: Optional[Dict] = None,
    apply_threshold: Optional[bool] = None,
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
    # gt_relabel_dir overrides the relabel subtree — used for difficulty tiers
    # (relabel_easy/relabel_medium/relabel_hard from apply_gt.py --tier). When
    # None, fall back to the single-rule relabel1/relabel0 subtree.
    _relabel_seg = gt_relabel_dir or ('relabel1' if relabel else 'relabel0')
    gt_base = (Path(gt_cache) / dataset / f'fold{fold}'
               / (gt_vocab_variant or vocab_variant)
               / _relabel_seg)
    gt_loaded: Dict[str, list] = {}
    gt_missing: List[str] = []
    _lookup_split = {'train': 'training', 'valid': 'valid', 'test': 'test'}
    _apply_thr = apply_threshold
    if _apply_thr is None and refresh_vocab is not None:
        _apply_thr = getattr(refresh_vocab, 'apply_threshold', False)
    if _apply_thr is None:
        _apply_thr = False
    _refresh_lookup = fold_motif_lookup
    for split in ('train', 'valid', 'test'):
        gt_path = gt_base / f'{split}_with_gt.pt'
        if gt_path.exists():
            gt_loaded[split] = torch.load(gt_path, weights_only=False)
            if refresh_vocab is not None and gt_vocab_variant and gt_vocab_variant != vocab_variant:
                from .graph_to_smiles import refresh_motif_annotations_on_graphs
                lookup = _refresh_lookup
                if lookup is None:
                    lookup = refresh_vocab.lookup_for_split(_lookup_split[split])
                refresh_motif_annotations_on_graphs(
                    gt_loaded[split],
                    lookup,
                    index_maps=refresh_index_maps,
                    apply_threshold=_apply_thr,
                    validate=True,
                )
            if verbose:
                print(f'  GT {split}: {len(gt_loaded[split])} graphs '
                      f'← {gt_path.name}')
        else:
            gt_missing.append(str(gt_path))

    if _apply_thr:
        for split in ('train', 'valid', 'test'):
            if split in gt_loaded:
                guard_excessive_all_unk_motifs(
                    gt_loaded[split],
                    apply_threshold=True,
                    label=f'{dataset} fold={fold} GT/{split}',
                )

    if gt_missing:
        raise FileNotFoundError(
            "use_gt=True but the ground-truth cache is incomplete. Missing:\n  "
            + "\n  ".join(gt_missing)
            + f"\nRun phase-4 relabelling (SharedModules/data/apply_gt.py) for "
              f"dataset={dataset} fold={fold} variant={gt_vocab_variant or vocab_variant} first, "
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
    # Single pass over the dataset: some dataset wrappers (e.g. MutagTUDataset)
    # do expensive per-item work in __getitem__ (clone + SMILES motif lookup),
    # so iterating twice doubled that cost. Grow the histogram as needed.
    deg = torch.zeros(1, dtype=torch.long)
    for data in dataset:
        if data.edge_index.numel() == 0:
            continue
        d = degree(data.edge_index[1], num_nodes=data.num_nodes).long()
        bc = torch.bincount(d)
        if bc.numel() > deg.numel():
            grown = torch.zeros(bc.numel(), dtype=torch.long)
            grown[:deg.numel()] = deg
            deg = grown
        deg[:bc.numel()] += bc
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
