"""dataset_routing.py — loader kind, path resolution, and training guards.

``Mutagenicity`` (CSV + synthetic GT) and ``mutag`` (TUDataset + source GT) are
distinct dataset keys and must never be aliased. Use :func:`loader_kind` and
:func:`resolve_data_root` so launchers and trainers route consistently.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .dataset_schema import DATASET_COLUMN, TASK_TYPE
from .ground_truth import GT_SUPPORTED_DATASETS

# TUDataset Mutagenicity with source explanation GT (NOT the CSV benchmark).
MUTAG_TUDATASET = 'mutag'

# CSV Mutagenicity benchmark (FOLDS); synthetic GT via phase-4 only.
MUTAG_CSV_DATASET = 'Mutagenicity'

OGB_DATASET_NAMES = frozenset({
    'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp', 'ogbg-molclintox',
    'ogbg-moltox21', 'ogbg-molsider', 'ogbg-molesol', 'ogbg-molfreesolv',
    'ogbg-mollipo',
})

SOURCE_GT_DATASETS = frozenset({MUTAG_TUDATASET})

SINGLE_FOLD_DATASETS = OGB_DATASET_NAMES | SOURCE_GT_DATASETS

REGRESSION_DATASETS = frozenset(
    ds for ds, task in TASK_TYPE.items() if task == 'Regression'
)


def loader_kind(dataset: str) -> str:
    """Return ``csv`` | ``tudataset_mutag`` | ``ogb``."""
    if dataset == MUTAG_TUDATASET:
        return 'tudataset_mutag'
    if dataset in OGB_DATASET_NAMES:
        return 'ogb'
    return 'csv'


def resolve_mutag_roots(data_root: str) -> tuple[str, Path]:
    """Return ``(tudataset_root, artifact_dir)`` for the mutag TUDataset.

    Accepts either:

    * **Parent layout** — ``data_root=…/data`` with ``mutag/raw/`` underneath.
      PyG ``root=…/data/mutag``; CSV/index maps live in ``…/data/``.
    * **Dataset layout** — ``data_root=…/data/mutag`` (the PyG folder itself,
      containing ``raw/``). Artifacts live alongside under the same directory.

    This lets ``MUTAG_DATA_ROOT`` be either ``$PROJECT/data`` or ``$PROJECT/data/mutag``.
    """
    root = Path(data_root)
    if (root / 'raw').is_dir():
        return str(root), root
    if (root / 'mutag').is_dir():
        return str(root / 'mutag'), root
    if root.name == 'mutag':
        return str(root), root
    return str(root / 'mutag'), root


def resolve_data_root(
    dataset: str,
    data_root: str,
    *,
    mutag_data_root: Optional[str] = None,
    ogb_data_root: Optional[str] = None,
) -> str:
    """Pick the data root for *dataset* (FOLDS vs bundled mutag/ OGB cache)."""
    if dataset == MUTAG_TUDATASET:
        return str(mutag_data_root or data_root)
    if dataset in OGB_DATASET_NAMES:
        return str(ogb_data_root or data_root)
    return str(data_root)


def resolve_node_encoder_for_dataset(dataset: str, cli_encoder: str) -> str:
    """OGB requires ``atom_encoder``; CSV/mutag honor the CLI."""
    if dataset in OGB_DATASET_NAMES:
        return 'atom_encoder'
    return cli_encoder


def effective_fold(dataset: str, fold: int) -> int:
    """OGB and mutag export artifacts default to fold 0 only."""
    if dataset in SINGLE_FOLD_DATASETS:
        return 0
    return int(fold)


def is_single_fold_dataset(dataset: str) -> bool:
    """True when only fold 0 has distinct data (OGB / mutag)."""
    return dataset in SINGLE_FOLD_DATASETS


def collapse_redundant_folds(df):
    """Drop fold>0 rows for OGB/mutag (duplicate of fold 0)."""
    import pandas as pd

    if df is None or len(df) == 0 or 'dataset' not in df.columns:
        return df
    ds = df['dataset'].astype(str)
    fold = pd.to_numeric(df.get('fold'), errors='coerce')
    mask = ds.isin(SINGLE_FOLD_DATASETS) & (fold > 0)
    dropped = int(mask.sum())
    if dropped:
        print(f'  [dedup] dropped {dropped} redundant fold>0 row(s) for OGB/mutag')
    return df.loc[~mask].copy()


def validate_use_gt(dataset: str, use_gt: bool, gt_cache: Optional[str]) -> None:
    """Fail fast on inconsistent synthetic-GT flags."""
    if not use_gt:
        return
    if not gt_cache:
        raise ValueError(
            '--use_gt requires --gt_cache pointing to the phase-4 GT cache '
            '(e.g. RESULTS/gt_cache). Without it, loaders stay on original '
            'labels and GT-ROC will not run on synthetic labels.')
    if dataset in SOURCE_GT_DATASETS:
        raise ValueError(
            f"--use_gt is not valid for dataset={dataset!r}: mutag carries "
            "source explanation GT at load time. Train without --use_gt.")
    if dataset not in GT_SUPPORTED_DATASETS:
        raise ValueError(
            f"--use_gt is not supported for dataset={dataset!r}. "
            f"Synthetic GT is defined for: {sorted(GT_SUPPORTED_DATASETS)}.")


def assert_vocab_rule_mining_allowed(dataset: str) -> None:
    """Motif rule mining requires discrete class labels."""
    if dataset in REGRESSION_DATASETS:
        raise ValueError(
            f"Vocabulary/rule mining is not compatible with regression dataset "
            f"{dataset!r} (continuous labels). Remove it from --datasets or "
            "use a classification benchmark.")


def mutag_artifact_paths(
    data_root: str,
    fold: int,
    *,
    index_maps_path: Optional[str] = None,
    smiles_csv_path: Optional[str] = None,
    splits_path: Optional[str] = None,
) -> Dict[str, str]:
    """Resolved mutag export paths for summary.json / regenerate."""
    _, artifact_dir = resolve_mutag_roots(data_root)
    f = int(fold)
    return {
        'mutag_index_maps_path': str(index_maps_path or artifact_dir / f'mutag_{f}_index_maps.pkl'),
        'mutag_smiles_csv_path': str(smiles_csv_path or artifact_dir / f'mutag_{f}.csv'),
        'mutag_splits_path': str(splits_path or artifact_dir / f'mutag_{f}_splits.pkl'),
    }


def default_processed_base(data_root: str, processed_root: Optional[str] = None) -> str:
    """Base PyG cache directory; CLI passes this, trainers append ``/{vocab_variant}``."""
    if processed_root not in (None, ''):
        return str(processed_root)
    return str(Path(data_root).parent / 'processed')


def variant_processed_root(base: str, vocab_variant: str) -> str:
    """Per-vocab PyG cache root passed to ``get_loaders``."""
    return f'{str(base).rstrip("/")}/{str(vocab_variant).strip("/")}'


def base_from_stored_processed_root(
    stored: str,
    vocab_variant: Optional[str] = None,
) -> str:
    """Strip a trailing ``/{vocab_variant}`` when replaying summary.json on the CLI."""
    root = str(stored).rstrip('/')
    if vocab_variant:
        suffix = f'/{vocab_variant}'
        if root.endswith(suffix):
            return root[: -len(suffix)]
    return root


def training_summary_extras(cfg) -> Dict:
    """Hyperparameters + mutag paths to merge into summary.json."""
    ds = getattr(cfg, 'dataset', '')
    out = {
        'conv_normalize': getattr(cfg, 'conv_normalize', 'l2'),
        'hidden_dim': getattr(cfg, 'hidden_dim', None),
        'num_layers': getattr(cfg, 'num_layers', None),
        'gin_inner_bn': getattr(cfg, 'gin_inner_bn', True),
        'info_loss_level': getattr(cfg, 'info_loss_level', None),
        'learn_edge_att': getattr(cfg, 'learn_edge_att', None),
        'processed_root': getattr(cfg, 'processed_root', None),
        'data_root': getattr(cfg, 'data_root', None),
        'seed': getattr(cfg, 'seed', None),
        'node_encoder': getattr(cfg, 'node_encoder', None),
        'loader_kind': loader_kind(ds),
        'w_feat': getattr(cfg, 'w_feat', None),
        'w_message': getattr(cfg, 'w_message', None),
        'w_readout': getattr(cfg, 'w_readout', None),
        'use_gt': bool(getattr(cfg, 'use_gt', False)),
        'gt_cache': getattr(cfg, 'gt_cache', None),
        'run_multi_explanation': bool(getattr(cfg, 'run_multi_explanation', False)),
    }
    if hasattr(cfg, 'unk_mode'):
        out['unk_mode'] = getattr(cfg, 'unk_mode', None)
    if ds == MUTAG_TUDATASET:
        out.update(mutag_artifact_paths(
            getattr(cfg, 'data_root', ''),
            getattr(cfg, 'fold', 0),
            index_maps_path=getattr(cfg, 'mutag_index_maps_path', None),
            smiles_csv_path=getattr(cfg, 'mutag_smiles_csv_path', None),
            splits_path=getattr(cfg, 'mutag_splits_path', None),
        ))
        out['mutag_seed'] = getattr(cfg, 'mutag_seed', 42)
    return out
