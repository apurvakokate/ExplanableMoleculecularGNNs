from .vocab import VocabData, load_vocab, compute_mask_cache
from .dataset import MolDataset, build_graph, NUM_ATOM_TYPES, EDGE_FEAT_DIM, ATOMS, BONDS
from .loader import (
    get_loaders, compute_pos_weights, apply_gt_loaders, LoaderMeta,
    DATASET_COLUMN, TASK_TYPE,
    MutagTUDataset, MUTAG_X_DIM, MUTAG_EDGE_DIM,
    OGB_DATASET_NAMES,
)
from .graph_to_smiles import (
    graph_to_mapped_smiles,
    unmap_smiles,
    verify_ogb_index_alignment,
    verify_mutag_index_alignment,
    apply_motif_lookup_with_index_map,
    build_mutag_smiles_df,
    MUTAG_ATOM_TYPE_MAP,
)
# attach_ground_truth is DORMANT (commented out in ground_truth.py) — the live
# GT path is SharedModules/data/apply_gt.py. Only GT_SUPPORTED_DATASETS is
# still exported.
from .ground_truth import GT_SUPPORTED_DATASETS
from .dataset_routing import (
    MUTAG_TUDATASET,
    OGB_DATASET_NAMES,
    SINGLE_FOLD_DATASETS,
    loader_kind,
    resolve_data_root,
    resolve_node_encoder_for_dataset,
    effective_fold,
    is_single_fold_dataset,
    collapse_redundant_folds,
    validate_use_gt,
    assert_vocab_rule_mining_allowed,
    mutag_artifact_paths,
    training_summary_extras,
    resolve_mutag_roots,
    default_processed_base,
    variant_processed_root,
    base_from_stored_processed_root,
)

__all__ = [
    'VocabData', 'load_vocab', 'compute_mask_cache',
    'MolDataset', 'build_graph', 'NUM_ATOM_TYPES', 'EDGE_FEAT_DIM', 'ATOMS', 'BONDS',
    'get_loaders', 'compute_pos_weights', 'apply_gt_loaders', 'LoaderMeta',
    'DATASET_COLUMN', 'TASK_TYPE',
    'MutagTUDataset', 'MUTAG_X_DIM', 'MUTAG_EDGE_DIM', 'OGB_DATASET_NAMES',
    'graph_to_mapped_smiles', 'unmap_smiles',
    'verify_ogb_index_alignment', 'verify_mutag_index_alignment',
    'apply_motif_lookup_with_index_map', 'build_mutag_smiles_df',
    'MUTAG_ATOM_TYPE_MAP',
    'GT_SUPPORTED_DATASETS',
    'MUTAG_TUDATASET', 'SINGLE_FOLD_DATASETS',
    'loader_kind', 'resolve_data_root', 'resolve_node_encoder_for_dataset',
    'effective_fold', 'is_single_fold_dataset', 'collapse_redundant_folds',
    'validate_use_gt', 'assert_vocab_rule_mining_allowed',
    'mutag_artifact_paths', 'training_summary_extras', 'resolve_mutag_roots',
    'default_processed_base', 'variant_processed_root', 'base_from_stored_processed_root',
]
