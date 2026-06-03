"""ground_truth.py — integrate the motif-rule ground truth pipeline
with our SharedModules data loading.

The pipeline (from MotifSAT/motif_label_pipeline.py + ground_truth_pipeline.py)
selects a DNF rule from the motif co-occurrence rules produced by
generate_vocab_rules.py, then for every graph:
  - Sets ``data.edge_label`` float [E]  — 1.0 for edges touching rule-active
    motif nodes, 0.0 otherwise.
  - Optionally relabels ``data.y`` with the rule-derived binary ground truth.

``data.edge_label`` is what the explainer ROC evaluation uses:
    AUC(node_att[src] * node_att[dst], data.edge_label)

File layout produced by generate_vocab_rules.py
-----------------------------------------------
{vocab_root}/{dataset}/{variant}/
    matrix.npz                   ← motif presence matrix [n_graphs × M]
    matrix_columns.csv           ← columns: motif_id, motif_identity, ...
    smiles_labels.csv            ← rows:    smiles, label, group, ...

motif_label_pipeline.load_dataset_rulebook expects
---------------------------------------------------
{data_root}/{dataset}_fold{fold}/
    graph_motif_matrix.npz
    graph_motif_matrix_columns.csv   (column: motif_identity)
    graph_motif_matrix_rows.csv      (column: smiles)

This module bridges the gap by building the expected directory structure
(symlinks or copies) from the vocab output, then calling the pipeline.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Datasets for which ground truth rules are meaningful
# ─────────────────────────────────────────────────────────────────────────────

GT_SUPPORTED_DATASETS = {
    'Mutagenicity', 'Benzene', 'BBBP', 'hERG', 'Alkane_Carbonyl',
    'Fluoride_Carbonyl',
}


# ─────────────────────────────────────────────────────────────────────────────
# File bridge: translate vocab output names to what motif_label_pipeline wants
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_rulebook_dir(
    vocab_root: str,
    dataset: str,
    variant: str,
    fold: int,
    rulebook_root: str,
    force: bool = False,
) -> Path:
    """Create the directory structure that motif_label_pipeline.load_dataset_rulebook
    expects, from the files produced by generate_vocab_rules.py.

    Writes (or symlinks) three files:
        {rulebook_root}/{dataset}_fold{fold}/graph_motif_matrix.npz
        {rulebook_root}/{dataset}_fold{fold}/graph_motif_matrix_columns.csv
        {rulebook_root}/{dataset}_fold{fold}/graph_motif_matrix_rows.csv

    Parameters
    ----------
    vocab_root : str
        Root of the vocabulary output (``--out_dir`` passed to generate_vocab_rules).
    dataset : str
    variant : str
        Vocabulary variant subdirectory (e.g. ``'all_fallback_bpe'``).
    fold : int
    rulebook_root : str
        Where to write the bridged files.
    force : bool
        Overwrite existing files.
    """
    vdir = Path(vocab_root) / dataset / variant
    out_dir = Path(rulebook_root) / f'{dataset}_fold{fold}'
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_src   = vdir / 'matrix.npz'
    cols_src  = vdir / 'matrix_columns.csv'
    rows_src  = vdir / 'smiles_labels.csv'

    npz_dst   = out_dir / 'graph_motif_matrix.npz'
    cols_dst  = out_dir / 'graph_motif_matrix_columns.csv'
    rows_dst  = out_dir / 'graph_motif_matrix_rows.csv'

    # matrix.npz → graph_motif_matrix.npz
    if force or not npz_dst.exists():
        shutil.copy2(npz_src, npz_dst)

    # matrix_columns.csv → graph_motif_matrix_columns.csv
    # Already has 'motif_identity' column — copy as-is
    if force or not cols_dst.exists():
        shutil.copy2(cols_src, cols_dst)

    # smiles_labels.csv → graph_motif_matrix_rows.csv
    # Needs column 'smiles' — our file already has it
    if force or not rows_dst.exists():
        df = pd.read_csv(rows_src)
        if 'smiles' not in df.columns:
            raise KeyError(
                f"smiles_labels.csv at {rows_src} must have a 'smiles' column")
        df[['smiles']].to_csv(rows_dst, index=False)

    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# Core ground truth builder
# ─────────────────────────────────────────────────────────────────────────────

def attach_ground_truth(
    split_datasets: Dict[str, Any],
    dataset: str,
    fold: int,
    vocab_root: str,
    variant: str,
    rulebook_root: str,
    motif_list: Optional[List[str]] = None,
    rule_index: Optional[int] = None,
    relabel_graphs: bool = True,
    cache_root: Optional[str] = None,
    force_rebuild: bool = False,
    interactive: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Attach ``edge_label`` (and optionally relabel ``y``) to every Data object
    using the motif-rule ground truth pipeline.

    Parameters
    ----------
    split_datasets : dict
        ``{'train': dataset, 'valid': dataset, 'test': dataset}``
        where each dataset is iterable and yields PyG Data objects that
        already have ``data.smiles`` and ``data.nodes_to_motifs`` set.
    dataset : str
        Dataset name (e.g. ``'Mutagenicity'``).
    fold : int
    vocab_root : str
        Root of generate_vocab_rules.py output.
    variant : str
        Vocabulary variant (e.g. ``'all_fallback_bpe'``).
    rulebook_root : str
        Working directory for the motif_label_pipeline files.
    motif_list : list[str] or None
        SMARTS list (from VocabData) used to resolve motif names to ids.
    rule_index : int or None
        Pre-select a rule by index, skipping interactive prompt.
    relabel_graphs : bool
        Replace ``data.y`` with the rule-derived binary label if True.
    cache_root : str or None
        Where to cache the annotated Data lists.  If None, no caching.
    force_rebuild : bool
        Ignore existing cache.
    interactive : bool
        Show rule selection prompt (set False for non-TTY HPC runs).

    Returns
    -------
    split_datasets : dict
        Same structure, Data objects now have ``data.edge_label`` [E].
    debug : dict
        Per-split statistics (edge_positive_fraction, n_graphs_relabelled, …).
    """
    # Add MotifSAT src to path so we can import the pipeline
    _ensure_motifsat_importable()

    # Bridge the file names
    _prepare_rulebook_dir(
        vocab_root, dataset, variant, fold,
        rulebook_root, force=force_rebuild,
    )

    # Try cache first
    if cache_root and not force_rebuild:
        cached = _load_cache(cache_root, dataset, fold, variant, relabel_graphs)
        if cached is not None:
            return cached

    from motif_label_pipeline import (
        load_dataset_rulebook,
        choose_rule_interactive,
        evaluate_rule_on_motifs,
        save_rulebook_json,
    )

    rulebook = load_dataset_rulebook(
        data_root=rulebook_root,
        dataset_name=dataset,
        fold=fold,
    )

    selected_rule = choose_rule_interactive(
        rulebook,
        selected_index=rule_index,
        interactive=interactive,
    )

    # Build motif_name → set[motif_id] map
    motif_name_to_ids = _motif_name_to_ids(motif_list)

    # Smiles → row indices in rulebook
    smiles_to_row = {}
    for i, smi in enumerate(rulebook['row_smiles']):
        smiles_to_row.setdefault(str(smi), []).append(i)

    out: Dict[str, list] = {}
    debug: Dict[str, Dict] = {}

    for split_name, ds in split_datasets.items():
        data_list = [ds[i].clone() for i in range(len(ds))]
        agg = _empty_agg(split_name)

        for data in data_list:
            smi = str(getattr(data, 'smiles', ''))
            present: set[str] = set()
            row_idxs = smiles_to_row.get(smi, [])
            if row_idxs:
                for ridx in row_idxs:
                    present |= rulebook['row_motif_sets'][ridx]
            else:
                agg['n_missing_smiles'] += 1
                n2m = getattr(data, 'nodes_to_motifs', None)
                if n2m is not None:
                    present = {
                        _motif_name(int(v), motif_list)
                        for v in n2m.detach().cpu().tolist()
                        if int(v) >= 0
                    }

            rule_positive, active_names = evaluate_rule_on_motifs(present, selected_rule)
            active_ids = _resolve_active_ids(active_names, motif_name_to_ids)

            edge_label, n_pos = _build_edge_label(data, active_ids)
            data.edge_label = edge_label

            old_y = float(torch.as_tensor(data.y).view(-1)[0].item())
            gt_y = 1.0 if rule_positive else 0.0
            if relabel_graphs:
                data.y = torch.tensor([gt_y], dtype=torch.float32)
                if old_y != gt_y:
                    agg['n_relabelled'] += 1

            n_edges = int(edge_label.numel())
            agg['n_graphs'] += 1
            agg['n_total_edges'] += n_edges
            agg['n_pos_edges'] += n_pos
            if n_pos > 0:
                agg['n_graphs_with_pos_edges'] += 1
            if gt_y > 0.5:
                agg['n_graphs_rule_positive'] += 1

        agg['edge_positive_fraction'] = (
            agg['n_pos_edges'] / agg['n_total_edges']
            if agg['n_total_edges'] > 0 else 0.0
        )
        out[split_name] = data_list
        debug[split_name] = agg
        print(
            f"  [{dataset} fold={fold} {split_name}]  "
            f"n={agg['n_graphs']}  "
            f"rule_positive={agg['n_graphs_rule_positive']}  "
            f"edge_pos_frac={agg['edge_positive_fraction']:.4f}  "
            f"relabelled={agg['n_relabelled']}"
        )

    # Save rulebook JSON
    if cache_root:
        rj = Path(cache_root) / dataset / f'fold{fold}' / variant / 'selected_rule.json'
        rj.parent.mkdir(parents=True, exist_ok=True)
        save_rulebook_json(rulebook, selected_rule, rj)
        _save_cache(out, cache_root, dataset, fold, variant, relabel_graphs)

    return out, debug


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_motifsat_importable() -> None:
    """Add the MotifSAT src directory to sys.path if it is not already importable."""
    try:
        import motif_label_pipeline  # noqa: F401
        return
    except ImportError:
        pass
    # Try common relative locations
    candidates = [
        Path(__file__).resolve().parents[3] / 'MotifBreakdown',
    ]
    import os
    env_path = os.environ.get('MOTIFSAT_SRC')
    if env_path:
        candidates.insert(0, Path(env_path))
    for p in candidates:
        if (p / 'motif_label_pipeline.py').exists():
            sys.path.insert(0, str(p))
            return
    raise ImportError(
        "Cannot import motif_label_pipeline. "
        "Either place MotifBreakdown/ alongside SharedModules/ or set "
        "MOTIFSAT_SRC=/path/to/MotifBreakdown."
    )


def _motif_name(motif_id: int, motif_list: Optional[List[str]]) -> str:
    if motif_list and 0 <= motif_id < len(motif_list):
        return str(motif_list[motif_id])
    return f'motif_{motif_id}'


def _motif_name_to_ids(motif_list: Optional[List[str]]) -> Dict[str, set]:
    out: Dict[str, set] = {}
    if motif_list is None:
        return out
    for mid, name in enumerate(motif_list):
        out.setdefault(str(name), set()).add(mid)
    return out


def _resolve_active_ids(
    active_names: set[str],
    motif_name_to_ids: Dict[str, set],
) -> set[int]:
    ids: set[int] = set()
    for name in active_names:
        ids.update(motif_name_to_ids.get(str(name), set()))
    return ids


def _build_edge_label(data, active_ids: set[int]) -> Tuple[torch.Tensor, int]:
    """Return (edge_label [E], n_positive_edges)."""
    n_edges = data.edge_index.size(1)
    edge_label = torch.zeros(n_edges, dtype=torch.float32)
    n2m = getattr(data, 'nodes_to_motifs', None)
    if n2m is None or not active_ids:
        return edge_label, 0
    node_active = torch.tensor(
        [int(v) in active_ids for v in n2m.detach().cpu().tolist()],
        dtype=torch.bool,
    )
    src, dst = data.edge_index.detach().cpu().long()
    pos = node_active[src] | node_active[dst]
    edge_label[pos] = 1.0
    return edge_label, int(pos.sum().item())


def _empty_agg(split_name: str) -> Dict[str, Any]:
    return {
        'split': split_name,
        'n_graphs': 0,
        'n_graphs_rule_positive': 0,
        'n_graphs_with_pos_edges': 0,
        'n_total_edges': 0,
        'n_pos_edges': 0,
        'n_relabelled': 0,
        'n_missing_smiles': 0,
        'edge_positive_fraction': 0.0,
    }


def _cache_paths(cache_root: str, dataset: str, fold: int,
                 variant: str, relabel: bool) -> Dict[str, Path]:
    tag = 'relabel1' if relabel else 'relabel0'
    base = Path(cache_root) / dataset / f'fold{fold}' / variant / tag
    base.mkdir(parents=True, exist_ok=True)
    return {s: base / f'{s}_with_gt.pt' for s in ('train', 'valid', 'test')}


def _save_cache(out: Dict[str, list], cache_root: str, dataset: str,
                fold: int, variant: str, relabel: bool) -> None:
    paths = _cache_paths(cache_root, dataset, fold, variant, relabel)
    for split_name, data_list in out.items():
        torch.save(data_list, paths[split_name])


def _load_cache(cache_root: str, dataset: str, fold: int,
                variant: str, relabel: bool) -> Optional[Tuple]:
    paths = _cache_paths(cache_root, dataset, fold, variant, relabel)
    if not all(p.exists() for p in paths.values()):
        return None
    out = {s: torch.load(paths[s], weights_only=False)
           for s in ('train', 'valid', 'test')}
    debug = {s: {'loaded_from_cache': True} for s in ('train', 'valid', 'test')}
    return out, debug
