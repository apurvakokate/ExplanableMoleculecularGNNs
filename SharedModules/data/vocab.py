"""vocab.py — load the MotifBreakdown vocabulary from pickle files.

Artifacts from generate_vocab_rules.py (fold-0 mining)
  {base}_lookup_all.pickle          SMILES → atom map, pre-threshold (annotation)
  {base}_mol_fragment_smarts.pickle per-SMILES fragment list (threshold support)
  {base}_graph_lookup.pickle        fold-0 train slice, thresholded (legacy/mining)
  {base}_valid_graph_lookup.pickle  fold-0 valid slice
  {base}_test_graph_lookup.pickle   fold-0 test slice
  {base}_motif_list.pickle
  {base}_kept_motif_ids.pickle      fold-0 kept ids (reference; trainers use per-fold)
  vocab_meta.json                   apply_threshold, threshold_pct, mining_fold

Usage
-----
    from SharedModules.data.vocab import load_vocab
    vocab = load_vocab('/path/to/vocab_root', 'Benzene', 'all_fallback_bpe_filter')
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch


@dataclass
class VocabData:
    """All vocabulary artefacts for one dataset × variant.

    lookup_all : pre-threshold SMILES → atom maps (use for GT rule presence).
    lookup_train/valid/test : fold-0 split slices after threshold (mining stats).
    Per-fold training annotations are built in get_loaders via fold_threshold.py.
    """

    motif_list: List[str]
    motif_counts: List[int]
    motif_lengths: List[int]
    motif_class: Dict[int, Dict[int, int]]
    lookup_train: Dict[str, Dict[int, Tuple[str, int]]]
    lookup_valid: Dict[str, Dict[int, Tuple[str, int]]]
    lookup_test: Dict[str, Dict[int, Tuple[str, int]]]
    gmi_train: Dict[str, Set[int]]
    gmi_test: Dict[str, Set[int]]
    mask_cache: Dict[str, Dict[int, Dict[str, torch.BoolTensor]]] = field(
        default_factory=dict
    )
    kept_motif_ids: Optional[List[int]] = None
    # Pre-threshold full-pool annotation (all SMILES in fold-0 CSV).
    lookup_all: Optional[Dict[str, Dict[int, Tuple[str, int]]]] = None
    # {smiles: [smarts, ...]} — one entry per fragment instance for support counts.
    mol_fragment_smarts: Optional[Dict[str, List[str]]] = None
    apply_threshold: bool = False
    threshold_pct: Optional[float] = None
    mining_fold: int = 0
    variant: str = ''
    dataset: str = ''
    vocab_dir: str = ''

    @property
    def num_motifs(self) -> int:
        return len(self.motif_list)

    def motif_id(self, smarts: str) -> Optional[int]:
        try:
            return self.motif_list.index(smarts)
        except ValueError:
            return None

    def lookup_for_split(self, split: str) -> Dict[str, Dict[int, Tuple[str, int]]]:
        """Fold-0 split slice (thresholded at mining time). Mining/eval legacy only."""
        if split in ('test',):
            return self.lookup_test
        if split in ('valid',):
            return self.lookup_valid
        return self.lookup_train

    @property
    def annotation_lookup(self) -> Dict[str, Dict[int, Tuple[str, int]]]:
        """Pre-threshold SMILES map for rule / fragmentation checks."""
        if self.lookup_all is None:
            raise FileNotFoundError(
                f"Vocab {self.dataset}/{self.variant}: missing _lookup_all.pickle. "
                f"Re-run phase 1 (generate_vocab_rules.py). Merged split-lookup "
                f"fallback is disabled — per-fold thresholding requires the full pool."
            )
        return self.lookup_all


def _load_pickle(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_vocab(
    out_dir: str,
    dataset: str,
    variant: str,
    load_mask_cache: bool = True,
) -> VocabData:
    vdir = Path(out_dir) / dataset / variant
    base = str(vdir / f'{dataset}_{variant}')

    def p(suffix: str) -> Path:
        return Path(f'{base}{suffix}')

    motif_list = list(_load_pickle(p('_motif_list.pickle')))
    motif_counts = list(_load_pickle(p('_motif_counts.pickle')))
    motif_lengths = list(_load_pickle(p('_motif_length.pickle')))
    motif_class = _load_pickle(p('_motif_class.pickle'))
    lookup_train = _load_pickle(p('_graph_lookup.pickle'))
    _valid_path = Path(f'{base}_valid_graph_lookup.pickle')
    lookup_valid = _load_pickle(_valid_path) if _valid_path.exists() else {}
    lookup_test = _load_pickle(p('_test_graph_lookup.pickle'))
    gmi_train = _load_pickle(p('_graph_motifidx.pickle'))
    gmi_test = _load_pickle(p('_test_graph_motifidx.pickle'))

    _kept_path = Path(f'{base}_kept_motif_ids.pickle')
    kept_motif_ids = (list(_load_pickle(_kept_path))
                      if _kept_path.exists() else None)

    _all_path = Path(f'{base}_lookup_all.pickle')
    if not _all_path.exists():
        raise FileNotFoundError(
            f"Missing required vocab artifact: {_all_path}\n"
            f"Re-run phase 1 for {dataset}/{variant} "
            f"(generate_vocab_rules.py writes _lookup_all.pickle)."
        )
    lookup_all = _load_pickle(_all_path)

    _frags_path = Path(f'{base}_mol_fragment_smarts.pickle')
    if not _frags_path.exists():
        raise FileNotFoundError(
            f"Missing required vocab artifact: {_frags_path}\n"
            f"Re-run phase 1 for {dataset}/{variant} "
            f"(generate_vocab_rules.py writes _mol_fragment_smarts.pickle)."
        )
    mol_fragment_smarts = _load_pickle(_frags_path)

    apply_threshold = False
    threshold_pct = None
    mining_fold = 0
    meta_path = vdir / 'vocab_meta.json'
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        apply_threshold = bool(meta.get('apply_threshold', False))
        if meta.get('threshold_pct') is not None:
            threshold_pct = float(meta['threshold_pct'])
        mining_fold = int(meta.get('mining_fold', 0))

    mask_cache: Dict[str, Dict[int, Dict[str, torch.BoolTensor]]] = {}
    if load_mask_cache:
        for split in ('training', 'valid', 'test', 'all'):
            cache_path = vdir / f'mask_cache_{split}.pickle'
            if cache_path.exists():
                mask_cache[split] = _load_pickle(cache_path)

    return VocabData(
        motif_list=motif_list,
        motif_counts=motif_counts,
        motif_lengths=motif_lengths,
        motif_class=motif_class,
        lookup_train=lookup_train,
        lookup_valid=lookup_valid,
        lookup_test=lookup_test,
        gmi_train=gmi_train,
        gmi_test=gmi_test,
        mask_cache=mask_cache,
        kept_motif_ids=kept_motif_ids,
        lookup_all=lookup_all,
        mol_fragment_smarts=mol_fragment_smarts,
        apply_threshold=apply_threshold,
        threshold_pct=threshold_pct,
        mining_fold=mining_fold,
        variant=variant,
        dataset=dataset,
        vocab_dir=str(vdir),
    )


def compute_mask_cache(
    smiles_list: List[str],
    groups_all: List[str],
    lookup_all: Dict[str, Dict[int, Tuple[str, int]]],
) -> Dict[str, Dict[int, Dict[str, torch.BoolTensor]]]:
    """Compute bool mask cache (mirrors generate_vocab_rules.build_mask_cache)."""
    splits = {'training', 'valid', 'test'}
    cache: Dict[str, Dict[int, Dict[str, torch.BoolTensor]]] = {
        'training': {}, 'valid': {}, 'test': {}, 'all': {}
    }
    for smi, grp in zip(smiles_list, groups_all):
        if grp not in splits:
            continue
        node_map = lookup_all.get(smi, {})
        if not node_map:
            continue
        n = max(node_map.keys()) + 1
        motif_atoms: Dict[int, List[int]] = {}
        for atom_idx, (_, mid) in node_map.items():
            if mid >= 0:
                motif_atoms.setdefault(mid, []).append(atom_idx)
        for mid, idxs in motif_atoms.items():
            mask = torch.zeros(n, dtype=torch.bool)
            mask[torch.tensor(idxs, dtype=torch.long)] = True
            for key in (grp, 'all'):
                cache[key].setdefault(mid, {})[smi] = mask
    return cache
