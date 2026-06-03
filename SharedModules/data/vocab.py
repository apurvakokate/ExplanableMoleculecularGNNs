"""vocab.py — load the MotifBreakdown vocabulary from pickle files.

File produced by generate_vocab_rules.py
  {base}_graph_lookup.pickle
  {base}_test_graph_lookup.pickle
  {base}_motif_list.pickle
  {base}_motif_counts.pickle
  {base}_motif_length.pickle
  {base}_motif_class.pickle
  {base}_graph_motifidx.pickle
  {base}_test_graph_motifidx.pickle
  mask_cache_training.pickle   ← added by patched generate_vocab_rules
  mask_cache_valid.pickle
  mask_cache_test.pickle
  mask_cache_all.pickle

Usage
-----
    from SharedModules.data.vocab import load_vocab
    vocab = load_vocab('/path/to/variant/dir', 'Mutagenicity', 'all_fallback_bpe')
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch


@dataclass
class VocabData:
    """All vocabulary artefacts for one dataset × variant.

    Attributes
    ----------
    motif_list : list[str]
        SMARTS strings, index = motif_id.
    motif_counts : list[int]
        Per-motif molecule occurrence counts.
    motif_lengths : list[int]
        Per-motif heavy-atom count.
    motif_class : dict[int, dict[int, int]]
        {motif_id: {0: n_neg, 1: n_pos}}.
    lookup_train : dict[str, dict[int, tuple[str, int]]]
        {smiles: {node_idx: (smarts, motif_id)}} — training + valid split.
    lookup_test : dict[str, dict[int, tuple[str, int]]]
        {smiles: {node_idx: (smarts, motif_id)}} — test split.
    gmi_train : dict[str, set[int]]
        {smiles: set[motif_id]} — training + valid.
    gmi_test : dict[str, set[int]]
        {smiles: set[motif_id]} — test.
    mask_cache : dict[str, dict[int, dict[str, torch.BoolTensor]]]
        {split: {motif_id: {smiles: bool_mask [n_atoms]}}}.
        Empty dict if cache files are absent (use compute_mask_cache()).
    num_motifs : int
        len(motif_list).
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

    @property
    def num_motifs(self) -> int:
        return len(self.motif_list)

    def motif_id(self, smarts: str) -> Optional[int]:
        """Return motif_id for a SMARTS string, or None if unknown."""
        try:
            return self.motif_list.index(smarts)
        except ValueError:
            return None

    def lookup_for_split(self, split: str) -> Dict[str, Dict[int, Tuple[str, int]]]:
        """Return the node lookup dict for 'train'/'training'/'test'/'valid'."""
        if split in ('test',):
            return self.lookup_test
        if split in ('valid',):
            return self.lookup_valid
        return self.lookup_train


def _load_pickle(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_vocab(
    out_dir: str,
    dataset: str,
    variant: str,
    load_mask_cache: bool = True,
) -> VocabData:
    """Load vocabulary artefacts produced by generate_vocab_rules.py.

    Parameters
    ----------
    out_dir : str
        Root output directory (the ``--out_dir`` passed to generate_vocab_rules).
    dataset : str
        Dataset name (e.g. ``'Mutagenicity'``).
    variant : str
        Variant string (e.g. ``'all_fallback_bpe'``).  This is the subdirectory
        name under ``{out_dir}/{dataset}/``.
    load_mask_cache : bool
        Whether to load the bool mask cache.  Set False to save memory when
        you only need graph structure.

    Returns
    -------
    VocabData
    """
    vdir = Path(out_dir) / dataset / variant
    base = str(vdir / f'{dataset}_{variant}')

    def p(suffix: str) -> Path:
        return Path(f'{base}{suffix}')

    motif_list = list(_load_pickle(p('_motif_list.pickle')))
    motif_counts = list(_load_pickle(p('_motif_counts.pickle')))
    motif_lengths = list(_load_pickle(p('_motif_length.pickle')))
    motif_class = _load_pickle(p('_motif_class.pickle'))
    lookup_train = _load_pickle(p('_graph_lookup.pickle'))
    # lookup_valid was added in a later version of generate_vocab_rules.
    # Fall back to empty dict for backwards compatibility with older vocab dirs.
    _valid_path = Path(f'{base}_valid_graph_lookup.pickle')
    lookup_valid = _load_pickle(_valid_path) if _valid_path.exists() else {}
    lookup_test = _load_pickle(p('_test_graph_lookup.pickle'))
    gmi_train = _load_pickle(p('_graph_motifidx.pickle'))
    gmi_test = _load_pickle(p('_test_graph_motifidx.pickle'))

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
    )


def compute_mask_cache(
    smiles_list: List[str],
    groups_all: List[str],
    lookup_all: Dict[str, Dict[int, Tuple[str, int]]],
) -> Dict[str, Dict[int, Dict[str, torch.BoolTensor]]]:
    """Compute the bool mask cache on-the-fly (when pickle files are absent).

    Mirrors the logic in MotifBreakdown/generate_vocab_rules.py:build_mask_cache.
    """
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
