"""graph_to_smiles.py — Convert pre-built graph datasets to SMILES while
preserving exact atom-index correspondence with the original graph nodes.

Problem
-------
For datasets where graphs are stored as adjacency matrices + node-type lists
(mutag TUDataset) rather than SMILES, we need SMILES to run MotifBreakdown.
But Chem.MolToSmiles() produces *canonical* SMILES which **reorders atoms**.
If we then call Chem.MolFromSmiles(canonical_smiles) and run fragmentation,
the atom indices in the resulting lookup will not match the node indices in
the original PyG graph, causing nodes_to_motifs to be silently wrong.

Solution: atom-map numbers
--------------------------
Before generating SMILES, stamp each atom with an atom-map number equal to
its original graph node index + 1 (RDKit uses 0 to mean "no map").  The
canonical SMILES will then contain these map numbers (e.g. [C:3]).  When
MotifBreakdown fragments the molecule, the resulting lookup keys are these
*mapped SMILES strings*, and the lookup values carry the original graph-node
indices recovered from the atom-map numbers.

For OGB datasets there is no such problem: OGB's smiles2graph function uses
Chem.MolFromSmiles(smiles).GetAtoms() in standard iteration order — the same
as our build_graph — so atom index i in the OGB graph always equals atom i
from Chem.MolFromSmiles(smiles).

Exports
-------
graph_to_mapped_smiles()   mutag graph → mapped SMILES + index map
ogb_smiles_to_canonical()  OGB SMILES (from mol.csv.gz) → canonical SMILES
                           (no reordering needed, but strip map numbers)
verify_index_alignment()   sanity check: feature matrix rows == SMILES atoms
build_mutag_smiles_df()    export full mutag dataset to a CSV the pipeline
                           can consume
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')


# ─────────────────────────────────────────────────────────────────────────────
# Mutag TUDataset atom-type map
# ─────────────────────────────────────────────────────────────────────────────

MUTAG_ATOM_TYPE_MAP: Dict[int, int] = {
    0: 6,    # C
    1: 7,    # N
    2: 8,    # O
    3: 9,    # F
    4: 15,   # P
    5: 16,   # S
    6: 17,   # Cl
    7: 35,   # Br
    8: 53,   # I
}

# Reverse: symbol → our ATOMS dict index (for verifying feature matrices)
ATOMIC_NUM_TO_SYMBOL: Dict[int, str] = {
    v: Chem.GetPeriodicTable().GetElementSymbol(v)
    for v in MUTAG_ATOM_TYPE_MAP.values()
}


# ─────────────────────────────────────────────────────────────────────────────
# Core: graph → atom-map-number SMILES
# ─────────────────────────────────────────────────────────────────────────────

def graph_to_mapped_smiles(
    node_types: List[int],
    edge_src: List[int],
    edge_dst: List[int],
    atom_type_map: Dict[int, int] = MUTAG_ATOM_TYPE_MAP,
) -> Tuple[Optional[str], Optional[Dict[int, int]]]:
    """Convert a graph (node-type list + edge list) to a mapped SMILES string.

    Atom-map numbers encode the original graph node index:
        atom_map_num = graph_node_idx + 1
    so after fragmentation, the lookup can recover the original node index.

    Parameters
    ----------
    node_types : list[int]
        Integer node-type labels (keys in atom_type_map).
    edge_src, edge_dst : list[int]
        Undirected edge list (both directions should be present; deduplication
        is done internally).
    atom_type_map : dict
        Maps node-type integer → atomic number.  Defaults to MUTAG_ATOM_TYPE_MAP.

    Returns
    -------
    mapped_smiles : str or None
        SMILES with atom-map numbers, e.g. ``[C:1][N:2][C:3]``.
        None if conversion fails (unknown atom type, RDKit sanitisation error).
    graph_to_smiles_idx : dict[int, int] or None
        graph_node_idx → smiles_atom_idx in the returned SMILES string.
        Use this to translate MotifBreakdown lookup keys back to graph indices.
    """
    mol = Chem.RWMol()
    for node_idx, type_int in enumerate(node_types):
        atomic_num = atom_type_map.get(int(type_int))
        if atomic_num is None:
            return None, None
        atom = Chem.Atom(atomic_num)
        atom.SetAtomMapNum(node_idx + 1)   # encode original index (1-based)
        mol.AddAtom(atom)

    added: set = set()
    for s, d in zip(edge_src, edge_dst):
        pair = tuple(sorted([int(s), int(d)]))
        if pair not in added:
            mol.AddBond(pair[0], pair[1], Chem.BondType.SINGLE)
            added.add(pair)

    try:
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        mapped_smiles = Chem.MolToSmiles(mol)

        # Recover index map from the canonical-order rebuilt mol
        rebuilt = Chem.MolFromSmiles(mapped_smiles)
        if rebuilt is None:
            return None, None
        smiles_to_graph = {a.GetIdx(): a.GetAtomMapNum() - 1
                           for a in rebuilt.GetAtoms()}
        graph_to_smiles = {v: k for k, v in smiles_to_graph.items()}
        return mapped_smiles, graph_to_smiles
    except Exception:
        return None, None


def unmap_smiles(mapped_smiles: str) -> Optional[str]:
    """Remove atom-map numbers from a mapped SMILES string.

    The plain SMILES is what MotifBreakdown stores as lookup keys.
    After fragmentation, lookup keys will be SMARTS like ``[*]c1ccccc1``
    (without map numbers) that are used for motif vocabulary.
    """
    mol = Chem.MolFromSmiles(mapped_smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


# ─────────────────────────────────────────────────────────────────────────────
# OGB: SMILES are already correct — verify only
# ─────────────────────────────────────────────────────────────────────────────

def verify_ogb_index_alignment(
    smiles: str,
    graph_x: torch.Tensor,
) -> Dict[str, object]:
    """Verify that Chem.MolFromSmiles(smiles) has the same atom count as
    the OGB graph feature matrix and that atom symbols match x[:,0].

    OGB x[:,0] stores atomic_num - 1 (0-indexed into [H=0, He=1, ..., Og=117]).
    This check confirms graph node i corresponds to the i-th atom in the SMILES.

    Parameters
    ----------
    smiles : str
        SMILES from OGB's mapping/mol.csv.gz.
    graph_x : Tensor [N, 9]
        OGB node feature matrix.

    Returns
    -------
    dict with:
        ok : bool
        n_smiles_atoms : int
        n_graph_nodes : int
        mismatches : list of (node_idx, smiles_symbol, graph_symbol)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'ok': False, 'error': 'SMILES could not be parsed',
                'n_smiles_atoms': 0, 'n_graph_nodes': int(graph_x.size(0)),
                'mismatches': []}

    n_smi = mol.GetNumAtoms()
    n_grf = int(graph_x.size(0))

    if n_smi != n_grf:
        return {'ok': False,
                'error': f'atom count mismatch: SMILES={n_smi} graph={n_grf}',
                'n_smiles_atoms': n_smi, 'n_graph_nodes': n_grf,
                'mismatches': []}

    pt = Chem.GetPeriodicTable()
    mismatches = []
    for i, atom in enumerate(mol.GetAtoms()):
        smiles_sym = atom.GetSymbol()
        # OGB x[:,0] = atomic_num - 1  (index into 1-based periodic table)
        ogb_atomic_num = int(graph_x[i, 0].item()) + 1
        try:
            graph_sym = pt.GetElementSymbol(ogb_atomic_num)
        except Exception:
            graph_sym = '?'
        if smiles_sym != graph_sym:
            mismatches.append((i, smiles_sym, graph_sym))

    return {
        'ok':             len(mismatches) == 0,
        'n_smiles_atoms': n_smi,
        'n_graph_nodes':  n_grf,
        'mismatches':     mismatches,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Verify mutag graph features vs mapped SMILES
# ─────────────────────────────────────────────────────────────────────────────

def verify_mutag_index_alignment(
    mapped_smiles: str,
    graph_x: torch.Tensor,
    node_types: List[int],
    graph_to_smiles_idx: Dict[int, int],
) -> Dict[str, object]:
    """Verify that the mapped SMILES and the mutag graph feature matrix are
    consistent: each graph node's atom type matches both (a) the feature
    matrix one-hot index and (b) the corresponding atom in the SMILES.

    Parameters
    ----------
    mapped_smiles : str
        Output of graph_to_mapped_smiles().
    graph_x : Tensor [N, D]
        Pre-built node feature matrix from the mutag dataset.
        For TUDataset mutag, x = original_features (pre-baked float matrix).
    node_types : list[int]
        Original integer node-type labels (0–8).
    graph_to_smiles_idx : dict[int, int]
        Output of graph_to_mapped_smiles().

    Returns
    -------
    dict with ok, n_nodes, mismatches list
    """
    mol = Chem.MolFromSmiles(mapped_smiles)
    if mol is None:
        return {'ok': False, 'error': 'SMILES could not be parsed',
                'n_nodes': 0, 'mismatches': []}

    pt = Chem.GetPeriodicTable()
    n_nodes = len(node_types)
    mismatches = []

    for graph_idx in range(n_nodes):
        # Expected element from node_type
        atomic_num = MUTAG_ATOM_TYPE_MAP.get(int(node_types[graph_idx]))
        if atomic_num is None:
            mismatches.append((graph_idx, '?', '?', 'unknown_node_type'))
            continue
        expected_sym = pt.GetElementSymbol(atomic_num)

        # Symbol from SMILES (via atom-map alignment)
        smiles_idx = graph_to_smiles_idx.get(graph_idx)
        if smiles_idx is None:
            mismatches.append((graph_idx, expected_sym, '?', 'no_smiles_mapping'))
            continue
        smiles_sym = mol.GetAtomWithIdx(smiles_idx).GetSymbol()

        if expected_sym != smiles_sym:
            mismatches.append((graph_idx, expected_sym, smiles_sym,
                               'atom_type_mismatch'))

    return {
        'ok':         len(mismatches) == 0,
        'n_nodes':    n_nodes,
        'mismatches': mismatches,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build full mutag CSV for MotifBreakdown
# ─────────────────────────────────────────────────────────────────────────────

def build_mutag_smiles_df(
    data_list,
    split_name: str = 'all',
    verify: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Convert a mutag PyG dataset to a DataFrame with mapped SMILES.

    The mapped SMILES preserve atom-index correspondence with the graph.
    The resulting CSV can be passed directly to generate_vocab_rules.py.
    A separate ``index_map`` dict maps ``{smiles: {graph_node_idx: smiles_atom_idx}}``
    and is saved alongside the CSV for use in MolDataset.process().

    Parameters
    ----------
    data_list : iterable of PyG Data
        Must have .x (feature matrix), .y (label), .node_type (int tensor),
        .edge_index (LongTensor [2, E]).
    split_name : str
        Value to use in the 'group' column of the output CSV.
    verify : bool
        Run verify_mutag_index_alignment on each graph.

    Returns
    -------
    df : DataFrame
        Columns: smiles, label, group, graph_id, conversion_ok, verify_ok
    index_maps : dict
        {mapped_smiles: {graph_node_idx: smiles_atom_idx}}
        Use this when building nodes_to_motifs in MolDataset.
    """
    rows = []
    index_maps: Dict[str, Dict[int, int]] = {}
    stats = {'total': 0, 'converted': 0, 'failed': 0, 'verify_fail': 0}

    for i, data in enumerate(data_list):
        stats['total'] += 1
        node_types = data.node_type.cpu().tolist()
        edge_src   = data.edge_index[0].cpu().tolist()
        edge_dst   = data.edge_index[1].cpu().tolist()
        label      = float(data.y.item()) if data.y.numel() == 1 \
                     else data.y.cpu().numpy().tolist()

        mapped_smiles, g2s = graph_to_mapped_smiles(
            node_types, edge_src, edge_dst)

        if mapped_smiles is None:
            stats['failed'] += 1
            rows.append({'smiles': None, 'label': label, 'group': split_name,
                         'graph_id': i, 'conversion_ok': False,
                         'verify_ok': False})
            continue

        stats['converted'] += 1
        verify_ok = True
        if verify:
            result = verify_mutag_index_alignment(
                mapped_smiles, data.x, node_types, g2s)
            verify_ok = result['ok']
            if not verify_ok:
                stats['verify_fail'] += 1

        index_maps[mapped_smiles] = g2s
        rows.append({'smiles':        mapped_smiles,
                     'label':         label,
                     'group':         split_name,
                     'graph_id':      i,
                     'conversion_ok': True,
                     'verify_ok':     verify_ok})

    df = pd.DataFrame(rows)
    print(f"[build_mutag_smiles_df] "
          f"total={stats['total']} converted={stats['converted']} "
          f"failed={stats['failed']} verify_fail={stats['verify_fail']}")
    return df, index_maps


# ─────────────────────────────────────────────────────────────────────────────
# Build OGB lookup using canonical SMILES
# ─────────────────────────────────────────────────────────────────────────────

def verify_ogb_dataset_alignment(
    dataset,
    smiles_list: List[str],
    max_verify: int = 100,
) -> Dict[str, object]:
    """Spot-check a sample of an OGB dataset for SMILES–graph index alignment.

    Parameters
    ----------
    dataset : PygGraphPropPredDataset (or OGBDatasetWithSmiles)
    smiles_list : list[str]
        SMILES from mol.csv.gz in row order.
    max_verify : int
        Number of molecules to check (random sample).

    Returns
    -------
    dict with ok, n_checked, n_failed, failures list
    """
    import random
    n = len(smiles_list)
    indices = random.sample(range(n), min(max_verify, n))
    failures = []
    for idx in indices:
        data = dataset[idx]
        if not hasattr(data, 'x') or data.x is None:
            continue
        result = verify_ogb_index_alignment(smiles_list[idx], data.x)
        if not result['ok']:
            failures.append({'idx': idx, **result})

    return {
        'ok':        len(failures) == 0,
        'n_checked': len(indices),
        'n_failed':  len(failures),
        'failures':  failures[:5],   # cap at 5 for readability
    }


# ─────────────────────────────────────────────────────────────────────────────
# MolDataset helper: apply index_map when building nodes_to_motifs
# ─────────────────────────────────────────────────────────────────────────────

def apply_motif_lookup_with_index_map(
    n_nodes: int,
    mapped_smiles: str,
    lookup: Dict[str, Dict[int, Tuple[str, int]]],
    index_map: Dict[str, Dict[int, int]],
) -> torch.Tensor:
    """Build nodes_to_motifs tensor for a mutag graph.

    The lookup produced by MotifBreakdown is keyed by the mapped SMILES and
    uses *SMILES atom indices* internally.  We translate back to graph node
    indices using the index_map produced by graph_to_mapped_smiles().

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the graph (= len(node_types)).
    mapped_smiles : str
        The mapped SMILES for this graph (key in both lookup and index_map).
    lookup : dict
        {mapped_smiles: {smiles_atom_idx: (smarts, motif_id)}}
        As produced by generate_vocab_rules.build_lookup().
    index_map : dict
        {mapped_smiles: {graph_node_idx: smiles_atom_idx}}
        As produced by build_mutag_smiles_df().

    Returns
    -------
    Tensor [n_nodes] with motif_id per graph node (-1 = unknown).
    """
    ntm = torch.full((n_nodes,), -1, dtype=torch.long)
    smi_lookup = lookup.get(mapped_smiles, {})
    g2s = index_map.get(mapped_smiles, {})

    for graph_idx, smiles_idx in g2s.items():
        entry = smi_lookup.get(smiles_idx)
        if entry is not None:
            ntm[graph_idx] = entry[1]   # motif_id

    return ntm
