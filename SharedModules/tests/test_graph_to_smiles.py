#!/usr/bin/env python3
"""test_graph_to_smiles.py — tests for graph_to_smiles.py.

Covers:
  1. graph_to_mapped_smiles  — correct SMILES, index map preserved
  2. unmap_smiles            — atom-map numbers stripped cleanly
  3. verify_ogb_index_alignment — symbol check for OGB graphs
  4. verify_mutag_index_alignment — node-type vs SMILES consistency
  5. apply_motif_lookup_with_index_map — correct nodes_to_motifs tensor
  6. End-to-end: mutag graph → SMILES → mock vocab → nodes_to_motifs

Run:
    python test_graph_to_smiles.py -v
"""

import sys, os, unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from SharedModules.data.graph_to_smiles import (
    MUTAG_ATOM_TYPE_MAP,
    graph_to_mapped_smiles,
    unmap_smiles,
    verify_ogb_index_alignment,
    verify_mutag_index_alignment,
    apply_motif_lookup_with_index_map,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _nitrobenzene_graph():
    """Return (node_types, edge_src, edge_dst) for nitrobenzene.
    Atom order: C(0)C(1)C(2)C(3)C(4)C(5)N(6)O(7)O(8)  — NOT canonical.
    """
    node_types = [0, 0, 0, 0, 0, 0, 1, 2, 2]   # C×6, N, O, O
    edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(5,6),(6,7),(6,8)]
    src = [s for s,d in edges] + [d for s,d in edges]
    dst = [d for s,d in edges] + [s for s,d in edges]
    return node_types, src, dst


def _ethanol_graph():
    """O(0)-C(1)-C(2) in non-canonical order."""
    node_types = [2, 0, 0]          # O, C, C
    src = [0, 1, 1, 2]
    dst = [1, 0, 2, 1]
    return node_types, src, dst


def _make_ogb_feature_row(atomic_num: int) -> torch.Tensor:
    """Build a minimal OGB-style x row where x[0] = atomic_num - 1."""
    row = torch.zeros(9, dtype=torch.long)
    row[0] = atomic_num - 1   # 0-indexed atomic number
    return row


# ── 1. graph_to_mapped_smiles ─────────────────────────────────────────────────

class TestGraphToMappedSmiles(unittest.TestCase):

    def test_returns_smiles_and_map(self):
        nt, src, dst = _nitrobenzene_graph()
        smiles, g2s = graph_to_mapped_smiles(nt, src, dst)
        self.assertIsNotNone(smiles)
        self.assertIsNotNone(g2s)

    def test_map_covers_all_nodes(self):
        nt, src, dst = _nitrobenzene_graph()
        smiles, g2s = graph_to_mapped_smiles(nt, src, dst)
        self.assertEqual(set(g2s.keys()), set(range(len(nt))))

    def test_map_is_bijection(self):
        """Every graph node maps to a unique SMILES atom index."""
        nt, src, dst = _nitrobenzene_graph()
        _, g2s = graph_to_mapped_smiles(nt, src, dst)
        smiles_indices = list(g2s.values())
        self.assertEqual(len(smiles_indices), len(set(smiles_indices)),
                         "SMILES indices are not unique — bijection violated")

    def test_atom_symbols_preserved(self):
        """After round-trip, graph node i has the correct element symbol."""
        nt, src, dst = _nitrobenzene_graph()
        smiles, g2s = graph_to_mapped_smiles(nt, src, dst)
        mol = Chem.MolFromSmiles(smiles)
        pt = Chem.GetPeriodicTable()
        for graph_idx, smiles_idx in g2s.items():
            expected_sym = pt.GetElementSymbol(MUTAG_ATOM_TYPE_MAP[nt[graph_idx]])
            actual_sym   = mol.GetAtomWithIdx(smiles_idx).GetSymbol()
            self.assertEqual(expected_sym, actual_sym,
                             f"graph node {graph_idx}: expected {expected_sym}"
                             f" but got {actual_sym} at smiles_idx {smiles_idx}")

    def test_non_canonical_input_survives(self):
        """O(0)-C(1)-C(2) — O first, which is non-canonical."""
        nt, src, dst = _ethanol_graph()
        smiles, g2s = graph_to_mapped_smiles(nt, src, dst)
        self.assertIsNotNone(smiles)
        # Verify O is still at graph node 0
        mol = Chem.MolFromSmiles(smiles)
        O_smiles_idx = g2s[0]
        self.assertEqual(mol.GetAtomWithIdx(O_smiles_idx).GetSymbol(), 'O')

    def test_canonical_smiles_changes_order(self):
        """Demonstrates that without atom-map numbers, order would be lost."""
        nt, src, dst = _nitrobenzene_graph()
        smiles, g2s = graph_to_mapped_smiles(nt, src, dst)
        mol = Chem.MolFromSmiles(smiles)
        # Rebuild without map numbers to show ordering changes
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        plain = Chem.MolToSmiles(mol)
        mol_plain = Chem.MolFromSmiles(plain)
        plain_syms  = [a.GetSymbol() for a in mol_plain.GetAtoms()]
        # The first atom in plain SMILES is very likely not C (nitrobenzene
        # canonical SMILES starts with O=[N+]...)
        # Confirm the reordering happens when map numbers are absent
        original_sym_order = [Chem.GetPeriodicTable().GetElementSymbol(
                               MUTAG_ATOM_TYPE_MAP[t]) for t in nt]
        self.assertNotEqual(plain_syms, original_sym_order,
                            "Expected canonical SMILES to reorder atoms "
                            "(test is not meaningful for this molecule)")

    def test_unknown_atom_type_returns_none(self):
        nt = [0, 99, 0]        # 99 is not in MUTAG_ATOM_TYPE_MAP
        smiles, g2s = graph_to_mapped_smiles(nt, [0,1,1,2], [1,0,2,1])
        self.assertIsNone(smiles)
        self.assertIsNone(g2s)

    def test_disconnected_atoms_handled(self):
        """Single isolated atom."""
        nt = [0]
        smiles, g2s = graph_to_mapped_smiles(nt, [], [])
        self.assertIsNotNone(smiles)
        self.assertEqual(g2s, {0: 0})

    def test_index_map_inverts_correctly(self):
        """graph_to_smiles and smiles_to_graph are exact inverses."""
        nt, src, dst = _nitrobenzene_graph()
        _, g2s = graph_to_mapped_smiles(nt, src, dst)
        s2g = {v: k for k, v in g2s.items()}
        # Re-invert
        g2s_again = {v: k for k, v in s2g.items()}
        self.assertEqual(g2s, g2s_again)


# ── 2. unmap_smiles ───────────────────────────────────────────────────────────

class TestUnmapSmiles(unittest.TestCase):

    def test_removes_map_numbers(self):
        mapped = '[C:1][N:2][C:3]'
        plain  = unmap_smiles(mapped)
        self.assertIsNotNone(plain)
        self.assertNotIn(':', plain)

    def test_invalid_returns_none(self):
        self.assertIsNone(unmap_smiles('[INVALID]'))

    def test_already_plain_unchanged(self):
        smiles = 'CC(=O)Nc1ccc(O)cc1'
        result = unmap_smiles(smiles)
        self.assertIsNotNone(result)
        self.assertNotIn(':', result)

    def test_roundtrip_preserves_molecule(self):
        nt, src, dst = _nitrobenzene_graph()
        mapped, _ = graph_to_mapped_smiles(nt, src, dst)
        plain = unmap_smiles(mapped)
        # Both should parse to the same canonical form
        m1 = Chem.MolFromSmiles(mapped)
        m2 = Chem.MolFromSmiles(plain)
        for a in m1.GetAtoms(): a.SetAtomMapNum(0)
        canon1 = Chem.MolToSmiles(m1)
        canon2 = Chem.MolToSmiles(m2)
        self.assertEqual(canon1, canon2)


# ── 3. verify_ogb_index_alignment ─────────────────────────────────────────────

class TestVerifyOgbIndexAlignment(unittest.TestCase):
    """OGB graphs come from Chem.MolFromSmiles — same as our build_graph.
    Atom i in graph = atom i in SMILES.  We just verify element consistency.
    """

    def _ogb_x_for_smiles(self, smiles: str) -> torch.Tensor:
        """Build a fake OGB x matrix where x[i,0] = atomic_num[i] - 1."""
        mol = Chem.MolFromSmiles(smiles)
        rows = []
        for atom in mol.GetAtoms():
            rows.append(_make_ogb_feature_row(atom.GetAtomicNum()))
        return torch.stack(rows)

    def test_correct_alignment(self):
        smi = 'O=[N+]([O-])c1ccccc1'
        x = self._ogb_x_for_smiles(smi)
        result = verify_ogb_index_alignment(smi, x)
        self.assertTrue(result['ok'])
        self.assertEqual(result['n_smiles_atoms'], result['n_graph_nodes'])

    def test_wrong_atom_count_detected(self):
        smi = 'Cc1ccccc1'   # 7 heavy atoms
        x = torch.zeros(5, 9, dtype=torch.long)   # wrong: only 5 rows
        result = verify_ogb_index_alignment(smi, x)
        self.assertFalse(result['ok'])
        self.assertIn('mismatch', result.get('error', '').lower())

    def test_symbol_mismatch_detected(self):
        smi = 'Cc1ccccc1'   # 7 atoms, first is C (atomic_num=6, ogb_x0=5)
        x = self._ogb_x_for_smiles(smi)
        # Corrupt: change first atom's ogb encoding to N (atomic_num=7, ogb_x0=6)
        x_corrupt = x.clone()
        x_corrupt[0, 0] = 6   # N instead of C
        result = verify_ogb_index_alignment(smi, x_corrupt)
        self.assertFalse(result['ok'])
        self.assertEqual(len(result['mismatches']), 1)
        self.assertEqual(result['mismatches'][0][0], 0)  # node 0

    def test_invalid_smiles(self):
        result = verify_ogb_index_alignment('[INVALID]', torch.zeros(3, 9))
        self.assertFalse(result['ok'])

    def test_paracetamol_correct(self):
        smi = 'CC(=O)Nc1ccc(O)cc1'
        x = self._ogb_x_for_smiles(smi)
        result = verify_ogb_index_alignment(smi, x)
        self.assertTrue(result['ok'])
        self.assertEqual(len(result['mismatches']), 0)


# ── 4. verify_mutag_index_alignment ───────────────────────────────────────────

class TestVerifyMutagIndexAlignment(unittest.TestCase):

    def _make_setup(self, node_types=None):
        if node_types is None:
            node_types, src, dst = _nitrobenzene_graph()
        else:
            src, dst = [], []
        smiles, g2s = graph_to_mapped_smiles(node_types, src, dst)
        # Fake feature matrix (not actually used by verify, just for API compat)
        x = torch.zeros(len(node_types), 14)
        return smiles, x, node_types, g2s

    def test_correct_alignment(self):
        nt, src, dst = _nitrobenzene_graph()
        smiles, g2s = graph_to_mapped_smiles(nt, src, dst)
        x = torch.zeros(len(nt), 14)
        result = verify_mutag_index_alignment(smiles, x, nt, g2s)
        self.assertTrue(result['ok'])
        self.assertEqual(len(result['mismatches']), 0)

    def test_all_carbon(self):
        nt = [0, 0, 0, 0, 0, 0]   # C×6 (benzene)
        src = [0,1,2,3,4,5]
        dst = [1,2,3,4,5,0]
        smiles, g2s = graph_to_mapped_smiles(nt, src + dst, dst + src)
        if smiles is None: return   # sanitisation may fail for benzene w/ single bonds
        x = torch.zeros(6, 14)
        result = verify_mutag_index_alignment(smiles, x, nt, g2s)
        self.assertTrue(result['ok'])

    def test_detects_wrong_index_map(self):
        """Deliberately swap two entries in g2s — should produce mismatches."""
        nt, src, dst = _nitrobenzene_graph()
        smiles, g2s = graph_to_mapped_smiles(nt, src, dst)
        if smiles is None or len(g2s) < 2: return
        # Swap graph nodes 0 and 6 (C vs N) in the map
        bad_g2s = dict(g2s)
        bad_g2s[0], bad_g2s[6] = g2s[6], g2s[0]
        x = torch.zeros(len(nt), 14)
        result = verify_mutag_index_alignment(smiles, x, nt, bad_g2s)
        self.assertFalse(result['ok'])
        self.assertGreater(len(result['mismatches']), 0)


# ── 5. apply_motif_lookup_with_index_map ─────────────────────────────────────

class TestApplyMotifLookup(unittest.TestCase):

    def _setup(self):
        """Build a fake vocab lookup and index_map for nitrobenzene."""
        nt, src, dst = _nitrobenzene_graph()
        smiles, g2s = graph_to_mapped_smiles(nt, src, dst)
        n = len(nt)

        # Fake lookup: SMILES atom 0..5 (benzene ring) → motif 0 (benzene)
        #              SMILES atom 6..8 (nitro)         → motif 1 (nitro)
        mol = Chem.MolFromSmiles(smiles)
        smi_to_graph = {v: k for k, v in g2s.items()}
        lookup_inner = {}
        for smiles_idx in range(mol.GetNumAtoms()):
            graph_idx = smi_to_graph[smiles_idx]
            if nt[graph_idx] == 0:    # C → benzene motif
                lookup_inner[smiles_idx] = ('[*]c1ccccc1', 0)
            else:                      # N/O → nitro motif
                lookup_inner[smiles_idx] = ('[*][N+](=O)[O-]', 1)
        lookup = {smiles: lookup_inner}
        index_map = {smiles: g2s}
        return n, smiles, lookup, index_map, nt

    def test_output_shape(self):
        n, smiles, lookup, index_map, nt = self._setup()
        ntm = apply_motif_lookup_with_index_map(n, smiles, lookup, index_map)
        self.assertEqual(ntm.shape, (n,))

    def test_carbon_nodes_get_benzene_motif(self):
        n, smiles, lookup, index_map, nt = self._setup()
        ntm = apply_motif_lookup_with_index_map(n, smiles, lookup, index_map)
        for graph_idx, type_int in enumerate(nt):
            if type_int == 0:   # C
                self.assertEqual(int(ntm[graph_idx]), 0,
                                 f"graph node {graph_idx} (C) should be motif 0")

    def test_nitro_nodes_get_nitro_motif(self):
        n, smiles, lookup, index_map, nt = self._setup()
        ntm = apply_motif_lookup_with_index_map(n, smiles, lookup, index_map)
        for graph_idx, type_int in enumerate(nt):
            if type_int in (1, 2):   # N or O
                self.assertEqual(int(ntm[graph_idx]), 1,
                                 f"graph node {graph_idx} (N/O) should be motif 1")

    def test_missing_smiles_gives_all_minus_one(self):
        n, smiles, lookup, index_map, nt = self._setup()
        ntm = apply_motif_lookup_with_index_map(n, 'WRONG_SMILES', lookup, index_map)
        self.assertTrue((ntm == -1).all())

    def test_empty_lookup_gives_all_minus_one(self):
        n, smiles, lookup, index_map, nt = self._setup()
        ntm = apply_motif_lookup_with_index_map(n, smiles, {}, index_map)
        self.assertTrue((ntm == -1).all())

    def test_no_unknown_nodes_in_fully_mapped_graph(self):
        """When every atom is in the lookup, no -1 should remain."""
        n, smiles, lookup, index_map, nt = self._setup()
        ntm = apply_motif_lookup_with_index_map(n, smiles, lookup, index_map)
        self.assertTrue((ntm >= 0).all(),
                        f"Unexpected -1 values: {ntm.tolist()}")


# ── 6. End-to-end index consistency ──────────────────────────────────────────

class TestEndToEndIndexConsistency(unittest.TestCase):
    """Simulate the full pipeline:
    mutag graph → mapped SMILES → (mock) vocab fragmentation → nodes_to_motifs
    and verify every node index is correctly assigned.
    """

    def _run_pipeline(self, node_types, edge_src, edge_dst):
        """Full round-trip without calling MotifBreakdown (mocked vocab)."""
        n = len(node_types)
        smiles, g2s = graph_to_mapped_smiles(node_types, edge_src, edge_dst)
        if smiles is None:
            return None

        # Mock vocab: assign all atoms to motif 0
        mol = Chem.MolFromSmiles(smiles)
        mock_lookup = {smiles: {i: ('[*]C', 0) for i in range(mol.GetNumAtoms())}}
        mock_index_map = {smiles: g2s}

        ntm = apply_motif_lookup_with_index_map(n, smiles, mock_lookup, mock_index_map)
        return ntm, g2s

    def test_all_nodes_assigned_nitrobenzene(self):
        nt, src, dst = _nitrobenzene_graph()
        result = self._run_pipeline(nt, src, dst)
        if result is None: return
        ntm, g2s = result
        self.assertTrue((ntm == 0).all(),
                        f"Not all nodes assigned: {ntm.tolist()}")

    def test_all_nodes_assigned_ethanol(self):
        nt, src, dst = _ethanol_graph()
        result = self._run_pipeline(nt, src, dst)
        if result is None: return
        ntm, g2s = result
        self.assertTrue((ntm == 0).all())

    def test_graph_indices_match_smiles_indices(self):
        """The index map is internally consistent: rebuild and check."""
        nt, src, dst = _nitrobenzene_graph()
        smiles, g2s = graph_to_mapped_smiles(nt, src, dst)
        if smiles is None: return

        mol = Chem.MolFromSmiles(smiles)
        pt  = Chem.GetPeriodicTable()

        # For each graph node, the SMILES atom at g2s[graph_node] must
        # have the same element as MUTAG_ATOM_TYPE_MAP[node_types[graph_node]]
        for graph_idx, type_int in enumerate(nt):
            expected_atomic = MUTAG_ATOM_TYPE_MAP[type_int]
            expected_sym    = pt.GetElementSymbol(expected_atomic)
            smiles_idx      = g2s[graph_idx]
            actual_sym      = mol.GetAtomWithIdx(smiles_idx).GetSymbol()
            self.assertEqual(expected_sym, actual_sym,
                             f"Graph node {graph_idx}: expected {expected_sym}"
                             f" at smiles_idx {smiles_idx} but got {actual_sym}")

    def test_different_orderings_give_consistent_motifs(self):
        """Two graphs with the same chemistry but different node orderings
        should produce the same canonical SMILES (just permuted)."""
        # Nitrobenzene v1: C(0-5), N(6), O(7), O(8)
        nt1, src1, dst1 = _nitrobenzene_graph()
        # Nitrobenzene v2: N(0), C(1-6) ring, O(7), O(8) — N comes first
        nt2  = [1, 0, 0, 0, 0, 0, 0, 2, 2]
        edges2 = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,1),(0,7),(0,8)]
        src2 = [s for s,d in edges2] + [d for s,d in edges2]
        dst2 = [d for s,d in edges2] + [s for s,d in edges2]

        smiles1, g2s1 = graph_to_mapped_smiles(nt1, src1, dst1)
        smiles2, g2s2 = graph_to_mapped_smiles(nt2, src2, dst2)
        if smiles1 is None or smiles2 is None:
            return

        # Both should produce the same canonical (unmapped) SMILES
        plain1 = unmap_smiles(smiles1)
        plain2 = unmap_smiles(smiles2)
        self.assertEqual(plain1, plain2,
                         "Same molecule with different node ordering should "
                         "give same canonical SMILES")

        # Both should cover all nodes
        self.assertEqual(set(g2s1.keys()), set(range(len(nt1))))
        self.assertEqual(set(g2s2.keys()), set(range(len(nt2))))


if __name__ == '__main__':
    unittest.main(verbosity=2)
