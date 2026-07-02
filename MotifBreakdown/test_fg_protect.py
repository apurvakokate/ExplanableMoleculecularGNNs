#!/usr/bin/env python3
"""test_fg_protect.py — tests for functional-group protection (nitro + aniline).

Covers detection (protected_atomsets) and the carve (carve_protected):
  * nitro owns {N, O, O}; aniline owns just {N} with the ipso ring C as key context
  * aliphatic amines / amides are NOT protected
  * the carve preserves the atom partition and keeps the ring intact
  * the aniline motif is labelled distinctly (not a generic bare `*N`)

Run:  python test_fg_protect.py -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

import chemfrag as C
import fg_protect as F


def _keyer(m, aset):
    return C.frag_key(m, aset)


def _norm(k):
    return str(k).replace('[*]', '*').replace('([*])', '(*)')


class TestProtectedAtomsets(unittest.TestCase):

    def test_nitro_owns_N_and_two_O(self):
        m = Chem.MolFromSmiles('O=[N+]([O-])c1ccccc1')  # nitrobenzene
        prot = F.protected_atomsets(m)
        self.assertIn('nitro', [t for t, _, _ in prot])
        nitro = [s for t, s, _ in prot if t == 'nitro'][0]
        self.assertEqual(sorted(m.GetAtomWithIdx(i).GetSymbol() for i in nitro),
                         ['N', 'O', 'O'])

    def test_aniline_owns_only_N_context_is_N_plus_ipsoC(self):
        m = Chem.MolFromSmiles('Nc1ccccc1')  # aniline
        anil = [(s, k) for t, s, k in F.protected_atomsets(m) if t == 'aniline']
        self.assertEqual(len(anil), 1)
        owned, kctx = anil[0]
        self.assertEqual(len(owned), 1)                       # owns just the N
        self.assertEqual(m.GetAtomWithIdx(next(iter(owned))).GetSymbol(), 'N')
        self.assertEqual(len(kctx), 2)                        # label context = N + ipso C
        self.assertTrue(owned < kctx)

    def test_aliphatic_primary_amine_not_protected(self):
        m = Chem.MolFromSmiles('CCN')  # ethylamine — no ring attachment
        self.assertEqual([t for t, _, _ in F.protected_atomsets(m)], [])

    def test_amide_not_protected(self):
        m = Chem.MolFromSmiles('CC(=O)Nc1ccccc1')  # acetanilide — amide N
        self.assertNotIn('aniline', [t for t, _, _ in F.protected_atomsets(m)])

    def test_nonoverlapping(self):
        m = Chem.MolFromSmiles('Nc1ccc([N+](=O)[O-])cc1')  # p-nitroaniline
        used = set()
        for _, s, _ in F.protected_atomsets(m):
            self.assertTrue(s.isdisjoint(used))
            used |= set(s)


class TestCarve(unittest.TestCase):

    def _carve(self, smi):
        m = Chem.MolFromSmiles(smi)
        whole = [(_keyer(m, set(range(m.GetNumAtoms()))), set(range(m.GetNumAtoms())))]
        return m, F.carve_protected(m, whole, _keyer)

    def test_partition_preserved(self):
        m, carved = self._carve('Nc1ccc([N+](=O)[O-])cc1')  # p-nitroaniline
        union = set().union(*[a for _, a in carved])
        self.assertEqual(union, set(range(m.GetNumAtoms())))
        self.assertEqual(sum(len(a) for _, a in carved), m.GetNumAtoms())

    def test_nitro_group_emitted_owning_three_atoms(self):
        m, carved = self._carve('O=[N+]([O-])c1ccccc1')  # nitrobenzene
        nitro = [(k, a) for k, a in carved
                 if len(a) == 3 and sorted(m.GetAtomWithIdx(i).GetSymbol() for i in a) == ['N', 'O', 'O']]
        self.assertEqual(len(nitro), 1)

    def test_aniline_distinct_label_owns_only_N(self):
        m, carved = self._carve('Nc1ccccc1')  # aniline
        anil = [(k, a) for k, a in carved
                if len(a) == 1 and m.GetAtomWithIdx(next(iter(a))).GetSymbol() == 'N']
        self.assertEqual(len(anil), 1)
        k = _norm(anil[0][0])
        self.assertNotIn(k, ('*N', 'N'))                     # not a generic bare amine
        self.assertTrue('N' in k and ('c' in k or 'C' in k))  # ring-C context in the label

    def test_ring_stays_intact(self):
        m, carved = self._carve('Nc1ccccc1')  # aniline → ring must remain whole (6 C)
        ring_frag = max((a for _, a in carved), key=len)
        self.assertEqual(len(ring_frag), 6)


class TestFlatMutagRepresentation(unittest.TestCase):
    """Validate detection/carve on the ACTUAL mutag export representation:
    atom-mapped, explicit-H, NON-aromatic single-bond SMILES (bond orders and
    aromaticity are lost in the TUDataset graph->SMILES step). Connectivity-based
    detection (N with >=2 O neighbours; ring-attached primary amine via IsInRing)
    must still fire here, not just on clean aromatic SMILES."""

    # a real exported mutag molecule carrying both toxicophores (nitro as N(OH)(OH),
    # aniline as -NH2 on a saturated ring carbon; note: no aromatic atoms)
    FLAT = ('[CH:1]1([N:4]([OH:8])[OH:9])[CH:2]([O:6][H:14])[CH:5]([H:13])'
            '[CH:10]([H:15])[CH:7]([N:11]([H:16])[H:17])[CH:3]1[H:12]')

    def test_parses_and_is_non_aromatic(self):
        m = Chem.MolFromSmiles(self.FLAT)
        self.assertIsNotNone(m)
        self.assertFalse(any(a.GetIsAromatic() for a in m.GetAtoms()))

    def test_both_toxicophores_detected_on_flat_smiles(self):
        m = Chem.MolFromSmiles(self.FLAT)
        tags = set(t for t, _, _ in F.protected_atomsets(m))
        self.assertEqual(tags, {'nitro', 'aniline'})

    def test_carve_partition_preserved_on_flat_smiles(self):
        m = Chem.MolFromSmiles(self.FLAT)
        whole = [(_keyer(m, set(range(m.GetNumAtoms()))), set(range(m.GetNumAtoms())))]
        carved = F.carve_protected(m, whole, _keyer)
        self.assertEqual(set().union(*[a for _, a in carved]), set(range(m.GetNumAtoms())))
        self.assertEqual(sum(len(a) for _, a in carved), m.GetNumAtoms())


if __name__ == '__main__':
    unittest.main(verbosity=2)
