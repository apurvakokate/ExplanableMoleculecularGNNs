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


if __name__ == '__main__':
    unittest.main(verbosity=2)
