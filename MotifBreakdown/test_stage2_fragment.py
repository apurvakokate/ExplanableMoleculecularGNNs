"""Rule Set 1 — Stage 2 (Fragment) unit tests.

Per-instance, per-atom, semantic checks on the three FG-first detectors, run over a
small real sample + hand-built oracle molecules. These target the failure modes an
aggregate coverage% cannot see: an atom owned by two motifs, an unclaimed atom, a
DISCONNECTED motif (the O.O.O class), a mislabeled ring, a leaked internal key.

Detectors: fg_first_frag / ertl_frag / rdkit_fg_frag  (.partition + rekey_structural).
Merge (cascade_bpe_linker) and threshold/filter get their own module.
"""
import os
import unittest
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import fg_first_frag as fgf
import ertl_frag as ef
import rdkit_fg_frag as rf

fgf.set_ring_identity('canonical')

DETECTORS = {'fg_first': fgf.partition, 'ertl_first': ef.partition, 'rdkit_fg_first': rf.partition}
PART_KW = dict(subcut_chains=False, whole_ring_systems=True)

_ROOT = os.path.join(os.path.dirname(__file__), '..', 'sweep', 'FOLDS')


def _sample(n=250):
    smis = []
    for ds in ('BBBP', 'Mutagenicity'):
        p = os.path.join(_ROOT, f'{ds}_0.csv')
        if not os.path.exists(p):
            continue
        import csv
        with open(p) as f:
            for row in csv.DictReader(f):
                s = row.get('smiles')
                if s and Chem.MolFromSmiles(s):
                    smis.append(s)
                if len(smis) >= n:
                    break
        if len(smis) >= n:
            break
    return smis


def _groups(owner):
    g = {}
    for a, f in enumerate(owner):
        g.setdefault(f, set()).add(a)
    return g


class Stage2Partition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.smis = _sample()

    def _each(self):
        """Yield (detector_name, part_fn, mol) over the sample; skip if no data."""
        if not self.smis:
            self.skipTest('no local FOLDS sample')
        for name, part in DETECTORS.items():
            for s in self.smis:
                yield name, part, Chem.MolFromSmiles(s), s

    def test_partition_exact(self):
        """Every atom in exactly one motif: union == all atoms AND pairwise disjoint."""
        for name, part, m, s in self._each():
            owner, _ = part(m, **PART_KW)
            self.assertEqual(len(owner), m.GetNumAtoms(), f'{name} {s}: owner length')
            g = _groups(owner)
            total = sum(len(v) for v in g.values())
            self.assertEqual(total, m.GetNumAtoms(),
                             f'{name} {s}: atoms owned {total} != {m.GetNumAtoms()} (overlap/drop)')
            self.assertEqual(set().union(*g.values()) if g else set(), set(range(m.GetNumAtoms())),
                             f'{name} {s}: union != all atoms')

    def test_no_unclaimed(self):
        """No atom left with owner -1 at partition time."""
        for name, part, m, s in self._each():
            owner, _ = part(m, **PART_KW)
            self.assertNotIn(-1, owner, f'{name} {s}: has an unclaimed atom')

    def test_motif_connected(self):
        """Each motif's atom set is connected (the O.O.O disconnected-motif class)."""
        for name, part, m, s in self._each():
            owner, _ = part(m, **PART_KW)
            for f, ats in _groups(owner).items():
                self.assertTrue(fgf._connected(m, ats),
                                f'{name} {s}: motif {f} atoms {ats} disconnected')

    def test_ring_semantic(self):
        """A motif keyed 'ring:...' really is a ring atom-set (label truthfulness)."""
        for name, part, m, s in self._each():
            owner, ident = part(m, **PART_KW)
            for f, ats in _groups(owner).items():
                if ident[f].startswith('ring:'):
                    self.assertTrue(fgf._is_ring_set(m, ats),
                                    f'{name} {s}: ring-keyed motif {ats} is not a ring set')

    def test_no_literal_key_leak(self):
        """After rekey_structural, no motif key leaks an internal fg:/chain:/frag: tag."""
        for name, part, m, s in self._each():
            owner, ident = part(m, **PART_KW)
            frags = [(ident[f], ats) for f, ats in _groups(owner).items()]
            for key, _ in fgf.rekey_structural(m, frags):
                self.assertFalse(key.startswith(('fg:', 'chain:', 'frag:')),
                                 f'{name} {s}: leaked internal key {key!r}')

    def test_partition_deterministic(self):
        """partition + rekey twice → identical (key, sorted atoms)."""
        for name, part, m, s in self._each():
            def run():
                owner, ident = part(m, **PART_KW)
                frags = [(ident[f], ats) for f, ats in _groups(owner).items()]
                return sorted((k, tuple(sorted(a))) for k, a in fgf.rekey_structural(m, frags))
            self.assertEqual(run(), run(), f'{name} {s}: non-deterministic')


class Stage2Oracle(unittest.TestCase):
    """Known decompositions — the semantic ground truth an FG detector must reproduce."""

    def _decomp(self, part, smi):
        m = Chem.MolFromSmiles(smi)
        owner, ident = part(m, **PART_KW)
        return m, _groups(owner), ident

    def test_nitrobenzene(self):
        """c1ccccc1[N+](=O)[O-]: one ring motif = 6 aromatic C; one nitro motif = {N,O,O}."""
        smi = 'c1ccccc1[N+](=O)[O-]'
        for name, part in DETECTORS.items():
            with self.subTest(detector=name):
                m, g, ident = self._decomp(part, smi)
                sym = {i: m.GetAtomWithIdx(i).GetSymbol() for i in range(m.GetNumAtoms())}
                arom_c = {i for i in sym if sym[i] == 'C' and m.GetAtomWithIdx(i).GetIsAromatic()}
                nitro = {i for i in sym if sym[i] in ('N', 'O')}
                motifs = list(g.values())
                self.assertIn(arom_c, motifs, f'{name}: benzene ring not one motif ({arom_c})')
                # the N and both O share a single motif (the whole nitro group)
                n_motif = next(a for a in motifs if any(sym[i] == 'N' for i in a))
                self.assertTrue(nitro.issubset(n_motif),
                                f'{name}: nitro atoms split across motifs: {n_motif}')
                self.assertTrue(fgf._connected(m, n_motif), f'{name}: nitro motif disconnected')

    def test_partition_exact_oracles(self):
        for smi in ('CC(=O)Oc1ccccc1C(=O)O', 'c1ccc2ccccc2c1', 'O=C(N)c1ccccc1'):  # aspirin, naphthalene, benzamide
            for name, part in DETECTORS.items():
                with self.subTest(detector=name, smi=smi):
                    m, g, _ = self._decomp(part, smi)
                    self.assertEqual(sum(len(v) for v in g.values()), m.GetNumAtoms())
                    for ats in g.values():
                        self.assertTrue(fgf._connected(m, ats))


class Stage2CrossMethod(unittest.TestCase):
    """ertl and rdkit_fg are different algorithms → their vocabs must differ (no silent
    cross-fallback where one secretly runs the other)."""

    def test_ertl_rdkit_vocab_differ(self):
        smis = _sample()
        if not smis:
            self.skipTest('no local FOLDS sample')

        def vocab(part):
            keys = set()
            for s in smis:
                m = Chem.MolFromSmiles(s)
                owner, ident = part(m, **PART_KW)
                frags = [(ident[f], ats) for f, ats in _groups(owner).items()]
                keys.update(k for k, _ in fgf.rekey_structural(m, frags))
            return keys
        self.assertNotEqual(vocab(ef.partition), vocab(rf.partition),
                            'ertl and rdkit_fg produced identical vocabularies')


if __name__ == '__main__':
    unittest.main(verbosity=2)
