#!/usr/bin/env python3
"""
test_pipeline.py
================
One unit test per public method across:
  molfragbpe5.py          — fragmentation, hierarchy, BPE
  motif_label_pipeline.py — rule mining utilities
  generate_vocab_rules.py — tracked fragmentation, vocab, lookup, matrix

All tests use the method= / use_fallback= / use_bpe= API introduced in
the unified refactor. No VARIANTS dict, no old-style tuple arguments.

Run:
    python test_pipeline.py -v
    python test_pipeline.py TestBuildCascade
"""

import sys, os, unittest
from copy import deepcopy

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r-BRICS'))

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

import molfragbpe5 as frag
import motif_label_pipeline as pipe
import generate_vocab_rules as gvr
import chemfrag_v4_adapter as v4


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def mol(smi):
    return Chem.MolFromSmiles(smi)

# reBRICS adds cuts when a post-rBRICS fragment still contains CCCCCC.
# Plain alkyl chains (e.g. decane) are often fully split in pass 1 already.
_REBRICS_DIFF_CANDIDATES = (
    'c1ccccc1OCCCCCC',
    'c1ccccc1OCCCCCCC',
    'c1ccccc1CCCCCCCC',
    'CCOc1ccccc1CCCCCCCC',
    'CCCCCCCCCCCCCCCC',
)


def _pick_rebrics_diff_case():
    """Return (smi, n_rbrics_only, n_rbrics) for a mol where reBRICS adds cuts."""
    for smi in _REBRICS_DIFF_CANDIDATES:
        frag._CACHE.clear()
        m = mol(smi)
        if m is None:
            continue
        n_only = len(frag.cut_rbrics_only(m))
        n_full = len(frag.cut_rbrics(m))
        if n_full > n_only:
            return smi, n_only, n_full
    return None, None, None

def atom_n(m):
    return sum(1 for a in m.GetAtoms() if a.GetAtomicNum() not in (0, 1))

def frags_sum(cuts, n=None):
    s = sum(frag.atom_count(f) for f in cuts)
    return s == n if n is not None else s


# ═════════════════════════════════════════════════════════════════════════════
# molfragbpe5 — utilities
# ═════════════════════════════════════════════════════════════════════════════

class TestStrip(unittest.TestCase):
    def test_numbered_wildcard(self):
        self.assertEqual(frag.strip('[16*]CC'), '[*]CC')
    def test_bare_star(self):
        self.assertEqual(frag.strip('*CC'), '[*]CC')
    def test_mixed_formats(self):
        self.assertEqual(frag.strip('[4*]c1cc([8*])ccc1'), '[*]c1cc([*])ccc1')
    def test_already_normalised(self):
        self.assertEqual(frag.strip('[*]c1ccccc1'), '[*]c1ccccc1')


class TestAtomCount(unittest.TestCase):
    def test_benzene(self):
        self.assertEqual(frag.atom_count('[*]c1ccccc1'), 6)
    def test_wildcard_excluded(self):
        self.assertEqual(frag.atom_count('[*]C[*]'), 1)
    def test_nitro(self):
        self.assertEqual(frag.atom_count('[*][N+](=O)[O-]'), 3)
    def test_ethyl(self):
        self.assertEqual(frag.atom_count('[*]CC'), 2)


class TestFragDiameter(unittest.TestCase):
    def test_ethyl(self):
        self.assertEqual(frag.frag_diameter('[*]CC'), 1)
    def test_propyl(self):
        self.assertEqual(frag.frag_diameter('[*]CCC'), 2)
    def test_benzene(self):
        self.assertEqual(frag.frag_diameter('[*]c1ccccc1'), 3)
    def test_single_atom(self):
        self.assertEqual(frag.frag_diameter('[*]C'), 0)


class TestHasRing(unittest.TestCase):
    def test_benzene(self):
        self.assertTrue(frag.has_ring('[*]c1ccccc1'))
    def test_chain(self):
        self.assertFalse(frag.has_ring('[*]CC'))
    def test_piperazine(self):
        self.assertTrue(frag.has_ring('[*]N1CCNCC1'))


class TestIsTrivial(unittest.TestCase):
    def test_linker(self):
        self.assertTrue(frag.is_trivial('[*]C[*]'))
    def test_single_atom(self):
        self.assertTrue(frag.is_trivial('[*]C'))
    def test_benzene_not_trivial(self):
        self.assertFalse(frag.is_trivial('[*]c1ccccc1'))


# ─── build_cascade ─────────────────────────────────────────────────────────

class TestBuildCascade(unittest.TestCase):
    """build_cascade — returns method-specific list of (name, cut_fn) tuples."""

    def test_all_has_five_methods(self):
        c = frag.build_cascade('all')
        self.assertEqual(len(c), 5)
        names = [n for n, _ in c]
        self.assertEqual(names, ['rbrics', 'rbrics_only', 'brics', 'recap', 'murcko'])

    def test_rbrics_has_one_method(self):
        c = frag.build_cascade('rbrics')
        self.assertEqual(len(c), 1)
        self.assertEqual(c[0][0], 'rbrics')

    def test_brics_has_one_method(self):
        c = frag.build_cascade('brics')
        self.assertEqual(len(c), 1)
        self.assertEqual(c[0][0], 'brics')

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            frag.build_cascade('recap')   # recap alone is not a valid method key

    def test_returns_callable_fns(self):
        for _, fn in frag.build_cascade('all'):
            self.assertTrue(callable(fn))

    def test_all_includes_chemistry_cascade(self):
        # build_cascade('all') must equal CHEMISTRY_CASCADE
        self.assertEqual(frag.build_cascade('all'), frag.CHEMISTRY_CASCADE)

    def test_rbrics_only_has_one_method(self):
        c = frag.build_cascade('rbrics_only')
        self.assertEqual(len(c), 1)
        self.assertEqual(c[0][0], 'rbrics_only')


# ─── cut functions ──────────────────────────────────────────────────────────

@unittest.skipUnless(frag.RBRICS_OK, "rBRICS_public.py not found — skipping rBRICS tests")
class TestCutRbrics(unittest.TestCase):
    def setUp(self): frag._CACHE.clear()
    def test_ar_no2_cut(self):
        self.assertGreaterEqual(len(frag.cut_rbrics(mol('O=[N+]([O-])c1ccccc1'))), 2)
    def test_atom_conservation(self):
        m = mol('O=[N+]([O-])c1ccccc1')
        self.assertTrue(frags_sum(frag.cut_rbrics(m), atom_n(m)))
    def test_bare_ring_returns_empty(self):
        self.assertEqual(frag.cut_rbrics(mol('c1ccccc1')), [])


class TestCutBrics(unittest.TestCase):
    def setUp(self): frag._CACHE.clear()
    def test_ar_no2_not_cut(self):
        self.assertEqual(frag.cut_brics(mol('O=[N+]([O-])c1ccccc1')), [])
    def test_paracetamol(self):
        self.assertGreaterEqual(len(frag.cut_brics(mol('CC(=O)Nc1ccc(O)cc1'))), 2)
    def test_atom_conservation(self):
        m = mol('CC(=O)Nc1ccc(O)cc1')
        self.assertTrue(frags_sum(frag.cut_brics(m), atom_n(m)))


class TestCutRecap(unittest.TestCase):
    def setUp(self): frag._CACHE.clear()
    def test_amide(self):
        self.assertGreaterEqual(len(frag.cut_recap(mol('CC(=O)NC'))), 2)
    def test_sulfonamide(self):
        self.assertGreaterEqual(len(frag.cut_recap(mol('CNS(=O)(=O)c1ccccc1'))), 2)
    def test_ar_no2_blocked(self):
        self.assertEqual(frag.cut_recap(mol('O=[N+]([O-])c1ccccc1')), [])
    def test_atom_conservation(self):
        m = mol('CNS(=O)(=O)c1ccccc1')
        self.assertTrue(frags_sum(frag.cut_recap(m), atom_n(m)))


class TestCutMurcko(unittest.TestCase):
    def setUp(self): frag._CACHE.clear()
    def test_toluene(self):
        self.assertGreaterEqual(len(frag.cut_murcko(mol('Cc1ccccc1'))), 2)
    def test_bare_benzene_blocked(self):
        self.assertEqual(frag.cut_murcko(mol('c1ccccc1')), [])
    def test_atom_conservation(self):
        m = mol('Cc1ccccc1')
        self.assertTrue(frags_sum(frag.cut_murcko(m), atom_n(m)))


class TestCutRingChain(unittest.TestCase):
    def setUp(self): frag._CACHE.clear()
    def test_substituted_ring(self):
        self.assertGreaterEqual(len(frag.cut_ring_chain(mol('Cc1cc(O)c2c(=O)ccc2c1'))), 2)
    def test_pure_ring_blocked(self):
        self.assertEqual(frag.cut_ring_chain(mol('c1ccc2ccccc2c1')), [])
    def test_pure_chain_blocked(self):
        self.assertEqual(frag.cut_ring_chain(mol('CCCCCC')), [])
    def test_atom_conservation(self):
        m = mol('Cc1cc(O)c2c(=O)ccc2c1')
        self.assertTrue(frags_sum(frag.cut_ring_chain(m), atom_n(m)))


class TestCutAcyclicBonds(unittest.TestCase):
    def setUp(self): frag._CACHE.clear()
    def test_perfluoroalkyl(self):
        self.assertGreaterEqual(len(frag.cut_acyclic_bonds(mol('O=C(O)C(F)(F)F'))), 2)
    def test_benzene_blocked(self):
        self.assertEqual(frag.cut_acyclic_bonds(mol('c1ccccc1')), [])
    def test_atom_conservation(self):
        m = mol('O=C(O)C(F)(F)F')
        self.assertTrue(frags_sum(frag.cut_acyclic_bonds(m), atom_n(m)))


# ─── Hierarchy ──────────────────────────────────────────────────────────────

class TestHierarchy(unittest.TestCase):
    def test_add_molecule_root(self):
        h = frag.Hierarchy()
        h.add_molecule_root('c1ccccc1')
        self.assertEqual(h.nodes['c1ccccc1'].support, 1)
        self.assertEqual(h.nodes['c1ccccc1'].depth, 0)

    def test_touch_registers_once(self):
        h = frag.Hierarchy()
        n1 = h.touch('[*]c1ccccc1', None, depth=1, method='rbrics')
        n2 = h.touch('[*]c1ccccc1', None, depth=1, method='rbrics')
        self.assertIs(n1, n2)

    def test_add_cut_increments_support(self):
        h = frag.Hierarchy()
        h.add_molecule_root('Cc1ccccc1')
        h.add_cut('Cc1ccccc1', ['[*]c1ccccc1', '[*]C'], depth=1, method='rbrics')
        self.assertEqual(h.nodes['[*]c1ccccc1'].support, 1)

    def test_internal_fragment_nodes(self):
        h = frag.Hierarchy()
        h.add_molecule_root('Cc1ccccc1')
        h.add_cut('Cc1ccccc1', ['[*]c1ccccc1', '[*]C'], depth=1, method='rbrics')
        h.add_cut('[*]c1ccccc1', ['[*]c1ccc([*])cc1'], depth=2, method='murcko')
        self.assertIn('[*]c1ccccc1', h.internal_fragment_nodes())
        self.assertNotIn('Cc1ccccc1', h.internal_fragment_nodes())


# ─── fragment_recursive ─────────────────────────────────────────────────────

class TestFragmentRecursive(unittest.TestCase):
    def setUp(self): frag._CACHE.clear()

    def test_leaf_when_no_cut(self):
        h = frag.Hierarchy()
        h.add_molecule_root('[*]c1ccccc1')
        h.touch('[*]c1ccccc1', None, depth=1, method='test')
        result = frag.fragment_recursive('[*]c1ccccc1', h,
                                         frag.CHEMISTRY_CASCADE, depth=1)
        self.assertEqual(result, ['[*]c1ccccc1'])

    def test_toluene_splits_with_all(self):
        h = frag.Hierarchy()
        h.add_molecule_root('Cc1ccccc1')
        h.touch('[*]c1ccc(C)cc1', 'Cc1ccccc1', depth=1, method='rbrics')
        h.nodes['[*]c1ccc(C)cc1'].support = 1
        result = frag.fragment_recursive(
            '[*]c1ccc(C)cc1', h, frag.build_cascade('all'), depth=1)
        self.assertGreater(len(result), 1)

    def test_rbrics_cascade_only_uses_rbrics(self):
        # With rbrics-only cascade, BRICS/RECAP/Murcko should not fire
        h = frag.Hierarchy()
        h.add_molecule_root('[*]CC(=O)NC')
        h.touch('[*]CC(=O)NC', None, depth=1, method='test')
        h.nodes['[*]CC(=O)NC'].support = 1
        result = frag.fragment_recursive(
            '[*]CC(=O)NC', h, frag.build_cascade('rbrics'), depth=1)
        # rBRICS may or may not cut this — but must only use rBRICS
        methods_used = {n.cut_method for n in h.nodes.values()
                        if n.cut_method not in ('root','test','')}
        if methods_used:
            self.assertTrue(all(m == 'rbrics' for m in methods_used),
                            f"Non-rbrics method used: {methods_used}")


# ─── fragment_molecule ──────────────────────────────────────────────────────

class TestFragmentMolecule(unittest.TestCase):
    def setUp(self): frag._CACHE.clear()

    def test_nitrobenzene_splits_with_all(self):
        # rBRICS is the primary cutter for 'all' — only meaningful when available
        if not frag.RBRICS_OK:
            self.skipTest("rBRICS_public.py not found")
        h = frag.Hierarchy()
        leaves = frag.fragment_molecule(mol('O=[N+]([O-])c1ccccc1'), h,
                                         use_fallback=False, method='all')
        self.assertGreaterEqual(len(leaves), 2)

    @unittest.skipUnless(frag.RBRICS_OK, "rBRICS_public.py not found")
    def test_nitrobenzene_splits_with_rbrics(self):
        h = frag.Hierarchy()
        leaves = frag.fragment_molecule(mol('O=[N+]([O-])c1ccccc1'), h,
                                         use_fallback=False, method='rbrics')
        self.assertGreaterEqual(len(leaves), 2)

    def test_nitrobenzene_not_split_with_brics(self):
        # BRICS does not cut Ar-NO2 — molecule stays as single fragment
        h = frag.Hierarchy()
        leaves = frag.fragment_molecule(mol('O=[N+]([O-])c1ccccc1'), h,
                                         use_fallback=False, method='brics')
        self.assertEqual(len(leaves), 1)

    def test_fallback_fires_when_enabled(self):
        h1 = frag.Hierarchy()
        m = mol('O=C1Cc2cccc3cccc1c23')   # acenaphthenone
        leaves_no_fb = frag.fragment_molecule(m, h1, use_fallback=False, method='all')
        frag._CACHE.clear()
        h2 = frag.Hierarchy()
        leaves_fb = frag.fragment_molecule(m, h2, use_fallback=True, method='all')
        self.assertGreaterEqual(len(leaves_fb), len(leaves_no_fb))

    def test_method_all_default(self):
        # method defaults to 'all' — should work without specifying method
        h = frag.Hierarchy()
        leaves = frag.fragment_molecule(mol('Cc1ccccc1'), h)
        self.assertGreater(len(leaves), 0)

    def test_molecule_in_hierarchy(self):
        h = frag.Hierarchy()
        m = mol('CC(=O)Nc1ccc(O)cc1')
        canon_smi = frag.canon(m)
        frag.fragment_molecule(m, h, method='all')
        self.assertIn(canon_smi, h.nodes)

    def test_invalid_method_raises(self):
        h = frag.Hierarchy()
        with self.assertRaises(ValueError):
            frag.fragment_molecule(mol('c1ccccc1'), h, method='unknown')


# ─── bpe_merge ──────────────────────────────────────────────────────────────

class TestBpeMerge(unittest.TestCase):
    def _make_hier(self):
        h = frag.Hierarchy()
        h.nodes['[*]c1ccc(C)cc1'] = frag.HNode(
            '[*]c1ccc(C)cc1', None,
            {'[*]c1ccccc1', '[*]C'}, support=10, depth=1, cut_method='rbrics')
        h.nodes['[*]c1ccccc1'] = frag.HNode(
            '[*]c1ccccc1', '[*]c1ccc(C)cc1', set(), support=10, depth=2,
            cut_method='brics')
        h.nodes['[*]C'] = frag.HNode(
            '[*]C', '[*]c1ccc(C)cc1', set(), support=10, depth=2,
            cut_method='brics')
        return h

    def test_merge_fires(self):
        h = self._make_hier()
        mf = [['[*]c1ccccc1', '[*]C']] * 10
        _, hist = frag.bpe_merge(deepcopy(mf), h, n=10,
                                  min_atoms=3, max_diam=3, sz_max=18, min_abs=5,
                                  max_child_sup=1.0)
        self.assertGreaterEqual(len(hist), 1)

    def test_merge_blocked_by_high_child_support(self):
        # Guard 8: child [*]c1ccccc1 appears in 10/10 mols → 100% > max_child_sup=5%
        h = self._make_hier()
        mf = [['[*]c1ccccc1', '[*]C']] * 10
        _, hist = frag.bpe_merge(deepcopy(mf), h, n=10,
                                  min_atoms=3, max_diam=3, sz_max=18, min_abs=5,
                                  max_child_sup=0.05)
        self.assertEqual(len(hist), 0,
            "Merge should be blocked when child support > max_child_sup")

    def test_merge_allowed_with_high_max_child_sup(self):
        # Guard 8 relaxed to 100% — merge should fire
        h = self._make_hier()
        mf = [['[*]c1ccccc1', '[*]C']] * 10
        _, hist = frag.bpe_merge(deepcopy(mf), h, n=10,
                                  min_atoms=3, max_diam=3, sz_max=18, min_abs=5,
                                  max_child_sup=1.0)
        self.assertGreaterEqual(len(hist), 1)

    def test_enc_saved_in_history(self):
        # enc_saved = n_merged × (n_children - 1)
        h = self._make_hier()
        mf = [['[*]c1ccccc1', '[*]C']] * 10
        _, hist = frag.bpe_merge(deepcopy(mf), h, n=10,
                                  min_atoms=3, max_diam=3, sz_max=18, min_abs=5,
                                  max_child_sup=1.0)
        if hist:
            self.assertIn('enc_saved', hist[0])
            self.assertEqual(hist[0]['enc_saved'],
                             hist[0]['n_merged'] * (len(hist[0]['children']) - 1))

    def test_guard5_removed_parent_in_vocab(self):
        # Guard 5 removed: if P is already in current_vocab the merge still fires
        # for molecules that have its children (molecules with P already are untouched)
        h = self._make_hier()
        # Half molecules have parent already; half have children
        mf = ([['[*]c1ccc(C)cc1']] * 5 +      # P already in mol as leaf
              [['[*]c1ccccc1', '[*]C']] * 10)   # children — should be merged
        _, hist = frag.bpe_merge(deepcopy(mf), h, n=15,
                                  min_atoms=3, max_diam=3, sz_max=18, min_abs=5,
                                  max_child_sup=1.0)
        # Merge should still fire on the 10 molecules that have children
        if hist:
            self.assertGreaterEqual(hist[0]['n_merged'], 5)

    def test_merge_correct_parent(self):
        h = self._make_hier()
        mf = [['[*]c1ccccc1', '[*]C']] * 10
        _, hist = frag.bpe_merge(deepcopy(mf), h, n=10,
                                  min_atoms=3, max_diam=3, sz_max=18, min_abs=5,
                                  max_child_sup=1.0)
        self.assertEqual(hist[0]['parent'], '[*]c1ccc(C)cc1')

    def test_no_merge_below_min_abs(self):
        h = self._make_hier()
        mf = [['[*]c1ccccc1', '[*]C']] * 3
        _, hist = frag.bpe_merge(deepcopy(mf), h, n=3,
                                  min_atoms=3, max_diam=3, sz_max=18, min_abs=5,
                                  max_child_sup=1.0)
        self.assertEqual(len(hist), 0)

    def test_vocab_contains_parent_after_merge(self):
        h = self._make_hier()
        mf = [['[*]c1ccccc1', '[*]C']] * 10
        mf_out, hist = frag.bpe_merge(deepcopy(mf), h, n=10,
                                       min_atoms=3, max_diam=3, sz_max=18, min_abs=5,
                                       max_child_sup=1.0)
        if hist:
            parent = hist[0]['parent']
            self.assertTrue(any(parent in f for f in mf_out))


# ─── vocab_stats ────────────────────────────────────────────────────────────

class TestVocabStats(unittest.TestCase):
    def test_vocab_size(self):
        mf = [['[*]c1ccccc1', '[*]C'], ['[*]c1ccccc1', '[*]N']]
        s = frag.vocab_stats(mf, n=2, label='t')
        self.assertEqual(s['vocab_size'], 3)

    def test_above_1pct(self):
        mf = [['[*]c1ccccc1']] * 10 + [['[*]CC']]
        s = frag.vocab_stats(mf, n=11, label='t')
        self.assertGreaterEqual(s['above_1pct'], 1)

    def test_single_frag_mols(self):
        mf = [['[*]c1ccccc1'], ['[*]c1ccccc1', '[*]C']]
        s = frag.vocab_stats(mf, n=2, label='t')
        self.assertEqual(s['single_frag_mols'], 1)


# ═════════════════════════════════════════════════════════════════════════════
# motif_label_pipeline
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildCatalogPipe(unittest.TestCase):
    def test_returns_filter_catalog(self):
        from rdkit.Chem.FilterCatalog import FilterCatalog
        self.assertIsInstance(pipe.build_catalog(), FilterCatalog)

    def test_nitro_is_alert(self):
        cat = pipe.build_catalog()
        self.assertGreater(len(list(cat.GetMatches(mol('O=[N+]([O-])c1ccccc1')))), 0)


class TestPipeAtomCount(unittest.TestCase):
    def test_benzene_with_wildcard(self):
        # atom_count excludes wildcards — matches molfragbpe5.atom_count
        self.assertEqual(pipe.atom_count('[*]c1ccccc1'), 6)
    def test_nitro_with_wildcard(self):
        self.assertEqual(pipe.atom_count('[*][N+](=O)[O-]'), 3)
    def test_trivial_linker_excluded(self):
        # [*]O[*] has 1 heavy atom — must not pass the >= 2 filter
        self.assertEqual(pipe.atom_count('[*]O[*]'), 1)
        self.assertEqual(pipe.atom_count('[*]N[*]'), 1)
        self.assertEqual(pipe.atom_count('[*]C[*]'), 1)


class TestJaccard(unittest.TestCase):
    def test_identical(self):
        a = np.array([1,1,0,1], dtype=bool)
        self.assertEqual(pipe.jaccard(a, a), 1.0)
    def test_disjoint(self):
        a = np.array([1,1,0,0], dtype=bool)
        b = np.array([0,0,1,1], dtype=bool)
        self.assertEqual(pipe.jaccard(a, b), 0.0)
    def test_partial(self):
        # intersection=2, union=4 → 0.5
        a = np.array([1,1,0,0,1], dtype=bool)
        b = np.array([1,0,1,0,1], dtype=bool)
        self.assertEqual(pipe.jaccard(a, b), 0.5)
    def test_empty(self):
        a = np.zeros(3, dtype=bool)
        self.assertEqual(pipe.jaccard(a, a), 0.0)


class TestLabelDist(unittest.TestCase):
    def test_basic(self):
        mask = np.array([1,1,0,1,0], dtype=bool)
        d = pipe.label_dist(mask, n=5)
        self.assertEqual(d['n1'], 3)
        self.assertEqual(d['pct1'], 60.0)
    def test_all_covered(self):
        d = pipe.label_dist(np.ones(4, dtype=bool), n=4)
        self.assertEqual(d['n1'], 4)


class TestGetCore(unittest.TestCase):
    def test_benzene(self):
        core = pipe.get_core('[*]c1ccccc1')
        self.assertIsNotNone(core)
        self.assertEqual(core.GetNumAtoms(), 6)
    def test_invalid_returns_none(self):
        self.assertIsNone(pipe.get_core('[INVALID]'))


class TestTooGeneric(unittest.TestCase):
    def test_single_carbon(self):
        core = pipe.get_core('[*]C')
        self.assertIsNotNone(core)
        self.assertTrue(pipe.too_generic(core))
    def test_benzene_not_generic(self):
        core = pipe.get_core('[*]c1ccccc1')
        self.assertIsNotNone(core)
        self.assertFalse(pipe.too_generic(core))


class TestCheckSub(unittest.TestCase):
    def test_benzene_in_tolyl(self):
        self.assertTrue(pipe.check_sub('[*]c1ccccc1', '[*]c1ccc(C)cc1'))
    def test_unrelated(self):
        self.assertFalse(pipe.check_sub('[*]c1ccccc1', '[*][N+](=O)[O-]'))


class TestComputeAlertFamilies(unittest.TestCase):
    def test_nitro_has_alerts(self):
        cat = pipe.build_catalog()
        top = ['[*][N+](=O)[O-]']
        all_cands = [(0, '[*][N+](=O)[O-]', 0.5)]
        top_alerts, _ = pipe.compute_alert_families(top, all_cands, cat)
        self.assertIn('[*][N+](=O)[O-]', top_alerts)
        self.assertGreater(len(top_alerts['[*][N+](=O)[O-]']), 0)


class TestComputeSubsumingFamilies(unittest.TestCase):
    def test_specific_detected(self):
        top = ['[*]c1ccccc1']
        all_cands = [(0,'[*]c1ccccc1',0.9),(1,'[*]c1ccc(C)cc1',0.5)]
        groups = pipe.compute_subsuming_families(top, all_cands)
        if '[*]c1ccccc1' in groups:
            directions = [m['direction'] for m in groups['[*]c1ccccc1']]
            self.assertIn('specific', directions)


class TestCoocProfile(unittest.TestCase):
    def test_jaccard_value(self):
        masks = {'A': np.array([1,1,1,1,1,0,0,0,0,0], dtype=bool),
                 'B': np.array([1,1,0,0,0,0,0,0,0,0], dtype=bool)}
        all_cands = [(0,'A',0.5),(1,'B',0.3)]
        prof, _ = pipe.cooc_profile(['A','B'], all_cands, masks)
        self.assertIn(('A','B'), prof)
        self.assertAlmostEqual(prof[('A','B')]['J'], 2/5, places=2)

    def test_symmetry(self):
        masks = {'A': np.array([1,1,1,0,0,0], dtype=bool),
                 'B': np.array([1,1,0,1,0,0], dtype=bool)}
        all_cands = [(0,'A',0.5),(1,'B',0.5)]
        prof, _ = pipe.cooc_profile(['A','B'], all_cands, masks)
        self.assertEqual(prof[('A','B')]['J'], prof[('B','A')]['J'])


class TestBuildClauses(unittest.TestCase):
    def _setup(self):
        masks = {'A': np.array([1]*5+[0]*5, dtype=bool),
                 'B': np.array([0]*3+[1]*4+[0]*3, dtype=bool),
                 'C': np.array([1]*3+[0]*7, dtype=bool)}
        from collections import defaultdict
        return ['A','B','C'], masks, defaultdict(dict), 10

    def test_singleton_clauses(self):
        top, masks, prof, n = self._setup()
        clauses = pipe.build_clauses(top, masks, prof, n)
        k1 = [c for c in clauses if c['k'] == 1]
        self.assertEqual(len(k1), 3)

    def test_clause_n1_correct(self):
        top, masks, prof, n = self._setup()
        clauses = pipe.build_clauses(top, masks, prof, n)
        a_clause = next(c for c in clauses if c['motifs'] == ['A'])
        self.assertEqual(a_clause['n1'], 5)


class TestClauseMask(unittest.TestCase):
    def test_two_motif_and(self):
        masks = {'A': np.array([1,1,0,0], dtype=bool),
                 'B': np.array([1,0,1,0], dtype=bool)}
        np.testing.assert_array_equal(
            pipe.clause_mask({'motifs':['A','B']}, masks),
            [True,False,False,False])

    def test_single_passthrough(self):
        masks = {'A': np.array([1,0,1,1], dtype=bool)}
        np.testing.assert_array_equal(
            pipe.clause_mask({'motifs':['A']}, masks), masks['A'])


class TestBuildProxyLookup(unittest.TestCase):
    def test_high_conditional(self):
        prof = {('A','B'): {'J':0.5,'p_b_given_a':0.9,'p_a_given_b':0.5}}
        lup = pipe.build_proxy_lookup(prof)
        self.assertIn('A', lup)

    def test_low_conditional_absent(self):
        prof = {('A','B'): {'J':0.1,'p_b_given_a':0.3,'p_a_given_b':0.1}}
        lup = pipe.build_proxy_lookup(prof)
        self.assertNotIn('A', lup)


class TestBuildDnfRules(unittest.TestCase):
    def _setup(self, n=20):
        masks = {'A': np.array([1]*10+[0]*10, dtype=bool),
                 'B': np.array([0]*5+[1]*10+[0]*5, dtype=bool)}
        from collections import defaultdict
        clauses = pipe.build_clauses(['A','B'], masks, defaultdict(dict), n)
        return clauses, masks, defaultdict(dict), n

    def test_rules_produced(self):
        c, m, p, n = self._setup()
        rules = pipe.build_dnf_rules(c[:10], m, p, n)
        self.assertGreater(len(rules), 0)

    def test_required_keys(self):
        c, m, p, n = self._setup()
        rules = pipe.build_dnf_rules(c[:10], m, p, n)
        for k in ('n_clauses','clauses','ambiguity','n1','n0','pct1','pct0'):
            self.assertIn(k, rules[0])

    def test_sorted_by_tier_then_coverage(self):
        # build_dnf_rules emits tiers by n_clauses descending (4→1) and, WITHIN
        # each tier, sorts by n1 descending. It does NOT globally sort by pct1
        # (a 2-clause rule may out-cover a 3-clause one), so assert exactly the
        # ordering the code guarantees: n_clauses non-increasing, and n1
        # non-increasing within each equal-n_clauses run.
        c, m, p, n = self._setup()
        rules = pipe.build_dnf_rules(c[:10], m, p, n)
        ncs = [r['n_clauses'] for r in rules]
        self.assertEqual(ncs, sorted(ncs, reverse=True))
        for tier in set(ncs):
            n1s = [r['n1'] for r in rules if r['n_clauses'] == tier]
            self.assertEqual(n1s, sorted(n1s, reverse=True),
                             f"n1 not descending within n_clauses={tier} tier")


class TestScoreDnfRules(unittest.TestCase):
    """Balance-aware re-ranking: balance × separation × (1-spurious)."""

    def _rule(self, motifs, mask, n):
        import numpy as np
        d = pipe.label_dist(mask, n)
        d.update({'n_clauses': len(motifs),
                  'clauses': [{'motifs': [mm], 'k': 1, 'pair_stats': {}}
                              for mm in motifs],
                  'ambiguity': 0.0})
        return d

    def test_balanced_rule_ranks_first(self):
        import numpy as np
        n = 100
        mA = np.zeros(n, bool); mA[:90] = True   # 90% coverage -> imbalanced
        mB = np.zeros(n, bool); mB[:52] = True   # 52% coverage -> balanced
        masks = {'A': mA, 'B': mB}
        rules = [self._rule(['A'], mA, n), self._rule(['B'], mB, n)]
        tv = [[k for k in masks if masks[k][i]] for i in range(n)]
        ranked = pipe.score_dnf_rules(rules, masks, tv, {}, {}, n)
        self.assertEqual(ranked[0]['clauses'][0]['motifs'], ['B'])
        self.assertGreater(ranked[0]['balance'], ranked[1]['balance'])

    def test_score_components_present(self):
        import numpy as np
        n = 50
        mA = np.zeros(n, bool); mA[:25] = True
        rules = [self._rule(['A'], mA, n)]
        tv = [(['A'] if mA[i] else []) for i in range(n)]
        ranked = pipe.score_dnf_rules(rules, {'A': mA}, tv, {}, {}, n)
        for k in ('balance', 'separation', 'spurious', 'score', 'rule_pct_match'):
            self.assertIn(k, ranked[0])

    def test_spurious_penalizes_cooccurring_motifs(self):
        import numpy as np
        n = 100
        mA = np.zeros(n, bool); mA[:50] = True
        mB = np.zeros(n, bool); mB[:48] = True   # near-identical -> high Jaccard
        inter = int((mA & mB).sum()); u = int((mA | mB).sum()); J = inter / u
        prof = {('A', 'B'): {'J': J}, ('B', 'A'): {'J': J}}
        masks = {'A': mA, 'B': mB}
        rules = [self._rule(['A', 'B'], mA | mB, n)]
        tv = [[k for k in masks if masks[k][i]] for i in range(n)]
        ranked = pipe.score_dnf_rules(rules, masks, tv, prof, {}, n)
        self.assertGreater(ranked[0]['spurious'], 0.0)


# ═════════════════════════════════════════════════════════════════════════════
# generate_vocab_rules — tracked fragmentation
# ═════════════════════════════════════════════════════════════════════════════

class TestFobTracked(unittest.TestCase):
    def _make_mapped(self, m):
        rw = Chem.RWMol(m)
        for a in rw.GetAtoms(): a.SetAtomMapNum(a.GetIdx()+1)
        return rw.GetMol()

    def test_nitrobenzene_coverage(self):
        m = mol('O=[N+]([O-])c1ccccc1')
        mapped = self._make_mapped(m)
        idx = gvr._bond_indices_for(m, frag.cut_rbrics)
        pieces = gvr._fob_tracked(mapped, idx)
        all_orig = {a for _,s,_ in pieces for a in s}
        self.assertEqual(all_orig, set(range(m.GetNumAtoms())))

    def test_no_overlap(self):
        m = mol('CC(=O)Nc1ccc(O)cc1')
        mapped = self._make_mapped(m)
        idx = gvr._bond_indices_for(m, frag.cut_rbrics)
        pieces = gvr._fob_tracked(mapped, idx)
        seen = set()
        for _,s,_ in pieces:
            self.assertTrue(s.isdisjoint(seen))
            seen |= s

    def test_returns_three_tuple(self):
        m = mol('Cc1ccccc1')
        mapped = self._make_mapped(m)
        idx = gvr._bond_indices_for(m, frag.cut_rbrics)
        if idx:
            pieces = gvr._fob_tracked(mapped, idx)
            self.assertEqual(len(pieces[0]), 3)  # (smarts, orig_set, frag_map)

    def test_dummy_only_fragment_dropped_silently(self):
        # _fob_tracked returns all real-atom fragments even if only 1 results.
        # Dummy-only fragments (no orig_set) are excluded; caller checks count.
        m = mol('O=[N+]([O-])c1ccccc1')
        mapped = self._make_mapped(m)
        idx = gvr._bond_indices_for(m, frag.cut_rbrics)
        pieces = gvr._fob_tracked(mapped, idx)
        # All returned pieces must have non-empty orig_set
        for _, orig_set, _ in pieces:
            self.assertGreater(len(orig_set), 0)
        # All original atoms must be accounted for
        all_orig = {a for _,s,_ in pieces for a in s}
        self.assertEqual(all_orig, set(range(m.GetNumAtoms())))


class TestStampMol(unittest.TestCase):
    def test_map_numbers_set(self):
        idx_map = {0:3, 1:7, 2:5}
        m = gvr._stamp_mol('[*]CCC', idx_map)
        self.assertIsNotNone(m)
        for a in m.GetAtoms():
            if a.GetAtomicNum() not in (0,1):
                fi = a.GetIdx()
                if fi in idx_map:
                    self.assertEqual(a.GetAtomMapNum(), idx_map[fi]+1)

    def test_invalid_returns_none(self):
        self.assertIsNone(gvr._stamp_mol('[INVALID]', {}))


class TestBondIndicesFor(unittest.TestCase):
    @unittest.skipUnless(frag.RBRICS_OK, "rBRICS_public.py not found")
    def test_rbrics_nitrobenzene(self):
        idx = gvr._bond_indices_for(mol('O=[N+]([O-])c1ccccc1'), frag.cut_rbrics)
        self.assertGreater(len(idx), 0)
    def test_brics_paracetamol(self):
        idx = gvr._bond_indices_for(mol('CC(=O)Nc1ccc(O)cc1'), frag.cut_brics)
        self.assertGreater(len(idx), 0)
    def test_recap_amide(self):
        idx = gvr._bond_indices_for(mol('CC(=O)NC'), frag.cut_recap)
        self.assertGreater(len(idx), 0)
    def test_murcko_toluene(self):
        idx = gvr._bond_indices_for(mol('Cc1ccccc1'), frag.cut_murcko)
        self.assertGreater(len(idx), 0)
    def test_ring_chain_pure_ring_empty(self):
        self.assertEqual(gvr._bond_indices_for(mol('c1ccccc1'), frag.cut_ring_chain), [])


class TestFragmentMoleculeTracked(unittest.TestCase):
    """Tests the single-pass legacy engine fragment_molecule_tracked.

    Legacy is ONE chemistry pass: 'brics', 'rbrics_only', or 'rbrics'
    (rBRICS + reBRICS sub-pass). The recursive cascade and structural fallback
    are disabled; 'all' is handled by the v4 adapter, not this function.
    """

    def test_full_coverage_rbrics_only(self):
        m = mol('O=[N+]([O-])c1ccccc1')
        pieces = gvr.fragment_molecule_tracked(m, 'O=[N+]([O-])c1ccccc1',
                                                use_fallback=False,
                                                method='rbrics_only')
        covered = {a for _,s in pieces for a in s}
        self.assertEqual(covered, set(range(m.GetNumAtoms())))

    def test_full_coverage_rbrics(self):
        m = mol('CC(=O)Nc1ccc(O)cc1')
        smi = 'CC(=O)Nc1ccc(O)cc1'
        pieces = gvr.fragment_molecule_tracked(m, smi,
                                                use_fallback=False, method='rbrics')
        covered = {a for _,s in pieces for a in s}
        self.assertEqual(covered, set(range(m.GetNumAtoms())))

    def test_full_coverage_brics(self):
        m = mol('CC(=O)Nc1ccc(O)cc1')
        smi = 'CC(=O)Nc1ccc(O)cc1'
        pieces = gvr.fragment_molecule_tracked(m, smi,
                                                use_fallback=False, method='brics')
        covered = {a for _,s in pieces for a in s}
        self.assertEqual(covered, set(range(m.GetNumAtoms())))

    def test_no_overlap(self):
        m = mol('CC(=O)Nc1ccc(O)cc1')
        pieces = gvr.fragment_molecule_tracked(m, 'CC(=O)Nc1ccc(O)cc1',
                                                use_fallback=False, method='rbrics')
        seen = set()
        for _, atoms in pieces:
            self.assertTrue(atoms.isdisjoint(seen))
            seen |= atoms

    def test_atom_index_alignment(self):
        # Non-canonical SMILES — atom order must match Chem.MolFromSmiles(smi)
        smi = 'c1ccc(N)cc1'
        m = Chem.MolFromSmiles(smi)
        pieces = gvr.fragment_molecule_tracked(m, smi, use_fallback=False,
                                                method='rbrics')
        covered = {a for _,s in pieces for a in s}
        self.assertEqual(covered, set(range(m.GetNumAtoms())))

    def test_brics_leaves_nitro_intact(self):
        # BRICS method should leave nitrobenzene as one fragment
        smi = 'O=[N+]([O-])c1ccccc1'
        m = Chem.MolFromSmiles(smi)
        pieces = gvr.fragment_molecule_tracked(m, smi,
                                                use_fallback=False, method='brics')
        # No cut = single piece covering all atoms
        self.assertEqual(len(pieces), 1)

    def test_default_method_is_rbrics(self):
        smi = 'Cc1ccccc1'
        m = Chem.MolFromSmiles(smi)
        # Should not raise — uses method='rbrics' by default (legacy single pass)
        pieces = gvr.fragment_molecule_tracked(m, smi, use_fallback=False)
        self.assertGreater(len(pieces), 0)
        covered = {a for _, s in pieces for a in s}
        self.assertEqual(covered, set(range(m.GetNumAtoms())))

    def test_rbrics_tracked_matches_molfragbpe5_cut_rbrics(self):
        """Tracked rbrics path matches molfragbpe5.cut_rbrics fragment counts."""
        if not frag.RBRICS_OK:
            self.skipTest("rBRICS not installed")
        smi, _, n_full = _pick_rebrics_diff_case()
        if smi is None:
            self.skipTest("no candidate where reBRICS adds cuts")
        m = mol(smi)
        frag._CACHE.clear()
        p_full = gvr.fragment_molecule_tracked(m, smi, False, 'rbrics')
        self.assertEqual(len(p_full), n_full,
                         f"{smi}: tracked rbrics != cut_rbrics reference")

    def test_rbrics_rebrics_pass_differs_from_rbrics_only(self):
        """reBRICS post-pass adds cuts vs rbrics_only (molfragbpe5 reference)."""
        if not frag.RBRICS_OK:
            self.skipTest("rBRICS not installed")
        smi, n_only, n_full = _pick_rebrics_diff_case()
        if smi is None:
            self.skipTest("no candidate where reBRICS adds cuts")
        m = mol(smi)
        frag._CACHE.clear()
        p_only = gvr.fragment_molecule_tracked(m, smi, False, 'rbrics_only')
        frag._CACHE.clear()
        p_old = gvr.fragment_molecule_tracked(m, smi, False, 'rbrics_old')
        frag._CACHE.clear()
        p_full = gvr.fragment_molecule_tracked(m, smi, False, 'rbrics')
        self.assertEqual(len(p_only), n_only, f"{smi}: tracked rbrics_only mismatch")
        self.assertEqual(len(p_old), n_only, f"{smi}: rbrics_old should match rbrics_only")
        self.assertEqual(len(p_full), n_full, f"{smi}: tracked rbrics mismatch")
        self.assertGreater(len(p_full), len(p_only),
                           f"{smi}: reBRICS should add cuts beyond rbrics_only")


# ─── build_vocab ────────────────────────────────────────────────────────────

class TestBuildVocab(unittest.TestCase):
    def _frags(self):
        return [
            [('[*]c1ccccc1',{0,1,2,3,4,5}), ('[*]C',{6})],
            [('[*]c1ccccc1',{0,1,2,3,4,5}), ('[*]N',{6})],
            [('[*]CC',{0,1,2})],
        ]

    def test_all_kept_no_filter(self):
        ml, fid, _ = gvr.build_vocab(self._frags(), np.array([1,1,0]))
        for s in ['[*]c1ccccc1','[*]C','[*]N','[*]CC']:
            self.assertIn(s, ml)

    def test_trivial_still_included(self):
        mf = [[('[*]C[*]',{0})]]
        ml, _, _ = gvr.build_vocab(mf, np.array([0]))
        self.assertIn('[*]C[*]', ml)

    def test_sorted_by_count(self):
        ml, _, _ = gvr.build_vocab(self._frags(), np.array([1,1,0]))
        self.assertEqual(ml[0], '[*]c1ccccc1')

    def test_above_min_sup_flag(self):
        _, _, stats = gvr.build_vocab(self._frags(), np.array([1,1,0]),
                                       min_sup_for_rules=0.5)
        self.assertTrue(stats['[*]c1ccccc1']['above_min_sup'])
        self.assertFalse(stats['[*]C']['above_min_sup'])


# ─── build_lookup ───────────────────────────────────────────────────────────

class TestBuildLookup(unittest.TestCase):
    def _setup(self):
        mf = [
            [('[*]c1ccccc1',{0,1,2,3,4,5}), ('[*]C',{6})],
            [('[*]CC',{0,1,2})],
        ]
        ml, fid, _ = gvr.build_vocab(mf, np.array([1,0]))
        return ['s1','s2'], mf, fid

    def test_all_nodes_mapped(self):
        smis, mf, fid = self._setup()
        lup = gvr.build_lookup(smis, mf, fid)
        for smi, nm in lup.items():
            for ni,(smarts,mid) in nm.items():
                self.assertIsNotNone(mid, f"node {ni} in {smi} has None mid")

    def test_correct_atom_assignment(self):
        smis, mf, fid = self._setup()
        lup = gvr.build_lookup(smis, mf, fid)
        self.assertIn(6, lup['s1'])
        self.assertEqual(lup['s1'][6][0], '[*]C')

    def test_motif_ids_are_integers(self):
        smis, mf, fid = self._setup()
        lup = gvr.build_lookup(smis, mf, fid)
        for _, nm in lup.items():
            for _, (_, mid) in nm.items():
                self.assertIsInstance(mid, int)


# ─── build_matrix ───────────────────────────────────────────────────────────

class TestBuildMatrix(unittest.TestCase):
    def _setup(self):
        mf = [
            [('[*]c1ccccc1',{0,1,2,3,4,5}), ('[*]C',{6})],
            [('[*]c1ccccc1',{0,1,2,3,4,5})],
            [('[*]CC',{0,1,2})],
        ]
        ml, fid, _ = gvr.build_vocab(mf, np.array([1,1,0]))
        return mf, fid

    def test_shape(self):
        mf, fid = self._setup()
        X = gvr.build_matrix(mf, fid, 3)
        self.assertEqual(X.shape, (3, len(fid)))

    def test_row_entries(self):
        mf, fid = self._setup()
        X = gvr.build_matrix(mf, fid, 3).toarray()
        self.assertEqual(X[0, fid['[*]c1ccccc1']], 1)
        self.assertEqual(X[0, fid['[*]C']], 1)

    def test_binary(self):
        mf, fid = self._setup()
        X = gvr.build_matrix(mf, fid, 3).toarray()
        self.assertTrue(np.all((X == 0) | (X == 1)))

    def test_dtype_uint8(self):
        mf, fid = self._setup()
        X = gvr.build_matrix(mf, fid, 3)
        self.assertEqual(X.dtype, np.uint8)


# ─── Integration ────────────────────────────────────────────────────────────

class TestIntegration(unittest.TestCase):
    """Full pipeline: fragment → vocab → lookup → matrix, for each method."""

    SMILES = ['O=[N+]([O-])c1ccccc1', 'CC(=O)Nc1ccc(O)cc1', 'c1ccccc1']
    LABELS = np.array([1, 0, 1])

    def _run(self, method):
        frag._CACHE.clear()
        hier = frag.Hierarchy()
        mf_tracked, mf_plain = [], []
        for smi in self.SMILES:
            m = mol(smi)
            if m is None: continue
            mf_tracked.append(gvr.fragment_molecule_tracked(
                m, smi, use_fallback=True, method=method))
            mf_plain.append(frag.fragment_molecule(
                m, hier, use_fallback=True, method=method))

        n = len(self.LABELS)
        ml, fid, stats = gvr.build_vocab(mf_tracked, self.LABELS)
        lup = gvr.build_lookup(self.SMILES, mf_tracked, fid)
        X   = gvr.build_matrix(mf_tracked, fid, n)

        for smi in self.SMILES:
            m = mol(smi)
            if m is None: continue
            self.assertEqual(len(lup[smi]), m.GetNumAtoms(),
                             f"[{method}] {smi}: {len(lup[smi])} mapped != {m.GetNumAtoms()}")
        self.assertEqual(X.shape[0], n)
        self.assertEqual(X.shape[1], len(ml))

    def test_method_rbrics_only(self):
        self._run('rbrics_only')

    def test_method_rbrics(self):
        self._run('rbrics')

    def test_method_brics(self):
        self._run('brics')

    def test_nitro_split_only_with_rbrics(self):
        """rBRICS-family methods split nitrobenzene; BRICS alone does not."""
        smi = 'O=[N+]([O-])c1ccccc1'
        m = mol(smi)

        # rBRICS-dependent assertions — skip if rBRICS not installed
        if frag.RBRICS_OK:
            for method in ('rbrics', 'rbrics_only', 'rbrics_old'):
                frag._CACHE.clear()
                pieces = gvr.fragment_molecule_tracked(m, smi,
                                                        use_fallback=False,
                                                        method=method)
                self.assertGreater(len(pieces), 1,
                                   f"Expected split with method={method}")

        # BRICS must never split Ar-NO2 — always testable
        frag._CACHE.clear()
        pieces_b = gvr.fragment_molecule_tracked(m, smi,
                                                  use_fallback=False,
                                                  method='brics')
        self.assertEqual(len(pieces_b), 1,
                         "BRICS should NOT split Ar-NO2")


# ═════════════════════════════════════════════════════════════════════════════
# chemfrag_v4_adapter — THE production fragmentation engine (method='all')
# ═════════════════════════════════════════════════════════════════════════════

class TestV4Adapter(unittest.TestCase):
    """Contract tests for the v4 cascade + MDL-merge tokenizer used in production
    (generate_vocab_rules.run_dataset routes method='all' through here).

    The atom-index invariant (§2) requires that fragment_tracked_v4 partition a
    molecule's atoms EXACTLY — full coverage, no overlap — in the raw
    Chem.MolFromSmiles(smiles) index order, and that it FAIL LOUD rather than
    emit a silently-corrected (wrong) atom→motif lookup.
    """

    SMILES = ['O=[N+]([O-])c1ccccc1', 'CC(=O)Nc1ccc(O)cc1', 'c1ccccc1',
              'CCO', 'CC(=O)Oc1ccccc1C(=O)O']

    def test_exact_coverage_and_disjoint(self):
        # Both merge modes must conserve atoms exactly on every molecule.
        for use_merge in (False, True):
            ruleset, index = v4.learn_corpus_rulebook(self.SMILES, use_merge=use_merge)
            for smi in self.SMILES:
                m = mol(smi)
                toks = v4.fragment_tracked_v4(smi, ruleset, index)
                covered = set()
                for _, atoms in toks:
                    self.assertTrue(atoms.isdisjoint(covered),
                                    f"overlap in {smi} (merge={use_merge})")
                    covered |= atoms
                self.assertEqual(covered, set(range(m.GetNumAtoms())),
                                 f"coverage gap in {smi} (merge={use_merge})")

    def test_raw_atom_index_order(self):
        # Non-canonical SMILES: atomsets index into Chem.MolFromSmiles(smi)
        # directly, so the union must be exactly 0..n-1.
        smi = 'c1ccc(N)cc1'
        m = mol(smi)
        ruleset, index = v4.learn_corpus_rulebook([smi], use_merge=True)
        toks = v4.fragment_tracked_v4(smi, ruleset, index)
        covered = {a for _, atoms in toks for a in atoms}
        self.assertEqual(covered, set(range(m.GetNumAtoms())))

    def test_tokenize_without_index(self):
        # index=None forces on-the-fly fragmentation; contract must still hold.
        smi = 'CC(=O)Nc1ccc(O)cc1'
        ruleset, _ = v4.learn_corpus_rulebook([smi], use_merge=True)
        toks = v4.fragment_tracked_v4(smi, ruleset, index=None)
        covered = {a for _, atoms in toks for a in atoms}
        self.assertEqual(covered, set(range(mol(smi).GetNumAtoms())))

    def test_keys_are_strings(self):
        ruleset, index = v4.learn_corpus_rulebook(['CC(=O)Nc1ccc(O)cc1'])
        toks = v4.fragment_tracked_v4('CC(=O)Nc1ccc(O)cc1', ruleset, index)
        for key, _ in toks:
            self.assertIsInstance(key, str)

    def test_invalid_smiles_marked(self):
        ruleset, index = v4.learn_corpus_rulebook(['not_a_smiles'])
        toks = v4.fragment_tracked_v4('not_a_smiles', ruleset, index)
        self.assertEqual(toks, [('[INVALID]', {0})])

    def test_fail_loud_on_missing_atom(self):
        # A corrupt tokenizer that drops atoms must raise, never silently patch
        # them into token 0 (which would feed a wrong atom→motif lookup).
        smi = 'CCO'
        ruleset, index = v4.learn_corpus_rulebook([smi])
        orig = v4.M.apply_rulebook
        try:
            v4.M.apply_rulebook = lambda m, tree, rs: [('[*]C', {0})]  # only atom 0
            with self.assertRaises(ValueError):
                v4.fragment_tracked_v4(smi, ruleset, index)
        finally:
            v4.M.apply_rulebook = orig

    def test_fail_loud_on_overlap(self):
        # Full coverage but a duplicated atom across tokens must also raise.
        smi = 'CCO'
        n = mol(smi).GetNumAtoms()
        ruleset, index = v4.learn_corpus_rulebook([smi])
        orig = v4.M.apply_rulebook
        try:
            v4.M.apply_rulebook = lambda m, tree, rs: [('a', set(range(n))),
                                                       ('b', {0})]
            with self.assertRaises(ValueError):
                v4.fragment_tracked_v4(smi, ruleset, index)
        finally:
            v4.M.apply_rulebook = orig


# ─── v4 integration through build_vocab/lookup/matrix (method='all' path) ────

class TestV4Integration(unittest.TestCase):
    """End-to-end on the PRODUCTION path: v4 tokens → vocab → lookup → matrix.
    Mirrors TestIntegration but for method='all' (the default), which
    TestIntegration never exercises."""

    SMILES = ['O=[N+]([O-])c1ccccc1', 'CC(=O)Nc1ccc(O)cc1', 'c1ccccc1']
    LABELS = np.array([1, 0, 1])

    def test_full_pipeline_all(self):
        mf_tracked = v4.fragment_corpus_v4(self.SMILES, use_merge=True)
        n = len(self.LABELS)
        ml, fid, stats = gvr.build_vocab(mf_tracked, self.LABELS)
        lup = gvr.build_lookup(self.SMILES, mf_tracked, fid)
        X   = gvr.build_matrix(mf_tracked, fid, n)
        for smi in self.SMILES:
            self.assertEqual(len(lup[smi]), mol(smi).GetNumAtoms(),
                             f"[all] {smi}: {len(lup[smi])} mapped != "
                             f"{mol(smi).GetNumAtoms()}")
        self.assertEqual(X.shape, (n, len(ml)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
