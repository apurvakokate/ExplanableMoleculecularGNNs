"""Rule Set 1 — harvester unit tests.

Build a tiny fake summary.json tree and assert: 1:1 completeness, provenance read
(vs mtime fallback), idempotent + versioned dated CSV, count reconciliation, and the
paper-scope filters.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.harvest_results import (
    harvest, write_dated, scope_synthetic_labelling, scope_mose)


def _write(root, rel, obj):
    p = Path(root) / rel
    p.mkdir(parents=True, exist_ok=True)          # create the leaf dir that holds summary.json
    json.dump(obj, open(p / 'summary.json', 'w'))
    return p / 'summary.json'


class Harvester(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        # one with producer provenance, one legacy (none), one different model_type/variant
        _write(self.root, 'a', {'dataset': 'BBBP', 'model_type': 'VanillaGNN',
                                 'vocab_variant': 'rbrics', 'gt_tier': 'dnf_k2_r1',
                                 'git_sha': 'abc123', 'run_timestamp': '2026-01-01T00:00:00',
                                 'config_hash': 'deadbeef',
                                 'gnnexplainer_mean_instance_gt_roc_node_auc_mean': 0.7})
        _write(self.root, 'b', {'dataset': 'BBBP', 'model_type': 'MOSE',
                                 'vocab_variant': 'fg_first', 'gt_tier': 'dnf_k3_r1'})   # legacy
        _write(self.root, 'c', {'dataset': 'Mutagenicity', 'model_type': 'MotifSAT',
                                 'vocab_variant': 'rbrics_filter', 'gt_tier': 'none'})

    def test_completeness_and_provenance(self):
        df, found, skipped = harvest(self.root)
        self.assertEqual(found, 3)
        self.assertEqual(len(df), 3)
        self.assertEqual(skipped, 0)
        row_a = df[df['git_sha'] == 'abc123']
        self.assertEqual(len(row_a), 1, 'producer provenance not read')
        # legacy rows fall back to a non-"abc123" sha (mtime-stamped 'unknown')
        self.assertTrue((df['git_sha'] == 'unknown').sum() >= 2)

    def test_dated_csv_idempotent_versioned(self):
        df, *_ = harvest(self.root)
        with tempfile.TemporaryDirectory() as sd:
            p1 = write_dated(df.copy(), sd)
            n1 = len(pd.read_csv(p1, dtype=str))
            p2 = write_dated(df.copy(), sd)                 # re-harvest → no dup
            n2 = len(pd.read_csv(p2, dtype=str))
            self.assertEqual(p1, p2)
            self.assertEqual(n1, n2, 're-harvest duplicated rows (not idempotent)')
            self.assertEqual(n1, 3)

    def test_scope_filters(self):
        df, *_ = harvest(self.root)
        synth = scope_synthetic_labelling(df)               # excludes MOSE/MotifSAT
        self.assertEqual(set(synth['model_type']), {'VanillaGNN'})
        mose = scope_mose(df)                               # rbrics only (rbrics + rbrics_filter)
        self.assertEqual(set(mose['vocab_variant']), {'rbrics', 'rbrics_filter'})

    def test_reconciliation_holds(self):
        df, found, skipped = harvest(self.root)
        self.assertEqual(found, len(df) + skipped)


if __name__ == '__main__':
    unittest.main(verbosity=2)
