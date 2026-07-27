"""Rule Set 1 — CLI / end-to-end smoke.

Runs the REAL entry points the way HPC invokes them — env vars → argparse → script,
and the shell `_gt_tier_list` function — on a tiny synthetic fixture, asserting the
outer wiring that direct-method tests bypass:

  * RULE_ENGINE=dnf actually selects the DNF engine (vs silently the tier grader);
  * apply_gt.py --tier accepts a dnf key (no argparse `choices` rejection);
  * _gt_tier_list is variant-scoped (P0) and its fallback grid matches the real
    key scheme at N=1 (P1).

Heavier end-to-end paths that need the full training env (run_vanilla source-GT,
a full shell phase, the SKIP_SELF_EXPLAINING phase list) are covered by the HPC
integration runs (Rule Set 1 scans), not this local suite — stated, not pretended.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

PROJECT = os.path.dirname(os.path.abspath(__file__))
ENVBASE = {**os.environ, 'PYTHONPATH': PROJECT, 'WANDB_MODE': 'disabled'}


def _fixture_csv(path):
    import csv
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    cores = ['c1ccccc1', 'c1ccncc1', 'c1ccc2ccccc2c1', 'C1CCCCC1']
    subs = ['', 'C', 'CC', 'O', 'N', 'C(=O)O', 'Cl', 'C(=O)N', 'OC', 'CCN']
    rows, i = [], 0
    for c in cores:
        for s in subs:
            smi = c + s
            if Chem.MolFromSmiles(smi):
                rows.append((smi, i % 2, ['training', 'valid', 'test'][i % 3]))
                i += 1
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['smiles', 'BBBP', 'group']); w.writerows(rows)


def _run_gen(engine, out_vocab, data_root):
    env = dict(ENVBASE)
    if engine:
        env['RULE_ENGINE'] = engine
    return subprocess.run(
        [sys.executable, f'{PROJECT}/MotifBreakdown/generate_vocab_rules.py',
         '--datasets', 'BBBP', '--data_root', data_root, '--out_dir', out_vocab,
         '--method', 'rbrics', '--variant', 'rbrics', '--rule_tiers', '--fold', '0'],
        cwd=PROJECT, env=env, capture_output=True, text=True, timeout=300)


class CliEngineSelection(unittest.TestCase):
    """RULE_ENGINE=dnf must flip phase1 onto the DNF engine (the #1 blocker)."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        _fixture_csv(os.path.join(cls.tmp, 'BBBP_0.csv'))

    def test_dnf_engine_selected_with_env(self):
        r = _run_gen('dnf', os.path.join(self.tmp, 'vdnf'), self.tmp)
        self.assertEqual(r.returncode, 0, f'phase1 (dnf) failed:\n{r.stderr[-2000:]}')
        self.assertIn('[dnf]', r.stdout, 'RULE_ENGINE=dnf did not select the DNF engine')
        # any rule keys produced must be dnf_* (never easy/medium/hard)
        rt = os.path.join(self.tmp, 'vdnf', 'BBBP', 'rbrics', 'rule_tiers.json')
        if os.path.exists(rt):
            for k in json.load(open(rt)):
                self.assertTrue(k.startswith('dnf_'), f'non-dnf key {k!r} under RULE_ENGINE=dnf')

    def test_tier_engine_is_default(self):
        r = _run_gen(None, os.path.join(self.tmp, 'vtiers'), self.tmp)
        self.assertEqual(r.returncode, 0, f'phase1 (default) failed:\n{r.stderr[-2000:]}')
        self.assertNotIn('[dnf]', r.stdout,
                         'default engine unexpectedly ran DNF (env flip not isolating)')


class CliApplyGtAcceptsDnfTier(unittest.TestCase):
    """apply_gt.py --tier must accept a dnf key — the removed argparse `choices` bug."""
    def test_tier_not_rejected_by_argparse(self):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run(
                [sys.executable, f'{PROJECT}/SharedModules/data/apply_gt.py',
                 '--dataset', 'BBBP', '--vocab_root', d, '--variant', 'rbrics',
                 '--out_dir', d, '--tier', 'dnf_k2_r1', '--fold', '0'],
                cwd=PROJECT, env=ENVBASE, capture_output=True, text=True, timeout=120)
        # It will fail later (no vocab), but must NOT fail AT argparse on the tier value.
        self.assertNotIn('invalid choice', r.stderr,
                         f'--tier rejected a dnf key at argparse:\n{r.stderr[-800:]}')


class CliGtTierListShell(unittest.TestCase):
    """_gt_tier_list must be variant-scoped (P0) and match the real key scheme at N=1 (P1)."""

    def _call(self, env_kv, args):
        # sed-extract just the function; run in a clean bash subshell (avoid the dispatch case)
        # export the env for the WHOLE subshell (not just the eval that defines the fn),
        # then define the function and call it.
        script = (
            "FUNC=$(sed -n '/_gt_tier_list()/,/^}/p' run_experiments.sh); "
            f"export {env_kv}; eval \"$FUNC\"; _gt_tier_list {args}")
        r = subprocess.run(['bash', '-c', script], cwd=PROJECT, env=ENVBASE,
                           capture_output=True, text=True, timeout=60)
        return r.stdout.strip()

    def test_p0_variant_scoped(self):
        with tempfile.TemporaryDirectory() as vroot:
            for variant, keys in (('rbrics', ['dnf_k2_r1']), ('fg_first', ['dnf_k2_r1', 'dnf_k3_r1'])):
                d = os.path.join(vroot, 'BBBP', variant); os.makedirs(d)
                json.dump({k: {} for k in keys}, open(os.path.join(d, 'rule_tiers.json'), 'w'))
            env = f"RULE_ENGINE=dnf DATASETS=BBBP VOCAB_ROOT={vroot}"
            self.assertEqual(self._call(env, 'rbrics BBBP'), 'dnf_k2_r1')
            self.assertEqual(self._call(env, 'fg_first BBBP'), 'dnf_k2_r1 dnf_k3_r1')

    def test_p1_fallback_scheme_n1(self):
        env = "RULE_ENGINE=dnf DATASETS=BBBP VOCAB_ROOT=/nonexistent RULE_DNF_N=1"
        self.assertEqual(self._call(env, 'fg_first BBBP'), 'dnf_k2_r1 dnf_k3_r1')


if __name__ == '__main__':
    unittest.main(verbosity=2)
