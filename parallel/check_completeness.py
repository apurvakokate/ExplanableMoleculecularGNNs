#!/usr/bin/env python3
"""check_completeness.py — cadence check: are all target cells & metrics run and present?

Compares the config's EXPECTED cell set (for a dataset) against what's actually done
(from the timing log), reports done/failed/pending per celltype, lists failures, and
spot-checks that key metric columns are populated in results.csv.

Usage: check_completeness.py [--only-dataset BBBP] [--out-root final_v2]
"""
import os, sys, subprocess, argparse, csv, glob
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--only-dataset", default="BBBP")
ap.add_argument("--out-root", default="final_v2")
a = ap.parse_args()
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CFG = a.config or os.path.join(PROJ, "config", "experiment_matrix.yaml")
OUT = os.path.join(PROJ, a.out_root) if not os.path.isabs(a.out_root) else a.out_root

# expected cells from the config
exp_lines = subprocess.run(
    [sys.executable, os.path.join(PROJ, "parallel", "enumerate_from_config.py"),
     CFG, "--only-dataset", a.only_dataset],
    capture_output=True, text=True).stdout.strip().split("\n")
expected = set(); exp_ct = Counter()
for line in exp_lines:
    if not line:
        continue
    ds, variant, bb, fold, tier, ct, dev = line.split("\t")
    expected.add((ds, variant, bb, fold, tier, ct)); exp_ct[ct] += 1

# done / failed from the timing log (rc column is source of truth)
timing = os.path.join(OUT, "logs", "cell_timing.tsv")
# A cell can appear multiple times (retries): it is DONE if ANY attempt had rc==0,
# and FAILED only if NO attempt succeeded. (Fixes the retry double-count where a
# fail-then-succeed cell previously landed in both sets and inflated FAILED.)
_rc_ok = {}
if os.path.exists(timing):
    for line in open(timing):
        p = line.rstrip("\n").split("\t")
        if len(p) < 8:
            continue
        key = tuple(p[:6])          # ds,variant,bb,fold,tier,celltype
        _rc_ok[key] = _rc_ok.get(key, False) or (p[7] == "0")
done   = {k for k, ok in _rc_ok.items() if ok}
failed = {k for k, ok in _rc_ok.items() if not ok}
done_ct = Counter(k[5] for k in done)
fail_ct = Counter(k[5] for k in failed)

pending = expected - done - failed
pend_ct = Counter(k[5] for k in pending)

print(f"=== completeness: {a.only_dataset}  (out={os.path.basename(OUT)}) ===")
print(f"  {'celltype':10s} {'exp':>6} {'done':>6} {'fail':>6} {'pend':>6}")
for ct in ["vtrain", "posthoc", "mose", "gsat", "motifsat"]:
    print(f"  {ct:10s} {exp_ct[ct]:6d} {done_ct[ct]:6d} {fail_ct[ct]:6d} {pend_ct[ct]:6d}")
print(f"  {'TOTAL':10s} {len(expected):6d} {len(done):6d} {len(failed):6d} {len(pending):6d}")
if failed:
    print(f"\n  FAILED ({len(failed)}):")
    for k in sorted(failed)[:25]:
        print("    " + "/".join(k))

# Artifact reconciliation (M3 safety net): 'done' above is rc==0 from the timing log.
# A soft-skipped cell (missing gt_cache / vocab) exits rc==0 yet produces NO artifact,
# so it silently counts as done. Cross-check the on-disk artifact count per celltype
# so those gaps surface instead of reading as complete.
_codes = {k[0] for k in expected}
print("\n  artifact reconciliation (done=rc==0 vs files on disk):")
for _ct, _mdir, _fn in [("vtrain", "vanilla", "best_model.pt"),
                        ("posthoc", "baselines", "summary.json"),
                        ("mose", "mose", "summary.json"),
                        ("gsat", "base_gsat", "summary.json"),
                        ("motifsat", "motifsat", "summary.json")]:
    _files = glob.glob(os.path.join(OUT, _mdir, "**", _fn), recursive=True)
    _n = sum(1 for f in _files if any(("/" + code + "/") in f for code in _codes))
    _gap = done_ct[_ct] - _n
    _flag = "   <-- SOFT-SKIP / missing artifacts" if _gap > 0 else ""
    print(f"    {_ct:10s} done={done_ct[_ct]:5d}  artifacts={_n:5d}  gap={_gap:+d}{_flag}")

# metric-presence spot check on results.csv
rf = sorted(glob.glob(os.path.join(OUT, "tables", "results_*.csv")))
if rf:
    rows = list(csv.DictReader(open(rf[-1])))
    def npresent(col): return sum(1 for r in rows if str(r.get(col, "")).strip() not in ("", "nan", "None"))
    print(f"\n  metric coverage in results.csv ({len(rows)} rows):")
    checks = [
        ("pred AUC", "auc"), ("pred RMSE", "rmse"), ("pred RMSE_orig", "rmse_orig"),
        ("GNNEx GT-ROC", "gnnexplainer_max_gt_roc_node_auc_mean"),
        ("GNNEx global GT-ROC", "gnnexplainer_max_global_gt_roc_node_auc_mean"),
        ("GNNEx pearson_motif", "gnnexplainer_max_pearson_motif"),
        ("GNNEx pearson_instance", "gnnexplainer_max_pearson_instance"),
        ("MAGE GT-ROC", "mage_official_max_gt_roc_node_auc_mean"),
    ]
    for label, col in checks:
        present = "—" if not rows or col not in rows[0] else npresent(col)
        print(f"    {label:24s} {present}")
    print("  (ante-hoc MOSE/MotifSAT/GSAT native metrics appear once those cells land — "
          "node-mask-probe column populates after the probe runs.)")
else:
    print("\n  results.csv not harvested yet — run parallel/harvest_v2.py")
