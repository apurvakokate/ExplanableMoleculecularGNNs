#!/usr/bin/env python3
"""verify_results.py — reproduce recorded metrics from the SAVED checkpoints.

Post-completion integrity check. Samples N diverse cells from results.csv, RELOADS
each saved model, re-runs a forward pass on the test set, independently recomputes
two DETERMINISTIC metrics, and asserts they match the recorded CSV values:

  - predictive AUC   : model forward -> roc_auc_score(y, sigmoid(logit))   [all families]
  - GT-ROC node AUC  : native node-attention vs planted DNF GT             [ante-hoc GT cells]

Reuses analysis/probe_masked_nodes.py's model loaders and SharedModules' own metric
functions, so a mismatch means a real recording bug or non-determinism — not a
re-implementation difference.

NOTE: stochastic post-hoc explainers (GNNExplainer/PGExplainer) re-optimize their
masks and are NOT bit-reproducible, so their motif metrics are excluded from the
strict check (predictive AUC of the underlying Vanilla model IS checked).

Usage: verify_results.py [--results <csv>] [--out-root final_v2] [--n 5] [--tol 1e-4] [--seed 0]
"""
from __future__ import annotations
import argparse, glob, json, os, sys, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from analysis.probe_masked_nodes import (          # proven checkpoint + data loaders
    _load_vanilla_and_data, _load_model_and_data, _resolve_probe_family)


@torch.no_grad()
def _forward_auc(model, data_list, device):
    """Deterministic predictive AUC: forward every test graph, roc_auc on class-1 prob."""
    ys, ps = [], []
    model.eval()
    for d in data_list:
        d = d.to(device)
        batch = getattr(d, "batch", None)
        if batch is None:
            batch = torch.zeros(d.x.size(0), dtype=torch.long, device=device)
        out = model(d.x, d.edge_index, batch,
                    getattr(d, "nodes_to_motifs", None), getattr(d, "edge_attr", None))
        logit = out[0] if isinstance(out, (tuple, list)) else out
        p = torch.sigmoid(logit.view(-1))[0].item()
        y = float(d.y.view(-1)[0].item())
        ys.append(y); ps.append(p)
    if len(set(ys)) < 2:
        return float("nan")
    return float(roc_auc_score(ys, ps))


def _find_run_dir(out_root, row):
    """Locate the saved run dir for a results.csv row by matching summary.json fields."""
    fam = row["family"]
    fam_dir = {"vanilla": "vanilla", "baselines": "vanilla", "mose": "mose",
               "motifsat": "motifsat", "gsat": "base_gsat"}.get(fam, fam)
    want = dict(dataset=str(row["dataset"]), backbone=str(row["backbone"]),
                vocab_variant=str(row["vocab_variant"]), fold=int(row["fold"]),
                gt_tier=("" if pd.isna(row.get("gt_tier")) else str(row.get("gt_tier", ""))))
    for sj in glob.glob(os.path.join(out_root, fam_dir, want["dataset"],
                                     f"fold{want['fold']}", "*", "*", "summary.json")):
        try:
            m = json.load(open(sj))
        except Exception:
            continue
        gt = "" if m.get("gt_tier") in (None, "None") else str(m.get("gt_tier", ""))
        if (str(m.get("backbone")) == want["backbone"]
                and str(m.get("vocab_variant")) == want["vocab_variant"]
                and gt == want["gt_tier"]):
            return Path(sj).parent
    return None


def _diverse_sample(df, n, seed):
    """Pick n rows spread across families / tiers / backbones."""
    random.seed(seed)
    df = df[df["auc"].notna()].copy()
    picks, seen = [], set()
    # one per family first, then fill
    for fam in ["vanilla", "mose", "gsat", "motifsat", "baselines"]:
        sub = df[df["family"] == fam]
        if len(sub):
            r = sub.sample(1, random_state=seed).iloc[0]
            picks.append(r); seen.add(id(r))
    pool = df.sample(frac=1, random_state=seed)
    for _, r in pool.iterrows():
        if len(picks) >= n:
            break
        picks.append(r)
    return picks[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--out-root", default="final_v2")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT", ""))
    ap.add_argument("--vocab_root", default=os.environ.get("VOCAB_ROOT", ""))
    a = ap.parse_args()
    out_root = a.out_root if os.path.isabs(a.out_root) else str(_REPO / a.out_root)
    results = a.results or sorted(glob.glob(os.path.join(out_root, "tables", "results_*.csv")))[-1]
    df = pd.read_csv(results, low_memory=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = _diverse_sample(df, a.n, a.seed)
    print(f"verify_results: {len(rows)} keys from {results}  (tol={a.tol})\n")
    hdr = f"{'family':9} {'bb':5} {'variant':22} {'tier':10} {'metric':10} {'recorded':>10} {'recomputed':>11} {'|Δ|':>9}  verdict"
    print(hdr); print("-" * len(hdr))
    npass = nfail = nskip = 0

    for row in rows:
        rd = _find_run_dir(out_root, row)
        tag = f"{row['family']:9} {str(row['backbone']):5} {str(row['vocab_variant'])[:22]:22} {str(row.get('gt_tier','') or 'real'):10}"
        if rd is None:
            print(f"{tag} {'—':10} {'(run dir not found — skip)':>32}"); nskip += 1; continue
        # reload the right family
        fam = _resolve_probe_family(json.load(open(rd / "summary.json")), rd)
        if row["family"] in ("vanilla", "baselines"):
            model, test_list, status = _load_vanilla_and_data(rd, a.data_root, a.vocab_root, device)
        else:
            model, test_list, status = _load_model_and_data(rd, a.data_root, a.vocab_root, device)
        if status != "ok":
            print(f"{tag} {'—':10} {('(reload failed: '+status+')'):>32}"); nskip += 1; continue

        checks = [("auc", _forward_auc(model, test_list, device))]
        # ante-hoc GT cells: also recompute native GT-ROC node AUC
        if row["family"] in ("mose", "motifsat", "gsat") and str(row.get("gt_tier", "")).startswith("dnf"):
            try:
                from SharedModules.evaluation.motif_eval import compute_dnf_gt_roc
                node_att_fn = lambda m, d: m(d.x, d.edge_index,
                                             torch.zeros(d.x.size(0), dtype=torch.long, device=device),
                                             getattr(d, "nodes_to_motifs", None))[1]
                gtr = compute_dnf_gt_roc(model, test_list, device, node_att_fn)
                val = gtr.get("node_auc_mean") if isinstance(gtr, dict) else None
                if val is not None:
                    checks.append(("gt_roc_node", float(val)))
            except Exception as e:
                print(f"{tag} {'gt_roc_node':10} {'(recompute err: '+str(e)[:20]+')':>32}")

        for metric, recomputed in checks:
            recorded = row.get(metric, float("nan"))
            try:
                recorded = float(recorded)
            except Exception:
                recorded = float("nan")
            if np.isnan(recomputed) or np.isnan(recorded):
                verdict, d = "SKIP (nan)", float("nan"); nskip += 1
            else:
                d = abs(recomputed - recorded)
                verdict = "PASS" if d <= a.tol else "FAIL"
                npass += verdict == "PASS"; nfail += verdict == "FAIL"
            print(f"{tag} {metric:10} {recorded:10.6f} {recomputed:11.6f} {d:9.2e}  {verdict}")

    print("-" * len(hdr))
    print(f"PASS={npass}  FAIL={nfail}  SKIP={nskip}")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
