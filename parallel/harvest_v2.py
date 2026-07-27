#!/usr/bin/env python3
"""harvest_v2.py — refresh final_v2/tables/{results.csv, statistics.csv}. Idempotent;
run after every batch / on a cadence.

  results.csv    : every harvested summary (vanilla/baselines/mose/motifsat/gsat) + a
                   gt_rule column (the DNF/source rule string for that tier).
  statistics.csv : per (dataset, variant) stats — n_test, vocab_size, avg atoms/bonds,
                   GT tiers and their rule strings.

Usage: harvest_v2.py [--out-root final_v2]
"""
import os, sys, glob, json, csv, subprocess, argparse, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--out-root", default="final_v2")
args = ap.parse_args()
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(PROJ, args.out_root) if not os.path.isabs(args.out_root) else args.out_root
TABLES = os.path.join(OUT, "tables"); os.makedirs(TABLES, exist_ok=True)
VOCAB = os.path.join(PROJ, "vocab_" + os.path.basename(OUT))   # vocab_final_v2

# ---- 1. results.csv via the existing, tested harvester ----
subprocess.run([sys.executable, os.path.join(PROJ, "analysis", "harvest_results.py"),
                "--out_root", OUT, "--save_dir", TABLES], check=False)

def _rule_for(ds, variant, tier):
    """gt_rule string for (dataset, base-variant, tier) from rule_tiers.json."""
    if not tier or tier in ("real", "source", "none", ""):
        return "SOURCE_SMARTS" if tier == "source" else ""
    base = variant.replace("_filter", "")
    p = os.path.join(VOCAB, ds, base, "rule_tiers.json")
    if not os.path.exists(p):
        return ""
    try:
        v = json.load(open(p)).get(tier, "")
        # rule_tiers.json stores the rule under 'rule_str'; fall back to older keys.
        return v.get("rule_str", v.get("rule", v.get("best_rule", ""))) if isinstance(v, dict) else str(v)
    except Exception:
        return ""

# ---- 2. add gt_rule column to the latest results csv ----
rf = sorted(glob.glob(os.path.join(TABLES, "results_*.csv")))
if rf:
    path = rf[-1]
    rows = list(csv.DictReader(open(path)))
    if rows:
        for r in rows:
            r["gt_rule"] = _rule_for(r.get("dataset", ""), r.get("vocab_variant", ""), r.get("gt_tier", ""))
        cols = list(rows[0].keys())
        if "gt_rule" not in cols:
            cols.append("gt_rule")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
        print(f"results.csv: {len(rows)} rows (+ gt_rule) -> {path}")

# ---- 3. statistics.csv ----
try:
    from rdkit import Chem
    have_rdkit = True
except ImportError:
    have_rdkit = False

stat_rows = []
for vd in sorted(glob.glob(os.path.join(VOCAB, "*", "*"))):
    mp = os.path.join(vd, "vocab_meta.json")
    if not os.path.exists(mp):
        continue
    meta = json.load(open(mp)); ds = os.path.basename(os.path.dirname(vd)); variant = os.path.basename(vd)
    nodes = []; edges = []
    sp = os.path.join(vd, "smiles_labels.csv")
    if have_rdkit and os.path.exists(sp):
        for r in csv.DictReader(open(sp)):
            smi = r.get("smiles") or r.get("SMILES") or next(iter(r.values()))
            m = Chem.MolFromSmiles(smi) if smi else None
            if m:
                nodes.append(m.GetNumAtoms()); edges.append(m.GetNumBonds())
    rt = os.path.join(vd, "rule_tiers.json")
    tiers = json.load(open(rt)) if os.path.exists(rt) else {}
    reg = meta.get("task_type") == "Regression"
    stat_rows.append(dict(
        dataset=ds, variant=variant, task_type=meta.get("task_type"),
        n_total=meta.get("n_total"), n_trainval=meta.get("n_trainval"), n_test=meta.get("n_test"),
        n_neg=meta.get("n0_trainval"), n_pos=meta.get("n1_trainval"),
        pos_frac=("" if reg else round((meta.get("n1_trainval") or 0) / max(meta.get("n_trainval") or 1, 1), 3)),
        vocab_size=meta.get("vocab_size"),
        avg_atoms=(round(st.mean(nodes), 1) if nodes else ""), avg_bonds=(round(st.mean(edges), 1) if edges else ""),
        n_gt_tiers=len(tiers), gt_tiers=";".join(tiers.keys())))
if stat_rows:
    cols = list(stat_rows[0].keys())
    sp = os.path.join(TABLES, "statistics.csv")
    with open(sp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(stat_rows)
    print(f"statistics.csv: {len(stat_rows)} rows -> {sp}")
