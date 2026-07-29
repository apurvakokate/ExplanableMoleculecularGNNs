#!/usr/bin/env python3
"""build_topbottom_probe.py — EVAL-ONLY assembler for the two workbook sheets that were
"PENDING" in build_workbook.py: **Top 10 / Bottom 10** and the **node_mask_probe** field.

No retraining. The node_mask_probe metric itself is already implemented in
analysis/probe_masked_nodes.py (atom-type-recovery gap, all 7 methods) but was never run;
this script (a) ensures that probe output exists (runs it if missing), and (b) assembles the
per-motif Top/Bottom-10 rows from data already written per run, joins discriminativeness, and
broadcasts the method-level node_mask_probe onto each method's rows — matching the yaml spec
(output.sheets Top 10 / Bottom 10; experiment node_mask_probe run_for all_7_methods).

Per-motif sources (already on disk per run):
  ante-hoc (mose/gsat/motifsat): <run>/top_bottom.csv   (rank, top_motif_id, top_score,
                                 top_impact, bottom_motif_id, bottom_score, bottom_impact)
  post-hoc (4 explainers):       recomputed via top_bottom_motif_eval() from
                                 <run>/{method}_motif_scores_max.csv + per-motif impact
  discriminativeness:            <run>/discriminativeness.csv (motif_id -> abs_disc/presence_auc)
  node_mask_probe:               masked_node_probe.csv (one gap per run x method)

Output: Top_10.csv / Bottom_10.csv in <tables> with columns
  [dataset, backbone, fragmentation, filtered, gt_tier, gt_rule, method, motif_rank,
   top_n_motif_id, importance, impact, discriminativeness, node_mask_probe]
(and, if openpyxl is available, appended as sheets into the existing per-dataset workbooks).

NOTE: this is an assembler over per-run artifacts + the probe; validate on a couple of runs
(--limit) on the HPC before a full pass, since a few CSV column names vary by trainer version.

  python analysis/build_topbottom_probe.py --out_root final_v2 \
         --data_root <DATA_ROOT> --vocab_root vocab_final_v2 [--run_probe 1] [--limit N]
"""
import argparse, json, os, subprocess, sys, glob
from pathlib import Path
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

ARCHIVE = ("_archive", "_trash", "_old")
POSTHOC = ("gnnexplainer", "pgexplainer", "mage_official", "motif_occlusion")
ANTEHOC = {"mose": "mose", "base_gsat": "gsat", "motifsat": "motifsat"}
K = 10


def _cfg_from_meta(meta):
    v = str(meta.get("vocab_variant", ""))
    return dict(
        dataset=meta.get("dataset"), backbone=meta.get("backbone"),
        fragmentation=v.replace("_filter", ""), filtered=v.endswith("_filter"),
        gt_tier=str(meta.get("gt_tier", "") or ""), gt_rule="")  # gt_rule filled by harvest_v2


def _disc_map(run_dir):
    """motif_id -> discriminativeness (abs_disc, fallback presence_auc). Empty for regression."""
    p = Path(run_dir) / "discriminativeness.csv"
    if not p.exists():
        return {}
    try:
        d = pd.read_csv(p)
    except Exception:
        return {}
    key = "abs_disc" if "abs_disc" in d.columns else ("presence_auc" if "presence_auc" in d.columns else None)
    idc = "motif_id" if "motif_id" in d.columns else d.columns[0]
    return {} if key is None else dict(zip(d[idc], d[key]))


def _antehoc_rows(run_dir, disc):
    """20 rows (Top 1-10, Bottom 1-10) from the ante-hoc top_bottom.csv paired columns."""
    p = Path(run_dir) / "top_bottom.csv"
    if not p.exists():
        return []
    try:
        t = pd.read_csv(p)
    except Exception:
        return []
    rows = []
    for _, r in t.iterrows():
        rk = int(r.get("rank", 0))
        if "top_motif_id" in t.columns:
            mid = r.get("top_motif_id")
            rows.append(dict(section="Top", motif_rank=rk, top_n_motif_id=mid,
                             importance=r.get("top_score"), impact=r.get("top_impact"),
                             discriminativeness=disc.get(mid, "")))
        if "bottom_motif_id" in t.columns:
            mid = r.get("bottom_motif_id")
            rows.append(dict(section="Bottom", motif_rank=rk, top_n_motif_id=mid,
                             importance=r.get("bottom_score"), impact=r.get("bottom_impact"),
                             discriminativeness=disc.get(mid, "")))
    return rows


def _posthoc_rows(run_dir, method, disc):
    """Recompute Top/Bottom-K for a post-hoc method from its saved per-motif score CSV."""
    sp = Path(run_dir) / f"{method}_motif_scores_max.csv"
    ip = Path(run_dir) / f"{method}_max_score_vs_impact_own.csv"
    if not sp.exists():
        return []
    try:
        s = pd.read_csv(sp)
    except Exception:
        return []
    idc = "motif_id" if "motif_id" in s.columns else s.columns[0]
    sc = "score_max" if "score_max" in s.columns else [c for c in s.columns if c != idc][0]
    imp = {}
    if ip.exists():
        try:
            di = pd.read_csv(ip)
            ic = "impact_mean" if "impact_mean" in di.columns else None
            if ic:
                imp = dict(zip(di[di.columns[0]], di[ic]))
        except Exception:
            imp = {}
    s = s.sort_values(sc, ascending=False).reset_index(drop=True)
    rows = []
    for rk in range(min(K, len(s))):
        mid = s.iloc[rk][idc]
        rows.append(dict(section="Top", motif_rank=rk + 1, top_n_motif_id=mid,
                         importance=s.iloc[rk][sc], impact=imp.get(mid, ""),
                         discriminativeness=disc.get(mid, "")))
    for j in range(min(K, len(s))):
        rk = len(s) - min(K, len(s)) + j
        mid = s.iloc[rk][idc]
        rows.append(dict(section="Bottom", motif_rank=rk + 1, top_n_motif_id=mid,
                         importance=s.iloc[rk][sc], impact=imp.get(mid, ""),
                         discriminativeness=disc.get(mid, "")))
    return rows


def _probe_lookup(probe_csv):
    """(run_dir, method) -> node_mask_probe gap. Ante-hoc gated_ / post-hoc expl_ gap column."""
    out = {}
    if not os.path.exists(probe_csv):
        return out
    try:
        p = pd.read_csv(probe_csv)
    except Exception:
        return out
    gap_cols = [c for c in ("gated_gap_unmasked_minus_masked",
                            "expl_gap_unmasked_minus_masked") if c in p.columns]
    for _, r in p.iterrows():
        gap = next((r[c] for c in gap_cols if pd.notna(r.get(c))), "")
        out[(str(r.get("run_dir")), str(r.get("method", r.get("family", ""))))] = gap
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="final_v2")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--vocab_root", required=True)
    ap.add_argument("--run_probe", type=int, default=1, help="run probe_masked_nodes.py if its CSV is missing")
    ap.add_argument("--limit", type=int, default=0, help="validate on first N runs (0=all)")
    a = ap.parse_args()
    tables = Path(a.out_root) / "tables"; tables.mkdir(parents=True, exist_ok=True)
    probe_csv = str(tables / "masked_node_probe.csv")

    # (a) ensure node_mask_probe results exist (the metric is implemented; just never run)
    if a.run_probe and not os.path.exists(probe_csv):
        print("[probe] running analysis/probe_masked_nodes.py (never run before) ...")
        subprocess.run([sys.executable, str(_REPO / "analysis" / "probe_masked_nodes.py"),
                        "--out_root", a.out_root, "--data_root", a.data_root,
                        "--vocab_root", a.vocab_root, "--include_posthoc", "1",
                        "--save", probe_csv], check=False)
    probe = _probe_lookup(probe_csv)

    # (b) assemble Top/Bottom rows over every finished run
    all_rows = []
    run_dirs = [Path(p).parent for p in glob.glob(str(Path(a.out_root) / "*" / "**" / "best_model.pt"),
                                                  recursive=True)
                if not any(x in p for x in ARCHIVE)]
    if a.limit:
        run_dirs = run_dirs[:a.limit]
    for rd in run_dirs:
        sj = rd / "summary.json"
        if not sj.exists():
            continue
        try:
            meta = json.load(open(sj, encoding="utf-8"))
        except Exception:
            continue
        cfg = _cfg_from_meta(meta)
        disc = _disc_map(rd)
        fam = next((v for k, v in ANTEHOC.items() if f"/{k}/" in str(rd).replace(os.sep, "/")), None)
        method_rows = {}
        if fam:                                                  # ante-hoc: native explanation
            method_rows[fam] = _antehoc_rows(rd, disc)
        else:                                                    # baselines: 4 post-hoc explainers
            for m in POSTHOC:
                r = _posthoc_rows(rd, m, disc)
                if r:
                    method_rows[m] = r
        for method, rows in method_rows.items():
            gap = probe.get((str(rd), method), "")
            for row in rows:
                all_rows.append({**cfg, "method": method, "node_mask_probe": gap, **row})

    if not all_rows:
        print("no Top/Bottom rows assembled (no per-run top_bottom.csv / motif score CSVs found)")
        return
    df = pd.DataFrame(all_rows)
    cols = ["dataset", "backbone", "fragmentation", "filtered", "gt_tier", "gt_rule",
            "method", "section", "motif_rank", "top_n_motif_id", "importance", "impact",
            "discriminativeness", "node_mask_probe"]
    df = df[[c for c in cols if c in df.columns]]
    for section, name in (("Top", "Top_10"), ("Bottom", "Bottom_10")):
        sub = df[df["section"] == section].drop(columns=["section"])
        out = tables / f"{name}.csv"
        sub.to_csv(out, index=False)
        print(f"wrote {out}  ({len(sub)} rows, {sub[['dataset','method']].drop_duplicates().shape[0]} config·method blocks)")
    print("\nDONE. Merge Top_10.csv / Bottom_10.csv into the workbooks (or add as sheets in build_workbook).")


if __name__ == "__main__":
    main()
