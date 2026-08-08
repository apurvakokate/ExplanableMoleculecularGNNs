#!/usr/bin/env python3
"""build_thesis_results.py — assemble the thesis MAIN results file from the current
final_v2 base runs, WITHOUT touching any existing results.

Writes only under --out (default: a NEW `thesis_results/` in the PARENT of the code
dir). Nothing in final_v2 / final_local / the repo is modified or overwritten.

MAIN scope (per the thesis spec):
  datasets (10, molhiv excluded): Mutagenicity, BBBP, esol, Lipophilicity, hERG,
    ogbg-molbace  (original labels, gt_tier='none')  +  Benzene, Alkane_Carbonyl,
    Fluoride_Carbonyl, mutag  (source-GT, gt_tier='source').   [planted dnf_* -> SUPP]
  backbones(5) x methods(8) x fragmentations(4) x filtered(2) x rule_tier x fold.

KEY = (dataset, backbone, method, fragmentation, filtered, rule_tier, fold).
Each key is written ONCE: re-runs append only keys not already present (idempotent),
so a partial run can be resumed and duplicates are impossible. A coverage report flags
cells that are present-but-incomplete (missing backbones/folds) so gaps are caught.

Data source: a FRESH `harvest_all_runs.py` pass over final_v2 (so we get the 45 Aug-1
fills the stale scratch CSV misses). Metrics are taken as-is from the harvest; the
4-variant weighted×UNK pearson (thesis_pearson_variants.py) is a SEPARATE --with-variants
pass that reads each run's score_vs_impact.csv (no model re-eval).
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: analysis.* / SharedModules.*

MAIN_DATASETS = [
    "Mutagenicity", "BBBP", "esol", "Lipophilicity", "hERG", "ogbg-molbace",
    "Benzene_Verified_GT", "Alkane_Carbonyl_Verified_GT",
    "Fluoride_Carbonyl_Verified_GT", "mutag",
]
BACKBONES = ["GIN", "GCN", "GAT", "SAGE", "PNA"]
METHODS = ["(vanilla-model)", "mose", "motifsat", "gsat",
           "gnnexplainer", "pgexplainer", "mage_official", "motif_occlusion"]
FRAGS = ["rbrics", "rdkit_fg_first", "ertl_first", "fg_first"]
# sheet -> which gt_tier values belong to it. main = original-label + source-GT;
# supplementary = the planted-DNF (synthetic-relabelled) tiers.
SHEET_TIERS = {
    "main":          ["none", "source"],
    "supplementary": ["dnf_k2_r1", "dnf_k2_r2", "dnf_k3_r1", "dnf_k3_r2"],
}
POSTHOC = {"gnnexplainer", "pgexplainer", "mage_official", "motif_occlusion"}

KEY = ["dataset", "backbone", "method", "fragmentation", "filtered", "rule_tier", "fold"]
CTX = ["regime", "task_type", "node_encoder", "vocab_variant", "pooling", "status",
       "n_test", "summary_path"]
METRICS = ["auc", "val_auc", "train_auc", "rmse", "rmse_orig", "mae", "mae_orig",
           "grouped_pearson", "grouped_pearson_agnostic",
           "instance_pearson_own", "instance_pearson_agnostic",
           "grouped_spearman", "instance_spearman_own", "instance_spearman_agnostic",
           "grouped_gtroc", "instance_gtroc_fired", "global_gtroc_dnf",
           "instance_gtroc_dnf", "gtroc_edge", "spurious_roc"]
# Requested but NOT present in any summary -> require model RE-EVALUATION (forward passes).
NEEDS_REEVAL = ["train_rmse", "val_rmse", "train_mae", "val_mae",
                "train_rmse_orig", "val_rmse_orig", "train_mae_orig", "val_mae_orig"]


def run_harvest(final_v2: Path, repo: Path, save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(repo / "analysis" / "harvest_all_runs.py"),
           "--out-root", str(final_v2), "--repo", str(repo), "--save-dir", str(save_dir)]
    print("  [harvest]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    csv = save_dir / "all_runs_metrics.csv"
    if not csv.exists():
        sys.exit(f"harvest did not produce {csv}")
    return csv


def add_variants(main_path: Path, vocab_root: str, split_roots: list) -> None:
    """PHASE B (no model re-eval): append the 4-variant weighted×UNK grouped/instance
    pearson+spearman to each row, read from that run's score_vs_impact.csv. UNK is the
    EXACT per-fold filter via thesis_pearson_variants.kept_smarts_auto. Idempotent: any
    pre-existing variant_* columns are dropped and recomputed. The metric base matches
    the reported column — native -> own impact, post-hoc -> agnostic impact."""
    from analysis.thesis_pearson_variants import (
        variants_from_score_vs_impact, kept_smarts_auto)
    df = pd.read_csv(main_path, low_memory=False)
    df = df[[c for c in df.columns if not (c.startswith(("grouped_pearson_", "grouped_spearman_",
             "instance_pearson_", "instance_spearman_", "grouped_n_", "instance_n_"))
             or c == "variant_unk_source" or c == "variant_method")]]  # drop stale variant cols

    kept_cache: dict = {}
    def kept(ds, frag, fold):
        k = (ds, frag, int(fold))
        if k not in kept_cache:
            try:
                kept_cache[k] = kept_smarts_auto(ds, frag, int(fold), vocab_root, split_roots)
            except Exception as e:
                kept_cache[k] = None
                print(f"    [warn] kept_smarts failed {k}: {e}")
        return kept_cache[k]

    rows, n_done, n_skip = [], 0, 0
    for _, r in df.iterrows():
        method, sp = r["method"], r.get("summary_path")
        if method == "(vanilla-model)" or not isinstance(sp, str):
            rows.append({}); n_skip += 1; continue
        d = os.path.dirname(sp)
        if method in POSTHOC:
            csv = os.path.join(d, f"{method}_mean_score_vs_impact.csv"); mkind = "agnostic"
        else:
            csv = os.path.join(d, "score_vs_impact.csv"); mkind = "own"
        v = variants_from_score_vs_impact(csv, method=mkind,
                                          kept_smarts=kept(r["dataset"], r["fragmentation"], r["fold"]))
        rows.append(v)
        if v: n_done += 1
        else: n_skip += 1
        if (n_done + n_skip) % 1000 == 0:
            print(f"    variants: {n_done+n_skip}/{len(df)} (computed {n_done}, empty {n_skip})")
    vdf = pd.DataFrame(rows, index=df.index)
    out = pd.concat([df, vdf], axis=1)
    out.to_csv(main_path, index=False)
    src = vdf["variant_unk_source"].value_counts().to_dict() if "variant_unk_source" in vdf else {}
    print(f"  variants appended: computed {n_done}, empty/skip {n_skip}; unk_source={src}")
    print(f"  -> {main_path} (+{len([c for c in vdf.columns])} variant columns)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-v2", required=True, help="abs path to final_v2 root")
    ap.add_argument("--repo", required=True, help="abs path to the code repo")
    ap.add_argument("--out", required=True, help="thesis_results dir (parent, NOT code)")
    ap.add_argument("--sheet", choices=["main", "supplementary"], default="main",
                    help="main = original+source-GT; supplementary = planted-DNF (synthetic)")
    ap.add_argument("--skip-harvest", action="store_true",
                    help="reuse existing <out>/_harvest/all_runs_metrics.csv")
    ap.add_argument("--with-variants", action="store_true",
                    help="Phase B: append 4-variant weighted×UNK pearson/spearman (no re-eval)")
    ap.add_argument("--vocab-root", default=None,
                    help="vocab_final_v2 root (default: <repo>/vocab_final_v2)")
    ap.add_argument("--split-roots", default=None,
                    help="comma-separated fold-split roots, e.g. FOLDS,data,data/ogb")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    harvest_csv = out / "_harvest" / "all_runs_metrics.csv"
    if not args.skip_harvest:
        harvest_csv = run_harvest(Path(args.final_v2), Path(args.repo), out / "_harvest")
    df = pd.read_csv(harvest_csv, low_memory=False)

    # ── map to the thesis key space ───────────────────────────────────────────
    df["rule_tier"] = df["gt_tier"].fillna("none").replace("", "none")
    df["filtered"] = df["filtered"].astype(str)
    tiers = SHEET_TIERS[args.sheet]
    scope = (df["dataset"].isin(MAIN_DATASETS)
             & df["method"].isin(METHODS)
             & df["fragmentation"].isin(FRAGS)
             & df["rule_tier"].isin(tiers))
    m = df[scope].copy()

    # Node->motif aggregation: post-hoc explainers emit BOTH pooling='mean' and
    # pooling='max' rows; caveat #4 mandates MEAN. Drop 'max' so each cell is the
    # mean reduction. Native rows carry pooling in {'native','na'} and are untouched.
    # This is what makes the 7-dim KEY unique (otherwise mean/max collapse arbitrarily).
    if "pooling" in m.columns:
        _pool = m["pooling"].astype(str)
        n_max = int((_pool == "max").sum())
        m = m[_pool != "max"].copy()
        print(f"  dropped {n_max} pooling=max rows (kept MEAN per caveat #4)")

    keep = [c for c in (KEY + CTX + METRICS) if c in m.columns]
    m = m[keep]

    # ── dedup: one row per KEY (write once). Prefer a completed status. ────────
    if "status" in m.columns:
        m["_ok"] = (m["status"].astype(str).str.lower() == "ok").astype(int)
        m = m.sort_values("_ok", ascending=False).drop(columns="_ok")
    before = len(m)
    m = m.drop_duplicates(subset=KEY, keep="first")
    print(f"  scoped rows {before} -> {len(m)} unique keys")

    # ── idempotent append to an existing sheet file ───────────────────────────
    main_path = out / f"{args.sheet}_results.csv"
    if main_path.exists():
        prev = pd.read_csv(main_path, low_memory=False)
        prev_keys = set(map(tuple, prev[KEY].astype(str).values.tolist()))
        mask_new = ~m[KEY].astype(str).apply(tuple, axis=1).isin(prev_keys)
        added = m[mask_new]
        combined = pd.concat([prev, added], ignore_index=True)
        print(f"  existing {len(prev)} rows; appended {len(added)} new keys (write-once)")
    else:
        combined = m
        print(f"  new main file: {len(m)} rows")
    combined.to_csv(main_path, index=False)
    print(f"  -> {main_path}")

    # ── coverage report: catch present-but-incomplete cells ───────────────────
    grp = ["dataset", "method", "fragmentation", "filtered", "rule_tier"]
    cov = (combined.groupby(grp)
           .agg(n_rows=("fold", "size"),
                n_backbones=("backbone", "nunique"),
                n_folds=("fold", "nunique")).reset_index())
    cov["complete_25"] = (cov["n_backbones"] == 5) & (cov["n_folds"] == 5)
    cov_path = out / f"{args.sheet}_coverage_report.csv"
    cov.sort_values(["complete_25", "dataset", "method"]).to_csv(cov_path, index=False)
    incomplete = cov[~cov["complete_25"]]
    print(f"  coverage cells: {len(cov)}  incomplete(<5 bb x 5 folds): {len(incomplete)} "
          f"-> {cov_path}")

    # ── per-dimension presence + re-eval note ─────────────────────────────────
    print("\n  PRESENT per dimension:")
    for d in ["dataset", "backbone", "method", "fragmentation", "filtered", "rule_tier", "fold"]:
        print(f"    {d}: {sorted(combined[d].astype(str).unique())}")
    print("\n  NEEDS MODEL RE-EVALUATION (not in any summary; forward passes on train/val "
          "splits from best_model.pt):")
    for k in NEEDS_REEVAL:
        print(f"    - {k}")
    if not args.with_variants:
        print("  NEEDS SEPARATE PASS (no re-eval; run with --with-variants): the 4-variant "
              "weighted×UNK grouped/instance pearson via thesis_pearson_variants.py "
              "(reads each run's score_vs_impact.csv; TEST split only — JOINT needs re-eval).")
        return

    # ── Phase B ───────────────────────────────────────────────────────────────
    vocab_root = args.vocab_root or str(Path(args.repo) / "vocab_final_v2")
    if not args.split_roots:
        sys.exit("--with-variants requires --split-roots (e.g. <FOLDS>,<data>,<data/ogb>)")
    split_roots = [s.strip() for s in args.split_roots.split(",") if s.strip()]
    print(f"\n  PHASE B: 4-variant weighted×UNK pearson/spearman (vocab={vocab_root})")
    add_variants(Path(args.out) / f"{args.sheet}_results.csv", vocab_root, split_roots)


if __name__ == "__main__":
    main()
