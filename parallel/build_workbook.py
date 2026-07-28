#!/usr/bin/env python3
"""build_workbook.py — assemble the two result workbooks (the two regimes) from results.csv.

Matches config/experiment_matrix.yaml `output.sheets`: one fold-averaged row per config
(dataset, backbone, fragmentation, filtered, gt_tier, gt_rule); per-method or per-model columns.

  BBBP_original_labels.xlsx : Model Performance, Grouped Pearson, Instance Pearson
  BBBP_relabelled.xlsx      : + Grouped/Instance GTROC (relabelled only)

Post-hoc method metrics come from the `baselines`-family rows (explainer-prefixed columns);
MOSE/GSAT/MotifSAT metrics come from their own family rows (native unprefixed columns).
Partial runs are fine — cells not yet finished stay blank.

Usage: build_workbook.py <results.csv> <out_dir>
"""
import sys, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

RES = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(RES, low_memory=False)

# ---- identity / regime ----
df["filtered"] = df["vocab_variant"].astype(str).str.endswith("_filter")
df["fragmentation"] = df["vocab_variant"].astype(str).str.replace("_filter", "", regex=False)
df["gt_tier"] = df.get("gt_tier", "").fillna("").astype(str).replace("None", "")
df["gt_rule"] = df.get("gt_rule", "").fillna("").astype(str)
df["regime"] = np.where(df["gt_tier"].str.startswith("dnf"), "relabelled", "original_labels")
IDX = ["dataset", "backbone", "fragmentation", "filtered", "gt_tier", "gt_rule"]

POSTHOC = ["gnnexplainer", "pgexplainer", "mage_official", "motif_occlusion"]
ANTEHOC = ["mose", "gsat", "motifsat"]          # family == label
# whether BBBP-style (classification → auc) or regression (→ rmse_orig)
PERF_METRIC = "auc" if df["auc"].notna().any() else "rmse_orig"


def per_method(rdf, native_col, posthoc_suffix):
    """One fold-averaged row per config; one column per method (7)."""
    out = rdf[IDX].drop_duplicates().copy()
    bl = rdf[rdf["family"] == "baselines"]
    for m in POSTHOC:
        col = f"{m}_{posthoc_suffix}"
        if col in rdf.columns:
            out = out.merge(bl.groupby(IDX)[col].mean().rename(m), on=IDX, how="left")
        else:
            out[m] = np.nan
    for fam in ANTEHOC:
        sub = rdf[rdf["family"] == fam]
        if native_col in rdf.columns:
            out = out.merge(sub.groupby(IDX)[native_col].mean().rename(fam), on=IDX, how="left")
        else:
            out[fam] = np.nan
    return out.sort_values(IDX).reset_index(drop=True)


def model_perf(rdf):
    out = rdf[IDX].drop_duplicates().copy()
    for fam, label in [("vanilla", "Vanilla"), ("mose", "MOSE"), ("gsat", "GSAT"), ("motifsat", "MotifSAT")]:
        sub = rdf[rdf["family"] == fam]
        out = out.merge(sub.groupby(IDX)[PERF_METRIC].mean().rename(f"{label}_{PERF_METRIC}"),
                        on=IDX, how="left")
    return out.sort_values(IDX).reset_index(drop=True)


def _nonblank(t):
    m = [c for c in t.columns if c not in IDX]
    return int(t[m].notna().any(axis=1).sum())


for regime, rdf in df.groupby("regime"):
    sheets = {
        "Model Performance": model_perf(rdf),
        "Grouped Pearson": per_method(rdf, "pearson_motif", "max_pearson_motif"),
        "Instance Pearson": per_method(rdf, "pearson_instance", "max_pearson_instance"),
    }
    if regime == "relabelled":
        sheets["Grouped GTROC"] = per_method(rdf, "gt_roc_node_auc_mean", "max_gt_roc_node_auc_mean")
        sheets["Instance GTROC"] = per_method(rdf, "gt_roc_node_fired_auc_mean", "max_gt_roc_node_fired_auc_mean")
    # Explicit single-string config id as the first column of every sheet, so each
    # row is self-identifying (backbone·fragmentation·filtered·tier) rather than the
    # reader stitching the 6 index columns together.
    def _cfg_id(r):
        return "·".join([str(r["backbone"]), str(r["fragmentation"]),
                         "filter" if r["filtered"] else "full",
                         str(r["gt_tier"]) if str(r["gt_tier"]) else "real"])
    for _name, _t in sheets.items():
        _t.insert(0, "config", _t.apply(_cfg_id, axis=1))
    status = pd.DataFrame({"note": [
        f"Regime: {regime}",
        f"One row = one config (see the 'config' column). {len(sheets['Model Performance'])} configs = backbone x fragmentation x filtered x gt_tier.",
        f"Perf metric: {PERF_METRIC}   |   Fold-averaged.",
        "Partial run — blank cells = that (model x config) not finished yet.",
        "Post-hoc columns (gnnexplainer/pgexplainer/mage_official/motif_occlusion) fill in as CPU posthoc lands.",
        "KNOWN: PGExplainer NaN on high-dynamic-range backbones (mask collapse: PNA 100%, GAT/GCN/SAGE partial, GIN 0%). Honest limitation, not a gap.",
        "Top 10 / Bottom 10 + node_mask_probe: PENDING (need per-motif rank export + probe pass).",
        "GTROC uses gt_roc_node_auc_mean (grouped) / gt_roc_node_fired_auc_mean (instance proxy).",
    ]})
    path = os.path.join(OUT, f"BBBP_{regime}.xlsx")
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            status.to_excel(xw, "STATUS", index=False)
            for name, t in sheets.items():
                t.to_excel(xw, name[:31], index=False)
        print(f"wrote {path}  ({len(sheets)} sheets)  "
              + "  ".join(f"{n}:{_nonblank(t)}nb" for n, t in sheets.items()))
    except Exception as e:                       # openpyxl missing → CSV fallback
        for name, t in sheets.items():
            t.to_csv(os.path.join(OUT, f"BBBP_{regime}__{name.replace(' ', '_')}.csv"), index=False)
        print(f"[no xlsx: {e}] wrote CSV sheets for {regime}")
