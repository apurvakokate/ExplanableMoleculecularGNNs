#!/usr/bin/env python3
"""harvest_factorial_results.py — harvest the rbrics MoSE 24-combo architecture factorial.

Grid axes: node_encoder{onehot,linear} x conv_normalize{none,l2,layernorm}
           x graph_pool{add,mean} x unk_mode{fixed,learnable_shared}  (REAL tier, rbrics_filter).

Source of truth per cell = summary.json (flat: all config + task + faithfulness metrics),
augmented with mose_grouped_corr_pooled_testonly.csv for the SUPPORT-WEIGHTED grouped pearson
(pearson_w) alongside the unweighted (pearson_u) — both reported, per the unweighted-artifact
finding. Scans every tree that can hold a factorial cell and DEDUPES by the 7 axes, preferring
a run that has per-split eval (summary_splits.json), tie-broken by latest run_timestamp.

No hallucination / no abstraction: emits one RAW row per (ds,fold,bb,enc,norm,pool,unk) cell,
then a FOLD-AVERAGED table (mean+std+n over folds only; every other differing axis stays a
separate row).

Usage:
  python3 analysis/harvest_factorial_results.py \
      --root /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor \
      --out  /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/ablated_completely_v1/_dispatch_fac
Outputs: factorial_raw.csv, factorial_foldavg.csv in --out.
"""
import argparse, glob, json, os, csv, sys
import pandas as pd

VOCAB = "rbrics_filter"


def load_json_lenient(path):
    """Tolerate a double-written summary.json (one valid object + trailing junk bytes from an
    interrupted re-write) by decoding just the first JSON object via raw_decode."""
    try:
        raw = open(path).read()
    except Exception:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw)
            return obj
        except Exception:
            return None
# metric columns to carry from summary.json (flat keys)
TASK_COLS   = ["auc", "train_auc", "val_auc", "rmse", "mae", "rmse_orig", "mae_orig"]
FAITH_COLS  = ["pearson", "spearman", "pearson_motif", "spearman_motif",
               "pearson_instance", "spearman_instance",
               "pearson_instance_agnostic", "spearman_instance_agnostic"]
GTROC_COLS  = ["gt_roc_auc_mean", "gt_roc_node_auc_mean", "instance_gt_roc_node_auc_mean",
               "gt_roc_n_graphs"]
SCORE_COLS  = ["score_mean", "score_std", "score_min", "score_max", "top_k_abs_disc"]
AXES        = ["dataset", "fold", "backbone", "node_encoder", "conv_normalize",
               "graph_pool", "unk_mode"]
META        = ["apply_layer_norm", "vocab_variant", "num_layers", "hidden_dim",
               "w_feat", "w_message", "w_readout", "use_gt", "variant_tag",
               "run_timestamp", "git_sha", "config_hash"]


def grouped_weighted(cell_dir):
    """pearson_w/u (support-weighted vs unweighted, exclunk) from the pooled testonly csv."""
    p = os.path.join(cell_dir, "mose_grouped_corr_pooled_testonly.csv")
    out = {}
    if os.path.exists(p):
        try:
            with open(p) as f:
                row = next(csv.DictReader(f))
            for k in ("pearson_w_exclunk", "pearson_u_exclunk",
                      "spearman_w_exclunk", "spearman_u_exclunk", "n_motifs_exclunk"):
                v = row.get(k)
                out[k] = float(v) if v not in (None, "", "nan", "NaN") else float("nan")
        except Exception:
            pass
    return out


def is_factorial_cell(d):
    """A real-tier rbrics_filter MoSE wf+wr cell with the four axes populated."""
    return (d.get("family") == "mose"
            and d.get("vocab_variant") == VOCAB
            and d.get("dataset") != "mutag"           # mutag is excluded from this factorial
            and not d.get("use_gt", False)
            and bool(d.get("w_feat")) and bool(d.get("w_readout")) and not d.get("w_message")
            and d.get("node_encoder") in ("onehot", "linear")
            and d.get("conv_normalize") in ("none", "l2", "layernorm")
            and d.get("graph_pool") in ("add", "mean")
            and d.get("unk_mode") in ("fixed", "learnable_shared"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Claude+Cursor base dir")
    ap.add_argument("--out", required=True, help="output dir for CSVs")
    args = ap.parse_args()
    R = args.root
    trees = [f"{R}/ablated_completely_v1/mose/{VOCAB}",
             f"{R}/ablation_v2/normal/mose/{VOCAB}",
             f"{R}/ablation_v2/m1/mose/{VOCAB}",
             f"{R}/final_v2/mose/{VOCAB}"]
    os.makedirs(args.out, exist_ok=True)

    best = {}   # 7-axis key -> (has_splits, timestamp, row)
    n_seen = n_kept = 0
    for t in trees:
        for sj in glob.glob(f"{t}/*/fold*/*/summary.json"):
            n_seen += 1
            d = load_json_lenient(sj)
            if d is None:
                print(f"PARSE FAIL {sj}", file=sys.stderr); continue
            if not is_factorial_cell(d):
                continue
            cell_dir = os.path.dirname(sj)
            key = tuple(d.get(a) for a in AXES)
            has_splits = os.path.exists(os.path.join(cell_dir, "summary_splits.json"))
            ts = d.get("run_timestamp", "")
            prev = best.get(key)
            # prefer per-split-eval run, then latest timestamp
            if prev is not None and (prev[0], prev[1]) >= (has_splits, ts):
                continue
            row = {a: d.get(a) for a in AXES}
            for c in TASK_COLS + FAITH_COLS + GTROC_COLS + SCORE_COLS + META:
                row[c] = d.get(c)
            row.update(grouped_weighted(cell_dir))
            row["_cell_dir"] = cell_dir
            best[key] = (has_splits, ts, row)

    rows = [v[2] for v in best.values()]
    n_kept = len(rows)
    if not rows:
        print("No factorial cells found — check --root.", file=sys.stderr); sys.exit(1)
    raw = pd.DataFrame(rows).sort_values(AXES).reset_index(drop=True)
    raw_path = os.path.join(args.out, "factorial_raw.csv")
    raw.to_csv(raw_path, index=False)

    # fold-averaged: group by every axis EXCEPT fold
    gcols = [a for a in AXES if a != "fold"]
    num = raw.select_dtypes("number").columns.tolist()
    num = [c for c in num if c != "fold"]
    agg = raw.groupby(gcols)[num].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    # a single n_folds column (count of a reliable metric)
    base_count = "auc_count" if "auc_count" in agg.columns else (num[0] + "_count")
    agg["n_folds"] = agg[base_count]
    avg_path = os.path.join(args.out, "factorial_foldavg.csv")
    agg.to_csv(avg_path, index=False)

    # console summary
    print(f"scanned summary.json: {n_seen}   factorial cells kept (deduped): {n_kept}  (expect 4800)")
    print(f"distinct configs (fold-collapsed): {len(agg)}  (expect 960 = 8 ds x 5 bb x 24 combos)")
    print("cells per (enc,norm,pool,unk) combo  (each should be 200 = 8 ds x 5 bb x 5 folds):")
    combo = raw.groupby(["node_encoder", "conv_normalize", "graph_pool", "unk_mode"]).size()
    print(combo.to_string())
    print(f"\nRAW  -> {raw_path}")
    print(f"AVG  -> {avg_path}")
    # hygiene flags
    bad_ln = raw[(raw["conv_normalize"] == "none") & (raw["apply_layer_norm"] == True)]
    if len(bad_ln):
        print(f"\n[WARN] {len(bad_ln)} cells have conv_normalize=none AND apply_layer_norm=True "
              f"(tag-collision trap) — inspect before trusting their norm label.")


if __name__ == "__main__":
    main()
