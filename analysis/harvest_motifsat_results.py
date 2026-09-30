#!/usr/bin/env python3
"""harvest_motifsat_results.py — artifact-only harvest of the MotifSAT ablation.

Walks BOTH campaigns under a deliverables root:
  base_runs/<preset>/<ds>/fold<f>/<bb>/           (Stages 1-3: base GSAT / losses / readout)
  gnn1_ablation_runs/<config>/<ds>/fold<f>/<bb>/  (Stage 4: motif-level GNN, 24 configs)

For every (config x dataset x backbone) it reads the per-run artifacts and
aggregates over the 5 folds (mean + std). NO model forward pass — all metrics
were written at train time by run.py --per_split_eval:
  - summary_splits.json[<stem>][test] -> auc, rmse, instance_gt_roc_node_auc_mean
  - <stem>_grouped_corr_pooled_alltest.csv -> grouped Pearson (unweighted / support-weighted)

Method stem on disk: 'gsat' for base_gsat* and all motif_emb runs; 'motifsat'
for the loss/readout presets (matches run.py `_method` / native_complete.py).

Emits one CSV to --out (default stdout):
  campaign,config,dataset,backbone,n,auc_m,auc_s,rmse_m,rmse_s,
  gtroc_m,gtroc_s,pearU_m,pearU_s,pearW_m,pearW_s

Read-only. Run on the HPC (needs the deliverables filesystem).
Usage:
  python3 analysis/harvest_motifsat_results.py \
      --deliv /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/motifsat_paper_deliverables_v1 \
      --out harvest_all.csv
"""
import argparse
import collections
import csv
import glob
import json
import os
import statistics
import sys

# preset -> method stem written on disk (single source of truth: native_complete.py)
STEM_BASE = {
    "base_gsat": "gsat", "base_gsat_noanneal": "gsat",
    "intra_consistency": "motifsat", "inter_loss": "motifsat", "intra_inter_loss": "motifsat",
    "readout": "motifsat", "readout_inter": "motifsat",
    "readout_layernorm": "motifsat", "readout_nonorm": "motifsat",
}
# metrics pulled from summary_splits.json[<stem>][test]
SUMMARY_METRICS = ("auc", "rmse", "instance_gt_roc_node_auc_mean")


def grouped_pearson(csv_path):
    """(unweighted, support-weighted) grouped Pearson from a *_grouped_corr_pooled_alltest.csv.
    Prefers the exclude-UNK variant; for full-vocab motif_emb incl==excl anyway."""
    try:
        rows = list(csv.DictReader(open(csv_path)))
        if not rows:
            return (None, None)
        r = rows[0]
        pu = r.get("pearson_u_exclunk") or r.get("pearson_u_inclunk")
        pw = r.get("pearson_w_exclunk") or r.get("pearson_w_inclunk")
        return (float(pu) if pu not in (None, "") else None,
                float(pw) if pw not in (None, "") else None)
    except Exception:
        return (None, None)


def collect(data, campaign, config, stem, ds, bb, cell_dir):
    """Append this cell's per-fold metric values into data[(campaign,config,ds,bb)]."""
    try:
        test = json.load(open(os.path.join(cell_dir, "summary_splits.json")))[stem]["test"]
    except Exception:
        return
    key = (campaign, config, ds, bb)
    for m in SUMMARY_METRICS:
        v = test.get(m)
        if isinstance(v, (int, float)):
            data[key][m].append(v)
    pu, pw = grouped_pearson(os.path.join(cell_dir, stem + "_grouped_corr_pooled_alltest.csv"))
    if pu is not None:
        data[key]["pu"].append(pu)
    if pw is not None:
        data[key]["pw"].append(pw)


def mean_std(vals):
    if not vals:
        return ("", "")
    return (round(statistics.mean(vals), 4),
            round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deliv", default="/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/"
                                        "motifsat_paper_deliverables_v1",
                    help="deliverables root holding base_runs/ and gnn1_ablation_runs/")
    ap.add_argument("--out", default="-", help="output CSV path ('-' = stdout)")
    a = ap.parse_args()

    base = os.path.join(a.deliv, "base_runs")
    gnn1 = os.path.join(a.deliv, "gnn1_ablation_runs")
    data = collections.defaultdict(lambda: collections.defaultdict(list))

    for p in glob.glob(base + "/*/*/fold*/*/summary_splits.json"):
        rel = p.split("/base_runs/")[1].split("/")      # preset/ds/foldX/bb/summary_splits.json
        preset, ds, bb = rel[0], rel[1], rel[3]
        collect(data, "base", preset, STEM_BASE.get(preset, "gsat"), ds, bb, os.path.dirname(p))

    for p in glob.glob(gnn1 + "/*/*/fold*/*/summary_splits.json"):
        rel = p.split("/gnn1_ablation_runs/")[1].split("/")
        cfg, ds, bb = rel[0], rel[1], rel[3]
        collect(data, "motif_emb", cfg, "gsat", ds, bb, os.path.dirname(p))

    fh = sys.stdout if a.out == "-" else open(a.out, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["campaign", "config", "dataset", "backbone", "n",
                "auc_m", "auc_s", "rmse_m", "rmse_s", "gtroc_m", "gtroc_s",
                "pearU_m", "pearU_s", "pearW_m", "pearW_s"])
    for key in sorted(data):
        camp, cfg, ds, bb = key
        d = data[key]
        n = max(len(d.get("auc", [])), len(d.get("rmse", [])), len(d.get("pu", [])), 1)
        am, asd = mean_std(d.get("auc", []))
        rm, rsd = mean_std(d.get("rmse", []))
        gm, gsd = mean_std(d.get("instance_gt_roc_node_auc_mean", []))
        pum, pus = mean_std(d.get("pu", []))
        pwm, pws = mean_std(d.get("pw", []))
        w.writerow([camp, cfg, ds, bb, n, am, asd, rm, rsd, gm, gsd, pum, pus, pwm, pws])
    if fh is not sys.stdout:
        fh.close()
        sys.stderr.write("wrote %s (%d config-dataset-backbone rows)\n" % (a.out, len(data)))


if __name__ == "__main__":
    main()
