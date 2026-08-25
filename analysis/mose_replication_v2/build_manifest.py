#!/usr/bin/env python3
"""Merge the per-worker provenance logs into one manifest:
  mose_replication_v2/_deploy/manifest.csv

One row per shard: source artifact folder(s) -> destination rollup + artifacts in the new tree,
with completion status. Dedup by dest_rollup keeping the latest timestamp; re-verifies each
dest_rollup still exists on disk (status_now = PRESENT/MISSING).

Usage: python build_manifest.py --v2 /nfs/.../Claude+Cursor/mose_replication_v2
"""
import argparse, csv, glob, os

COLS = ["ts", "tier", "dataset", "rule", "method", "unk_mode", "bbgroup", "status", "rows",
        "source_ckpt", "source_artifacts", "dest_rollup", "dest_artifacts"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    a = ap.parse_args()
    md = os.path.join(a.v2, "_deploy", "manifest")
    parts = sorted(glob.glob(os.path.join(md, "*.tsv")))
    latest = {}                                  # dest_rollup -> row (latest ts)
    for fp in parts:
        for r in csv.DictReader(open(fp), delimiter="\t"):
            k = r.get("dest_rollup")
            if not k:
                continue
            if k not in latest or r.get("ts", "") > latest[k].get("ts", ""):
                latest[k] = r

    out = os.path.join(a.v2, "_deploy", "manifest.csv")
    n_ok = n_present = 0
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS + ["status_now"])
        w.writeheader()
        for k in sorted(latest):
            r = latest[k]
            rel = os.path.relpath(k, a.v2) if os.path.isabs(k) else k
            present = os.path.exists(k) and os.path.getsize(k) > 0
            row = {c: r.get(c, "") for c in COLS}
            row["dest_rollup"] = rel
            row["status_now"] = "PRESENT" if present else "MISSING"
            w.writerow(row)
            n_ok += (r.get("status") == "OK")
            n_present += present
    print(f"wrote {out}")
    print(f"  shards logged: {len(latest)}  |  status=OK: {n_ok}  |  dest present now: {n_present}")
    print(f"  (source_ckpt / source_artifacts columns hold the ORIGINAL run folders)")


if __name__ == "__main__":
    main()
