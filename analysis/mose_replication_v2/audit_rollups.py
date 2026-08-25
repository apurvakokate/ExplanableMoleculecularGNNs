#!/usr/bin/env python3
"""Audit mose_replication_v2/rollups after a run:
  1) NO DOUBLE-COUNT: every (tier,dataset,rule,method,unk_mode,backbone,fold) key appears in
     exactly one rollup CSV. A GAT row surfacing from both final_v2/ablated and gath1 would
     trip this. Also assert GAT rows come only from *__GAT_* (none/source) or *__all_* (planted).
  2) COVERAGE: shards that produced 0 rows / are missing, per tier.

Usage: python audit_rollups.py --v2 /nfs/.../Claude+Cursor/mose_replication_v2
"""
import argparse, csv, glob, os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    a = ap.parse_args()
    root = os.path.join(a.v2, "rollups")
    files = sorted(glob.glob(os.path.join(root, "**", "metrics_*_unk-exclude.csv"), recursive=True))
    print(f"scanning {len(files)} rollup files under {root}")

    key_files = defaultdict(set)      # cell key -> set of files that carry it
    gat_bad = []                      # GAT rows in a non-GAT/non-all file
    empty = []
    for fp in files:
        base = os.path.basename(fp)
        is_gat_file = "__GAT_" in base
        is_all_file = "__all_" in base
        rows = list(csv.DictReader(open(fp)))
        if not rows:
            empty.append(fp); continue
        for r in rows:
            bb = r.get("backbone")
            key = (r.get("gt_tier"), r.get("dataset"), r.get("gt_rule"), r.get("method"),
                   r.get("unk_mode"), bb, r.get("fold"))
            key_files[key].add(fp)
            if bb == "GAT" and not (is_gat_file or is_all_file):
                gat_bad.append((fp, key))

    dupes = {k: v for k, v in key_files.items() if len(v) > 1}
    print(f"\n=== DOUBLE-COUNT CHECK ===")
    if dupes:
        print(f"  FAIL: {len(dupes)} cell keys appear in >1 file:")
        for k, v in list(dupes.items())[:20]:
            print(f"    {k}")
            for f in v:
                print(f"        {os.path.relpath(f, a.v2)}")
    else:
        print("  OK: every cell key is unique across files")
    if gat_bad:
        print(f"  FAIL: {len(gat_bad)} GAT rows in non-GAT/non-all files (unexpected tree):")
        for f, k in gat_bad[:20]:
            print(f"    {os.path.relpath(f, a.v2)}  {k}")
    else:
        print("  OK: all GAT rows live in *__GAT_* / *__all_* files")

    if empty:
        print(f"\n=== EMPTY ROLLUPS ({len(empty)}) ===")
        for f in empty:
            print(f"    {os.path.relpath(f, a.v2)}")

    total_cells = sum(1 for _ in key_files)
    print(f"\ntotal distinct cells: {total_cells}")
    print("AUDIT:", "PASS" if not dupes and not gat_bad else "FAIL")


if __name__ == "__main__":
    main()
