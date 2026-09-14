"""Aggregate the per-unit fragnet_frag_perlayer_metrics.csv files of a base_runs tree into one
rollup.csv, and print a coverage report. Pure stdlib (csv) — runs in any env.

Walks   <base>/<regime>/<dataset>/eval/unk-{include,exclude}/rbrics[_filter]/fold<k>/fragnet_frag_perlayer_metrics.csv
Writes  <base>/rollup.csv    (every row from every unit, union of columns, source_path appended)

Each metrics row already self-identifies (dataset, fold, vocab, unk, regime, task_type, method, split,
layer, …), so the rollup is a faithful concatenation; the coverage report flags missing (dataset, fold,
unk) cells against the expected 5 folds × {include, exclude}.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

METRICS_NAME = "fragnet_frag_perlayer_metrics.csv"
EXPECT_FOLDS = 5
EXPECT_UNK = ("include", "exclude")


def rollup(base: str, out_path: str = None) -> None:
    base = Path(base)
    files = sorted(base.rglob(METRICS_NAME))
    if not files:
        raise FileNotFoundError(f"no {METRICS_NAME} under {base} — nothing to roll up")

    all_rows, cols = [], []
    seen = defaultdict(set)                       # (regime,dataset) -> {(fold,unk)}
    for f in files:
        with open(f, newline="") as fh:
            r = list(csv.DictReader(fh))
        for row in r:
            for k in row:
                if k not in cols:
                    cols.append(k)
            row["source_path"] = str(f.relative_to(base))
            all_rows.append(row)
        if r:
            d, reg = r[0].get("dataset", "?"), r[0].get("regime", "?")
            fold, unk = r[0].get("fold", "?"), r[0].get("unk", "?")
            seen[(reg, d)].add((str(fold), str(unk)))
    if "source_path" not in cols:
        cols.append("source_path")

    out = Path(out_path) if out_path else (base / "rollup.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    print(f"[rollup] {len(all_rows)} rows from {len(files)} metric files -> {out}")

    # featurization drops: n_dropped is per (regime,dataset,fold,split), repeated across layers+unk -> dedup
    seen_cell, drops = set(), defaultdict(int)
    for row in all_rows:
        key = (row.get("regime"), row.get("dataset"), row.get("fold"), row.get("split"))
        if key in seen_cell:
            continue
        seen_cell.add(key)
        nd = row.get("n_dropped")
        try:
            drops[(row.get("regime"), row.get("dataset"))] += int(nd) if nd not in (None, "") else 0
        except (ValueError, TypeError):
            pass
    any_drop = {k: v for k, v in drops.items() if v}
    if any_drop:
        print("[rollup] featurization drops (summed over folds × splits):")
        for (reg, d), n in sorted(any_drop.items()):
            print(f"    {reg:>6} / {d:<32} dropped {n}")
    else:
        print("[rollup] featurization drops: none")

    print("[rollup] coverage (expected 5 folds × 2 unk = 10 cells per dataset):")
    ok = True
    for (reg, d) in sorted(seen):
        have = seen[(reg, d)]
        want = {(str(fo), u) for fo in range(EXPECT_FOLDS) for u in EXPECT_UNK}
        missing = sorted(want - have)
        flag = "OK" if not missing else f"MISSING {len(missing)}: {missing}"
        if missing:
            ok = False
        print(f"    {reg:>6} / {d:<32} {len(have)}/10  {flag}")
    print("[rollup] ALL COMPLETE" if ok else "[rollup] INCOMPLETE — see MISSING above")


def _main():
    ap = argparse.ArgumentParser(description="Aggregate fragnet base_runs per-unit metrics into rollup.csv")
    ap.add_argument("--base", required=True, help=".../fragnet/base_runs")
    ap.add_argument("--out", default=None, help="default: <base>/rollup.csv")
    args = ap.parse_args()
    rollup(args.base, args.out)


if __name__ == "__main__":
    _main()
