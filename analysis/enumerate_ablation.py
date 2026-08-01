#!/usr/bin/env python3
"""enumerate_ablation.py — expand the LOCKED source-GT ablation spec into one CSV row per run.

Each row is a fully-explicit config the config-worker reads and routes. No behavior here —
pure enumeration so every combination can be inspected / smoke-tested.

Locked spec:
  datasets x regime (7): Benzene=normal; {Alkane,Fluoride,mutag} x {normal, m1}
  folds 5, backbones 5
  architecture (OFAT, 5 distinct): (norm, readout, node_encoder)
     1 none/add/onehot (default) | 2 l2/add/onehot | 3 layernorm/add/onehot
     4 none/mean/onehot          | 5 none/add/linear
  per base (ds,regime,fold,backbone,arch) -> the runs:
     vanilla x1 (celltype vtrain)      -> feeds post-hoc
     mose    x6 (frag{rbrics,rdkit} x {filt/learn, filt/fixed, unfilt/fixed})
     motifsat x2 (frag{rbrics,rdkit})
     gsat    x1 (frag-agnostic training; motif eval per frag inside the run)
     posthoc x2 (frag{rbrics,rdkit}, UNFILTERED) on the vanilla checkpoint
"""
import csv, sys

DATASETS_REGIME = [
    ("Benzene_Verified_GT",           ["normal"]),
    ("Alkane_Carbonyl_Verified_GT",   ["normal", "m1"]),
    ("Fluoride_Carbonyl_Verified_GT", ["normal", "m1"]),
    ("mutag",                          ["normal", "m1"]),
]
FOLDS = [0, 1, 2, 3, 4]
BACKBONES = ["GIN", "GCN", "GAT", "SAGE", "PNA"]
# arch_id -> (norm, readout, node_encoder)
ARCH = {
    1: ("none",      "add",  "onehot"),
    2: ("l2",        "add",  "onehot"),
    3: ("layernorm", "add",  "onehot"),
    4: ("none",      "mean", "onehot"),
    5: ("none",      "add",  "linear"),
}
FRAGS = ["rbrics", "rdkit_fg_first"]
# MoSE (filter, unk) triples (Unfiltered-learn dropped: no training UNK)
MOSE_FILTER_UNK = [("filtered", "learn"), ("filtered", "fixed"), ("unfiltered", "fixed")]

COLS = ["run_id", "category", "celltype", "method", "dataset", "fold", "backbone",
        "regime", "arch_id", "norm", "readout", "node_encoder",
        "fragmentation", "filter", "unk"]

def main(out="final_local/ablation/run_configs.csv"):
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    rid = 0
    def add(**k):
        nonlocal rid
        rid += 1
        rows.append({"run_id": f"r{rid:06d}", **k})
    for ds, regimes in DATASETS_REGIME:
        for regime in regimes:
            for fold in FOLDS:
                for bb in BACKBONES:
                    for aid, (norm, readout, enc) in ARCH.items():
                        base = dict(dataset=ds, fold=fold, backbone=bb, regime=regime,
                                    arch_id=aid, norm=norm, readout=readout, node_encoder=enc)
                        # vanilla (feeds post-hoc)
                        add(category="train", celltype="vtrain", method="vanilla",
                            fragmentation="NA", filter="NA", unk="NA", **base)
                        # mose x6
                        for frag in FRAGS:
                            for filt, unk in MOSE_FILTER_UNK:
                                add(category="train", celltype="mose", method="mose",
                                    fragmentation=frag, filter=filt, unk=unk, **base)
                        # motifsat x2
                        for frag in FRAGS:
                            add(category="train", celltype="motifsat", method="motifsat",
                                fragmentation=frag, filter="NA", unk="NA", **base)
                        # gsat x1 (frag-agnostic training)
                        add(category="train", celltype="gsat", method="gsat",
                            fragmentation="NA", filter="NA", unk="NA", **base)
                        # post-hoc x2 (unfiltered), one invocation per frag → 4 explainers each
                        for frag in FRAGS:
                            add(category="posthoc", celltype="posthoc", method="posthoc_4",
                                fragmentation=frag, filter="unfiltered", unk="NA", **base)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    # counts
    from collections import Counter
    by_ct = Counter(r["celltype"] for r in rows)
    print(f"wrote {out}  ({len(rows)} rows)")
    print("by celltype:", dict(by_ct))
    print(f"base configs = {len(rows and set((r['dataset'],r['regime'],r['fold'],r['backbone'],r['arch_id']) for r in rows))}")
    print(f"train rows   = {sum(1 for r in rows if r['category']=='train')}")
    print(f"posthoc rows = {sum(1 for r in rows if r['category']=='posthoc')}")

if __name__ == "__main__":
    main(*sys.argv[1:])
