#!/usr/bin/env python3
"""Emit the mose_replication_v2 re-eval worklist (one TSV row per shard).

A shard = one evaluate.py invocation (loops its folds x backbones internally), OR a
node-mask probe (one per dataset). The mapping below is the SINGLE SOURCE OF TRUTH for
which tree feeds which cell — in particular it guarantees GAT is sourced ONLY from
final_v2_gath1 (none/source) or planted_v2 (planted), never double-counted from
final_v2 (fixed) or ablated_completely_v1 (learnable).

Columns (tab-separated):
  tier  dataset  rule  method  unk_mode  backbones  bbgroup  ckpt_rel  art_rel  vocab

  tier      none | source | planted | probe
  rule      planted rule dir name, else '-'
  method    evaluate.py --method (mose/gsat/mage/motif_occlusion/pgexplainer/gnnexplainer);
            mose_U is method=mose with unk_mode=learnable_shared
  unk_mode  fixed | learnable_shared | '' (any)      -> evaluate.py --unk_mode
  backbones space-joined --backbone list, or '' (planted: no filter)
  bbgroup   nonGAT | GAT | all           -> used in the dest filename so GAT rows are isolated
  ckpt_rel  path under $base for --ckpt_root
  art_rel   path under $base for --artifacts_root, or '' (native)
  vocab     rbrics_filter (mose/mose_U) | rbrics (gsat/posthoc)

Run on the HPC (needs $base to enumerate planted rules):
  python make_worklist.py --base /nfs/.../Claude+Cursor --out $V2/_deploy/worklist.tsv
"""
import argparse, glob, os

NONE_DS = ["BBBP", "hERG", "Mutagenicity", "esol", "Lipophilicity",
           "Benzene_Verified_GT", "Alkane_Carbonyl_Verified_GT", "Fluoride_Carbonyl_Verified_GT"]
SOURCE_DS = ["Benzene_Verified_GT", "Alkane_Carbonyl_Verified_GT", "Fluoride_Carbonyl_Verified_GT"]
PLANTED_DS = ["BBBP", "hERG", "Mutagenicity"]

POSTHOC = ("mage", "motif_occlusion", "pgexplainer", "gnnexplainer")
# logical explainer -> (evaluate.py method, unk_mode)
NONSRC_METHODS = [
    ("mose",   "fixed"),
    ("mose_U", "learnable_shared"),
    ("gsat",   ""),
    ("mage",   ""), ("motif_occlusion", ""), ("pgexplainer", ""), ("gnnexplainer", ""),
]
PLANTED_METHODS = [("mose", "fixed"), ("gsat", ""),
                   ("mage", ""), ("motif_occlusion", ""), ("pgexplainer", ""), ("gnnexplainer", "")]

NONGAT = "GCN GIN PNA SAGE"


def eval_method(logical):
    return "mose" if logical == "mose_U" else logical


def vocab_for(logical):
    return "rbrics_filter" if logical in ("mose", "mose_U") else "rbrics"


def ckpt_rel(logical, bbgroup):
    if bbgroup == "GAT":
        return "final_v2_gath1"                       # GAT (heads=1) — the ONLY GAT source
    if logical == "mose_U":
        return "ablated_completely_v1"                # non-GAT learnable lives only here
    return "final_v2"                                 # non-GAT fixed / gsat / posthoc


def art_rel(logical, bbgroup):
    if eval_method(logical) not in POSTHOC:
        return ""
    return "final_v2_gath1" if bbgroup == "GAT" else "posthoc_v1"


def rows_for_nonplanted(tier, datasets):
    for ds in datasets:
        for logical, unk_mode in NONSRC_METHODS:
            m = eval_method(logical)
            for bbgroup, bbs in (("nonGAT", NONGAT), ("GAT", "GAT")):
                yield [tier, ds, "-", m, unk_mode, bbs, bbgroup,
                       ckpt_rel(logical, bbgroup), art_rel(logical, bbgroup), vocab_for(logical)]


def rows_for_planted(base):
    for ds in PLANTED_DS:
        rule_dirs = sorted(d for d in glob.glob(os.path.join(base, "planted_v2", ds, "*"))
                           if os.path.isdir(d) and not os.path.basename(d).startswith("_"))
        if not rule_dirs:
            print(f"# WARNING: no planted rules found under planted_v2/{ds} (base present?)")
        for rd in rule_dirs:
            rule = os.path.basename(rd)
            for logical, unk_mode in PLANTED_METHODS:
                m = eval_method(logical)
                ck = f"planted_v2/{ds}/{rule}"
                ar = f"{ck}/posthoc_gtroc" if m in POSTHOC else ""
                yield ["planted", ds, rule, m, unk_mode, "", "all", ck, ar, vocab_for(logical)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Claude+Cursor base (for planted rule enum)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-planted", action="store_true")
    ap.add_argument("--no-probe", action="store_true")
    a = ap.parse_args()

    rows = []
    rows += list(rows_for_nonplanted("none", NONE_DS))
    rows += list(rows_for_nonplanted("source", SOURCE_DS))
    if not a.no_planted:
        rows += list(rows_for_planted(a.base))
    if not a.no_probe:
        for ds in NONE_DS:                             # node-mask probe: one shard per dataset
            rows.append(["probe", ds, "-", "probe", "", "", "-", "", "", ""])

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("# tier\tdataset\trule\tmethod\tunk_mode\tbackbones\tbbgroup\tckpt_rel\tart_rel\tvocab\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    n_by_tier = {}
    for r in rows:
        n_by_tier[r[0]] = n_by_tier.get(r[0], 0) + 1
    print(f"wrote {a.out}  shards={len(rows)}  by_tier={n_by_tier}")


if __name__ == "__main__":
    main()
