#!/usr/bin/env python3
"""eval_antehoc_dnf_gtroc.py — EVAL-ONLY (no retraining).

Fills the known gap: the ante-hoc trainers (MOSE / GSAT / MotifSAT) run EvalPipeline, which
never calls compute_dnf_gt_roc, so their clause-based **Instance GT-ROC** was empty for the
planted-GT (dnf_*) tiers. This script reloads each finished ante-hoc dnf checkpoint, loads the
DNF clause-GT test set (graphs carrying node_label_clauses), computes compute_dnf_gt_roc using
the model's NATIVE node attention, and writes the metric into that run's summary.json — under
the exact unprefixed key build_workbook.py reads for Instance GTROC.

Writes (does NOT overwrite the whole-rule gt_roc_node_auc_mean = Grouped GTROC):
    instance_gt_roc_node_auc_mean <- dnf['instance_auc_mean'] # Instance GTROC (ante-hoc column)
    dnf_instance_auc_mean / dnf_global_auc_mean / dnf_n_graphs   # provenance

Only touches ante-hoc runs with use_gt=True and gt_tier starting 'dnf' (source_gt has no
clauses; its whole-rule GT-ROC already flows through). Re-run harvest_v2 + build_workbook after.

Reuses: analysis.probe_vs_vanilla._load_model_and_data (model reload) and the trainers'
apply_gt_loaders call (MOSE-GNN/run.py:124, MotifSAT/run.py:279) for the clause-GT test set.

  python analysis/eval_antehoc_dnf_gtroc.py --out_root final_v2 \
         --data_root <DATA_ROOT> --vocab_root vocab_final_v2 [--dry_run] [--dataset Mutagenicity]
"""
import argparse, json, sys
from pathlib import Path
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from analysis.probe_vs_vanilla import _load_model_and_data            # model reloader (traced)
from SharedModules.evaluation.motif_eval import compute_dnf_gt_roc, model_node_att_fn

ARCHIVE = ("_archive", "_trash", "_old")
FAMILIES = ("mose", "base_gsat", "motifsat")


def _iter_dnf_runs(out_root, only_ds):
    """Yield run dirs of finished ante-hoc dnf checkpoints (skip archives)."""
    for fam in FAMILIES:
        base = Path(out_root) / fam
        if not base.exists():
            continue
        for ck in base.rglob("best_model.pt"):
            if any(a in str(ck) for a in ARCHIVE):
                continue
            rd = ck.parent
            sj = rd / "summary.json"
            if not sj.exists():
                continue
            try:
                meta = json.load(open(sj, encoding="utf-8"))
            except Exception:
                continue
            tier = str(meta.get("gt_tier", "") or "")
            if not (meta.get("use_gt") and tier.startswith("dnf")):
                continue                                  # only planted-GT dnf runs carry clauses
            if only_ds and meta.get("dataset") not in only_ds:
                continue
            yield rd, meta


def _load_clause_gt_test(meta, data_root, vocab_root, batch_size=128):
    """Load the dnf clause-GT test set (node_label_clauses) — mirrors MOSE-GNN/run.py:124."""
    from SharedModules.data.vocab import load_vocab
    from SharedModules.data.loader import get_loaders, apply_gt_loaders, TASK_TYPE
    from SharedModules.data.dataset_routing import default_processed_base, variant_processed_root
    ds = meta["dataset"]; fold = int(meta.get("fold", 0))
    variant = meta.get("vocab_variant")
    droot = meta.get("data_root", data_root)
    vocab = load_vocab(vocab_root, ds, variant)
    task_type = TASK_TYPE.get(ds, "BinaryClass")
    proc_root = meta.get("processed_root") or variant_processed_root(
        default_processed_base(droot, None), variant)
    loaders, test_ds, dmeta = get_loaders(
        dataset=ds, data_root=droot, fold=fold, vocab=vocab,
        processed_root=proc_root, batch_size=batch_size,
        normalize=(task_type == "Regression"))
    gt_tier = meta.get("gt_tier")
    # The phase-4 gt_cache is keyed by the BASE variant — relabel dirs live under
    # e.g. gt_cache/<ds>/fold<k>/rdkit_fg_first/relabel_<tier>/, NOT the _filter name.
    # So strip any _filter suffix for the clause-GT lookup (matches where apply_gt wrote it).
    gt_vocab = (meta.get("gt_vocab_variant") or variant).replace("_filter", "")
    loaders, test_ds = apply_gt_loaders(
        loaders, test_ds, gt_cache=meta["gt_cache"], dataset=ds, fold=fold,
        vocab_variant=variant, batch_size=batch_size, gt_vocab_variant=gt_vocab,
        gt_relabel_dir=(f"relabel_{gt_tier}" if gt_tier else None),
        refresh_vocab=vocab if gt_vocab != variant else None,
        fold_motif_lookup=getattr(dmeta, "motif_lookup", None),
        apply_threshold=getattr(dmeta, "threshold_pct", None) is not None)
    return list(loaders["test"].dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True,
                   help="results tree (no default: `final_v2` is the RETIRED greedy-engine tree whose gt_roc_node_auc_mean is the OLD whole-rule metric — defaulting to it silently merges pre/post-2026-08 numbers)")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--vocab_root", required=True)
    ap.add_argument("--dataset", nargs="*", default=None, help="restrict to these datasets")
    ap.add_argument("--dry_run", action="store_true", help="compute + print, do NOT write summary.json")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    only_ds = set(a.dataset) if a.dataset else None

    n_ok = n_skip = 0
    for rd, meta in _iter_dnf_runs(a.out_root, only_ds):
        tag = f"{meta.get('dataset')}/{meta.get('vocab_variant')}/{meta.get('backbone')}/{meta.get('gt_tier')}"
        model, _plain, status = _load_model_and_data(rd, a.data_root, a.vocab_root, device)
        if status != "ok":
            print(f"[skip] {tag}: reload {status}"); n_skip += 1; continue
        try:
            test_list = _load_clause_gt_test(meta, a.data_root, a.vocab_root)
        except Exception as e:
            print(f"[skip] {tag}: clause-GT load failed: {e}"); n_skip += 1; continue
        try:
            dnf = compute_dnf_gt_roc(model, test_list, device, model_node_att_fn(model, device))
        except Exception as e:
            print(f"[skip] {tag}: compute_dnf_gt_roc err: {e}"); n_skip += 1; continue
        if not dnf or dnf.get("n_graphs", 0) == 0:
            print(f"[skip] {tag}: n_graphs=0 (no fired clauses on test)"); n_skip += 1; continue
        upd = {
            "instance_gt_roc_node_auc_mean": dnf["instance_auc_mean"],   # -> Instance GTROC
            "dnf_instance_auc_mean": dnf["instance_auc_mean"],
            "dnf_global_auc_mean": dnf["global_auc_mean"],
            "dnf_n_graphs": dnf["n_graphs"],
        }
        print(f"[{'dry' if a.dry_run else 'ok '}] {tag}: instance={dnf['instance_auc_mean']:.3f} "
              f"global={dnf['global_auc_mean']:.3f} n={dnf['n_graphs']}")
        if not a.dry_run:
            meta.update(upd)
            with open(rd / "summary.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        n_ok += 1
    print(f"\nDONE: {'(dry-run) ' if a.dry_run else ''}computed {n_ok}, skipped {n_skip}. "
          f"Re-run harvest_v2 + build_workbook to surface Instance GTROC.")


if __name__ == "__main__":
    main()
