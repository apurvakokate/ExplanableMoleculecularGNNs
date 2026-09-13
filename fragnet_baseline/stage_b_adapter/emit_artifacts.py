"""Stage B (part 2) — turn FragNet's neutral export into the SAME per-run artifact set every
other method produces, by REUSING analysis/evaluate.py's own writers + metric helpers. Runs in
the l2xgnn env. No live model: FragNet is ante-hoc, so its per-node attention (score/GT-ROC) and
its OWN per-motif impact both come precomputed from Stage A.

Neutral export schema (Stage A → here), per fold file (keyed by our split-local index):
  {split: {str(src_idx): {"atts":[...per atom, our order...], "atom_syms":[...], "pred": float,
                          "own_impact": {motif_id(str): float_per_graph}, "n_atoms": int}}}

We reuse, from evaluate.py:
  score_cache_from_atts, instance_corr, grouped_corr_variants, gtroc_all, _keep_fn,
  _w_importance, _w_impact, _w_pergraph, _w_grouped, _w_instance, _w_global, _w_summary_splits
so FragNet's numbers are computed by the exact same code path as MoSE/MotifSAT/GSAT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from align_and_aggregate import (
    _evaluate_module, _np1, align, kept_set, load_our_graphs,
)

METHOD = "fragnet"


def _pred_metrics(preds: Dict[int, float], graphs, task_type: str) -> dict:
    """AUC (classification) or RMSE/MAE (regression) from FragNet's per-graph predictions vs
    our graph labels. Model-free — FragNet already produced the predictions in Stage A."""
    from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
    nan = float("nan")
    ys, ps = [], []
    for gi, g in enumerate(graphs):
        if gi not in preds:
            continue
        y = getattr(g, "y", None)
        if y is None:
            raise ValueError(f"graph {gi} has no label y — cannot score predictions")
        ys.append(float(_np1(y)[0])); ps.append(float(preds[gi]))
    if not ys:
        return {"auc": nan, "rmse": nan, "mae": nan}
    ys, ps = np.asarray(ys), np.asarray(ps)
    if task_type == "Regression":
        rmse = float(np.sqrt(mean_squared_error(ys, ps)))
        return {"auc": nan, "rmse": rmse, "mae": float(mean_absolute_error(ys, ps))}
    # BinaryClass — AUC needs both classes present
    auc = float(roc_auc_score(ys, ps)) if len(set(ys.tolist())) > 1 else nan
    return {"auc": auc, "rmse": nan, "mae": nan}


def emit(dataset: str, fold: int, vocab: str, unk: str,
         data_root: str, processed_root: str, neutral_path: str, dest_root: str,
         regime: str = "source", planted_root: str = None, rule_id: str = None,
         vocab_root: str = None) -> dict:
    ev = _evaluate_module()
    split_lists, gt, vocab_obj, dmeta, task_type = load_our_graphs(
        dataset, fold, vocab, data_root, processed_root, regime=regime,
        planted_root=planted_root, rule_id=rule_id, vocab_root=vocab_root)

    neutral = json.loads(Path(neutral_path).read_text())
    att_by_split = align(neutral, split_lists)          # {split: {gi: [N] atts}} — asserts alignment

    kept = kept_set(dataset, fold, vocab, data_root, vocab_root) if unk == "exclude" else None
    keep_fn = ev._keep_fn(kept, unk)
    do_gtroc = regime in ("source", "planted")

    rows_by_split, inst_by_split, summary, pergraph = {}, {}, {}, {}
    for s, sl in split_lists.items():
        if not sl:
            continue
        rec = neutral.get(s) or {}
        # neutral is keyed by str(src_idx) == our split-local graph index gi (align() verified it)
        ic_cache: Dict[int, Dict[int, float]] = {}
        preds: Dict[int, float] = {}
        for idx_s, r in rec.items():
            gi = int(idx_s)
            preds[gi] = float(r["pred"])
            for mid_s, val in (r.get("own_impact") or {}).items():
                ic_cache.setdefault(int(mid_s), {})[gi] = float(val)

        sc_cache = ev.score_cache_from_atts(att_by_split.get(s, {}), sl)   # {mid:{gi: mean att}}

        # grouped rows: per motif, score = mean att over graphs, impact = mean OWN impact over graphs
        rows = []
        motif_list = getattr(vocab_obj, "motif_list", [])
        for mid in sorted(set(sc_cache) & set(ic_cache)):
            if unk == "exclude" and kept is not None and mid not in kept:
                continue
            sc_vals = list(sc_cache[mid].values())
            im_vals = list(ic_cache[mid].values())
            if not sc_vals or not im_vals:
                continue
            rows.append(dict(
                motif_id=int(mid), score=float(np.mean(sc_vals)),
                impact=float(np.mean(im_vals)), support=len(im_vals),
                motif_smarts=(str(motif_list[mid]) if mid < len(motif_list) else "")))
        rows_by_split[s] = rows
        inst_by_split[s] = ev.instance_corr(sc_cache, ic_cache, kept, unk)
        pergraph[s] = (sc_cache, ic_cache)
        summary[s] = _pred_metrics(preds, sl, task_type)

    if do_gtroc:
        for s, block in ev._gtroc_summary(att_by_split, gt, split_lists, keep_fn, unk).items():
            summary.setdefault(s, {}).update(block)

    # dest path (POC scratch): <dest_root>/unk-<unk>/<rbrics|rbrics_filter>/fold<f>/
    vocab_seg = (vocab + "_filter") if unk == "exclude" else vocab
    dest = Path(dest_root) / f"unk-{unk}" / vocab_seg / f"fold{int(fold)}"
    dest.mkdir(parents=True, exist_ok=True)

    # write the per-node atts (explainer_importances.json) in the SAME schema read_saved_atts expects
    (dest / "explainer_importances.json").write_text(json.dumps({"importances_by_split": {
        s: {METHOD: {int(gi): a.tolist() for gi, a in att_by_split.get(s, {}).items()}}
        for s in split_lists}}))

    for s in split_lists:
        ev._w_importance(dest, METHOD, s, rows_by_split.get(s, []))
        ev._w_impact(dest, METHOD, s, rows_by_split.get(s, []))
        if s in pergraph:
            ev._w_pergraph(dest, METHOD, s, pergraph[s][0], pergraph[s][1], kept, unk)
    ev._w_grouped(dest, METHOD, rows_by_split)
    ev._w_instance(dest, METHOD, inst_by_split)
    ev._w_global(dest, METHOD, rows_by_split)
    ev._w_summary_splits(dest, METHOD, summary)

    g_all = ev.grouped_corr_variants([dict(r) for s in split_lists for r in rows_by_split.get(s, [])])
    tsum = summary.get("test", {})
    rollup = dict(dataset=dataset, method=METHOD, fold=int(fold), vocab=vocab, unk=unk,
                  grouped_pearson_u=g_all.get("pearson_u_exclunk"),
                  n_motifs=g_all.get("n_motifs_exclunk"),
                  gtroc_global=tsum.get("global_gt_roc_node_auc_mean", float("nan")),
                  gtroc_instance=tsum.get("instance_gt_roc_node_auc_mean", float("nan")),
                  pred_auc=tsum.get("auc", float("nan")))
    print(f"[emit] {dataset} fold{fold} unk={unk} -> {dest}")
    print(f"       n_motifs={rollup['n_motifs']} grouped_pearson_u={rollup['grouped_pearson_u']} "
          f"gtroc_instance={rollup['gtroc_instance']} pred_auc={rollup['pred_auc']}")
    return rollup


def _main():
    ap = argparse.ArgumentParser(description="Stage B: FragNet neutral export -> artifact set")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--unk", required=True, choices=["include", "exclude"])
    ap.add_argument("--regime", default="source", choices=["source", "none", "planted"])
    ap.add_argument("--vocab_root", default=None, help="load_vocab root (e.g. .../vocab_final_v2)")
    ap.add_argument("--planted_root", default=None, help="planted regime: planted_v2 root")
    ap.add_argument("--rule_id", default=None, help="planted regime: e.g. dnf_k2_r1")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--processed_root", required=True)
    ap.add_argument("--neutral", required=True, help="Stage A fragnet_neutral.json for this fold")
    ap.add_argument("--dest_root", required=True)
    args = ap.parse_args()
    emit(args.dataset, args.fold, args.vocab, args.unk,
         args.data_root, args.processed_root, args.neutral, args.dest_root, regime=args.regime,
         planted_root=args.planted_root, rule_id=args.rule_id, vocab_root=args.vocab_root)


if __name__ == "__main__":
    _main()
