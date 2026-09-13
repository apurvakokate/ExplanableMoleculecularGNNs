"""Stage B (frag) — evaluate FragNet's PER-LAYER attention as importance, from export_frag_attention.py.

Runs in the l2xgnn env. Consumes fragnet_frag_neutral.json (per-layer atom + fragment attention +
frag_to_motif, keyed by our src_idx) and produces, PER LAYER (no cross-layer aggregation), the node-
and motif-level GT-ROC over ALL samples. Attention-only — no contribution/own-impact here.

Three GT-ROC columns per layer (source-GT regime):
  * atom_node_gtroc  — per-atom ATOM attention vs per-atom node_label. Uses evaluate.py's OWN
                       _per_graph_mean_auc, so it is the pipeline's exact node GT-ROC (comparable to
                       every other method).
  * frag_node_gtroc  — FRAGMENT attention broadcast to its atoms (motif score -> each atom), then the
                       SAME node GT-ROC. This is the fragment score judged by the pipeline's node
                       metric (directly comparable to a motif-level method's node GT-ROC).
  * frag_motif_gtroc — FRAGMENT attention at MOTIF granularity: per graph, AUC over motifs (positive =
                       motifs containing a GT atom) using the per-motif fragment score; per-graph mean.

Fragment attention is only meaningful when the model was finetuned with frag_type='custom' (so
FragNet's fragments == our rbrics motifs). frag_to_motif maps each fragment to its motif; repeated
motif types (several fragments) are aggregated to the type by mean.

Reuses evaluate.py via _evaluate_module(): _per_graph_mean_auc, _auc, _keep_fn — so the metric math is
identical to the rest of the pipeline. NO edit to evaluate.py.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from align_and_aggregate import _evaluate_module, _np1, our_node_symbols, load_our_graphs, kept_set

METHOD = "fragnet_frag"


def _align(neutral: dict, split_lists) -> dict:
    """{split: {gi: rec}} with the same element-sequence verification as align() — refuse misaligned."""
    out, problems = {}, []
    for split, sl in split_lists.items():
        by_idx = neutral.get(split) or {}
        out[split] = {}
        for gi, g in enumerate(sl):
            rec = by_idx.get(str(gi))
            if rec is None:
                problems.append(f"{split}[{gi}] no FragNet record (dropped in featurization?)")
                continue
            our_syms = our_node_symbols(g)
            fn_syms = list(rec.get("atom_syms") or [])
            if fn_syms != our_syms:
                j = next((k for k in range(min(len(fn_syms), len(our_syms)))
                          if fn_syms[k] != our_syms[k]), -1)
                problems.append(f"{split}[{gi}] ELEMENT MISMATCH at atom {j} — mapping untrustworthy")
                continue
            out[split][gi] = rec
    if problems:
        raise AssertionError("FragNet↔graph alignment failed:\n  " + "\n  ".join(problems[:20]))
    return out


def _motif_scores(rec: dict, layer: int) -> dict:
    """Per-motif fragment score for one layer: mean over the fragment instances of that motif."""
    f2m = {int(k): int(v) for k, v in rec["frag_to_motif"].items()}
    fa = rec["frag_att_by_layer"][str(layer)]
    agg = {}
    for fid, a in enumerate(fa):
        m = f2m.get(fid)
        if m is None:
            continue
        agg.setdefault(m, []).append(float(a))
    return {m: float(np.mean(v)) for m, v in agg.items()}


def _motif_gt(g) -> dict:
    """motif -> 1 if it contains any GT atom (node_label>0), else 0."""
    n2m = _np1(getattr(g, "nodes_to_motifs")).astype(int)
    nl = _np1(getattr(g, "node_label")).astype(float)
    gt = {}
    for i, m in enumerate(n2m):
        m = int(m)
        gt[m] = 1 if (gt.get(m, 0) or (nl[i] > 0)) else 0
    return gt


def _atom_att_by_i(aligned_split: dict, layer: int) -> dict:
    return {gi: np.asarray(rec["atom_att_by_layer"][str(layer)], dtype=float)
            for gi, rec in aligned_split.items()}


def _frag_broadcast_by_i(aligned_split: dict, gl, layer: int) -> dict:
    """Per-atom fragment score: each atom gets its motif's fragment score (broadcast)."""
    out = {}
    for gi, rec in aligned_split.items():
        g = gl[gi]
        ms = _motif_scores(rec, layer)
        n2m = _np1(getattr(g, "nodes_to_motifs")).astype(int)
        out[gi] = np.asarray([ms.get(int(m), np.nan) for m in n2m], dtype=float)
    return out


def _frag_motif_auc_mean(aligned_split: dict, gl, layer: int, keep_fn, ev) -> float:
    """Per-graph mean motif-level AUC: per-motif fragment score vs motif-GT, over kept motifs."""
    vals = []
    for gi, rec in aligned_split.items():
        g = gl[gi]
        ms = _motif_scores(rec, layer)
        mg = _motif_gt(g)
        keep = keep_fn(g)
        n2m = _np1(getattr(g, "nodes_to_motifs")).astype(int)
        motif_kept = {}
        for i, m in enumerate(n2m):
            m = int(m)
            motif_kept[m] = motif_kept.get(m, False) or bool(keep[i])
        motifs = [m for m in ms if m in mg and motif_kept.get(m, True)]
        pos = [m for m in motifs if mg[m] > 0]
        neg = [m for m in motifs if mg[m] == 0]
        v = ev._auc(ms, pos, neg)              # ms is a dict; _auc indexes ms[m]
        if v == v:
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def _pred_auc(aligned_split: dict, gl) -> float:
    from sklearn.metrics import roc_auc_score
    ys, ps = [], []
    for gi, rec in aligned_split.items():
        y = getattr(gl[gi], "y", None)
        if y is None:
            continue
        ys.append(float(_np1(y)[0])); ps.append(float(rec["pred"]))
    if len(set(ys)) < 2:
        return float("nan")
    return float(roc_auc_score(ys, ps))


def evaluate(dataset, fold, vocab, unk, data_root, processed_root, neutral_path, dest_root,
             vocab_root=None):
    ev = _evaluate_module()
    split_lists, gt, vocab_obj, dmeta, task_type = load_our_graphs(
        dataset, fold, vocab, data_root, processed_root, regime="source", vocab_root=vocab_root)
    neutral = json.loads(Path(neutral_path).read_text())
    aligned = _align(neutral, split_lists)

    kept = kept_set(dataset, fold, vocab, data_root, vocab_root) if unk == "exclude" else None
    keep_fn = ev._keep_fn(kept, unk)

    # number of layers from any record
    any_rec = next(iter(next(v for v in aligned.values() if v).values()))
    n_layers = len(any_rec["atom_att_by_layer"])

    rows = []
    for s in ("train", "valid", "test"):
        gl = gt.get(s)
        asp = aligned.get(s) or {}
        if not gl or not asp:
            continue
        pauc = _pred_auc(asp, gl)
        for L in range(n_layers):
            atom_by_i = _atom_att_by_i(asp, L)
            frag_by_i = _frag_broadcast_by_i(asp, gl, L)
            rows.append(dict(
                dataset=dataset, fold=int(fold), vocab=vocab, unk=unk, method=METHOD,
                split=s, layer=L, n_graphs=len(asp),
                atom_node_gtroc=ev._per_graph_mean_auc(atom_by_i, gl, "node_label", keep_fn),
                frag_node_gtroc=ev._per_graph_mean_auc(frag_by_i, gl, "node_label", keep_fn),
                frag_motif_gtroc=_frag_motif_auc_mean(asp, gl, L, keep_fn, ev),
                pred_auc=pauc))

    vocab_seg = (vocab + "_filter") if unk == "exclude" else vocab
    dest = Path(dest_root) / f"unk-{unk}" / vocab_seg / f"fold{int(fold)}"
    dest.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "fold", "vocab", "unk", "method", "split", "layer", "n_graphs",
            "atom_node_gtroc", "frag_node_gtroc", "frag_motif_gtroc", "pred_auc"]
    with open(dest / "fragnet_frag_perlayer_gtroc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)
    (dest / "fragnet_frag_perlayer_gtroc.json").write_text(json.dumps(rows, indent=2))
    print(f"[eval_frag] {dataset} fold{fold} unk={unk} -> {dest}")
    for r in rows:
        if r["split"] == "test":
            print("  L%d  atom_node=%.4f  frag_node=%.4f  frag_motif=%.4f  pred_auc=%.4f (n=%d)" % (
                r["layer"], r["atom_node_gtroc"], r["frag_node_gtroc"], r["frag_motif_gtroc"],
                r["pred_auc"], r["n_graphs"]))
    return rows


def _main():
    ap = argparse.ArgumentParser(description="Stage B (frag): per-layer FragNet attention GT-ROC")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--unk", required=True, choices=["include", "exclude"])
    ap.add_argument("--vocab_root", default=None)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--processed_root", required=True)
    ap.add_argument("--neutral", required=True, help="fragnet_frag_neutral.json from export_frag_attention")
    ap.add_argument("--dest_root", required=True)
    args = ap.parse_args()
    evaluate(args.dataset, args.fold, args.vocab, args.unk, args.data_root, args.processed_root,
             args.neutral, args.dest_root, vocab_root=args.vocab_root)


if __name__ == "__main__":
    _main()
