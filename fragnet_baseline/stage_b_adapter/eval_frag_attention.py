"""Stage B (frag) — evaluate FragNet's PER-LAYER attention, from export_frag_attention.py, on BOTH
axes the pipeline reports, WITHOUT cross-layer aggregation, over ALL splits (train/valid/test).

Runs in the l2xgnn env. Consumes fragnet_frag_neutral.json (per-layer atom + fragment attention +
frag_to_motif + per-motif own_impact, keyed by our src_idx).

Axis 1 — GT-ROC (correctness): attention score vs ground-truth atoms/motifs. Three columns per layer:
  * atom_node_gtroc  — per-atom ATOM attention vs per-atom node_label (evaluate.py's own
                       _per_graph_mean_auc — the pipeline's exact node GT-ROC).
  * frag_node_gtroc  — FRAGMENT attention broadcast to its atoms, then the SAME node GT-ROC.
  * frag_motif_gtroc — FRAGMENT attention at MOTIF granularity: per-graph mean motif-level AUC.

Axis 2 — Pearson (faithfulness): per-layer attention score vs OWN-IMPACT (|Δpred| from masking the
motif's heavy-atom rows; layer-independent, so ONE impact cache serves every layer). Computed for TWO
score sources, each grouped + per-instance, reusing evaluate.py's grouped_corr_variants + instance_corr
so the math is identical to MoSE/MotifSAT/GSAT:
  * atom_* — motif score = mean ATOM attention over the motif's atoms (score_cache_from_atts).
  * frag_* — motif score = FragNet's NATIVE fragment attention for that motif.
FULL vs FILTERED is controlled SOLELY by the kept (support-gated) motif list — the SAME lever as
GT-ROC's keep_fn and instance_corr's internal gate: full (unk=include) uses every motif present;
filtered (unk=exclude) keeps only motifs in `kept`. Our nodes_to_motifs never contains the UNK
sentinel (UNK_ID = -1), so grouped_corr_variants' inclunk/exclunk are identical here; we report
inclunk (the correlation over EXACTLY the kept-gated rows we build), so the kept list is the only
filter. Running the eval with unk=include vs unk=exclude yields the UNFILTERED vs FILTERED numbers.

Fragment attention is only meaningful when the model was finetuned with frag_type='custom' (FragNet's
fragments == our rbrics motifs). frag_to_motif maps each fragment to its motif; repeated motif types
(several fragments) are aggregated to the type by mean. Attention score, own-impact and the pooled atom
score are all keyed by our motif ids, so the three caches align.

Reuses evaluate.py via _evaluate_module(): _per_graph_mean_auc, _auc, _keep_fn, score_cache_from_atts,
grouped_corr_variants, instance_corr — no edit to evaluate.py.
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
    """Per-atom fragment score: each atom gets its motif's fragment score (broadcast). COVERAGE ASSERT:
    every atom must resolve to a finite motif score — a gap (a motif with no fragment score, i.e. a
    frag→motif coverage hole) would become NaN and silently drop the whole graph from frag_node_gtroc
    via the _auc NaN path, so we fail loud instead."""
    out = {}
    for gi, rec in aligned_split.items():
        g = gl[gi]
        ms = _motif_scores(rec, layer)
        n2m = _np1(getattr(g, "nodes_to_motifs")).astype(int)
        vals = np.asarray([ms.get(int(m), np.nan) for m in n2m], dtype=float)
        if not np.all(np.isfinite(vals)):
            bad = sorted({int(m) for m, v in zip(n2m, vals) if not np.isfinite(v)})
            raise AssertionError(
                f"graph {gi} layer {layer}: motifs {bad} have no fragment score (frag→motif coverage "
                f"gap) — frag_node_gtroc would silently drop this graph.")
        out[gi] = vals
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


def _own_impact_cache(aligned_split: dict) -> dict:
    """{mid: {gi: own_impact}} from the neutral per-graph per-motif own_impact (layer-independent)."""
    ic = {}
    for gi, rec in aligned_split.items():
        for mid_s, val in (rec.get("own_impact") or {}).items():
            ic.setdefault(int(mid_s), {})[gi] = float(val)
    return ic


def _frag_sc_cache(aligned_split: dict, layer: int) -> dict:
    """{mid: {gi: native fragment attention for that motif}} — the frag analogue of score_cache."""
    out = {}
    for gi, rec in aligned_split.items():
        for mid, sc in _motif_scores(rec, layer).items():
            out.setdefault(int(mid), {})[gi] = float(sc)
    return out


def _pearson_block(sc_cache: dict, ic_cache: dict, kept, unk: str, prefix: str, ev) -> dict:
    """Grouped + per-instance Pearson of a score cache vs own-impact, reusing evaluate.py. The kept
    (support-gated) list is the ONLY filter: grouped rows and instances are dropped iff unk=exclude and
    the motif is not in `kept`. We report grouped_corr_variants' INCLUNK variant — the correlation over
    exactly the kept-gated rows we pass — NOT exclunk: exclunk would additionally drop motif_id == -1,
    but our motif ids never take that sentinel, so exclunk is a vacuous no-op and inclunk is the honest
    number. instance_corr applies the identical kept/unk gate internally."""
    rows = []
    for mid in sorted(set(sc_cache) & set(ic_cache)):
        if unk == "exclude" and kept is not None and mid not in kept:
            continue
        sv = list(sc_cache[mid].values()); iv = list(ic_cache[mid].values())
        if not sv or not iv:
            continue
        rows.append(dict(motif_id=int(mid), score=float(np.mean(sv)),
                         impact=float(np.mean(iv)), support=len(iv)))
    g = ev.grouped_corr_variants(rows)
    inst = ev.instance_corr(sc_cache, ic_cache, kept, unk)
    return {
        f"{prefix}_grp_pearson_u": g.get("pearson_u_inclunk"),
        f"{prefix}_grp_pearson_w": g.get("pearson_w_inclunk"),
        f"{prefix}_grp_spearman_u": g.get("spearman_u_inclunk"),
        f"{prefix}_grp_n_motifs": g.get("n_motifs_inclunk"),
        f"{prefix}_inst_pearson": inst.get("pearson_instance"),
        f"{prefix}_inst_spearman": inst.get("spearman_instance"),
        f"{prefix}_inst_n": inst.get("n_instances"),
    }


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


COLS = ["dataset", "fold", "vocab", "unk", "method", "split", "layer", "n_graphs",
        "atom_node_gtroc", "frag_node_gtroc", "frag_motif_gtroc", "pred_auc",
        "atom_grp_pearson_u", "atom_grp_pearson_w", "atom_grp_spearman_u", "atom_grp_n_motifs",
        "atom_inst_pearson", "atom_inst_spearman", "atom_inst_n",
        "frag_grp_pearson_u", "frag_grp_pearson_w", "frag_grp_spearman_u", "frag_grp_n_motifs",
        "frag_inst_pearson", "frag_inst_spearman", "frag_inst_n"]


def evaluate(dataset, fold, vocab, unk, data_root, processed_root, neutral_path, dest_root,
             vocab_root=None):
    ev = _evaluate_module()
    split_lists, gt, vocab_obj, dmeta, task_type = load_our_graphs(
        dataset, fold, vocab, data_root, processed_root, regime="source", vocab_root=vocab_root)
    neutral = json.loads(Path(neutral_path).read_text())
    aligned = _align(neutral, split_lists)

    kept = kept_set(dataset, fold, vocab, data_root, vocab_root) if unk == "exclude" else None
    keep_fn = ev._keep_fn(kept, unk)

    any_rec = next(iter(next(v for v in aligned.values() if v).values()))
    n_layers = len(any_rec["atom_att_by_layer"])

    rows = []
    for s in ("train", "valid", "test"):
        gl = gt.get(s)
        asp = aligned.get(s) or {}
        if not gl or not asp:
            continue
        pauc = _pred_auc(asp, gl)
        missing_oi = [gi for gi, rec in asp.items() if "own_impact" not in rec]
        if missing_oi:
            raise AssertionError(
                f"{s}: {len(missing_oi)} of {len(asp)} records lack 'own_impact' (e.g. gi={missing_oi[0]}) "
                f"— stale/partial neutral export predating own-impact; re-run export_frag_attention.py so "
                f"the Pearson columns are not silently empty.")
        ic_cache = _own_impact_cache(asp)                      # layer-independent
        for L in range(n_layers):
            atom_by_i = _atom_att_by_i(asp, L)
            frag_by_i = _frag_broadcast_by_i(asp, gl, L)
            atom_sc = ev.score_cache_from_atts(atom_by_i, gl)  # {mid:{gi: mean atom att}}
            frag_sc = _frag_sc_cache(asp, L)                   # {mid:{gi: native frag att}}
            row = dict(
                dataset=dataset, fold=int(fold), vocab=vocab, unk=unk, method=METHOD,
                split=s, layer=L, n_graphs=len(asp),
                atom_node_gtroc=ev._per_graph_mean_auc(atom_by_i, gl, "node_label", keep_fn),
                frag_node_gtroc=ev._per_graph_mean_auc(frag_by_i, gl, "node_label", keep_fn),
                frag_motif_gtroc=_frag_motif_auc_mean(asp, gl, L, keep_fn, ev),
                pred_auc=pauc)
            row.update(_pearson_block(atom_sc, ic_cache, kept, unk, "atom", ev))
            row.update(_pearson_block(frag_sc, ic_cache, kept, unk, "frag", ev))
            rows.append(row)

    vocab_seg = (vocab + "_filter") if unk == "exclude" else vocab
    dest = Path(dest_root) / f"unk-{unk}" / vocab_seg / f"fold{int(fold)}"
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "fragnet_frag_perlayer_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for r in rows:
            w.writerow(r)
    (dest / "fragnet_frag_perlayer_metrics.json").write_text(json.dumps(rows, indent=2))
    print(f"[eval_frag] {dataset} fold{fold} unk={unk} -> {dest}")

    def _f(v):
        return float("nan") if v is None else float(v)
    for s in ("train", "valid", "test"):
        srows = [r for r in rows if r["split"] == s]
        if not srows:
            continue
        print(f"  [{s}] n={srows[0]['n_graphs']}  (GT-ROC | instance-Pearson r | grouped-Pearson r)")
        for r in srows:
            print("    L%d  atom_node=%.3f frag_node=%.3f frag_motif=%.3f | "
                  "atom_inst=%.3f frag_inst=%.3f | atom_grp=%.3f(n%s) frag_grp=%.3f(n%s)  auc=%.3f" % (
                      r["layer"], r["atom_node_gtroc"], r["frag_node_gtroc"], r["frag_motif_gtroc"],
                      _f(r["atom_inst_pearson"]), _f(r["frag_inst_pearson"]),
                      _f(r["atom_grp_pearson_u"]), r["atom_grp_n_motifs"],
                      _f(r["frag_grp_pearson_u"]), r["frag_grp_n_motifs"], r["pred_auc"]))
    return rows


def _main():
    ap = argparse.ArgumentParser(description="Stage B (frag): per-layer FragNet attention GT-ROC + Pearson")
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
