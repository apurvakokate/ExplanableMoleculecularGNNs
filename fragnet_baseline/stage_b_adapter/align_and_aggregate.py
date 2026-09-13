"""Stage B (part 1) — load OUR graphs, align FragNet's per-atom attention to our node
order, and build the kept-set for the filtered view. Runs in the l2xgnn env.

Two entry points:
  * ``dump_context`` (CLI) — write ``graph_context.json`` for the FragNet env: per split,
    per graph, the canonical SMILES + node order + nodes_to_motifs + node_label. This is
    the our_env -> fragnet_env half of the neutral handoff (so Stage A can define motifs
    for its own-impact masking and key its outputs by canonical SMILES).
  * ``load_our_graphs`` / ``align`` / ``kept_set`` — library functions used by emit_artifacts.py.

We deliberately REUSE analysis/evaluate.py's loader (build_gt_loaders, split_lists_and_gt)
so FragNet sees byte-identical graphs/splits/GT to every other method. Nothing here needs a
live model.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[2]          # .../ExplanableMoleculecularGNNs


def _evaluate_module():
    """Import analysis/evaluate.py as a module (reuses its loader + scoring helpers)."""
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    spec = importlib.util.spec_from_file_location(
        "chemintuit_evaluate", _REPO / "analysis" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                      # main() is __main__-guarded; safe to import
    return mod


def _np1(x) -> np.ndarray:
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().view(-1).numpy()
    return np.asarray(x).reshape(-1)


def canon(smiles: str) -> str:
    """RDKit-canonical SMILES — the alignment key between our graphs and FragNet's export.
    Both sides featurize from SMILES in RDKit GetIdx() atom order, so canonical SMILES
    equality is the safe join key (atom count asserted on top).

    FAIL LOUD: an empty or unparseable SMILES means the graph↔FragNet join cannot be
    trusted, so we raise rather than return '' — a '' fallback would collapse every bad
    graph onto one key and silently cross-wire attributions."""
    from rdkit import Chem
    if not smiles or not str(smiles).strip():
        raise ValueError("empty SMILES — cannot align FragNet attributions")
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return Chem.MolToSmiles(m)


_ATOMS_INV = None


def our_node_symbols(graph) -> list:
    """Decode our graph's per-node element symbols from the one-hot atom features, in OUR node
    order. This is the ground truth for atom↔atom verification against FragNet. Our features are
    one-hot of atom.GetSymbol() indexed by ATOMS (dataset.py:_atom_features), node_encoder='onehot'
    is an identity passthrough, and node order is MolFromSmiles(smiles).GetAtoms() order."""
    import torch
    global _ATOMS_INV
    if _ATOMS_INV is None:
        from SharedModules.data.dataset import ATOMS, NUM_ATOM_TYPES
        _ATOMS_INV = ({v: k for k, v in ATOMS.items()}, NUM_ATOM_TYPES)
    inv, ncls = _ATOMS_INV
    x = getattr(graph, "x", None)
    if x is None:
        raise ValueError("graph has no node features x — cannot decode element symbols")
    x = torch.as_tensor(x)
    if x.dim() != 2 or x.shape[1] != ncls:
        raise ValueError(f"expected one-hot x [N,{ncls}] for element decoding, got {tuple(x.shape)} "
                         f"(only node_encoder='onehot' is supported for atom verification)")
    return [inv[int(i)] for i in x.argmax(dim=1).tolist()]


# Dataset taxonomy — TWO categories, THREE kinds:
#   REAL labels → 'source'  : native node_label GT  (the *_Verified_GT sets + mutag)
#               → 'none'    : real labels, NO node GT (BBBP, Mutagenicity, hERG, esol,
#                             Lipophilicity, ogbg-*) → GT-ROC undefined, skipped downstream
#   PLANTED GT  → 'planted' : DNF-relabelled; needs gt_cache + relabel_<tier> + a rule id
SOURCE_GT = {"Benzene_Verified_GT", "Fluoride_Carbonyl_Verified_GT",
             "Alkane_Carbonyl_Verified_GT", "mutag"}


def build_meta(dataset: str, fold: int, vocab_variant: str, regime: str) -> dict:
    """Loader meta for a REGIME. The POC uses 'source' (Benzene). 'none' is wired (real
    labels; node-GT simply absent → GT-ROC skipped). 'planted' is intentionally NOT wired
    here — it needs gt_cache + relabel_<tier> + a DNF rule id, which the full sweep must
    supply; we fail loud rather than silently mislabel a planted run as source."""
    if regime == "source":
        if dataset not in SOURCE_GT:
            raise ValueError(f"{dataset!r} is not a source-GT dataset {sorted(SOURCE_GT)}")
        return {"dataset": dataset, "fold": int(fold), "vocab_variant": vocab_variant,
                "use_gt": False, "gt_cache": None, "gt_tier": "source"}
    if regime == "none":
        return {"dataset": dataset, "fold": int(fold), "vocab_variant": vocab_variant,
                "use_gt": False, "gt_cache": None, "gt_tier": "none"}
    if regime == "planted":
        # VERIFIED planted_v2 layout (2026-09-13): planted_v2/<dataset>/dnf_k{K}_r{R}/ holds, per rule:
        #   gt_cache/<dataset>/fold{k}/rbrics/   — the planted GT SOURCE (relabelled y + node_label)
        #   <method>/rbrics[_filter]_relabelled_dnf_k{K}_r{R}/  — trained models (rule id in variant)
        #   eval/metrics_<method>_unk-*.csv
        # (No worker/launch scripts live inside planted_v2; those are at the restart base.)
        # Loader recipe: use_gt=True, gt_tier='planted', gt_vocab_variant='rbrics',
        #   gt_cache=<planted_root>/<dataset>/<rule_id>/gt_cache, vocab_variant carrying _relabelled_<rule_id>.
        # NOT wired here: the exact processed-vs-relabel loader semantics must be validated against a
        # KNOWN planted run before trusting FragNet-on-planted — we fail loud rather than guess.
        raise RuntimeError(
            "build_meta(regime='planted') is not used — planted graphs load directly from the cached "
            "{split}_with_gt.pt via load_planted_graphs(); see the load_our_graphs dispatch.")
    raise ValueError(f"unknown regime {regime!r} (expected source|none|planted)")


def load_our_graphs(dataset: str, fold: int, vocab_variant: str,
                    data_root: str, processed_root: str,
                    regime: str = "source", batch_size: int = 128,
                    planted_root: Optional[str] = None, rule_id: Optional[str] = None,
                    vocab_root: Optional[str] = None):
    """Returns (split_lists, gt, vocab, dmeta, task_type).
    - source/none: via evaluate.py's own loader (``processed_root`` is the BASE; the variant is
      appended internally, matching evaluate.py --processed_root semantics).
    - planted: bypasses the loader and reads the cached ``{split}_with_gt.pt`` directly
      (they carry the rule-derived y + node_label + smiles + nodes_to_motifs)."""
    if regime == "planted":
        if not (planted_root and rule_id):
            raise ValueError("planted regime requires planted_root and rule_id (e.g. 'dnf_k2_r1')")
        return load_planted_graphs(dataset, fold, planted_root, rule_id, vocab_variant, vocab_root)
    ev = _evaluate_module()
    meta = build_meta(dataset, fold, vocab_variant, regime)
    loaders, vocab, dmeta, task_type = ev.build_gt_loaders(
        meta, data_root, vocab_root, processed_root, batch_size)
    split_lists, gt = ev.split_lists_and_gt(loaders, meta)
    return split_lists, gt, vocab, dmeta, task_type


def load_planted_graphs(dataset: str, fold: int, planted_root: str, rule_id: str,
                        vocab: str = "rbrics", vocab_root: Optional[str] = None):
    """Planted regime — load the pre-cached relabelled graphs directly (verified layout
    2026-09-13): planted_v2/<ds>/<rule_id>/gt_cache/<ds>/fold{k}/<vocab>/relabel_<rule_id>/
    {train,valid,test}_with_gt.pt. Each Data carries the rule-derived y + node_label +
    edge_label + smiles + nodes_to_motifs (apply_gt.py / apply_gt_loaders, loader.py:1081).
    So planted reuses the ENTIRE source pipeline; only the data source differs.

    FAIL LOUD on any missing file — a partial planted load would train/score FragNet on the
    wrong target."""
    import torch
    relabel_dir = (Path(planted_root) / dataset / rule_id / "gt_cache" / dataset
                   / f"fold{int(fold)}" / vocab / f"relabel_{rule_id}")
    if not relabel_dir.exists():
        raise FileNotFoundError(f"planted relabel dir not found: {relabel_dir}")
    split_lists: Dict[str, list] = {}
    for s, fn in (("train", "train_with_gt.pt"), ("valid", "valid_with_gt.pt"),
                  ("test", "test_with_gt.pt")):
        p = relabel_dir / fn
        if not p.exists():
            raise FileNotFoundError(f"planted split cache missing: {p}")
        split_lists[s] = list(torch.load(p, weights_only=False))
    # planted graphs carry node_label → they ARE the GT eval lists
    gt = {s: split_lists[s] for s in split_lists}
    from SharedModules.data.vocab import load_vocab
    vocab_obj = load_vocab(vocab_root, dataset, vocab)   # for motif_list (motif_smarts in rows)
    task_type = "BinaryClass"                      # planted DNF targets are binary (BBBP/hERG/Mutagenicity)
    return split_lists, gt, vocab_obj, None, task_type


def kept_set(dataset: str, fold: int, vocab_base: str,
             data_root: str, vocab_root: Optional[str] = None) -> set:
    """The filtered (--unk exclude) kept motif-id set for one fold, via the *_filter vocab's
    per-fold support threshold. Faithful port of evaluate.py:1028-1050."""
    from SharedModules.data.vocab import load_vocab
    from SharedModules.data.fold_threshold import build_fold_annotation
    from SharedModules.data.dataset_schema import DATASET_COLUMN
    want_variant = vocab_base + "_filter"
    filt = load_vocab(str(vocab_root) if vocab_root else None, dataset, want_variant)
    # FAIL LOUD: we explicitly requested the _filter vocab, so its .variant must agree.
    # (evaluate.py:1045 used `filt.variant or want_variant` as a silent fallback; we assert
    # instead and pass the known name, so a vocab-loading mismatch surfaces rather than hides.)
    if filt.variant and filt.variant != want_variant:
        raise ValueError(f"loaded filter vocab variant {filt.variant!r} != requested {want_variant!r}")
    csv = Path(data_root) / f"{dataset}_{int(fold)}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"fold CSV not found: {csv}")
    _, kept, _, _ = build_fold_annotation(
        lookup_all=filt.lookup_all, motif_list=filt.motif_list,
        mol_fragment_smarts=filt.mol_fragment_smarts, csv_path=str(csv),
        label_col=DATASET_COLUMN[dataset], dataset=dataset,
        variant=want_variant,
        vocab_dir=Path(filt.vocab_dir) if filt.vocab_dir else Path("."),
        apply_threshold=True, threshold_pct=filt.threshold_pct)
    if kept is None:
        raise ValueError(f"no per-fold kept set for {dataset} fold {fold}")
    return {int(x) for x in kept}


def align(neutral: dict, split_lists: Dict[str, list]) -> Dict[str, Dict[int, np.ndarray]]:
    """Map FragNet's per-atom attention onto our split-local graph index.

    Keyed by src_idx (== our split-local index), NOT SMILES — so duplicate SMILES and FragNet's
    molecule drops are both handled. FragNet featurized from the IDENTICAL (verbatim) SMILES, so
    its heavy-atom order equals ours and atts are already in our node order. The atom↔atom mapping
    is VERIFIED, not assumed: we assert FragNet's per-atom element sequence equals our decoded
    element sequence (catches any H/reorder/canonicalization drift) plus the count. Returns
    {split: {gi: [N] atts}}.

    neutral schema (Stage A): {split: {str(src_idx): {"atts":[...], "atom_syms":[...],
    "pred":float, "own_impact":{mid:val}, "n_atoms":int}}}.
    """
    out: Dict[str, Dict[int, np.ndarray]] = {}
    problems: List[str] = []
    for split, sl in split_lists.items():
        by_idx = (neutral.get(split) or {})
        out[split] = {}
        for gi, g in enumerate(sl):
            rec = by_idx.get(str(gi))
            if rec is None:
                problems.append(f"{split}[{gi}] no FragNet record (dropped in featurization?)")
                continue
            atts = np.asarray(rec["atts"], dtype=float)
            our_syms = our_node_symbols(g)
            fn_syms = list(rec.get("atom_syms") or [])
            if atts.shape[0] != len(our_syms):
                problems.append(f"{split}[{gi}] atom-count: FragNet {atts.shape[0]} vs graph {len(our_syms)}")
                continue
            if fn_syms != our_syms:
                j = next((k for k in range(min(len(fn_syms), len(our_syms))) if fn_syms[k] != our_syms[k]), -1)
                problems.append(f"{split}[{gi}] ELEMENT MISMATCH at atom {j}: FragNet={fn_syms[max(0,j-1):j+2]} "
                                f"graph={our_syms[max(0,j-1):j+2]} — atom order differs, mapping untrustworthy")
                continue
            out[split][gi] = atts
    if problems:
        raise AssertionError(
            "FragNet↔graph alignment failed (refusing to emit misaligned scores):\n  "
            + "\n  ".join(problems[:20])
            + (f"\n  ...(+{len(problems) - 20} more)" if len(problems) > 20 else ""))
    return out


def dump_context(dataset: str, fold: int, vocab_variant: str,
                 data_root: str, processed_root: str, out_path: str,
                 regime: str = "source",
                 planted_root: Optional[str] = None, rule_id: Optional[str] = None,
                 vocab_root: Optional[str] = None) -> None:
    """Write graph_context.json for the FragNet env. Per split, per graph: VERBATIM SMILES (so
    FragNet featurizes the identical string → identical atom order), y (FragNet's training
    target — relabelled for planted), per-atom element symbols (for atom↔atom verification),
    nodes_to_motifs (so Stage A groups atoms into motifs for own-impact masking), and node_label."""
    split_lists, _gt, _vocab, _dmeta, _tt = load_our_graphs(
        dataset, fold, vocab_variant, data_root, processed_root,
        regime=regime, planted_root=planted_root, rule_id=rule_id, vocab_root=vocab_root)
    ctx: Dict[str, list] = {}
    for split, sl in split_lists.items():
        rows = []
        for gi, g in enumerate(sl):
            smi = getattr(g, "smiles", None)
            if not smi:
                raise ValueError(f"{split}[{gi}] graph carries no SMILES — cannot dump context")
            nl = getattr(g, "node_label", None)
            y = getattr(g, "y", None)
            if y is None:
                raise ValueError(f"{split}[{gi}] graph has no label y — Stage A needs the training target")
            n2m = _np1(getattr(g, "nodes_to_motifs")).astype(int)
            syms = our_node_symbols(g)
            if len(syms) != n2m.shape[0]:
                raise AssertionError(f"{split}[{gi}] element count {len(syms)} != node count {n2m.shape[0]}")
            rows.append({
                "idx": gi,
                "smiles": smi,                               # VERBATIM — FragNet must featurize the identical string
                "y": _np1(y).astype(float).tolist(),         # training target (relabelled for planted)
                "atom_syms": syms,                           # our per-atom element, our order — atom↔atom verification
                "n_atoms": int(n2m.shape[0]),
                "nodes_to_motifs": n2m.tolist(),
                "node_label": (_np1(nl).astype(float).tolist() if nl is not None else None),
            })
        ctx[split] = rows
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(ctx))
    print(f"[dump_context] {dataset} fold{fold}: wrote {out_path} "
          f"({sum(len(v) for v in ctx.values())} graphs)")


def _main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump_context")
    d.add_argument("--dataset", required=True)
    d.add_argument("--fold", type=int, required=True)
    d.add_argument("--vocab", required=True)
    d.add_argument("--data_root", required=True)
    d.add_argument("--processed_root", required=True)
    d.add_argument("--regime", default="source", choices=["source", "none", "planted"])
    d.add_argument("--vocab_root", default=None, help="load_vocab root (e.g. .../vocab_final_v2)")
    d.add_argument("--planted_root", default=None, help="planted regime: planted_v2 root")
    d.add_argument("--rule_id", default=None, help="planted regime: e.g. dnf_k2_r1")
    d.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "dump_context":
        dump_context(args.dataset, args.fold, args.vocab,
                     args.data_root, args.processed_root, args.out, regime=args.regime,
                     planted_root=args.planted_root, rule_id=args.rule_id, vocab_root=args.vocab_root)


if __name__ == "__main__":
    _main()
