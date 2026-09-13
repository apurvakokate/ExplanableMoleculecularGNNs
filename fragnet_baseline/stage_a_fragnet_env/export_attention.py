"""Stage A (3/3) — runs in the FragNet env. Load the finetuned FragNet and export the NEUTRAL
handoff consumed by Stage B, keyed by OUR split-local index (src_idx):

  {split: {str(src_idx): {"atts":[per-atom], "atom_syms":[per-atom element], "pred":float,
                          "own_impact":{motif_id: float}, "n_atoms":int}}}

- atts       : per-atom attention from FragNetViz (the model's native node-level importance).
- atom_syms  : per-atom element (FragNet x_atoms order) — Stage B asserts this equals our graph's
               element sequence, proving atom i == atom i (not just equal counts).
- pred       : FragNetFineTune prediction (sigmoid for classification).
- own_impact : ante-hoc OWN impact = |p_full - p_masked|, ablating a motif's atoms (x_atoms rows
               zeroed). Motif membership comes from OUR nodes_to_motifs (graph_context), which is
               order-aligned to FragNet's atoms (verified via atom_syms before masking).

VALIDATE-ON-FIRST-RUN (fail loud, never silent): FragNetViz backbone-weight transfer; attn_atom ->
per-atom reduction. Use --impact none for GT-ROC-only.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch


def _load_models(ft_ckpt: str, device, n_classes=1):
    from fragnet.model.gat.gat2 import FragNetFineTune, FragNetViz
    # Head dims MUST match finetune_fragnet.build_config's finetune.model, or the FTHead3 state_dict
    # won't load: the checkpoint's predictor is 256->128->1024->1024->512->1 (h1=128,h2=1024,h3=1024,
    # h4=512). FragNetFineTune's OWN defaults are h1..h4=256, which is the mismatch that bit us.
    # Keep these in sync with finetune_fragnet.py (or, better, read <work>/config.yaml — see TODO).
    ft = FragNetFineTune(n_classes=n_classes, atom_features=167, frag_features=167,
                         edge_features=17, num_layer=4, num_heads=4, emb_dim=128,
                         h1=128, h2=1024, h3=1024, h4=512, act="relu", fthead="FTHead3")
    ft.load_state_dict(torch.load(ft_ckpt, map_location=device))
    ft.to(device).eval()
    viz = FragNetViz(num_layer=4, emb_dim=128, num_heads=4, return_attentions=True)
    missing, unexpected = viz.load_state_dict(ft.pretrain.state_dict(), strict=False)
    if len(ft.pretrain.state_dict()) - len(unexpected) == 0:
        raise RuntimeError("FragNetViz loaded 0 backbone tensors from ft.pretrain — name mismatch; "
                           "inspect ft.pretrain.state_dict() keys vs FragNetViz.")
    print(f"[export] viz backbone transfer: missing={len(missing)} unexpected={len(unexpected)}")
    viz.to(device).eval()
    return ft, viz


def _batch1(data, device):
    from fragnet.dataset.data import collate_fn
    b = collate_fn([data])
    if not isinstance(b, dict):
        raise TypeError("collate_fn did not return a dict batch — inspect fragnet/dataset/data.py")
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}


def _per_atom_atts(attn_atom, n_atoms: int) -> np.ndarray:
    """Reduce FragNetViz's attn_atom to one scalar per atom (x_atoms order). Asserts length ==
    n_atoms after averaging trailing (head/feature) dims; a per-edge shape trips the assert."""
    a = attn_atom.detach().cpu().float()
    if a.dim() > 1:
        a = a.mean(dim=tuple(range(1, a.dim())))
    a = a.view(-1).numpy()
    if a.shape[0] != n_atoms:
        raise AssertionError(
            f"attn_atom length {a.shape[0]} != n_atoms {n_atoms} — attn_atom is likely not per-atom "
            f"(per-edge?); inspect FragNetViz last-layer return and reduce accordingly.")
    return a


@torch.no_grad()
def _predict(ft, batch) -> float:
    out = ft(batch).view(-1)
    return float(torch.sigmoid(out[0]))          # classification; for regression use out[0] directly


@torch.no_grad()
def _own_impact(ft, data, device, nodes_to_motifs, p_full: float) -> dict:
    """|p_full - p_masked| per motif; ablation = zero the motif's atom feature rows (x_atoms),
    mirroring the pipeline's feature-zeroing _ablate_motif. nodes_to_motifs is our per-atom motif
    id, order-aligned to x_atoms (verified by atom_syms in the caller)."""
    n2m = np.asarray(nodes_to_motifs, dtype=int)
    if n2m.shape[0] != int(data.x_atoms.shape[0]):
        raise AssertionError(f"nodes_to_motifs {n2m.shape[0]} != x_atoms {int(data.x_atoms.shape[0])}")
    out = {}
    for mid in sorted({int(m) for m in n2m if m >= 0}):
        mask = torch.as_tensor(n2m == mid, dtype=torch.bool)
        if int(mask.sum()) == 0:
            continue
        d = data.clone()
        d.x_atoms = d.x_atoms.clone()
        d.x_atoms[mask] = 0.0
        p_masked = _predict(ft, _batch1(d, device))
        out[int(mid)] = abs(p_full - p_masked)
    return out


def export(graph_context: str, work: str, ft_ckpt: str, out_path: str,
           impact: str = "own", device: str = "auto") -> None:
    from fragnet.dataset.dataset import load_pickle_dataset
    dev = torch.device("cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available()))
                       else "cpu")
    ctx = json.loads(Path(graph_context).read_text())
    # per split: idx -> {atom_syms, nodes_to_motifs}
    ctx_by_idx = {s: {int(r["idx"]): r for r in (ctx.get(s) or [])} for s in ("train", "valid", "test")}

    ft, viz = _load_models(ft_ckpt, dev)
    work = Path(work)
    pkl_name = {"train": "train.pkl", "valid": "val.pkl", "test": "test.pkl"}
    neutral = {}
    for split, fn in pkl_name.items():
        p = work / fn
        if not p.exists():
            raise FileNotFoundError(f"missing FragNet pkl {p} — run prep_data.py/finetune first")
        ds = load_pickle_dataset(str(p))
        out_split = {}
        for data in ds:
            idx = int(data.src_idx)
            row = ctx_by_idx[split].get(idx)
            if row is None:
                raise KeyError(f"{split}: src_idx {idx} not in graph_context — handoff mismatch")
            n_atoms = int(data.x_atoms.shape[0])
            fn_syms = list(data.atom_syms)
            # atom↔atom verification BEFORE using any per-atom quantity
            if fn_syms != list(row["atom_syms"]):
                j = next((k for k in range(min(len(fn_syms), len(row["atom_syms"])))
                          if fn_syms[k] != row["atom_syms"][k]), -1)
                raise AssertionError(
                    f"{split} src_idx {idx}: ELEMENT MISMATCH at atom {j} — FragNet atom order "
                    f"differs from ours; attention/impact would be misaligned.")
            batch = _batch1(data, dev)
            with torch.no_grad():
                _, _, _, attn_atom, _, _ = viz(batch)
            atts = _per_atom_atts(attn_atom, n_atoms)
            pred = _predict(ft, batch)
            oi = _own_impact(ft, data, dev, row["nodes_to_motifs"], pred) if impact == "own" else {}
            out_split[str(idx)] = {"atts": atts.tolist(), "atom_syms": fn_syms, "pred": pred,
                                   "own_impact": {str(k): v for k, v in oi.items()}, "n_atoms": n_atoms}
        neutral[split] = out_split
        print(f"[export] {split}: {len(out_split)} graphs")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(neutral))
    print(f"[export] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Stage A/3: FragNet -> neutral atts+impact+pred JSON")
    ap.add_argument("--work", required=True)
    ap.add_argument("--vendor", required=True)
    ap.add_argument("--graph_context", required=True)
    ap.add_argument("--ft_ckpt", default=None, help="default: <work>/ft.pt")
    ap.add_argument("--impact", choices=["own", "none"], default="own",
                    help="'none' = GT-ROC-only (skip own-impact masked re-forward)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ft_ckpt = args.ft_ckpt or str(Path(args.work) / "ft.pt")
    export(args.graph_context, args.work, ft_ckpt, args.out, impact=args.impact, device=args.device)


if __name__ == "__main__":
    main()
