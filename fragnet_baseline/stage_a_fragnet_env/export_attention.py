"""Stage A (3/3) — runs in the FragNet env. Load the finetuned FragNet and export the NEUTRAL
handoff consumed by Stage B, keyed by OUR split-local index (src_idx):

  {split: {str(src_idx): {"atts":[per-atom], "atom_syms":[per-atom element], "pred":float,
                          "own_impact":{motif_id: float}, "n_atoms":int}}}

We use FragNet's OWN visualization model, fragnet.vizualize.model.FragNetFineTuneViz: a single forward
returns (prediction, attn_atoms, attn_frags, attn_bonds, attn_fbonds). Per-atom attention is attn_atoms
summed over the head dim — matching FragNet's own fragnet.vizualize.viz.vizualize_atom_weights, which
does `summed_attn_weights_atoms.sum(1)`. This replaces an earlier hand-rolled transfer into gat2.py's
FragNetViz, whose replacement last layer (FragNetLayerA) did not match the checkpoint.

The model architecture is read from the <work>/config.yaml that finetune_fragnet.py wrote, so it can
NEVER drift from the checkpoint (an earlier drift — default head dims vs the trained h1..h4 — is exactly
what bit us). Weights transfer via a shape-safe partial load (mirrors fragnet.vizualize.viz
.load_partial_weights); we FAIL LOUD if the backbone or head did not transfer, so untrained attention
or predictions can never slip through silently.

- atts       : per-atom attention (FragNet's native node-level importance).
- atom_syms  : per-atom element (FragNet x_atoms order) — Stage B asserts this equals our graph's
               element sequence, proving atom i == atom i (not just equal counts).
- pred       : FragNetFineTuneViz prediction (sigmoid for classification).
- own_impact : ante-hoc OWN impact = |p_full - p_masked|, ablating a motif's atoms (x_atoms rows
               zeroed). Motif membership comes from OUR nodes_to_motifs (graph_context), order-aligned
               to FragNet's atoms (verified via atom_syms before masking).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _read_model_cfg(work) -> dict:
    """Read finetune's config.yaml so export reconstructs EXACTLY the model that was trained (no drift).
    Returns the finetune.model dict, augmented with the top-level atom/frag/edge feature dims."""
    import yaml
    p = Path(work) / "config.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found — finetune_fragnet.py writes it; export needs the exact model dims to "
            f"rebuild FragNetFineTuneViz. No silent default: the head dims MUST match the checkpoint.")
    full = yaml.safe_load(p.read_text())
    cfg = dict((full.get("finetune") or {}).get("model") or {})
    for k in ("atom_features", "frag_features", "edge_features"):
        if full.get(k) is not None:
            cfg.setdefault(k, full[k])
    return cfg


def _load_model(ft_ckpt: str, device, cfg: dict):
    """Build fragnet.vizualize.model.FragNetFineTuneViz (FragNet's own attention model) and load the
    finetuned checkpoint with a shape-safe partial loader. One forward -> (pred, attn_atoms, ...)."""
    from fragnet.vizualize.model import FragNetFineTuneViz
    model = FragNetFineTuneViz(
        n_classes=cfg.get("n_classes", 1),
        atom_features=cfg.get("atom_features", 167),
        frag_features=cfg.get("frag_features", 167),
        edge_features=cfg.get("edge_features", 17),
        num_layer=cfg.get("num_layer", 4), num_heads=cfg.get("num_heads", 4),
        drop_ratio=cfg.get("drop_ratio", 0.1),
        h1=cfg.get("h1", 128), h2=cfg.get("h2", 1024), h3=cfg.get("h3", 1024), h4=cfg.get("h4", 512),
        act=cfg.get("act", "relu"), emb_dim=cfg.get("emb_dim", 128),
        fthead=cfg.get("fthead", "FTHead3"))
    ckpt = torch.load(ft_ckpt, map_location=device)
    msd = model.state_dict()
    # shape-safe partial transfer: keep only checkpoint tensors whose name AND shape match the model.
    transfer = {k: v for k, v in ckpt.items() if k in msd and v.shape == msd[k].shape}
    skipped = [k for k in ckpt if k not in transfer]
    msd.update(transfer)
    model.load_state_dict(msd)
    n_bb = sum(1 for k in transfer if k.startswith("pretrain."))
    n_head = sum(1 for k in transfer if k.startswith("fthead."))
    print(f"[export] FragNetFineTuneViz: transferred {len(transfer)}/{len(ckpt)} tensors "
          f"(backbone={n_bb}, head={n_head}); skipped={len(skipped)}")
    if skipped:
        print(f"[export] NOTE skipped keys (name/shape mismatch — expected only attention-layer "
              f"variant params, if any): {skipped}")
    if n_bb == 0 or n_head == 0:
        raise RuntimeError(
            "FragNetFineTuneViz: backbone or head did not transfer from the checkpoint — attention "
            "or predictions would be untrained. Inspect checkpoint key names/shapes vs the model.")
    model.to(device).eval()
    return model


def _batch1(data, device):
    from fragnet.dataset.data import collate_fn
    b = collate_fn([data])
    if not isinstance(b, dict):
        raise TypeError("collate_fn did not return a dict batch — inspect fragnet/dataset/data.py")
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}


@torch.no_grad()
def _forward(model, batch):
    """One FragNetFineTuneViz forward -> (pred_prob, attn_atoms). pred_prob = sigmoid(logit)."""
    out = model(batch)
    if not (isinstance(out, (tuple, list)) and len(out) >= 2):
        n = len(out) if hasattr(out, "__len__") else "?"
        raise TypeError(f"FragNetFineTuneViz returned {type(out)} len {n} — expected "
                        f"(pred, attn_atoms, attn_frags, attn_bonds, attn_fbonds).")
    logit = out[0].view(-1)
    return float(torch.sigmoid(logit[0])), out[1]


def _per_atom_atts(attn_atoms, n_atoms: int) -> np.ndarray:
    """Reduce attn_atoms to one scalar per atom by summing over the head dim — matches FragNet's own
    vizualize_atom_weights (summed_attn_weights_atoms.sum(1)). Length must equal n_atoms (x_atoms order)."""
    a = attn_atoms.detach().cpu().float()
    if a.dim() > 1:
        a = a.sum(dim=tuple(range(1, a.dim())))
    a = a.view(-1).numpy()
    if a.shape[0] != n_atoms:
        raise AssertionError(
            f"attn_atoms length {a.shape[0]} != n_atoms {n_atoms} — attention is not per-atom in "
            f"x_atoms order; inspect FragNetFineTuneViz attn_atoms shape.")
    return a


@torch.no_grad()
def _own_impact(model, data, device, nodes_to_motifs, p_full: float) -> dict:
    """|p_full - p_masked| per motif; ablation = zero the motif's atom feature rows (x_atoms). Motif
    membership is our per-atom motif id, order-aligned to x_atoms (verified by atom_syms in caller)."""
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
        p_masked, _ = _forward(model, _batch1(d, device))
        out[int(mid)] = abs(p_full - p_masked)
    return out


def export(graph_context: str, work: str, ft_ckpt: str, out_path: str,
           impact: str = "own", device: str = "auto") -> None:
    from fragnet.dataset.dataset import load_pickle_dataset
    dev = torch.device("cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available()))
                       else "cpu")
    ctx = json.loads(Path(graph_context).read_text())
    ctx_by_idx = {s: {int(r["idx"]): r for r in (ctx.get(s) or [])} for s in ("train", "valid", "test")}

    cfg = _read_model_cfg(work)
    model = _load_model(ft_ckpt, dev, cfg)
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
            # atom<->atom verification BEFORE using any per-atom quantity
            if fn_syms != list(row["atom_syms"]):
                j = next((k for k in range(min(len(fn_syms), len(row["atom_syms"])))
                          if fn_syms[k] != row["atom_syms"][k]), -1)
                raise AssertionError(
                    f"{split} src_idx {idx}: ELEMENT MISMATCH at atom {j} — FragNet atom order "
                    f"differs from ours; attention/impact would be misaligned.")
            pred, attn_atoms = _forward(model, _batch1(data, dev))
            atts = _per_atom_atts(attn_atoms, n_atoms)
            oi = _own_impact(model, data, dev, row["nodes_to_motifs"], pred) if impact == "own" else {}
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
    ap.add_argument("--vendor", required=True, help="vendored FragNet (kept for CLI parity; fragnet is pip-installed)")
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
