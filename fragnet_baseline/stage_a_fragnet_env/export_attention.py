"""Stage A (3/3) — runs in the FragNet env. Load the finetuned FragNet and export the NEUTRAL
handoff consumed by Stage B, keyed by OUR split-local index (src_idx):

  {split: {str(src_idx): {"atts":[per-atom], "atom_syms":[per-atom element], "pred":float,
                          "own_impact":{motif_id: float}, "n_atoms":int}}}

Model: FragNet's OWN fragnet.vizualize.model.FragNetFineTuneViz (one forward returns
(prediction, attn_atoms, attn_frags, attn_bonds, attn_fbonds)). Per-atom attention is attn_atoms
summed over the head dim (matches FragNet's vizualize_atom_weights). The architecture is read from the
<work>/config.yaml finetune wrote (no drift); weights load via a shape-safe partial loader.

FragNet keeps explicit H (get_3Dcoords adds them for the 3D conformer); our graph is heavy-atom-only.
We map via FragNet's HEAVY-atom subset, which must equal our atom sequence (verified per graph).

BATCHED: forwards are batched across molecules (collate_fn of `batch_graphs` graphs per forward) for
BOTH the base attention/prediction pass and the own-impact masked pass. A batch-1 implementation did
~84k single-graph forwards (12k base + ~72k masked) at ~15% GPU util (per-call overhead, not compute);
batching cuts that to ~1.3k forwards and fills the GPU. Correctness is unchanged — masking, the
heavy-atom verification, and the output schema are identical; only the number of forwards differs.

- atts       : per-atom attention (heavy atoms, our order).
- atom_syms  : per-atom element (our heavy-atom sequence).
- pred       : FragNetFineTuneViz prediction (sigmoid for classification).
- own_impact : ante-hoc OWN impact = |p_full - p_masked|, zeroing a motif's heavy-atom feature rows.
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
    ft = full.get("finetune") or {}
    cfg = dict(ft.get("model") or {})
    for k in ("atom_features", "frag_features", "edge_features"):
        if full.get(k) is not None:
            cfg.setdefault(k, full[k])
    # target_type drives the read-out activation: 'clsf' -> sigmoid, 'regr' -> raw output.
    # Default 'clsf' preserves existing (classification) behaviour when the key is absent.
    cfg["target_type"] = ft.get("target_type", "clsf")
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
    transfer = {k: v for k, v in ckpt.items() if k in msd and v.shape == msd[k].shape}
    skipped = [k for k in ckpt if k not in transfer]
    msd.update(transfer)
    model.load_state_dict(msd)
    n_bb = sum(1 for k in transfer if k.startswith("pretrain."))
    n_head = sum(1 for k in transfer if k.startswith("fthead."))
    print(f"[export] FragNetFineTuneViz: transferred {len(transfer)}/{len(ckpt)} tensors "
          f"(backbone={n_bb}, head={n_head}); skipped={len(skipped)}")
    if skipped:
        print(f"[export] NOTE skipped keys (name/shape mismatch): {skipped}")
    if n_bb == 0 or n_head == 0:
        raise RuntimeError(
            "FragNetFineTuneViz: backbone or head did not transfer from the checkpoint — attention "
            "or predictions would be untrained. Inspect checkpoint key names/shapes vs the model.")
    model.to(device).eval()
    return model


def _collate(datas, device):
    """collate_fn a LIST of Data into one batched dict, on device."""
    from fragnet.dataset.data import collate_fn
    b = collate_fn(datas)
    if not isinstance(b, dict):
        raise TypeError("collate_fn did not return a dict batch — inspect fragnet/dataset/data.py")
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}


def _reduce_heads(attn_atoms) -> torch.Tensor:
    """Sum attn_atoms over the head dim(s) -> 1D [total_atoms] (FragNet's vizualize_atom_weights does
    summed_attn_weights_atoms.sum(1))."""
    a = attn_atoms.detach().cpu().float()
    if a.dim() > 1:
        a = a.sum(dim=tuple(range(1, a.dim())))
    return a.view(-1)


@torch.no_grad()
def _run_batches(model, payload, batch_graphs, device):
    """Forward `payload` (list of Data) in chunks of `batch_graphs`. Yields (chunk_slice, logits,
    per_atom, atom_batch) per chunk: logits [n_chunk], per_atom [tot_atoms], atom_batch [tot_atoms]."""
    for s in range(0, len(payload), batch_graphs):
        chunk = payload[s:s + batch_graphs]
        batch = _collate([d for d in chunk], device)
        out = model(batch)
        if not (isinstance(out, (tuple, list)) and len(out) >= 2):
            n = len(out) if hasattr(out, "__len__") else "?"
            raise TypeError(f"FragNetFineTuneViz returned {type(out)} len {n} — expected "
                            f"(pred, attn_atoms, attn_frags, attn_bonds, attn_fbonds).")
        logits = out[0].view(-1).detach().cpu()                 # [n_chunk]
        per_atom = _reduce_heads(out[1])                        # [tot_atoms]
        atom_batch = batch["batch"].detach().cpu().view(-1)     # [tot_atoms] graph index per atom
        yield s, chunk, logits, per_atom, atom_batch


def export(graph_context: str, work: str, ft_ckpt: str, out_path: str,
           impact: str = "own", device: str = "auto", batch_graphs: int = 64) -> None:
    from fragnet.dataset.dataset import load_pickle_dataset
    dev = torch.device("cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available()))
                       else "cpu")
    print(f"[export] device={dev} batch_graphs={batch_graphs} impact={impact}")
    ctx = json.loads(Path(graph_context).read_text())
    ctx_by_idx = {s: {int(r["idx"]): r for r in (ctx.get(s) or [])} for s in ("train", "valid", "test")}

    cfg = _read_model_cfg(work)
    model = _load_model(ft_ckpt, dev, cfg)
    is_clf = (str(cfg.get("target_type", "clsf")) == "clsf")   # regression -> raw output, no sigmoid squash
    print(f"[export] target_type={cfg.get('target_type')} is_clf={is_clf}")
    work = Path(work)
    pkl_name = {"train": "train.pkl", "valid": "val.pkl", "test": "test.pkl"}
    neutral = {}
    for split, fn in pkl_name.items():
        p = work / fn
        if not p.exists():
            raise FileNotFoundError(f"missing FragNet pkl {p} — run prep_data.py/finetune first")
        ds = load_pickle_dataset(str(p))

        # --- resolve + verify every graph up front (fail loud) -------------------------------------
        # items: per graph -> (data, idx, heavy_idx[list], our_syms[list], nodes_to_motifs[list])
        items = []
        for data in ds:
            idx = int(data.src_idx)
            row = ctx_by_idx[split].get(idx)
            if row is None:
                raise KeyError(f"{split}: src_idx {idx} not in graph_context — handoff mismatch")
            fn_syms = list(data.atom_syms)
            our_syms = list(row["atom_syms"])                    # our graph: heavy atoms only
            heavy_idx = [i for i, s in enumerate(fn_syms) if s != "H"]
            heavy_syms = [fn_syms[i] for i in heavy_idx]
            if heavy_syms != our_syms:                           # atom<->atom verification
                j = next((k for k in range(min(len(heavy_syms), len(our_syms)))
                          if heavy_syms[k] != our_syms[k]), -1)
                raise AssertionError(
                    f"{split} src_idx {idx}: HEAVY-ATOM ELEMENT MISMATCH at {j} — FragNet heavy atoms "
                    f"({len(heavy_syms)}) differ from ours ({len(our_syms)}); mapping would be wrong.")
            items.append((data, idx, heavy_idx, our_syms, row["nodes_to_motifs"]))

        # --- PASS 1: base forward (attention + prediction), batched across molecules ---------------
        preds, atts_by_idx = {}, {}
        base_payload = [it[0] for it in items]
        for s, chunk, logits, per_atom, atom_batch in _run_batches(model, base_payload, batch_graphs, dev):
            for gi, (data, idx, heavy_idx, our_syms, _n2m) in enumerate(items[s:s + len(chunk)]):
                preds[idx] = float(torch.sigmoid(logits[gi]) if is_clf else logits[gi])
                a = per_atom[atom_batch == gi].numpy()           # this graph's atoms, FragNet order (incl H)
                if a.shape[0] != int(data.x_atoms.shape[0]):
                    raise AssertionError(
                        f"{split} src_idx {idx}: batched attention split gave {a.shape[0]} atoms != "
                        f"x_atoms {int(data.x_atoms.shape[0])} — batch['batch'] misalignment.")
                atts_by_idx[idx] = a[heavy_idx]                  # heavy-atom attention, our order

        # --- PASS 2: own-impact, masked forwards batched ACROSS molecules --------------------------
        oi_by_idx = {idx: {} for (_d, idx, _h, _s, _n) in items}
        if impact == "own":
            buf_data, buf_key = [], []                           # parallel: Data clone, (idx, mid)

            def _flush():
                if not buf_data:
                    return
                off = 0
                for _s2, chunk, logits, _pa, _ab in _run_batches(model, buf_data, batch_graphs, dev):
                    for k in range(len(chunk)):
                        idx, mid = buf_key[off + k]
                        oi_by_idx[idx][mid] = abs(
                            preds[idx] - float(torch.sigmoid(logits[k]) if is_clf else logits[k]))
                    off += len(chunk)
                buf_data.clear(); buf_key.clear()

            for (data, idx, heavy_idx, _our, n2m) in items:
                n2m_a = np.asarray(n2m, dtype=int)
                heavy = np.asarray(heavy_idx, dtype=int)
                if n2m_a.shape[0] != heavy.shape[0]:
                    raise AssertionError(f"{split} src_idx {idx}: nodes_to_motifs {n2m_a.shape[0]} "
                                         f"!= heavy atoms {heavy.shape[0]}")
                for mid in sorted({int(m) for m in n2m_a if m >= 0}):
                    rows = heavy[n2m_a == mid]                   # FragNet x_atoms rows for this motif
                    if rows.size == 0:
                        continue
                    d = data.clone()
                    d.x_atoms = d.x_atoms.clone()
                    d.x_atoms[torch.as_tensor(rows, dtype=torch.long)] = 0.0
                    buf_data.append(d); buf_key.append((idx, int(mid)))
                    if len(buf_data) >= batch_graphs:
                        _flush()
            _flush()

        # --- assemble ------------------------------------------------------------------------------
        out_split = {}
        for (data, idx, heavy_idx, our_syms, _n2m) in items:
            out_split[str(idx)] = {
                "atts": atts_by_idx[idx].tolist(), "atom_syms": our_syms, "pred": preds[idx],
                "own_impact": {str(m): v for m, v in oi_by_idx[idx].items()},
                "n_atoms": len(heavy_idx)}
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
    ap.add_argument("--batch_graphs", type=int, default=64, help="graphs per batched forward")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ft_ckpt = args.ft_ckpt or str(Path(args.work) / "ft.pt")
    export(args.graph_context, args.work, ft_ckpt, args.out, impact=args.impact,
           device=args.device, batch_graphs=args.batch_graphs)


if __name__ == "__main__":
    main()
