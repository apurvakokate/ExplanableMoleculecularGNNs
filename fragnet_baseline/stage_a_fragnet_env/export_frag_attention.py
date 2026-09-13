"""Stage A (3b) — PER-LAYER native attention export (atom + fragment), for the attention-as-importance
evaluation. Runs in the FragNet env.

We evaluate FragNet's ATTENTION scores themselves as importance (prediction-decoupled) — NOT the
contribution (Property^unmasked - Property^masked), which measures the masking's effect on the output
rather than the quality of the scores. So there is no own-impact / masked re-forward here.

For each graph (keyed by OUR src_idx) we emit FragNet's native attention at EVERY message-passing layer
(no cross-layer aggregation — each layer is a distinct signal; last-layer attention did not localise
while an intermediate layer did, so layers must be reported separately):

  {split: {str(src_idx): {
       "pred": float,
       "atom_syms": [heavy-atom element, our order],
       "n_atoms": int,               # heavy atoms
       "n_frags": int,               # FragNet fragments (== our motif INSTANCES under frag_type=custom)
       "frag_to_motif": {str(frag_idx): motif_id},   # each fragment -> our rbrics motif
       "atom_att_by_layer": {str(layer): [per heavy atom, our order]},
       "frag_att_by_layer": {str(layer): [per fragment]},
  }}}

FRAGMENT-level attention is meaningful only when the model was finetuned with frag_type='custom'
(prep_data --frag_type custom), so FragNet's fragment graph == our rbrics motifs. `frag_to_motif` then
maps each FragNet fragment to its motif (all heavy atoms in a fragment share one motif by construction);
a repeated motif type appears as several fragments and is re-aggregated to the type in Stage B.

FragNetViz is UNMODIFIED: we replicate its forward here (flipping return_attentions at runtime) so we
can read every layer. The loop mirrors fragnet.vizualize.model.FragNetViz.forward exactly — validated:
the last layer reproduces the native pipeline number.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import export_attention as EA          # reuse _read_model_cfg, _load_model, _collate (same dir)


def _reduce_heads(a) -> np.ndarray:
    """Sum an attention tensor over its head/trailing dims -> 1D per-node scalar (FragNet's
    vizualize_atom_weights convention: summed_attn_weights.sum(1))."""
    a = a.detach().cpu().float()
    if a.dim() > 1:
        a = a.sum(dim=tuple(range(1, a.dim())))
    return a.view(-1).numpy()


@torch.no_grad()
def _forward_all_layers(viz, batch):
    """Replicate fragnet.vizualize.model.FragNetViz.forward, capturing EVERY layer's atom and fragment
    attention. Returns (per_layer_atom_attn[list], per_layer_frag_attn[list], x_atoms, x_frags) where
    x_atoms/x_frags are the final post-activation embeddings (for the read-out prediction)."""
    xa = batch["x_atoms"]; ei = batch["edge_index"]; fi = batch["frag_index"]
    xf = batch["x_frags"]; ea = batch["edge_attr"]; a2f = batch["atom_to_frag_ids"]
    nfb = batch["node_features_bonds"]; eib = batch["edge_index_bonds_graph"]; eab = batch["edge_attr_bonds"]
    nffb = batch["node_features_fbonds"]; eifb = batch["edge_index_fbonds"]; eafb = batch["edge_attr_fbonds"]
    act = viz.act
    for L in viz.layers:
        L.return_attentions = True     # runtime flip only — FragNetViz source is unchanged
    atom_attn, frag_attn = [], []
    # layer 0 consumes the raw bond / fbond graph features (matches FragNetViz.forward exactly)
    o = viz.layers[0](xa, ei, ea, fi, xf, a2f, nfb, eib, eab, nffb, eifb, eafb)
    xa, xf, edf, fedf = o[0], o[1], o[2], o[3]
    atom_attn.append(o[4]); frag_attn.append(o[5])
    xa, xf = act(xa), act(xf); edf = act(edf); fedf = act(fedf)
    for L in viz.layers[1:]:
        o = L(xa, ei, edf, fi, xf, a2f, edf, eib, eab, fedf, eifb, eafb)
        xa, xf, edf, fedf = o[0], o[1], o[2], o[3]
        atom_attn.append(o[4]); frag_attn.append(o[5])
        xa, xf = act(xa), act(xf); edf = act(edf); fedf = act(fedf)
    return atom_attn, frag_attn, xa, xf


@torch.no_grad()
def _predict(model, batch, x_atoms, x_frags) -> float:
    """Read-out prediction, replicating FragNetFineTuneViz.forward: sum-pool atoms and frags, concat,
    FTHead. Uses the SAME final embeddings the per-layer forward produced (no second forward)."""
    from torch_scatter import scatter_add
    x_atoms_pooled = scatter_add(x_atoms, batch["batch"], dim=0)
    x_frags_pooled = scatter_add(x_frags, batch["frag_batch"], dim=0)
    cat = torch.cat((x_atoms_pooled, x_frags_pooled), 1)
    logit = model.fthead(cat).view(-1)
    return float(torch.sigmoid(logit[0]))


def export(graph_context: str, work: str, ft_ckpt: str, out_path: str, device: str = "auto") -> None:
    from fragnet.dataset.dataset import load_pickle_dataset
    dev = torch.device("cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available()))
                       else "cpu")
    print(f"[frag_export] device={dev}")
    ctx = json.loads(Path(graph_context).read_text())
    ctx_by_idx = {s: {int(r["idx"]): r for r in (ctx.get(s) or [])} for s in ("train", "valid", "test")}

    cfg = EA._read_model_cfg(work)
    model = EA._load_model(ft_ckpt, dev, cfg)
    viz = model.pretrain
    n_layers = len(viz.layers)
    print(f"[frag_export] layers={n_layers}")

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
            our_syms = list(row["atom_syms"])
            n2m = list(row["nodes_to_motifs"])
            fn_syms = list(data.atom_syms)
            heavy = [i for i, s in enumerate(fn_syms) if s != "H"]
            heavy_syms = [fn_syms[i] for i in heavy]
            if heavy_syms != our_syms:                     # atom<->atom verification (heavy atoms)
                j = next((k for k in range(min(len(heavy_syms), len(our_syms)))
                          if heavy_syms[k] != our_syms[k]), -1)
                raise AssertionError(
                    f"{split} src_idx {idx}: HEAVY-ATOM ELEMENT MISMATCH at {j} — attention would be "
                    f"misaligned.")
            if len(n2m) != len(heavy):
                raise AssertionError(
                    f"{split} src_idx {idx}: nodes_to_motifs {len(n2m)} != heavy atoms {len(heavy)}")

            batch = EA._collate([data], dev)
            atom_attn, frag_attn, x_atoms, x_frags = _forward_all_layers(viz, batch)
            pred = _predict(model, batch, x_atoms, x_frags)

            # per-layer atom attention, heavy subset in our order
            atom_by_layer = {}
            for L in range(n_layers):
                a = _reduce_heads(atom_attn[L])
                if a.shape[0] != int(data.x_atoms.shape[0]):
                    raise AssertionError(
                        f"{split} src_idx {idx}: layer {L} atom-attn {a.shape[0]} != x_atoms "
                        f"{int(data.x_atoms.shape[0])}")
                atom_by_layer[str(L)] = a[heavy].tolist()

            # fragment -> our motif (each fragment's heavy atoms share one motif by construction).
            # Use the COLLATED atom->fragment map (raw Data stores it as atom_id_frag_id; collate_fn
            # exposes it as atom_to_frag_ids — exactly what the forward consumed). batch-1: indices are
            # 0..n_atoms-1 / 0..n_frags-1, so no batch-offset bookkeeping.
            a2f_t = batch["atom_to_frag_ids"]
            a2f = (a2f_t.detach().cpu().numpy() if torch.is_tensor(a2f_t) else np.asarray(a2f_t)).astype(int).reshape(-1)
            n_frags = int(data.x_frags.shape[0])
            frag_to_motif = {}
            for k, i in enumerate(heavy):                  # i = FragNet atom index of our heavy atom k
                fid = int(a2f[i])
                frag_to_motif.setdefault(str(fid), int(n2m[k]))

            # per-layer fragment attention (one scalar per fragment)
            frag_by_layer = {}
            for L in range(n_layers):
                f = _reduce_heads(frag_attn[L])
                if f.shape[0] != n_frags:
                    raise AssertionError(
                        f"{split} src_idx {idx}: layer {L} frag-attn {f.shape[0]} != n_frags {n_frags}")
                frag_by_layer[str(L)] = f.tolist()

            out_split[str(idx)] = {
                "pred": pred, "atom_syms": our_syms, "n_atoms": len(heavy), "n_frags": n_frags,
                "frag_to_motif": frag_to_motif,
                "atom_att_by_layer": atom_by_layer, "frag_att_by_layer": frag_by_layer}
        neutral[split] = out_split
        print(f"[frag_export] {split}: {len(out_split)} graphs")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(neutral))
    print(f"[frag_export] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Stage A/3b: FragNet per-layer atom+fragment attention")
    ap.add_argument("--work", required=True)
    ap.add_argument("--vendor", required=True, help="vendored FragNet (CLI parity; fragnet is pip-installed)")
    ap.add_argument("--graph_context", required=True)
    ap.add_argument("--ft_ckpt", default=None, help="default: <work>/ft.pt")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ft_ckpt = args.ft_ckpt or str(Path(args.work) / "ft.pt")
    export(args.graph_context, args.work, ft_ckpt, args.out, device=args.device)


if __name__ == "__main__":
    main()
