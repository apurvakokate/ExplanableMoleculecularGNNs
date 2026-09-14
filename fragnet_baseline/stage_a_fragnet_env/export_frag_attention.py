"""Stage A (3b) — PER-LAYER native attention export (atom + fragment) + per-motif OWN-IMPACT.

Runs in the FragNet env. For each graph (keyed by our src_idx) it emits FragNet's native attention at
EVERY message-passing layer (atom-level and fragment-level, no cross-layer aggregation), the
fragment->motif map, the prediction, AND the per-motif own-impact.

Two evaluation axes (computed in Stage B from this file):
  * GT-ROC (correctness): attention scores vs ground-truth atoms/motifs — attention only, NOT impact.
  * Pearson (faithfulness): per-layer attention score vs OWN-IMPACT. own_impact is the ONLY use of the
    masked re-forward here; it is never used as the importance for GT-ROC.

own_impact[motif] = |p_full - p_masked|, zeroing the motif's HEAVY-atom feature rows (layer-independent;
mirrors the atom-level path). Masked clones are pooled ACROSS molecules and flushed in batches of
batch_graphs (the export_attention.py Path-A batching), so own-impact fills the GPU; each clone is
still an independent single-motif mask, so only the number of forwards changes. The attention pass
stays batch-1 per graph (keeps the per-layer/per-graph attention split trivially correct).

Fragment attention is meaningful only for a model finetuned with frag_type='custom' (FragNet fragments
== our rbrics motifs). frag_to_motif maps each fragment to its motif; repeated types re-aggregate to
type in Stage B.

FragNetViz is UNMODIFIED: we replicate its forward here (flipping return_attentions at runtime) to read
every layer. The loop mirrors fragnet.vizualize.model.FragNetViz.forward exactly — validated by the
self-check (last layer == the stock model prediction).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import export_attention as EA          # reuse _read_model_cfg, _load_model, _collate (same dir)


def _collate(model, datas, dev):
    """collate a LIST of graphs, fixing the single-connection degeneracy: a molecule with one
    fragment-bond connection has 0 fbond-graph edges, whose edge-attr FragNet stores as an empty 1-D
    tensor (0,). That crashes the fbond edge Linear (0-feature input). Batched collate normally absorbs
    it into a proper (E, F) tensor; when the WHOLE list is degenerate it stays (0,), so we reshape to
    (0, F). Identical to the batched path (0 fbond edges -> zero fbond messages). No FragNet change."""
    b = EA._collate(datas, dev)
    e = b.get("edge_attr_fbonds")
    if torch.is_tensor(e) and e.dim() < 2:
        F = model.pretrain.layers[0].edge_attr_fbond_embed.in_features
        b["edge_attr_fbonds"] = e.new_zeros((0, F))
    return b


def _reduce_heads(a) -> np.ndarray:
    """Sum an attention tensor over head/trailing dims -> 1D per-node scalar (FragNet's
    vizualize_atom_weights convention: summed_attn_weights.sum(1))."""
    a = a.detach().cpu().float()
    if a.dim() > 1:
        a = a.sum(dim=tuple(range(1, a.dim())))
    return a.view(-1).numpy()


@torch.no_grad()
def _forward_all_layers(viz, batch):
    """Replicate fragnet.vizualize.model.FragNetViz.forward, capturing every layer's atom and fragment
    attention. Returns (per_layer_atom[list], per_layer_frag[list], x_atoms, x_frags) — x_atoms/x_frags
    are the final post-activation embeddings (for the read-out prediction)."""
    xa = batch["x_atoms"]; ei = batch["edge_index"]; fi = batch["frag_index"]
    xf = batch["x_frags"]; ea = batch["edge_attr"]; a2f = batch["atom_to_frag_ids"]
    nfb = batch["node_features_bonds"]; eib = batch["edge_index_bonds_graph"]; eab = batch["edge_attr_bonds"]
    nffb = batch["node_features_fbonds"]; eifb = batch["edge_index_fbonds"]; eafb = batch["edge_attr_fbonds"]
    act = viz.act
    for L in viz.layers:
        L.return_attentions = True     # runtime flip only — FragNetViz source unchanged
    atom_attn, frag_attn = [], []
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
    """Single-graph read-out prediction (replicates FragNetFineTuneViz.forward: sum-pool + FTHead)."""
    from torch_scatter import scatter_add
    xap = scatter_add(x_atoms, batch["batch"], dim=0)
    xfp = scatter_add(x_frags, batch["frag_batch"], dim=0)
    logit = model.fthead(torch.cat((xap, xfp), 1)).view(-1)
    return float(torch.sigmoid(logit[0]))


@torch.no_grad()
def _predict_batch(model, batch, x_atoms, x_frags) -> np.ndarray:
    """Batched read-out predictions -> [num_graphs] probabilities."""
    from torch_scatter import scatter_add
    xap = scatter_add(x_atoms, batch["batch"], dim=0)
    xfp = scatter_add(x_frags, batch["frag_batch"], dim=0)
    logit = model.fthead(torch.cat((xap, xfp), 1)).view(-1)
    return torch.sigmoid(logit).detach().cpu().numpy()


@torch.no_grad()
def _flush_own_impact(model, viz, buf_data, buf_key, preds, oi_by_idx, dev) -> None:
    """Run ONE buffered batch of single-motif-masked clones, POOLED ACROSS molecules (each clone has
    exactly one motif's heavy rows zeroed), and record own_impact[idx][mid] = |preds[idx] - p_masked|
    for every clone. buf_data / buf_key are parallel; both are cleared in place. This is the Path-A
    own-impact batching (export_attention.py) applied to the fragment export: it fills the GPU instead
    of one small forward per molecule, and is numerically identical (each clone is still an independent
    single-motif mask; only the number of forwards changes)."""
    if not buf_data:
        return
    b = _collate(model, buf_data, dev)
    _, _, xa, xf = _forward_all_layers(viz, b)
    pm = _predict_batch(model, b, xa, xf)                    # [len(buf_data)] in buffer order
    for k, (idx, mid) in enumerate(buf_key):
        oi_by_idx[idx][mid] = abs(preds[idx] - float(pm[k]))
    buf_data.clear(); buf_key.clear()


@torch.no_grad()
def _self_check(model, viz, sample, dev, tol: float = 1e-4) -> None:
    """FIRST-RUN GUARD: our manual per-layer read-out MUST equal FragNet's stock forward. Compute the
    stock predictions FIRST (layers in stock config), THEN the manual ones (which flip return_attentions
    on all layers), and assert they match."""
    theirs = []
    for data in sample:
        b = _collate(model, [data], dev)
        out = model(b)
        theirs.append(float(torch.sigmoid(out[0].view(-1)[0])))
    for k, data in enumerate(sample):
        b = _collate(model, [data], dev)
        _, _, x_atoms, x_frags = _forward_all_layers(viz, b)
        ours = _predict(model, b, x_atoms, x_frags)
        if abs(ours - theirs[k]) > tol:
            raise AssertionError(
                f"SELF-CHECK FAILED on sample {k}: manual read-out {ours:.6f} != model(batch) "
                f"{theirs[k]:.6f} (|diff|={abs(ours - theirs[k]):.2e} > {tol}).")
    print(f"[frag_export] SELF-CHECK OK: manual read-out == model(batch) on {len(sample)} graphs (tol {tol})")


def export(graph_context: str, work: str, ft_ckpt: str, out_path: str, device: str = "auto",
           batch_graphs: int = 64) -> None:
    from fragnet.dataset.dataset import load_pickle_dataset
    dev = torch.device("cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available()))
                       else "cpu")
    print(f"[frag_export] device={dev} batch_graphs={batch_graphs}")
    ctx = json.loads(Path(graph_context).read_text())
    ctx_by_idx = {s: {int(r["idx"]): r for r in (ctx.get(s) or [])} for s in ("train", "valid", "test")}

    cfg = EA._read_model_cfg(work)
    model = EA._load_model(ft_ckpt, dev, cfg)
    viz = model.pretrain
    n_layers = len(viz.layers)
    print(f"[frag_export] layers={n_layers}")

    work = Path(work)
    pkl_name = {"train": "train.pkl", "valid": "val.pkl", "test": "test.pkl"}
    for _s, _fn in pkl_name.items():
        _p = work / _fn
        if _p.exists():
            _self_check(model, viz, load_pickle_dataset(str(_p))[:3], dev)
            break
    neutral = {}
    for split, fn in pkl_name.items():
        p = work / fn
        if not p.exists():
            raise FileNotFoundError(f"missing FragNet pkl {p} — run prep_data.py/finetune first")
        ds = load_pickle_dataset(str(p))
        out_split = {}
        preds, oi_by_idx = {}, {}
        buf_data, buf_key = [], []        # cross-molecule buffer of single-motif-masked clones
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
            if heavy_syms != our_syms:
                j = next((k for k in range(min(len(heavy_syms), len(our_syms)))
                          if heavy_syms[k] != our_syms[k]), -1)
                raise AssertionError(
                    f"{split} src_idx {idx}: HEAVY-ATOM ELEMENT MISMATCH at {j} — attention misaligned.")
            if len(n2m) != len(heavy):
                raise AssertionError(
                    f"{split} src_idx {idx}: nodes_to_motifs {len(n2m)} != heavy atoms {len(heavy)}")

            batch = _collate(model, [data], dev)
            atom_attn, frag_attn, x_atoms, x_frags = _forward_all_layers(viz, batch)
            pred = _predict(model, batch, x_atoms, x_frags)
            preds[idx] = pred; oi_by_idx[idx] = {}

            atom_by_layer = {}
            for L in range(n_layers):
                a = _reduce_heads(atom_attn[L])
                if a.shape[0] != int(data.x_atoms.shape[0]):
                    raise AssertionError(
                        f"{split} src_idx {idx}: layer {L} atom-attn {a.shape[0]} != x_atoms "
                        f"{int(data.x_atoms.shape[0])}")
                atom_by_layer[str(L)] = a[heavy].tolist()

            a2f_t = batch["atom_to_frag_ids"]
            a2f = (a2f_t.detach().cpu().numpy() if torch.is_tensor(a2f_t) else np.asarray(a2f_t)).astype(int).reshape(-1)
            n_frags = int(data.x_frags.shape[0])
            frag_to_motif = {}
            for k, i in enumerate(heavy):
                fid = str(int(a2f[i])); mid = int(n2m[k])
                if frag_to_motif.get(fid, mid) != mid:      # fragment already claimed by another motif
                    raise AssertionError(
                        f"{split} src_idx {idx}: fragment {fid} spans motifs {frag_to_motif[fid]} and "
                        f"{mid} — FragNet fragmentation != our rbrics motifs (custom-alignment premise "
                        f"violated); this fragment's attention cannot be attributed to a single motif.")
                frag_to_motif[fid] = mid

            frag_by_layer = {}
            for L in range(n_layers):
                f = _reduce_heads(frag_attn[L])
                if f.shape[0] != n_frags:
                    raise AssertionError(
                        f"{split} src_idx {idx}: layer {L} frag-attn {f.shape[0]} != n_frags {n_frags}")
                frag_by_layer[str(L)] = f.tolist()

            # own_impact is attached after the split's buffer is fully flushed (below)
            out_split[str(idx)] = {
                "pred": pred, "atom_syms": our_syms, "n_atoms": len(heavy), "n_frags": n_frags,
                "frag_to_motif": frag_to_motif, "own_impact": None,
                "atom_att_by_layer": atom_by_layer, "frag_att_by_layer": frag_by_layer}

            # enqueue this molecule's single-motif-masked clones (exactly one motif zeroed each)
            n2m_a = np.asarray(n2m, dtype=int); heavy_a = np.asarray(heavy, dtype=int)
            for mid in sorted({int(m) for m in n2m_a if m >= 0}):
                rows = heavy_a[n2m_a == mid]
                if rows.size == 0:
                    continue
                d = data.clone(); d.x_atoms = d.x_atoms.clone()
                d.x_atoms[torch.as_tensor(rows, dtype=torch.long)] = 0.0
                buf_data.append(d); buf_key.append((idx, int(mid)))
                if len(buf_data) >= batch_graphs:
                    _flush_own_impact(model, viz, buf_data, buf_key, preds, oi_by_idx, dev)
        _flush_own_impact(model, viz, buf_data, buf_key, preds, oi_by_idx, dev)   # tail of split
        for idx_str, rec in out_split.items():
            rec["own_impact"] = {str(m): v for m, v in oi_by_idx[int(idx_str)].items()}
        neutral[split] = out_split
        print(f"[frag_export] {split}: {len(out_split)} graphs")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(neutral))
    print(f"[frag_export] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Stage A/3b: FragNet per-layer atom+fragment attention + own-impact")
    ap.add_argument("--work", required=True)
    ap.add_argument("--vendor", required=True, help="vendored FragNet (CLI parity; fragnet is pip-installed)")
    ap.add_argument("--graph_context", required=True)
    ap.add_argument("--ft_ckpt", default=None, help="default: <work>/ft.pt")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch_graphs", type=int, default=64,
                    help="own-impact masked clones per batched forward (attention pass stays batch-1)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ft_ckpt = args.ft_ckpt or str(Path(args.work) / "ft.pt")
    export(args.graph_context, args.work, ft_ckpt, args.out, device=args.device,
           batch_graphs=args.batch_graphs)


if __name__ == "__main__":
    main()
