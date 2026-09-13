"""Stage A (1/3) — runs in the FragNet env. graph_context.json -> FragNet .pkl datasets.

We featurize each molecule with FragNet's OWN pipeline (get_3Dcoords + CreateData.create_data_point,
via FinetuneData) so features + 3D-conformer embedding match FragNet's paper runs exactly — BUT we
drive the loop ourselves so we can attach, to each Data:
  * src_idx   — OUR split-local graph index (the stable key; survives duplicate SMILES and drops)
  * atom_syms — per-atom element symbols in FragNet's x_atoms order (for atom↔atom verification)

graph_context.json (from dump_context, l2xgnn env) gives, per split, each graph's VERBATIM smiles +
y (relabelled for planted) + idx. We featurize from the verbatim smiles so FragNet's heavy-atom
order equals ours. The pkl is a pickled list of Data (exactly what load_pickle_dataset expects).

No fallbacks: a dropped molecule (conformer/featurization failure) is recorded and later shows up as
a missing src_idx at alignment (loud), never silently skipped into a wrong slot.
"""
import argparse
import json
import pickle
from pathlib import Path

# FragNet names the validation split "val".
_SPLIT_OUT = {"train": "train", "valid": "val", "test": "test"}


def featurize_split(rows, create_data, get_3Dcoords, frag_type):
    """Featurize one split's rows IN ORDER. Returns (list[Data], dropped_idxs)."""
    out, dropped = [], []
    for r in rows:
        idx, smiles, y = int(r["idx"]), r["smiles"], r["y"]
        mol = get_3Dcoords(smiles)                      # FragNet's own 3D embed (None on failure)
        if mol is None:
            dropped.append(idx); continue
        try:
            conf = mol.GetConformer(id=0)
        except Exception:
            dropped.append(idx); continue
        data = create_data.create_data_point([smiles, y, mol, conf, frag_type])
        if data is None:                                 # no-edge / bad mol — FragNet drops it
            dropped.append(idx); continue
        data.src_idx = idx
        data.atom_syms = [a.GetSymbol() for a in mol.GetAtoms()]
        n_x = int(data.x_atoms.shape[0])
        if len(data.atom_syms) != n_x:
            # x_atoms and our symbol list disagree -> FragNet kept Hs or reordered; fail loud so the
            # atom↔atom verification in Stage B is never fed a misaligned symbol list.
            raise AssertionError(
                f"src_idx {idx}: atom_syms {len(data.atom_syms)} != x_atoms {n_x} "
                f"(get_3Dcoords may have kept explicit H; inspect before trusting attention).")
        out.append(data)
    return out, dropped


def main():
    ap = argparse.ArgumentParser(description="Stage A/1: graph_context.json -> FragNet pkls")
    ap.add_argument("--graph_context", required=True, help="graph_context.json from dump_context")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--vendor", required=True, help="path to vendored pnnl/FragNet repo (on sys.path)")
    ap.add_argument("--data_type", default="exp1s", help="feature scheme pt.pt was pretrained on")
    ap.add_argument("--frag_type", default="brics")
    args = ap.parse_args()

    # FragNet must be importable (pip install -e vendor/FragNet, per run_poc.sh setup).
    from fragnet.dataset.dataset import FinetuneData
    from fragnet.dataset.fragments import get_3Dcoords

    fd = FinetuneData(target_name="y", data_type=args.data_type, frag_type=args.frag_type)
    create_data = fd.create_data

    ctx = json.loads(Path(args.graph_context).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        rows = ctx.get(split) or []
        if not rows:
            raise ValueError(f"graph_context has no '{split}' rows")
        ds, dropped = featurize_split(rows, create_data, get_3Dcoords, args.frag_type)
        pkl = out_dir / f"{_SPLIT_OUT[split]}.pkl"
        with open(pkl, "wb") as f:
            pickle.dump(ds, f)
        msg = f"[prep_data] {split}: featurized {len(ds)}/{len(rows)} -> {pkl}"
        if dropped:
            msg += f"  (DROPPED {len(dropped)} src_idx: {dropped[:10]}{'...' if len(dropped) > 10 else ''})"
        print(msg)


if __name__ == "__main__":
    main()
