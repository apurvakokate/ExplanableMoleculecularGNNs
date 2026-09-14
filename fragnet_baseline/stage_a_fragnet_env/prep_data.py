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


def _frag_bonds_from_motifs(mol, nodes_to_motifs):
    """Bonds to cut so FragNet's fragments equal OUR rbrics motifs: every HEAVY-HEAVY bond whose two
    atoms are in DIFFERENT motifs. Heavy atoms are taken in RDKit order and aligned 1:1 to
    nodes_to_motifs (our heavy-atom graph order == FragNet's heavy-atom order; get_3Dcoords appends H).
    Cutting exactly these bonds -> connected components == our motifs (proven: exact when a motif type
    is not repeated in the molecule; otherwise FragNet splits a repeated type into its connected
    instances, which we later re-aggregate to the motif type). Returns [(a1, a2), ...] for frag_type
    'custom' — the same atom-index-pair format FragNet's brics branch produces."""
    n2m = list(nodes_to_motifs)
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
    if len(heavy) != len(n2m):
        raise AssertionError(
            f"nodes_to_motifs {len(n2m)} != heavy atoms {len(heavy)} — heavy-atom alignment broken; "
            f"custom fragmentation would cut the wrong bonds.")
    motif_of = {heavy[i]: n2m[i] for i in range(len(heavy))}
    fb = []
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a1 in motif_of and a2 in motif_of and motif_of[a1] != motif_of[a2]:
            fb.append((a1, a2))
    return fb


def featurize_split(rows, create_data, get_3Dcoords, frag_type):
    """Featurize one split's rows IN ORDER. Returns (list[Data], dropped_idxs)."""
    out, dropped = [], []
    for r in rows:
        idx, smiles, y = int(r["idx"]), r["smiles"], r["y"]
        # get_3Dcoords may RETURN None or RAISE: a SMILES RDKit can't sanitize (e.g. hypervalent N in
        # Mutagenicity/hERG nitro compounds) becomes None, and FragNet's AddHs(None) then throws. Either
        # way this molecule cannot be featurized -> DROP it (counted), never let it abort the whole split.
        try:
            mol = get_3Dcoords(smiles)
        except Exception:
            dropped.append(idx); continue
        if mol is None:
            dropped.append(idx); continue
        try:
            conf = mol.GetConformer(id=0)
        except Exception:
            dropped.append(idx); continue
        if frag_type == "custom":
            # align FragNet's fragment graph to OUR rbrics motifs — pass the cut-bonds derived from
            # nodes_to_motifs; FragNet's fragment/connection machinery is otherwise unchanged.
            if "nodes_to_motifs" not in r:
                raise KeyError(f"src_idx {idx}: frag_type='custom' needs 'nodes_to_motifs' in graph_context")
            frag_bonds = _frag_bonds_from_motifs(mol, r["nodes_to_motifs"])
            data = create_data.create_data_point([smiles, y, mol, conf, frag_type, frag_bonds])
        else:
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
    ap.add_argument("--frag_type", default="brics",
                    help="brics|murcko (FragNet native) or 'custom' = align FragNet's fragment graph "
                         "to OUR rbrics motifs (cut bonds derived from graph_context nodes_to_motifs)")
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
        pkl = out_dir / f"{_SPLIT_OUT[split]}.pkl"
        # Reuse a previously-featurized pkl (conformer embedding is the ~19-min/fold bottleneck).
        # pickle.dump runs once at the end of a split, so an existing non-empty pkl = a complete split.
        if pkl.exists() and pkl.stat().st_size > 0:
            print(f"[prep_data] {split}: reuse cached {pkl} ({pkl.stat().st_size} bytes) — skip featurization")
            continue
        rows = ctx.get(split) or []
        if not rows:
            raise ValueError(f"graph_context has no '{split}' rows")
        ds, dropped = featurize_split(rows, create_data, get_3Dcoords, args.frag_type)
        with open(pkl, "wb") as f:
            pickle.dump(ds, f)
        msg = f"[prep_data] {split}: featurized {len(ds)}/{len(rows)} -> {pkl}"
        if dropped:
            msg += f"  (DROPPED {len(dropped)} src_idx: {dropped[:10]}{'...' if len(dropped) > 10 else ''})"
        print(msg)


if __name__ == "__main__":
    main()
