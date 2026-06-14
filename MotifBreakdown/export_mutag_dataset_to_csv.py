#!/usr/bin/env python3
"""export_mutag_dataset_to_csv.py — produce the mutag fold CSV + index-map pickle
the rest of the pipeline expects.

The vocab generator and the mutag training loader both need mutag converted from
its TUDataset/PyG form into:
    {out_dir}/mutag_{fold}.csv             (columns: smiles, label, group, ...)
    {out_dir}/mutag_{fold}_index_maps.pkl  ({mapped_smiles: {graph_idx: smiles_idx}})

This driver wraps SharedModules.data.graph_to_smiles.build_mutag_smiles_df (which
does the graph→mapped-SMILES conversion but does not save anything). Run it once
before `generate_vocab_rules.py --datasets mutag ...`.

Usage:
    python3 export_mutag_dataset_to_csv.py \
        --data_root <DIR containing mutag/>  \
        --out_dir   <FOLDS dir>              \
        --fold 0

The mutag PyG dataset is loaded via `datasets.mutag.Mutag(root=<data_root>/mutag)`
— the same class the training loader uses — so the graph order (and therefore the
labels and the index maps) matches what training will see.
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys
from pathlib import Path


def _load_mutag(data_root: str):
    """Load the mutag PyG dataset exactly as the training loader does."""
    # datasets.mutag lives under the repo's datasets/ package; make sure it's importable.
    here = Path(__file__).resolve()
    for cand in (here.parents[1],            # repo root (…/SharedModules/.. )
                 here.parents[1] / 'SharedModules',
                 Path(data_root)):
        p = str(cand)
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from datasets.mutag import Mutag
    except Exception as e:
        raise ImportError(
            "Could not import datasets.mutag.Mutag. Run this script from the repo "
            "root (so the 'datasets' package is importable), or add it to "
            f"PYTHONPATH. Original error: {e}")
    return Mutag(root=str(Path(data_root) / 'mutag'))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_root', required=True,
                   help="Directory containing the mutag/ TUDataset folder")
    p.add_argument('--out_dir', required=True,
                   help="FOLDS dir where mutag_{fold}.csv + _index_maps.pkl are written")
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--split_name', default='training',
                   help="Value for the CSV 'group' column. Mutag has no canonical "
                        "split here; default 'training' (the vocab pipeline only "
                        "needs the group column to exist).")
    p.add_argument('--no_verify', action='store_true',
                   help="Skip per-graph index-alignment verification (faster).")
    args = p.parse_args()

    # import the converter from SharedModules
    here = Path(__file__).resolve()
    sm = here.parents[1] / 'SharedModules'
    if str(sm) not in sys.path:
        sys.path.insert(0, str(sm))
    if str(here.parents[1]) not in sys.path:
        sys.path.insert(0, str(here.parents[1]))
    from SharedModules.data.graph_to_smiles import build_mutag_smiles_df

    dataset = _load_mutag(args.data_root)
    print(f"Loaded mutag: {len(dataset)} graphs from {args.data_root}/mutag")

    df, index_maps = build_mutag_smiles_df(
        dataset, split_name=args.split_name, verify=not args.no_verify)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"mutag_{args.fold}.csv"
    pkl_path = out_dir / f"mutag_{args.fold}_index_maps.pkl"

    # Drop rows that failed SMILES conversion so the vocab pipeline never sees a
    # null SMILES (the index map already only contains successful conversions).
    n_before = len(df)
    df_ok = df[df['conversion_ok']].copy()
    if len(df_ok) < n_before:
        print(f"  dropped {n_before - len(df_ok)} graphs that failed conversion")

    df_ok.to_csv(csv_path, index=False)
    with open(pkl_path, 'wb') as f:
        pickle.dump(index_maps, f)

    n_verify_fail = int((~df_ok['verify_ok']).sum()) if 'verify_ok' in df_ok else 0
    print(f"  wrote {csv_path}  ({len(df_ok)} molecules)")
    print(f"  wrote {pkl_path}  ({len(index_maps)} index maps)")
    if n_verify_fail:
        print(f"  [warn] {n_verify_fail} graphs failed index-alignment verification; "
              f"inspect before trusting their motif annotations.")


if __name__ == '__main__':
    main()
