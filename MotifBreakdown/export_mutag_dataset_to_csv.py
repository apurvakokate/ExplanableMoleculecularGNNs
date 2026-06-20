#!/usr/bin/env python3
"""export_mutag_dataset_to_csv.py — produce the mutag fold CSV + index-map pickle
the rest of the pipeline expects.

The vocab generator and the mutag training loader both need mutag converted from
its TUDataset/PyG form into:
    {out_dir}/mutag_{fold}.csv             (columns: smiles, label, group, ...)
    {out_dir}/mutag_{fold}_index_maps.pkl  ({mapped_smiles: {graph_idx: smiles_idx}})
    {out_dir}/mutag_{fold}_splits.pkl      (GSAT-style train/valid/test indices)

Split logic mirrors Graph-COM/GSAT (``mutag_x=True`` by default):
    80% train / 20% valid; test = all graphs with y==0 and edge_label.sum()>0.

Usage:
    python3 export_mutag_dataset_to_csv.py \\
        --data_root <DIR containing mutag/>  \\
        --out_dir   <FOLDS dir>              \\
        --fold 0 --seed 42

Run once before ``generate_vocab_rules.py --datasets mutag ...``.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


def _load_mutag(data_root: str):
    """Load the mutag PyG dataset exactly as the training loader does."""
    here = Path(__file__).resolve()
    for cand in (here.parents[1],
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
                   help="FOLDS dir where mutag_{fold}.* artifacts are written")
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--seed', type=int, default=42,
                   help="RNG seed for the shuffle (use seed+fold for multi-fold)")
    p.add_argument('--no_mutag_x', action='store_true',
                   help="Standard 80/10/10 random split instead of GSAT mutag_x")
    p.add_argument('--no_verify', action='store_true',
                   help="Skip per-graph index-alignment verification (faster).")
    args = p.parse_args()

    here = Path(__file__).resolve()
    sm = here.parents[1] / 'SharedModules'
    if str(sm) not in sys.path:
        sys.path.insert(0, str(sm))
    if str(here.parents[1]) not in sys.path:
        sys.path.insert(0, str(here.parents[1]))

    from SharedModules.data.graph_to_smiles import build_mutag_smiles_df
    from SharedModules.data.mutag_splits import (
        get_mutag_split_idx, group_for_graph, save_mutag_splits,
    )

    dataset = _load_mutag(args.data_root)
    split_seed = args.seed + args.fold
    mutag_x = not args.no_mutag_x
    split_idx = get_mutag_split_idx(
        dataset, seed=split_seed, mutag_x=mutag_x)
    groups = [
        group_for_graph(i, split_idx, mutag_x=mutag_x)
        for i in range(len(dataset))
    ]

    mode = 'mutag_x (GSAT)' if mutag_x else 'standard 80/10/10'
    print(f"Loaded mutag: {len(dataset)} graphs from {args.data_root}/mutag")
    print(f"  split mode: {mode}  seed={split_seed}")
    print(f"  train={len(split_idx['train'])}  valid={len(split_idx['valid'])}  "
          f"test={len(split_idx['test'])}")

    df, index_maps = build_mutag_smiles_df(
        dataset, groups=groups, verify=not args.no_verify)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"mutag_{args.fold}.csv"
    pkl_path = out_dir / f"mutag_{args.fold}_index_maps.pkl"
    splits_path = out_dir / f"mutag_{args.fold}_splits.pkl"

    n_before = len(df)
    df_ok = df[df['conversion_ok']].copy()
    if len(df_ok) < n_before:
        print(f"  dropped {n_before - len(df_ok)} graphs that failed conversion")

    df_ok.to_csv(csv_path, index=False)
    with open(pkl_path, 'wb') as f:
        pickle.dump(index_maps, f)
    save_mutag_splits(
        splits_path, split_idx, seed=split_seed, mutag_x=mutag_x)

    n_verify_fail = int((~df_ok['verify_ok']).sum()) if 'verify_ok' in df_ok else 0
    print(f"  wrote {csv_path}  ({len(df_ok)} molecules)")
    print(f"  wrote {pkl_path}  ({len(index_maps)} index maps)")
    print(f"  wrote {splits_path}")
    if n_verify_fail:
        print(f"  [warn] {n_verify_fail} graphs failed index-alignment verification; "
              f"inspect before trusting their motif annotations.")


if __name__ == '__main__':
    main()
