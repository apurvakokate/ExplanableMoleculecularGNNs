#!/usr/bin/env python3
"""export_mutag_dataset_to_csv.py — produce the mutag fold CSV + index-map pickle
the rest of the pipeline expects.

The vocab generator and the mutag training loader both need mutag converted from
its TUDataset/PyG form into:
    {out_dir}/mutag_{fold}.csv             (columns: smiles, label, group, ...)
    {out_dir}/mutag_{fold}_index_maps.pkl  ({mapped_smiles: {graph_idx: smiles_idx}})
    {out_dir}/mutag_{fold}_splits.pkl      (disjoint train/valid/test indices)

Split logic: random disjoint 80% train / 10% valid / 10% test (seed + fold).
Graphs that fail SMILES reconstruction are dropped from the CSV, splits, and
index maps (they are never loaded for vocab mining or training).
GT-ROC at train time uses test mutagens only (see ``mutag_gt_eval_graphs``).

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
    sm = here.parents[1] / 'SharedModules'
    if str(sm) not in sys.path:
        sys.path.insert(0, str(sm))
    from SharedModules.data.dataset_routing import resolve_mutag_roots
    tudataset_root, _ = resolve_mutag_roots(data_root)
    return Mutag(root=tudataset_root)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_root', required=True,
                   help='MUTAG_DATA_ROOT: parent of mutag/ or the mutag/ folder itself')
    p.add_argument('--out_dir', required=True,
                   help='Directory for mutag_{fold}.* artifacts (use MUTAG_DATA_ROOT)')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--seed', type=int, default=42,
                   help="RNG seed for the shuffle (use seed+fold for multi-fold)")
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
        exclude_graph_ids_from_splits, mutag_gt_eval_graphs,
    )

    dataset = _load_mutag(args.data_root)
    split_seed = args.seed + args.fold
    split_idx = get_mutag_split_idx(dataset, seed=split_seed)
    groups = [
        group_for_graph(i, split_idx)
        for i in range(len(dataset))
    ]

    print(f"Loaded mutag: {len(dataset)} graphs from {args.data_root}/mutag")
    print(f"  split: 80/10/10 disjoint  seed={split_seed}")
    print(f"  train={len(split_idx['train'])}  valid={len(split_idx['valid'])}  "
          f"test={len(split_idx['test'])}")
    _test_graphs = [dataset[i] for i in split_idx['test']]
    print(f"  test mutagens w/ source GT (GT-ROC eval): "
          f"{len(mutag_gt_eval_graphs(_test_graphs))}")

    df, index_maps = build_mutag_smiles_df(
        dataset, groups=groups, verify=not args.no_verify)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"mutag_{args.fold}.csv"
    pkl_path = out_dir / f"mutag_{args.fold}_index_maps.pkl"
    splits_path = out_dir / f"mutag_{args.fold}_splits.pkl"

    n_before = len(df)
    df_ok = df[df['conversion_ok']].copy()
    n_drop = n_before - len(df_ok)
    if n_drop:
        failed_ids = df.loc[~df['conversion_ok'], 'graph_id'].astype(int).tolist()
        print(f"  {n_drop} graphs failed SMILES conversion — excluding from dataset:")
        print(f"    graph_ids={failed_ids}")
        split_before = {k: len(v) for k, v in split_idx.items()}
        split_idx = exclude_graph_ids_from_splits(split_idx, failed_ids)
        for k in ('train', 'valid', 'test'):
            dropped = split_before[k] - len(split_idx[k])
            if dropped:
                print(f"    removed from {k}: {dropped}")
        _test_graphs = [dataset[i] for i in split_idx['test']]
        print(f"  after exclusion: train={len(split_idx['train'])}  "
              f"valid={len(split_idx['valid'])}  test={len(split_idx['test'])}")
        print(f"  test mutagens w/ source GT: "
              f"{len(mutag_gt_eval_graphs(_test_graphs))}")

    df_ok.to_csv(csv_path, index=False)
    with open(pkl_path, 'wb') as f:
        pickle.dump(index_maps, f)
    save_mutag_splits(splits_path, split_idx, seed=split_seed)

    from SharedModules.data.mutag_artifacts import validate_mutag_artifacts
    info = validate_mutag_artifacts(csv_path, splits_path, pkl_path,
                                    dataset_size=len(dataset))
    print(f"  artifact check OK: {info['n_graphs']} graphs "
          f"(train={info['n_train']} valid={info['n_valid']} test={info['n_test']})")

    n_verify_fail = int((~df_ok['verify_ok']).sum()) if 'verify_ok' in df_ok else 0
    print(f"  wrote {csv_path}  ({len(df_ok)} molecules)")
    print(f"  wrote {pkl_path}  ({len(index_maps)} index maps)")
    print(f"  wrote {splits_path}")
    if n_verify_fail:
        print(f"  [warn] {n_verify_fail} graphs failed heavy-atom alignment verification "
              f"(explicit-H graph nodes are expected to be absent from RDKit SMILES).")


if __name__ == '__main__':
    main()
