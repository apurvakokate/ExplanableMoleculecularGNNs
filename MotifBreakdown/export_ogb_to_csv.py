#!/usr/bin/env python3
"""export_ogb_to_csv.py — bridge OGB molecular datasets into the CSV fold format
the vocabulary pipeline (generate_vocab_rules.py) consumes.

The vocab generator is CSV-based: it reads {data_root}/{dataset}_{fold}.csv with
columns (smiles, <label>, group). OGB datasets are not CSV — they are
PygGraphPropPredDataset objects — but OGB ships the original SMILES and labels in
<root>/<name>/mapping/mol.csv.gz, and provides an official train/valid/test
split. This script joins the two into a fold CSV so OGB datasets can be
fragmented and thresholded exactly like the CSV datasets.

Usage:
    python3 export_ogb_to_csv.py --dataset ogbg-molhiv \
        --ogb_root /path/to/ogb_download --out_dir /path/to/FOLDS --fold 0

Output:
    {out_dir}/{dataset}_{fold}.csv   (columns: smiles, label, group)
    group ∈ {training, valid, test} from the official OGB split.

Notes:
- Only single-task datasets (molhiv, molbace, molbbbp, molesol, molfreesolv,
  mollipo) export a single 'label' column. Multi-task OGB sets (moltox21,
  molsider, molclintox) have multiple label columns; this script refuses them
  unless --label_col is given, because the vocab pipeline expects one label.
- The OGB split is deterministic; --fold is recorded in the filename only (OGB
  provides ONE official split, not k folds). Pass --fold 0 to match the default
  the loaders/vocab pipeline use.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def _ensure_ogb_cached(dataset: str, ogb_root: str) -> Path:
    """Return path to mol.csv.gz, downloading the OGB dataset if needed."""
    name_hyphen = dataset.replace('_', '-')
    mol_csv = Path(ogb_root) / name_hyphen / 'mapping' / 'mol.csv.gz'
    if mol_csv.exists():
        return mol_csv
    try:
        from ogb.graphproppred import PygGraphPropPredDataset
    except ImportError:
        raise ImportError("OGB not installed. Run: pip install ogb")
    print(f"  OGB cache missing for {name_hyphen}; downloading to {ogb_root!r} ...")
    PygGraphPropPredDataset(root=ogb_root, name=name_hyphen)
    if not mol_csv.exists():
        raise FileNotFoundError(
            f"OGB download finished but mapping file still missing: {mol_csv}")
    return mol_csv


def export(dataset: str, ogb_root: str, out_dir: str, fold: int,
           label_col: str | None) -> Path:
    name_hyphen = dataset.replace('_', '-')
    mol_csv = _ensure_ogb_cached(dataset, ogb_root)
    df = pd.read_csv(mol_csv)
    if 'smiles' not in df.columns:
        raise KeyError(f"'smiles' column not in {mol_csv}; columns={list(df.columns)}")

    # Resolve the label column.
    non_label = {'smiles', 'mol_id'}
    candidates = [c for c in df.columns if c not in non_label]
    if label_col is not None:
        if label_col not in df.columns:
            raise KeyError(f"--label_col {label_col!r} not in {list(df.columns)}")
        use_col = label_col
    elif len(candidates) == 1:
        use_col = candidates[0]
    else:
        raise ValueError(
            f"{dataset} has multiple label columns {candidates}; this is a "
            f"multi-task dataset. Pass --label_col <name> to pick one (the vocab "
            f"pipeline expects a single label).")

    # Official OGB split → group labels.
    try:
        from ogb.graphproppred import PygGraphPropPredDataset
    except ImportError:
        raise ImportError("OGB not installed. Run: pip install ogb")
    ds = PygGraphPropPredDataset(root=ogb_root, name=name_hyphen)
    split = ds.get_idx_split()
    group = ['training'] * len(df)
    for g, key in (('valid', 'valid'), ('test', 'test')):
        for i in split[key].tolist():
            group[int(i)] = g

    out = pd.DataFrame({
        'smiles': df['smiles'].astype(str),
        'label':  df[use_col],
        'group':  group,
    })
    # Drop rows OGB marks with NaN label (some multi-task rows); single-task is fine.
    n_before = len(out)
    out = out.dropna(subset=['label']).reset_index(drop=True)
    if len(out) < n_before:
        print(f"  dropped {n_before-len(out)} rows with missing label")

    out_path = Path(out_dir) / f"{dataset}_{fold}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path}  ({len(out)} molecules; "
          f"label='{use_col}'; "
          f"train={group.count('training')} valid={group.count('valid')} "
          f"test={group.count('test')})")
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True,
                   help='e.g. ogbg-molhiv | ogbg-molbace')
    p.add_argument('--ogb_root', required=True,
                   help='Root passed to PygGraphPropPredDataset(root=...)')
    p.add_argument('--out_dir', required=True,
                   help='FOLDS dir where {dataset}_{fold}.csv is written')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--label_col', default=None,
                   help='Explicit label column (required for multi-task OGB sets)')
    args = p.parse_args()
    export(args.dataset, args.ogb_root, args.out_dir, args.fold, args.label_col)


if __name__ == '__main__':
    main()
