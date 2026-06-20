# Dataset raw files

## mutag (Mutagenicity with source ground truth)

Vendored loader: `datasets/mutag.py` (from [Graph-COM/GSAT](https://github.com/Graph-COM/GSAT)).

### Raw files (bundled in-repo)

Text files live under `data/mutag/raw/` (see `data/mutag/README.md`):

```
Mutagenicity_A.txt
Mutagenicity_edge_gt.txt          ← source edge_label ground truth
Mutagenicity_edge_labels.txt
Mutagenicity_graph_indicator.txt
Mutagenicity_graph_labels.txt
Mutagenicity_label_readme.txt     ← y=0 mutagen, y=1 nonmutagen
Mutagenicity_node_labels.txt
```

Optional: `Mutagenicity.pkl` (~6 GB, PGExplainer pre-baked features). Without it,
the loader builds 14-dim one-hot features from node labels.

Set `data_root` to the repo `data/` directory so the loader finds `data/mutag/`.

### Pipeline artifacts

Before training with motifs, export fold CSV + splits:

```bash
python MotifBreakdown/export_mutag_dataset_to_csv.py \
    --data_root /path/to/data --out_dir /path/to/FOLDS --fold 0
```

Produces:

| File | Purpose |
|------|---------|
| `mutag_{fold}.csv` | Mapped SMILES + labels + group for vocab generation |
| `mutag_{fold}_index_maps.pkl` | Graph node → SMILES atom index (motif lookup) |
| `mutag_{fold}_splits.pkl` | GSAT-style train/valid/test indices |

Default split (`mutag_x=True`, GSAT): 80% train / 20% valid; test = all **mutagen** graphs (`y==0`) with annotated NO2/NH2 motif edges.

### Label encoding

From `Mutagenicity_label_readme.txt` (TUDataset / PGExplainer bundle):

| `y` | Class |
|-----|-------|
| 0 | mutagen |
| 1 | nonmutagen |

`Mutagenicity_edge_gt.txt`: `1` = edge inside NO2/NH2 motif, `0` = outside.

### Ground truth

Each processed `Data` carries **source** GT (do **not** use `--use_gt`):

- `edge_label` [E] — motif edges (NO2/NH2), kept only when `y==0` (mutagen)
- `node_label` [N] — nodes on those motif edges when `y==0`; zeroed when `y==1`

## OGB (ogbg-molhiv, ogbg-molbace, …)

Auto-downloaded on first load via `ogb`:

```python
from ogb.graphproppred import PygGraphPropPredDataset
PygGraphPropPredDataset(root='/path/to/ogb', name='ogbg-molhiv')
```

Export to fold CSV for vocab generation:

```bash
python MotifBreakdown/export_ogb_to_csv.py \
    --dataset ogbg-molhiv --ogb_root /path/to/ogb --out_dir /path/to/FOLDS
```

OGB has **no** source explanation GT; use `--use_gt` + Phase 4 synthetic relabelling when needed.

SMILES come from `{ogb_root}/{name}/mapping/mol.csv.gz` and match graph node order directly.
