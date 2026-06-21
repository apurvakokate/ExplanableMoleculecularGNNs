# Dataset raw files

Path conventions match `experiment_config.sh`:

| Variable | Role |
|----------|------|
| `DATA_ROOT` | CSV FOLDS (`{dataset}_{fold}.csv`) for standard benchmarks |
| `MUTAG_DATA_ROOT` | Parent of `mutag/` TUDataset **and** mutag export artifacts |
| `OGB_DATA_ROOT` | OGB PyG cache **and** OGB export CSVs for vocab mining |

Run **phase 0** to create special-dataset CSV bridges:

```bash
source experiment_config.sh
bash run_experiments.sh phase0
```

---

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

Set `MUTAG_DATA_ROOT` to the repo `data/` directory (parent of `mutag/`).

### Pipeline artifacts

Before phase 1 / training with motifs, export fold CSV + splits to **`MUTAG_DATA_ROOT`**:

```bash
python MotifBreakdown/export_mutag_dataset_to_csv.py \
    --data_root "$MUTAG_DATA_ROOT" \
    --out_dir   "$MUTAG_DATA_ROOT" \
    --fold 0 --seed 42
```

Produces under `$MUTAG_DATA_ROOT/`:

| File | Purpose |
|------|---------|
| `mutag_{fold}.csv` | Mapped SMILES + labels + group for vocab generation |
| `mutag_{fold}_index_maps.pkl` | Graph node → SMILES atom index (motif lookup) |
| `mutag_{fold}_splits.pkl` | Disjoint train/valid/test indices (80/10/10) |

Default split: random disjoint **80% train / 10% valid / 10% test**.
GT-ROC uses **test mutagens** (`y==0`) with source motif labels only.

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

---

## OGB (ogbg-molhiv, ogbg-molbace, …)

Auto-downloaded on first load via `ogb`:

```python
from ogb.graphproppred import PygGraphPropPredDataset
PygGraphPropPredDataset(root='/path/to/ogb', name='ogbg-molhiv')
```

Export to fold CSV for vocab generation under **`OGB_DATA_ROOT`**:

```bash
python MotifBreakdown/export_ogb_to_csv.py \
    --dataset ogbg-molhiv \
    --ogb_root "$OGB_DATA_ROOT" \
    --out_dir  "$OGB_DATA_ROOT" \
    --fold 0
```

Phase 1 reads `{dataset}_0.csv` from `OGB_DATA_ROOT` (same root used for PyG training).

OGB has **no** source explanation GT; use `--use_gt` + Phase 4 synthetic relabelling when needed.

SMILES come from `{ogb_root}/ogbg_molhiv/mapping/mol.csv.gz` (underscore cache dir).

---

## Dataset lists in config

`experiment_config.sh` splits datasets for clarity:

- **`DATASETS_CSV`** — FOLDS CSV benchmarks (Mutagenicity, BBBP, …)
- **`DATASETS_SPECIAL`** — mutag / OGB (need phase 0 export)
- **`DATASETS`** — union of both (override entirely if desired)

Synthetic GT (phase 4) applies to **`DATASETS_CSV`** only.
