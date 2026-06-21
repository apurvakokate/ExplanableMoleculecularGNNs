# Mutagenicity (mutag) dataset

Bundled TUDataset raw files for the PGExplainer / GSAT **Mutagenicity** benchmark
(4337 graphs). Used by the training loader as `dataset=mutag` with
`data_root=<repo>/data`.

## Layout

```
data/
  mutag/
    raw/                          ← committed (text files, ~5 MB)
      Mutagenicity_*.txt
      Mutagenicity_label_readme.txt
    processed/data.pt             ← generated on first load (gitignored)
  mutag_0.csv                     ← optional: export for vocab (see below)
  mutag_0_index_maps.pkl
  mutag_0_splits.pkl
```

## Label encoding (`Mutagenicity_label_readme.txt`)

| Value | Meaning |
|-------|---------|
| `y = 0` | mutagen |
| `y = 1` | nonmutagen |

`Mutagenicity_edge_gt.txt`: `1` = edge inside NO₂/NH₂ motif (source explanation GT).

## Optional: `Mutagenicity.pkl` (~6 GB)

PGExplainer also ships a pre-baked feature pickle. It is **not** in git (too large).
Download if you need bit-identical features to the original PGExplainer runs:

```bash
curl -L -o /tmp/Mutagenicity.pkl.zip \
  https://github.com/flyingdoog/PGExplainer/raw/master/dataset/Mutagenicity.pkl.zip
unzip /tmp/Mutagenicity.pkl.zip -d data/mutag/raw/
```

Without the pickle, `datasets.mutag.Mutag` builds equivalent **14-dim atom-type
one-hot** features from `Mutagenicity_node_labels.txt`.

## Usage

```bash
# Default: point data_root at repo data/ directory
export DATA_ROOT=/path/to/ExplanableMoleculecularGNNs/data

# First load processes raw → data/mutag/processed/data.pt
python -c "from datasets.mutag import Mutag; ds=Mutag('data/mutag'); print(len(ds))"

# Before motif training, export fold CSV + splits (once):
python MotifBreakdown/export_mutag_dataset_to_csv.py \
    --data_root "$DATA_ROOT" --out_dir "$DATA_ROOT" --fold 0
```

## Ground-truth evaluation

mutag uses **source** GT (`node_label`, `edge_label`) attached at load time.
Do **not** pass `--use_gt` (that is for synthetic rule GT on CSV datasets).

**Splits:** disjoint random 80% / 10% / 10% (train / valid / test). Re-export
after upgrading from legacy GSAT overlapping splits:

```bash
python MotifBreakdown/export_mutag_dataset_to_csv.py \
    --data_root "$DATA_ROOT" --out_dir "$DATA_ROOT" --fold 0
```

**GT-ROC:** computed on **held-out test mutagens** only (`y=0` with annotated
NO₂/NH₂ motif edges). Property metrics (AUC) use the full test split.
