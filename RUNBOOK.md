# ChemIntuit Experiment Runbook

Complete guide for running the full pipeline from raw FOLDS to trained models.

---

## Repository layout

```
project/
├── experiment_config.sh          ← edit once, source before any run
├── run_experiments.sh            ← phase dispatcher
├── MotifBreakdown/
│   ├── generate_vocab_rules.py   ← fragmentation + vocab (CLI)
│   ├── molfragbpe5.py            ← fragmentation engine + BPE
│   ├── motif_label_pipeline.py   ← phase 3 synthetic rule generation
│   ├── analyse_phase3.py         ← full analysis script → JSON for widget
│   ├── coverage_vs_threshold.py  ← phase 2 threshold sweep
│   └── test_pipeline.py          ← 133 unit tests
├── SharedModules/
│   ├── data/                     ← dataset, vocab, loader
│   ├── baselines/
│   │   ├── vanilla_gnn.py
│   │   ├── run_vanilla.py        ← vanilla training + GNNExplainer + PGExplainer
│   │   └── mage.py
│   ├── evaluation/               ← metrics, motif_eval, embedding_viz
│   └── models/                   ← GIN/GCN/GAT/SAGE conv layers + gnn_base
├── MOSE-GNN/
│   └── run.py                    ← MOSE training (CLI)
└── MotifSAT/
    └── run.py                    ← MotifSAT training (CLI)
```

---

## One-time setup

```bash
# 1. Edit paths and datasets
nano experiment_config.sh
# Set: PROJECT, DATA_ROOT, VOCAB_ROOT, OUT_ROOT, DATASETS, FOLDS

# 2. Source it (do this at the start of every session)
source experiment_config.sh

# 3. Check tests pass on your environment
cd $PROJECT/MotifBreakdown && python3 test_pipeline.py -v
cd $PROJECT/SharedModules && python3 tests/test_shared_modules.py
cd $PROJECT/SharedModules && python3 tests/test_graph_to_smiles.py
cd $PROJECT/MOSE-GNN     && python3 tests/test_mose_gnn.py
cd $PROJECT/MotifSAT     && python3 tests/test_motifsat.py
```

### Fragmentation variant names used throughout

| Name | Method | Fallback | BPE | Notes |
|---|---|---|---|---|
| `rbrics_old` | `rbrics_old` | no | no | CreateMotifVocab plot path (BreakrBRICSBonds + ToSmiles) |
| `rbrics` | `rbrics` | no | no | rBRICS+reBRICS, no fallback, no BPE |
| `all_fallback_bpe` | `all` | yes | yes | Full cascade with fallback and BPE |

### Weights & Biases logging (optional, OFF by default)

W&B is **disabled by default** — `WANDB_FLAGS` is empty, so no run is created
and nothing is logged. To enable it for the phase-5 training runs, export
`WANDB_FLAGS` before running any phase:

```bash
export WANDB_FLAGS="--use_wandb --wandb_project ChemIntuit --wandb_entity your_team"
```

`WANDB_FLAGS` is appended to all phase-5 runners (vanilla, mose, gsat,
motifsat, baselines). **When W&B is enabled it now defaults to OFFLINE mode**
(logs to `./wandb/`, no network) so a blocked HPC compute node can never crash
training with a BrokenPipe error. Sync afterwards from a node with internet:
```bash
wandb sync --sync-all
```
To stream live instead (only on a node with outbound internet):
```bash
export WANDB_MODE=online
```
Leave `WANDB_FLAGS` unset to skip W&B entirely.

---

## Step 1 — Fragmentation

Produces vocabulary files for all three variants. No threshold applied yet.

```bash
# All three variants in one go
for ds in $DATASETS; do

  # 1a. rbrics_old  (CreateMotifVocab plot replication)
  python3 $PROJECT/MotifBreakdown/generate_vocab_rules.py \
    --datasets $ds --data_root $DATA_ROOT \
    --out_dir $VOCAB_ROOT \
    --method rbrics_old --variant rbrics_old --fold 0

  # 1b. rbrics  (rBRICS + reBRICS, clean chemistry)
  python3 $PROJECT/MotifBreakdown/generate_vocab_rules.py \
    --datasets $ds --data_root $DATA_ROOT \
    --out_dir $VOCAB_ROOT/$ds/rbrics \
    --method rbrics --fold 0

  # 1c. all+fallback+bpe  (maximum coverage)
  python3 $PROJECT/MotifBreakdown/generate_vocab_rules.py \
    --datasets $ds --data_root $DATA_ROOT \
    --out_dir $VOCAB_ROOT/$ds/all_fallback_bpe \
    --method all --fallback --bpe --fold 0

done

# Or equivalently via the runner:
bash run_experiments.sh phase1
```

**Outputs** per dataset/variant: `vocab.pkl`, `matrix.pkl`, `matrix_columns.csv`,
`meta.json`, `rules.json`, `vocab_meta.json`, `bpe_*.json`

---

## Step 2 — Coverage vs threshold (phase 2)

Sweeps support thresholds from 0% to 30% and plots vocabulary size vs node
coverage. Run this before deciding on a threshold.

```bash
for ds in $DATASETS; do
  for variant in rbrics all_fallback_bpe; do
    python3 $PROJECT/MotifBreakdown/coverage_vs_threshold.py \
      --dataset $ds \
      --vocab_root $VOCAB_ROOT \
      --variant $variant \
      --out_dir $OUT_ROOT/coverage_plots
  done
done

# Or via runner:
bash run_experiments.sh phase2
```

**Review the plots** in `$OUT_ROOT/coverage_plots/`.

### Threshold selection

The threshold is the minimum motif support % for inclusion in the final
vocabulary used for training. Look for the **elbow point** on each curve
where vocabulary size drops steeply but node coverage is still above ~80%.

Thresholds are **not** a single global value. They are set **per fragmentation
variant × per dataset** in the `CHOSEN_THRESHOLD` dict inside
`MotifBreakdown/generate_vocab_rules.py`. After reviewing the coverage plots,
edit that dict — there is no `THRESHOLD` environment variable.

```python
# MotifBreakdown/generate_vocab_rules.py
CHOSEN_THRESHOLD = {
    'all_fallback_bpe_filter': {
        'Mutagenicity': 0.002,   # vocab 212→108, coverage 87.6%→81.8%
        'Benzene':      0.006,
        'BBBP':         0.006,
        # ... one entry per dataset
    },
    'rbrics_filter':     { ... },
    'rbrics_old_filter': { ... },
}
```

phase3 reads these automatically (`--apply_threshold` with no value). If a
`variant × dataset` entry is missing, `generate_vocab_rules.py` raises a clear
error telling you which key to add.

---

## Step 3 — Thresholded vocabularies (phase 3)

Regenerates vocabularies with the per-variant × per-dataset thresholds from
`CHOSEN_THRESHOLD` applied. This is the vocabulary used for model training.

```bash
# No THRESHOLD env var. Edit CHOSEN_THRESHOLD in generate_vocab_rules.py first,
# then just run phase3 — it passes --apply_threshold and the dict is consulted.
bash run_experiments.sh phase3
```

To regenerate a single dataset/variant by hand, you may optionally override the
dict value with `--threshold_pct` (a manual one-off; the runner never uses it):

```bash
python3 $PROJECT/MotifBreakdown/generate_vocab_rules.py \
  --datasets Mutagenicity --data_root $DATA_ROOT \
  --out_dir $VOCAB_ROOT \
  --method all --fallback --bpe \
  --apply_threshold --threshold_pct 0.002 --fold 0   # override; usually omit
```


---

### Rule ranking (how rule_index is ordered)

`rules.json` is sorted so `rule_index=0` is the "best" rule. The sort key is
controlled by `RULE_RANK` (default `balanced`):

- **`balanced`** (default) — `score = balance × separation × (1 − spurious)`:
  - `balance = 1 − |coverage% − 50| / 50` — rewards a ~50/50 synthetic split.
  - `separation` — structural SNR (label-free): how structurally distinct the
    matched vs non-matched molecules are.
  - `spurious` — mean pairwise Jaccard among the rule's motifs **plus** a
    subsuming-family penalty, so redundant/co-occurring and general-variant
    motifs are penalised.
- **`pct1`** — legacy: sort by positive coverage only (ignores balance/spurious).
  This is what produced the imbalanced 4-clause OR rules.

```bash
export RULE_RANK=balanced   # default; set to pct1 for legacy behaviour
```

> **Variant note.** `rbrics` / `rbrics_old` (no fallback, no BPE) produce larger
> fragments, so rule coverage often only *just* crosses 50% — that's expected
> from the algorithm. `balanced` degrades gracefully here (a 40% rule still
> scores `balance=0.8` and beats a 90% rule at `0.2`). `all_fallback_bpe` reaches
> much higher coverage, so the balance/spurious penalties change the ranking
> substantially versus `pct1` — inspect both if comparing variants.

### 4a. Inspect available rules (manual verification)

Every rule carries its score components, written to `rules_summary.csv` next to
`rules.json`. Review coverage and balance before committing to an index:

```bash
cd $VOCAB_ROOT/<dataset>/<variant>
python3 -c "import pandas as pd; \
df=pd.read_csv('rules_summary.csv'); \
print(df[['rank','score','balance','separation','spurious','cover_pct','pct1','n_clauses','rule_str']].head(15).to_string(index=False))"
```

`cover_pct` (= `pct1`) is the fraction of ALL molecules the rule labels as 1 —
read it directly to see how balanced the synthetic task will be (~50% is ideal).

### 4b. Pick / override the rule index

`rule_index=0` is the top-ranked rule under `RULE_RANK`. To use a different one,
just set `RULE_INDEX` to its `rank` from the table above:

```bash
export RULE_INDEX=3          # any rank from rules_summary.csv
bash run_experiments.sh phase4
```

To find the most balanced rule regardless of the full score:

```bash
python3 -c "import pandas as pd; \
df=pd.read_csv('rules_summary.csv'); df['imbal']=(df.cover_pct-50).abs(); \
print(df.sort_values('imbal')[['rank','cover_pct','balance','spurious','rule_str']].head(10).to_string(index=False))"
```

### 4c. Apply synthetic labels

```bash
export RULE_INDEX=0          # index into rules.json (default = balance-aware best)
bash run_experiments.sh phase4
```

The phase4 step writes relabelled graph objects to `$OUT_ROOT/gt_cache/`. The
selected rule (with its score components) is also saved to `selected_rule.json`
in the same directory for provenance.


---

## Step 5 — Vanilla GNN

Trains a plain GNN with no motif injection. Used as the backbone for
post-hoc explainers (GNNExplainer, PGExplainer, MAGE).

```bash
for ds in $DATASETS; do
  for fold in $FOLDS; do
    python3 $PROJECT/SharedModules/baselines/run_vanilla.py \
      --dataset $ds --fold $fold \
      --backbone $BACKBONE --node_encoder $NODE_ENCODER \
      --epochs $EPOCHS \
      --data_root $DATA_ROOT \
      --vocab_root $VOCAB_ROOT \
      --vocab_variant rbrics \
      --out_dir $OUT_ROOT/vanilla
  done
done

# Or via runner:
bash run_experiments.sh phase5_vanilla
```

---

## Step 6 — Baseline explainers (post-hoc on vanilla)

GNNExplainer, PGExplainer, and MAGE run on the trained vanilla model weights.
Motif-level evaluation is done using each vocabulary variant.

```bash
for ds in $DATASETS; do
  for fold in $FOLDS; do
    python3 $PROJECT/SharedModules/baselines/run_vanilla.py \
      --dataset $ds --fold $fold \
      --backbone $BACKBONE --node_encoder $NODE_ENCODER \
      --epochs 0 \                        # load weights, skip training
      --data_root $DATA_ROOT \
      --vocab_root $VOCAB_ROOT \
      --vocab_variant rbrics_filter \
      --out_dir $OUT_ROOT/baselines
  done
done

# Or via runner:
bash run_experiments.sh phase5_baselines
```

The runner applies all four vocab variants automatically for cross-evaluation.

---

## Step 7 — MOSE-GNN

Trains MOSE-GNN with motif feature injection (w_feat) and readout injection
(w_readout) across all three thresholded vocabulary variants.

```bash
for ds in $DATASETS; do
  for fold in $FOLDS; do
    for variant in rbrics_filter all_fallback_bpe_filter all_fallback_bpe; do
      python3 $PROJECT/MOSE-GNN/run.py \
        --dataset $ds --fold $fold \
        --backbone $BACKBONE --node_encoder $NODE_ENCODER \
        --w_feat --w_readout \
        --epochs $EPOCHS \
        --data_root $DATA_ROOT \
        --vocab_root $VOCAB_ROOT \
        --vocab_variant $variant \
        --out_dir $OUT_ROOT/mose/$variant
    done
  done
done

# Or via runner:
bash run_experiments.sh phase5_mose
```

### MOSE configuration matrix

| Variant | w_feat | w_readout | Notes |
|---|---|---|---|
| rbrics_filter | ✓ | ✓ | rBRICS, thresholded vocab |
| all_fallback_bpe_filter | ✓ | ✓ | Full cascade, thresholded |
| all_fallback_bpe | ✓ | ✓ | Full cascade, no threshold |
| all_fallback_bpe + gt | ✓ | ✓ | + synthetic relabelling |

---

## Step 8 — MotifSAT

MotifSAT with readout-level motif aggregation, no information bottleneck
(learn_edge_att=False required for valid motif score aggregation).

> **Message injection (`--w_message`).** `--w_message` is now a normal opt-in
> flag (it previously defaulted to `True` and could not be turned off). The
> `phase5_gsat` and `phase5_motifsat` runners pass it automatically, controlled
> by the `MOTIFSAT_W_MESSAGE` environment variable (default `1` = on, which
> reproduces the historical behaviour). Set `export MOTIFSAT_W_MESSAGE=0`
> before running either phase to train without message injection. When invoking
> `MotifSAT/run.py` by hand, add `--w_message` explicitly if you want it.

> **Attention scores at eval.** At evaluation the model's returned `node_att`
> (and `edge_att`) is a hard 0/1 mask — the GSAT prediction gate. Ranking-based
> metrics (GT ROC, per-motif attention aggregation) instead use the continuous
> `node_att_soft` / `edge_att_soft` probabilities exposed in the model's `aux`
> dict, so ROC-AUC can rank atoms rather than tie on 0/1. This is handled
> automatically inside `compute_gt_roc` and `_aggregate_att_to_motif`; no flag
> needed.

```bash
for ds in $DATASETS; do
  for fold in $FOLDS; do

    # Base GSAT (no motif method — comparison point)
    python3 $PROJECT/MotifSAT/run.py \
      --dataset $ds --fold $fold \
      --backbone $BACKBONE --node_encoder $NODE_ENCODER \
      --motif_method none \
      --noise none --info_loss_level none --info_loss_coef 0.0 \
      --w_message \
      --epochs $EPOCHS \
      --data_root $DATA_ROOT --vocab_root $VOCAB_ROOT \
      --vocab_variant rbrics \
      --out_dir $OUT_ROOT/base_gsat

    # MotifSAT readout — rbrics
    python3 $PROJECT/MotifSAT/run.py \
      --dataset $ds --fold $fold \
      --backbone $BACKBONE --node_encoder $NODE_ENCODER \
      --motif_method readout \
      --noise none --info_loss_level none --info_loss_coef 0.0 \
      --w_feat --w_readout --w_message \
      --epochs $EPOCHS \
      --data_root $DATA_ROOT --vocab_root $VOCAB_ROOT \
      --vocab_variant rbrics \
      --out_dir $OUT_ROOT/motifsat/rbrics

    # MotifSAT readout — all+fallback+bpe
    python3 $PROJECT/MotifSAT/run.py \
      --dataset $ds --fold $fold \
      --backbone $BACKBONE --node_encoder $NODE_ENCODER \
      --motif_method readout \
      --noise none --info_loss_level none --info_loss_coef 0.0 \
      --w_feat --w_readout --w_message \
      --epochs $EPOCHS \
      --data_root $DATA_ROOT --vocab_root $VOCAB_ROOT \
      --vocab_variant all_fallback_bpe \
      --out_dir $OUT_ROOT/motifsat/all_fallback_bpe

  done
done

# Or via runner (passes --w_message by default; set MOTIFSAT_W_MESSAGE=0 to disable):
bash run_experiments.sh phase5_gsat
bash run_experiments.sh phase5_motifsat
```

---

## Collecting results

```bash
bash run_experiments.sh collect
# Prints a table and writes $OUT_ROOT/all_results.csv
```

---

## Full sequential run (non-interactive)

For batch HPC submission without manual threshold/rule review steps:

```bash
source experiment_config.sh

# Thresholds: edit CHOSEN_THRESHOLD (per variant × dataset) in
#   MotifBreakdown/generate_vocab_rules.py — there is no THRESHOLD env var.
export RULE_INDEX=0

# Optional knobs:
#   MOTIFSAT_W_MESSAGE=0   train MotifSAT/GSAT without message injection
#                          (default 1 = on, reproduces historical behaviour)
#   MOTIFSAT_VERIFY_FIXES=1 emit one-time "[FIX#N active]" confirmation logs
export MOTIFSAT_W_MESSAGE=1

bash run_experiments.sh phase1
bash run_experiments.sh phase3   # reads CHOSEN_THRESHOLD dict
bash run_experiments.sh phase4   # uses $RULE_INDEX
bash run_experiments.sh phase5_vanilla
bash run_experiments.sh phase5_mose
bash run_experiments.sh phase5_gsat
bash run_experiments.sh phase5_motifsat
bash run_experiments.sh phase5_baselines
bash run_experiments.sh collect
```

For interactive runs, insert `bash run_experiments.sh phase2` between
phase1 and phase3 and review the coverage plots before editing the
`CHOSEN_THRESHOLD` dict in `generate_vocab_rules.py`.

---

## Vocabulary variant summary

After all phases are complete you will have trained models on six configurations:

| Config | Frag | Threshold | Synthetic GT | Purpose |
|---|---|---|---|---|
| `rbrics` | rBRICS | no | no | Baseline vocab |
| `rbrics_filter` | rBRICS | yes | no | Cleaner training vocab |
| `all_fallback_bpe` | all+fb+bpe | no | no | Full motif coverage |
| `all_fallback_bpe_filter` | all+fb+bpe | yes | no | Filtered full vocab |
| `all_fallback_bpe` + gt | all+fb+bpe | no | yes | Synthetic relabelling |
| `rbrics_old` | CreateMotifVocab plot | no | no | Plot replication baseline |

---

## Analysis & explainability diagnostics

The eval pipeline now emits, per run (alongside `summary.json`):
- `motif_impact.csv` — per-motif mask-removal impact.
- `discriminativeness.csv` — per-motif **class-discriminativeness** (model-free):
  `presence_auc`, `delta_p1`, `abs_disc`. Answers "is this motif actually
  predictive of the label?" — high learned score + low `abs_disc` = a
  non-discriminative motif the explainer latched onto.
- `score_vs_impact.csv` — joined `motif_id, score, impact, abs_disc` for plots.
- `correlation.csv` / `top_disc_check.csv` — score↔impact correlation and
  whether the top-scored motifs are the discriminative ones.

New `summary.json` fields (now carried into `all_results.csv` by `collect`):
- `pearson`, `spearman` — score-vs-impact correlation (MOSE, MotifSAT, and per
  explainer for baselines: `gnnexplainer_pearson`, `pgexplainer_pearson`, ...).
- `top_k_abs_disc`, `mean_abs_disc`, `score_disc_spearman` — top-scored-motif
  discriminativeness check.
- `score_min/max/mean/std/median/mode/count` — learned motif-score distribution.

### Reformatted results table (dataset×run rows, backbone cols)
```bash
python analysis/make_results_table.py $OUT_ROOT/all_results.csv --metric auc
python analysis/make_results_table.py $OUT_ROOT/all_results.csv --metric val_auc --md val.md
```

### Motif score-vs-impact plots (binned box-plot grid + count histogram)
Box plots of motif IMPACT binned by learned SCORE, one coloured box-series per
group (default model family), with an orange motif-COUNT histogram per bin.
The per-bin counts are also written as a table (CSV + markdown).
```bash
python analysis/plot_score_vs_impact.py --out_root $OUT_ROOT \
    --save_dir $OUT_ROOT/plots \
    --counts_table $OUT_ROOT/score_impact_counts.csv \
    --group family --facet vocab_variant --nbins 6
```

### Baseline (post-hoc) motif scores: mean AND max node aggregation
Each explainer (GNNExplainer / PGExplainer / MAGE) is node-level; we aggregate
to motif level by both mean and max over the motif's atoms. The summary / table
report both, e.g. `gnnexplainer_mean_pearson`, `gnnexplainer_max_pearson`,
`pgexplainer_mean_score_disc_spearman`, etc.

### Degenerate-explanation reference
The discriminativeness check (`abs_disc`, `presence_auc`) targets the failure
mode in Azzolin et al., "GNN Explanations that do not Explain and How to find
Them" (arXiv:2601.20815): a motif with `presence_auc ~ 0.5` has no
class-discriminative power, so a high learned score on it is a degenerate /
unfaithful explanation. The paper's full EST metric is not implemented here.

### Masked-node feature-recovery probe
Tests whether masked (low-attention) nodes' embeddings still leak their input
features. Import `probe_run` where a trained `model` + `test_list` are in scope:
```python
from analysis.probe_masked_nodes import probe_run
res = probe_run(model, test_list, device)   # gated vs raw, masked vs unmasked
# a positive gated_gap_unmasked_minus_masked >> raw gap => masking hides features
```

### Regenerate explainability metrics post-hoc (no retraining)
The new metrics are computed at EVAL time, so they can be regenerated from saved
`best_model.pt` checkpoints without retraining. MOSE and MotifSAT now accept
`--eval_only --load_weights_from <run_dir>`; vanilla uses `--epochs 0
--load_weights_from <run_dir>`.

Regenerate everything under an output tree in one go:
```bash
python analysis/regenerate_eval.py --out_root $OUT_ROOT \
    --data_root $DATA_ROOT --vocab_root $VOCAB_ROOT \
    --processed_root $PROCESSED_ROOT          # optional
    # add --dry_run first to preview the commands
bash run_experiments.sh collect               # refresh all_results.csv
```
Pair each checkpoint with the vocab it was TRAINED on — if vocabularies were
regenerated after training, the masks differ and impact/discriminativeness would
be computed against a different vocab than the model saw.

---

## Single analysis entry point

`analysis/run_analysis.py` ties every analysis step together. Subcommands:
`regenerate` (eval-only on checkpoints), `collect` (rebuild all_results.csv),
`table` (pivot tables per metric), `plots` (score-vs-impact grid + count table),
and `all` (runs them in order).

```bash
# everything, end to end (regenerate eval -> collect -> tables -> plots)
python analysis/run_analysis.py all \
    --out_root $OUT_ROOT --data_root $DATA_ROOT --vocab_root $VOCAB_ROOT \
    --processed_root $PROCESSED_ROOT

# use existing summaries, skip the eval regeneration step
python analysis/run_analysis.py all --out_root $OUT_ROOT --skip_regenerate

# individual steps
python analysis/run_analysis.py table --out_root $OUT_ROOT --metrics auc gt_roc_auc_mean
python analysis/run_analysis.py plots --out_root $OUT_ROOT --facet backbone --nbins 6
```

Or via the experiment runner (runs `all`):
```bash
bash run_experiments.sh analyze
# pass extra flags through ANALYZE_ARGS, e.g. to skip regeneration:
ANALYZE_ARGS="--skip_regenerate" bash run_experiments.sh analyze
```

The masked-node probe stays a library call (needs a live model in memory):
`from analysis.probe_masked_nodes import probe_run`.

---

## Per (architecture × dataset) regularization (MOSE)

MOSE-GNN resolves `ent_reg` / `size_reg` automatically per (backbone, dataset)
from `MOSE-GNN/reg_config.py`. **PNA reuses GIN's configuration.** Any pair not
in the table falls back to `DEFAULT_REG = (0.01, 0.0)`.

* The `phase5_mose` runner passes neither `--ent_reg` nor `--size_reg`, so the
  table is used automatically — no action needed.
* Pass `--ent_reg <v>` and/or `--size_reg <v>` explicitly to override the table
  for a single run (explicit flags always win; partial overrides are allowed).
* The resolved values are printed at run start (`[reg_config] GIN×BBBP: ...`)
  and recorded in `summary.json` / `all_results.csv` (`ent_reg`, `size_reg`).

To change coefficients, edit the `REG_CONFIG` dict in `MOSE-GNN/reg_config.py`.

---

## MOSE learning rates & depth

* **Two learning rates** (separate Adam param groups): the explainer
  (`motif_params` / `unk_param`) trains at `explainer_lr=0.01`, the GNN backbone
  at `gnn_lr=0.001`. Defaults live in `MOSEConfig`; the active values print at
  run start (`[lr] explainer=... gnn=...`) and are saved to the summary. Pass
  `--explainer_lr` / `--gnn_lr` (or omit to use the defaults). Set neither and
  the single `lr` is used (legacy behaviour).
* **GNN depth per dataset**: BBBP uses 2 layers, all others 3, resolved in
  `MOSE-GNN/reg_config.py` (`NUM_LAYERS_BY_DATASET`). `--num_layers <n>`
  overrides. The resolved value prints (`[reg_config] <ds>: num_layers=...`) and
  is recorded in the summary.

The `phase5_mose` runner passes none of these, so every run picks up the
per-dataset depth and the 0.01/0.001 LR split automatically.

---

## Shared backbone normalization (MOSE + MotifSAT)

Both models share `SharedModules/models/gnn_base.py`. Two knobs control the conv
stack, applied **after each conv, before ReLU**:

* `--conv_normalize {l2,layernorm,none}` (**default `l2`**). `l2` rescales each
  node embedding to unit length (matches the original DomainDrivenGlobalExpl
  reference). This is intentional: it cancels embedding-magnitude differences so
  the soft motif-weight scaling acts on direction rather than norm. `layernorm`
  uses a learned per-layer LayerNorm; `none` disables per-conv normalization.
  (Back-compat: `apply_layer_norm=True` still forces `layernorm`.)
* `--no_gin_inner_bn` disables the BatchNorm inside the GIN MLP. **Default: on**
  — GIN layers are `Linear→ReLU→Linear→ReLU→BatchNorm` (Xu et al. / reference).

Both are recorded in `summary.json` / `all_results.csv` (`conv_normalize`,
`gin_inner_bn`). `phase5_mose` / `phase5_motifsat` pass neither, so the L2 +
GIN-BN defaults apply automatically.

**Note:** these change the architecture, so models trained before this build are
not comparable to L2 runs — retrain (or eval-only won't help, since weights
differ structurally for GIN due to the added BatchNorm).

---

## Plot & table fixes (score-vs-impact, baselines)

* **Count histogram now hangs DOWN from the top** of each panel (inverted twin
  axis), so the orange motif-count bars no longer overlap the boxes near the
  x-axis.
* **Default `--group` is now `backbone`** (was `family`), so the different
  architectures tested appear as distinct coloured boxes per score bin. Use
  `--group family` or `--group variant` for other axes.
* **Family detection is now from `motif_method`** (mose / readout→motifsat /
  none→vanilla), not the `exp_dir` path — some run dirs start with the dataset
  name, which previously collapsed everything into one box / mis-grouped rows.
* **Baselines in the table**: they are the `vanilla` family rows.
  `make_results_table.py` derives family from `motif_method`, and `--metric`
  now accepts any column, including baseline explainer columns
  (`gnnexplainer_mean_pearson`, `pgexplainer_max_spearman`, …) once present.

NOTE: explainability columns (pearson/spearman, discriminativeness, baseline
explainer metrics) are only populated by runs trained with the current build —
older `all_results.csv` files have prediction AUCs only.

---

## Explainability diagnostics now in the sweep

* **Multiple-explanation (H0/H1/H2)** — `MOSE-GNN/run.py` gained `--run_multi_explanation`
  (config field `run_multi_explanation`). `run_experiments.sh` passes it when
  `MOSE_RUN_MULTI_EXPLANATION=1` (default in `experiment_config.sh`), so
  each MOSE run writes `multi_explanation.*` (HH/HL/LH/LL categorisation + ratio_H1/H2).
* **Masked-node probe** — `analysis/probe_masked_nodes.py main()` is now a real CLI: it
  rebuilds the MOSE model from summary.json + best_model.pt (inferring hidden_dim from the
  checkpoint when the summary omits it), builds the test loader, runs the probe, and writes
  `masked_node_probe.csv`. Run post-hoc:

    python3 analysis/probe_masked_nodes.py --out_root $OUT_ROOT \
        --data_root $DATA_ROOT --vocab_root $VOCAB_ROOT

  Interpretation: `gated_gap_unmasked_minus_masked > raw_gap_...` means the attention gate
  removes recoverable input-feature info from masked nodes (masking genuinely hides features).

## score-vs-impact plot: per-group counts

The motif-count bars are now **per group** (per family), colour-matched to each box and
positioned under it — previously a single pooled bar summed every family's motif rows,
double-counting motifs scored by more than one family. The counts table gains a `group` column.
