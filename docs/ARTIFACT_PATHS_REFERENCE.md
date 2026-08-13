# Artifact Paths Reference (ChemIntuit / Explainable Molecular GNN)

Where every artifact lives on the HPC, for future agents and sessions. HPC base
(symlinked, both paths are the same tree):

```
/nfs/stak/users/kokatea/hpc-share/ChemIntuit/Claude+Cursor
/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor      # same via symlink
```

Local repo: `/Users/apurvakokate/ExplanableMolecularGNN/ExplanableMoleculecularGNNs`
(commit locally, user pushes; HPC pulls — NEVER scp/rsync to HPC).

---

## Environments (source before any HPC run)
| env script | roots it sets | campaign |
|---|---|---|
| `parallel/final_ertlmdl_env.sh` | `OUT_ROOT=final_ertlmdl`, `VOCAB_ROOT=vocab_ertlmdl`, `PROCESSED_ROOT=processed_ertlmdl`, `MOSE_BASE=0`, `RULE_ENGINE=none`, `REUSE_VANILLA_FROM=final_v2` | conservative-Ertl-ring + ring_mdl (FCOL) |
| `experiment_config.sh` | `DATA_ROOT`, `NODE_ENCODER`, `MUTAG_DATA_ROOT`, `OGB_DATA_ROOT`, backbones, epochs | base config sourced by the above |

Conda env: `l2xgnn` (torch 2.2, rdkit 2023.09; `torch_scatter` present on HPC).

## Data
- Standard folds: `DATA_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/`
  (all datasets except `mutag`/OGB; each has `{dataset}_{fold}.csv`, folds 0–4).
- `mutag` (NOT `Mutagenicity`): `MUTAG_DATA_ROOT=.../Claude+Cursor/data`
- OGB: `.../Claude+Cursor/data/ogb`

## Fragmentation vocabularies  `VOCAB_ROOT/{dataset}/{variant}/`
- ERTL campaign root: `vocab_ertlmdl/` — variants `conservative_ertl_ring_mdl`,
  `conservative_ertl_ring_bpe`, **`ring_mdl`** (rings-only + MDL, no FG heads).
- rBRICS / older: `vocab_final_v2/{dataset}/rbrics/`, `vocab_v2/`, `vocab_final_v1/`.
- Key files per variant dir: `rules.json`, `matrix_columns.csv` (motif_id→smarts+class
  counts), `motif_vocabulary.csv`, `{ds}_{variant}_graph_lookup.pickle` (atom→motif),
  `{ds}_{variant}_lookup_all.pickle` (ALL molecules — this is why vocab is fold-0-shared),
  `{ds}_{variant}_graph_motifidx.pickle` (graph→motif-id set), `smiles_labels.csv`,
  `stats_motifs.csv` (per-motif freq + `above_threshold`).
- **Vocab is fold-0-mined and fold-shared** (no fold in the path). The `_filter`
  variant is NOT a separate dir — it is the base vocab with a per-fold support
  threshold applied at train time (`SharedModules/data/threshold_config.py`).

## Processed graphs  `PROCESSED_ROOT/{variant}/`
- `processed_ertlmdl/{variant}/` — created lazily by the first train/eval run that
  needs it (per fold split, motif assignments are vocab-fixed).

## Vanilla (base GNN) checkpoints  `{OUT_ROOT}/vanilla/{ds}/fold{F}/{variant}/bb-{BB}_enc-onehot_norm-none_real/best_model.pt`
- **Vocab-INDEPENDENT** (atom features only) → the same checkpoint is symlinked across
  variants (`parallel/seed_vanilla_from_final_v2.sh`, seeded from `final_v2`).
- ERTL campaign: `final_ertlmdl/vanilla/...`. Used directly by post-hoc explainers.

## MoSE (ante-hoc) results  `{OUT_ROOT}/mose/{variant}/{ds}/fold{F}/{BB}_..._unk-{mode}_..._{tier}_.../`
- ERTL: `final_ertlmdl/mose/conservative_ertl_ring_mdl_filter/...`, `.../ring_mdl_filter/...`
- **rBRICS (comparison) — CORRECTED root (use this):**
  `antehoc_recompute_v1/mose/rbrics_filter/{ds}/fold{F}/{BB}_..._unk-{mode}_real_.../`
  (Aug-8 re-eval, *after* the Aug 7–8 eval-path fix; 8 datasets, both `unk-fixed` and
  `unk-learnable_shared` present; planted tiers under
  `antehoc_recompute_v1/mose/rbrics_filter_relabelled_dnf_k{2,3}_r{1,2}/`; `ertl_first_filter`
  also recomputed here). Schema = **`summary_splits.json`** (method→`{train,valid,test}`) +
  derived CSVs: pooled Pearson in **`mose_grouped_corr_pooled_alltest.csv`** (`pearson_u_exclunk`
  = the Tab-3.4 per-motif metric), per-instance in `mose_instance_corr_{split}.csv`,
  impact/importance in `mose_{impact,importance}_{split}.csv`. rBRICS **GT-ROC** →
  `gtroc_filtered_exclunk_v1` (authoritative, drop-UNK), *not* a summary field.
  - ⚠️ **STALE — do NOT use:** `final_v2/mose/rbrics_filter/…/summary.json` (July-29, pre-fix,
    flat `pearson_motif_all` field). Superseded by the recompute root above.
- Per-run files (ERTL campaign, old-schema `summary.json`): `summary.json`, `score_vs_impact.csv`
  (per-motif score+impact), `correlation.csv`, `gt_roc.csv`, `motif_impact.csv`, `best_model.pt`,
  `epoch_scalars.csv`/`epoch_motifs.csv` (per-epoch; stdout is buffered — progress is here).
- unk modes: `unk-fixed` (value 0.5) | `unk-learnable_shared`.

## Post-hoc explainers (MAGE=`mage_v2`, motif-occlusion, GNNExplainer, PGExplainer)
- SCAFFOLD (vanilla weights loaded under an eval vocab, `run_vanilla --epochs 0`):
  `{OUT_ROOT}/baselines/{ds}/fold{F}/{variant}/bb-{BB}_.../` → `summary.json`, `summary_splits.json`.
- SCORED (importances + GT-ROC + Pearson):
  - ERTL: `final_ertlmdl/posthoc_ertl/{ds}/fold{F}/{variant}/bb-{BB}_.../`
  - rBRICS-era: `posthoc_v1/baselines/{ds}/fold{F}/{variant}/bb-{BB}_.../`
- Per-run files: `{method}_grouped_corr_pooled_alltest.csv` (Pearson — `pearson_u_exclunk`
  matches MoSE `pearson_motif`), `{method}_importance_{split}.csv`, `{method}_impact_{split}.csv`,
  `summary_splits.json` (holds `{method}/{split}/gt_roc_node_auc_mean`).
- Use `mage_v2` NOT `mage_official` (the latter is the wrong/deprecated version).

## Hyperparameter tuning sweep (GAT, ERTL, all folds)
- `fragmentation_tuning_experiment/runs/conservative_ertl_ring_mdl_filter/{ds}/fold*/GAT_*/summary.json`
- Analysis is generated on-demand by `fragmentation_tuning_experiment/collect.py`
  (fold-averaged, mean±std, auc-guarded) — **not saved to a file** unless you redirect it.
- Scripts: `fragmentation_tuning_experiment/{grid.tsv,tune_mose.sbatch,collect.py,README.md}`.

## Vocab-health (LOCAL ONLY, not on HPC)
- `analysis/vocab_health.py`, `analysis/vocab_health_{BBBP,Mutagenicity,hERG,Benzene}.csv`,
  `reports/vocab_health_summary.png` (7 ertl configs; `ring_mdl` not yet added).

## Key scripts
| script | role |
|---|---|
| `MotifBreakdown/generate_vocab_rules.py` | vocab generation (`--method {conservative_ertl_ring_mdl, ring_mdl, rbrics, ...}`) |
| `MotifBreakdown/mdl_linker.py` | rings + FG heads + MDL linker (`head_source {ertl,rbrics,none}`) |
| `MotifBreakdown/ring_mdl.py` | rings-only + MDL (no FG heads); functions replicated from mdl_linker |
| `MOSE-GNN/run.py` | MoSE trainer; `MOSE-GNN/reg_config.py` = base (ent_reg,size_reg) per (backbone,dataset); PNA reuses GIN |
| `SharedModules/baselines/run_vanilla.py` | vanilla train + post-hoc scaffold (`--epochs 0 --load_weights_from`) |
| `analysis/eval_driver_posthoc.py` | scores post-hoc explainers (`--methods motif_occlusion mage_v2 --out_root ... --dest_root ...`) |
| `run_experiments.sh` | phase driver (`phase5_mose`, `phase5_baselines`); `VOCAB_FOCUS` token → variant |
| `parallel/run_cell.sh` | one (ds,variant,bb,fold,tier,method) cell — variant is the BASE (e.g. `ring_mdl`); `_filter` is derived |

## Conventions / gotchas
- Launch a cell with the **base** variant (`ring_mdl`), not `ring_mdl_filter` — phase5_mose
  derives the filtered variant via `_vocab_focus_filtered_for`.
- "base params" = `reg_config.py` defaults (NOT the tuning-sweep values).
- `run_cell.sh` "COMPLETED" ≠ success — check the log for `resolved to no variants` /
  `unknown VOCAB_FOCUS token` (a silent no-op if a variant isn't registered).
- Bad GPU node: exclude `cn-gpu3` (persistent CUDA-init failure → CPU fallback).
- QOS `normal` GrpCPU cap is shared across users; add `-p ...,preempt` (PreemptMode=REQUEUE)
  for capacity. `sbatch --wrap` is disabled site-wide; pipe the script to `sbatch` via stdin.
- Metrics scope: explainer metrics use the `_all` (all-splits) fields
  (`pearson_motif_all`, `gt_roc_node_auc_mean_all`); predictive `auc`/`rmse_orig` = test split.
