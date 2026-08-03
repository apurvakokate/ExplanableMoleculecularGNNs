# Ablation Study — `ablated_completely_v1`

Standalone context for the study. This doc states the TASK and WHERE THE DATA LIVES
only. It deliberately contains NO implementation (no run commands, no driver, no
scope/matrix decisions) — those are to be decided and approved separately.

## Task

Retrain and fully evaluate **MoSE** and **MotifSAT** under three one-factor-at-a-time
(OFAT) config changes, across all datasets. Only ONE parameter changes per setup.
Output goes under **`ablated_completely_v1`**.

## The three setups (one knob at a time)

1. **Use L2 norm instead of no norm.** (MoSE + MotifSAT; both keep the default fixed
   unknown.)
2. **Run MoSE with learn-unknown instead of fixed unknown.** (MoSE only — MotifSAT has
   no unknown-motif parameter; keep no-norm.)
3. **Node embedding = linear instead of one-hot.** (MoSE + MotifSAT.)

## Baseline for comparison

**`antehoc_v1`** — the post-fix ante-hoc per-split metrics of the default-config
models. (NOT final_v2's training-time summaries.)

## Metrics

The complete thesis metric set (per-split `summary_splits.json`, same format as
`antehoc_v1`) is produced **END-TO-END within training** via the `--per_split_eval`
flag on `run.py` — no separate eval pass. After training, `run.py` calls the shared
`evaluate_native_all_splits` on the in-memory model (train/valid/test evaluated
separately). The ablation scripts pass `--per_split_eval`.

---

## Where the data lives (VERIFIED on osuhpc, 2026-08-03)

Base: `/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/` (= `$REPO`)

| role | path |
|---|---|
| new output | `$REPO/ablated_completely_v1` |
| baseline models (config source) | `$REPO/final_v2/{mose,motifsat}` |
| baseline metrics (comparison) | `$REPO/antehoc_v1/{mose,motifsat}` |
| vocab_root | `$REPO/vocab_final_v2` |
| processed (per variant) | `$REPO/processed_final_v2/<variant>` |

**Vocabulary layout:** `vocab_final_v2/<dataset>/<variant>/<dataset>_<variant>_*.pickle`
- MoSE variants (filtered): `rbrics_filter`, `rdkit_fg_first_filter`,
  `ertl_first_filter`, `fg_first_filter`
- MotifSAT variants (unfiltered): `rbrics`, `rdkit_fg_first`, `ertl_first`, `fg_first`

**Per-dataset raw `data_root`:**
- FOLDS set (`*_Verified_GT`, hERG, Lipophilicity, BBBP, esol, Mutagenicity):
  `/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/`
- mutag: `$REPO/data`
- OGB (`ogbg-molbace`): `$REPO/data/ogb`

**What exists under the baseline (per model):**
- Datasets (10): Alkane_Carbonyl_Verified_GT, BBBP, Benzene_Verified_GT, esol,
  Fluoride_Carbonyl_Verified_GT, hERG, Lipophilicity, mutag, Mutagenicity, ogbg-molbace
- Folds present: 0–4. Backbones present: GIN, GCN, GAT, SAGE, PNA.
- Vocab variants: the 4 above per model, plus planted `_relabelled_dnf_k*_r*` variants
  for the datasets that support planting.

---

## Implementation (approved 2026-08-03)

The single **self-contained** worker `ablation_worker.sh` (repo root) is the canonical
implementation — it calls the models' `run.py` directly (no `run_experiments.sh` /
env-sourcing chain; only `conda` is sourced), enumerating **setup → dataset → variant →
backbone → fold** for both models. REAL vs PLANTED (relabelled) is selected by the `TIER`
env var (`real` default), so the two tiers run as independent submissions. See
**Deployment** below for the worker/launcher and the full worker copy.

Each cell flips exactly one knob off the default:

| setup | knob change | MoSE | MotifSAT |
|---|---|---|---|
| L2 | conv_normalize none→l2 | `--conv_normalize l2` | `--conv_normalize l2` |
| learn-unk | unk_mode fixed→learnable_shared | `--unk_mode learnable_shared` | — |
| linear | node_encoder onehot→linear | `--node_encoder linear` | `--node_encoder linear` |

**Faithfulness rests on verified `run.py` behavior**, not reconstructed values:
- `run.py` self-resolves `size_reg`/`ent_reg`/`num_layers` per (backbone,dataset) from
  `reg_config.py` (so they are NOT passed).
- `--processed_root` is the **base** `processed_final_v2`; `run.py` appends `/<variant>`.
- Static injection inlined: MoSE `--w_feat --w_readout`; MotifSAT
  `--w_feat --w_message --w_readout --motif_method readout --noise motif
  --info_loss_level motif`. Both `--epochs 500`; `graph_pool` left at default `add`.
- mutag gets its `--mutag_*` split/index/seed flags.

**Scope covered:**
- Datasets (9): Mutagenicity, BBBP, hERG, Benzene_Verified_GT, Alkane_Carbonyl_Verified_GT, Fluoride_Carbonyl_Verified_GT,
  Lipophilicity, esol, mutag. **OGB (`ogbg-molbace`) is EXCLUDED — needs further work.**
- Variants (4/model): MoSE filtered set / MotifSAT unfiltered set (see above).
- Backbones (5): GIN, GCN, SAGE, GAT, PNA. Folds: 0–4 for all EXCEPT **mutag = fold0
  only** (1 fold; Mutagenicity has the full 5).
- **FULL SCOPE = real + planted.** REAL: the 4 base variants × 9 datasets. PLANTED
  (DNF): each script globs `final_v2/<family>/*_relabelled_dnf_*`, parses base+tier from
  the dir name, and reproduces every planted baseline run with the knob flipped
  (`--use_gt --gt_cache final_v2/gt_cache --gt_tier <tier>` [+ `--gt_vocab_variant
  <base>` for MoSE]). Planted datasets = BBBP, hERG, Mutagenicity (classification only;
  4 tiers dnf_k2_r1/k2_r2/k3_r1/k3_r2). **OGB skipped** in both real and planted.

Output: `ablated_completely_v1/<family>/<variant>/<dataset>/fold<k>/<run_dir>/` (run.py
names the run dir; the flipped knob is encoded in the name).

## Metrics hook (added 2026-08-03)
`--per_split_eval` on `MOSE-GNN/run.py` + `MotifSAT/run.py` (opt-in; default behavior
unchanged) makes training END-TO-END: after `summary.json`, it builds per-split GT
lists (inlined `split_lists_and_gt` logic) + motif scores (MoSE `get_motif_scores`;
MotifSAT `_aggregate_att_to_motif` over all graphs) and calls the shared
`evaluate_native_all_splits` → `summary_splits.json`, no checkpoint reload. Threaded as
a `run()` kwarg (NOT a cfg field, to avoid perturbing config_hash/variant_tag).

## Deployment — `ablation_worker.sh` + `ablation_launch.sh`

The **worker** (`ablation_worker.sh`) is the canonical deployment executor. It is
self-contained: it enumerates every cell (`setup × dataset × variant × backbone × fold`,
both models, real or planted per `TIER`) and runs `run.py --per_split_eval` with a
**claim-based pull** guard. It is the sole executor (the earlier per-tier sequential
scripts were removed to keep one source of truth).

**Dispatch model (per cell):**
- **DONE** = `$DONE_FILE` (default `summary.json`) exists in the cell's run dir → skip.
  (Set `DONE_FILE=summary_splits.json` to also require the per-split eval.)
- **CLAIM** = atomic `mkdir` of a cell-keyed dir under `_dispatch/claims/` → prevents
  duplicate concurrent runs; shared across GPU+CPU pools (so a cell can never run twice).
- **FAILURE** = `run.py` rc≠0 → appended to `_dispatch/failures.tsv` (`ts, host_job,
  cell_id, rc`) + a `.failed` marker. **NO auto-redeploy** — failed/orphaned cells are
  skipped by every worker; re-kick is manual.
- **DYNAMIC** — launch any number of workers; add more anytime (re-run the launcher);
  move datasets between GPU/CPU pools freely (claims are cell-keyed → running cells never
  duplicate, only not-yet-started cells relocate).

**Launcher** (`ablation_launch.sh`) — deploys the two pools; re-runnable to add workers:
```bash
# first attempt: REAL only (planted staged). GPU=big datasets, CPU=small.
NGPU=44 NCPU=125 CPU_CORES=2 TIER=real bash ablation_launch.sh
# add more workers later (safe): re-run it.
# planted, when ready:  TIER=planted bash ablation_launch.sh
# move a dataset GPU->CPU mid-run: edit GPU_DATASETS/CPU_DATASETS, re-run. Safe.
```
Env: `NGPU NCPU CPU_CORES TIER GPU_DATASETS CPU_DATASETS GPU_PART CPU_PART DONE_FILE`.
Defaults: GPU=`Benzene_Verified_GT hERG Alkane_Carbonyl_Verified_GT Mutagenicity`,
CPU=`Fluoride_Carbonyl_Verified_GT Lipophilicity mutag BBBP esol`, parts `preempt` /
`preempt,share`.

### Killing / pausing / re-kicking
```bash
B=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor; D=$B/ablated_completely_v1/_dispatch
# KILL all ablation workers (GPU+CPU):
squeue -u kokatea -h -o "%i %j" | awk '$2 ~ /^abl_/ {print $1}' | xargs -r scancel
# kill one pool:  scancel -u kokatea --name=abl_gpu   (or --name=abl_cpu)
# kill one worker: scancel <JOBID>
# PAUSE (let running cells finish, stop new): cancel only PENDING workers:
squeue -u kokatea -h -t PD -o "%i %j" | awk '$2 ~ /^abl_/ {print $1}' | xargs -r scancel

# FAILURES (exact setups that failed): $D/failures.tsv
# RE-KICK (manual; no auto-redeploy). With NO workers running:
#   retry all FAILED cells:
for c in "$D"/claims/*/; do [ -e "$c/.failed" ] && rm -rf "$c"; done
#   also reset ORPHANED claims from killed/preempted workers (claim, no .failed, not done):
#   (safe only when no workers are running) then re-launch:
bash "$B/ablation_launch.sh"    # workers re-pick the now-unclaimed cells
```

### Worker script (`ablation_worker.sh`) — full copy
```bash
#!/usr/bin/env bash
# ablation_worker.sh — claim-based PULL worker. One SLURM job = one worker; launch many
# (via ablation_launch.sh) and they self-balance over the filesystem. DONE=summary.json,
# CLAIM=atomic mkdir, FAILURE->failures.tsv (+ .failed), NO auto-redeploy. ENV: POOL_DATASETS
# (req), DEVICE=cuda|cpu, TIER=real|planted, BACKBONES, DONE_FILE.
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
cd "$REPO"; export PYTHONPATH=$REPO WANDB_MODE=disabled
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
[ "${DEVICE:-cuda}" = cpu ] && export CUDA_VISIBLE_DEVICES=""
VOCAB=$REPO/vocab_final_v2; PROC=$REPO/processed_final_v2; OUT=$REPO/ablated_completely_v1
BASE=$REPO/final_v2; GTCACHE=$BASE/gt_cache
FOLDS_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/; MUTAG_ROOT=$REPO/data
DISPATCH=$OUT/_dispatch; CLAIMS=$DISPATCH/claims; FAILURES=$DISPATCH/failures.tsv
mkdir -p "$CLAIMS"; [ -s "$FAILURES" ] || printf 'ts\thost_job\tcell_id\trc\n' > "$FAILURES"
WHO="$(hostname -s):${SLURM_JOB_ID:-$$}"
TIER="${TIER:-real}"; DONE_FILE="${DONE_FILE:-summary.json}"
BACKBONES="${BACKBONES:-GIN GCN SAGE GAT PNA}"
: "${POOL_DATASETS:?set POOL_DATASETS to the space-separated datasets for this pool}"
_data_root(){ [ "$1" = mutag ] && echo "$MUTAG_ROOT" || echo "$FOLDS_ROOT"; }
_folds(){ [ "$1" = mutag ] && echo 0 || echo "0 1 2 3 4"; }
_mutag(){ [ "$1" = mutag ] && echo "--mutag_index_maps_path $MUTAG_ROOT/mutag_$2_index_maps.pkl --mutag_smiles_csv_path $MUTAG_ROOT/mutag_$2.csv --mutag_splits_path $MUTAG_ROOT/mutag_$2_splits.pkl --mutag_seed 42"; }
_inpool(){ case " $POOL_DATASETS " in *" $1 "*) return 0;; *) return 1;; esac; }
run_cell(){
  local cellid=$1 family=$2 outvar=$3 ds=$4 f=$5 glob=$6; shift 6; shift
  local parent="$OUT/$family/$outvar/$ds/fold$f"
  compgen -G "$parent/$glob/$DONE_FILE" >/dev/null 2>&1 && return 0
  mkdir "$CLAIMS/$cellid" 2>/dev/null || return 0
  echo "$WHO $(date +%s)" > "$CLAIMS/$cellid/info"; echo "[run $DEVICE] $cellid"
  "$@"; local rc=$?
  if [ "$rc" -eq 0 ] && compgen -G "$parent/$glob/$DONE_FILE" >/dev/null 2>&1; then return 0; fi
  printf '%s\t%s\t%s\t%s\n' "$(date +%s)" "$WHO" "$cellid" "$rc" >> "$FAILURES"
  touch "$CLAIMS/$cellid/.failed"; echo "[FAIL rc=$rc] $cellid"
}
mose_run(){   # setup norm unk enc  MODE=real|planted
  local setup=$1 norm=$2 unk=$3 enc=$4 mode=$5 bb f ds dr variant G
  if [ "$mode" = real ]; then
    for ds in $POOL_DATASETS; do _inpool "$ds" || continue; dr=$(_data_root "$ds")
      for variant in rbrics_filter rdkit_fg_first_filter ertl_first_filter fg_first_filter; do
        for bb in $BACKBONES; do for f in $(_folds "$ds"); do
          G="${bb}_${enc}_norm-${norm}_*unk-${unk}_*"
          run_cell "mose__real__${variant}__${ds}__f${f}__${bb}__${setup}" mose "$variant" "$ds" "$f" "$G" -- \
            python3 MOSE-GNN/run.py --dataset "$ds" --fold "$f" --backbone "$bb" \
              --node_encoder "$enc" --conv_normalize "$norm" --unk_mode "$unk" \
              --w_feat --w_readout --epochs 500 --per_split_eval \
              --data_root "$dr" --vocab_root "$VOCAB" --vocab_variant "$variant" \
              --processed_root "$PROC" --out_dir "$OUT/mose/$variant" $(_mutag "$ds" "$f")
        done; done; done; done
  else
    for gtdir in "$BASE"/mose/*_relabelled_dnf_*; do [ -d "$gtdir" ] || continue
      local gv=$(basename "$gtdir") bv tier gtvoc; bv="${gv%%_relabelled_*}"; tier="${gv#*_relabelled_}"; gtvoc="${bv%_filter}"
      for dsdir in "$gtdir"/*/; do ds=$(basename "$dsdir"); case "$ds" in ogbg-*) continue;; esac
        _inpool "$ds" || continue; dr=$(_data_root "$ds")
        for bb in $BACKBONES; do for f in $(_folds "$ds"); do [ -d "$gtdir/$ds/fold$f" ] || continue
          G="${bb}_${enc}_norm-${norm}_*unk-${unk}_*"
          run_cell "mose__planted__${gv}__${ds}__f${f}__${bb}__${setup}" mose "$gv" "$ds" "$f" "$G" -- \
            python3 MOSE-GNN/run.py --dataset "$ds" --fold "$f" --backbone "$bb" \
              --node_encoder "$enc" --conv_normalize "$norm" --unk_mode "$unk" \
              --w_feat --w_readout --epochs 500 --per_split_eval \
              --use_gt --gt_cache "$GTCACHE" --gt_tier "$tier" --gt_vocab_variant "$gtvoc" \
              --data_root "$dr" --vocab_root "$VOCAB" --vocab_variant "$bv" \
              --processed_root "$PROC" --out_dir "$OUT/mose/$gv" $(_mutag "$ds" "$f")
        done; done; done; done
  fi
}
ms_run(){     # setup norm enc  MODE=real|planted
  local setup=$1 norm=$2 enc=$3 mode=$4 bb f ds dr variant G
  if [ "$mode" = real ]; then
    for ds in $POOL_DATASETS; do _inpool "$ds" || continue; dr=$(_data_root "$ds")
      for variant in rbrics rdkit_fg_first ertl_first fg_first; do
        for bb in $BACKBONES; do for f in $(_folds "$ds"); do
          G="${bb}_readout_${enc}_norm-${norm}_*"
          run_cell "motifsat__real__${variant}__${ds}__f${f}__${bb}__${setup}" motifsat "$variant" "$ds" "$f" "$G" -- \
            python3 MotifSAT/run.py --dataset "$ds" --fold "$f" --backbone "$bb" \
              --node_encoder "$enc" --conv_normalize "$norm" \
              --motif_method readout --noise motif --info_loss_level motif \
              --w_feat --w_message --w_readout --epochs 500 --per_split_eval \
              --data_root "$dr" --vocab_root "$VOCAB" --vocab_variant "$variant" \
              --processed_root "$PROC" --out_dir "$OUT/motifsat/$variant" $(_mutag "$ds" "$f")
        done; done; done; done
  else
    for gtdir in "$BASE"/motifsat/*_relabelled_dnf_*; do [ -d "$gtdir" ] || continue
      local gv=$(basename "$gtdir") bv tier; bv="${gv%%_relabelled_*}"; tier="${gv#*_relabelled_}"
      for dsdir in "$gtdir"/*/; do ds=$(basename "$dsdir"); case "$ds" in ogbg-*) continue;; esac
        _inpool "$ds" || continue; dr=$(_data_root "$ds")
        for bb in $BACKBONES; do for f in $(_folds "$ds"); do [ -d "$gtdir/$ds/fold$f" ] || continue
          G="${bb}_readout_${enc}_norm-${norm}_*"
          run_cell "motifsat__planted__${gv}__${ds}__f${f}__${bb}__${setup}" motifsat "$gv" "$ds" "$f" "$G" -- \
            python3 MotifSAT/run.py --dataset "$ds" --fold "$f" --backbone "$bb" \
              --node_encoder "$enc" --conv_normalize "$norm" \
              --motif_method readout --noise motif --info_loss_level motif \
              --w_feat --w_message --w_readout --epochs 500 --per_split_eval \
              --use_gt --gt_cache "$GTCACHE" --gt_tier "$tier" \
              --data_root "$dr" --vocab_root "$VOCAB" --vocab_variant "$bv" \
              --processed_root "$PROC" --out_dir "$OUT/motifsat/$gv" $(_mutag "$ds" "$f")
        done; done; done; done
  fi
}
echo "worker $WHO  DEVICE=${DEVICE:-cuda}  TIER=$TIER  DONE_FILE=$DONE_FILE  POOL=[$POOL_DATASETS]"
mose_run l2       l2   fixed            onehot "$TIER"
mose_run learnunk none learnable_shared onehot "$TIER"
mose_run linear   none fixed            linear "$TIER"
ms_run   l2       l2   onehot                  "$TIER"
ms_run   linear   none linear                  "$TIER"
echo "worker $WHO DONE (TIER=$TIER)."
```

## Still OPEN
- **Harvester** — `build_metric_set.py` needs to parse the ablation knob from the run-dir
  name (post-run; doesn't block deployment).
- **OGB** — deferred (excluded from both real and planted).
