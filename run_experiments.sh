#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh
# Phased experiment pipeline.
#
# Setup (once):
#   1. Edit experiment_config.sh with your paths
#   2. source experiment_config.sh
#   3. bash run_experiments.sh phase1
#
# Phases:
#   phase0           export mutag/OGB CSV bridges (DATASETS_SPECIAL)
#   phase1           fragmentation, no threshold (all 3 variants)
#   phase2           coverage vs threshold sweep  (review, then edit CHOSEN_THRESHOLD)
#   phase3           thresholded vocabularies     (reads CHOSEN_THRESHOLD dict)
#   phase4           synthetic GT                 (requires RULE_INDEX)
#   phase5_vanilla   train Vanilla GNN
#   phase5_mose      train MOSE-GNN
#   phase5_gsat      train base GSAT
#   phase5_baselines post-hoc explainers on vanilla
#   phase5_motifsat  train MotifSAT
#   collect          print results table
#
# Three fragmentation variants:
#   rbrics_old       — rbrics_only (legacy DomainDrivenGlobalExpl, ablation)
#   rbrics           — rBRICS + reBRICS, no fallback, no BPE
#   all_fallback_bpe — full cascade, fallback, BPE
# =============================================================================
set -e

# ── Load config ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPT_CONFIG="${EXPT_CONFIG:-${SCRIPT_DIR}/experiment_config.sh}"

if [ -f "$EXPT_CONFIG" ]; then
    source "$EXPT_CONFIG"
    echo "Loaded config: $EXPT_CONFIG"
else
    echo "[warn] Config file not found: $EXPT_CONFIG"
fi

# ── Defaults ──────────────────────────────────────────────────────────────────
PROJECT="${PROJECT:-/path/to/project}"
DATA_ROOT="${DATA_ROOT:-./FOLDS}"
VOCAB_ROOT="${VOCAB_ROOT:-./vocab_output}"
OUT_ROOT="${OUT_ROOT:-./results}"
FOLDS="${FOLDS:-0 1 2 3 4}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${PROJECT}/processed}"  # PyG .pt cache
MUTAG_DATA_ROOT="${MUTAG_DATA_ROOT:-${PROJECT}/data}"
OGB_DATA_ROOT="${OGB_DATA_ROOT:-${PROJECT}/data/ogb}"
DATASETS_CSV="${DATASETS_CSV:-Mutagenicity BBBP Benzene}"
DATASETS_SPECIAL="${DATASETS_SPECIAL:-mutag ogbg-molhiv}"
DATASETS="${DATASETS:-$DATASETS_CSV $DATASETS_SPECIAL}"
CONV_NORMALIZE="${CONV_NORMALIZE:-l2}"
MOSE_RUN_MULTI_EXPLANATION="${MOSE_RUN_MULTI_EXPLANATION:-1}"
RULE_INDEX="${RULE_INDEX:-}"
WANDB_FLAGS="${WANDB_FLAGS:-}"      # e.g. "--use_wandb --wandb_project MyProject"
# MotifSAT message injection (w_message).  Prior to the argparse fix, w_message
# defaulted to True and could not be disabled, so every base-GSAT and MotifSAT
# run used message injection.  This toggle preserves that behaviour by default.
# Set MOTIFSAT_W_MESSAGE=0 to train without message injection.
MOTIFSAT_W_MESSAGE="${MOTIFSAT_W_MESSAGE:-1}"
if [ "$MOTIFSAT_W_MESSAGE" = "1" ]; then
    WM_FLAG="--w_message"
else
    WM_FLAG=""
fi
_mose_extra_flags() {
    local extra=""
    [ "$MOSE_RUN_MULTI_EXPLANATION" = "1" ] && extra="--run_multi_explanation"
    echo "$extra"
}
# Rule ranking for rules.json (rule_index 0 = best). 'balanced' sorts by
# balance × separation × (1-spurious) to target a ~50/50 synthetic GT split and
# penalise spurious/subsuming motifs; 'pct1' is the legacy positive-coverage
# sort. Either way, all score components are written to rules_summary.csv so you
# can inspect and override RULE_INDEX.
RULE_RANK="${RULE_RANK:-balanced}"
# NOTE: thresholds are now per-dataset dicts — edit CHOSEN_THRESHOLD
#       in MotifBreakdown/generate_vocab_rules.py instead.

# ── Validate ──────────────────────────────────────────────────────────────────
_check_paths() {
    local ok=1
    [ "$PROJECT" = "/path/to/project" ] && \
        echo "ERROR: PROJECT is still placeholder. Edit experiment_config.sh" && ok=0
    [ ! -d "$PROJECT" ] && \
        echo "ERROR: PROJECT not found: $PROJECT" && ok=0
    [ ! -d "$DATA_ROOT" ] && \
        echo "ERROR: DATA_ROOT not found: $DATA_ROOT" && ok=0
    [ "$ok" = "0" ] && exit 1
    mkdir -p "$VOCAB_ROOT" "$OUT_ROOT"
    echo "  PROJECT    = $PROJECT"
    echo "  DATA_ROOT  = $DATA_ROOT"
    echo "  VOCAB_ROOT = $VOCAB_ROOT"
    echo "  OUT_ROOT   = $OUT_ROOT"
    echo "  DATASETS   = $DATASETS"
    echo "  CSV        = $DATASETS_CSV"
    echo "  SPECIAL    = $DATASETS_SPECIAL"
    echo "  CONV_NORM  = $CONV_NORMALIZE"
    echo "  FOLDS      = $FOLDS"
    echo "  BACKBONES  = $BACKBONES"
    echo "  PROCESSED  = $PROCESSED_ROOT"
    echo "  MUTAG_ROOT = $MUTAG_DATA_ROOT"
    echo "  OGB_ROOT   = $OGB_DATA_ROOT"
    echo "  EPOCHS     = $EPOCHS"
    echo "  Thresholds = per-dataset dict in generate_vocab_rules.py"
}

# ── Vocabulary variant names ───────────────────────────────────────────────────
# Three base variants (no threshold):
V_OLD="rbrics_old"               # method=rbrics_only, legacy behaviour
V_RBRICS="rbrics"                # method=rbrics, no fallback, no BPE
V_ALL="all_fallback_bpe"         # method=all, fallback, BPE
V_ALL_SHATTER="all_fallback_bpe_shatter"  # method=all + mild-shatter floor (auto-suffixed)

# Three filtered variants (threshold applied):
V_OLD_TH="rbrics_old_filter"
V_RBRICS_TH="rbrics_filter"
V_ALL_TH="all_fallback_bpe_filter"

# GT-relabelled variant (from phase4):
V_ALL_GT="${V_ALL}_relabelled"

# ── Dataset routing (Mutagenicity CSV ≠ mutag TUDataset; OGB uses fold 0) ─────
_dataset_data_root() {
    case "$1" in
        mutag) echo "$MUTAG_DATA_ROOT" ;;
        ogbg-*) echo "$OGB_DATA_ROOT" ;;
        *) echo "$DATA_ROOT" ;;
    esac
}

_dataset_node_encoder() {
    case "$1" in
        ogbg-*) echo "atom_encoder" ;;
        *) echo "$NODE_ENCODER" ;;
    esac
}

# Skip fold>0 for OGB/mutag (artifacts are fold-0 only).
_skip_redundant_fold() {
    case "$1" in mutag|ogbg-*) [ "$2" != "0" ] && return 0 ;; esac
    return 1
}

_mutag_train_flags() {
    local ds=$1 fold=$2
    [ "$ds" != "mutag" ] && return 0
    local root="$(_dataset_data_root mutag)"
    echo "--mutag_index_maps_path $root/mutag_${fold}_index_maps.pkl" \
         "--mutag_smiles_csv_path $root/mutag_${fold}.csv" \
         "--mutag_splits_path $root/mutag_${fold}_splits.pkl" \
         "--mutag_seed 42"
}

# ── Helper: fragment one variant ──────────────────────────────────────────────
# Usage: run_frag <method> <fallback:0|1> <bpe:0|1> <out_variant> [shatter:0|1]
run_frag() {
    local method=$1 use_fallback=$2 use_bpe=$3 variant=$4 use_shatter=${5:-0}
    echo "  [$variant] method=$method fallback=$use_fallback bpe=$use_bpe shatter=$use_shatter"
    for ds in $DATASETS; do
        ds_root="$(_dataset_data_root "$ds")"
        ds_fold=0
        python3 "$PROJECT/MotifBreakdown/generate_vocab_rules.py" \
            --datasets  "$ds" \
            --data_root "$ds_root" \
            --out_dir   "$VOCAB_ROOT" \
            --method    "$method" \
            --variant   "$variant" \
            $( [ "$use_fallback" = "1" ] && echo "--fallback" ) \
            $( [ "$use_bpe"      = "1" ] && echo "--bpe" ) \
            $( [ "$use_shatter"  = "1" ] && echo "--shatter" ) \
            --rule_rank "$RULE_RANK" \
            --fold 0
    done
}

# Usage: run_frag_thresh <method> <fallback:0|1> <bpe:0|1> <threshold_pct> <out_variant>
run_frag_thresh() {
    # Threshold is looked up from CHOSEN_THRESHOLD in generate_vocab_rules.py
    # keyed by variant name × dataset name — no shell variable needed.
    local method=$1 use_fallback=$2 use_bpe=$3 variant=$4
    echo "  [$variant] method=$method (threshold from CHOSEN_THRESHOLD dict)"
    for ds in $DATASETS; do
        ds_root="$(_dataset_data_root "$ds")"
        python3 "$PROJECT/MotifBreakdown/generate_vocab_rules.py" \
            --datasets      "$ds" \
            --data_root     "$ds_root" \
            --out_dir       "$VOCAB_ROOT" \
            --method        "$method" \
            --variant       "$variant" \
            $( [ "$use_fallback" = "1" ] && echo "--fallback" ) \
            $( [ "$use_bpe"      = "1" ] && echo "--bpe" ) \
            --apply_threshold \
            --rule_rank "$RULE_RANK" \
            --fold 0
    done
}

# ── Helper: training runners ───────────────────────────────────────────────────
run_vanilla() {
    local variant=$1
    for backbone in $BACKBONES; do
        echo "  [Vanilla] backbone=$backbone encoder=$NODE_ENCODER vocab=$variant"
        for ds in $DATASETS; do
            for fold in $FOLDS; do
                _skip_redundant_fold "$ds" "$fold" && continue
                local ds_root="$(_dataset_data_root "$ds")"
                local enc="$(_dataset_node_encoder "$ds")"
                local eff_fold="$fold"
                case "$ds" in mutag|ogbg-*) eff_fold=0 ;; esac
                python3 "$PROJECT/SharedModules/baselines/run_vanilla.py" \
                    --dataset      "$ds" --fold "$eff_fold" \
                    --backbone     "$backbone" --node_encoder "$enc" \
                    --epochs       "$EPOCHS" \
                    --data_root    "$ds_root" \
                    --vocab_root   "$VOCAB_ROOT" \
                    --vocab_variant "$variant" \
                    --conv_normalize "$CONV_NORMALIZE" \
                    --processed_root "$PROCESSED_ROOT" \
                    --out_dir      "$OUT_ROOT/vanilla/${variant}" \
                    $(_mutag_train_flags "$ds" "$eff_fold") \
                    $WANDB_FLAGS
            done
        done
    done
}

run_mose() {
    local variant=$1 inj_args=$2
    for backbone in $BACKBONES; do
        echo "  [MOSE] backbone=$backbone vocab=$variant inj=$inj_args"
        for ds in $DATASETS; do
            for fold in $FOLDS; do
                _skip_redundant_fold "$ds" "$fold" && continue
                local ds_root="$(_dataset_data_root "$ds")"
                local enc="$(_dataset_node_encoder "$ds")"
                local eff_fold="$fold"
                case "$ds" in mutag|ogbg-*) eff_fold=0 ;; esac
                python3 "$PROJECT/MOSE-GNN/run.py" \
                    --dataset      "$ds" --fold "$eff_fold" \
                    --backbone     "$backbone" --node_encoder "$enc" \
                    $inj_args \
                    --epochs       "$EPOCHS" \
                    --data_root    "$ds_root" \
                    --vocab_root   "$VOCAB_ROOT" \
                    --vocab_variant "$variant" \
                    --conv_normalize "$CONV_NORMALIZE" \
                    --processed_root "$PROCESSED_ROOT" \
                    --out_dir      "$OUT_ROOT/mose/${variant}" \
                    $(_mutag_train_flags "$ds" "$eff_fold") \
                    $(_mose_extra_flags) \
                    $WANDB_FLAGS
            done
        done
    done
}

run_gsat() {
    local variant=$1
    for backbone in $BACKBONES; do
        echo "  [BaseGSAT] backbone=$backbone vocab=$variant"
        for ds in $DATASETS; do
            for fold in $FOLDS; do
                _skip_redundant_fold "$ds" "$fold" && continue
                local ds_root="$(_dataset_data_root "$ds")"
                local enc="$(_dataset_node_encoder "$ds")"
                local eff_fold="$fold"
                case "$ds" in mutag|ogbg-*) eff_fold=0 ;; esac
                python3 "$PROJECT/MotifSAT/run.py" \
                    --dataset         "$ds" --fold "$eff_fold" \
                    --backbone        "$backbone" --node_encoder "$enc" \
                    --motif_method    none \
                    --learn_edge_att \
                    --noise           node \
                    --info_loss_level node \
                    --info_loss_coef  1.0 \
                    --epochs          "$EPOCHS" \
                    --data_root       "$ds_root" \
                    --vocab_root      "$VOCAB_ROOT" \
                    --vocab_variant   "$variant" \
                    --conv_normalize  "$CONV_NORMALIZE" \
                    --processed_root  "$PROCESSED_ROOT" \
                    --out_dir         "$OUT_ROOT/base_gsat/${variant}" \
                    $(_mutag_train_flags "$ds" "$eff_fold") \
                    $WANDB_FLAGS
            done
        done
    done
}

run_motifsat() {
    local variant=$1
    for backbone in $BACKBONES; do
        echo "  [MotifSAT readout] backbone=$backbone vocab=$variant"
        for ds in $DATASETS; do
            for fold in $FOLDS; do
                _skip_redundant_fold "$ds" "$fold" && continue
                local ds_root="$(_dataset_data_root "$ds")"
                local enc="$(_dataset_node_encoder "$ds")"
                local eff_fold="$fold"
                case "$ds" in mutag|ogbg-*) eff_fold=0 ;; esac
                python3 "$PROJECT/MotifSAT/run.py" \
                    --dataset         "$ds" --fold "$eff_fold" \
                    --backbone        "$backbone" --node_encoder "$enc" \
                    --motif_method    readout \
                    --noise           none \
                    --info_loss_level none \
                    --info_loss_coef  0.0 \
                    --w_feat --w_readout $WM_FLAG \
                    --epochs          "$EPOCHS" \
                    --data_root       "$ds_root" \
                    --vocab_root      "$VOCAB_ROOT" \
                    --vocab_variant   "$variant" \
                    --conv_normalize  "$CONV_NORMALIZE" \
                    --processed_root  "$PROCESSED_ROOT" \
                    --out_dir         "$OUT_ROOT/motifsat/${variant}" \
                    $(_mutag_train_flags "$ds" "$eff_fold") \
                    $WANDB_FLAGS
            done
        done
    done
}

run_baselines() {
    # Re-run vanilla with epochs=0 (load weights) to apply post-hoc explainers
    # under a specific vocabulary for motif-level evaluation.
    # For filtered variants (*_filter), load the weights trained on the
    # corresponding unfiltered variant (model weights are independent of
    # the vocabulary threshold — only the motif eval vocab changes).
    local eval_variant=$1
    # Resolve which vanilla weights to load
    local weight_variant="$eval_variant"
    case "$eval_variant" in
        "${V_OLD_TH}")    weight_variant="$V_OLD" ;;
        "${V_RBRICS_TH}") weight_variant="$V_RBRICS" ;;
        "${V_ALL_TH}")    weight_variant="$V_ALL" ;;
    esac
    for backbone in $BACKBONES; do
        echo "  [Baselines eval] backbone=$backbone vocab=$eval_variant  weights=vanilla/$weight_variant"
        for ds in $DATASETS; do
            for fold in $FOLDS; do
                _skip_redundant_fold "$ds" "$fold" && continue
                local ds_root="$(_dataset_data_root "$ds")"
                local enc="$(_dataset_node_encoder "$ds")"
                local eff_fold="$fold"
                case "$ds" in mutag|ogbg-*) eff_fold=0 ;; esac
                python3 "$PROJECT/SharedModules/baselines/run_vanilla.py" \
                    --dataset      "$ds" --fold "$eff_fold" \
                    --backbone     "$backbone" --node_encoder "$enc" \
                    --epochs       0 \
                    --data_root    "$ds_root" \
                    --vocab_root   "$VOCAB_ROOT" \
                    --vocab_variant "$eval_variant" \
                    --conv_normalize "$CONV_NORMALIZE" \
                    --processed_root "$PROCESSED_ROOT" \
                    --load_weights_from "$OUT_ROOT/vanilla/${weight_variant}" \
                    --weight_vocab_variant "$weight_variant" \
                    --out_dir      "$OUT_ROOT/baselines/${eval_variant}" \
                    $(_mutag_train_flags "$ds" "$eff_fold") \
                    $WANDB_FLAGS
            done
        done
    done
}

apply_gt() {
    # Write relabelled graph objects to gt_cache for CSV datasets × folds.
    local variant=$1 rule_idx=$2
    echo "  [SyntheticGT] vocab=$variant rule=$rule_idx"
    for ds in $DATASETS_CSV; do
        for fold in $FOLDS; do
            python3 "$PROJECT/SharedModules/data/apply_gt.py" \
                --dataset    "$ds" \
                --fold       "$fold" \
                --vocab_root "$VOCAB_ROOT" \
                --variant    "$variant" \
                --out_dir    "$OUT_ROOT/gt_cache" \
                --rule_index "$rule_idx" \
                --data_root  "$DATA_ROOT" \
                --processed_root "$PROCESSED_ROOT" \
             || { echo "  [error] apply_gt.py failed for $ds fold $fold — see output above"; exit 1; }
        done
    done
}

# =============================================================================
# PHASE 0 — Export mutag / OGB CSV bridges for vocab generation
#   Writes {mutag|ogbg-*}_0.csv (+ mutag index maps / splits) under the
#   dataset-specific roots so phase1 _dataset_data_root finds them.
# =============================================================================
phase0() {
    _check_paths
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " PHASE 0 — Export mutag / OGB CSV bridges"
    echo "══════════════════════════════════════════════════════════"

    for ds in $DATASETS_SPECIAL; do
        case "$ds" in
            mutag)
                local csv="$MUTAG_DATA_ROOT/mutag_0.csv"
                echo "  [mutag] → $MUTAG_DATA_ROOT"
                if [ -f "$csv" ]; then
                    echo "    skip (exists): $csv"
                else
                    python3 "$PROJECT/MotifBreakdown/export_mutag_dataset_to_csv.py" \
                        --data_root "$MUTAG_DATA_ROOT" \
                        --out_dir   "$MUTAG_DATA_ROOT" \
                        --fold 0 --seed 42
                fi
                ;;
            ogbg-*)
                local csv="$OGB_DATA_ROOT/${ds}_0.csv"
                echo "  [$ds] → $OGB_DATA_ROOT"
                if [ -f "$csv" ]; then
                    echo "    skip (exists): $csv"
                else
                    python3 "$PROJECT/MotifBreakdown/export_ogb_to_csv.py" \
                        --dataset  "$ds" \
                        --ogb_root   "$OGB_DATA_ROOT" \
                        --out_dir    "$OGB_DATA_ROOT" \
                        --fold 0
                fi
                ;;
            *)
                echo "  [warn] unknown special dataset (no export script): $ds"
                ;;
        esac
    done

    echo ""
    echo "Phase 0 complete."
    echo "Next: bash run_experiments.sh phase1"
}

# =============================================================================
# PHASE 1 — Fragmentation, no threshold
#   Three variants: rbrics_old, rbrics, all_fallback_bpe
# =============================================================================
phase1() {
    _check_paths
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " PHASE 1 — Fragmentation (no threshold, 3 variants)"
    echo "══════════════════════════════════════════════════════════"

    echo "1a. rbrics_old  (legacy rbrics_only — ablation baseline)"
    run_frag rbrics_only 0 0 "$V_OLD"

    echo "1b. rbrics  (rBRICS + reBRICS, no fallback, no BPE)"
    run_frag rbrics 0 0 "$V_RBRICS"

    echo "1c. all_fallback_bpe  (full cascade, fallback, BPE)"
    run_frag all 1 1 "$V_ALL"

    echo "1d. all_fallback_bpe_shatter  (full cascade + mild-shatter floor)"
    # NOTE: --shatter auto-appends '_shatter' to the variant name, so we pass the
    # BASE name "$V_ALL" here; the output lands in "$V_ALL_SHATTER".
    run_frag all 1 1 "$V_ALL" 1

    echo ""
    echo "Phase 1 complete. Vocabularies in: $VOCAB_ROOT"
    echo "Variants: $V_OLD  $V_RBRICS  $V_ALL  $V_ALL_SHATTER"
    echo "Next: bash run_experiments.sh phase2  (review coverage plots)"
}

# =============================================================================
# PHASE 2 — Coverage vs threshold sweep
#   All three base variants swept so you can compare curves side by side.
# =============================================================================
phase2() {
    _check_paths
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " PHASE 2 — Coverage vs threshold sweep (3 variants)"
    echo "══════════════════════════════════════════════════════════"

    for ds in $DATASETS; do
        for variant in "$V_OLD" "$V_RBRICS" "$V_ALL"; do
            echo "  [$ds / $variant]"
            python3 "$PROJECT/MotifBreakdown/coverage_vs_threshold.py" \
                --dataset    "$ds" \
                --vocab_root "$VOCAB_ROOT" \
                --variant    "$variant" \
                --out_dir    "$OUT_ROOT/coverage_plots"
        done
    done

    echo ""
    echo "Phase 2 complete. Review plots in: $OUT_ROOT/coverage_plots"
    echo "Then:  edit CHOSEN_THRESHOLD in MotifBreakdown/generate_vocab_rules.py"
    echo "       bash run_experiments.sh phase3"
}

# =============================================================================
# PHASE 3 — Thresholded vocabularies
#   All three variants re-fragmented with threshold applied.
# =============================================================================
phase3() {
    # Thresholds are read from CHOSEN_THRESHOLD in generate_vocab_rules.py.
    # Edit that dict (keyed by variant name × dataset) instead of setting
    # a shell variable.  No THRESHOLD env var needed.
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " PHASE 3 — Thresholded vocabularies (per-dataset CHOSEN_THRESHOLD)"
    echo "══════════════════════════════════════════════════════════"

    echo "3a. rbrics_old_filter"
    run_frag_thresh rbrics_only 0 0 "$V_OLD_TH"

    echo "3b. rbrics_filter"
    run_frag_thresh rbrics 0 0 "$V_RBRICS_TH"

    echo "3c. all_fallback_bpe_filter"
    run_frag_thresh all 1 1 "$V_ALL_TH"

    echo ""
    echo "Phase 3 complete.  Six vocabularies now available:"
    echo "  No threshold: $V_OLD  $V_RBRICS  $V_ALL"
    echo "  Filtered:     $V_OLD_TH  $V_RBRICS_TH  $V_ALL_TH"
    echo ""
    echo "To change thresholds: edit CHOSEN_THRESHOLD in"
    echo "  MotifBreakdown/generate_vocab_rules.py"
    echo ""
    echo "Next: review rules then:"
    echo "  export RULE_INDEX=<n>"
    echo "  bash run_experiments.sh phase4"
}

# =============================================================================
# PHASE 4 — Synthetic GT
#   Applied to V_ALL (no-threshold full cascade) only.
#   Result cached in $OUT_ROOT/gt_cache; loaded at training time via --use_gt.
# =============================================================================
phase4() {
    [ -z "$RULE_INDEX" ] && \
        echo "ERROR: set RULE_INDEX first.  export RULE_INDEX=0" && exit 1

    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " PHASE 4 — Synthetic GT (rule=$RULE_INDEX, vocab=$V_ALL)"
    echo "══════════════════════════════════════════════════════════"

    apply_gt "$V_ALL" "$RULE_INDEX"

    echo ""
    echo "Phase 4 complete.  GT cache: $OUT_ROOT/gt_cache"
    echo "Seven configurations now available:"
    echo "  1. $V_OLD             (no threshold)"
    echo "  2. $V_RBRICS          (no threshold)"
    echo "  3. $V_ALL             (no threshold)"
    echo "  4. $V_OLD_TH          (threshold per CHOSEN_THRESHOLD dict)"
    echo "  5. $V_RBRICS_TH       (threshold per CHOSEN_THRESHOLD dict)"
    echo "  6. $V_ALL_TH          (threshold per CHOSEN_THRESHOLD dict)"
    echo "  7. $V_ALL + GT        (relabelled, rule=$RULE_INDEX)"
    echo ""
    echo "Next: bash run_experiments.sh phase5_vanilla"
}

# =============================================================================
# PHASE 5a — Vanilla GNN
#   Train on all three base (no-threshold) variants so post-hoc explainers
#   can be evaluated under each fragmentation scheme independently.
# =============================================================================
phase5_vanilla() {
    _check_paths
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " PHASE 5a — Vanilla GNN (3 vocab variants)"
    echo "══════════════════════════════════════════════════════════"

    run_vanilla "$V_OLD"
    run_vanilla "$V_RBRICS"
    run_vanilla "$V_ALL"

    echo "Vanilla training complete."
}

# =============================================================================
# PHASE 5b — MOSE-GNN
#   Six configurations: all three filtered variants + all_fallback_bpe (no
#   threshold) + all_fallback_bpe with GT relabelling.
#   rbrics_old is an ablation — run with and without threshold.
# =============================================================================
phase5_mose() {
    _check_paths
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " PHASE 5b — MOSE-GNN (w_feat + w_readout)"
    echo "══════════════════════════════════════════════════════════"

    # Filtered variants (main comparison)
    run_mose "$V_OLD_TH"    "--w_feat --w_readout"
    run_mose "$V_RBRICS_TH" "--w_feat --w_readout"
    run_mose "$V_ALL_TH"    "--w_feat --w_readout"

    # No-threshold full cascade (ablation: does filtering help?)
    run_mose "$V_ALL"       "--w_feat --w_readout"

    # Synthetic GT relabelling (main novel contribution)
    # Requires phase4 to have been run.
    if [ -d "$OUT_ROOT/gt_cache" ]; then
        # GT-relabelled run: separate function to avoid word-splitting in inj_args
        for backbone in $BACKBONES; do
            echo "  [MOSE+GT] backbone=$backbone vocab=$V_ALL_GT"
            for ds in $DATASETS_CSV; do
                for fold in $FOLDS; do
                    local enc="$(_dataset_node_encoder "$ds")"
                    python3 "$PROJECT/MOSE-GNN/run.py" \
                        --dataset      "$ds" --fold "$fold" \
                        --backbone     "$backbone" --node_encoder "$enc" \
                        --w_feat --w_readout \
                        --use_gt --gt_cache "$OUT_ROOT/gt_cache" \
                        --epochs       "$EPOCHS" \
                        --data_root    "$DATA_ROOT" \
                        --vocab_root   "$VOCAB_ROOT" \
                        --vocab_variant "$V_ALL" \
                        --conv_normalize "$CONV_NORMALIZE" \
                        --processed_root "$PROCESSED_ROOT" \
                        --out_dir      "$OUT_ROOT/mose/${V_ALL_GT}" \
                        $(_mose_extra_flags) \
                        $WANDB_FLAGS
                done
            done
        done
    else
        echo "  [skip] $V_ALL_GT — run phase4 first"
    fi

    echo "MOSE training complete."
}

# =============================================================================
# PHASE 5c — Base GSAT
#   Comparison point: GSAT with no motif method, across all three base variants.
# =============================================================================
phase5_gsat() {
    _check_paths
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " PHASE 5c — Base GSAT (no motif method, 3 variants)"
    echo "══════════════════════════════════════════════════════════"

    run_gsat "$V_OLD"
    run_gsat "$V_RBRICS"
    run_gsat "$V_ALL"

    echo "Base GSAT training complete."
}

# =============================================================================
# PHASE 5d — Post-hoc baselines
#   GNNExplainer, PGExplainer, MAGE applied to each trained vanilla model,
#   evaluated under all six vocab variants for cross-comparison.
# =============================================================================
phase5_baselines() {
    _check_paths
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " PHASE 5d — Post-hoc baselines on vanilla (6 eval vocabs)"
    echo "══════════════════════════════════════════════════════════"

    # Evaluate each vocab variant (model weights fixed; vocab changes which
    # motifs get mapped to nodes for scoring).
    for eval_variant in \
        "$V_OLD"       "$V_RBRICS"    "$V_ALL" \
        "$V_OLD_TH"    "$V_RBRICS_TH" "$V_ALL_TH"; do
        run_baselines "$eval_variant"
    done

    echo "Baseline evaluation complete."
}

# =============================================================================
# PHASE 5e — MotifSAT
#   Readout-level motif aggregation, no IB, no noise.
#   learn_edge_att must be False for valid motif score aggregation.
#   Three base variants + GT-relabelled.
# =============================================================================
phase5_motifsat() {
    _check_paths
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " PHASE 5e — MotifSAT readout | no IB | no noise (3 variants)"
    echo "══════════════════════════════════════════════════════════"

    run_motifsat "$V_OLD"
    run_motifsat "$V_RBRICS"
    run_motifsat "$V_ALL"

    # GT-relabelled run: same V_ALL vocab + fragmentation; only data.y
    # and data.edge_label differ (set by phase4 apply_gt.py).
    # --use_gt replaces all three loaders with GT data; the model trains on
    # the rule-derived synthetic label and is evaluated against it.
    if [ -d "$OUT_ROOT/gt_cache" ]; then
        for backbone in $BACKBONES; do
            echo "  [MotifSAT+GT] backbone=$backbone vocab=$V_ALL_GT"
            for ds in $DATASETS_CSV; do
                for fold in $FOLDS; do
                    local enc="$(_dataset_node_encoder "$ds")"
                    python3 "$PROJECT/MotifSAT/run.py" \
                        --dataset         "$ds" --fold "$fold" \
                        --backbone        "$backbone" --node_encoder "$enc" \
                        --motif_method    readout \
                        --noise           none \
                        --info_loss_level none \
                        --info_loss_coef  0.0 \
                        --w_feat --w_readout $WM_FLAG \
                        --use_gt --gt_cache "$OUT_ROOT/gt_cache" \
                        --epochs          "$EPOCHS" \
                        --data_root       "$DATA_ROOT" \
                        --vocab_root      "$VOCAB_ROOT" \
                        --vocab_variant   "$V_ALL" \
                        --conv_normalize  "$CONV_NORMALIZE" \
                        --processed_root  "$PROCESSED_ROOT" \
                        --out_dir         "$OUT_ROOT/motifsat/${V_ALL_GT}" \
                        $WANDB_FLAGS
                done
            done
        done
    else
        echo "  [skip] $V_ALL_GT — run phase4 first"
    fi

    echo "MotifSAT training complete."
}

# =============================================================================
# Collect results
# =============================================================================
collect_results() {
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " Collecting results"
    echo "══════════════════════════════════════════════════════════"
    python3 - << PYEOF
import json, pandas as pd
from pathlib import Path

rows = []
for p in Path("$OUT_ROOT").rglob("summary.json"):
    try:
        d = json.load(open(p))
        d["exp_dir"] = str(p.parent.relative_to("$OUT_ROOT"))
        rows.append(d)
    except Exception:
        pass

if rows:
    df = pd.DataFrame(rows)
    # Core identifying + prediction columns, in a fixed order when present.
    core = [c for c in ["exp_dir","dataset","backbone","vocab_variant",
                        "motif_method","noise","info_loss_coef",
                        "ent_reg","size_reg","num_layers","explainer_lr","gnn_lr","conv_normalize","gin_inner_bn",
                        "train_auc","val_auc","auc",
                        "gt_roc_auc_mean","gt_roc_node_auc_mean","gt_roc_edge_auc_mean",
                        "gt_roc_node_mean_auc_mean","gt_roc_node_max_auc_mean",
                        "pearson","spearman",
                        "top_k_abs_disc","mean_abs_disc","score_disc_spearman",
                        "score_min","score_max","score_mean","score_std",
                        "score_median","score_mode","score_count"] if c in df]
    # Plus any explainer-specific metric columns (gnnexplainer_*, pgexplainer_*,
    # mage_*) so the baselines keep their per-explainer numbers.
    extra = sorted(c for c in df.columns
                   if c not in core and any(c.startswith(p) for p in
                   ("gnnexplainer_","pgexplainer_","mage_")))
    want = core + extra
    out = df[want].sort_values(["dataset","exp_dir"])
    print(out.to_string(index=False))
    out.to_csv("$OUT_ROOT/all_results.csv", index=False)
    print(f"\nSaved: $OUT_ROOT/all_results.csv  ({len(want)} columns)")
else:
    print("No summary.json files found yet.")
PYEOF
}

# =============================================================================
# Dispatcher
# =============================================================================
PHASE="${1:-}"
case "$PHASE" in
    phase0)           phase0 ;;
    phase1)           phase1 ;;
    phase2)           phase2 ;;
    phase3)           phase3 ;;
    phase4)           phase4 ;;
    phase5_vanilla)   phase5_vanilla ;;
    phase5_mose)      phase5_mose ;;
    phase5_gsat)      phase5_gsat ;;
    phase5_baselines) phase5_baselines ;;
    phase5_motifsat)  phase5_motifsat ;;
    collect)          collect_results ;;
    analyze|analysis)
        # Single entry point for all analysis + plots. Regenerates eval metrics
        # from checkpoints, rebuilds all_results.csv, writes pivot tables, and
        # draws the score-vs-impact grid. Pass --skip_regenerate via ANALYZE_ARGS
        # to use existing summaries.
        python3 "$PROJECT/analysis/run_analysis.py" all \
            --out_root "$OUT_ROOT" \
            --data_root "$DATA_ROOT" --vocab_root "$VOCAB_ROOT" \
            --mutag_data_root "$MUTAG_DATA_ROOT" \
            --ogb_data_root "$OGB_DATA_ROOT" \
            ${PROCESSED_ROOT:+--processed_root "$PROCESSED_ROOT"} \
            $ANALYZE_ARGS
        ;;
    "")
        echo "Usage: bash run_experiments.sh <phase>"
        echo ""
        echo "Phases:"
        echo "  phase0            export mutag/OGB CSV bridges (DATASETS_SPECIAL)"
        echo "  phase1            fragment all 3 variants (rbrics_old, rbrics, all_fallback_bpe)"
        echo "  phase2            coverage vs threshold sweep (review, then edit CHOSEN_THRESHOLD)"
        echo "  phase3            threshold all 3 variants  (reads CHOSEN_THRESHOLD)"
        echo "  phase4            synthetic GT               (requires RULE_INDEX)"
        echo "  phase5_vanilla    vanilla GNN (3 variants)"
        echo "  phase5_mose       MOSE-GNN (6 configs)"
        echo "  phase5_gsat       base GSAT (3 variants)"
        echo "  phase5_baselines  post-hoc on vanilla (6 eval vocabs)"
        echo "  phase5_motifsat   MotifSAT (3 variants + GT)"
        echo "  collect           print results table"
        echo "  analyze           regenerate eval + tables + plots (single entry point)"
        echo ""
        echo "Required env (set in experiment_config.sh):"
        echo "  PROJECT  DATA_ROOT  VOCAB_ROOT  OUT_ROOT"
        echo "  RULE_INDEX (phase4)   Thresholds: edit CHOSEN_THRESHOLD in generate_vocab_rules.py"
        ;;
    *)
        echo "Unknown phase: $PHASE"; exit 1 ;;
esac