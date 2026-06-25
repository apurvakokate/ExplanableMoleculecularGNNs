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
#   phase1           fragmentation, no threshold (4 variants)
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
# Four fragmentation variants:
#   rbrics_old       — CreateMotifVocab plot path (BreakrBRICSBonds + ToSmiles)
#   rbrics           — BreakrBRICSBonds (rBRICS else BRICS fallback) + reBRICS
#   rbrics_with_struct_fallback — same + structural fallback on single fragments
#   all_fallback_bpe — full v4 cascade, fallback, BPE
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
DATASETS_CSV="${DATASETS_CSV:-Mutagenicity BBBP hERG Benzene Alkane_Carbonyl Fluoride_Carbonyl Lipophilicity esol}"
DATASETS_SPECIAL="${DATASETS_SPECIAL:-mutag ogbg-molhiv ogbg-molbace}"
DATASETS="${DATASETS:-$DATASETS_CSV $DATASETS_SPECIAL}"
BACKBONES="${BACKBONES:-GIN GCN SAGE GAT PNA}"
CONV_NORMALIZE="${CONV_NORMALIZE:-l2}"
MOSE_RUN_MULTI_EXPLANATION="${MOSE_RUN_MULTI_EXPLANATION:-0}"
RULE_INDEX="${RULE_INDEX:-}"
# Optional phase4/5 subset: comma-separated short names, e.g. rbrics,all_fallback_bpe
# Aliases: old→rbrics_old  struct|struct_fallback→rbrics_with_struct_fallback  all|v4→all_fallback_bpe
# Unset = all four base fragmentation variants.
VOCAB_FOCUS="${VOCAB_FOCUS:-}"
WANDB_FLAGS="${WANDB_FLAGS:-}"      # e.g. "--use_wandb --wandb_project MyProject"
ENCODER_NORM="${ENCODER_NORM:-off}"
# Default injection presets (3-bit: w_feat / w_message / w_readout):
#   MOSE     101  (--w_feat --w_readout)
#   MotifSAT 111  (--w_feat --w_message --w_readout)
#   GSAT     010  (--w_message; node att → edge att via src×dst in gnn_base)
MOSE_INJ="${MOSE_INJ:---w_feat --w_readout}"
MOTIFSAT_INJ="${MOTIFSAT_INJ:---w_feat --w_message --w_readout}"
GSAT_INJ="${GSAT_INJ:---w_message}"
GSAT_LEARN_EDGE_ATT="${GSAT_LEARN_EDGE_ATT:-0}"
_mose_extra_flags() {
    local extra=""
    [ "$MOSE_RUN_MULTI_EXPLANATION" = "1" ] && extra="--run_multi_explanation"
    echo "$extra"
}
_gsat_learn_edge_att_flag() {
    [ "$GSAT_LEARN_EDGE_ATT" = "1" ] && echo "--learn_edge_att"
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
    if [ -n "$VOCAB_FOCUS" ]; then
        echo "  VOCAB_FOCUS= $VOCAB_FOCUS  → base:$(_vocab_focus_base_variants)"
    else
        echo "  VOCAB_FOCUS= (all four base variants)"
    fi
}

# ── Vocabulary variant names ───────────────────────────────────────────────────
# Three base variants (no threshold):
V_OLD="rbrics_old"               # method=rbrics_old, CreateMotifVocab plot path
V_RBRICS="rbrics"                # method=rbrics, no structural fallback, no BPE
V_RBRICS_SF="rbrics_with_struct_fallback"  # method=rbrics + structural fallback
V_ALL="all_fallback_bpe"         # method=all, fallback, BPE
# V_ALL_SHATTER="all_fallback_bpe_shatter"  # ablation; phase1d disabled (no phase5)

# Three filtered variants (threshold applied):
V_OLD_TH="rbrics_old_filter"
V_RBRICS_TH="rbrics_filter"
V_RBRICS_SF_TH="rbrics_with_struct_fallback_filter"
V_ALL_TH="all_fallback_bpe_filter"

# GT-relabelled out_dir suffix (from phase4): {base_variant}_relabelled
_gt_variant_name() { echo "${1}_relabelled"; }

# ── VOCAB_FOCUS — subset phase4/5 to selected fragmentation algorithms ────────
_vocab_focus_resolve_one() {
    local raw="${1// /}"
    case "$raw" in
        rbrics_old|old)              echo "$V_OLD" ;;
        rbrics)                      echo "$V_RBRICS" ;;
        rbrics_with_struct_fallback|struct_fallback|struct)
                                     echo "$V_RBRICS_SF" ;;
        all_fallback_bpe|all|v4)   echo "$V_ALL" ;;
        *)
            echo "  [warn] unknown VOCAB_FOCUS token: '$raw' (ignored)" >&2
            return 1
            ;;
    esac
}

_vocab_focus_base_variants() {
    if [ -z "$VOCAB_FOCUS" ]; then
        echo "$V_OLD $V_RBRICS $V_RBRICS_SF $V_ALL"
        return
    fi
    local resolved="" token v
    local IFS=','
    for token in $VOCAB_FOCUS; do
        v=$(_vocab_focus_resolve_one "$token") || continue
        case " $resolved " in
            *" $v "*) ;;
            *) resolved="$resolved $v" ;;
        esac
    done
    # shellcheck disable=SC2086
    set -- $resolved
    if [ $# -eq 0 ]; then
        echo "ERROR: VOCAB_FOCUS='$VOCAB_FOCUS' resolved to no variants" >&2
        exit 1
    fi
    echo "$resolved"
}

_vocab_focus_filtered_for() {
    case "$1" in
        "$V_OLD")      echo "$V_OLD_TH" ;;
        "$V_RBRICS")   echo "$V_RBRICS_TH" ;;
        "$V_RBRICS_SF") echo "$V_RBRICS_SF_TH" ;;
        "$V_ALL")      echo "$V_ALL_TH" ;;
        *)             echo "$1" ;;
    esac
}

_vocab_focus_filtered_variants() {
    local base filtered
    for base in $(_vocab_focus_base_variants); do
        filtered=$(_vocab_focus_filtered_for "$base")
        echo "$filtered"
    done
}

_vocab_focus_all_eval_variants() {
    _vocab_focus_base_variants
    _vocab_focus_filtered_variants
}

_baseline_weight_variant() {
    case "$1" in
        "$V_OLD_TH")       echo "$V_OLD" ;;
        "$V_RBRICS_TH")    echo "$V_RBRICS" ;;
        "$V_RBRICS_SF_TH") echo "$V_RBRICS_SF" ;;
        "$V_ALL_TH")       echo "$V_ALL" ;;
        *)                 echo "$1" ;;
    esac
}

_gt_split_cached() {
    local variant=$1 ds=$2 fold=$3 split=${4:-train}
    [ -f "$OUT_ROOT/gt_cache/$ds/fold${fold}/$variant/relabel1/${split}_with_gt.pt" ]
}

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

# Regression datasets skip rule mining and phase-4 synthetic GT only; phases 1–3
# still fragment, threshold, and plot node coverage.
# Synthetic GT (phase 4) only applies to GT_SUPPORTED_DATASETS CSV benchmarks.
_skip_synthetic_gt_dataset() {
    # Return 0 = apply GT; return 1 = skip (regression, mutag, OGB, etc.).
    PYTHONPATH="$PROJECT:${PYTHONPATH:-}" python3 -c "
from SharedModules.data.ground_truth import GT_SUPPORTED_DATASETS
import sys
sys.exit(0 if sys.argv[1] in GT_SUPPORTED_DATASETS else 1)
" "$1" 2>/dev/null || {
        case "$1" in Mutagenicity|Benzene|BBBP|hERG|Alkane_Carbonyl|Fluoride_Carbonyl)
            return 0 ;;
        esac
        return 1
    }
}

# Phase 1 writes three base variants per dataset (no threshold).
_phase1_variant_done() {
    local ds=$1 variant=$2
    [ -f "$VOCAB_ROOT/$ds/$variant/rules.json" ] && \
    [ -f "$VOCAB_ROOT/$ds/$variant/vocab_meta.json" ]
}

_phase1_complete() {
    local ds=$1
    _phase1_variant_done "$ds" "$V_OLD" && \
    _phase1_variant_done "$ds" "$V_RBRICS" && \
    _phase1_variant_done "$ds" "$V_ALL"
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

# Config slug matching run_experiments.py _cfg_slug (vanilla/baselines omit inj/ep).
_vanilla_cfg_slug() {
    local syn="${1:-real}" bb="${2:?backbone required}"
    local nrm="norm-${CONV_NORMALIZE}"
    [ "${ENCODER_NORM:-off}" = "on" ] && nrm="${nrm}+encLN"
    echo "bb-${bb}_enc-${NODE_ENCODER}_${nrm}_${syn}"
}

# Canonical run dirs (same layout as run_experiments.py config_tag + --final_out_dir).
_vanilla_run_dir() {
    local ds=$1 fold=$2 variant=$3 bb=$4 syn=${5:-real}
    echo "$OUT_ROOT/vanilla/${ds}/fold${fold}/${variant}/$(_vanilla_cfg_slug "$syn" "$bb")"
}

_baseline_run_dir() {
    local ds=$1 fold=$2 eval_variant=$3 bb=$4 syn=${5:-real}
    echo "$OUT_ROOT/baselines/${ds}/fold${fold}/${eval_variant}/$(_vanilla_cfg_slug "$syn" "$bb")"
}

# ── Helper: fragment one variant ──────────────────────────────────────────────
# Usage: run_frag <method> <fallback:0|1> <bpe:0|1> <out_variant> [shatter:0|1]
run_frag() {
    local method=$1 use_fallback=$2 use_bpe=$3 variant=$4 use_shatter=${5:-0}
    echo "  [$variant] method=$method fallback=$use_fallback bpe=$use_bpe shatter=$use_shatter"
    for ds in $DATASETS; do
        if [ "${FORCE_PHASE1:-0}" != "1" ] && _phase1_variant_done "$ds" "$variant"; then
            echo "  [skip] $ds / $variant — already exists"
            continue
        fi
        # Skip only when re-running the standard phase1 trio and all three exist.
        if [ "${FORCE_PHASE1:-0}" != "1" ] && _phase1_complete "$ds"; then
            case "$variant" in
                "$V_OLD"|"$V_RBRICS"|"$V_ALL")
                    echo "  [skip] $ds — phase1 complete ($V_OLD, $V_RBRICS, $V_ALL)"
                    continue
                    ;;
            esac
        fi
        ds_root="$(_dataset_data_root "$ds")"
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
                local out_dir="$(_vanilla_run_dir "$ds" "$eff_fold" "$variant" "$backbone")"
                python3 "$PROJECT/SharedModules/baselines/run_vanilla.py" \
                    --dataset      "$ds" --fold "$eff_fold" \
                    --backbone     "$backbone" --node_encoder "$enc" \
                    --epochs       "$EPOCHS" \
                    --data_root    "$ds_root" \
                    --vocab_root   "$VOCAB_ROOT" \
                    --vocab_variant "$variant" \
                    --conv_normalize "$CONV_NORMALIZE" \
                    --processed_root "$PROCESSED_ROOT" \
                    --out_dir      "$out_dir" \
                    --final_out_dir \
                    $( [ "$ENCODER_NORM" = "on" ] && echo "--apply_layer_norm" ) \
                    $(_mutag_train_flags "$ds" "$eff_fold") \
                    $WANDB_FLAGS
            done
        done
    done
}

run_mose() {
    local variant=$1 inj_args=$2
    # Nested out_dir (mose/<variant>/<ds>/fold<N>/<tag>/): trainers append ds/fold/tag.
    # Differs from vanilla/baselines canonical layout; regenerate_eval + collect handle both.
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
    local variant=$1 inj_args=${2:-$GSAT_INJ}
    for backbone in $BACKBONES; do
        echo "  [BaseGSAT] backbone=$backbone vocab=$variant inj=$inj_args learn_edge_att=$GSAT_LEARN_EDGE_ATT"
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
                    $inj_args \
                    $(_gsat_learn_edge_att_flag) \
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
    local variant=$1 inj_args=${2:-$MOTIFSAT_INJ}
    for backbone in $BACKBONES; do
        echo "  [MotifSAT readout] backbone=$backbone vocab=$variant inj=$inj_args"
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
                    $inj_args \
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
    local weight_variant
    weight_variant=$(_baseline_weight_variant "$eval_variant")
    for backbone in $BACKBONES; do
        echo "  [Baselines eval] backbone=$backbone vocab=$eval_variant  weight_vocab=$weight_variant"
        for ds in $DATASETS; do
            for fold in $FOLDS; do
                _skip_redundant_fold "$ds" "$fold" && continue
                local ds_root="$(_dataset_data_root "$ds")"
                local enc="$(_dataset_node_encoder "$ds")"
                local eff_fold="$fold"
                case "$ds" in mutag|ogbg-*) eff_fold=0 ;; esac
                local wdir="$(_vanilla_run_dir "$ds" "$eff_fold" "$weight_variant" "$backbone")"
                local out_dir="$(_baseline_run_dir "$ds" "$eff_fold" "$eval_variant" "$backbone")"
                python3 "$PROJECT/SharedModules/baselines/run_vanilla.py" \
                    --dataset      "$ds" --fold "$eff_fold" \
                    --backbone     "$backbone" --node_encoder "$enc" \
                    --epochs       0 \
                    --data_root    "$ds_root" \
                    --vocab_root   "$VOCAB_ROOT" \
                    --vocab_variant "$eval_variant" \
                    --conv_normalize "$CONV_NORMALIZE" \
                    --processed_root "$PROCESSED_ROOT" \
                    --load_weights_from "$wdir" \
                    --weight_vocab_variant "$weight_variant" \
                    --out_dir      "$out_dir" \
                    --final_out_dir \
                    $( [ "$ENCODER_NORM" = "on" ] && echo "--apply_layer_norm" ) \
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
        if _skip_synthetic_gt_dataset "$ds"; then
            echo "  [skip] $ds — not in GT_SUPPORTED_DATASETS"
            continue
        fi
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

run_mose_gt() {
    local variant=$1
    local gt_variant
    gt_variant=$(_gt_variant_name "$variant")
    for backbone in $BACKBONES; do
        echo "  [MOSE+GT] backbone=$backbone base=$variant → $gt_variant"
        for ds in $DATASETS_CSV; do
            if _skip_synthetic_gt_dataset "$ds"; then
                echo "  [skip] $ds — not in GT_SUPPORTED_DATASETS"
                continue
            fi
            for fold in $FOLDS; do
                if ! _gt_split_cached "$variant" "$ds" "$fold" train; then
                    echo "  [skip] $gt_variant $ds fold$fold — no gt_cache (run phase4)"
                    continue
                fi
                local enc="$(_dataset_node_encoder "$ds")"
                python3 "$PROJECT/MOSE-GNN/run.py" \
                    --dataset      "$ds" --fold "$fold" \
                    --backbone     "$backbone" --node_encoder "$enc" \
                    --w_feat --w_readout \
                    --use_gt --gt_cache "$OUT_ROOT/gt_cache" \
                    --epochs       "$EPOCHS" \
                    --data_root    "$DATA_ROOT" \
                    --vocab_root   "$VOCAB_ROOT" \
                    --vocab_variant "$variant" \
                    --conv_normalize "$CONV_NORMALIZE" \
                    --processed_root "$PROCESSED_ROOT" \
                    --out_dir      "$OUT_ROOT/mose/${gt_variant}" \
                    $(_mose_extra_flags) \
                    $WANDB_FLAGS
            done
        done
    done
}

run_motifsat_gt() {
    local variant=$1
    local gt_variant
    gt_variant=$(_gt_variant_name "$variant")
    for backbone in $BACKBONES; do
        echo "  [MotifSAT+GT] backbone=$backbone base=$variant → $gt_variant"
        for ds in $DATASETS_CSV; do
            if _skip_synthetic_gt_dataset "$ds"; then
                echo "  [skip] $ds — not in GT_SUPPORTED_DATASETS"
                continue
            fi
            for fold in $FOLDS; do
                if ! _gt_split_cached "$variant" "$ds" "$fold" train; then
                    echo "  [skip] $gt_variant $ds fold$fold — no gt_cache (run phase4)"
                    continue
                fi
                local enc="$(_dataset_node_encoder "$ds")"
                python3 "$PROJECT/MotifSAT/run.py" \
                    --dataset         "$ds" --fold "$fold" \
                    --backbone        "$backbone" --node_encoder "$enc" \
                    --motif_method    readout \
                    --noise           none \
                    --info_loss_level none \
                    --info_loss_coef  0.0 \
                    --w_feat --w_readout --w_message \
                    --use_gt --gt_cache "$OUT_ROOT/gt_cache" \
                    --epochs          "$EPOCHS" \
                    --data_root       "$DATA_ROOT" \
                    --vocab_root      "$VOCAB_ROOT" \
                    --vocab_variant   "$variant" \
                    --conv_normalize  "$CONV_NORMALIZE" \
                    --processed_root  "$PROCESSED_ROOT" \
                    --out_dir         "$OUT_ROOT/motifsat/${gt_variant}" \
                    $WANDB_FLAGS
            done
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

_check_special_exports() {
    for ds in $DATASETS_SPECIAL; do
        case "$ds" in
            mutag)
                [ -f "$MUTAG_DATA_ROOT/mutag_0.csv" ] || \
                    echo "  [warn] missing $MUTAG_DATA_ROOT/mutag_0.csv — run phase0 first"
                ;;
            ogbg-*)
                [ -f "$OGB_DATA_ROOT/${ds}_0.csv" ] || \
                    echo "  [warn] missing $OGB_DATA_ROOT/${ds}_0.csv — run phase0 first"
                ;;
        esac
    done
}

# =============================================================================
# PHASE 1 — Fragmentation, no threshold
#   Four variants: rbrics_old, rbrics, rbrics_with_struct_fallback, all_fallback_bpe
# =============================================================================
phase1() {
    _check_paths
    _check_special_exports
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " PHASE 1 — Fragmentation (no threshold, 4 variants)"
    echo "  Skips: datasets with phase1 trio done ($V_OLD, $V_RBRICS, $V_ALL);"
    echo "         individual variants already on disk (set FORCE_PHASE1=1 to redo)"
    echo "══════════════════════════════════════════════════════════"

    echo "1a. rbrics_old  (CreateMotifVocab plot path — BreakrBRICSBonds + ToSmiles)"
    run_frag rbrics_old 0 0 "$V_OLD"

    echo "1b. rbrics  (BreakrBRICSBonds + BRICS fallback + reBRICS)"
    run_frag rbrics 0 0 "$V_RBRICS"

    echo "1d. rbrics_with_struct_fallback  (rBRICS/BRICS fallback + reBRICS + structural fallback)"
    run_frag rbrics 1 0 "$V_RBRICS_SF"

    echo "1c. all_fallback_bpe  (full cascade, fallback, BPE)"
    run_frag all 1 1 "$V_ALL"

    # echo "1d. all_fallback_bpe_shatter  (full cascade + mild-shatter floor)"
    # run_frag all 1 1 "$V_ALL" 1   # → vocab dir all_fallback_bpe_shatter (no phase5 yet)

    echo ""
    echo "Phase 1 complete. Vocabularies in: $VOCAB_ROOT"
    echo "Variants: $V_OLD  $V_RBRICS  $V_RBRICS_SF  $V_ALL"
    echo "Next: bash run_experiments.sh phase2  (review coverage plots)"
}

# =============================================================================
# PHASE 2 — Coverage vs threshold sweep
#   All four base variants swept so you can compare curves side by side.
# =============================================================================
phase2() {
    _check_paths
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " PHASE 2 — Coverage vs threshold sweep (4 variants)"
    echo "══════════════════════════════════════════════════════════"

    local vocab_datasets="$DATASETS"

    for variant in "$V_OLD" "$V_RBRICS" "$V_RBRICS_SF" "$V_ALL"; do
        echo ""
        echo "  [combined / $variant]"
        python3 "$PROJECT/MotifBreakdown/coverage_vs_threshold.py" \
            --vocab_root "$VOCAB_ROOT" \
            --datasets   $vocab_datasets \
            --variant    "$variant" \
            --out_dir    "$OUT_ROOT/coverage_plots" \
            --combine_plot
    done

    echo ""
    echo "Phase 2 complete. Review plots in: $OUT_ROOT/coverage_plots"
    echo "  Per-dataset:  {dataset}_{variant}_coverage.png"
    echo "  Combined:     all_datasets_{variant}_coverage.png"
    echo "Then:  edit CHOSEN_THRESHOLD in MotifBreakdown/generate_vocab_rules.py"
    echo "       bash run_experiments.sh phase3"
}

# =============================================================================
# PHASE 3 — Thresholded vocabularies
#   All four filtered variants re-fragmented with threshold applied.
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
    run_frag_thresh rbrics_old 0 0 "$V_OLD_TH"

    echo "3b. rbrics_filter"
    run_frag_thresh rbrics 0 0 "$V_RBRICS_TH"

    echo "3d. rbrics_with_struct_fallback_filter"
    run_frag_thresh rbrics 1 0 "$V_RBRICS_SF_TH"

    echo "3c. all_fallback_bpe_filter"
    run_frag_thresh all 1 1 "$V_ALL_TH"

    echo ""
    echo "Phase 3 complete.  Vocabularies now available:"
    echo "  No threshold: $V_OLD  $V_RBRICS  $V_RBRICS_SF  $V_ALL"
    echo "  Filtered:     $V_OLD_TH  $V_RBRICS_TH  $V_RBRICS_SF_TH  $V_ALL_TH"
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
#   All four unfiltered base variants (respects VOCAB_FOCUS when set).
#   Result cached in $OUT_ROOT/gt_cache/{dataset}/fold{N}/{variant}/relabel1/
# =============================================================================
phase4() {
    [ -z "$RULE_INDEX" ] && \
        echo "ERROR: set RULE_INDEX first.  export RULE_INDEX=0" && exit 1

    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " PHASE 4 — Synthetic GT (rule=$RULE_INDEX, VOCAB_FOCUS=${VOCAB_FOCUS:-all four})"
    echo "══════════════════════════════════════════════════════════"

    local variant
    for variant in $(_vocab_focus_base_variants); do
        echo ""
        echo "  --- apply_gt: $variant ---"
        apply_gt "$variant" "$RULE_INDEX"
    done

    echo ""
    echo "Phase 4 complete.  GT cache: $OUT_ROOT/gt_cache"
    echo "Per-variant GT training dirs (phase5): {mose,motifsat}/{variant}_relabelled"
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
    echo " PHASE 5a — Vanilla GNN (VOCAB_FOCUS=${VOCAB_FOCUS:-all four})"
    echo "══════════════════════════════════════════════════════════"

    local variant
    for variant in $(_vocab_focus_base_variants); do
        run_vanilla "$variant"
    done

    echo "Vanilla training complete."
}

# =============================================================================
# PHASE 5b — MOSE-GNN
#   Per VOCAB_FOCUS: filtered + unfiltered base vocabs + GT relabelled
#   (when phase4 gt_cache exists). Default injection: 101 (--w_feat --w_readout).
# =============================================================================
phase5_mose() {
    _check_paths
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " PHASE 5b — MOSE-GNN (VOCAB_FOCUS=${VOCAB_FOCUS:-all four})"
    echo "══════════════════════════════════════════════════════════"

    local variant
    for variant in $(_vocab_focus_filtered_variants); do
        run_mose "$variant" "$MOSE_INJ"
    done
    for variant in $(_vocab_focus_base_variants); do
        run_mose "$variant" "$MOSE_INJ"
    done

    if [ -d "$OUT_ROOT/gt_cache" ]; then
        for variant in $(_vocab_focus_base_variants); do
            run_mose_gt "$variant"
        done
    else
        echo "  [skip] *_relabelled — run phase4 first"
    fi

    echo "MOSE training complete."
}

# =============================================================================
# PHASE 5c — Base GSAT
#   Node-level attention + message injection (default 010: --w_message).
#   learn_edge_att=False by default (node att scaled to edges via src×dst).
#   Set GSAT_LEARN_EDGE_ATT=1 for the legacy edge-attention MLP path.
# =============================================================================
phase5_gsat() {
    _check_paths
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " PHASE 5c — Base GSAT (VOCAB_FOCUS=${VOCAB_FOCUS:-all four})"
    echo "══════════════════════════════════════════════════════════"

    local variant
    for variant in $(_vocab_focus_base_variants); do
        run_gsat "$variant"
    done

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
    echo " PHASE 5d — Post-hoc baselines (VOCAB_FOCUS=${VOCAB_FOCUS:-all four})"
    echo "══════════════════════════════════════════════════════════"

    local eval_variant
    for eval_variant in $(_vocab_focus_all_eval_variants); do
        run_baselines "$eval_variant"
    done

    echo "Baseline evaluation complete."
}

# =============================================================================
# PHASE 5e — MotifSAT
#   Readout-level motif aggregation; default injection 111 (w_feat+w_message+w_readout).
#   Unfiltered base vocabs only (MotifSAT builds embeddings — filtering N/A).
# =============================================================================
phase5_motifsat() {
    _check_paths
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " PHASE 5e — MotifSAT readout (VOCAB_FOCUS=${VOCAB_FOCUS:-all four})"
    echo "══════════════════════════════════════════════════════════"

    local variant
    for variant in $(_vocab_focus_base_variants); do
        run_motifsat "$variant" "$MOTIFSAT_INJ"
    done

    if [ -d "$OUT_ROOT/gt_cache" ]; then
        for variant in $(_vocab_focus_base_variants); do
            run_motifsat_gt "$variant"
        done
    else
        echo "  [skip] *_relabelled — run phase4 first"
    fi

    echo "MotifSAT training complete."
}

# =============================================================================
# Post-hoc H0/H1/H2 multi-explanation (MOSE, MotifSAT, GSAT node-attention)
#   Run after phase5 training completes. Skips vanilla/baselines and
#   learn_edge_att GSAT runs automatically.
# =============================================================================
multi_explanation() {
    _check_paths
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " Post-hoc multi-explanation (H0/H1/H2)"
    echo "══════════════════════════════════════════════════════════"
    python3 "$PROJECT/analysis/run_multi_explanation.py" \
        --out_root   "$OUT_ROOT" \
        --data_root  "$DATA_ROOT" \
        --vocab_root "$VOCAB_ROOT"
}

# =============================================================================
# Post-hoc masked-node feature probe (MOSE / MotifSAT readout / GSAT node-att)
# =============================================================================
probe_masked_nodes() {
    _check_paths
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " Masked-node feature probe"
    echo "══════════════════════════════════════════════════════════"
    python3 "$PROJECT/analysis/probe_masked_nodes.py" \
        --out_root   "$OUT_ROOT" \
        --data_root  "$DATA_ROOT" \
        --vocab_root "$VOCAB_ROOT" \
        --save       masked_node_probe.csv
}

# =============================================================================
# Collect results (delegates to analysis/run_analysis.py collect)
# =============================================================================
collect_results() {
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " Collecting results (run_analysis.py collect)"
    echo "══════════════════════════════════════════════════════════"
    python3 "$PROJECT/analysis/run_analysis.py" collect \
        --out_root "$OUT_ROOT"
}

# Superseded inline collector (no config.json merge / axis normalization):
# collect_results() {
#     python3 - << PYEOF
# ... rglob summary.json only ...
# PYEOF
# }

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
    multi_explanation) multi_explanation ;;
    probe_masked_nodes) probe_masked_nodes ;;
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
        echo "  phase1            fragment all 4 variants (rbrics_old, rbrics, struct_fallback, all_fallback_bpe)"
        echo "  phase2            coverage vs threshold sweep (review, then edit CHOSEN_THRESHOLD)"
        echo "  phase3            threshold all 4 variants  (reads CHOSEN_THRESHOLD)"
        echo "  phase4            synthetic GT (all 4 base variants; VOCAB_FOCUS)"
        echo "  phase5_vanilla    vanilla GNN (VOCAB_FOCUS base variants)"
        echo "  phase5_mose       MOSE-GNN (filtered + base + GT per VOCAB_FOCUS)"
        echo "  phase5_gsat       base GSAT (VOCAB_FOCUS base variants)"
        echo "  phase5_baselines  post-hoc on vanilla (VOCAB_FOCUS eval variants)"
        echo "  phase5_motifsat   MotifSAT (VOCAB_FOCUS base + GT variants)"
        echo "  multi_explanation post-hoc H0/H1/H2 on MOSE/MotifSAT/GSAT"
        echo "  probe_masked_nodes post-hoc masked-node feature probe"
        echo "  collect           print results table"
        echo "  analyze           regenerate eval + tables + plots (single entry point)"
        echo ""
        echo "Required env (set in experiment_config.sh):"
        echo "  PROJECT  DATA_ROOT  VOCAB_ROOT  OUT_ROOT"
        echo "  RULE_INDEX (phase4)   Thresholds: edit CHOSEN_THRESHOLD in generate_vocab_rules.py"
        echo "  VOCAB_FOCUS (phase4/5) e.g. rbrics,all_fallback_bpe  (default: all four)"
        ;;
    *)
        echo "Unknown phase: $PHASE"; exit 1 ;;
esac