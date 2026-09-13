#!/usr/bin/env bash
# FragNet POC orchestrator — Benzene_Verified_GT, folds 0 & 1. Run ON THE HPC.
# Three phases: `setup` (once), `stage_a` (FragNet env), `stage_b` (l2xgnn env).
# This script RUNS NOTHING by itself beyond what you invoke; it documents the exact flow.
set -euo pipefail

# ── fixed coordinates (verified) ───────────────────────────────────────────────
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"                         # ExplanableMoleculecularGNNs
FRAGNET_REPO="https://github.com/pnnl/FragNet.git"
FRAGNET_PIN="master"                                  # TODO: pin to a specific commit SHA
VENDOR="$HERE/vendor/FragNet"
WORK="$HERE/work"                                     # neutral handoff + ft checkpoints

DATASET="Benzene_Verified_GT"
FOLDS="${FOLDS:-0 1}"
VOCAB="rbrics"                                        # comparison vocab (full); filtered = rbrics_filter
DATA_ROOT="/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS"
# BASE processed root — build_gt_loaders appends the variant (rbrics) internally.
PROC_ROOT="/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/processed_final_v2"
# Vocab root (load_vocab reads <VOCAB_ROOT>/<dataset>/<variant>/...); same generation as final_v2.
VOCAB_ROOT="/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/vocab_final_v2"
# POC writes to a SCRATCH tree — NEVER the authoritative mose_replication_v2 during validation.
# Production target (only AFTER the POC passes + we re-run to write there):
#   mose_replication_v2/artifacts/source/${DATASET}/fragnet/...
POC_SCRATCH="${POC_SCRATCH:-/nfs/stak/users/kokatea/hpc-share/ChemIntuit/Claude+Cursor/_fragnet_poc_scratch}"
DEST_ROOT="${POC_SCRATCH}/artifacts/source/${DATASET}/fragnet"
PT_CKPT="$VENDOR/fragnet/exps/pt/unimol_exp1s4/pt.pt"

phase="${1:-help}"

case "$phase" in
  setup)
    mkdir -p "$HERE/vendor" "$WORK"
    [ -d "$VENDOR" ] || git clone "$FRAGNET_REPO" "$VENDOR"
    ( cd "$VENDOR" && git checkout "$FRAGNET_PIN" )
    test -f "$PT_CKPT" || { echo "MISSING pretrained weights: $PT_CKPT" >&2; exit 1; }
    echo "[setup] FragNet vendored; pretrained pt.pt present."
    echo "[setup] Create the env:  conda env create -f $HERE/environment.yml"
    echo "[setup] Then:            conda activate fragnet && pip install -e $VENDOR"
    ;;

  stage_a)   # FragNet env. Produces work/<fold>/fragnet_neutral_<split>.json
    : "${CONDA_DEFAULT_ENV:?activate the 'fragnet' env first}"
    for f in $FOLDS; do
      echo "=== Stage A fold $f ==="
      test -f "$WORK/$f/graph_context.json" || { echo "run '$0 dump_context' (l2xgnn) first" >&2; exit 1; }
      python "$HERE/stage_a_fragnet_env/prep_data.py" \
          --graph_context "$WORK/$f/graph_context.json" \
          --out_dir "$WORK/$f" --vendor "$VENDOR"
      # batch_size default raised 16 -> 256: the bs512 test proved a large batch trains Benzene to
      # val AUC 1.0 in ~15 min (122 epochs) vs ~31 s/epoch at batch 16 (9,600 train / 16 = 600
      # batches/epoch of tiny-kernel + single-threaded-collate overhead). 256 fits comfortably on an
      # 8 GB GPU; override with BATCH_SIZE for a larger card (512 OOMs the old M60).
      python "$HERE/stage_a_fragnet_env/finetune_fragnet.py" \
          --work "$WORK/$f" --pt_ckpt "$PT_CKPT" --vendor "$VENDOR" \
          --task clf --n_classes 1 --batch_size "${BATCH_SIZE:-256}"
      python "$HERE/stage_a_fragnet_env/export_attention.py" \
          --work "$WORK/$f" --vendor "$VENDOR" \
          --graph_context "$WORK/$f/graph_context.json" --impact own \
          --out "$WORK/$f/fragnet_neutral.json"
    done
    ;;

  dump_context)  # l2xgnn env, BEFORE stage_a export: dump node order+nodes_to_motifs+node_label
    for f in $FOLDS; do
      python "$HERE/stage_b_adapter/align_and_aggregate.py" dump_context \
          --dataset "$DATASET" --fold "$f" --vocab "$VOCAB" --regime source \
          --data_root "$DATA_ROOT" --processed_root "$PROC_ROOT" --vocab_root "$VOCAB_ROOT" \
          --out "$WORK/$f/graph_context.json"
    done
    ;;

  stage_b)   # l2xgnn env. Consumes work/<fold>/fragnet_neutral.json -> dest_root artifacts
    : "${CONDA_DEFAULT_ENV:?activate the 'l2xgnn' env first}"
    for f in $FOLDS; do
      for unk in include exclude; do
        echo "=== Stage B fold $f unk=$unk ==="
        python "$HERE/stage_b_adapter/emit_artifacts.py" \
            --dataset "$DATASET" --fold "$f" --vocab "$VOCAB" --unk "$unk" \
            --data_root "$DATA_ROOT" --processed_root "$PROC_ROOT" --vocab_root "$VOCAB_ROOT" \
            --neutral "$WORK/$f/fragnet_neutral.json" \
            --dest_root "$DEST_ROOT"
      done
    done
    echo "[stage_b] POC artifacts -> $DEST_ROOT  (SCRATCH — validate, then 'clean')"
    ;;

  clean)     # remove the POC scratch tree + intermediates once validated
    echo "[clean] removing POC scratch: $POC_SCRATCH  and work: $WORK"
    rm -rf "$POC_SCRATCH" "$WORK"
    ;;

  *)
    echo "usage: $0 {setup|dump_context|stage_a|stage_b|clean}" >&2
    echo "order: setup -> dump_context (l2xgnn) -> stage_a (fragnet) -> stage_b (l2xgnn) -> clean" >&2
    echo "POC artifacts go to a SCRATCH tree ($POC_SCRATCH), never mose_replication_v2." >&2
    ;;
esac
