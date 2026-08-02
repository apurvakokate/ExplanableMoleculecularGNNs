#!/usr/bin/env python3
"""eval_driver_posthoc.py — per-split re-eval driver for the POST-HOC families
(vanilla / baselines). This is the LONG path: it re-fits the explainers on train
and scores every split.

For each finished VanillaGNN checkpoint it reloads the model, builds the
train/valid/test GT loaders, and calls evaluate_posthoc_all_splits:
  GNNExplainer + Motif-Occlusion + MAGE  → GPU-batched via --batch_size
  PGExplainer                            → sequential / CPU-bound regardless

Device is chosen with --device (auto|cpu|cuda) so a scheduler can route these
(slow) runs to GPU jobs; --methods can split a run into a GPU group
(gnnexplainer mage_official) and a CPU group (pgexplainer motif_occlusion).
Writes into --dest_root (mirrors the run tree; source stays read-only).

  python analysis/eval_driver_posthoc.py --out_root final_v2 --dest_root final_v3 \
      --data_root <DATA> --vocab_root <VOCAB> [--processed_root <PROC>] \
      --device cuda --batch_size 256 [--dataset BBBP] [--methods ...] \
      [--shard i/N] [--limit N] [--dry_run]
"""
import argparse
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.eval_driver_common import (
    build_gt_loaders, split_lists_and_gt, denorm_of, iter_runs, out_dir_for)
from SharedModules.evaluation.split_eval import evaluate_posthoc_all_splits

FAMILIES = ('vanilla', 'baselines')


def load_vanilla_model(run_dir, meta, dmeta, device):
    """Rebuild VanillaGNN from summary meta + dataset dmeta and load best_model.pt
    (mirrors run_vanilla.py's construction so the state_dict loads cleanly)."""
    from SharedModules.baselines.vanilla_gnn import VanillaGNN
    from SharedModules.data.loader import NUM_CLASSES
    model = VanillaGNN(
        x_dim=dmeta.x_dim, hidden_dim=int(meta.get('hidden_dim', 64)),
        num_layers=int(meta.get('num_layers', 3)), backbone=meta['backbone'],
        node_encoder=meta.get('node_encoder', 'onehot'),
        apply_layer_norm=bool(meta.get('apply_layer_norm', False)),
        dropout=float(meta.get('dropout', 0.0) or 0.0),
        conv_normalize=meta.get('conv_normalize', 'none'),
        graph_pool=meta.get('graph_pool', 'add'),
        gin_inner_bn=bool(meta.get('gin_inner_bn', True)),
        num_classes=NUM_CLASSES.get(meta['dataset'], 1),
        deg=getattr(dmeta, 'deg', None))
    # Match run_vanilla / train_vanilla_gnn(epochs=0) EXACTLY: load the raw
    # state_dict with strict=True (the default) so any construction mismatch
    # (wrong pool / gin_inner_bn / encoder / PNA deg) fails LOUD here, instead of
    # strict=False silently dropping keys and loading a partial, wrong model.
    model.load_state_dict(
        torch.load(run_dir / 'best_model.pt', map_location='cpu', weights_only=False))
    return model.to(device).eval()


def process(run_dir, meta, args, device):
    loaders, vocab, dmeta, task_type = build_gt_loaders(
        meta, args.data_root, args.vocab_root, args.processed_root, args.loader_batch_size)
    split_lists, gt = split_lists_and_gt(loaders, meta)
    model = load_vanilla_model(run_dir, meta, dmeta, device)
    out_dir = out_dir_for(run_dir, args.out_root, args.dest_root)
    kw = {'method_names': tuple(args.methods)} if args.methods else {}
    evaluate_posthoc_all_splits(
        model, vocab, loaders, split_lists, gt, device, task_type,
        denorm_of(dmeta, task_type), out_dir, args.max_motifs_eval,
        dataset=meta['dataset'], batch_size=args.batch_size, **kw)


def main():
    ap = argparse.ArgumentParser(description='Post-hoc per-split re-eval driver')
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--vocab_root', required=True)
    ap.add_argument('--processed_root', default=None)
    ap.add_argument('--dest_root', default=None)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--batch_size', type=int, default=256,
                    help='GPU batch for GNNExplainer / Motif-Occlusion / MAGE embedding')
    ap.add_argument('--loader_batch_size', type=int, default=128)
    ap.add_argument('--max_motifs_eval', type=int, default=None)
    ap.add_argument('--dataset', nargs='*', default=None)
    ap.add_argument('--methods', nargs='*', default=None,
                    help='post-hoc subset (default all 4). e.g. GPU group '
                         '"gnnexplainer mage_official" / CPU group "pgexplainer motif_occlusion".')
    ap.add_argument('--shard', default=None, help='i/N: process runs[i::N]')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    device = torch.device(
        'cuda' if (args.device == 'cuda'
                   or (args.device == 'auto' and torch.cuda.is_available()))
        else 'cpu')
    print(f'device: {device}  (post-hoc driver, families={FAMILIES})')

    ok = failed = 0
    for run_dir, meta, fam in iter_runs(args.out_root, set(FAMILIES),
                                        args.dataset, args.shard):
        if args.limit and (ok + failed) >= args.limit:
            break
        tag = f"{meta.get('dataset')}/{fam}/{meta.get('backbone')}/{meta.get('vocab_variant')}/fold{meta.get('fold')}"
        if args.dry_run:
            print(f'  [dry] {tag}')
            ok += 1
            continue
        try:
            process(run_dir, meta, args, device)
            print(f'  [ok] {tag}')
            ok += 1
        except Exception as e:
            print(f'  [FAIL] {tag}: {type(e).__name__}: {e}')
            failed += 1
    print(f'\nDONE (post-hoc): processed={ok}  failed={failed}')


if __name__ == '__main__':
    main()
