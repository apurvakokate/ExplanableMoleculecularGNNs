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
(gnnexplainer mage_v2) and a CPU group (pgexplainer motif_occlusion).
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
from analysis.posthoc_complete import methods_complete, method_state
from SharedModules.evaluation.split_eval import evaluate_posthoc_all_splits

FAMILIES = ('vanilla', 'baselines')
# the 4 post-hoc explainers, passed EXPLICITLY to evaluate_posthoc_all_splits when
# --methods is unset (kept in sync with that function's signature default).
DEFAULT_POSTHOC_METHODS = ('gnnexplainer', 'motif_occlusion', 'mage_v2', 'pgexplainer')


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
    # Pass method_names EXPLICITLY (not via **kw) so a mismatched key can't be
    # silently dropped. When --methods is unset, pass the full default set here
    # rather than relying on the callee's signature default.
    method_names = (tuple(args.methods) if args.methods
                    else DEFAULT_POSTHOC_METHODS)
    evaluate_posthoc_all_splits(
        model, vocab, loaders, split_lists, gt, device, task_type,
        denorm_of(dmeta, task_type), out_dir, args.max_motifs_eval,
        dataset=meta['dataset'], batch_size=args.batch_size,
        method_names=method_names, unk_mode=args.unk_mode)


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
    ap.add_argument('--unk_mode', default=None, choices=['zero', 'half', 'exclude'],
                    help='Filtered-vocab UNK-atom handling for the node-direct GT-ROC '
                         '(default None = full/unfiltered). zero/half fill UNK atoms; '
                         'exclude drops them (instance/global only). Folds in the old '
                         'gtroc_filtered_{v1,unk05,exclunk} trees — use a distinct '
                         '--dest_root per mode.')
    ap.add_argument('--dataset', nargs='*', default=None)
    ap.add_argument('--methods', nargs='*', default=None,
                    help='post-hoc subset (default all 4). e.g. GPU group '
                         '"gnnexplainer mage_v2" / CPU group "pgexplainer motif_occlusion".')
    ap.add_argument('--mage_only', action='store_true',
                    help='run ONLY MAGE (fit + score mage_v2; skip GNNExplainer / '
                         'PGExplainer / Motif-Occlusion). Reuses the vanilla checkpoint; '
                         'the per-split agnostic-impact cache still computes (it is the '
                         'model-side y-axis for Pearson, not an explainer). Shorthand for '
                         '--methods mage_v2.')
    ap.add_argument('--shard', default=None, help='i/N: process runs[i::N]')
    ap.add_argument('--only', default=None,
                    help='absolute path of ONE run directory to process; every other '
                         'run under --out_root is skipped. Lets a queue worker claim a '
                         'single cell (so a slow PGExplainer unit fits inside the wall '
                         'limit) without depending on iter_runs ordering the way '
                         '--shard does.')
    ap.add_argument('--families', nargs='*', default=None,
                    help="model families to include (default: baselines ONLY). This is "
                         "the post-hoc/baselines re-eval; vanilla is excluded unless "
                         "explicitly requested here.")
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--allow_filter_refit', action='store_true',
                    help='DANGER: permit running baselines on a *_filter vocab, which '
                         'RE-FITS MAGE/PGExplainer per vocab (unseeded, stochastic) and '
                         'produces a re-fit mage_attention.pt / pgexplainer_mlp.pt that is '
                         'WRONG for the paper. The filtered baseline column must instead '
                         'come from analysis/evaluate.py with --unk exclude (drops UNK '
                         'motifs/atoms from the FULL run). Off by default; runs are skipped.')
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--skip_done', action='store_true', default=True,
                    help='skip runs whose dest summary_splits.json is already non-empty '
                         '(resumable; makes it safe to add workers for rebalancing)')
    ap.add_argument('--no_skip_done', dest='skip_done', action='store_false')
    args = ap.parse_args()
    if args.mage_only:
        args.methods = ['mage_v2']   # fit + score MAGE only

    device = torch.device(
        'cuda' if (args.device == 'cuda'
                   or (args.device == 'auto' and torch.cuda.is_available()))
        else 'cpu')
    fams = set(args.families) if args.families else {'baselines'}
    if not fams <= set(FAMILIES):
        raise SystemExit(f'--families must be subset of {FAMILIES}, got {sorted(fams)}')
    print(f'device: {device}  (post-hoc driver, families={sorted(fams)})')

    # Resolve the method set ONCE, here, so the per-method skip below and process()
    # below agree on exactly which methods this invocation is responsible for.
    method_names = tuple(args.methods) if args.methods else DEFAULT_POSTHOC_METHODS
    print(f'methods: {list(method_names)}')

    # --only: address exactly one run directory by absolute path. Used by the queue
    # worker so a task is one cell, which keeps a PGExplainer unit inside the wall
    # limit and makes the claim boundary exact. Resolved to a canonical path so a
    # trailing slash or symlink cannot silently miss.
    only = Path(args.only).resolve() if args.only else None
    if only is not None and not only.is_dir():
        raise SystemExit(f'--only: not a directory: {only}')

    ok = failed = skipped_only = 0
    for run_dir, meta, fam in iter_runs(args.out_root, fams,
                                        args.dataset, args.shard):
        if only is not None and Path(run_dir).resolve() != only:
            skipped_only += 1
            continue
        if args.limit and (ok + failed) >= args.limit:
            break
        tag = f"{meta.get('dataset')}/{fam}/{meta.get('backbone')}/{meta.get('vocab_variant')}/fold{meta.get('fold')}"
        # GUARDRAIL: this is the FULL-vocab post-hoc PRODUCER — never re-fit a baseline
        # explainer on a *_filter vocab. The filtered view is analysis/evaluate.py with
        # --unk exclude (drops UNK/below-threshold motifs & atoms from this FULL run),
        # NOT a re-fit here (which re-fits MAGE/PGExplainer, unseeded stochastic => wrong).
        if (fam == 'baselines' and str(meta.get('vocab_variant', '')).endswith('_filter')
                and not args.allow_filter_refit):
            print(f'  [skip:filter-refit] {tag} -> filtered view is analysis/evaluate.py '
                  f'--unk exclude on the FULL run; re-fit here only with --allow_filter_refit')
            continue
        # Skip only what is ACTUALLY done, PER METHOD.
        #
        # This used to key on `summary_splits.json` existing, which is written once per
        # run dir as a whole-directory completion marker. That made the skip blind to
        # WHICH methods are inside: once any pass stamped a directory, every later pass
        # skipped it wholesale. Measured consequence: GNNExplainer and PGExplainer ended
        # up covering nearly disjoint cell sets, and a corrected MAGE pass would have
        # skipped all 1,337 directories GNNExplainer had already touched.
        #
        # methods_complete() re-derives completion from the artifacts themselves — the
        # 11 output files non-empty, the fitted checkpoint present, non-constant
        # importances, and non-degenerate per-atom attributions. Same function the
        # runner's post-flight assertion and the tracker use, so they cannot disagree.
        _dest = (Path(args.dest_root) / run_dir.relative_to(args.out_root)
                 if args.dest_root else run_dir)
        if args.skip_done and methods_complete(_dest, method_names):
            continue
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
