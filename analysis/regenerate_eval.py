#!/usr/bin/env python3
"""regenerate_eval.py — re-run eval-only on existing checkpoints.

Walks an experiment output tree, finds every run with a best_model.pt + a
summary.json, and re-invokes the appropriate run.py with --eval_only so the new
explainability metrics (score↔impact correlation, discriminativeness,
score_vs_impact.csv, motif-score stats; plus baseline mean/max) are regenerated
WITHOUT retraining.

It reconstructs each run's CLI flags from its summary.json (dataset, fold,
backbone, vocab_variant, motif_method, w_feat/w_message/w_readout, node_encoder).
Vanilla/baseline runs use the existing --epochs 0 --load_weights_from path.

Usage
-----
    python analysis/regenerate_eval.py --out_root results \
        --data_root $DATA_ROOT --vocab_root $VOCAB_ROOT \
        [--processed_root $PROCESSED_ROOT] [--families mose motifsat vanilla] \
        [--dry_run]

IMPORTANT: pair each checkpoint with the vocab it was TRAINED on. If you
regenerated vocabularies after training (e.g. the balanced-ranking change), the
masks may differ and impact/discriminativeness would be computed against a
different vocab than the model saw. Point --vocab_root at the matching vocab.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _family(meta: dict) -> str:
    mt = (meta.get('model_type') or '').lower()
    if 'mose' in mt:
        return 'mose'
    if 'motifsat' in mt or 'gsat' in mt:
        return 'motifsat'
    if 'vanilla' in mt:
        return 'vanilla'
    return 'unknown'


def _flag(b):
    return bool(b)


def build_cmd(meta: dict, run_dir: Path, args) -> list[str] | None:
    fam = _family(meta)
    ds = meta.get('dataset'); fold = meta.get('fold', 0)
    bb = meta.get('backbone'); var = meta.get('vocab_variant')
    enc = meta.get('node_encoder', 'onehot')
    if not (ds and bb and var):
        return None
    # Canonical runs (written by run_experiments.py) carry a config.json and use
    # the single-level FINAL layout: regenerate must write back into run_dir
    # itself (--out_dir run_dir --final_out_dir) instead of re-deriving the
    # nested path under out_root.
    canonical = (run_dir / 'config.json').exists()
    out_dir = str(run_dir) if canonical else args.out_root
    common = [
        '--dataset', str(ds), '--fold', str(fold), '--backbone', str(bb),
        '--node_encoder', str(enc),
        '--data_root', args.data_root, '--vocab_root', args.vocab_root,
        '--vocab_variant', str(var), '--out_dir', out_dir,
    ]
    if canonical:
        common += ['--final_out_dir']
    if args.processed_root:
        common += ['--processed_root', f'{args.processed_root}']

    if fam == 'mose':
        cmd = [sys.executable, str(REPO / 'MOSE-GNN' / 'run.py')] + common
        for f, name in ((meta.get('w_feat'), '--w_feat'),
                        (meta.get('w_message'), '--w_message'),
                        (meta.get('w_readout'), '--w_readout')):
            if _flag(f):
                cmd.append(name)
        cmd += ['--eval_only', '--load_weights_from', str(run_dir)]
        return cmd

    if fam == 'motifsat':
        cmd = [sys.executable, str(REPO / 'MotifSAT' / 'run.py')] + common
        cmd += ['--motif_method', str(meta.get('motif_method', 'readout')),
                '--noise', str(meta.get('noise', 'none')),
                '--info_loss_level', str(meta.get('info_loss_level', 'none')),
                '--info_loss_coef', str(meta.get('info_loss_coef', 0.0))]
        for f, name in ((meta.get('w_feat'), '--w_feat'),
                        (meta.get('w_message'), '--w_message'),
                        (meta.get('w_readout'), '--w_readout'),
                        (meta.get('learn_edge_att'), '--learn_edge_att')):
            if _flag(f):
                cmd.append(name)
        cmd += ['--eval_only', '--load_weights_from', str(run_dir)]
        return cmd

    if fam == 'vanilla':
        # Vanilla already supports eval-only via --epochs 0 + load_weights_from.
        cmd = [sys.executable, str(REPO / 'SharedModules' / 'baselines' / 'run_vanilla.py')] + common
        cmd += ['--epochs', '0', '--load_weights_from', str(run_dir)]
        return cmd

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--vocab_root', required=True)
    ap.add_argument('--processed_root', default=None)
    ap.add_argument('--families', nargs='*',
                    default=['mose', 'motifsat', 'vanilla'])
    ap.add_argument('--dry_run', action='store_true',
                    help='Print the commands without running them.')
    args = ap.parse_args()

    out_root = Path(args.out_root)
    runs = sorted({p.parent for p in out_root.rglob('best_model.pt')})
    print(f'Found {len(runs)} checkpoint(s) under {out_root}\n')

    ran = skipped = failed = 0
    for run_dir in runs:
        sj = run_dir / 'summary.json'
        if not sj.exists():
            skipped += 1
            continue
        try:
            meta = json.load(open(sj))
        except Exception:
            skipped += 1
            continue
        if _family(meta) not in args.families:
            skipped += 1
            continue
        cmd = build_cmd(meta, run_dir, args)
        if cmd is None:
            skipped += 1
            continue
        print('»', ' '.join(cmd))
        if args.dry_run:
            ran += 1
            continue
        try:
            subprocess.run(cmd, check=True, env={**__import__('os').environ,
                                                 'PYTHONPATH': str(REPO)})
            ran += 1
        except subprocess.CalledProcessError as e:
            print(f'  [fail] {run_dir}: {e}')
            failed += 1

    print(f'\nDone. regenerated={ran}  skipped={skipped}  failed={failed}')
    if not args.dry_run and ran:
        print('Now re-run: bash run_experiments.sh collect   (to refresh all_results.csv)')


if __name__ == '__main__':
    main()
