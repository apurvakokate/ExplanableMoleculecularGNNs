#!/usr/bin/env python3
"""regenerate_eval.py — re-run eval-only on existing checkpoints.

Walks an experiment output tree, finds every run with a best_model.pt + a
summary.json, and re-invokes the appropriate run.py with --eval_only so the new
explainability metrics (score↔impact correlation, discriminativeness,
score_vs_impact.csv, motif-score stats; plus baseline mean/max) are regenerated
WITHOUT retraining.

It reconstructs each run's CLI flags from its summary.json (dataset, fold,
backbone, vocab_variant, motif_method, w_feat/w_message/w_readout, node_encoder).
Vanilla/baseline dirs (when included via ``--families vanilla baselines``) reload
the GNN checkpoint with ``--epochs 0`` but still **re-fit post-hoc explainers**
(PGExplainer/GNNExplainer/MAGE). They do NOT create missing ``baselines/`` runs —
use ``bash run_experiments.sh phase5_baselines`` for that.

Default ``--families``: ante-hoc only (mose, motifsat, gsat) with true ``--eval_only``.

Usage
-----
    python analysis/regenerate_eval.py --out_root results \
        --data_root $DATA_ROOT --vocab_root $VOCAB_ROOT \
        [--processed_root $PROCESSED_ROOT] \
        [--families mose motifsat gsat] \
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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.aggregate_experiments import (
    resolve_family, dataset_allowed, ARCHIVE_PREFIXES)
from SharedModules.data.dataset_routing import (
    base_from_stored_processed_root,
    mutag_artifact_paths,
    resolve_data_root,
)


def _flag(b):
    return bool(b)


def _meta_float(meta: dict, key: str, default: float = 0.0) -> float:
    val = meta.get(key)
    if val is None:
        return default
    return float(val)


def _family_allowed(fam: str, allowed: set[str]) -> bool:
    if fam in allowed:
        return True
    if fam == 'gsat' and 'motifsat' in allowed:
        return True
    return False


def _append_hparams(cmd: list[str], meta: dict, *, data_root: str | None = None) -> None:
    """Forward training hyperparameters stored in summary.json when present."""
    if meta.get('conv_normalize') not in (None, ''):
        cmd += ['--conv_normalize', str(meta['conv_normalize'])]
    if meta.get('num_layers') is not None:
        cmd += ['--num_layers', str(meta['num_layers'])]
    if meta.get('hidden_dim') is not None:
        cmd += ['--hidden_dim', str(meta['hidden_dim'])]
    if meta.get('gnn_lr') is not None:
        cmd += ['--gnn_lr', str(meta['gnn_lr'])]
    if meta.get('explainer_lr') is not None:
        cmd += ['--explainer_lr', str(meta['explainer_lr'])]
    if meta.get('gin_inner_bn') is False:
        cmd += ['--no_gin_inner_bn']
    if _flag(meta.get('apply_layer_norm')):
        cmd += ['--apply_layer_norm']
    _append_mutag_flags(cmd, meta, data_root=data_root)


def _append_mutag_flags(cmd: list[str], meta: dict, *, data_root: str | None = None) -> None:
    """Forward mutag artifact paths from summary or reconstruct from data_root."""
    if meta.get('dataset') != 'mutag':
        return
    dr = data_root or meta.get('data_root', '')
    paths = mutag_artifact_paths(
        dr,
        int(meta.get('fold', 0)),
        index_maps_path=meta.get('mutag_index_maps_path'),
        smiles_csv_path=meta.get('mutag_smiles_csv_path'),
        splits_path=meta.get('mutag_splits_path'),
    )
    for key, flag in (
        ('mutag_index_maps_path', '--mutag_index_maps_path'),
        ('mutag_smiles_csv_path', '--mutag_smiles_csv_path'),
        ('mutag_splits_path', '--mutag_splits_path'),
    ):
        val = meta.get(key) or paths.get(key)
        if val:
            cmd += [flag, str(val)]
    if meta.get('mutag_seed') is not None:
        cmd += ['--mutag_seed', str(meta['mutag_seed'])]


def _append_gt_flags(cmd: list[str], meta: dict) -> None:
    """Replay synthetic-GT training/eval when recorded in summary.json."""
    if _flag(meta.get('use_gt')) and meta.get('gt_cache'):
        cmd += ['--use_gt', '--gt_cache', str(meta['gt_cache'])]


def _resolve_run_data_root(meta: dict, args) -> str:
    """Prefer data_root recorded at training time; fall back to CLI/env routing."""
    if meta.get('data_root'):
        return str(meta['data_root'])
    ds = str(meta.get('dataset', ''))
    return resolve_data_root(
        ds,
        str(args.data_root),
        mutag_data_root=getattr(args, 'mutag_data_root', None),
        ogb_data_root=getattr(args, 'ogb_data_root', None),
    )


def _processed_root(meta: dict, args) -> str | None:
    """Base processed_root for trainer CLI (trainers append vocab variant)."""
    if meta.get('processed_root'):
        return base_from_stored_processed_root(
            str(meta['processed_root']), meta.get('vocab_variant'))
    if not args.processed_root:
        return None
    return str(args.processed_root)


def build_cmd(meta: dict, run_dir: Path, args) -> list[str] | None:
    try:
        exp_dir = str(run_dir.relative_to(Path(args.out_root)))
    except ValueError:
        exp_dir = str(run_dir)
    fam = resolve_family(meta, exp_dir)
    ds = meta.get('dataset')
    fold = meta.get('fold', 0)
    bb = meta.get('backbone')
    var = meta.get('vocab_variant')
    enc = meta.get('node_encoder', 'onehot')
    if not (ds and bb and var):
        return None

    data_root = _resolve_run_data_root(meta, args)

    # Always write back into the checkpoint directory (canonical or shell-nested layout).
    common = [
        '--dataset', str(ds), '--fold', str(fold), '--backbone', str(bb),
        '--node_encoder', str(enc),
        '--data_root', data_root,
        '--vocab_root', args.vocab_root,
        '--vocab_variant', str(var), '--out_dir', str(run_dir),
        '--final_out_dir',
    ]
    proc = _processed_root(meta, args)
    if proc:
        common += ['--processed_root', proc]
    _append_hparams(common, meta, data_root=data_root)
    _append_gt_flags(common, meta)

    train_fam = fam
    if fam in ('gsat', 'motifsat'):
        train_fam = 'motifsat'
    elif fam in ('vanilla', 'baselines'):
        train_fam = 'vanilla'

    if train_fam == 'mose':
        cmd = [sys.executable, str(REPO / 'MOSE-GNN' / 'run.py')] + common
        if meta.get('unk_mode') not in (None, ''):
            cmd += ['--unk_mode', str(meta['unk_mode'])]
        for f, name in ((meta.get('w_feat'), '--w_feat'),
                        (meta.get('w_message'), '--w_message'),
                        (meta.get('w_readout'), '--w_readout')):
            if _flag(f):
                cmd.append(name)
        # multi-explanation is post-hoc (analysis/run_multi_explanation.py)
        cmd += ['--eval_only', '--load_weights_from', str(run_dir)]
        return cmd

    if train_fam == 'motifsat':
        cmd = [sys.executable, str(REPO / 'MotifSAT' / 'run.py')] + common
        cmd += ['--motif_method', str(meta.get('motif_method', 'readout')),
                '--noise', str(meta.get('noise', 'none')),
                '--info_loss_level', str(meta.get('info_loss_level', 'none')),
                '--info_loss_coef', str(_meta_float(meta, 'info_loss_coef', 0.0))]
        for f, name in ((meta.get('w_feat'), '--w_feat'),
                        (meta.get('w_message'), '--w_message'),
                        (meta.get('w_readout'), '--w_readout'),
                        (meta.get('learn_edge_att'), '--learn_edge_att')):
            if _flag(f):
                cmd.append(name)
        cmd += ['--eval_only', '--load_weights_from', str(run_dir)]
        return cmd

    if train_fam == 'vanilla':
        cmd = [sys.executable, str(REPO / 'SharedModules' / 'baselines' / 'run_vanilla.py')] + common
        wvv = meta.get('weight_vocab_variant')
        if wvv:
            cmd += ['--weight_vocab_variant', str(wvv)]
        cmd += ['--epochs', '0', '--load_weights_from', str(run_dir)]
        return cmd

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--mutag_data_root', default=None,
                    help='mutag TUDataset root (defaults to $MUTAG_DATA_ROOT)')
    ap.add_argument('--ogb_data_root', default=None,
                    help='OGB cache root (defaults to $OGB_DATA_ROOT)')
    ap.add_argument('--vocab_root', required=True)
    ap.add_argument('--processed_root', default=None)
    ap.add_argument('--families', nargs='*',
                    default=['mose', 'motifsat', 'gsat'],
                    help='Run dirs to re-evaluate (default: ante-hoc only). '
                         'Add vanilla baselines to re-fit GNNExplainer/PGExplainer/MAGE '
                         'on existing vanilla/baselines checkpoints — does NOT create '
                         'missing baselines/ runs (use phase5_baselines for that).')
    ap.add_argument('--dataset', nargs='*', default=None,
                    help='only regenerate runs for these dataset(s), e.g. --dataset mutag')
    ap.add_argument('--dry_run', action='store_true',
                    help='Print the commands without running them.')
    args = ap.parse_args()
    import os
    if not args.mutag_data_root:
        args.mutag_data_root = os.environ.get('MUTAG_DATA_ROOT')
    if not args.ogb_data_root:
        args.ogb_data_root = os.environ.get('OGB_DATA_ROOT')

    out_root = Path(args.out_root)
    allowed = set(args.families)
    datasets = set(args.dataset) if args.dataset else None
    runs = sorted({p.parent for p in out_root.rglob('best_model.pt')})
    # Skip archived/scratch trees (matches iter_summaries) so regenerate never
    # overwrites summaries under _archive/_trash/_old.
    runs = [rd for rd in runs if not any(
        part.startswith(ARCHIVE_PREFIXES)
        for part in rd.relative_to(out_root).parts)]
    if datasets:
        runs = [rd for rd in runs if dataset_allowed(rd, datasets)]
        print(f'Dataset filter {sorted(datasets)}: {len(runs)} checkpoint(s)\n')
    else:
        print(f'Found {len(runs)} checkpoint(s) under {out_root}\n')

    ran = skipped = failed = 0
    for run_dir in runs:
        sj = run_dir / 'summary.json'
        if not sj.exists():
            print(f'  [skip] no summary.json: {run_dir}')
            skipped += 1
            continue
        try:
            with open(sj, encoding='utf-8') as f:
                meta = json.load(f)
        except Exception as e:
            print(f'  [skip] corrupt summary {sj}: {e}')
            skipped += 1
            continue
        try:
            exp_dir = str(run_dir.relative_to(out_root))
        except ValueError:
            exp_dir = str(run_dir)
        fam = resolve_family(meta, exp_dir)
        if not _family_allowed(fam, allowed):
            skipped += 1
            continue
        cmd = build_cmd(meta, run_dir, args)
        if cmd is None:
            print(f'  [skip] incomplete metadata: {run_dir}')
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
        print('Now re-run: python analysis/run_analysis.py collect --out_root '
              f'{args.out_root}')


if __name__ == '__main__':
    main()
