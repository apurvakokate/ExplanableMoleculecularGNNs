#!/usr/bin/env python3
"""eval_driver_antehoc.py — per-split re-eval driver for the ANTE-HOC families
(mose / motifsat / gsat). Reloads each native checkpoint (rebuilt to MATCH
training, ``strict=True`` so any mismatch fails LOUD), extracts its OWN per-motif
scores, and calls ``evaluate_native_all_splits`` — which produces, per
train/valid/test split: grouped + instance correlation (own axis; MEAN node
reduction; 4 variants w/u x incl/excl-UNK), GT-ROC fired (global/instance),
spurious/family ROC, and AUC/RMSE/MAE.

OWN per-motif scores:
  * mose            → ``model.get_motif_scores()`` (learned, global)
  * motifsat / gsat → ``_aggregate_att_to_motif(..., MEAN)`` over train+val+test

Writes into --dest_root (mirrors the run tree; --out_root stays read-only).

  python analysis/eval_driver_antehoc.py --out_root final_v2 --dest_root posthoc_v1 \
      --data_root <DATA> --vocab_root <VOCAB> [--processed_root <PROC>] \
      --device cuda --families mose [--dataset BBBP] [--shard i/N]

NOTE: MOSE-GNN/ and MotifSAT/ are script-style dirs with clashing bare module
names (config/model/run/train/...). We import one family at a time, purging the
clashing modules on a family change, so a worker can safely handle either. For
throughput, LAUNCH ONE FAMILY PER POOL (--families mose | motifsat | gsat) so the
import happens once. mutag needs --data_root=<repo>/data (MUTAG_DATA_ROOT).
"""
import argparse
import importlib
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.eval_driver_common import (
    build_gt_loaders, split_lists_and_gt, denorm_of, iter_runs, out_dir_for)
from SharedModules.evaluation.split_eval import evaluate_native_all_splits

FAMILIES = ('mose', 'motifsat', 'gsat')
_FAM_DIR = {'mose': 'MOSE-GNN', 'motifsat': 'MotifSAT', 'gsat': 'MotifSAT'}
# bare module names that clash between MOSE-GNN/ and MotifSAT/ (script-style dirs)
_CLASH = ('config', 'model', 'run', 'train', 'reg_config', 'losses', 'motif_modules')
_loaded = {'fam': None, 'run': None, 'Config': None}


def _family_module(fam):
    """Import (run, Config) for one family, purging clashing bare modules on a
    family change so MOSE-GNN and MotifSAT never collide in one process."""
    want_dir = _FAM_DIR[fam]
    if _loaded['fam'] == fam:
        return _loaded['run'], _loaded['Config']
    for m in _CLASH:
        sys.modules.pop(m, None)
    for d in ('MOSE-GNN', 'MotifSAT'):
        p = str(_REPO / d)
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(_REPO / want_dir))
    run_mod = importlib.import_module('run')
    cfg_mod = importlib.import_module('config')
    Config = cfg_mod.MOSEConfig if fam == 'mose' else cfg_mod.MotifSATConfig
    _loaded.update(fam=fam, run=run_mod, Config=Config)
    return run_mod, Config


def _cfg_from_meta(Config, meta):
    """Reconstruct the training config from summary.json meta (fields not recorded
    keep their dataclass default — those are non-architectural training knobs)."""
    cfg = Config()
    for k, v in meta.items():
        if v is not None and hasattr(cfg, k):
            try:
                setattr(cfg, k, v)
            except Exception:
                pass
    return cfg


def _flatten_mose_scores(ms):
    """get_motif_scores() → {mid: score}. Multi-channel variant returns
    {channel: {mid: score}} — flatten (mirrors MOSE-GNN/run.py's collapse)."""
    if isinstance(ms, dict) and ms and isinstance(next(iter(ms.values())), dict):
        flat = {}
        for chan in ms.values():
            for mid, s in chan.items():
                flat[int(mid)] = float(s)
        return flat
    return {int(m): float(s) for m, s in (ms or {}).items()}


def load_native_model_and_scores(run_dir, meta, fam, loaders, vocab, dmeta, device, task_type):
    """Rebuild the native model to MATCH training + strict load; return (model, own
    per-motif scores). strict=True: any state_dict mismatch aborts LOUD rather than
    silently loading a partial (wrong) motif head."""
    run_mod, Config = _family_module(fam)
    cfg = _cfg_from_meta(Config, meta)
    sd = torch.load(run_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    sd = sd.get('model_state_dict', sd) if isinstance(sd, dict) else sd

    # build_model's ``meta`` arg is the DATASET metadata object (dmeta: .x_dim/.deg),
    # NOT the summary.json dict — cfg already carries the run config from summary meta.
    if fam == 'mose':
        kept = (getattr(dmeta, 'kept_motif_ids', None)
                if getattr(dmeta, 'kept_motif_ids', None) is not None
                else getattr(vocab, 'kept_motif_ids', None))
        model = run_mod.build_model(cfg, vocab.num_motifs, task_type, dmeta, kept_motif_ids=kept)
        model.load_state_dict(sd)                       # strict=True (default)
        model = model.to(device).eval()
        scores = _flatten_mose_scores(model.get_motif_scores())
    else:  # motifsat / gsat  — both the MotifSAT GSAT class (gsat = learn_edge_att)
        model = run_mod.build_model(cfg, task_type, dmeta)
        model.load_state_dict(sd)                       # strict=True
        model = model.to(device).eval()
        all_graphs = [g for s in ('train', 'valid', 'test')
                      for g in list(loaders[s].dataset)]
        agg = run_mod._aggregate_att_to_motif(
            model, all_graphs, device,
            learn_edge_att=getattr(cfg, 'learn_edge_att', False), max_graphs=None)
        scores = {int(m): float(s) for m, s in (agg.get('mean') or {}).items()}
    return model, scores


def process(run_dir, meta, fam, args, device):
    loaders, vocab, dmeta, task_type = build_gt_loaders(
        meta, args.data_root, args.vocab_root, args.processed_root, args.loader_batch_size)
    split_lists, gt = split_lists_and_gt(loaders, meta)
    model, scores = load_native_model_and_scores(
        run_dir, meta, fam, loaders, vocab, dmeta, device, task_type)
    if not scores:
        raise RuntimeError(f'{fam}: empty motif scores (no signal to correlate)')
    out_dir = out_dir_for(run_dir, args.out_root, args.dest_root)
    evaluate_native_all_splits(
        model, vocab, loaders, split_lists, gt, device, task_type,
        denorm_of(dmeta, task_type), out_dir, args.max_motifs_eval,
        motif_scores=scores, method_name=fam)


def main():
    ap = argparse.ArgumentParser(description='Ante-hoc per-split re-eval driver')
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--vocab_root', required=True)
    ap.add_argument('--processed_root', default=None)
    ap.add_argument('--dest_root', default=None)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--loader_batch_size', type=int, default=128)
    ap.add_argument('--max_motifs_eval', type=int, default=None)
    ap.add_argument('--dataset', nargs='*', default=None)
    ap.add_argument('--families', nargs='*', default=None,
                    help='subset of mose/motifsat/gsat (default all). LAUNCH ONE '
                         'FAMILY PER POOL for import safety + throughput.')
    ap.add_argument('--shard', default=None, help='i/N: process runs[i::N]')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--skip_done', action='store_true', default=True)
    ap.add_argument('--no_skip_done', dest='skip_done', action='store_false')
    args = ap.parse_args()

    device = torch.device(
        'cuda' if (args.device == 'cuda'
                   or (args.device == 'auto' and torch.cuda.is_available()))
        else 'cpu')
    fams = set(args.families) if args.families else set(FAMILIES)
    if not fams <= set(FAMILIES):
        raise SystemExit(f'--families must be subset of {FAMILIES}, got {sorted(fams)}')
    print(f'device: {device}  (ante-hoc driver, families={sorted(fams)})')

    ok = failed = 0
    for run_dir, meta, fam in iter_runs(args.out_root, fams, args.dataset, args.shard):
        if args.limit and (ok + failed) >= args.limit:
            break
        tag = f"{meta.get('dataset')}/{fam}/{meta.get('backbone')}/{meta.get('vocab_variant')}/fold{meta.get('fold')}"
        _dest = (Path(args.dest_root) / run_dir.relative_to(args.out_root)
                 if args.dest_root else run_dir)
        _ss = Path(_dest) / 'summary_splits.json'
        if args.skip_done and _ss.exists() and _ss.stat().st_size > 2:
            continue
        if args.dry_run:
            print(f'  [dry] {tag}')
            ok += 1
            continue
        try:
            process(run_dir, meta, fam, args, device)
            print(f'  [ok] {tag}')
            ok += 1
        except Exception as e:
            print(f'  [FAIL] {tag}: {type(e).__name__}: {e}')
            failed += 1
    print(f'\nDONE (ante-hoc): processed={ok}  failed={failed}')


if __name__ == '__main__':
    main()
