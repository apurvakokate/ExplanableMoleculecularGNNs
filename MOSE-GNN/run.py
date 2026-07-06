#!/usr/bin/env python3
"""run.py — MOSE-GNN experiment entry point.

Usage
-----
    python run.py --dataset Mutagenicity --fold 0 --backbone GIN
    python run.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from SharedModules.data.vocab import load_vocab
from SharedModules.data.loader import (
    get_loaders, compute_pos_weights, apply_gt_loaders, TASK_TYPE
)
from SharedModules.evaluation.pipeline import EvalPipeline
from SharedModules.evaluation.embedding_viz import EmbeddingVizLogger, build_impact_cache_from_eval
from SharedModules.evaluation.wandb_logger import WandbLogger
from SharedModules.evaluation.metrics import evaluate_predictions
from SharedModules.evaluation.multi_explanation_posthoc import run_multi_explanation_posthoc
from SharedModules.utils import set_seed, get_device

from config import MOSEConfig
from model import SingleChannelGNN, MultiChannelGNN
from train import train_mose_gnn


def build_model(cfg: MOSEConfig, num_motifs: int, task_type: str, meta,
                kept_motif_ids=None):
    """Construct the appropriate model variant."""
    if task_type == 'MultiLabel':
        raise ValueError(
            f"MOSE-GNN does not support MultiLabel dataset {cfg.dataset!r}. "
            f"OGB multi-label benchmarks (e.g. ogbg-moltox21) are excluded.")

    common = dict(
        x_dim=meta.x_dim,                 # 52 (CSV), 14 (mutag), 9 (OGB)
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        backbone=cfg.backbone,
        node_encoder=cfg.node_encoder,   # resolved in run(): CLI for CSV, atom_encoder for OGB
        apply_layer_norm=cfg.apply_layer_norm,
        num_motifs=num_motifs,
        kept_motif_ids=kept_motif_ids,
        unk_mode=cfg.unk_mode,
        unk_value=cfg.unk_value,
        w_feat=cfg.w_feat,
        w_message=cfg.w_message,
        w_readout=cfg.w_readout,
        dropout=cfg.dropout,
        deg=meta.deg,   # degree histogram for PNA; None for GIN/GCN/SAGE/GAT
        conv_normalize=getattr(cfg, 'conv_normalize', 'none'),
        gin_inner_bn=getattr(cfg, 'gin_inner_bn', True),
        self_gate=getattr(cfg, 'self_gate', False),
    )
    return SingleChannelGNN(**common)


def run(cfg: MOSEConfig) -> dict:
    from SharedModules.data.dataset_routing import validate_use_gt, training_summary_extras

    validate_use_gt(cfg.dataset, cfg.use_gt, cfg.gt_cache)
    set_seed(cfg.seed)
    device = get_device()

    task_type = TASK_TYPE.get(cfg.dataset, 'BinaryClass')
    if task_type == 'MultiLabel':
        raise ValueError(
            f"MOSE-GNN does not support MultiLabel dataset {cfg.dataset!r}. "
            f"OGB multi-label benchmarks (e.g. ogbg-moltox21) are excluded.")

    print(f'\n{"="*60}')
    print(f'  MOSE-GNN  {cfg.dataset}  fold={cfg.fold}  backbone={cfg.backbone}')
    print(f'{"="*60}')

    # Load vocabulary
    print('Loading vocabulary...')
    vocab = load_vocab(cfg.vocab_root, cfg.dataset, cfg.vocab_variant)
    print(f'  {vocab.num_motifs} motifs  mask_cache splits: {list(vocab.mask_cache.keys())}')

    # Data loaders
    loaders, test_ds, meta = get_loaders(
        dataset=cfg.dataset,
        data_root=cfg.data_root,
        fold=cfg.fold,
        vocab=vocab,
        processed_root=cfg.processed_root,
        batch_size=cfg.batch_size,
        normalize=(task_type == 'Regression'),
        mutag_index_maps_path=getattr(cfg, 'mutag_index_maps_path', None),
        mutag_smiles_csv_path=getattr(cfg, 'mutag_smiles_csv_path', None),
        mutag_splits_path=getattr(cfg, 'mutag_splits_path', None),
        mutag_seed=getattr(cfg, 'mutag_seed', 42),
    )
    print(f'  Task: {task_type}  train={len(loaders["train"].dataset)}'
          f'  val={len(loaders["valid"].dataset)}'
          f'  test={len(loaders["test"].dataset)}')

    # Honor --node_encoder for CSV datasets (force atom_encoder for OGB), and
    # store the resolved value on cfg so the model, variant_tag (output dir) and
    # logged config all agree. Previously the model used meta.node_encoder and
    # the CLI flag was silently ignored for CSV datasets.
    from SharedModules.data.loader import resolve_node_encoder
    cfg.node_encoder = resolve_node_encoder(getattr(cfg, 'node_encoder', None),
                                            meta.node_encoder)

    # ── GT loader replacement (use_gt=True: train on synthetic rule labels) ──
    # apply_gt.py writes train/valid/test_with_gt.pt for every split; the shared
    # helper swaps all three loaders (fail-fast on an incomplete cache) so the
    # model trains on/eval against the rule label. pos_weights are recomputed
    # below from the GT training distribution.
    if getattr(cfg, 'use_gt', False) and getattr(cfg, 'gt_cache', None):
        _gt_vocab = getattr(cfg, 'gt_vocab_variant', None) or cfg.vocab_variant
        loaders, test_ds = apply_gt_loaders(
            loaders, test_ds,
            gt_cache=cfg.gt_cache, dataset=cfg.dataset, fold=cfg.fold,
            vocab_variant=cfg.vocab_variant, batch_size=cfg.batch_size,
            gt_vocab_variant=_gt_vocab,
            refresh_vocab=vocab if _gt_vocab != cfg.vocab_variant else None,
            fold_motif_lookup=getattr(meta, 'motif_lookup', None),
            apply_threshold=getattr(meta, 'threshold_pct', None) is not None,
        )

    # Model
    _fold_kept = getattr(meta, 'kept_motif_ids', None)
    _kept_ids = (_fold_kept if _fold_kept is not None else vocab.kept_motif_ids)
    model = build_model(cfg, vocab.num_motifs, task_type, meta,
                        kept_motif_ids=_kept_ids)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _kept = len(_kept_ids) if _kept_ids is not None else vocab.num_motifs
    print(f'  Model: {model.__class__.__name__}  params={n_params:,}  '
          f'motif_params rows={_kept}/{vocab.num_motifs} (kept/total)')

    # Guard: with no injection the motif-importance params never enter the
    # forward pass, so they receive no task gradient (scores stay at sigmoid(0)
    # =0.5) and the explanation is meaningless — the model is effectively a
    # vanilla GNN. This is easy to hit from a bare CLI run because the
    # --w_feat/--w_message/--w_readout flags default OFF (the launchers always
    # pass them). Warn loudly so a degenerate run is never mistaken for MOSE.
    if vocab.num_motifs > 0 and not (cfg.w_feat or cfg.w_message or cfg.w_readout):
        print('  [WARN] No motif injection enabled (w_feat/w_message/w_readout '
              'all False): motif_params get no task gradient and MOSE degrades '
              'to a vanilla GNN. Pass at least one of --w_feat/--w_readout.')

    # Positive class weights
    pos_w = compute_pos_weights(loaders['train'].dataset) \
        if task_type in ('BinaryClass', 'MultiLabel') else None

    # Train
    if getattr(cfg, 'final_out_dir', False):
        out_dir = Path(cfg.out_dir)
    else:
        out_dir = Path(cfg.out_dir) / cfg.dataset / f'fold{cfg.fold}' / cfg.variant_tag()
    out_dir.mkdir(parents=True, exist_ok=True)

    # W&B initialisation (only when --use_wandb is passed)
    try:
        import os as _os
        import wandb as _wandb
        if getattr(cfg, 'use_wandb', False) and _wandb.run is None:
            _wb_kwargs = dict(
                project=getattr(cfg, 'wandb_project', 'ChemIntuit'),
                entity=getattr(cfg, 'wandb_entity', None),
                name=f'{cfg.dataset}_fold{cfg.fold}_{cfg.variant_tag()}',
                config=cfg.to_dict(),
                reinit=True,
            )
            # On HPC compute nodes outbound internet is usually blocked, which
            # makes the ONLINE wandb client spawn a network-polling thread that
            # throws BrokenPipe mid-run (and can abort training under set -e).
            # To make crashes impossible by default we DEFAULT TO OFFLINE: logs
            # are written to ./wandb/ and synced later with `wandb sync`.
            # Set WANDB_MODE=online explicitly to stream live.
            _mode = _os.environ.get('WANDB_MODE') or 'offline'
            try:
                _wandb.init(mode=_mode, **_wb_kwargs)
            except Exception as _e:
                print(f'  [warn] wandb init (mode={_mode}) failed ({_e}); '
                      f'retrying offline.')
                try:
                    _wandb.init(mode='offline', **_wb_kwargs)
                except Exception as _e2:
                    print(f'  [warn] wandb offline init also failed ({_e2}); '
                          f'continuing without W&B.')
        _wb_run = _wandb.run
    except ImportError:
        _wb_run = None

    wandb_logger = WandbLogger(
        model=model, vocab=vocab,
        task_type=task_type, model_type='mose',
        log_scores_every=getattr(cfg, 'log_scores_every', 5),
        wandb_run=_wb_run,
    ) if _wb_run is not None else None

    viz_logger = EmbeddingVizLogger(
        model=model, vocab=vocab, device=device,
        motif_scores=None,           # updated each epoch via update_motif_scores
        task_type=task_type,
        viz_every=getattr(cfg, 'viz_every', 5),
        max_points=getattr(cfg, 'viz_max_points', 3000),
        wandb_run=_wb_run,
    ) if _wb_run is not None else None

    if getattr(cfg, 'eval_only', False):
        # Load a trained checkpoint and skip training entirely. The eval
        # pipeline below regenerates summary.json + per-motif CSVs (impact,
        # discriminativeness, score_vs_impact, correlation) from the loaded
        # weights, so new metrics can be produced post-hoc without retraining.
        ckpt_src = cfg.load_weights_from or str(out_dir / 'best_model.pt')
        ckpt_path = Path(ckpt_src)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / 'best_model.pt'
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f'--eval_only set but no checkpoint at {ckpt_path}. '
                f'Pass --load_weights_from <run_dir or best_model.pt>.')
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = state.get('model_state_dict', state) if isinstance(state, dict) else state
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f'  [eval_only] loaded with {len(missing)} missing / '
                  f'{len(unexpected)} unexpected keys (non-strict).')
        print(f'  [eval_only] loaded weights from {ckpt_path}; skipping training.')
        model = model.to(device)
        history = {}
    else:
        model, history = train_mose_gnn(
            model, loaders, task_type, device,
            epochs=cfg.epochs, lr=cfg.lr,
            explainer_lr=getattr(cfg, 'explainer_lr', None),
            gnn_lr=getattr(cfg, 'gnn_lr', None),
            weight_decay=cfg.weight_decay,
            pos_weights=pos_w, size_reg=cfg.size_reg, ent_reg=cfg.ent_reg,
            top_tau=cfg.top_tau, ignore_unknowns=cfg.ignore_unknowns,
            patience=cfg.patience, min_epochs=cfg.min_epochs,
            clip_grad=cfg.clip_grad,
            save_path=str(out_dir / 'best_model.pt'),
            verbose=cfg.verbose,
            viz_logger=viz_logger,
            wandb_logger=wandb_logger,
        )

    # Evaluate all splits (train / valid / test). For regression, also report
    # MAE/RMSE in the original target units (denormalised via the train z-score
    # std) alongside the normalised values.
    _denorm = ((meta.norm_mean, meta.norm_std)
               if task_type == 'Regression' else None)
    split_metrics = {}
    for split_name in ('train', 'valid', 'test'):
        m = evaluate_predictions(model, loaders[split_name], device, task_type,
                                 denorm=_denorm)
        split_metrics[split_name] = m
        if cfg.verbose:
            print(f'  {split_name}: {m}')

    test_list = list(test_ds)
    from SharedModules.data.mutag_splits import mutag_gt_eval_graphs
    _gt_eval = (mutag_gt_eval_graphs(test_list)
                if cfg.dataset == 'mutag' else None)
    motif_scores = model.get_motif_scores() if hasattr(model, 'get_motif_scores') else None
    # For MultiLabel, average across classes for correlation
    if isinstance(motif_scores, dict) and motif_scores and isinstance(
            next(iter(motif_scores.values())), dict):
        import numpy as np
        all_scores = list(motif_scores.values())  # list of {mid: score}
        common_ids = set(all_scores[0].keys())
        flat_scores = {
            mid: float(np.mean([sc[mid] for sc in all_scores if mid in sc]))
            for mid in common_ids
        }
    else:
        flat_scores = motif_scores

    pipeline = EvalPipeline(
        model, vocab, loaders['test'], test_list, device, task_type,
        max_motifs_eval=cfg.max_motifs_eval,
        denorm=_denorm,
        gt_eval_list=_gt_eval,
    )
    results = pipeline.run(
        motif_scores=flat_scores,
        run_motif_impact=cfg.run_motif_impact,
    )

    # Multi-explanation (optional inline; default is post-hoc via analysis/run_multi_explanation.py)
    if cfg.run_multi_explanation and flat_scores:
        run_multi_explanation_posthoc(
            model, vocab, test_list, device, task_type, out_dir,
            motif_scores=flat_scores,
            max_motifs=cfg.max_motifs_eval,
        )

    # Save results
    dfs = pipeline.to_dataframe(results)
    for name, df in dfs.items():
        df.to_csv(out_dir / f'{name}.csv', index=False)

    pred = results.get('prediction', {})
    corr = results.get('correlation', {})
    gt   = results.get('gt_roc', {})
    gt_node = results.get('gt_roc_node', {})
    gt_node_mean = results.get('gt_roc_node_mean', {})
    gt_node_max  = results.get('gt_roc_node_max', {})
    gt_edge = results.get('gt_roc_edge', {})
    tdc  = results.get('top_disc_check', {})
    from SharedModules.evaluation.metrics import motif_score_stats
    sstats = motif_score_stats(flat_scores)
    summary = {
        'model_type':    'MOSE-GNN',
        'family':        'mose',
        'motif_method':  'mose',
        'dataset':       cfg.dataset,
        'fold':          cfg.fold,
        'backbone':      cfg.backbone,
        'variant_tag':   cfg.variant_tag(),
        'vocab_variant': cfg.vocab_variant,
        'node_encoder':  cfg.node_encoder,
        'apply_layer_norm': cfg.apply_layer_norm,
        'w_feat':        cfg.w_feat,
        'w_message':     cfg.w_message,
        'w_readout':     cfg.w_readout,
        'ent_reg':       cfg.ent_reg,
        'size_reg':      cfg.size_reg,
        'num_layers':    cfg.num_layers,
        'hidden_dim':    cfg.hidden_dim,
        'explainer_lr':  getattr(cfg, 'explainer_lr', None),
        'gnn_lr':        getattr(cfg, 'gnn_lr', None),
        'conv_normalize': getattr(cfg, 'conv_normalize', 'none'),
        'gin_inner_bn':  getattr(cfg, 'gin_inner_bn', True),
        **training_summary_extras(cfg),  # includes self_gate
        # prediction
        'train_auc': split_metrics.get('train', {}).get('auc', split_metrics.get('train', {}).get('auc_mean', float('nan'))),
        'val_auc':   split_metrics.get('valid', {}).get('auc', split_metrics.get('valid', {}).get('auc_mean', float('nan'))),
        'auc':       pred.get('auc', pred.get('auc_mean', float('nan'))),
        'rmse':      pred.get('rmse', float('nan')),
        'mae':       pred.get('mae',  float('nan')),
        # regression metrics in original target units (denormalised); NaN for
        # classification tasks
        'rmse_orig': split_metrics.get('test', {}).get('rmse_orig', float('nan')),
        'mae_orig':  split_metrics.get('test', {}).get('mae_orig',  float('nan')),
        # correlation (score vs impact)
        'pearson':   corr.get('pearson',  float('nan')),
        'spearman':  corr.get('spearman', float('nan')),
        # GT ROC (primary = configured level; node & edge reported alongside)
        'gt_roc_auc_mean': gt.get('auc_mean', float('nan')),
        'gt_roc_n_graphs': gt.get('n_graphs', 0),
        'gt_roc_node_auc_mean': gt_node.get('auc_mean', float('nan')),
        'gt_roc_node_mean_auc_mean': gt_node_mean.get('auc_mean', float('nan')),
        'gt_roc_node_max_auc_mean':  gt_node_max.get('auc_mean', float('nan')),
        'gt_roc_edge_auc_mean': gt_edge.get('auc_mean', float('nan')),
        # top-scored motifs class-discriminative?
        'top_k_abs_disc':      tdc.get('top_k_abs_disc', float('nan')),
        'mean_abs_disc':       tdc.get('mean_abs_disc', float('nan')),
        'score_disc_spearman': tdc.get('score_disc_spearman', float('nan')),
        # motif-score distribution
        **sstats,
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n  Results:')
    for k, v in results.get('prediction', {}).items():
        print(f'    {k}: {v:.4f}')
    if 'correlation' in results:
        c = results['correlation']
        print(f"    pearson={c['pearson']:.3f}  spearman={c['spearman']:.3f}")

    # Log final results to W&B
    if wandb_logger is not None:
        wandb_logger.log_final_results(
            split_metrics=split_metrics,
            correlation=results.get('correlation'),
            gt_roc=results.get('gt_roc'),
            gt_roc_node=results.get('gt_roc_node'),
            gt_roc_node_mean=results.get('gt_roc_node_mean'),
            gt_roc_node_max=results.get('gt_roc_node_max'),
            gt_roc_edge=results.get('gt_roc_edge'),
            top_bottom=results.get('top_bottom'),
        )

    return results


def main():
    parser = argparse.ArgumentParser(description='MOSE-GNN')
    parser.add_argument('--config',      default=None)
    parser.add_argument('--dataset',     default='Mutagenicity')
    parser.add_argument('--fold',        type=int, default=0)
    parser.add_argument('--backbone',    default='GIN')
    parser.add_argument('--node_encoder', default='onehot',
                        choices=['onehot','linear','atom_encoder'])
    parser.add_argument('--hidden_dim',  type=int, default=64)
    parser.add_argument('--num_layers',  type=int, default=None,
                        help='GNN depth. If omitted, resolved per dataset '
                             '(BBBP=2, others=3) from reg_config.py.')
    parser.add_argument('--unk_mode',    default='fixed')
    parser.add_argument('--w_feat',      action='store_true')
    parser.add_argument('--w_message',   action='store_true')
    parser.add_argument('--w_readout',   action='store_true')
    parser.add_argument('--ent_reg',     type=float, default=None,
                        help='Entropy reg. If omitted, resolved per '
                             '(backbone, dataset) from reg_config.py (PNA→GIN).')
    parser.add_argument('--size_reg',    type=float, default=None,
                        help='Size/sparsity reg. If omitted, resolved per '
                             '(backbone, dataset) from reg_config.py (PNA→GIN).')
    parser.add_argument('--epochs',      type=int, default=150)
    parser.add_argument('--gnn_lr',      type=float, default=None,
                        help='LR for GNN backbone params (default 0.001 from MOSEConfig).')
    parser.add_argument('--explainer_lr', type=float, default=None,
                        help='LR for motif-importance / explainer params (default 0.01).')
    parser.add_argument('--data_root',   default='./datasets/FOLDS')
    parser.add_argument('--vocab_root',  default='./motifsat_output')
    parser.add_argument('--vocab_variant', default='all_fallback_bpe')
    parser.add_argument('--out_dir',     default='./mose_results')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--use_wandb',   action='store_true',
                        help='Initialise a W&B run for this experiment')
    parser.add_argument('--wandb_project', default='ChemIntuit',
                        help='W&B project name')
    parser.add_argument('--wandb_entity',  default=None,
                        help='W&B entity (team/user)')
    parser.add_argument('--processed_root', default=os.environ.get('PROCESSED_ROOT'),
                        help='PyG .pt cache base ($PROCESSED_ROOT; per-vocab subdir appended)')
    parser.add_argument('--use_gt',      action='store_true',
                        help='Load ground-truth relabelled graphs from gt_cache')
    parser.add_argument('--gt_cache',    default=None,
                        help='Path to gt_cache directory written by phase4')
    parser.add_argument('--gt_vocab_variant', default=None,
                        help='Base vocab variant for gt_cache lookup when '
                             'training on a *_filter vocab.')
    parser.add_argument('--mutag_index_maps_path', default=None,
                        help='mutag only: override path to '
                             'mutag_<fold>_index_maps.pkl (default: convention '
                             'under --data_root).')
    parser.add_argument('--mutag_smiles_csv_path', default=None,
                        help='mutag only: override path to mutag_<fold>.csv '
                             '(default: convention under --data_root).')
    parser.add_argument('--mutag_splits_path', default=None,
                        help='mutag only: override path to mutag_<fold>_splits.pkl.')
    parser.add_argument('--mutag_seed', type=int, default=42,
                        help='mutag only: RNG seed when splits pickle is absent.')
    parser.add_argument('--final_out_dir', action='store_true',
                        help='Treat --out_dir as the FINAL run dir (no '
                             '<dataset>/fold<k>/<variant_tag> append). Set by the '
                             'unified launcher to avoid double dataset/fold nesting.')
    parser.add_argument('--eval_only',   action='store_true',
                        help='Skip training; load a checkpoint and only run the '
                             'eval pipeline (regenerates summary + per-motif CSVs).')
    parser.add_argument('--load_weights_from', default=None,
                        help='Run dir or path to best_model.pt for --eval_only. '
                             'Defaults to the out_dir for this config.')
    parser.add_argument('--conv_normalize', default='none',
                        choices=['l2', 'layernorm', 'none'],
                        help='Per-conv normalization (default none for MOSE; '
                             'l2 cancels motif-weight magnitude scaling).')
    parser.add_argument('--run_multi_explanation', action='store_true',
                        help='Run multiple-explanation / co-occurrence (H0/H1/H2) '
                             'analysis after training and save multi_explanation.*')
    parser.add_argument('--no_gin_inner_bn', dest='gin_inner_bn',
                        action='store_false',
                        help='Disable BatchNorm inside the GIN MLP (default on).')
    parser.set_defaults(gin_inner_bn=True)
    parser.add_argument('--self_gate', action='store_true',
                        help='EXPERIMENTAL (default off): gate GIN/SAGE self-term '
                             'by node attention so the w_message gate controls all '
                             "of a node's signal. No-op for GCN/GAT/PNA.")
    args = parser.parse_args()

    if args.config:
        cfg = MOSEConfig.from_yaml(args.config)
    else:
        from reg_config import resolve_reg, resolve_num_layers
        from SharedModules.data.dataset_routing import (
            default_processed_base,
            variant_processed_root,
        )
        # Resolve regularization per (backbone, dataset) unless given explicitly.
        _ent, _size, _from_tbl = resolve_reg(
            args.backbone, args.dataset, args.ent_reg, args.size_reg)
        if _from_tbl:
            print(f'  [reg_config] {args.backbone}×{args.dataset}: '
                  f'ent_reg={_ent} size_reg={_size}'
                  + (' (PNA→GIN)' if args.backbone == 'PNA' else ''))
        # Resolve GNN depth per dataset unless given explicitly (BBBP=2, else 3).
        _nlayers, _nl_from_tbl = resolve_num_layers(args.dataset, args.num_layers)
        if _nl_from_tbl:
            print(f'  [reg_config] {args.dataset}: num_layers={_nlayers}')
        _base_proc = default_processed_base(args.data_root, args.processed_root)
        _proc_root = variant_processed_root(_base_proc, args.vocab_variant)
        cfg = MOSEConfig(
            dataset=args.dataset, fold=args.fold,
            backbone=args.backbone, hidden_dim=args.hidden_dim,
            num_layers=_nlayers, unk_mode=args.unk_mode,
            w_feat=args.w_feat, w_message=args.w_message,
            w_readout=args.w_readout,
            ent_reg=_ent, size_reg=_size,
            epochs=args.epochs,
            gnn_lr=0.001 if args.gnn_lr is None else args.gnn_lr,
            explainer_lr=0.01 if args.explainer_lr is None else args.explainer_lr,
            data_root=args.data_root,
            vocab_root=args.vocab_root, vocab_variant=args.vocab_variant,
            node_encoder=args.node_encoder,
            processed_root=_proc_root,
            out_dir=args.out_dir, seed=args.seed,
            final_out_dir=args.final_out_dir,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            use_gt=args.use_gt, gt_cache=args.gt_cache,
            gt_vocab_variant=args.gt_vocab_variant,
            mutag_index_maps_path=args.mutag_index_maps_path,
            mutag_smiles_csv_path=args.mutag_smiles_csv_path,
            mutag_splits_path=args.mutag_splits_path,
            mutag_seed=args.mutag_seed,
            eval_only=args.eval_only,
            load_weights_from=args.load_weights_from,
            conv_normalize=args.conv_normalize,
            gin_inner_bn=args.gin_inner_bn,
            self_gate=args.self_gate,
            run_multi_explanation=args.run_multi_explanation,
        )
    run(cfg)


if __name__ == '__main__':
    main()
