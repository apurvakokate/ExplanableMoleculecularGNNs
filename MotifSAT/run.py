#!/usr/bin/env python3
"""run.py -- MotifSAT experiment entry point.

Usage
-----
    python run.py --dataset Mutagenicity --fold 0 --backbone GIN \
        --motif_method node_emb --noise none --info_loss_level node \
        --w_message --vocab_root ./vocab_output --data_root ./FOLDS
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
from SharedModules.data.loader import get_loaders, compute_pos_weights, TASK_TYPE
from SharedModules.evaluation.pipeline import EvalPipeline
from SharedModules.evaluation.embedding_viz import EmbeddingVizLogger, build_impact_cache_from_eval
from SharedModules.evaluation.wandb_logger import WandbLogger
from SharedModules.evaluation.metrics import evaluate_predictions
from SharedModules.evaluation.multi_explanation import MultiExplanationAnalysis
from SharedModules.utils import set_seed, get_device

from config import MotifSATConfig
from model import GSAT
from train import train_gsat


def build_model(cfg: MotifSATConfig, task_type: str, meta) -> GSAT:
    from SharedModules.data.loader import NUM_CLASSES
    num_classes = NUM_CLASSES.get(cfg.dataset, 1)
    return GSAT(
        x_dim=meta.x_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        backbone_name=cfg.backbone,
        node_encoder=cfg.node_encoder,   # resolved in run(): CLI for CSV, atom_encoder for OGB
        apply_layer_norm=cfg.apply_layer_norm,
        dropout=cfg.dropout,
        num_classes=num_classes,
        task_type=task_type,
        motif_method=cfg.motif_method,
        pool_mode=cfg.pool_mode,
        extractor_hidden_mult=cfg.extractor_hidden_mult,
        extractor_dropout_p=cfg.extractor_dropout_p,
        noise=cfg.noise,
        info_loss_level=cfg.info_loss_level,
        motif_info_size_normalize=cfg.motif_info_size_normalize,
        w_feat=cfg.w_feat,
        w_message=cfg.w_message,
        w_readout=cfg.w_readout,
        learn_edge_att=cfg.learn_edge_att,
        init_r=cfg.init_r,
        final_r=cfg.final_r,
        decay_interval=cfg.decay_interval,
        decay_r=cfg.decay_r,
        info_loss_coef=cfg.info_loss_coef,
        motif_loss_coef=cfg.motif_loss_coef,
        between_motif_coef=cfg.between_motif_coef,
        within_node_coef=cfg.within_node_coef,
        deg=meta.deg,   # degree histogram for PNA; None for GIN/GCN/SAGE/GAT
        conv_normalize=getattr(cfg, 'conv_normalize', 'l2'),
        gin_inner_bn=getattr(cfg, 'gin_inner_bn', True),
    )


def _aggregate_att_to_motif(
    model: torch.nn.Module,
    data_list: list,
    device: torch.device,
    learn_edge_att: bool = False,
    max_graphs: int = 500,
) -> dict:
    """Aggregate node attention to per-motif-vocabulary scores (mean and max).

    Only meaningful when learn_edge_att=False (node attention path).

    When learn_edge_att=False:
        out = (logits, node_att [N,1], aux)
        node_att[i] reflects how much node i contributed to the prediction.
        local_mean(m,g) = mean node_att[nodes where n2m == m]
        local_max(m,g)  = max  node_att[nodes where n2m == m]
        score_mean(m)   = mean_g local_mean(m, g)
        score_max(m)    = mean_g local_max(m, g)

    When learn_edge_att=True:
        The edge extractor is a SEPARATE MLP on concatenated node embeddings
        (not derived from node attention). node_att is None. Edge attention
        values are not semantically tied to vocabulary motifs — they reflect
        edge-level relevance, not motif-level importance. Motif aggregation
        is therefore skipped and an empty dict is returned.

    Returns {'mean': {motif_id: float}, 'max': {motif_id: float}}
    or {'mean': {}, 'max': {}} when learn_edge_att=True.
    """
    if learn_edge_att:
        # Edge attention is a separate MLP, not comparable at motif level.
        return {'mean': {}, 'max': {}}

    model.eval()
    mean_sum: dict = {}
    mean_cnt: dict = {}
    max_sum:  dict = {}
    max_cnt:  dict = {}

    with torch.no_grad():
        for data in data_list[:max_graphs]:
            data  = data.to(device)
            n     = data.x.size(0)
            n2m   = getattr(data, "nodes_to_motifs", None)
            if n2m is None:
                continue
            batch = (data.batch if data.batch is not None
                     else torch.zeros(n, dtype=torch.long, device=device))

            try:
                out      = model(data.x, data.edge_index, batch, n2m,
                                 getattr(data, "edge_attr", None))
                # Prefer the clean (noise-free) soft attention for aggregation;
                # the returned att is a soft sigmoid but at train time carries
                # injected noise. node_att_soft is the noise-free probability.
                node_att = None
                if len(out) >= 3 and isinstance(out[2], dict) \
                        and out[2].get("node_att_soft") is not None:
                    node_att = out[2]["node_att_soft"]
                if node_att is None:
                    node_att = out[1]
                if node_att is None:
                    continue
                node_score = node_att.view(-1).detach().cpu()
            except Exception:
                continue

            n2m_cpu = n2m.cpu()
            for mid in n2m_cpu[n2m_cpu >= 0].unique().tolist():
                s_m = node_score[n2m_cpu == mid]
                if s_m.numel() == 0:
                    continue
                mean_sum[mid] = mean_sum.get(mid, 0.0) + float(s_m.mean())
                mean_cnt[mid] = mean_cnt.get(mid, 0)   + 1
                max_sum[mid]  = max_sum.get(mid, 0.0)  + float(s_m.max())
                max_cnt[mid]  = max_cnt.get(mid, 0)    + 1

    return {
        "mean": {mid: mean_sum[mid] / mean_cnt[mid]
                 for mid in mean_sum if mean_cnt[mid] > 0},
        "max":  {mid: max_sum[mid]  / max_cnt[mid]
                 for mid in max_sum  if max_cnt[mid]  > 0},
    }


def run(cfg: MotifSATConfig) -> dict:
    set_seed(cfg.seed)
    device = get_device()

    print(f'\n{"="*60}')
    print(f'  MotifSAT  {cfg.dataset}  fold={cfg.fold}')
    print(f'{"="*60}')

    # Vocabulary
    print("Loading vocabulary...")
    vocab = load_vocab(cfg.vocab_root, cfg.dataset, cfg.vocab_variant)
    print(f"  {vocab.num_motifs} motifs")

    # Data
    task_type = TASK_TYPE.get(cfg.dataset, "BinaryClass")
    loaders, test_ds, meta = get_loaders(
        dataset=cfg.dataset,
        data_root=cfg.data_root,
        fold=cfg.fold,
        vocab=vocab,
        processed_root=cfg.processed_root,
        batch_size=cfg.batch_size,
        normalize=(task_type == "Regression"),
    )
    print(f"  Task: {task_type}  "
          f"train={len(loaders['train'].dataset)}  "
          f"val={len(loaders['valid'].dataset)}  "
          f"test={len(loaders['test'].dataset)}")

    # Honor --node_encoder for CSV (force atom_encoder for OGB); store on cfg so
    # model build and variant_tag agree. tag computed AFTER this resolution.
    from SharedModules.data.loader import resolve_node_encoder
    cfg.node_encoder = resolve_node_encoder(getattr(cfg, 'node_encoder', None),
                                            meta.node_encoder)
    tag = cfg.variant_tag()
    print(f'  variant: {tag}')

    # ── GT loader replacement (use_gt=True: train on synthetic rule labels) ──
    # apply_gt.py writes train/valid/test_with_gt.pt for every split.
    # When use_gt=True ALL three loaders are replaced so the model trains
    # to predict the rule-derived label, not the original activity label.
    # pos_weights are recomputed below from the GT training distribution.
    if getattr(cfg, 'use_gt', False) and getattr(cfg, 'gt_cache', None):
        from torch_geometric.loader import DataLoader as _DataLoader
        _gt_base = (Path(cfg.gt_cache) / cfg.dataset
                    / f'fold{cfg.fold}' / cfg.vocab_variant / 'relabel1')
        _gt_loaded: dict = {}
        _gt_missing: list = []
        for _split in ('train', 'valid', 'test'):
            _gt_path = _gt_base / f'{_split}_with_gt.pt'
            if _gt_path.exists():
                _gt_loaded[_split] = torch.load(_gt_path, weights_only=False)
                print(f'  GT {_split}: {len(_gt_loaded[_split])} graphs ← {_gt_path.name}')
            else:
                _gt_missing.append(str(_gt_path))
        # FAIL FAST: use_gt requested but cache incomplete -> do not silently mix
        # GT and original-label loaders (would train on the wrong target).
        if _gt_missing:
            raise FileNotFoundError(
                "use_gt=True but the ground-truth cache is incomplete. Missing:\n  "
                + "\n  ".join(_gt_missing)
                + f"\nRun phase-4 relabelling for dataset={cfg.dataset} "
                  f"fold={cfg.fold} variant={cfg.vocab_variant} first, or unset --use_gt.")
        for _split, _shuffle in (('train', True), ('valid', False), ('test', False)):
            if _split in _gt_loaded:
                loaders[_split] = _DataLoader(
                    _gt_loaded[_split], batch_size=cfg.batch_size,
                    shuffle=_shuffle, num_workers=0,
                )
        if 'test' in _gt_loaded:
            test_ds = _gt_loaded['test']
        if _gt_loaded:
            print('  Training on GT-relabelled data '
                  '(data.y = rule-based synthetic labels)')
            print(f'  [FIX#6 active] GT loaders replaced: '
                  f'{sorted(_gt_loaded.keys())} '
                  f"(test loader now GT-backed: {'test' in _gt_loaded})")

    # Model
    model = build_model(cfg, task_type, meta)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: GSAT({tag})  params={n_params:,}")
    print(f"  [FIX#5 active] injection flags from CLI/config: "
          f"w_feat={cfg.w_feat} w_message={cfg.w_message} w_readout={cfg.w_readout} "
          f"(w_message is now opt-in, not forced True)")

    # Positive class weights
    pos_w = (compute_pos_weights(loaders["train"].dataset)
             if task_type in ("BinaryClass", "MultiLabel") else None)

    # Output dir
    if getattr(cfg, 'final_out_dir', False):
        out_dir = Path(cfg.out_dir)
    else:
        out_dir = (Path(cfg.out_dir) / cfg.dataset / f"fold{cfg.fold}" / tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train
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
            # HPC nodes usually block internet → the ONLINE client's polling
            # thread throws BrokenPipe mid-run. DEFAULT TO OFFLINE so crashes
            # are impossible; set WANDB_MODE=online to stream live.
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
        task_type=task_type, model_type='motifsat',
        log_scores_every=getattr(cfg, 'log_scores_every', 5),
        wandb_run=_wb_run,
    ) if _wb_run is not None else None

    viz_logger = EmbeddingVizLogger(
        model=model, vocab=vocab, device=device,
        motif_scores=None,   # MotifSAT: per-instance attention used (no global scores)
        task_type=task_type,
        viz_every=getattr(cfg, 'viz_every', 5),
        max_points=getattr(cfg, 'viz_max_points', 3000),
        wandb_run=_wb_run,
    ) if _wb_run is not None else None

    if getattr(cfg, "eval_only", False):
        # Skip training; load a checkpoint and only run the eval pipeline so the
        # new metrics (correlation, discriminativeness, score_vs_impact, score
        # stats) regenerate post-hoc without retraining.
        from pathlib import Path as _P
        ckpt_src = cfg.load_weights_from or str(out_dir / "best_model.pt")
        ckpt_path = _P(ckpt_src)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "best_model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"--eval_only set but no checkpoint at {ckpt_path}. "
                f"Pass --load_weights_from <run_dir or best_model.pt>.")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"  [eval_only] loaded with {len(missing)} missing / "
                  f"{len(unexpected)} unexpected keys (non-strict).")
        print(f"  [eval_only] loaded weights from {ckpt_path}; skipping training.")
        model = model.to(device)
        history = {}
    else:
        model, history = train_gsat(
            model, loaders, task_type, device,
            epochs=cfg.epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            pos_weights=pos_w,
            motif_lengths=vocab.motif_lengths if vocab else None,
            patience=cfg.patience,
            min_epochs=cfg.min_epochs,
            clip_grad=cfg.clip_grad,
            save_path=str(out_dir / "best_model.pt"),
            verbose=cfg.verbose,
            viz_logger=viz_logger,
            wandb_logger=wandb_logger,
        )

    # Evaluate all splits
    split_metrics = {}
    for split_name in ("train", "valid", "test"):
        m = evaluate_predictions(model, loaders[split_name], device, task_type)
        split_metrics[split_name] = m
        if cfg.verbose:
            print(f"  {split_name}: {m}")

    # Determine GT ROC level: edge for learn_edge_att, node otherwise
    gt_level = "edge" if cfg.learn_edge_att else "node"
    test_list = list(test_ds)
    pipeline = EvalPipeline(
        model, vocab, loaders["test"], test_list, device, task_type,
        max_motifs_eval=cfg.max_motifs_eval,
        gt_level=gt_level,
    )
    # For base GSAT (learn_edge_att=True, motif_method='none') aggregate node
    # attention to vocabulary-level scores (mean and max) for correlation eval.
    # For motif-aware variants, motif_scores comes from the model directly.
    if cfg.motif_method == "none":   # base GSAT: node att OR edge att
        gsat_agg = _aggregate_att_to_motif(
            model, test_list, device,
            learn_edge_att=cfg.learn_edge_att,
            max_graphs=cfg.max_motifs_eval or 500,
        )
        # Save both aggregations as CSVs
        import pandas as pd
        motif_list = getattr(vocab, "motif_list", [])
        for agg in ("mean", "max"):
            rows = [
                {"motif_id": mid,
                 f"score_{agg}": s,
                 "motif_smarts": motif_list[mid] if mid < len(motif_list) else "?"}
                for mid, s in gsat_agg[agg].items()
            ]
            if rows:
                pd.DataFrame(rows).to_csv(
                    out_dir / f"base_gsat_att_{agg}.csv", index=False)

        # Run pipeline twice — once per aggregation
        for agg in ("mean", "max"):
            agg_scores = gsat_agg[agg]
            r = pipeline.run(
                motif_scores=agg_scores if agg_scores else None,
                run_motif_impact=cfg.run_motif_impact,
            )
            dfs_agg = pipeline.to_dataframe(r)
            for name, df in dfs_agg.items():
                df.to_csv(out_dir / f"{name}_att_{agg}.csv", index=False)
        # Also run without motif_scores for the plain prediction result
        results = pipeline.run(run_motif_impact=False)
        summary_scores = gsat_agg.get("mean", {})
    else:
        # readout/motif-aware: aggregate attention to per-motif scores so the
        # correlation, discriminativeness and score-distribution stats populate.
        summary_scores = _aggregate_att_to_motif(
            model, test_list, device,
            learn_edge_att=cfg.learn_edge_att,
            max_graphs=cfg.max_motifs_eval or 500,
        ).get("mean", {})
        results = pipeline.run(
            motif_scores=summary_scores if summary_scores else None,
            run_motif_impact=cfg.run_motif_impact,
        )

    pipeline.print_summary(results)

    # Multi-explanation analysis (motif_method=readout only, noise comparison)
    if cfg.run_multi_explanation and cfg.motif_method in ("readout", "node_emb", "motif_emb"):
        try:
            print("\n  Running multi-explanation analysis ...")
            # Use aggregated attention as motif scores for MotifSAT
            _me_scores = _aggregate_att_to_motif(
                model, test_list, device,
                learn_edge_att=cfg.learn_edge_att,
                max_graphs=cfg.max_motifs_eval or 500,
            ).get("mean", {})
            analysis = MultiExplanationAnalysis(
                model, vocab, test_list, device,
                motif_scores=_me_scores,  # per-vocabulary aggregated attention scores
                task_type=task_type,
                max_motifs=cfg.max_motifs_eval,
            )
            analysis.run(local_filter="p75")
            analysis.save(str(out_dir / "multi_explanation"))
        except Exception as e:
            print(f"  [warn] Multi-explanation failed: {e}")

    # Save
    dfs = pipeline.to_dataframe(results)
    for name, df in dfs.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)

    pred = results.get("prediction", {})
    corr = results.get("correlation", {})
    gt   = results.get("gt_roc", {})
    tdc  = results.get("top_disc_check", {})
    from SharedModules.evaluation.metrics import motif_score_stats
    sstats = motif_score_stats(summary_scores)
    summary = {
        "model_type":       "MotifSAT",
        "dataset":          cfg.dataset,
        "fold":             cfg.fold,
        "backbone":         cfg.backbone,
        "variant_tag":      tag,
        "vocab_variant":    cfg.vocab_variant,
        "motif_method":     cfg.motif_method,
        "node_encoder":     cfg.node_encoder,
        "apply_layer_norm": cfg.apply_layer_norm,
        "w_feat":           cfg.w_feat,
        "w_message":        cfg.w_message,
        "w_readout":        cfg.w_readout,
        "noise":            cfg.noise,
        "info_loss_coef":   cfg.info_loss_coef,
        "learn_edge_att":   cfg.learn_edge_att,
        "gt_level":         gt_level,
        # prediction
        "train_auc": split_metrics.get("train", {}).get("auc", split_metrics.get("train", {}).get("auc_mean", float("nan"))),
        "val_auc":   split_metrics.get("valid", {}).get("auc", split_metrics.get("valid", {}).get("auc_mean", float("nan"))),
        "auc":    pred.get("auc", pred.get("auc_mean", float("nan"))),
        "rmse":   pred.get("rmse", float("nan")),
        "mae":    pred.get("mae",  float("nan")),
        # correlation (score vs impact)
        "pearson":  corr.get("pearson",  float("nan")),
        "spearman": corr.get("spearman", float("nan")),
        # GT ROC
        "gt_roc_auc_mean": gt.get("auc_mean", float("nan")),
        "gt_roc_n_graphs": gt.get("n_graphs", 0),
        # top-scored motifs class-discriminative?
        "top_k_abs_disc":      tdc.get("top_k_abs_disc", float("nan")),
        "mean_abs_disc":       tdc.get("mean_abs_disc", float("nan")),
        "score_disc_spearman": tdc.get("score_disc_spearman", float("nan")),
        # motif-score distribution
        **sstats,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Log final results to W&B
    if wandb_logger is not None:
        wandb_logger.log_final_results(
            split_metrics=split_metrics,
            correlation=results.get('correlation'),
            gt_roc=results.get('gt_roc'),
            top_bottom=results.get('top_bottom'),
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="MotifSAT")
    parser.add_argument("--config",          default=None)
    parser.add_argument("--dataset",         default="Mutagenicity")
    parser.add_argument("--fold",            type=int, default=0)
    parser.add_argument("--backbone",        default="GIN")
    parser.add_argument("--node_encoder",    default="onehot",
                        choices=["onehot","linear","atom_encoder"])
    parser.add_argument("--motif_method",    default="none",
                        choices=["none","loss","node_emb","motif_emb","readout"])
    parser.add_argument("--noise",           default="none",
                        choices=["none","node","motif"])
    parser.add_argument("--info_loss_level", default="node",
                        choices=["none","node","motif"])
    parser.add_argument("--w_feat",          action="store_true")
    parser.add_argument("--w_message",       action="store_true")
    parser.add_argument("--w_readout",       action="store_true")
    parser.add_argument("--learn_edge_att",  action="store_true")
    parser.add_argument("--hidden_dim",      type=int, default=64)
    parser.add_argument("--num_layers",      type=int, default=3)
    parser.add_argument("--info_loss_coef",  type=float, default=1.0)
    parser.add_argument("--motif_loss_coef", type=float, default=0.0)
    parser.add_argument("--epochs",          type=int, default=100)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--data_root",       default="./datasets/FOLDS")
    parser.add_argument("--vocab_root",      default="./motifsat_output")
    parser.add_argument("--vocab_variant",   default="all_fallback_bpe")
    parser.add_argument("--out_dir",         default="./motifsat_results")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--processed_root", default=None,
                        help="PyG .pt cache root ($PROCESSED_ROOT in config)")
    parser.add_argument("--use_wandb",       action="store_true",
                        help="Initialise a W&B run for this experiment")
    parser.add_argument("--wandb_project",   default="ChemIntuit")
    parser.add_argument("--wandb_entity",    default=None)
    parser.add_argument("--use_gt",          action="store_true",
                        help="Load ground-truth relabelled train/valid/test sets "
                             "from gt_cache; all three loaders are replaced and the "
                             "model trains on the rule-derived synthetic label")
    parser.add_argument("--gt_cache",        default=None,
                        help="Path to gt_cache directory written by phase4")
    parser.add_argument("--final_out_dir",   action="store_true",
                        help="Treat --out_dir as the FINAL run dir (no "
                             "<dataset>/fold<k>/<variant_tag> append). Set by the "
                             "unified launcher to avoid double dataset/fold nesting.")
    parser.add_argument("--eval_only",       action="store_true",
                        help="Skip training; load a checkpoint and only run the "
                             "eval pipeline (regenerates summary + per-motif CSVs).")
    parser.add_argument("--load_weights_from", default=None,
                        help="Run dir or path to best_model.pt for --eval_only.")
    parser.add_argument("--conv_normalize", default="l2",
                        choices=["l2", "layernorm", "none"],
                        help="Per-conv normalization (default l2).")
    parser.add_argument("--no_gin_inner_bn", dest="gin_inner_bn",
                        action="store_false",
                        help="Disable BatchNorm inside the GIN MLP (default on).")
    parser.set_defaults(gin_inner_bn=True)
    args = parser.parse_args()

    if args.config:
        cfg = MotifSATConfig.from_yaml(args.config)
    else:
        from pathlib import Path as _P
        _base_proc = args.processed_root or str(_P(args.data_root).parent / 'processed')
        # Make processed_root vocab-variant-specific so different vocab
        # variants never share cached .pt files (nodes_to_motifs differs).
        _proc_root = f'{_base_proc}/{args.vocab_variant}'
        cfg = MotifSATConfig(
            dataset=args.dataset, fold=args.fold,
            backbone=args.backbone, motif_method=args.motif_method,
            noise=args.noise, info_loss_level=args.info_loss_level,
            w_feat=args.w_feat, w_message=args.w_message,
            w_readout=args.w_readout, learn_edge_att=args.learn_edge_att,
            hidden_dim=args.hidden_dim, num_layers=args.num_layers,
            info_loss_coef=args.info_loss_coef,
            motif_loss_coef=args.motif_loss_coef,
            epochs=args.epochs, lr=args.lr,
            data_root=args.data_root, vocab_root=args.vocab_root,
            vocab_variant=args.vocab_variant, out_dir=args.out_dir,
            node_encoder=args.node_encoder,
            processed_root=_proc_root,
            seed=args.seed,
            final_out_dir=args.final_out_dir,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            use_gt=args.use_gt,
            gt_cache=args.gt_cache,
            eval_only=args.eval_only,
            load_weights_from=args.load_weights_from,
            conv_normalize=args.conv_normalize,
            gin_inner_bn=args.gin_inner_bn,
        )
    run(cfg)


if __name__ == "__main__":
    main()
