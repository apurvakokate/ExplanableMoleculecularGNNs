#!/usr/bin/env python3
"""run.py -- MotifSAT experiment entry point.

Usage
-----
    python run.py --dataset Mutagenicity --fold 0 --backbone GIN \
        --motif_method readout --noise none --info_loss_level node \
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
from SharedModules.data.loader import (
    get_loaders, compute_pos_weights, apply_gt_loaders, TASK_TYPE
)
from SharedModules.evaluation.pipeline import EvalPipeline
from SharedModules.evaluation.embedding_viz import EmbeddingVizLogger, build_impact_cache_from_eval
from SharedModules.evaluation.wandb_logger import WandbLogger
from SharedModules.evaluation.metrics import evaluate_predictions
from SharedModules.evaluation.multi_explanation_posthoc import run_multi_explanation_posthoc
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
            data  = data.clone().to(device)
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
    from SharedModules.data.dataset_routing import validate_use_gt, training_summary_extras
    from reg_config import resolve_gsat_r

    validate_use_gt(cfg.dataset, cfg.use_gt, cfg.gt_cache)
    _init, _final, _dec_int, _dec_r, _from_tbl = resolve_gsat_r(
        cfg.dataset, cfg.init_r, cfg.final_r, cfg.decay_interval, cfg.decay_r,
    )
    if _from_tbl:
        print(f'  [reg_config] {cfg.dataset}: init_r={_init} final_r={_final} '
              f'decay_interval={_dec_int} decay_r={_dec_r}')
    cfg.init_r = _init
    cfg.final_r = _final
    cfg.decay_interval = _dec_int
    cfg.decay_r = _dec_r

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
        mutag_index_maps_path=getattr(cfg, 'mutag_index_maps_path', None),
        mutag_smiles_csv_path=getattr(cfg, 'mutag_smiles_csv_path', None),
        mutag_splits_path=getattr(cfg, 'mutag_splits_path', None),
        mutag_seed=getattr(cfg, 'mutag_seed', 42),
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
    # apply_gt.py writes train/valid/test_with_gt.pt for every split; the shared
    # helper swaps all three loaders (fail-fast on an incomplete cache) so the
    # model trains on/eval against the rule label. pos_weights are recomputed
    # below from the GT training distribution.
    if getattr(cfg, 'use_gt', False) and getattr(cfg, 'gt_cache', None):
        loaders, test_ds = apply_gt_loaders(
            loaders, test_ds,
            gt_cache=cfg.gt_cache, dataset=cfg.dataset, fold=cfg.fold,
            vocab_variant=cfg.vocab_variant, batch_size=cfg.batch_size,
        )

    # Model
    model = build_model(cfg, task_type, meta)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: GSAT({tag})  params={n_params:,}")
    print(f"  [FIX#5 active] injection flags from CLI/config: "
          f"w_feat={cfg.w_feat} w_message={cfg.w_message} w_readout={cfg.w_readout} "
          f"(w_message is now opt-in, not forced True)")

    # Guard (parity with MOSE): with no injection AND no edge-attention path the
    # sampled/extracted attention is never applied in the 2nd forward pass, so it
    # receives no task gradient and the explanation is inert — the model reduces
    # to a plain backbone. learn_edge_att applies attention via edge_atten
    # regardless of the w_* flags, so it is exempt.
    if (not getattr(cfg, 'learn_edge_att', False)
            and not (cfg.w_feat or cfg.w_message or cfg.w_readout)):
        print("  [WARN] No attention injection enabled "
              "(w_feat/w_message/w_readout all False) and learn_edge_att off: "
              "the extractor attention is never applied, so it gets no task "
              "gradient and the explanation is inert. Pass at least one --w_* "
              "flag (or --learn_edge_att for base GSAT).")

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

    # Evaluate all splits. For regression, also report MAE/RMSE in the original
    # target units (denormalised via the train z-score std).
    _denorm = ((meta.norm_mean, meta.norm_std)
               if task_type == 'Regression' else None)
    split_metrics = {}
    for split_name in ("train", "valid", "test"):
        m = evaluate_predictions(model, loaders[split_name], device, task_type,
                                 denorm=_denorm)
        split_metrics[split_name] = m
        if cfg.verbose:
            print(f"  {split_name}: {m}")

    # Determine GT ROC level: edge for learn_edge_att, node otherwise
    gt_level = "edge" if cfg.learn_edge_att else "node"
    test_list = list(test_ds)
    from SharedModules.data.mutag_splits import mutag_gt_eval_graphs
    _gt_eval = (mutag_gt_eval_graphs(test_list)
                if cfg.dataset == 'mutag' else None)
    from SharedModules.data.dataset_routing import load_mutag_eval_index_maps
    _mutag_maps = (load_mutag_eval_index_maps(
        cfg.data_root, cfg.fold,
        index_maps_path=getattr(cfg, 'mutag_index_maps_path', None))
        if cfg.dataset == 'mutag' else None)
    pipeline = EvalPipeline(
        model, vocab, loaders["test"], test_list, device, task_type,
        max_motifs_eval=cfg.max_motifs_eval,
        gt_level=gt_level,
        denorm=_denorm,
        gt_eval_list=_gt_eval,
        index_maps=_mutag_maps,
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
        # readout/motif-aware: aggregate node attention to per-motif scores so
        # the correlation, discriminativeness and score-distribution stats
        # populate. Run once per aggregation (mean & max) — same treatment as
        # base GSAT — so both flavours are saved. The 'mean' run is the headline
        # used for summary.json.
        motif_agg = _aggregate_att_to_motif(
            model, test_list, device,
            learn_edge_att=cfg.learn_edge_att,
            max_graphs=cfg.max_motifs_eval or 500,
        )
        results = None
        for agg in ("mean", "max"):
            agg_scores = motif_agg.get(agg, {})
            r = pipeline.run(
                motif_scores=agg_scores if agg_scores else None,
                run_motif_impact=cfg.run_motif_impact,
            )
            for name, df in pipeline.to_dataframe(r).items():
                df.to_csv(out_dir / f"{name}_att_{agg}.csv", index=False)
            if agg == "mean":
                results = r
        summary_scores = motif_agg.get("mean", {})

    pipeline.print_summary(results)

    # Multi-explanation (optional inline; default is post-hoc via analysis/run_multi_explanation.py)
    if cfg.run_multi_explanation and not cfg.learn_edge_att:
        agg_fn = _aggregate_att_to_motif if cfg.motif_method in ('readout', 'none') else None
        _me_scores = summary_scores
        if not _me_scores and agg_fn is not None:
            _me_scores = agg_fn(
                model, test_list, device,
                learn_edge_att=cfg.learn_edge_att,
                max_graphs=cfg.max_motifs_eval or 500,
            ).get("mean", {})
        run_multi_explanation_posthoc(
            model, vocab, test_list, device, task_type, out_dir,
            motif_scores=_me_scores or None,
            learn_edge_att=cfg.learn_edge_att,
            att_aggregate_fn=agg_fn,
            max_motifs=cfg.max_motifs_eval,
        )

    # Save
    dfs = pipeline.to_dataframe(results)
    for name, df in dfs.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)

    pred = results.get("prediction", {})
    corr = results.get("correlation", {})
    gt   = results.get("gt_roc", {})
    gt_node = results.get("gt_roc_node", {})
    gt_node_mean = results.get("gt_roc_node_mean", {})
    gt_node_max  = results.get("gt_roc_node_max", {})
    gt_edge = results.get("gt_roc_edge", {})
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
        "motif_loss_coef":  cfg.motif_loss_coef,
        "within_node_coef": cfg.within_node_coef,
        "between_motif_coef": cfg.between_motif_coef,
        "learn_edge_att":   cfg.learn_edge_att,
        "gt_level":         gt_level,
        **training_summary_extras(cfg),
        # prediction
        "train_auc": split_metrics.get("train", {}).get("auc", split_metrics.get("train", {}).get("auc_mean", float("nan"))),
        "val_auc":   split_metrics.get("valid", {}).get("auc", split_metrics.get("valid", {}).get("auc_mean", float("nan"))),
        "auc":    pred.get("auc", pred.get("auc_mean", float("nan"))),
        "rmse":   pred.get("rmse", float("nan")),
        "mae":    pred.get("mae",  float("nan")),
        # regression metrics in original target units (denormalised); NaN for
        # classification tasks
        "rmse_orig": split_metrics.get("test", {}).get("rmse_orig", float("nan")),
        "mae_orig":  split_metrics.get("test", {}).get("mae_orig",  float("nan")),
        # correlation (score vs impact)
        "pearson":  corr.get("pearson",  float("nan")),
        "spearman": corr.get("spearman", float("nan")),
        # GT ROC (primary = configured level; node & edge reported alongside)
        "gt_roc_auc_mean": gt.get("auc_mean", float("nan")),
        "gt_roc_n_graphs": gt.get("n_graphs", 0),
        "gt_roc_node_auc_mean": gt_node.get("auc_mean", float("nan")),
        "gt_roc_node_mean_auc_mean": gt_node_mean.get("auc_mean", float("nan")),
        "gt_roc_node_max_auc_mean":  gt_node_max.get("auc_mean", float("nan")),
        "gt_roc_edge_auc_mean": gt_edge.get("auc_mean", float("nan")),
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
            gt_roc_node=results.get('gt_roc_node'),
            gt_roc_node_mean=results.get('gt_roc_node_mean'),
            gt_roc_node_max=results.get('gt_roc_node_max'),
            gt_roc_edge=results.get('gt_roc_edge'),
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
                        choices=["none","loss","readout","motif_emb"])
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
    parser.add_argument("--init_r",          type=float, default=None,
                        help="IB prior retention at start (default 0.9).")
    parser.add_argument("--final_r",         type=float, default=None,
                        help="IB prior floor; resolved from reg_config when omitted "
                             "(mutag/Mutagenicity=0.5, OGB=0.7).")
    parser.add_argument("--decay_interval",  type=int, default=None,
                        help="Anneal r every N epochs (default 10; OGB=20).")
    parser.add_argument("--decay_r",         type=float, default=None,
                        help="Subtract this from r each decay step (default 0.1).")
    parser.add_argument("--motif_loss_coef", type=float, default=0.0,
                        help="Outer multiplier on the motif consistency loss. "
                             "The consistency term is "
                             "motif_loss_coef * (within_node_coef*within_var "
                             "- between_motif_coef*between_var), so this must be "
                             ">0 AND at least one of the two coefs below set for "
                             "the consistency loss to have any effect.")
    parser.add_argument("--within_node_coef", type=float, default=0.0,
                        help="Weight on within-motif attention variance "
                             "(penalise; encourages consistent attention within "
                             "a motif). Gated by --motif_loss_coef.")
    parser.add_argument("--between_motif_coef", type=float, default=0.0,
                        help="Weight on between-motif attention variance "
                             "(reward; encourages discrimination across motifs). "
                             "Gated by --motif_loss_coef.")
    parser.add_argument("--epochs",          type=int, default=100)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--data_root",       default="./datasets/FOLDS")
    parser.add_argument("--vocab_root",      default="./motifsat_output")
    parser.add_argument("--vocab_variant",   default="all_fallback_bpe")
    parser.add_argument("--out_dir",         default="./motifsat_results")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--processed_root", default=os.environ.get("PROCESSED_ROOT"),
                        help="PyG .pt cache base ($PROCESSED_ROOT; per-vocab subdir appended)")
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
    parser.add_argument("--mutag_index_maps_path", default=None,
                        help="mutag only: override path to "
                             "mutag_<fold>_index_maps.pkl (default: convention "
                             "under --data_root).")
    parser.add_argument("--mutag_smiles_csv_path", default=None,
                        help="mutag only: override path to mutag_<fold>.csv "
                             "(default: convention under --data_root).")
    parser.add_argument("--mutag_splits_path", default=None,
                        help="mutag only: override path to mutag_<fold>_splits.pkl.")
    parser.add_argument("--mutag_seed", type=int, default=42,
                        help="mutag only: RNG seed when splits pickle is absent.")
    parser.add_argument("--final_out_dir",   action="store_true",
                        help="Treat --out_dir as the FINAL run dir (no "
                             "<dataset>/fold<k>/<variant_tag> append). Set by the "
                             "unified launcher to avoid double dataset/fold nesting.")
    parser.add_argument("--eval_only",       action="store_true",
                        help="Skip training; load a checkpoint and only run the "
                             "eval pipeline (regenerates summary + per-motif CSVs).")
    parser.add_argument("--load_weights_from", default=None,
                        help="Run dir or path to best_model.pt for --eval_only.")
    parser.add_argument("--run_multi_explanation", action="store_true",
                        help="Run H0/H1/H2 multi-explanation inline after eval "
                             "(default: post-hoc via analysis/run_multi_explanation.py).")
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
        from SharedModules.data.dataset_routing import (
            default_processed_base,
            variant_processed_root,
        )
        _base_proc = default_processed_base(args.data_root, args.processed_root)
        _proc_root = variant_processed_root(_base_proc, args.vocab_variant)
        cfg = MotifSATConfig(
            dataset=args.dataset, fold=args.fold,
            backbone=args.backbone, motif_method=args.motif_method,
            noise=args.noise, info_loss_level=args.info_loss_level,
            w_feat=args.w_feat, w_message=args.w_message,
            w_readout=args.w_readout, learn_edge_att=args.learn_edge_att,
            hidden_dim=args.hidden_dim, num_layers=args.num_layers,
            info_loss_coef=args.info_loss_coef,
            init_r=args.init_r,
            final_r=args.final_r,
            decay_interval=args.decay_interval,
            decay_r=args.decay_r,
            motif_loss_coef=args.motif_loss_coef,
            within_node_coef=args.within_node_coef,
            between_motif_coef=args.between_motif_coef,
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
            mutag_index_maps_path=args.mutag_index_maps_path,
            mutag_smiles_csv_path=args.mutag_smiles_csv_path,
            mutag_splits_path=args.mutag_splits_path,
            mutag_seed=args.mutag_seed,
            eval_only=args.eval_only,
            load_weights_from=args.load_weights_from,
            conv_normalize=args.conv_normalize,
            gin_inner_bn=args.gin_inner_bn,
            run_multi_explanation=args.run_multi_explanation,
        )
    run(cfg)


if __name__ == "__main__":
    main()
