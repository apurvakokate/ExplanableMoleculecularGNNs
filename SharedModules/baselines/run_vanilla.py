#!/usr/bin/env python3
"""run_vanilla.py — Train VanillaGNN and run post-hoc explainers.

Trains a vanilla GNN (no motif parameters) then runs GNNExplainer,
PGExplainer, and MAGE to compute motif-level explanations comparable
to MOSE-GNN and MotifSAT.

Usage
-----
    python run_vanilla.py --dataset Mutagenicity --fold 0 --backbone GIN \\
        --vocab_root ./vocab_output --vocab_variant rbrics_nofall_nobpe_nofilter \\
        --data_root ./FOLDS --out_dir ./vanilla_results
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict

import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from SharedModules.data.dataset_routing import (
    default_processed_base,
    variant_processed_root,
)
from SharedModules.data.vocab import load_vocab
from SharedModules.data.loader import (
    get_loaders, compute_pos_weights, apply_gt_loaders, TASK_TYPE
)
from SharedModules.evaluation.pipeline import EvalPipeline
from SharedModules.evaluation.metrics import evaluate_predictions
from SharedModules.evaluation.motif_eval import compute_gt_roc
from SharedModules.utils import set_seed, get_device
from SharedModules.baselines.vanilla_gnn import VanillaGNN, train_vanilla_gnn
from SharedModules.baselines.gnn_explainer import run_gnnexplainer
from SharedModules.baselines.pg_explainer import run_pgexplainer
from SharedModules.baselines.mage import run_mage


@dataclass
class VanillaConfig:
    dataset: str          = 'Mutagenicity'
    data_root: str        = './datasets/FOLDS'
    vocab_root: str       = './vocab_output'
    vocab_variant: str    = 'rbrics_nofall_nobpe_nofilter'
    fold: int             = 0
    processed_root: str   = './processed'
    backbone: str         = 'GIN'
    node_encoder: str     = 'onehot'
    hidden_dim: int       = 64
    num_layers: int       = 3
    apply_layer_norm: bool = False
    conv_normalize: str = 'l2'
    gin_inner_bn: bool = True
    dropout: float        = 0.5
    epochs: int           = 100
    lr: float             = 1e-3
    weight_decay: float   = 1e-5
    batch_size: int       = 128
    patience: int         = 20
    seed: int             = 42
    out_dir: str          = './vanilla_results'
    verbose: bool         = True
    run_gnnexplainer: bool = True
    run_pgexplainer: bool  = True
    run_mage: bool         = True
    run_motif_impact: bool = True
    max_motifs_eval: Optional[int] = None
    load_weights_from: Optional[str] = None  # dir containing best_model.pt
    weight_vocab_variant: Optional[str] = None  # vocab variant of loaded weights
    # When True, treat --out_dir as the FINAL run directory and do NOT append
    # <dataset>/fold<k>/<variant_tag>. Required by run_experiments.py / run_experiments.sh.
    final_out_dir: bool = False
    use_wandb: bool = False
    wandb_project: str = 'ChemIntuit'
    wandb_entity: Optional[str] = None
    # Synthetic ground-truth (Phase 4). When use_gt=True the train/valid/test
    # loaders are swapped for the rule-relabelled caches written by
    # SharedModules/data/apply_gt.py, exactly like MOSE-GNN / MotifSAT. This
    # makes the test graphs carry node_label/edge_label so the post-hoc
    # explainers can be scored with GT-ROC.
    use_gt: bool = False
    gt_cache: Optional[str] = None

    # mutag motif-annotation artifacts (optional overrides; default to the
    # conventional {data_root}/mutag_{fold}.csv + _index_maps.pkl paths).
    mutag_index_maps_path: Optional[str] = None
    mutag_smiles_csv_path: Optional[str] = None
    mutag_splits_path: Optional[str] = None
    mutag_seed: int = 42

    def variant_tag(self) -> str:
        enc = self.node_encoder
        # Effective inter-layer norm (none|l2|layernorm). apply_layer_norm=True
        # forces layernorm. Encoded so none/l2/layernorm runs never collide.
        _norm = 'layernorm' if self.apply_layer_norm else getattr(self, 'conv_normalize', 'l2')
        # NOTE: epochs is deliberately NOT in the tag. A baseline/explainer run
        # uses --epochs 0 to LOAD the checkpoint trained at epochs>0; if epochs
        # were in the tag the load would look in the wrong directory.
        # Synthetic-GT marker so GT and real-label runs never share a dir/
        # checkpoint (mirrors MOSE/MotifSAT config tags). Kept in lockstep with
        # the _ckpt_tag built in run().
        _gt = 'gt' if getattr(self, 'use_gt', False) else 'real'
        try:
            from SharedModules.data.loader import hp_suffix
            hp = hp_suffix(self)
        except Exception:
            hp = ''
        base = f'{self.backbone}_{enc}_norm-{_norm}_{_gt}_{self.vocab_variant}'
        return f'{base}_{hp}' if hp else base

    def to_dict(self) -> Dict:
        return asdict(self)

    # Unused — no --config CLI (unlike MOSE/MotifSAT).
    # def save(self, path: str) -> None:
    #     import yaml
    #     Path(path).parent.mkdir(parents=True, exist_ok=True)
    #     with open(path, 'w') as f:
    #         yaml.dump(self.to_dict(), f)
    #
    # @classmethod
    # def from_yaml(cls, path: str) -> 'VanillaConfig':
    #     import yaml
    #     with open(path) as f:
    #         d = yaml.safe_load(f)
    #     return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def run(cfg: VanillaConfig) -> dict:
    from SharedModules.data.dataset_routing import (
        default_processed_base,
        validate_use_gt,
        training_summary_extras,
        variant_processed_root,
    )

    validate_use_gt(cfg.dataset, cfg.use_gt, cfg.gt_cache)
    set_seed(cfg.seed)
    device = get_device()

    print(f'\n{"="*60}')
    print(f'  VanillaGNN  {cfg.dataset}  fold={cfg.fold}')
    print(f'{"="*60}')

    # Vocabulary (needed for motif-level evaluation)
    vocab = load_vocab(cfg.vocab_root, cfg.dataset, cfg.vocab_variant)
    print(f'  {vocab.num_motifs} motifs')

    task_type = TASK_TYPE.get(cfg.dataset, 'BinaryClass')
    loaders, test_ds, meta = get_loaders(
        dataset=cfg.dataset, data_root=cfg.data_root, fold=cfg.fold,
        vocab=vocab, processed_root=cfg.processed_root,
        batch_size=cfg.batch_size, normalize=(task_type == 'Regression'),
        mutag_index_maps_path=getattr(cfg, 'mutag_index_maps_path', None),
        mutag_smiles_csv_path=getattr(cfg, 'mutag_smiles_csv_path', None),
        mutag_splits_path=getattr(cfg, 'mutag_splits_path', None),
        mutag_seed=getattr(cfg, 'mutag_seed', 42),
    )

    # ── GT loader replacement (use_gt=True: synthetic rule labels) ─────────────
    # Swap all three loaders for the apply_gt.py caches so the baseline trains
    # on/eval against the same rule target as MOSE-GNN / MotifSAT, and the test
    # graphs carry node_label/edge_label for post-hoc explainer GT-ROC.
    if getattr(cfg, 'use_gt', False) and getattr(cfg, 'gt_cache', None):
        loaders, test_ds = apply_gt_loaders(
            loaders, test_ds,
            gt_cache=cfg.gt_cache, dataset=cfg.dataset, fold=cfg.fold,
            vocab_variant=cfg.vocab_variant, batch_size=cfg.batch_size,
        )

    from SharedModules.data.loader import NUM_CLASSES, resolve_node_encoder
    num_classes = NUM_CLASSES.get(cfg.dataset, 1)
    # Resolve the encoder ONCE (honors CLI for CSV; forces atom_encoder for OGB)
    # and store it back on cfg so the output dir tag, checkpoint tag, and the
    # model all use the SAME value — otherwise a baseline (epochs=0) run could
    # look for a checkpoint under a different tag than training wrote.
    cfg.node_encoder = resolve_node_encoder(getattr(cfg, 'node_encoder', None),
                                            meta.node_encoder)
    tag = cfg.variant_tag()   # computed AFTER resolution so the dir reflects reality
    print(f'  variant: {tag}')
    _node_encoder = cfg.node_encoder
    model = VanillaGNN(
        x_dim=meta.x_dim, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
        backbone=cfg.backbone, node_encoder=_node_encoder,
        apply_layer_norm=cfg.apply_layer_norm, dropout=cfg.dropout,
        conv_normalize=getattr(cfg,'conv_normalize','l2'),
        gin_inner_bn=getattr(cfg,'gin_inner_bn',True),
        num_classes=num_classes,
        deg=meta.deg,   # degree histogram for PNA; None for GIN/GCN/SAGE/GAT
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  VanillaGNN  params={n_params:,}')

    pos_w = (compute_pos_weights(loaders['train'].dataset)
             if task_type in ('BinaryClass', 'MultiLabel') else None)

    if getattr(cfg, 'final_out_dir', False):
        out_dir = Path(cfg.out_dir)            # launcher already encoded ds/fold/cfg
    else:
        out_dir = Path(cfg.out_dir) / cfg.dataset / f'fold{cfg.fold}' / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    from SharedModules.data.loader import write_hparams as _wh; _wh(out_dir, cfg)

    # When load_weights_from is set, resolve the checkpoint path from that dir.
    # The checkpoint was saved with the weight variant's tag (which may differ
    # from the eval variant when running post-hoc baselines under a filtered vocab).
    _ckpt_variant = (cfg.weight_vocab_variant
                     if cfg.weight_vocab_variant
                     else cfg.vocab_variant)
    # Checkpoint tag MUST match VanillaConfig.variant_tag() exactly (the dir the
    # training run wrote to), except the vocab variant may differ when a post-hoc
    # baseline evaluates under a filtered vocab but loads weights trained on the
    # unfiltered one. Keep this in lockstep with variant_tag().
    _norm = 'layernorm' if cfg.apply_layer_norm else getattr(cfg, 'conv_normalize', 'l2')
    _gt = 'gt' if getattr(cfg, 'use_gt', False) else 'real'
    try:
        from SharedModules.data.loader import hp_suffix as _hp_suffix
        _hp = _hp_suffix(cfg)
    except Exception:
        _hp = ''
    _ckpt_tag = f'{cfg.backbone}_{cfg.node_encoder}_norm-{_norm}_{_gt}_{_ckpt_variant}'
    if _hp:
        _ckpt_tag = f'{_ckpt_tag}_{_hp}'
    if not cfg.load_weights_from:
        _ckpt_dir = out_dir
    elif getattr(cfg, 'final_out_dir', False):
        # FINAL layout: --load_weights_from IS the trained vanilla run dir, which
        # contains best_model.pt directly (no <ds>/fold/<tag> re-append). If a
        # best_model.pt path was passed, use its parent.
        _lw = Path(cfg.load_weights_from)
        _ckpt_dir = _lw.parent if _lw.name == 'best_model.pt' else _lw
    else:
        _ckpt_dir = Path(cfg.load_weights_from) / cfg.dataset / f'fold{cfg.fold}' / _ckpt_tag

    model = train_vanilla_gnn(
        model, loaders, task_type, device,
        epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
        pos_weights=pos_w, patience=cfg.patience,
        save_path=str(_ckpt_dir / 'best_model.pt'), verbose=cfg.verbose,
    )
    # Baseline runs (epochs=0) load vanilla weights; mirror ckpt into out_dir so
    # summary.json and best_model.pt live together (required by regenerate_eval).
    if (cfg.epochs == 0 and getattr(cfg, 'final_out_dir', False)
            and cfg.load_weights_from):
        _run_ckpt = out_dir / 'best_model.pt'
        _src_ckpt = _ckpt_dir / 'best_model.pt'
        if _run_ckpt.resolve() != _src_ckpt.resolve():
            torch.save(model.state_dict(), _run_ckpt)

    # ── Prediction performance on all splits ──────────────────────────────────
    # For regression, also report MAE/RMSE in the original target units
    # (denormalised via the train z-score std).
    _denorm = ((meta.norm_mean, meta.norm_std)
               if task_type == 'Regression' else None)
    all_preds = {}
    for split_name in ('train', 'valid', 'test'):
        m = evaluate_predictions(model, loaders[split_name], device, task_type,
                                 denorm=_denorm)
        all_preds[split_name] = m
        print(f'  {split_name}: {m}')

    # ── EvalPipeline: motif impact + correlation ──────────────────────────────
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
        model, vocab, loaders['test'], test_list, device, task_type,
        max_motifs_eval=cfg.max_motifs_eval,
        denorm=_denorm,
        gt_eval_list=_gt_eval,
        index_maps=_mutag_maps,
    )
    eval_results = pipeline.run(run_motif_impact=cfg.run_motif_impact)
    dfs = pipeline.to_dataframe(eval_results)
    for name, df in dfs.items():
        df.to_csv(out_dir / f'{name}.csv', index=False)

    results: Dict = {'prediction': all_preds, 'eval': eval_results}

    # ── Post-hoc explainers ───────────────────────────────────────────────────
    # GNNExplainer
    if cfg.run_gnnexplainer:
        try:
            print('\n  Running GNNExplainer ...')
            gnnex_scores = run_gnnexplainer(model, test_list, vocab, device, task_type)
            _save_explainer_scores(gnnex_scores, out_dir / 'gnnexplainer_motif_scores', vocab)
            results['gnnexplainer_mean'] = gnnex_scores.get('mean', {})
            results['gnnexplainer_max']  = gnnex_scores.get('max', {})
        except Exception as e:
            print(f'  [warn] GNNExplainer failed: {e}')

    # PGExplainer
    if cfg.run_pgexplainer:
        try:
            print('\n  Running PGExplainer ...')
            pgex_scores = run_pgexplainer(model, loaders, test_list, vocab, device, task_type)
            _save_explainer_scores(pgex_scores, out_dir / 'pgexplainer_motif_scores', vocab)
            results['pgexplainer_mean'] = pgex_scores.get('mean', {})
            results['pgexplainer_max']  = pgex_scores.get('max', {})
        except Exception as e:
            print(f'  [warn] PGExplainer failed: {e}')

    # MAGE
    if cfg.run_mage:
        try:
            print('\n  Running MAGE ...')
            mage_scores = run_mage(model, test_list, vocab, device, task_type)
            _save_explainer_scores(mage_scores, out_dir / 'mage_motif_scores', vocab)
            results['mage_mean'] = mage_scores.get('mean', {})
            results['mage_max']  = mage_scores.get('max', {})
        except Exception as e:
            print(f'  [warn] MAGE failed: {e}')

    # ── Summary JSON ──────────────────────────────────────────────────────────
    pred  = all_preds.get('test', {})
    corr  = eval_results.get('correlation', {})
    gt    = eval_results.get('gt_roc', {})
    gt_node = eval_results.get('gt_roc_node', {})
    gt_edge = eval_results.get('gt_roc_edge', {})

    # Per-explainer score-vs-impact correlation, top-motif discriminativeness,
    # and score distribution. The post-hoc explainer's attribution IS its motif
    # score, so we correlate each against the same mask-based impact / the
    # label-aware discriminativeness computed in eval_results.
    from SharedModules.evaluation.motif_eval import (
        score_impact_correlation, top_motifs_discriminative_check)
    from SharedModules.evaluation.metrics import motif_score_stats
    _impacts = eval_results.get('motif_impact', {})
    _disc    = eval_results.get('discriminativeness', {})
    _topk = cfg.top_k if hasattr(cfg, 'top_k') else 10
    explainer_metrics = {}
    # Each post-hoc explainer produces NODE-level attributions; we aggregate
    # them to motif level by both mean and max over the motif's atoms. Report
    # correlation/discriminativeness/score-stats for BOTH aggregations so the
    # baselines are directly comparable to the motif-aware models.
    for _ex in ('gnnexplainer', 'pgexplainer', 'mage'):
        for _agg in ('mean', 'max'):
            _sc = results.get(f'{_ex}_{_agg}', {})
            if not _sc:
                continue
            _pfx = f'{_ex}_{_agg}'   # e.g. gnnexplainer_mean, gnnexplainer_max
            if _impacts:
                _c = score_impact_correlation(_sc, _impacts)
                explainer_metrics[f'{_pfx}_pearson']  = _c.get('pearson', float('nan'))
                explainer_metrics[f'{_pfx}_spearman'] = _c.get('spearman', float('nan'))
            if _disc:
                _t = top_motifs_discriminative_check(_sc, _disc, k=_topk)
                explainer_metrics[f'{_pfx}_top_k_abs_disc']      = _t.get('top_k_abs_disc', float('nan'))
                explainer_metrics[f'{_pfx}_score_disc_spearman'] = _t.get('score_disc_spearman', float('nan'))
            _st = motif_score_stats(_sc)
            explainer_metrics[f'{_pfx}_score_mean'] = _st['score_mean']
            explainer_metrics[f'{_pfx}_score_std']  = _st['score_std']

    # ── Per-explainer GT-ROC (node & edge) ─────────────────────────────────────
    # The vanilla model has no intrinsic node attention, so its GT-ROC comes from
    # each post-hoc explainer: broadcast the explainer's per-motif score onto its
    # atoms (node_att[i] = score[nodes_to_motifs[i]]) and score that node
    # attribution against the synthetic GT, reusing compute_gt_roc's node_att_fn
    # path. The explainer already reduces its per-node mask to per-motif scores
    # by mean AND max, so we score BOTH aggregations (the agg IS the node→motif
    # reduction). Requires --use_gt so the test graphs carry node/edge labels.
    _gt_present = any(
        getattr(d, 'node_label', None) is not None
        or getattr(d, 'edge_label', None) is not None
        for d in (_gt_eval if _gt_eval is not None else test_list)
    )
    if _gt_present:
        _gt_roc_list = _gt_eval if _gt_eval is not None else test_list
        for _ex in ('gnnexplainer', 'pgexplainer', 'mage'):
            for _agg in ('mean', 'max'):
                _sc = results.get(f'{_ex}_{_agg}', {})
                if not _sc:
                    continue
                _fn = _motif_score_node_att_fn(_sc)
                _gn = compute_gt_roc(model, _gt_roc_list, device,
                                     node_att_fn=_fn, level='node')
                _ge = compute_gt_roc(model, _gt_roc_list, device,
                                     node_att_fn=_fn, level='edge')
                explainer_metrics[f'{_ex}_{_agg}_gt_roc_node_auc_mean'] = _gn['auc_mean']
                explainer_metrics[f'{_ex}_{_agg}_gt_roc_edge_auc_mean'] = _ge['auc_mean']

    summary = {
        'dataset':          cfg.dataset,
        'fold':             cfg.fold,
        'backbone':         cfg.backbone,
        'variant_tag':      tag,
        'vocab_variant':    cfg.vocab_variant,
        'node_encoder':     cfg.node_encoder,
        'apply_layer_norm': cfg.apply_layer_norm,
        'model_type':       'VanillaGNN',
        'motif_method':     'none',
        'auc':              pred.get('auc', pred.get('auc_mean', float('nan'))),
        'rmse':             pred.get('rmse', float('nan')),
        'mae':              pred.get('mae', float('nan')),
        # regression metrics in original target units (denormalised); NaN for
        # classification tasks
        'rmse_orig':        pred.get('rmse_orig', float('nan')),
        'mae_orig':         pred.get('mae_orig', float('nan')),
        'train_auc':        all_preds.get('train', {}).get('auc', float('nan')),
        'val_auc':          all_preds.get('valid', {}).get('auc', float('nan')),
        'pearson':          corr.get('pearson', float('nan')),
        'spearman':         corr.get('spearman', float('nan')),
        'gt_roc_auc_mean':  gt.get('auc_mean', float('nan')),
        'gt_roc_n_graphs': gt.get('n_graphs', 0),
        'gt_roc_node_auc_mean': gt_node.get('auc_mean', float('nan')),
        'gt_roc_edge_auc_mean': gt_edge.get('auc_mean', float('nan')),
        **training_summary_extras(cfg),
        **explainer_metrics,
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Optional W&B logging (only when --use_wandb is passed) ─────────────────
    # The vanilla baseline does not stream per-epoch metrics; it logs the final
    # summary so it appears alongside MOSE/MotifSAT runs in the same project.
    if getattr(cfg, 'use_wandb', False):
        try:
            import os as _os
            import wandb as _wandb
            if _wandb.run is None:
                _wb_kwargs = dict(
                    project=getattr(cfg, 'wandb_project', 'ChemIntuit'),
                    entity=getattr(cfg, 'wandb_entity', None),
                    name=f'{cfg.dataset}_fold{cfg.fold}_{tag}',
                    config=cfg.to_dict(),
                    reinit=True,
                )
                # HPC nodes usually block internet → the ONLINE client's polling
                # thread throws BrokenPipe. DEFAULT TO OFFLINE so crashes are
                # impossible; set WANDB_MODE=online to stream live.
                _mode = _os.environ.get('WANDB_MODE') or 'offline'
                try:
                    _wandb.init(mode=_mode, **_wb_kwargs)
                except Exception as _e:
                    print(f'  [warn] wandb init (mode={_mode}) failed ({_e}); '
                          f'retrying offline.')
                    _wandb.init(mode='offline', **_wb_kwargs)
            _wandb.log({k: v for k, v in summary.items()
                        if isinstance(v, (int, float))})
            _wandb.finish()
        except ImportError:
            print('  [warn] --use_wandb set but wandb is not installed; skipping')
        except Exception as _e:
            print(f'  [warn] wandb logging failed ({_e}); continuing without W&B.')

    return results


def _motif_score_node_att_fn(motif_scores: Dict[int, float]):
    """Build a node_att_fn for compute_gt_roc from per-motif scores.

    Broadcasts the per-motif score onto each atom via nodes_to_motifs:
    ``node_att[i] = motif_scores[nodes_to_motifs[i]]`` (0.0 for unassigned
    atoms, motif id < 0). This turns a post-hoc explainer's motif-level
    attribution into a per-node attribution comparable against the
    (motif-granular) synthetic GT.
    """
    def fn(data):
        n2m = getattr(data, 'nodes_to_motifs', None)
        n = data.x.size(0)
        if n2m is None:
            return torch.zeros(n, device=data.x.device)
        vals = [float(motif_scores.get(int(m), 0.0)) if int(m) >= 0 else 0.0
                for m in n2m.tolist()]
        return torch.tensor(vals, dtype=torch.float32, device=data.x.device)
    return fn


def _save_explainer_scores(
    scores: Dict[str, Dict[int, float]],
    stem: Path,
    vocab,
) -> None:
    """Save mean and max aggregation CSVs.

    scores : {'mean': {motif_id: float}, 'max': {motif_id: float}}
    stem   : base path without extension — writes stem_mean.csv and stem_max.csv
    """
    import pandas as pd
    motif_list = getattr(vocab, 'motif_list', [])
    for agg in ('mean', 'max'):
        agg_scores = scores.get(agg, {})
        rows = [
            {
                'motif_id':      mid,
                f'score_{agg}':  s,
                'motif_smarts':  motif_list[mid] if mid < len(motif_list) else '?',
            }
            for mid, s in agg_scores.items()
        ]
        if rows:
            pd.DataFrame(rows).to_csv(
                Path(str(stem) + f'_{agg}.csv'), index=False)


def main():
    parser = argparse.ArgumentParser(description='VanillaGNN + post-hoc explainers')
    parser.add_argument('--dataset',         default='Mutagenicity')
    parser.add_argument('--fold',            type=int, default=0)
    parser.add_argument('--backbone',        default='GIN')
    parser.add_argument('--node_encoder',    default='onehot')
    parser.add_argument('--apply_layer_norm', action='store_true')
    parser.add_argument('--conv_normalize', default='l2', choices=['l2','layernorm','none'])
    parser.add_argument('--no_gin_inner_bn', dest='gin_inner_bn', action='store_false')
    parser.set_defaults(gin_inner_bn=True)
    parser.add_argument('--hidden_dim',      type=int, default=64)
    parser.add_argument('--num_layers',      type=int, default=3)
    parser.add_argument('--epochs',          type=int, default=100)
    parser.add_argument('--lr',              type=float, default=1e-3)
    parser.add_argument('--data_root',       default='./datasets/FOLDS')
    parser.add_argument('--vocab_root',      default='./vocab_output')
    parser.add_argument('--vocab_variant',   default='rbrics_nofall_nobpe_nofilter')
    parser.add_argument('--out_dir',         default='./vanilla_results')
    parser.add_argument('--processed_root',  default=os.environ.get('PROCESSED_ROOT'),
                        help='PyG .pt cache base ($PROCESSED_ROOT; per-vocab subdir appended)')
    parser.add_argument('--weight_vocab_variant', default=None,
                        help='Vocab variant used when training the weights to load. '
                             'Only needed with --load_weights_from when the eval vocab '
                             'differs from the vocab the model was trained with '
                             '(e.g. rbrics_old_filter eval, rbrics_old weights). '
                             'Defaults to --vocab_variant if not set.')
    parser.add_argument('--seed',            type=int, default=42)
    parser.add_argument('--load_weights_from', default=None,
                        help='Directory of a previous run to load best_model.pt from. '
                             'Used with --epochs 0 for post-hoc explainer evaluation.')
    parser.add_argument('--final_out_dir', action='store_true',
                        help='Treat --out_dir as the FINAL run dir (do not append '
                             '<dataset>/fold<k>/<variant_tag>). Set by the unified '
                             'launcher to avoid double dataset/fold nesting.')
    parser.add_argument('--no_gnnexplainer', action='store_true')
    parser.add_argument('--no_pgexplainer',  action='store_true')
    parser.add_argument('--no_mage',         action='store_true')
    parser.add_argument('--use_wandb',       action='store_true',
                        help='Initialise a W&B run and log the final summary.')
    parser.add_argument('--wandb_project',   default='ChemIntuit')
    parser.add_argument('--wandb_entity',    default=None)
    parser.add_argument('--use_gt',          action='store_true',
                        help='Load ground-truth relabelled graphs from gt_cache '
                             '(Phase 4) so post-hoc explainers get GT-ROC.')
    parser.add_argument('--gt_cache',        default=None,
                        help='Path to gt_cache directory written by phase4 '
                             '(SharedModules/data/apply_gt.py).')
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
    args = parser.parse_args()

    base_proc = default_processed_base(args.data_root, args.processed_root)
    proc_root = variant_processed_root(base_proc, args.vocab_variant)
    cfg = VanillaConfig(
        dataset=args.dataset, fold=args.fold, backbone=args.backbone,
        node_encoder=args.node_encoder, apply_layer_norm=args.apply_layer_norm,
        conv_normalize=args.conv_normalize, gin_inner_bn=args.gin_inner_bn,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        epochs=args.epochs, lr=args.lr, data_root=args.data_root,
        vocab_root=args.vocab_root, vocab_variant=args.vocab_variant,
        processed_root=proc_root,
        out_dir=args.out_dir, seed=args.seed,
        run_gnnexplainer=not args.no_gnnexplainer,
        run_pgexplainer=not args.no_pgexplainer,
        run_mage=not args.no_mage,
        load_weights_from=args.load_weights_from,
        weight_vocab_variant=args.weight_vocab_variant,
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
    )
    run(cfg)


if __name__ == '__main__':
    main()
