"""config.py — MotifSAT configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional
import yaml


@dataclass
class MotifSATConfig:
    # Data
    dataset: str = 'Mutagenicity'
    data_root: str = './datasets/FOLDS'
    vocab_root: str = './motifsat_output'
    vocab_variant: str = 'all_fallback_bpe'
    fold: int = 0
    processed_root: str = './processed'

    # GNN backbone
    backbone: str = 'GIN'              # GIN | GAT | GCN | SAGE | PNA
    node_encoder: str = 'onehot'       # onehot | linear | atom_encoder (ogbg)
    hidden_dim: int = 64
    num_layers: int = 3
    apply_layer_norm: bool = False
    conv_normalize: str = 'l2'      # l2 | layernorm | none (per-conv norm)
    gin_inner_bn: bool = True       # BatchNorm inside GIN MLP (Xu et al. design)
    dropout: float = 0.5

    # Motif method (orthogonal to noise and info loss)
    motif_method: str = 'none'         # none | loss | node_emb | motif_emb | readout
    pool_mode: str = 'mean'            # mean | max | max_mean | multi
    extractor_hidden_mult: int = 2
    extractor_dropout_p: float = 0.5

    # Noise
    noise: str = 'none'                # none | node | motif

    # IB info loss
    info_loss_level: str = 'node'      # none | node | motif
    motif_info_size_normalize: bool = False
    info_loss_coef: float = 1.0
    motif_loss_coef: float = 0.0
    between_motif_coef: float = 0.0
    within_node_coef: float = 0.0

    # Attention injection
    w_feat: bool = False
    w_message: bool = True
    w_readout: bool = False
    learn_edge_att: bool = False       # True = base GSAT (edge attention)

    # Temperature schedule
    init_r: float = 0.9
    final_r: float = 0.1
    decay_interval: Optional[int] = 10
    decay_r: Optional[float] = 0.1
    logit_clamp: float = 3.0        # |ℓ| ≤ clamp before sigmoid; 3.0 = original paper

    # Training
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    patience: int = 20
    min_epochs: int = 20
    clip_grad: float = 2.0
    seed: int = 42

    # Output
    out_dir: str = './motifsat_results'
    run_name: str = 'motifsat_run'
    verbose: bool = True
    # When True, treat out_dir as the FINAL run dir (no <ds>/fold/<tag> append).
    # Set by the unified launcher (run_experiments.py) to avoid double nesting.
    final_out_dir: bool = False

    # Evaluation
    run_motif_impact: bool = True
    max_motifs_eval: Optional[int] = None
    run_multi_explanation: bool = False
    # Eval-only mode: skip training, load weights, run eval pipeline only.
    eval_only: bool = False
    load_weights_from: Optional[str] = None  # dir or path to best_model.pt

    # Ground-truth relabelling (phase4)
    # When use_gt=True, the GT-annotated train/valid/test sets are loaded from
    # gt_cache and replace all three loaders. The model trains to predict the
    # rule-derived synthetic label (data.y), not the original activity label.
    use_gt: bool = False
    gt_cache: Optional[str] = None

    # W&B
    use_wandb: bool = False
    wandb_project: str = 'ChemIntuit'
    wandb_entity: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str) -> 'MotifSATConfig':
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, d: Dict) -> 'MotifSATConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def variant_tag(self) -> str:
        """Unique tag encoding all axes of variation — no two different configs can collide."""
        enc  = self.node_encoder
        _norm = 'layernorm' if self.apply_layer_norm else getattr(self, 'conv_normalize', 'l2')
        ln   = f'norm-{_norm}'
        inj  = '+'.join(filter(None, [
            'wf' if self.w_feat    else '',
            'wm' if self.w_message else '',
            'wr' if self.w_readout else '',
        ])) or 'noinj'
        if self.learn_edge_att:
            inj = 'edge'
        noise_str = f'noise-{self.noise}'
        il_str    = f'il-{self.info_loss_level}'
        frag      = self.vocab_variant
        gt        = 'gt' if getattr(self, 'use_gt', False) else 'real'
        ep        = f'ep{self.epochs}'
        try:
            from SharedModules.data.loader import hp_suffix
            hp = hp_suffix(self)
        except Exception:
            hp = ''
        base = f'{self.backbone}_{self.motif_method}_{enc}_{ln}_{inj}_{noise_str}_{il_str}_{gt}_{ep}_{frag}'
        return f'{base}_{hp}' if hp else base
