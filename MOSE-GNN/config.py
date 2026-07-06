"""config.py — MOSE-GNN configuration dataclass with YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional
import yaml


@dataclass
class MOSEConfig:
    # Data
    dataset: str = 'Mutagenicity'
    data_root: str = './datasets/FOLDS'
    vocab_root: str = './motifsat_output'
    vocab_variant: str = 'all_fallback_bpe'
    fold: int = 0
    processed_root: str = './processed'

    # Model
    backbone: str = 'GIN'                    # GIN | GAT | GCN | SAGE | PNA
    node_encoder: str = 'onehot'             # onehot | linear
    hidden_dim: int = 64
    num_layers: int = 3
    apply_layer_norm: bool = False
    conv_normalize: str = 'none'    # l2 | layernorm | none (per-conv norm; MOSE default none)
    gin_inner_bn: bool = True       # BatchNorm inside GIN MLP (Xu et al. design)
    self_gate: bool = False         # EXPERIMENTAL (off): gate GIN/SAGE self-term by node att
    dropout: float = 0.5
    w_feat: bool = True
    w_message: bool = False
    w_readout: bool = True

    # Motif importance
    unk_mode: str = 'fixed'                  # fixed | learnable_shared
    unk_value: float = 0.5

    # Training
    epochs: int = 150
    lr: float = 1e-3              # base LR (kept for backward compat / fallback)
    explainer_lr: float = 0.01   # LR for motif-importance params (the explainer)
    gnn_lr: float = 0.001        # LR for GNN backbone params
    weight_decay: float = 1e-5
    batch_size: int = 128
    size_reg: float = 0.0
    ent_reg: float = 0.01
    top_tau: int = 10
    ignore_unknowns: bool = False
    patience: int = 30
    min_epochs: int = 20
    clip_grad: float = 2.0
    seed: int = 42

    # Output
    out_dir: str = './mose_results'
    run_name: str = 'mose_run'
    verbose: bool = True
    # When True, treat out_dir as the FINAL run dir (no <ds>/fold/<tag> append).
    # Set by the unified launcher (run_experiments.py) to avoid double nesting.
    final_out_dir: bool = False

    # Evaluation
    run_motif_impact: bool = True
    max_motifs_eval: Optional[int] = None
    run_multi_explanation: bool = False
    # Eval-only mode: skip training, load weights, run the eval pipeline only.
    eval_only: bool = False
    load_weights_from: Optional[str] = None  # dir or path to best_model.pt

    # W&B
    use_wandb: bool = False
    wandb_project: str = 'ChemIntuit'
    wandb_entity: Optional[str] = None

    # Ground-truth relabelling (phase4)
    use_gt: bool = False        # load GT relabelled graphs from gt_cache
    gt_cache: Optional[str] = None  # path to gt_cache directory
    gt_vocab_variant: Optional[str] = None  # base variant for gt_cache lookup when training on *_filter

    # mutag motif-annotation artifacts (optional overrides; default to the
    # conventional {data_root}/mutag_{fold}.csv + _index_maps.pkl paths).
    mutag_index_maps_path: Optional[str] = None
    mutag_smiles_csv_path: Optional[str] = None
    mutag_splits_path: Optional[str] = None
    mutag_seed: int = 42

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str) -> 'MOSEConfig':
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, d: Dict) -> 'MOSEConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def variant_tag(self) -> str:
        """Short string uniquely identifying this configuration for output directory naming.

        Encodes all axes of variation so no two different configs can collide:
            backbone  - GIN / GCN / GAT / SAGE / PNA
            encoder   - onehot / linear
            ln        - noLN / LN
            injection - wf+wr, wm+wr, wf+wm+wr, etc.
            unk       - fixed / learn
            frag      - vocab_variant (fragmentation algorithm)
        """
        enc  = self.node_encoder            # onehot | linear
        # Inter-layer norm type (none|l2|layernorm). apply_layer_norm=True is a
        # back-compat alias that forces layernorm, so derive the effective value
        # the model will actually use, and put it in the tag so none/l2/layernorm
        # runs never collide.
        _norm = 'layernorm' if self.apply_layer_norm else getattr(self, 'conv_normalize', 'none')
        ln   = f'norm-{_norm}'
        inj  = '+'.join(filter(None, [
            'wf' if self.w_feat    else '',
            'wm' if self.w_message else '',
            'wr' if self.w_readout else '',
        ])) or 'noinj'
        unk  = f'unk-{self.unk_mode}'
        frag = self.vocab_variant           # e.g. rbrics_nofall_nobpe_nofilter
        gt   = 'gt' if getattr(self, 'use_gt', False) else 'real'
        ep   = f'ep{self.epochs}'
        try:
            from SharedModules.data.loader import hp_suffix
            hp = hp_suffix(self)
        except Exception:
            hp = ''
        base = f'{self.backbone}_{enc}_{ln}_{inj}_{unk}_{gt}_{ep}_{frag}'
        return f'{base}_{hp}' if hp else base
