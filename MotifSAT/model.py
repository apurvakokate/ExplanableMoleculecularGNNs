"""model.py — MotifSAT GSAT model.

Implements the motif_method choices × three noise levels ×
three info loss levels × three attention injection points.

motif_method  : none | loss | readout   (motif_emb -> NotImplementedError)
noise         : none | node | motif
info_loss_level: none | node | motif
w_feat / w_message / w_readout : bool flags (orthogonal to method)
learn_edge_att : bool (base GSAT path — edge attention instead of node)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import global_add_pool

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from SharedModules.models.gnn_base import BaseGNN
from SharedModules.data.dataset import NUM_ATOM_TYPES, EDGE_FEAT_DIM
from SharedModules.evaluation.metrics import evaluate_predictions

from motif_modules import (
    ExtractorMLP, MotifReadoutScorer, MotifPooling,
    compute_inverse_idx, lift_motif_to_node,
)
from losses import info_loss, motif_consistency_loss, motif_size_weights


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

LOGIT_CLAMP = 3.0   # clamp |ℓ| ≤ 3 before sigmoid; att ∈ [0.047, 0.953]
                    # Prevents saturation that kills IB gradient and Concrete sampler.


def _concrete_sample(log_logits: Tensor, r: float, training: bool) -> Tensor:
    """Concrete / soft-sigmoid sampling with logit clamping.

    Clamps |log_logits| to [-LOGIT_CLAMP, LOGIT_CLAMP] before sampling.
    Without clamping, MLP outputs can saturate to ±inf, collapsing
    sigmoid to 0/1 and killing IB gradient flow.

    During training: Concrete (Gumbel-sigmoid) sample.
    During eval: the deterministic soft sigmoid sigma(logits) — NOT a hard
    0/1 threshold. This matches reference GSAT (Miao et al., 2022): the
    attention is always a continuous gate in (0, 1). A hard threshold would
    discard the magnitude information that downstream ranking metrics
    (GT ROC, per-motif aggregation) and the readout injection rely on.
    """
    log_logits = log_logits.clamp(-LOGIT_CLAMP, LOGIT_CLAMP)
    if training:
        u = torch.empty_like(log_logits).uniform_().clamp(1e-6, 1 - 1e-6)
        log_u = u.log() - (1 - u).log()
        return torch.sigmoid((log_logits + log_u) / r)
    else:
        return log_logits.sigmoid()


def _add_logistic_noise(log_logits: Tensor) -> Tuple[Tensor, Tensor]:
    """Add logistic noise ε ~ Logistic(0,1) and return (noisy_logits, noise)."""
    u = torch.empty_like(log_logits).uniform_().clamp(1e-6, 1 - 1e-6)
    noise = u.log() - (1 - u).log()
    return log_logits + noise, noise


# ─────────────────────────────────────────────────────────────────────────────
# GSAT model
# ─────────────────────────────────────────────────────────────────────────────

class GSAT(nn.Module):
    """GSAT extended to chemical motifs.

    Parameters
    ----------
    x_dim, hidden_dim, num_layers, backbone, node_encoder, apply_layer_norm,
    dropout, deg, edge_dim
        Passed to BaseGNN backbone.
    num_classes : int
        1 (binary/regression) or C (multi-label).
    task_type : str
    backbone_name : str  (GIN | GAT | GCN | SAGE | PNA)

    motif_method : str
        'none'      — base GSAT, node-level attention only.
        'loss'      — node attention + motif consistency regularisation.
        'readout'   — MotifReadoutScorer: pool node embeddings to motif level
                      (max+mean pooling), score each motif, broadcast the motif
                      score back to its atoms.
        'motif_emb' — NOT IMPLEMENTED; raises NotImplementedError.

    noise : str
        'none' | 'node' | 'motif'

    info_loss_level : str
        'none' | 'node' | 'motif'

    w_feat, w_message, w_readout : bool  (attention injection flags)
    learn_edge_att : bool  (base GSAT edge-level attention; overrides motif_method)

    pool_mode : str  (pooling for 'readout'; default 'max_mean')
    extractor_hidden_mult : int
    extractor_dropout_p : float

    motif_info_size_normalize : bool
        Divide info loss by motif length when info_loss_level='motif'.

    init_r, final_r, decay_interval, decay_r : float / int
        Temperature schedule for Concrete sampling.

    info_loss_coef, motif_loss_coef : float
    between_motif_coef, within_node_coef : float  (consistency loss coefficients)
    """

    def __init__(
        self,
        x_dim: int = NUM_ATOM_TYPES,
        hidden_dim: int = 64,
        num_layers: int = 3,
        backbone_name: str = 'GIN',
        node_encoder: str = 'onehot',
        apply_layer_norm: bool = False,
        dropout: float = 0.5,
        num_classes: int = 1,
        task_type: str = 'BinaryClass',
        deg=None,
        edge_dim: Optional[int] = None,
        conv_normalize: str = 'l2',
        gin_inner_bn: bool = True,
        # ── Motif method ──
        motif_method: str = 'none',
        pool_mode: str = 'max_mean',
        extractor_hidden_mult: int = 2,
        extractor_dropout_p: float = 0.5,
        # ── Noise / IB ──
        noise: str = 'none',
        info_loss_level: str = 'node',
        motif_info_size_normalize: bool = False,
        # ── Injection flags ──
        w_feat: bool = False,
        w_message: bool = True,
        w_readout: bool = False,
        learn_edge_att: bool = False,
        # ── Temperature ──
        init_r: float = 0.9,
        final_r: float = 0.1,
        decay_interval: Optional[int] = None,
        decay_r: Optional[float] = None,
        # ── Loss coefficients ──
        info_loss_coef: float = 1.0,
        motif_loss_coef: float = 0.0,
        between_motif_coef: float = 0.0,
        within_node_coef: float = 0.0,
    ):
        super().__init__()

        if motif_method == 'motif_emb':
            raise NotImplementedError(
                "motif_method='motif_emb' is not implemented. Use 'readout' for "
                "the motif-pooling scorer, or 'none'/'loss' for node-level "
                "attention."
            )
        assert motif_method in ('none', 'loss', 'readout'), (
            f"unknown motif_method={motif_method!r}; "
            f"expected one of none | loss | readout"
        )
        assert noise in ('none', 'node', 'motif')
        assert info_loss_level in ('none', 'node', 'motif')

        self.motif_method = motif_method
        self.noise = noise
        self.info_loss_level = info_loss_level
        self.motif_info_size_normalize = motif_info_size_normalize
        self.w_feat = w_feat
        self.w_message = w_message
        self.w_readout = w_readout
        self.learn_edge_att = learn_edge_att
        self.init_r = init_r
        self.final_r = final_r
        self.decay_interval = decay_interval
        self.decay_r = decay_r
        self.info_loss_coef = info_loss_coef
        self.motif_loss_coef = motif_loss_coef
        self.between_motif_coef = between_motif_coef
        self.within_node_coef = within_node_coef
        self.task_type = task_type
        self.num_classes = num_classes

        # Backbone
        self.clf = BaseGNN(
            x_dim=x_dim, hidden_dim=hidden_dim, num_layers=num_layers,
            backbone=backbone_name, node_encoder=node_encoder,
            apply_layer_norm=apply_layer_norm, dropout=dropout,
            deg=deg, edge_dim=edge_dim,
            conv_normalize=conv_normalize, gin_inner_bn=gin_inner_bn,
        )
        self.clf.lin2 = nn.Linear(hidden_dim, num_classes)

        # Extractor (node-level)
        self.extractor = ExtractorMLP(hidden_dim, extractor_hidden_mult,
                                      extractor_dropout_p)

        # Motif-level modules
        if motif_method == 'readout':
            self.motif_scorer = MotifReadoutScorer(
                in_dim=hidden_dim,
                pool_mode=pool_mode,
                hidden_mult=extractor_hidden_mult,
                dropout_p=extractor_dropout_p,
            )
        else:
            self.motif_scorer = None

        # Edge attention extractor (learn_edge_att=True path)
        if learn_edge_att:
            self.edge_extractor = ExtractorMLP(
                hidden_dim * 2, extractor_hidden_mult, extractor_dropout_p
            )
        else:
            self.edge_extractor = None

        # Current temperature (updated by anneal_r)
        self.register_buffer('r', torch.tensor(float(init_r)))

    # ── Temperature annealing ────────────────────────────────────────────────

    def anneal_r(self, epoch: int) -> None:
        """Update temperature r based on decay schedule."""
        if self.decay_interval is not None and self.decay_r is not None:
            new_r = max(
                self.final_r,
                self.init_r - self.decay_r * (epoch // self.decay_interval)
            )
        else:
            new_r = self.init_r
        self.r.fill_(new_r)

    # ── Attention computation ────────────────────────────────────────────────

    def _get_node_logits(
        self,
        node_emb: Tensor,
        nodes_to_motifs: Optional[Tensor],
        batch: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Compute node-level attention log-logits.

        Returns (node_log_logits [N,1], motif_log_logits [M,1] or None,
                 inverse_indices [N] or None).
        """
        if self.learn_edge_att:
            # Edge attention: will be computed in forward; return stub
            return self.extractor(node_emb), None, None

        if self.motif_method == 'readout' and nodes_to_motifs is not None:
            inv_idx, motif_batch, _ = compute_inverse_idx(
                nodes_to_motifs, batch)
            motif_logits, node_logits = self.motif_scorer(node_emb, inv_idx)
            return node_logits, motif_logits, inv_idx

        else:
            return self.extractor(node_emb), None, None

    def _apply_noise(
        self,
        node_logits: Tensor,
        motif_logits: Optional[Tensor],
        inv_idx: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Add Logistic noise at node or motif level.

        Logits are clamped to [-LOGIT_CLAMP, LOGIT_CLAMP] before noise
        so that noise is added on a controlled scale rather than on top
        of already-saturated values.
        """
        node_logits = node_logits.clamp(-LOGIT_CLAMP, LOGIT_CLAMP)
        if motif_logits is not None:
            motif_logits = motif_logits.clamp(-LOGIT_CLAMP, LOGIT_CLAMP)
        if self.noise == 'node':
            node_logits, _ = _add_logistic_noise(node_logits)
        elif self.noise == 'motif' and motif_logits is not None:
            motif_logits, noise = _add_logistic_noise(motif_logits)
            # Broadcast motif noise to nodes
            node_logits = node_logits + lift_motif_to_node(noise, inv_idx)
        return node_logits, motif_logits

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
        nodes_to_motifs: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        epoch: int = 0,
        motif_lengths: Optional[list] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Dict]:
        """Forward pass.

        Returns (logits [B, C], node_att [N, 1] or None, aux_dict)

        aux_dict contains tensors needed for loss computation:
          node_att, motif_att (if applicable), inv_idx, motif_batch
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        r = float(self.r.item())

        # Step 1: Backbone embedding (no attention injection yet)
        _, node_emb = self.clf.get_embedding(
            x, edge_index, edge_attr=edge_attr, batch=batch
        )

        # Step 2: Extractor → log-logits
        node_logits, motif_logits, inv_idx = self._get_node_logits(
            node_emb, nodes_to_motifs, batch
        )

        # Step 3: Noise injection
        if self.training:
            node_logits, motif_logits = self._apply_noise(
                node_logits, motif_logits, inv_idx
            )

        # Step 4: Edge attention (learn_edge_att path)
        edge_att = None
        # node_att / edge_att are soft gates in (0,1): at eval the deterministic
        # sigmoid, at train the Concrete (Gumbel-sigmoid) sample. We ALSO keep
        # the clean (noise-free) sigmoid as *_soft for ranking metrics, since at
        # train time the sampled att carries injected logistic noise that would
        # add variance to ROC / motif-aggregation scores.
        node_att_soft = None
        edge_att_soft = None
        if self.learn_edge_att:
            src, dst = edge_index
            edge_logits = self.edge_extractor(
                torch.cat([node_emb[src], node_emb[dst]], dim=-1)
            )
            edge_logits = edge_logits.clamp(-LOGIT_CLAMP, LOGIT_CLAMP)
            if self.training:
                edge_logits, _ = _add_logistic_noise(edge_logits)
            edge_att = _concrete_sample(edge_logits, r, self.training)
            edge_att_soft = edge_logits.sigmoid()
            node_att = None
        else:
            node_att = _concrete_sample(node_logits, r, self.training)
            node_att_soft = node_logits.sigmoid()

        # Step 5: Re-run backbone with attention injection
        graph_emb, node_emb_final = self.clf.get_embedding(
            x, edge_index,
            edge_attr=edge_attr,
            node_att=node_att,
            edge_atten=edge_att,
            w_feat=self.w_feat,
            w_message=self.w_message,
            w_readout=self.w_readout,
            batch=batch,
        )
        logits = self.clf.classify(graph_emb)

        # Collect aux info for loss
        aux = {
            'node_att':      node_att,
            'edge_att':      edge_att,
            'node_att_soft': node_att_soft,   # continuous probs for ranking/ROC
            'edge_att_soft': edge_att_soft,
            'node_logits':   node_logits,
            'motif_logits':  motif_logits,
            'inv_idx':       inv_idx,
            'r':             r,
        }
        return logits, node_att, aux

    # ── Loss computation ──────────────────────────────────────────────────────

    def compute_loss(
        self,
        task_loss: Tensor,
        aux: Dict,
        nodes_to_motifs: Optional[Tensor],
        batch: Optional[Tensor],
        motif_lengths: Optional[list] = None,
    ) -> Tuple[Tensor, Dict]:
        """Compute total loss = task + IB + consistency.

        Returns (total_loss, loss_breakdown_dict).
        """
        breakdown: Dict[str, float] = {'task': float(task_loss.item())}
        total = task_loss

        r = aux['r']

        # ── IB (information) loss ─────────────────────────────────────────
        if self.info_loss_level != 'none' and self.info_loss_coef > 0:
            if self.learn_edge_att and aux['edge_att'] is not None:
                ib = info_loss(aux['edge_att'].view(-1), r)
            elif self.info_loss_level == 'motif' and aux['motif_logits'] is not None:
                motif_att = aux['motif_logits'].sigmoid().view(-1)
                sw = None
                if self.motif_info_size_normalize and motif_lengths is not None \
                        and aux['inv_idx'] is not None:
                    # motif-level size weights: use motif vocab ids
                    from motif_modules import compute_inverse_idx
                    # Use motif_lengths at motif-instance level
                    # Approximate: size_weight for motif instance = weight of vocab motif
                    # inv_idx maps nodes → motif rows; motif vocab ids from compute_inverse_idx
                    n = nodes_to_motifs.size(0) if nodes_to_motifs is not None else 0
                    node_sw = motif_size_weights(
                        nodes_to_motifs if nodes_to_motifs is not None
                        else torch.zeros(n, dtype=torch.long),
                        motif_lengths
                    )
                    try:
                        from torch_scatter import scatter_mean as _sm
                    except ImportError:
                        from torch_geometric.utils import scatter as _tg
                        def _sm(src, index, dim=0, dim_size=None):
                            return _tg(src, index, dim=dim, dim_size=dim_size, reduce="mean")
                    sw = _sm(node_sw, aux['inv_idx'], dim=0,
                             dim_size=int(aux['inv_idx'].max().item()) + 1)
                ib = info_loss(motif_att, r, sw)
            elif aux['node_att'] is not None:
                node_att_v = aux['node_att'].view(-1)
                sw = None
                if self.motif_info_size_normalize and motif_lengths is not None \
                        and nodes_to_motifs is not None:
                    sw = motif_size_weights(nodes_to_motifs, motif_lengths)
                ib = info_loss(node_att_v, r, sw)
            else:
                ib = torch.tensor(0.0, device=task_loss.device)

            total = total + self.info_loss_coef * ib
            breakdown['info_loss'] = float(ib.item())

        # ── Motif consistency loss ────────────────────────────────────────
        if (self.motif_loss_coef > 0 or self.between_motif_coef > 0 or
                self.within_node_coef > 0):
            if (aux['node_att'] is not None and nodes_to_motifs is not None
                    and batch is not None):
                within_v, between_v = motif_consistency_loss(
                    aux['node_att'], nodes_to_motifs, batch
                )
                cons = (self.within_node_coef * within_v
                        - self.between_motif_coef * between_v)
                total = total + self.motif_loss_coef * cons
                breakdown['within_var'] = float(within_v.item())
                breakdown['between_var'] = float(between_v.item())

        breakdown['total'] = float(total.item())
        return total, breakdown
