"""model.py — MotifSAT GSAT model.

motif_method  : none | loss | readout
noise         : none | node | motif  (where Concrete stochasticity is applied)
    none  — per-node extractor logits; sample independently per node (base GSAT).
    node  — motif_emb → MLP → motif logits broadcast to nodes; sample per node.
    motif — motif_emb → MLP → sample once per motif; broadcast att to nodes.
info_loss_level: none | node | motif
w_feat / w_message / w_readout : attention injection flags
learn_edge_att : bool (edge-level GSAT extractor)
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

CONCRETE_TEMP = 1.0  # official GSAT: fixed temp=1 for Concrete sampling (r is IB-only)


def _clamp_logits(log_logits: Tensor, logit_clamp: Optional[float]) -> Tensor:
    """Optional |ℓ| clamp before sigmoid / Concrete (off when logit_clamp is None/≤0)."""
    if logit_clamp is not None and logit_clamp > 0:
        return log_logits.clamp(-logit_clamp, logit_clamp)
    return log_logits


def _concrete_sample(
    log_logits: Tensor,
    training: bool,
    temp: float = CONCRETE_TEMP,
    logit_clamp: Optional[float] = None,
    deterministic: bool = False,
) -> Tensor:
    """Concrete / soft-sigmoid sampling (official GSAT temp=1).

    When ``deterministic=True``, always use soft sigmoid (no Gumbel draw), even
    during training. Default ``deterministic=False`` matches official GSAT.
    """
    log_logits = _clamp_logits(log_logits, logit_clamp)
    if training and not deterministic:
        u = torch.empty_like(log_logits).uniform_().clamp(1e-6, 1 - 1e-6)
        log_u = u.log() - (1 - u).log()
        return torch.sigmoid((log_logits + log_u) / temp)
    else:
        return log_logits.sigmoid()


def _reorder_like(from_edge_index: Tensor, to_edge_index: Tensor,
                  values: Tensor) -> Tensor:
    """Align edge values from one edge_index ordering to another (GSAT util)."""
    from torch_geometric.utils import sort_edge_index
    from_edge_index, values = sort_edge_index(from_edge_index, values)
    ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
    ranking = ranking_score.argsort().argsort()
    if not (from_edge_index[:, ranking] == to_edge_index).all():
        raise ValueError("Edge index mismatch in _reorder_like.")
    return values[ranking]


def _symmetrize_edge_att(edge_index: Tensor, edge_att: Tensor) -> Tensor:
    """Average (u,v) and (v,u) edge attention on undirected graphs (official GSAT)."""
    from torch_geometric.utils import is_undirected
    if not is_undirected(edge_index):
        return edge_att
    att = edge_att.view(-1)
    try:
        from torch_sparse import transpose
    except ImportError as e:
        # Fail loudly: previously this silently returned the UN-symmetrized
        # edge attention, so an undirected-graph edge-GSAT run (learn_edge_att=
        # True) would train with ASYMMETRIC per-direction attention — a
        # different model — with no warning. Raise instead. Only the edge path
        # on undirected graphs reaches here; node/motif paths never call this.
        raise ImportError(
            "_symmetrize_edge_att requires torch_sparse to average edge "
            "attention over (u,v)/(v,u) on undirected graphs, but it is not "
            "installed. Refusing to silently train an asymmetric edge-attention "
            "model (learn_edge_att=True). Install torch_sparse, or use a "
            "node/motif attention path instead."
        ) from e
    trans_idx, trans_val = transpose(edge_index, att, None, None, coalesced=False)
    trans_val_perm = _reorder_like(trans_idx, edge_index, trans_val)
    return ((att + trans_val_perm) / 2).view_as(edge_att)


def _uses_motif_scorer(motif_method: str, noise: str) -> bool:
    return motif_method == 'readout' or noise in ('node', 'motif')


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
        'none'  — node-level extractor; independent Concrete sample per node.
        'node'  — motif pool → MLP → broadcast logits → sample per node.
        'motif' — motif pool → MLP → Concrete sample per motif → broadcast att.

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
        IB prior-retention schedule for info_loss (official GSAT annealing).
        Concrete sampling uses fixed temp=CONCRETE_TEMP (1.0), not r.

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
        conv_normalize: str = 'none',
        gin_inner_bn: bool = True,
        graph_pool: str = 'add',
        # ── Motif method ──
        motif_method: str = 'none',
        pool_mode: str = 'max_mean',
        extractor_hidden_mult: int = 2,
        extractor_dropout_p: float = 0.5,
        motif_scorer_norm: Optional[str] = None,  # motif scorer norm: instance|layer|none (REQUIRED for scorer runs; no default)
        # ── Noise / IB ──
        noise: str = 'none',
        info_loss_level: str = 'node',
        motif_info_size_normalize: bool = False,
        # ── Injection flags ──
        w_feat: bool = False,
        w_message: bool = True,
        w_readout: bool = False,
        learn_edge_att: bool = False,
        # ── IB prior retention (info_loss only) ──
        init_r: float = 0.9,
        final_r: float = 0.5,
        decay_interval: Optional[int] = None,
        decay_r: Optional[float] = None,
        # ── Loss coefficients ──
        info_loss_coef: float = 1.0,
        motif_loss_coef: float = 0.0,
        between_motif_coef: float = 0.0,
        within_node_coef: float = 0.0,
        logit_clamp: Optional[float] = None,
        deterministic_att: bool = False,
        self_gate: bool = False,
    ):
        super().__init__()

        if motif_method == 'motif_emb':
            raise NotImplementedError(
                "motif_method='motif_emb' is not implemented. Use 'readout' for "
                "the motif-pooling scorer, or 'none'/'loss' for node-level "
                "attention."
            )
        # Explicit raises (not assert) so validation survives `python -O`, where
        # asserts are stripped — otherwise a removed method like 'node_emb' would
        # silently fall back to the base-GSAT extractor path.
        if motif_method not in ('none', 'loss', 'readout'):
            raise ValueError(
                f"unknown motif_method={motif_method!r}; "
                f"expected one of none | loss | readout"
            )
        if noise not in ('none', 'node', 'motif'):
            raise ValueError(f"unknown noise={noise!r}; expected none | node | motif")
        if info_loss_level not in ('none', 'node', 'motif'):
            raise ValueError(
                f"unknown info_loss_level={info_loss_level!r}; "
                f"expected none | node | motif"
            )
        if learn_edge_att and noise != 'none':
            raise ValueError(
                f"learn_edge_att=True is incompatible with noise={noise!r}; "
                f"use noise='none' for the edge-attention GSAT path."
            )
        if learn_edge_att and not w_message:
            raise ValueError(
                "learn_edge_att=True requires w_message=True: the learned edge "
                "attention is injected only through the message-passing channel "
                "(BaseGNN builds edge_atten only when w_message is set). With "
                "w_message=False the edge extractor would train but never affect "
                "the prediction (silent no-op)."
            )
        # The whole motif-consistency term is scaled by motif_loss_coef:
        #   total += motif_loss_coef * (within_node_coef*within_var - between_motif_coef*between_var)
        # so between_motif_coef / within_node_coef have ZERO effect while motif_loss_coef == 0.
        # Guard the silent no-op (a coef that is set but never acts).
        if (between_motif_coef > 0 or within_node_coef > 0) and motif_loss_coef == 0:
            raise ValueError(
                f"between_motif_coef={between_motif_coef} / within_node_coef={within_node_coef} have "
                f"NO effect while motif_loss_coef=0 — the consistency loss is scaled by motif_loss_coef "
                f"(total += motif_loss_coef * (within_node_coef*within_var - between_motif_coef*"
                f"between_var)). Set motif_loss_coef > 0 to activate them (the effective weights are "
                f"motif_loss_coef*within_node_coef and motif_loss_coef*between_motif_coef), or leave "
                f"both sub-coefs at 0."
            )
        # Fail loudly: info_loss_level='motif' needs a motif scorer, otherwise
        # compute_loss SILENTLY falls back to the node-level info loss (motif_logits
        # is None). No silent degradation.
        if (info_loss_level == 'motif' and not learn_edge_att
                and not _uses_motif_scorer(motif_method, noise)):
            raise ValueError(
                f"info_loss_level='motif' requires a motif scorer "
                f"(motif_method='readout' or noise in {{'node','motif'}}); got "
                f"motif_method={motif_method!r} noise={noise!r}. Without one, the "
                f"motif info loss would silently degrade to node-level. Enable "
                f"readout/motif-noise, or use info_loss_level='node'."
            )
        # Fail loudly: no injection channel AND no edge path => the sampled
        # attention is never applied in the 2nd forward pass, so it gets no task
        # gradient and the explanation is inert (the model reduces to a plain
        # backbone). Same class of silent no-op as the learn_edge_att/w_message
        # guard above, so it raises too (was previously only a run.py warning).
        if not learn_edge_att and not (w_feat or w_message or w_readout):
            raise ValueError(
                "no attention injection enabled: w_feat/w_message/w_readout are "
                "all False and learn_edge_att is False, so the extractor "
                "attention is never applied and receives no task gradient (inert "
                "explanation). Enable at least one of w_feat/w_message/w_readout, "
                "or use learn_edge_att=True for the edge-attention path."
            )
        # Fail loudly: a partial decay schedule. anneal_r needs BOTH decay_interval
        # and decay_r, or NEITHER (both None disables annealing). Exactly one set
        # would silently NOT anneal (new_r stays init_r). Spelled out per case.
        _only_interval = decay_interval is not None and decay_r is None
        _only_decay_r  = decay_r is not None and decay_interval is None
        if _only_interval or _only_decay_r:
            raise ValueError(
                f"decay_interval={decay_interval} and decay_r={decay_r} must be "
                f"provided together: both None disables annealing, but exactly one "
                f"set would silently NOT anneal. Provide both or neither."
            )
        # Fail loudly: an inverted IB schedule. final_r > init_r pins r=final_r from
        # epoch 0 (the anneal max() clamps immediately) — never the intended decay.
        if (init_r is not None and final_r is not None and final_r > init_r):
            raise ValueError(
                f"final_r={final_r} must be <= init_r={init_r}: final_r > init_r "
                f"pins r=final_r from the first epoch (no annealing)."
            )
        # Fail loudly: the motif consistency loss operates on node_att, which is
        # None on the edge path — so consistency coefs with learn_edge_att=True
        # are a SILENT no-op (the inner gate in compute_loss skips). Same family
        # as the inert-config guards above.
        if learn_edge_att and (motif_loss_coef > 0 or within_node_coef > 0
                               or between_motif_coef > 0):
            raise ValueError(
                "motif consistency loss (motif_loss_coef / within_node_coef / "
                "between_motif_coef) requires node-level attention, but "
                "learn_edge_att=True has no node_att — the consistency term would "
                "be a silent no-op. Drop the consistency coefs, or use a "
                "node/motif attention path (learn_edge_att=False)."
            )

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
        self.logit_clamp = logit_clamp
        self.deterministic_att = deterministic_att
        self.task_type = task_type
        self.num_classes = num_classes

        # Backbone
        self.clf = BaseGNN(
            x_dim=x_dim, hidden_dim=hidden_dim, num_layers=num_layers,
            backbone=backbone_name, node_encoder=node_encoder,
            apply_layer_norm=apply_layer_norm, dropout=dropout,
            deg=deg, edge_dim=edge_dim,
            conv_normalize=conv_normalize, gin_inner_bn=gin_inner_bn,
            self_gate=self_gate, graph_pool=graph_pool,
        )
        self.clf.lin2 = nn.Linear(hidden_dim, num_classes)

        # Node-level extractor (base GSAT path when noise='none')
        self.extractor = ExtractorMLP(hidden_dim, dropout_p=extractor_dropout_p)

        # Motif pool → MLP scorer (readout method or noise=node|motif)
        if _uses_motif_scorer(motif_method, noise):
            if motif_scorer_norm is None:
                raise ValueError(
                    "motif_scorer_norm must be set explicitly (instance | layer | "
                    "none) for a run that uses the motif scorer (motif_method="
                    "'readout' or noise in {'node','motif'}) — there is no default."
                )
            self.motif_scorer = MotifReadoutScorer(
                in_dim=hidden_dim,
                pool_mode=pool_mode,
                dropout_p=extractor_dropout_p,
                norm=motif_scorer_norm,
            )
        else:
            self.motif_scorer = None

        # Edge attention extractor (learn_edge_att=True path)
        if learn_edge_att:
            self.edge_extractor = ExtractorMLP(
                hidden_dim * 2, dropout_p=extractor_dropout_p, edge_mode=True,
            )
        else:
            self.edge_extractor = None

        # Current IB prior retention (updated by anneal_r; used in info_loss)
        self.register_buffer('r', torch.tensor(float(init_r)))

    # ── IB prior annealing ───────────────────────────────────────────────────

    def anneal_r(self, epoch: int) -> None:
        """Update IB prior retention r based on decay schedule.

        ``epoch`` is 0-indexed (matches official GSAT ``get_r(current_epoch)``).
        """
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
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """Compute attention log-logits.

        Returns (node_logits [N,1], motif_logits [M,1]|None,
                 inverse_indices [N]|None, motif_batch [M]|None).
        """
        # Only called on the node/motif paths — forward computes edge attention
        # from self.edge_extractor separately and never calls this when
        # learn_edge_att is True. self.motif_scorer is not None iff readout /
        # noise in {node,motif} (built under the same predicate in __init__), so
        # it is the single condition selecting the motif path.
        if self.motif_scorer is not None:
            if nodes_to_motifs is None:
                raise ValueError(
                    "motif scorer active (motif_method='readout' or noise in "
                    "{'node','motif'}) but nodes_to_motifs is None — motif "
                    "vocabulary annotations are required on every graph.")
            inv_idx, motif_batch, _ = compute_inverse_idx(nodes_to_motifs, batch)
            motif_logits, node_logits = self.motif_scorer(
                node_emb, inv_idx, motif_batch)
            return node_logits, motif_logits, inv_idx, motif_batch

        # Base node extractor (motif_method in {none,loss}, noise='none').
        return self.extractor(node_emb, batch), None, None, None

    def _sample_node_attention(
        self,
        node_logits: Tensor,
        motif_logits: Optional[Tensor],
        inv_idx: Optional[Tensor],
        training: bool,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Concrete / soft-sigmoid sampling at node or motif granularity."""
        lc = self.logit_clamp
        det = self.deterministic_att
        node_logits = _clamp_logits(node_logits, lc)
        if motif_logits is not None:
            motif_logits = _clamp_logits(motif_logits, lc)

        if (motif_logits is not None and inv_idx is not None
                and self.noise == 'motif'):
            motif_att = _concrete_sample(
                motif_logits, training, logit_clamp=lc, deterministic=det)
            node_att = lift_motif_to_node(motif_att, inv_idx)
            node_att_soft = lift_motif_to_node(motif_logits.sigmoid(), inv_idx)
            return node_att, node_att_soft, motif_att

        node_att = _concrete_sample(
            node_logits, training, logit_clamp=lc, deterministic=det)
        return node_att, node_logits.sigmoid(), None

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
        node_weights: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Dict]:
        """Forward pass.

        Returns (logits [B, C], node_att [N, 1] or None, aux_dict)

        aux_dict contains tensors needed for loss computation:
          node_att, motif_att (if applicable), inv_idx, motif_batch

        When ``node_weights`` is set (eval counterfactual), the graph is left
        intact and the bool/float mask is injected as ``node_att`` directly.
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if node_weights is not None:
            # node_weights is already the [N,1] float node attention (prepared by
            # the evaluator); apply it directly through the injection channels.
            nw = node_weights.to(x.device).float().view(-1, 1)
            graph_emb, node_emb_final = self.clf.get_embedding(
                x, edge_index,
                edge_attr=edge_attr,
                node_att=nw,
                w_feat=self.w_feat,
                w_message=self.w_message,
                w_readout=self.w_readout,
                batch=batch,
            )
            logits = self.clf.classify(graph_emb)
            return logits, nw, {'node_att': nw, 'node_att_soft': nw}

        r = float(self.r.item())

        # Step 1: Backbone embedding (no attention injection yet)
        _, node_emb = self.clf.get_embedding(
            x, edge_index, edge_attr=edge_attr, batch=batch
        )

        # Step 2/3: score → sample. The edge path scores edges from node_emb
        # directly; the node/motif paths go through _get_node_logits, which is
        # therefore computed ONLY when used (never on the edge path). The
        # nodes_to_motifs-None validation now lives inside _get_node_logits (one
        # local raise), so it is not duplicated here.
        node_logits = motif_logits = inv_idx = None
        edge_att = edge_att_mp = edge_att_soft = None
        node_att_soft = motif_att = None
        lc = self.logit_clamp
        if self.learn_edge_att:
            src, dst = edge_index
            edge_logits = self.edge_extractor(
                torch.cat([node_emb[src], node_emb[dst]], dim=-1),
                batch[src],
            )
            edge_logits = _clamp_logits(edge_logits, lc)
            edge_att = _concrete_sample(
                edge_logits, self.training, logit_clamp=lc,
                deterministic=self.deterministic_att)
            edge_att_mp = _symmetrize_edge_att(edge_index, edge_att)
            edge_att_soft = edge_logits.sigmoid()
            node_att = None
        else:
            node_logits, motif_logits, inv_idx, _ = self._get_node_logits(
                node_emb, nodes_to_motifs, batch)
            node_att, node_att_soft, motif_att = self._sample_node_attention(
                node_logits, motif_logits, inv_idx, self.training,
            )

        # Step 4: Re-run backbone with attention injection
        graph_emb, node_emb_final = self.clf.get_embedding(
            x, edge_index,
            edge_attr=edge_attr,
            node_att=node_att,
            edge_atten=edge_att_mp if edge_att_mp is not None else edge_att,
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
            'motif_att':     motif_att,
            'node_att_soft': node_att_soft,
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
                if aux.get('motif_att') is not None:
                    motif_att = aux['motif_att'].view(-1)
                else:
                    motif_att = aux['motif_logits'].sigmoid().view(-1)
                # No size weighting on the motif path (sw=None). motif_att is
                # already ONE value per motif, so info_loss's mean over the M
                # motif rows is per-motif-equal BY CONSTRUCTION. Averaging the
                # per-node 1/len weight onto motif rows collapses to 1/L_m and
                # would re-introduce a spurious size penalty on large motifs, so
                # motif_info_size_normalize has NO effect at motif granularity —
                # it applies only to the node-level info loss below.
                ib = info_loss(motif_att, r)
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
