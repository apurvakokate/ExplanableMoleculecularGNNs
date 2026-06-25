"""vanilla_gnn.py — vanilla GNN (no motif weights) training and evaluation.

The vanilla model is a plain BaseGNN with no attention or motif parameters.
It is used as the input to post-hoc explainers (GNNExplainer, PGExplainer,
MAGE) and as a comparison baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import global_add_pool

from ..models.gnn_base import BaseGNN
from ..data.dataset import NUM_ATOM_TYPES, EDGE_FEAT_DIM
from ..evaluation.metrics import evaluate_predictions


class VanillaGNN(nn.Module):
    """Vanilla GNN with no motif weights.

    Accepts the same ``forward(x, edge_index, batch, nodes_to_motifs)``
    signature as MOSE-GNN / MotifSAT for compatibility with the eval pipeline,
    but ignores ``nodes_to_motifs``.

    Parameters
    ----------
    x_dim, hidden_dim, num_layers, backbone, node_encoder, apply_layer_norm,
    dropout, deg, edge_dim
        Passed through to BaseGNN.
    num_classes : int
        1 for binary / regression; C for multi-label.
    """

    def __init__(
        self,
        x_dim: int = NUM_ATOM_TYPES,
        hidden_dim: int = 64,
        num_layers: int = 3,
        backbone: str = 'GIN',
        node_encoder: str = 'onehot',
        apply_layer_norm: bool = False,
        dropout: float = 0.5,
        num_classes: int = 1,
        deg=None,
        edge_dim: Optional[int] = None,
        conv_normalize: str = 'l2',
        gin_inner_bn: bool = True,
    ):
        super().__init__()
        self.backbone_net = BaseGNN(
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            backbone=backbone,
            node_encoder=node_encoder,
            apply_layer_norm=apply_layer_norm,
            dropout=dropout,
            deg=deg,
            edge_dim=edge_dim,
            conv_normalize=conv_normalize,
            gin_inner_bn=gin_inner_bn,
        )
        self.backbone_net.lin2 = nn.Linear(hidden_dim, num_classes)
        self.num_classes = num_classes

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        nodes_to_motifs: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph_emb, _ = self.backbone_net.get_embedding(
            x, edge_index, edge_attr=edge_attr, batch=batch
        )
        return self.backbone_net.classify(graph_emb), None

    def get_emb(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return node-level embeddings (needed by MAGE)."""
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        _, node_emb = self.backbone_net.get_embedding(
            x, edge_index, edge_attr=edge_attr, batch=batch
        )
        return node_emb


def train_vanilla_gnn(
    model: VanillaGNN,
    loaders: Dict,
    task_type: str,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    pos_weights: Optional[torch.Tensor] = None,
    patience: int = 20,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> VanillaGNN:
    """Train a VanillaGNN with early stopping.

    Returns the model loaded with the best validation checkpoint.
    """
    model.to(device)

    if task_type in ('BinaryClass', 'MultiLabel'):
        pw = pos_weights.to(device) if pos_weights is not None else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5,
                                  min_lr=1e-5)

    # When epochs=0, skip training and load existing weights. A baseline/explainer
    # run with epochs=0 is meaningless on random weights, so REQUIRE the checkpoint.
    if epochs == 0:
        if save_path is not None and Path(save_path).exists():
            model.load_state_dict(
                torch.load(save_path, map_location='cpu', weights_only=False)
            )
            if verbose:
                print(f'  Loaded weights from {save_path}')
            return model
        raise FileNotFoundError(
            f"epochs=0 (load-and-evaluate / baseline-explainer mode) but no "
            f"checkpoint found at {save_path}. Refusing to run on randomly "
            f"initialized weights. Train the vanilla model first (epochs>0) so "
            f"the checkpoint exists, then re-run with epochs=0.")

    best_val = float('inf') if task_type == 'Regression' else 0.0
    no_improve = 0
    best_state = None

    train_loader = loaders['train']
    val_loader = loaders['valid']

    for epoch in range(1, epochs + 1):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out, _ = model(data.x, data.edge_index, data.batch,
                           data.nodes_to_motifs, data.edge_attr)
            loss = _compute_loss(criterion, out, data.y, task_type)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_predictions(model, val_loader, device, task_type)
        val_score = _val_score(val_metrics, task_type)
        scheduler.step(val_score if task_type == 'Regression' else -val_score)

        improved = (val_score < best_val if task_type == 'Regression'
                    else val_score > best_val)
        if improved:
            best_val = val_score
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if verbose and epoch % 10 == 0:
            print(f'  Epoch {epoch:3d}  val={val_score:.4f}  '
                  f'best={best_val:.4f}  patience={no_improve}/{patience}')

        if no_improve >= patience:
            if verbose:
                print(f'  Early stopping at epoch {epoch}')
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)

    return model


def _compute_loss(criterion, out, y, task_type):
    if task_type == 'MultiLabel':
        # out, y are [B, C]. Mask NaN targets PER ELEMENT (a molecule may have
        # observed labels for only some tasks), then average BCE over the
        # observed entries only. Masking per-row (valid.any) would still feed
        # NaN targets from unobserved columns into the loss → NaN gradient.
        if out.shape != y.shape:
            out = out.view_as(y)
        valid = ~torch.isnan(y)
        y0 = torch.nan_to_num(y, nan=0.0).float()
        per = F.binary_cross_entropy_with_logits(
            out, y0, pos_weight=getattr(criterion, 'pos_weight', None),
            reduction='none')
        per = per[valid]
        return per.mean() if per.numel() > 0 else (out.sum() * 0.0)
    out = out.reshape(-1)
    y = y.reshape(-1)
    valid = ~torch.isnan(y)
    return criterion(out[valid], y[valid].float())


def _val_score(metrics: Dict, task_type: str) -> float:
    if task_type == 'Regression':
        return metrics.get('rmse', float('inf'))
    elif task_type == 'MultiLabel':
        return metrics.get('auc_mean', 0.0)
    return metrics.get('auc', 0.0)
