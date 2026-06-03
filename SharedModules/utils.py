"""utils.py — common utilities used across SharedModules, MOSE-GNN, MotifSAT."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic ops (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: Optional[str] = None) -> torch.device:
    """Return torch device.  Auto-selects CUDA if available and device_str is None."""
    if device_str is not None:
        return torch.device(device_str)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_checkpoint(
    state: Dict[str, Any],
    path: str,
    is_best: bool = False,
    best_suffix: str = '_best',
) -> None:
    """Save model checkpoint.  Optionally also saves a copy marked as best."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    if is_best:
        best_path = path.with_stem(path.stem + best_suffix)
        torch.save(state, best_path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> int:
    """Load a checkpoint into model (and optionally optimizer).

    Returns the epoch number stored in the checkpoint (0 if absent).
    """
    ckpt = torch.load(path, map_location=device or 'cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt.get('epoch', 0)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def process_data(data, use_edge_attr: bool = False):
    """Ensure data has a batch tensor and optional edge_attr.

    Utility used in training loops that receive single-graph Data objects.
    """
    if data.batch is None:
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long,
                                 device=data.x.device)
    if not use_edge_attr:
        data.edge_attr = None
    return data
