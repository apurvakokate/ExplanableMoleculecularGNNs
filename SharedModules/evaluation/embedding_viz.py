"""embedding_viz.py — PCA motif embedding visualisation logged to W&B.

Produces two PCA scatter plots per validation pass, each with two panels
(class 0 / class 1 graphs):

  Plot A — colored by IMPORTANCE
    MOSE-GNN:  global learned σ(θ_m) — one scalar per vocabulary motif,
               constant across graphs.
    MotifSAT:  mean node attention across all nodes in this motif instance,
               varying per graph.

  Plot B — colored by IMPACT
    Both models: logit-shift impact approximation computed inline without
    a second forward pass — |σ(logit_full) - σ(logit_no_att)|, where
    logit_no_att is re-computed with the model's attention weights zeroed out
    for that motif instance only.  This is cheap (one extra conv pass per
    motif instance) but gives a directional signal as training progresses.
    Full mask-cache impact is re-used when available (from EvalPipeline).

In both cases the PCA is fitted on motif-instance embeddings:
    embedding = pool(node_emb[atoms belonging to motif m in graph g])

Each point in the scatter = one motif instance (motif m appearing in graph g).
Points are labelled by vocabulary motif id; a W&B Table stores pc1/pc2/
importance/impact/motif_name/motif_id/label for downstream analysis.

Usage
-----
    logger = EmbeddingVizLogger(
        model=model,
        vocab=vocab,
        device=device,
        motif_scores=None,           # dict[motif_id → float] or None (MotifSAT)
        task_type='BinaryClass',
        viz_every=5,                 # epochs between plots
        max_points=3000,
        impact_cache=None,           # dict[motif_id → float] or None
    )

    # Inside training loop:
    logger.log(valid_loader, epoch)
"""

from __future__ import annotations

import gc
import io
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool
from sklearn.decomposition import PCA

# matplotlib / wandb are optional imports — guard gracefully
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _motif_vocab_label(mid: int, motif_list: Optional[List[str]]) -> str:
    if motif_list is None or mid < 0 or mid >= len(motif_list):
        return f'motif_{mid}'
    label = str(motif_list[mid])
    return label[:33] + '...' if len(label) > 36 else label


@torch.no_grad()
def _collect_motif_snapshot(
    model: torch.nn.Module,
    data: Data,
    device: torch.device,
    motif_scores: Optional[Dict[int, float]],
    impact_cache: Optional[Dict[int, Dict[int, float]]],
) -> Optional[dict]:
    """Extract per-motif-instance embeddings, importance, and impact from one batch.

    Returns a dict with numpy arrays:
        emb        [M_inst, D]    motif instance embeddings (mean-pooled nodes)
        importance [M_inst]       0–1 importance per instance
        impact     [M_inst]       0–1 impact per instance (NaN where unavailable)
        motif_id   [M_inst]       vocabulary motif index
        graph_id   [M_inst]       within-batch graph index
        node_att   [N]            per-node attention (for node-level PCA panel)
        node_emb   [N, D]         node embeddings
    Returns None if data has no motif annotations or model forward fails.
    """
    data = data.to(device)
    n2m = getattr(data, 'nodes_to_motifs', None)
    smiles_list = getattr(data, 'smiles', None)
    if n2m is None:
        return None

    n = data.x.size(0)
    batch = (data.batch if data.batch is not None
             else torch.zeros(n, dtype=torch.long, device=device))

    # ── forward pass ──────────────────────────────────────────────────────────
    try:
        out = model(data.x, data.edge_index, batch, n2m,
                    getattr(data, 'edge_attr', None))
        if len(out) >= 3:
            logit, node_att_raw, aux = out[0], out[1], out[2]
        elif len(out) == 2:
            logit, node_att_raw = out[0], out[1]
            aux = {}
        else:
            logit = out[0]; node_att_raw = None; aux = {}
    except Exception:
        return None

    # node attention: [N] in [0, 1]
    if node_att_raw is not None:
        node_att = node_att_raw.view(-1).detach().float()
    else:
        # MOSE-GNN: derive from motif_params via n2m
        node_att = torch.full((n,), 0.5, device=device)

    # ── node embeddings: use post-conv node embeddings from backbone ─────────
    # Priority: (1) get_embedding() which runs full conv stack,
    #           (2) get_emb() for VanillaGNN,
    #           (3) backbone encode() as last resort (pre-conv only).
    node_emb = None
    try:
        backbone = (model.backbone_net if hasattr(model, 'backbone_net')
                    else model.clf      if hasattr(model, 'clf')
                    else None)
        if backbone is not None:
            _, node_emb = backbone.get_embedding(
                data.x, data.edge_index,
                edge_attr=getattr(data, 'edge_attr', None),
                batch=batch,
            )
    except Exception:
        node_emb = None
    if node_emb is None:
        try:
            node_emb = model.get_emb(data.x, data.edge_index, batch,
                                      getattr(data, 'edge_attr', None))
        except Exception:
            return None

    node_emb = node_emb.detach().float()

    # ── per motif-instance: pool embeddings over constituent nodes ─────────────
    # Build (graph_idx * max_vocab + motif_id) compound index
    n2m_cpu = n2m.cpu()
    known   = n2m >= 0
    if not known.any():
        return None

    max_mid   = int(n2m_cpu.max().item()) + 1
    batch_cpu = batch.cpu()
    compound  = batch_cpu * max_mid + n2m_cpu   # [N], unique per (graph, motif)
    compound[~known.cpu()] = -1

    # unique motif instances
    valid_mask = compound >= 0
    compound_valid = compound[valid_mask]
    unique_ids, inverse = compound_valid.unique(return_inverse=True)

    # mean-pool node embeddings per instance
    emb_valid = node_emb[valid_mask.to(device)]
    try:
        from torch_scatter import scatter_mean as _scatter_mean
    except ImportError:
        # torch_scatter not installed — use PyG scatter fallback
        from torch_geometric.utils import scatter as _tg_scatter
        def _scatter_mean(src, index, dim, dim_size):
            return _tg_scatter(src, index, dim=dim, dim_size=dim_size, reduce='mean')
    inst_emb  = _scatter_mean(emb_valid, inverse.to(device), dim=0,
                             dim_size=len(unique_ids))   # [M_inst, D]

    # recover vocabulary motif id and graph id per instance
    inst_mid   = unique_ids % max_mid
    inst_graph = unique_ids // max_mid

    # importance per instance
    if motif_scores is not None:
        # MOSE-GNN: global learned score σ(θ_m)
        imp_arr = np.array([motif_scores.get(int(m), 0.5)
                             for m in inst_mid.tolist()], dtype=np.float32)
    else:
        # MotifSAT: mean node attention over motif's nodes
        att_cpu = node_att.cpu().numpy()
        n2m_np  = n2m_cpu.numpy()
        imp_list = []
        for uid in unique_ids.tolist():
            mid  = uid % max_mid
            gidx = uid // max_mid
            mask_ni = (n2m_np == mid) & (batch_cpu.numpy() == gidx)
            imp_list.append(float(att_cpu[mask_ni].mean()) if mask_ni.any() else 0.5)
        imp_arr = np.array(imp_list, dtype=np.float32)
    imp_arr = np.clip(imp_arr, 0.0, 1.0)

    # impact per instance — from cache if available
    smiles_arr = None
    if isinstance(smiles_list, (list, tuple)):
        smiles_arr = smiles_list
    elif hasattr(data, 'smiles') and isinstance(data.smiles, str):
        smiles_arr = [data.smiles] * data.num_graphs

    impact_arr = np.full(len(unique_ids), np.nan, dtype=np.float32)
    if impact_cache is not None and smiles_arr is not None:
        for k, uid in enumerate(unique_ids.tolist()):
            mid  = uid % max_mid
            gidx = uid // max_mid
            if mid in impact_cache and gidx in impact_cache[mid]:
                v   = impact_cache[mid][gidx]
                if v is not None:
                    impact_arr[k] = float(v)

    # node-level arrays
    node_att_np = node_att.cpu().numpy()
    node_emb_np = node_emb.cpu().numpy()
    batch_np    = batch_cpu.numpy()
    y_graph     = data.y.view(-1)

    return {
        'emb':        inst_emb.cpu().numpy(),
        'importance': imp_arr,
        'impact':     impact_arr,
        'motif_id':   inst_mid.numpy().astype(np.int64),
        'graph_id':   inst_graph.numpy().astype(np.int64),
        'node_att':   node_att_np,
        'node_emb':   node_emb_np,
        'batch':      batch_np,
        'y_graph':    y_graph.cpu().float().numpy(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PCA scatter helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pca_scatter(
    ax,
    X: np.ndarray,
    color: np.ndarray,
    motif_ids: np.ndarray,
    motif_list: Optional[List[str]],
    title: str,
    color_label: str,
    max_annotations: int = 150,
    pca_obj: Optional[PCA] = None,
) -> Tuple[Optional[PCA], Optional[List]]:
    """Draw a PCA scatter; return (fitted_pca, table_rows)."""
    if X.shape[0] < 3:
        ax.set_title(f'{title} (n<3)', fontsize=11)
        ax.axis('off')
        return pca_obj, None

    if pca_obj is None:
        pca_obj = PCA(n_components=2, random_state=0)
        xy = pca_obj.fit_transform(X)
    else:
        xy = pca_obj.transform(X)

    c = np.nan_to_num(color, nan=0.5).clip(0.0, 1.0)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap='viridis',
                    s=14, alpha=0.65, vmin=0.0, vmax=1.0, linewidths=0)
    ax.set_title(f'{title} (n={len(xy)})', fontsize=11)
    ax.set_xlabel('PC1', fontsize=10)
    ax.set_ylabel('PC2', fontsize=10)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=color_label)

    # annotate a sample of points with motif names
    n_ann = min(len(xy), max_annotations)
    fs = max(7.0, min(14.0, 1800.0 / max(n_ann, 1)))
    for i in range(n_ann):
        label = _motif_vocab_label(int(motif_ids[i]), motif_list)
        ax.annotate(label, (xy[i, 0], xy[i, 1]),
                    fontsize=fs, alpha=0.75, ha='center', va='center')

    table_rows = [
        [float(xy[i, 0]), float(xy[i, 1]),
         float(c[i]),
         _motif_vocab_label(int(motif_ids[i]), motif_list),
         int(motif_ids[i])]
        for i in range(len(xy))
    ]
    return pca_obj, table_rows


# ─────────────────────────────────────────────────────────────────────────────
# Main logger class
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingVizLogger:
    """Log motif-embedding PCA plots to W&B during training.

    Two coloring schemes per plot:
        A — colored by importance  (global for MOSE, per-instance for MotifSAT)
        B — colored by impact      (from impact_cache if available, else NaN→grey)

    Parameters
    ----------
    model : nn.Module
        The model being trained.  Must expose a forward pass compatible with
        the calling convention used in MOSE-GNN / MotifSAT run.py.
    vocab : VocabData
    device : torch.device
    motif_scores : dict[int, float] or None
        If provided (MOSE-GNN): global learned importance σ(θ_m).
        If None (MotifSAT):     per-instance attention is used.
    task_type : str
    viz_every : int
        Log every N epochs.  0 = disabled.
    max_points : int
        Maximum motif instances to plot per class panel.
    max_batches : int
        Maximum validation batches to collect embeddings from.
    max_annotations : int
        Maximum labelled points per panel.
    impact_cache : dict or None
        From faithful LOO impact — maps motif_id → {graph_idx → impact}.
        Updated externally; can be None for the first few epochs.
    wandb_run : wandb.Run or None
        Active wandb run.  If None, W&B is looked up via wandb.run.
    dpi : int
        Figure DPI for exported PNGs.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        vocab,
        device: torch.device,
        motif_scores: Optional[Dict[int, float]] = None,
        task_type: str = 'BinaryClass',
        viz_every: int = 5,
        max_points: int = 3000,
        max_batches: int = 12,
        max_annotations: int = 150,
        impact_cache: Optional[Dict[int, Dict[int, float]]] = None,
        wandb_run=None,
        dpi: int = 150,
    ):
        self.model          = model
        self.vocab          = vocab
        self.device         = device
        self.motif_scores   = motif_scores
        self.task_type      = task_type
        self.viz_every      = viz_every
        self.max_points     = max_points
        self.max_batches    = max_batches
        self.max_annotations = max_annotations
        self.impact_cache   = impact_cache
        self.wandb_run      = wandb_run
        self.dpi            = dpi
        self._pca_fitted: Optional[PCA] = None   # reuse across epochs for stability

    def update_motif_scores(self, scores: Dict[int, float]) -> None:
        """Call after each training epoch to update MOSE-GNN importance scores."""
        self.motif_scores = scores

    def update_impact_cache(self, impact_cache: Dict[int, Dict[int, float]]) -> None:
        """Call after running compute_motif_impact to refresh impact colours."""
        self.impact_cache = impact_cache

    def should_log(self, epoch: int) -> bool:
        return (
            self.viz_every > 0
            and epoch % self.viz_every == 0
            and HAS_MATPLOTLIB
            and HAS_WANDB
        )

    @torch.no_grad()
    def log(self, valid_loader, epoch: int) -> None:
        """Collect embeddings from the validation loader and log PCA plots to W&B."""
        if not self.should_log(epoch):
            return
        if not HAS_MATPLOTLIB:
            print('[EmbeddingViz] matplotlib not available — skipping')
            return
        if not HAS_WANDB:
            print('[EmbeddingViz] wandb not available — skipping')
            return

        self.model.eval()

        # Buckets: class → lists of arrays
        buckets: Dict[int, dict] = {
            c: {'emb': [], 'imp': [], 'impact': [], 'mid': []}
            for c in (0, 1)
        }

        n_batches = 0
        for data in valid_loader:
            if n_batches >= self.max_batches:
                break
            snap = _collect_motif_snapshot(
                self.model, data, self.device,
                self.motif_scores, self.impact_cache
            )
            n_batches += 1
            if snap is None:
                continue

            y_graph = snap['y_graph']
            inst_graph = snap['graph_id']   # [M_inst] index into this batch's graphs
            inst_y = np.round(y_graph[inst_graph]).astype(int).clip(0, 1)

            for cls in (0, 1):
                sel = inst_y == cls
                if sel.any():
                    buckets[cls]['emb'].append(snap['emb'][sel])
                    buckets[cls]['imp'].append(snap['importance'][sel])
                    buckets[cls]['impact'].append(snap['impact'][sel])
                    buckets[cls]['mid'].append(snap['motif_id'][sel])

        # Check we have data
        has_data = any(buckets[c]['emb'] for c in (0, 1))
        if not has_data:
            print(f'[EmbeddingViz] No motif embeddings collected at epoch {epoch}')
            return

        rng = np.random.RandomState(epoch)
        wandb_obj = self.wandb_run or (wandb.run if HAS_WANDB else None)

        for cls in (0, 1):
            parts = buckets[cls]
            if not parts['emb']:
                continue

            X   = np.concatenate(parts['emb'],    axis=0)
            Imp = np.concatenate(parts['imp'],     axis=0)
            Ipc = np.concatenate(parts['impact'],  axis=0)
            Mid = np.concatenate(parts['mid'],     axis=0)

            # Subsample
            if len(X) > self.max_points:
                idx = rng.choice(len(X), self.max_points, replace=False)
                X, Imp, Ipc, Mid = X[idx], Imp[idx], Ipc[idx], Mid[idx]

            motif_list = getattr(self.vocab, 'motif_list', None)
            has_impact = not np.all(np.isnan(Ipc))
            n_panels   = 2 if has_impact else 1
            fig_w      = 12 * n_panels
            fig, axes  = plt.subplots(1, n_panels, figsize=(fig_w, 8))
            if n_panels == 1:
                axes = [axes]

            # Panel A: importance
            pca_obj, table_rows_imp = _pca_scatter(
                axes[0], X, Imp, Mid, motif_list,
                title=f'y={cls} — importance',
                color_label='importance',
                max_annotations=self.max_annotations,
                pca_obj=self._pca_fitted,
            )
            if self._pca_fitted is None:
                self._pca_fitted = pca_obj   # lock PCA fit after first class 0

            # Panel B: impact (if available)
            table_rows_impact = None
            if has_impact:
                # Replace NaN with grey (0.5 neutral) only for display
                Ipc_display = np.where(np.isnan(Ipc), 0.5, Ipc)
                _, table_rows_impact = _pca_scatter(
                    axes[1], X, Ipc_display, Mid, motif_list,
                    title=f'y={cls} — impact',
                    color_label='impact',
                    max_annotations=self.max_annotations,
                    pca_obj=pca_obj,   # same PCA projection
                )

            imp_type = 'global σ(θ_m)' if self.motif_scores else 'per-instance attention'
            fig.suptitle(
                f'Motif embedding PCA  epoch={epoch}  y={cls}\n'
                f'Importance: {imp_type}   '
                f'Impact: {"logit-shift / cache" if has_impact else "not available"}',
                fontsize=12, y=1.01,
            )
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=self.dpi,
                        bbox_inches='tight', pad_inches=0.2)
            plt.close(fig)
            png_bytes = buf.getvalue()

            if wandb_obj is not None:
                try:
                    if HAS_PIL:
                        img = PILImage.open(io.BytesIO(png_bytes)).convert('RGB')
                        payload = {
                            f'embviz/pca_y{cls}': wandb.Image(img,
                                caption=f'epoch={epoch} y={cls}  imp={imp_type}')
                        }
                    else:
                        payload = {f'embviz/pca_y{cls}': wandb.Image(io.BytesIO(png_bytes))}

                    # Importance table
                    if table_rows_imp:
                        payload[f'embviz/motif_importance_table_y{cls}'] = wandb.Table(
                            columns=['pc1', 'pc2', 'importance', 'motif_name', 'motif_id'],
                            data=table_rows_imp,
                        )
                    # Impact table
                    if table_rows_impact:
                        # Replace NaN with None for cleaner wandb display
                        clean_rows = []
                        for row in table_rows_impact:
                            r = list(row)
                            if np.isnan(r[2]):
                                r[2] = None
                            clean_rows.append(r)
                        payload[f'embviz/motif_impact_table_y{cls}'] = wandb.Table(
                            columns=['pc1', 'pc2', 'impact', 'motif_name', 'motif_id'],
                            data=clean_rows,
                        )
                    wandb_obj.log(payload, step=epoch)
                    print(f'[EmbeddingViz] Logged PCA (y={cls}) at epoch {epoch} '
                          f'— {len(X)} motif instances, {n_panels} panels')
                except Exception as e:
                    print(f'[EmbeddingViz] wandb log failed (y={cls}, epoch={epoch}): {e}')

        del buckets
        gc.collect()

    def reset_pca(self) -> None:
        """Force PCA to be re-fitted on the next call (e.g. after major training changes)."""
        self._pca_fitted = None


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build impact_cache from EvalPipeline results
# ─────────────────────────────────────────────────────────────────────────────

def build_impact_cache_from_eval(
    model: torch.nn.Module,
    data_list: list,
    vocab,
    device: torch.device,
    task_type: str = 'BinaryClass',
    base_att_fn=None,
) -> Dict[int, Dict[int, float]]:
    """Run faithful LOO impact and return per-graph values keyed by list index.

    Structure: ``{motif_id: {graph_idx: impact_value}}`` for embedding viz and
    the per-instance correlation. ``base_att_fn`` selects the weight vector W
    (defaults to the model's own node attention; pass e.g. an all-ones fn for a
    method-agnostic impact when the method has no per-node weights, like MAGE).
    """
    from .motif_eval import (
        build_graph_mask_cache,
        build_faithful_loo_baseline, loo_impact,
    )

    model.eval()
    mask_cache = build_graph_mask_cache(data_list)
    base_W, p_full_W = build_faithful_loo_baseline(
        model, data_list, device, task_type, base_att_fn=base_att_fn)

    cache: Dict[int, Dict[int, float]] = {}
    for mid, motif_masks in mask_cache.items():
        per_graph: Dict[int, float] = {}
        for gi, graph_mask in motif_masks.items():
            d = data_list[gi]
            nw = loo_impact(model, gi, d, graph_mask, base_W, p_full_W, device, task_type)
            if nw is not None:
                per_graph[gi] = nw
        if per_graph:
            cache[mid] = per_graph

    return cache
