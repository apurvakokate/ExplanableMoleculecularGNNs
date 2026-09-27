"""motif_modules.py — MotifSAT motif-level building blocks.

MotifPooling          — pool node embeddings to motif-instance level
MotifReadoutScorer    — score each motif instance with an MLP
compute_inverse_idx   — map nodes to dense motif-row indices
lift_motif_to_node    — broadcast motif-level values back to nodes
ExtractorMLP          — official GSAT extractor (InstanceNorm MLP)
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import InstanceNorm

logger = logging.getLogger("motifsat.motif_modules")

_VERIFY_FIXES = os.environ.get("MOTIFSAT_VERIFY_FIXES", "0") == "1"
_logged_inverse_idx_fix = False
try:
    from torch_scatter import scatter_mean, scatter_max, scatter_add
except ImportError:
    from torch_geometric.utils import scatter as _sc
    def scatter_mean(src, index, dim=0, dim_size=None):
        return _sc(src, index, dim=dim, dim_size=dim_size, reduce="mean")
    def scatter_add(src, index, dim=0, dim_size=None):
        return _sc(src, index, dim=dim, dim_size=dim_size, reduce="sum")
    def scatter_max(src, index, dim=0, dim_size=None):
        out = _sc(src, index, dim=dim, dim_size=dim_size, reduce="max")
        return out, None


# ─────────────────────────────────────────────────────────────────────────────
# Index helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_inverse_idx(
    nodes_to_motifs: Tensor,
    batch: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Map each node to a dense motif-row index for scatter operations."""
    if batch is None:
        batch = torch.zeros(nodes_to_motifs.size(0), dtype=torch.long,
                            device=nodes_to_motifs.device)
    batch = batch.long()
    offset = nodes_to_motifs.long() + 1
    max_mid = int(offset.max().item()) + 1
    gm_id = batch * max_mid + offset
    unique, inverse_indices = gm_id.unique(return_inverse=True)
    motif_batch = unique // max_mid
    motif_vocab_ids = (unique % max_mid) - 1

    global _logged_inverse_idx_fix
    if _VERIFY_FIXES and not _logged_inverse_idx_fix:
        n_unknown = int((nodes_to_motifs < 0).sum().item())
        n_unknown_rows = int((motif_vocab_ids < 0).sum().item())
        logger.info(
            "[FIX#2 active] compute_inverse_idx: unknown nodes isolated "
            "(this batch: %d unknown nodes -> %d dedicated -1 rows; "
            "%d real motif rows uncontaminated)",
            n_unknown, n_unknown_rows,
            int((motif_vocab_ids >= 0).sum().item()),
        )
        _logged_inverse_idx_fix = True

    return inverse_indices, motif_batch, motif_vocab_ids


def lift_motif_to_node(
    motif_vals: Tensor,
    inverse_indices: Tensor,
) -> Tensor:
    """Broadcast motif-level values back to node level."""
    return motif_vals[inverse_indices]


# ─────────────────────────────────────────────────────────────────────────────
# Motif pooling
# ─────────────────────────────────────────────────────────────────────────────

class MotifPooling(nn.Module):
    """Pool node embeddings to motif-instance level."""

    def __init__(self, mode: str = 'mean'):
        super().__init__()
        if mode not in ('mean', 'max', 'max_mean', 'multi'):
            raise ValueError(f"Unknown pool mode: {mode}")
        self.mode = mode

    @property
    def out_mult(self) -> int:
        return {'mean': 1, 'max': 1, 'max_mean': 2, 'multi': 3}[self.mode]

    def forward(
        self,
        emb: Tensor,
        inverse_indices: Tensor,
        num_motifs: Optional[int] = None,
    ) -> Tensor:
        M = (inverse_indices.max().item() + 1 if num_motifs is None
             else num_motifs)
        M = int(M)
        if self.mode == 'mean':
            return scatter_mean(emb, inverse_indices, dim=0, dim_size=M)
        elif self.mode == 'max':
            out, _ = scatter_max(emb, inverse_indices, dim=0, dim_size=M)
            return out
        elif self.mode == 'max_mean':
            mean = scatter_mean(emb, inverse_indices, dim=0, dim_size=M)
            mx, _ = scatter_max(emb, inverse_indices, dim=0, dim_size=M)
            return torch.cat([mx, mean], dim=1)
        else:
            mean = scatter_mean(emb, inverse_indices, dim=0, dim_size=M)
            mx, _ = scatter_max(emb, inverse_indices, dim=0, dim_size=M)
            s = scatter_add(emb, inverse_indices, dim=0, dim_size=M)
            return torch.cat([mean, mx, s], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Official GSAT Extractor MLP  (Graph-COM/GSAT run_gsat.py ExtractorMLP + MLP)
# ─────────────────────────────────────────────────────────────────────────────

class BatchSequential(nn.Sequential):
    """Sequential module that passes ``batch`` into InstanceNorm layers."""

    def forward(self, inputs: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        for module in self._modules.values():
            if isinstance(module, InstanceNorm):
                if batch is None:
                    raise ValueError("InstanceNorm in ExtractorMLP requires batch indices")
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


def _gsat_mlp_channels(in_dim: int, edge_mode: bool) -> List[int]:
    """Channel sizes matching official GSAT ExtractorMLP."""
    if edge_mode:
        # [2H, 4H, H, 1] when in_dim = 2H
        h = in_dim // 2
        return [in_dim, in_dim * 2, h, 1]
    # node: [H, 2H, H, 1]
    return [in_dim, in_dim * 2, in_dim, 1]


class ExtractorMLP(nn.Module):
    """Official GSAT extractor: graph-wise InstanceNorm MLP → scalar logit.

    Node path (``edge_mode=False``): ``[D, 2D, D, 1]``.
    Edge path (``edge_mode=True``): ``[2D, 4D, D, 1]`` with ``in_dim=2D``.

    ``batch`` indexes the graph id per row (nodes, motif instances, or edge
    source nodes for the edge extractor — same as official GSAT).
    """

    def __init__(
        self,
        in_dim: int,
        dropout_p: float = 0.5,
        edge_mode: bool = False,
        hidden_mult: int = 2,  # kept for API compat; official widths are fixed
        norm: str = 'instance',  # per-hidden-layer norm: instance | layer | none
    ):
        super().__init__()
        del hidden_mult  # official architecture uses fixed channel schedule
        if norm not in ('instance', 'layer', 'none'):
            raise ValueError(
                f"unknown norm={norm!r}; expected one of instance | layer | none")
        channels = _gsat_mlp_channels(in_dim, edge_mode)
        layers: list[nn.Module] = []
        for i in range(1, len(channels)):
            layers.append(nn.Linear(channels[i - 1], channels[i], bias=True))
            if i < len(channels) - 1:
                # norm choice. InstanceNorm is graph-wise (official GSAT, needs
                # batch — routed by BatchSequential); LayerNorm/none are for the
                # MOTIF scorer where per-graph InstanceNorm over the few motif
                # rows in a graph is unstable. 'none' skips normalization.
                if norm == 'instance':
                    layers.append(InstanceNorm(channels[i]))
                elif norm == 'layer':
                    layers.append(nn.LayerNorm(channels[i]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_p))
        self.net = BatchSequential(*layers)
        self.edge_mode = edge_mode
        self.norm = norm

    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        return self.net(x, batch)


# ─────────────────────────────────────────────────────────────────────────────
# Motif Readout Scorer
# ─────────────────────────────────────────────────────────────────────────────

class MotifReadoutScorer(nn.Module):
    """Pool node embeddings → motif MLP → motif logits (+ optional node broadcast).

    Used when ``motif_method='readout'`` or ``noise in ('node', 'motif')``.
    """

    def __init__(
        self,
        in_dim: int,
        pool_mode: str = 'mean',
        dropout_p: float = 0.5,
        hidden_mult: int = 2,
        norm: str = 'none',
    ):
        super().__init__()
        del hidden_mult
        self.pooling = MotifPooling(pool_mode)
        pooled_dim = in_dim * self.pooling.out_mult
        self.scorer = ExtractorMLP(pooled_dim, dropout_p=dropout_p, norm=norm)

    def forward(
        self,
        node_emb: Tensor,
        inverse_indices: Tensor,
        motif_batch: Tensor,
        num_motifs: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return (motif_logits [M, 1], node_logits_broadcast [N, 1])."""
        motif_emb = self.pooling(node_emb, inverse_indices, num_motifs)
        motif_logits = self.scorer(motif_emb, motif_batch)
        node_logits = lift_motif_to_node(motif_logits, inverse_indices)
        return motif_logits, node_logits


# ─────────────────────────────────────────────────────────────────────────────
# Mechanism ③ — fragment-level (motif) graph: coarsening + featurizer + GNN
# ─────────────────────────────────────────────────────────────────────────────

# Per-motif chemistry descriptors (variant B), computed from the motif SMARTS.
# Robust on SMARTS query mols (counts + rings + aromaticity), so no reliance on
# sanitization / Lipschitz descriptors that fail on `[*]`-wildcard patterns.
MOTIF_DESC_NAMES = ['n_heavy', 'n_attach', 'n_rings', 'n_aromatic',
                    'n_C', 'n_N', 'n_O', 'n_S', 'n_halogen']
MOTIF_DESC_DIM = len(MOTIF_DESC_NAMES)
_HALOGENS = {9, 17, 35, 53}          # ATOMIC NUMBERS (F,Cl,Br,I) — used by descriptors

# ── D-chemistry junction edge feature (edge_mode='chem') ─────────────────────
# Direction-aware, per-junction, summed over the crossing atom-bonds:
#   [0]      multiplicity (constant 1 per crossing bond; sum -> # crossing bonds)
#   [1:5]    bond-order histogram  {single, double, triple, aromatic}  (atom edge_attr[:, :4])
#   [5:11]   source-side attachment-atom element  {C,N,O,S,halogen,other}
#   [11:17]  target-side attachment-atom element  {C,N,O,S,halogen,other}
_BONDORDER_DIM = 4                    # SharedModules.data.dataset.BONDS (single/double/triple/aromatic)
_ELEM_BUCKET_DIM = 6                  # C, N, O, S, halogen, other
MOTIF_EDGE_CHEM_DIM = 1 + _BONDORDER_DIM + 2 * _ELEM_BUCKET_DIM   # = 17
# Element buckets keyed on the ATOMS ONE-HOT INDEX (NOT atomic number):
#   ATOMS = {H:0, C:1, N:2, O:3, S:4, F:5, P:6, Cl:7, Br:8, I:9, ...}
_ELEM_BUCKET_C, _ELEM_BUCKET_N, _ELEM_BUCKET_O, _ELEM_BUCKET_S = 0, 1, 2, 3
_ELEM_BUCKET_HALOGEN, _ELEM_BUCKET_OTHER = 4, 5
_HALOGEN_ATOMIDX = (5, 7, 8, 9)       # F, Cl, Br, I in the ATOMS one-hot index space


def _elem_bucket(atom_type_idx: Tensor) -> Tensor:
    """Map ATOMS one-hot indices [K] -> element bucket ids [K] in
    {0:C, 1:N, 2:O, 3:S, 4:halogen, 5:other}."""
    b = torch.full_like(atom_type_idx, _ELEM_BUCKET_OTHER)
    b[atom_type_idx == 1] = _ELEM_BUCKET_C
    b[atom_type_idx == 2] = _ELEM_BUCKET_N
    b[atom_type_idx == 3] = _ELEM_BUCKET_O
    b[atom_type_idx == 4] = _ELEM_BUCKET_S
    for h in _HALOGEN_ATOMIDX:
        b[atom_type_idx == h] = _ELEM_BUCKET_HALOGEN
    return b


def build_motif_descriptors(motif_list: List[str]) -> Tensor:
    """[num_motifs, MOTIF_DESC_DIM] chemistry descriptors from each motif's SMARTS.

    Attachment points are the ``[*]`` wildcards (atomic num 0). FAILS LOUD: an empty
    or unparseable SMARTS, or a SMARTS whose ring info cannot be perceived, raises —
    a motif we cannot read is a vocabulary bug, never silently masked with a zero row
    (which would corrupt that motif's id_desc features without a trace)."""
    from rdkit import Chem
    rows = []
    for i, smarts in enumerate(motif_list):
        s = str(smarts)
        if not s:
            raise ValueError(
                f"build_motif_descriptors: motif {i} has an empty SMARTS.")
        m = Chem.MolFromSmarts(s)
        if m is None:
            raise ValueError(
                f"build_motif_descriptors: motif {i} SMARTS failed to parse: {s!r}")
        Chem.GetSymmSSSR(m)                 # initialise ring info; raises loudly on failure
        n_heavy = n_attach = n_arom = nC = nN = nO = nS = nX = 0
        for a in m.GetAtoms():
            z = a.GetAtomicNum()
            if z == 0:
                n_attach += 1; continue
            if z > 1:
                n_heavy += 1
            if a.GetIsAromatic():
                n_arom += 1
            if z == 6: nC += 1
            elif z == 7: nN += 1
            elif z == 8: nO += 1
            elif z == 16: nS += 1
            elif z in _HALOGENS: nX += 1
        n_rings = m.GetRingInfo().NumRings()
        rows.append([n_heavy, n_attach, n_rings, n_arom, nC, nN, nO, nS, nX])
    return torch.tensor(rows, dtype=torch.float)


def build_motif_graph(
    inv_idx: Tensor,
    edge_index: Tensor,
    num_motifs: int,
    edge_mode: str = 'mult',
    x: Optional[Tensor] = None,
    atom_edge_attr: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Coarsen the atom graph to the fragment graph (quotient graph).

    A motif edge (a→b) exists wherever an atom bond crosses two motif instances
    (``inv_idx[u] != inv_idx[v]``). The atom ``edge_index`` is already
    bidirectional, so crossing bonds yield both directions -> symmetric motif
    adjacency. The junction edge feature depends on ``edge_mode``:
      * 'none' — no edge feature (plain GIN); returns (edge_index, None).
      * 'mult' — scalar junction MULTIPLICITY (variant D): # crossing bonds per
                 motif edge (single-bond link vs. fused/multi-bond). [E, 1].
      * 'chem' — D-CHEMISTRY [E, MOTIF_EDGE_CHEM_DIM]: multiplicity ⊕ bond-order
                 histogram ⊕ source-side element ⊕ target-side element, summed over
                 the crossing bonds (direction-aware). REQUIRES ``x`` (atom one-hot)
                 and ``atom_edge_attr`` (bond features); FAILS LOUD if either is
                 None. Assumes the ATOMS one-hot node encoding (node_encoder=onehot).
    """
    from torch_geometric.utils import coalesce
    if edge_mode not in ('none', 'mult', 'chem'):
        raise ValueError(f"edge_mode must be none|mult|chem, got {edge_mode!r}")
    dev = inv_idx.device
    src, dst = edge_index
    mi_s, mi_d = inv_idx[src], inv_idx[dst]
    cross = mi_s != mi_d
    n_cross = int(cross.sum().item())

    if edge_mode == 'none':
        if n_cross == 0:
            return torch.empty((2, 0), dtype=torch.long, device=dev), None
        raw = torch.stack([mi_s[cross], mi_d[cross]], dim=0)
        return coalesce(raw, num_nodes=num_motifs), None

    feat_dim = 1 if edge_mode == 'mult' else MOTIF_EDGE_CHEM_DIM
    if n_cross == 0:                                     # single-fragment molecule(s)
        me = torch.empty((2, 0), dtype=torch.long, device=dev)
        return me, torch.zeros((0, feat_dim), device=dev)

    raw = torch.stack([mi_s[cross], mi_d[cross]], dim=0)
    if edge_mode == 'mult':
        feat = torch.ones((n_cross, 1), device=dev)
    else:  # chem
        if x is None or atom_edge_attr is None:
            raise ValueError(
                "edge_mode='chem' requires x (atom one-hot) and atom_edge_attr "
                "(bond features); got None. Thread them from _motif_emb_logits.")
        cs, cd = src[cross], dst[cross]                  # atom endpoints of crossing bonds
        mult = torch.ones((n_cross, 1), device=dev)
        bo = atom_edge_attr[cross][:, :_BONDORDER_DIM].float()      # bond-order one-hot
        at = x.argmax(dim=1)                             # ATOMS one-hot index per atom
        src_oh = F.one_hot(_elem_bucket(at[cs]), _ELEM_BUCKET_DIM).float()
        dst_oh = F.one_hot(_elem_bucket(at[cd]), _ELEM_BUCKET_DIM).float()
        feat = torch.cat([mult, bo, src_oh, dst_oh], dim=-1)        # [n_cross, 17]
    me, ea = coalesce(raw, feat, num_nodes=num_motifs, reduce='sum')
    return me, ea


class MotifFeaturizer(nn.Module):
    """Fragment-node input features F_m for GNN1 — DECOUPLED from the node GNN.

    mode='id_desc'  : learned motif-id embedding (A) ⊕ fixed descriptors (B).
    mode='multihot' : scatter_add of the raw atom features x per fragment (the
                      atom-type count vector) — no vocab id needed.
    """

    def __init__(self, mode: str, num_motifs: int, x_dim: int,
                 id_dim: int = 64, desc_table: Optional[Tensor] = None):
        super().__init__()
        if mode not in ('id_desc', 'multihot'):
            raise ValueError(f"unknown motif_feat={mode!r}; use id_desc | multihot")
        self.mode = mode
        if mode == 'id_desc':
            # id_desc REQUIRES the descriptor table — no silent id-only fallback.
            if desc_table is None:
                raise ValueError(
                    "motif_feat='id_desc' requires a descriptor table (desc_table); "
                    "got None. Build it via build_motif_descriptors(vocab.motif_list).")
            if desc_table.size(0) != num_motifs:
                raise ValueError(
                    f"desc_table has {desc_table.size(0)} rows but num_motifs="
                    f"{num_motifs}; they must match.")
            if desc_table.size(1) == 0:
                raise ValueError("desc_table has 0 descriptor columns (empty).")
            self.id_emb = nn.Embedding(num_motifs, id_dim)
            self.register_buffer('desc_table', desc_table.float())
            self.out_dim = id_dim + self.desc_table.size(1)
        else:
            self.out_dim = x_dim

    def forward(self, x: Tensor, inv_idx: Tensor, motif_vocab_ids: Tensor,
                num_motifs: int) -> Tensor:
        if self.mode == 'id_desc':
            mid = motif_vocab_ids.long()
            return torch.cat([self.id_emb(mid), self.desc_table[mid]], dim=-1)
        return scatter_add(x, inv_idx, dim=0, dim_size=num_motifs)


class MotifGNN(nn.Module):
    """GNN1 — message passing over the fragment graph → per-motif logits.

    An HIMP-style (Fey et al., 2020) GIN/GINE fragment-graph encoder: edge-aware
    GINE (using the junction-multiplicity edge feature, variant D) when ``edge_dim``
    is set, else plain GIN. Separate parameters from the node GNN.

    Two knobs, both defaulting OFF (consensus-aligned for a single-branch encoder
    on tiny fragment graphs; see the GNN1-design ablation):
      * ``norm``     — 'none' (default) or 'layer'. NOT batch/instance: BatchNorm
                       breaks and per-graph InstanceNorm is unstable when a
                       molecule has a single fragment (M=1).
      * ``residual`` — per-layer skip connection h = h + block(h). Default False;
                       HIMP's residuals live on its cross-level atom↔clique
                       exchange, which this single branch does not have, so a
                       within-branch skip is not inherited from the consensus.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2,
                 edge_dim: Optional[int] = None, dropout: float = 0.0,
                 norm: str = 'none', residual: bool = False):
        super().__init__()
        from torch_geometric.nn import GINConv, GINEConv
        if norm not in ('none', 'layer'):
            raise ValueError(f"MotifGNN norm must be 'none' or 'layer', got {norm!r}")
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.residual = residual
        self.input_lin = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(
                GINEConv(mlp, train_eps=True, edge_dim=edge_dim) if edge_dim
                else GINConv(mlp, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim) if norm == 'layer'
                              else nn.Identity())
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1))

    def forward(self, Fm: Tensor, edge_index: Tensor,
                edge_attr: Optional[Tensor]) -> Tensor:
        h = self.input_lin(Fm)
        for conv, norm in zip(self.convs, self.norms):
            m = (conv(h, edge_index, edge_attr) if self.edge_dim is not None
                 else conv(h, edge_index))
            block = F.dropout(F.relu(norm(m)), p=self.dropout, training=self.training)
            h = h + block if self.residual else block
        return self.scorer(h)                # [M, 1] per-motif logits
