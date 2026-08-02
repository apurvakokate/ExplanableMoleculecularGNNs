"""mage.py — MAGE Stage-2 class-wise motif importance scores.

Faithful reimplementation of the *motif identification* stage (§3.2) of

    MAGE: Model-Level Graph Neural Networks Explanations via Motif-based
    Graph Generation.  Yu et al., arXiv:2405.12519.
    Official code: https://github.com/ZhaoningYu1996/MAGE

MAGE proper is a three-stage model-level explainer (motif extraction →
attention-based class-wise motif scoring → VAE motif-graph generation). We only
reimplement **Stage 2**, the per-motif class-relevance matrix ``S_cm``, because
we evaluate those scores directly as a per-motif explanation proxy. There is no
generator here.

This is a completely separate implementation from ``motif_occlusion.py`` (the
cosine-embedding occlusion baseline that was previously mislabelled "MAGE").

Pipeline identifier: this method is wired into ``run_vanilla`` under the name
``mage_official`` (result keys ``mage_official_*``, CSV stem
``mage_official_motif_scores``) — deliberately NOT ``mage`` — so its outputs
never collide with legacy ``mage_*`` result dirs that actually hold the renamed
Motif-Occlusion. The public entry point below is still named ``run_mage``.

Algorithm (Stage 2, §3.2)
-------------------------
Let ``f`` be the frozen target GNN. Write its graph encoder as
``phi(G) = pool(f.get_emb(G))`` (the pooled graph embedding that ``f`` classifies)
and its head as ``f_cls`` (``phi(G) -> logits``).

2.1  Attention.  For each molecule ``G_i`` with motifs ``{m}`` present in it,
     query ``h_{G_i}=phi(G_i)``, keys/values ``h_m=phi(motif_graph_m)``:

         e_{i,m} = LeakyReLU( a_q·(W h_{G_i}) + a_k·(W h_m) )
         alpha_{i,m} = softmax_m( e_{i,m} )                       (Eq. 1)
         h'_{G_i} = Σ_m alpha_{i,m} · (W h_m)

     This is a single-head GAT layer over the (motif → molecule) bipartite graph,
     exactly the ``GATConv(heads=1, add_self_loops=False, bias=False)`` the
     official repo uses. The attention params ``(W, a_q, a_k)`` are trained to
     make the aggregated encoding reproduce the ORIGINAL prediction of the
     FROZEN model:

         minimize  Σ_i  CE( f_cls(h'_{G_i}),  softmax(f_cls(h_{G_i})).detach() )

     (Reference-repo bug fixed here: it trained a *fresh* linear classifier on
     hard argmax labels instead of distilling the frozen model's soft output.)

2.2  Scores.  ``S_mm[m,i] = alpha_{i,m}`` (0 if motif ``m`` ∉ molecule ``i``);
     ``P[i,:] = softmax(f_cls(h_{G_i}))`` — the frozen target model's predicted
     class probabilities (reference-repo bug fixed: it used the surrogate
     classifier's probs). Then

         S_cm = S_mm @ P,   then divide row ``m`` by ``freq(m)``
                            (the number of molecules containing ``m``).

     Supports arbitrary ``num_classes`` (reference was binary-hardcoded).

2.3  Expose a single per-motif score. Two modes (see ``use_predicted_class``):
       * fixed positive class (default): ``score[m] = S_cm[m, positive_class]``
         — reproduces the prior MOSE-GNN behaviour (fixed class column).
       * predicted class per graph: ``score[m] = (1/freq_m) Σ_i alpha_{i,m}
         P[i, argmax_c P[i,c]]`` — each graph votes with its own predicted class,
         then averaged; no single global class survives the averaging.

The returned score is NOT a masking effect, so — unlike ``motif_occlusion`` — it
is not tautological with the mask-based ``score_impact_correlation`` metric.

Task-type handling
------------------
* ``BinaryClass`` (model emits one logit ``z``): treated as 2-class with logits
  ``[0, z]`` so ``P = [1-σ(z), σ(z)]`` and the CE distillation is well defined.
* ``Regression``: no class probabilities — the reconstruction objective becomes
  MSE and ``P`` carries the scalar prediction (single column). ``positive_class``
  / ``use_predicted_class`` collapse to the one column. Documented extension
  beyond the paper (which is classification-only).
* ``MultiLabel``: MAGE's class-probability formulation assumes mutually
  exclusive classes, which multi-label targets are not. Skipped with a warning.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_add_pool

NodeScoreResult = Dict[str, Dict[int, float]]


# ─────────────────────────────────────────────────────────────────────────────
# Motif-graph construction from vocab SMARTS
# ─────────────────────────────────────────────────────────────────────────────

def _uses_atom_encoder(model) -> bool:
    """True if the frozen model encodes nodes with the OGB ``AtomEncoder`` (which
    expects an ``[N, 9]`` *integer* atom-feature tensor) rather than the 51-dim
    one-hot scheme. Checked on the model and its ``backbone_net``. Used to pick
    the motif-graph featurisation so OGB datasets get matching features; non-OGB
    (one-hot) datasets are unaffected."""
    for m in (model, getattr(model, 'backbone_net', None)):
        if m is not None and type(getattr(m, 'node_encoder', None)).__name__ == 'AtomEncoder':
            return True
    return False


def _motif_feature_scheme(model, sample) -> str:
    """Which node featurisation the frozen model expects for motif graphs:
      'ogb'    — OGB AtomEncoder ([N, 9] integer features),
      'mutag'  — 14-dim one-hot over the Mutagenicity TUDataset atom-type map,
      'onehot' — the generic 51-dim one-hot scheme (default).
    OGB is detected from the encoder; mutag from the molecule feature width (it is
    the only 14-dim dataset), so all other one-hot datasets stay on 'onehot'."""
    if _uses_atom_encoder(model):
        return 'ogb'
    try:
        from ..data.graph_to_smiles import MUTAG_ATOM_TYPE_MAP
        if sample is not None and int(sample.x.shape[1]) == len(MUTAG_ATOM_TYPE_MAP):
            return 'mutag'
    except Exception:
        pass
    return 'onehot'


def _motif_graph_from_smarts(smarts: str, scheme: str = 'onehot') -> Optional[Data]:
    """Build a model-input graph from a vocab motif SMARTS, stripping rBRICS
    ``[*]`` attachment (dummy) atoms.

    Node features match the frozen model's featurisation (see ``_motif_feature_scheme``):
    the generic 51-dim one-hot by default (``scheme='onehot'``), the OGB ``[N, 9]``
    integer atom features (``scheme='ogb'``), or the 14-dim Mutagenicity-TUDataset
    one-hot (``scheme='mutag'``).

    Returns None if the fragment cannot be parsed, contains an atom outside the
     atom vocabulary, or has no real atoms after stripping.
    """
    from rdkit import Chem
    from rdkit.Chem import RWMol, GetAdjacencyMatrix
    import numpy as np
    from ..data.dataset import _atom_features

    mol = Chem.MolFromSmiles(smarts)
    if mol is None:
        mol = Chem.MolFromSmarts(smarts)
    if mol is None:
        return None

    # Remove wildcard/dummy attachment atoms (atomic number 0), high index first
    # so earlier indices stay valid.
    rw = RWMol(mol)
    dummies = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
    for idx in sorted(dummies, reverse=True):
        rw.RemoveAtom(idx)
    mol = rw.GetMol()
    if mol.GetNumAtoms() == 0:
        return None

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:  # aromatic fragment left without its ring context — skip kekulize
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except Exception:
            return None

    if scheme == 'ogb':
        # OGB AtomEncoder models need [N, 9] integer atom features, built the
        # same way the OGB molecules were (atom_to_feature_vector). A blind
        # dtype cast of the one-hot features would be the wrong schema.
        try:
            from ogb.utils.features import atom_to_feature_vector
            feats = [atom_to_feature_vector(a) for a in mol.GetAtoms()]
        except Exception:
            return None
        if not feats:
            return None
        x = torch.tensor(feats, dtype=torch.long)
    elif scheme == 'mutag':
        # mutag (Mutagenicity TUDataset) uses 14-dim one-hot over its own atom-type
        # map (C,O,Cl,H,N,F,Br,S,P,I,Na,K,Li,Ca) — NOT the generic 51-dim scheme.
        from ..data.graph_to_smiles import MUTAG_ATOM_TYPE_MAP
        _num = len(MUTAG_ATOM_TYPE_MAP)                       # 14
        _atnum_to_idx = {an: i for i, an in MUTAG_ATOM_TYPE_MAP.items()}
        idxs = []
        for a in mol.GetAtoms():
            j = _atnum_to_idx.get(a.GetAtomicNum())
            if j is None:
                return None                                  # atom outside mutag's 14 types
            idxs.append(j)
        if not idxs:
            return None
        x = torch.zeros((len(idxs), _num), dtype=torch.float32)
        x[torch.arange(len(idxs)), torch.tensor(idxs)] = 1.0
    else:
        x = _atom_features(mol)      # None if an atom is outside ATOMS
        if x is None:
            return None

    A = np.array(GetAdjacencyMatrix(mol))
    rows, cols = np.nonzero(A)
    edge_index = torch.stack([
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(cols, dtype=torch.long),
    ], dim=0) if len(rows) else torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


# ─────────────────────────────────────────────────────────────────────────────
# Frozen-model encoder / classifier access
# ─────────────────────────────────────────────────────────────────────────────

def _classify(model: nn.Module, graph_emb: torch.Tensor) -> torch.Tensor:
    """Apply the frozen model's classification head to a pooled graph embedding."""
    if hasattr(model, 'classify'):
        return model.classify(graph_emb)
    bn = getattr(model, 'backbone_net', None)
    if bn is not None and hasattr(bn, 'classify'):
        return bn.classify(graph_emb)
    raise AttributeError(
        'MAGE requires the target model to expose classify(graph_emb) '
        '(directly or via .backbone_net) — see SharedModules/models/gnn_base.py')


@torch.no_grad()
def _graph_embed(model: nn.Module, data: Data, device: torch.device) -> torch.Tensor:
    """phi(G): pooled graph embedding, matching the backbone's global_add_pool."""
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    edge_attr = getattr(data, 'edge_attr', None)
    try:
        node_emb = model.get_emb(x, edge_index, batch, edge_attr)
    except TypeError:
        node_emb = model.get_emb(x, edge_index, batch)
    return global_add_pool(node_emb, batch)       # [1, D]


@torch.no_grad()
def _embed_graphs(model: nn.Module, data_list: List[Data], device: torch.device,
                  batch_size: Optional[int] = None) -> Optional[torch.Tensor]:
    """Pooled embeddings [len(data_list), D] for a list of graphs.

    ``batch_size=None`` → per-graph (mirrors ``_graph_embed`` exactly; the reference
    path). ``batch_size>0`` → GPU-batched: pack each chunk into ONE disconnected PyG
    ``Batch``, a single ``get_emb`` forward, then ``global_add_pool`` per graph via
    the batch vector. Because add-pool is a per-graph segment sum over a disconnected
    batch, the batched result is BIT-IDENTICAL to the per-graph loop — it just
    replaces N tiny launches with N/batch_size big ones (the GPU win). Verified by
    ab_mage_batched.py."""
    if not data_list:
        return None
    if batch_size is None:
        return torch.stack([_graph_embed(model, g, device).squeeze(0)
                            for g in data_list]).to(device)
    embs: List[torch.Tensor] = []
    for start in range(0, len(data_list), batch_size):
        batch = Batch.from_data_list(data_list[start:start + batch_size]).to(device)
        ea = getattr(batch, 'edge_attr', None)
        try:
            node_emb = model.get_emb(batch.x, batch.edge_index, batch.batch, ea)
        except TypeError:
            node_emb = model.get_emb(batch.x, batch.edge_index, batch.batch)
        embs.append(global_add_pool(node_emb, batch.batch))      # [k, D]
    return torch.cat(embs, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Single-head additive (GAT-style) attention over motif → molecule edges
# ─────────────────────────────────────────────────────────────────────────────

class _MotifAttention(nn.Module):
    """One GAT head: shared value/query/key projection ``W`` plus additive
    attention vectors ``a_q`` (molecule/query) and ``a_k`` (motif/key)."""

    def __init__(self, dim: int, negative_slope: float = 0.2):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.a_q = nn.Parameter(torch.empty(dim))
        self.a_k = nn.Parameter(torch.empty(dim))
        self.negative_slope = negative_slope
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.a_q)
        nn.init.zeros_(self.a_k)

    def forward(
        self,
        h_mol: torch.Tensor,        # [N, D] molecule embeddings
        h_motif: torch.Tensor,      # [M, D] motif embeddings
        edge_mol: torch.Tensor,     # [E] molecule index per edge
        edge_motif: torch.Tensor,   # [E] motif index per edge
        num_mol: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (alpha [E], h'_mol [N, D])."""
        Wm = self.W(h_mol)          # [N, D]
        Wk = self.W(h_motif)        # [M, D]
        e = (Wm[edge_mol] * self.a_q).sum(-1) + (Wk[edge_motif] * self.a_k).sum(-1)
        e = F.leaky_relu(e, self.negative_slope)          # [E]

        # Softmax of e over the motifs of each molecule (segment softmax, no
        # torch_scatter dependency).
        gmax = e.new_full((num_mol,), float('-inf'))
        gmax = gmax.scatter_reduce(0, edge_mol, e, reduce='amax', include_self=True)
        e = e - gmax[edge_mol]
        exp_e = e.exp()
        denom = e.new_zeros(num_mol).index_add_(0, edge_mol, exp_e)
        alpha = exp_e / denom[edge_mol].clamp_min(1e-12)   # [E]

        # Aggregate value = W h_motif, weighted by alpha, into each molecule node.
        h_prime = h_mol.new_zeros(num_mol, Wk.size(1))
        h_prime.index_add_(0, edge_mol, alpha.unsqueeze(1) * Wk[edge_motif])
        return alpha, h_prime


# ─────────────────────────────────────────────────────────────────────────────
# Split-aware fit / score (train-on-train, score-per-split)
#
# ``run_mage`` (below) fits the attention on the SAME graphs it scores — correct
# for a single-set model-level explanation, but a train/val/test evaluation must
# FIT the attention on TRAIN only and then SCORE each split with that frozen
# attention (otherwise the test score is a train-on-test leakage artifact). The
# ``_MotifAttention`` module is graph-set-agnostic (it is a single GAT head over
# embeddings), so a train-fit instance applies unchanged to any split's tensors.
# The three helpers below factor run_mage's monolith into build → fit → score so
# the driver can reuse a single fitted attention across all splits.
# ─────────────────────────────────────────────────────────────────────────────

def _build_mage_tensors(
    model: nn.Module,
    graphs: List[Data],
    vocab,
    device: torch.device,
    task_type: str,
    max_graphs: Optional[int] = None,
    verbose: bool = True,
    embed_batch_size: Optional[int] = None,
):
    """Build the (motif→molecule) bipartite tensors + frozen predictions P for a
    graph list. Returns a dict T (see keys below) or None if nothing embeddable.

    Mirrors the tensor-building inside ``run_mage``; shared by fit and score so a
    single fitted attention can be evaluated on a DIFFERENT graph list (a split)
    than the one it was trained on.

    ``embed_batch_size`` — None keeps the per-graph embedding loop (reference);
    a positive int GPU-batches the molecule + motif encoder forwards (bit-identical,
    just far fewer kernel launches; see ``_embed_graphs``)."""
    is_reg = (task_type == 'Regression')
    model.eval()
    model.to(device)
    motif_list = getattr(vocab, 'motif_list', [])
    graphs = graphs if max_graphs is None else graphs[:max_graphs]

    present: set = set()
    for d in graphs:
        n2m = getattr(d, 'nodes_to_motifs', None)
        if n2m is None:
            continue
        present.update(int(m) for m in n2m[n2m >= 0].unique().tolist())
    if not present:
        if verbose:
            print('  [warn] MAGE: no motif assignments (nodes_to_motifs) found.')
        return None

    _scheme = _motif_feature_scheme(model, graphs[0] if graphs else None)
    motif_ids: List[int] = []
    motif_graphs: List[Data] = []          # collect first, embed once (batched)
    for mid in sorted(present):
        if mid >= len(motif_list):
            continue
        g = _motif_graph_from_smarts(str(motif_list[mid]), scheme=_scheme)
        if g is None:
            continue
        motif_graphs.append(g)
        motif_ids.append(mid)
    if not motif_ids:
        if verbose:
            print('  [warn] MAGE: no motif graphs could be built from vocab SMARTS.')
        return None

    h_motif = _embed_graphs(model, motif_graphs, device, embed_batch_size)  # [M, D]
    mid_to_col = {mid: j for j, mid in enumerate(motif_ids)}
    dim = h_motif.size(1)

    kept_graphs: List[Data] = []           # collect first, embed once (batched)
    kept_gi: List[int] = []                # index into `graphs` for each molecule row
    edge_mol: List[int] = []
    edge_motif: List[int] = []
    for gi, d in enumerate(graphs):
        n2m = getattr(d, 'nodes_to_motifs', None)
        if n2m is None:
            continue
        mids = [int(m) for m in n2m[n2m >= 0].unique().tolist() if int(m) in mid_to_col]
        if not mids:
            continue
        row = len(kept_graphs)
        kept_graphs.append(d)
        kept_gi.append(gi)
        for m in mids:
            edge_mol.append(row)
            edge_motif.append(mid_to_col[m])
    if not kept_graphs:
        if verbose:
            print('  [warn] MAGE: no molecules with embeddable motifs.')
        return None

    h_mol = _embed_graphs(model, kept_graphs, device, embed_batch_size)  # [N, D]
    num_mol = h_mol.size(0)
    edge_mol_t = torch.tensor(edge_mol, dtype=torch.long, device=device)
    edge_motif_t = torch.tensor(edge_motif, dtype=torch.long, device=device)

    with torch.no_grad():
        logits_orig = _classify(model, h_mol)             # [N, 1] or [N, C]
        if is_reg:
            P = logits_orig.detach()
            target_soft = None
        else:
            if logits_orig.size(1) == 1:                  # binary: logits [0, z]
                z = logits_orig
                logits_orig2 = torch.cat([torch.zeros_like(z), z], dim=1)
            else:
                logits_orig2 = logits_orig
            P = F.softmax(logits_orig2, dim=1).detach()   # [N, C]
            target_soft = P
    n_classes = P.size(1)

    return dict(
        h_motif=h_motif, h_mol=h_mol, edge_mol_t=edge_mol_t, edge_motif_t=edge_motif_t,
        num_mol=num_mol, P=P, target_soft=target_soft, n_classes=n_classes,
        motif_ids=motif_ids, mid_to_col=mid_to_col, kept_gi=kept_gi,
        is_reg=is_reg, dim=dim,
    )


def _fit_attention(model, T, task_type, attn_epochs=300, attn_lr=1e-2, verbose=True):
    """Train the single-head attention to reproduce the frozen model's prediction
    (distillation), exactly as in ``run_mage`` §2.1, on the tensors ``T``."""
    is_reg = T['is_reg']
    device = T['h_mol'].device
    attn = _MotifAttention(T['dim']).to(device)
    opt = torch.optim.Adam(attn.parameters(), lr=attn_lr)
    first_loss = last_loss = float('nan')
    for ep in range(attn_epochs):
        attn.train()
        opt.zero_grad()
        _, h_prime = attn(T['h_mol'], T['h_motif'], T['edge_mol_t'],
                          T['edge_motif_t'], T['num_mol'])
        logits_prime = _classify(model, h_prime)
        if is_reg:
            loss = F.mse_loss(logits_prime, T['P'])
        else:
            if logits_prime.size(1) == 1:
                z = logits_prime
                logits_prime = torch.cat([torch.zeros_like(z), z], dim=1)
            loss = -(T['target_soft'] * F.log_softmax(logits_prime, dim=1)).sum(1).mean()
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
        if ep == 0:
            first_loss = last_loss
    if verbose:
        kind = 'MSE' if is_reg else 'CE'
        print(f'  MAGE attention recon {kind}: {first_loss:.4f} -> {last_loss:.4f} '
              f'({T["num_mol"]} molecules, {len(T["motif_ids"])} motifs, '
              f'{T["edge_mol_t"].numel()} edges, {attn_epochs} epochs) [FIT]')
    return attn


def _score_from_attn(model, attn, T, positive_class=1,
                     use_predicted_class=False, return_per_instance=False):
    """Compute S_cm per-motif scores (and optional per-instance contributions)
    from a PRE-FIT attention applied to the tensors ``T`` — ``run_mage`` §2.2/2.3.
    ``per_instance`` graph indices are ``T['kept_gi']`` (index into the scored
    graph list)."""
    is_reg = T['is_reg']
    motif_ids = T['motif_ids']
    P = T['P']
    n_classes = T['n_classes']
    edge_mol_t, edge_motif_t = T['edge_mol_t'], T['edge_motif_t']
    num_mol = T['num_mol']
    kept_gi = T['kept_gi']
    device = P.device
    attn.eval()
    with torch.no_grad():
        alpha, _ = attn(T['h_mol'], T['h_motif'], edge_mol_t, edge_motif_t, num_mol)
        freq = torch.zeros(len(motif_ids), device=device).index_add_(
            0, edge_motif_t, torch.ones_like(alpha))
        S_cm = torch.zeros(len(motif_ids), n_classes, device=device)
        contrib = alpha.unsqueeze(1) * P[edge_mol_t]
        S_cm.index_add_(0, edge_motif_t, contrib)
        S_cm = S_cm / freq.clamp_min(1.0).unsqueeze(1)

        if is_reg:
            col = 0
            score_vec = S_cm[:, 0]
        elif use_predicted_class:
            pred_c = P.argmax(dim=1)
            p_pred = P.gather(1, pred_c.unsqueeze(1)).squeeze(1)
            num = torch.zeros(len(motif_ids), device=device).index_add_(
                0, edge_motif_t, alpha * p_pred[edge_mol_t])
            score_vec = num / freq.clamp_min(1.0)
            col = None
        else:
            col = int(positive_class)
            if col < 0 or col >= n_classes:
                col = min(1, n_classes - 1)
            score_vec = S_cm[:, col]

    scores = {motif_ids[j]: float(score_vec[j].item()) for j in range(len(motif_ids))}
    result = {'mean': scores, 'max': dict(scores)}
    if not return_per_instance:
        return result

    per_instance: Dict[int, Dict[int, float]] = {}
    with torch.no_grad():
        if is_reg:
            pcol = P[:, 0]
        elif use_predicted_class:
            c_of_row = P.argmax(dim=1)
            pcol = P.gather(1, c_of_row.unsqueeze(1)).squeeze(1)
        else:
            pcol = P[:, col]
        a_cpu = alpha.cpu()
        pcol_cpu = pcol.cpu()
        em_cpu = edge_mol_t.cpu()
        ek_cpu = edge_motif_t.cpu()
        for e in range(a_cpu.numel()):
            row = int(em_cpu[e].item())
            mid = motif_ids[int(ek_cpu[e].item())]
            gi = kept_gi[row]
            per_instance.setdefault(mid, {})[gi] = float(
                a_cpu[e].item() * pcol_cpu[row].item())
    return result, per_instance


def fit_mage(model, train_list, vocab, device, task_type='BinaryClass',
             attn_epochs=300, attn_lr=1e-2, max_graphs=None, verbose=True,
             embed_batch_size=None):
    """Fit MAGE Stage-2 attention on TRAIN graphs (no scoring). Returns the fitted
    ``_MotifAttention`` module, or None if the train set has nothing embeddable.
    Pair with ``score_mage`` to score each split with this one frozen attention.
    ``embed_batch_size`` GPU-batches the encoder forwards (bit-identical)."""
    if task_type == 'MultiLabel':
        if verbose:
            print("  [warn] MAGE: class-probability scoring is undefined for "
                  "MultiLabel; skipping fit.")
        return None
    if not hasattr(model, 'get_emb'):
        if verbose:
            print('  [warn] MAGE requires model.get_emb(...); skipping fit.')
        return None
    T = _build_mage_tensors(model, train_list, vocab, device, task_type,
                            max_graphs=max_graphs, verbose=verbose,
                            embed_batch_size=embed_batch_size)
    if T is None:
        return None
    return _fit_attention(model, T, task_type, attn_epochs=attn_epochs,
                          attn_lr=attn_lr, verbose=verbose)


def score_mage(model, attn, score_list, vocab, device, task_type='BinaryClass',
               positive_class=1, use_predicted_class=False, max_graphs=None,
               return_per_instance=False, verbose=True, embed_batch_size=None):
    """Score a graph list with a PRE-FIT attention (from ``fit_mage``). The scores
    S_cm are recomputed over ``score_list`` (frozen model preds + attention), so
    calling this for train/valid/test yields the per-split MAGE scores. Returns
    the same ``{'mean':…, 'max':…}`` (+ per_instance) interface as ``run_mage``.
    ``embed_batch_size`` GPU-batches the encoder forwards (bit-identical)."""
    empty = {'mean': {}, 'max': {}}
    if attn is None or not hasattr(model, 'get_emb'):
        return (empty, {}) if return_per_instance else empty
    T = _build_mage_tensors(model, score_list, vocab, device, task_type,
                            max_graphs=max_graphs, verbose=verbose,
                            embed_batch_size=embed_batch_size)
    if T is None:
        return (empty, {}) if return_per_instance else empty
    return _score_from_attn(model, attn, T, positive_class=positive_class,
                            use_predicted_class=use_predicted_class,
                            return_per_instance=return_per_instance)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_mage(
    model: nn.Module,
    test_list: List[Data],
    vocab,
    device: torch.device,
    task_type: str = 'BinaryClass',
    positive_class: int = 1,
    use_predicted_class: bool = False,
    attn_epochs: int = 300,
    attn_lr: float = 1e-2,
    max_graphs: Optional[int] = None,
    return_per_instance: bool = False,
    verbose: bool = True,
    fit_list: Optional[List[Data]] = None,
):
    """MAGE Stage-2 per-motif class-relevance scores.

    Thin wrapper over ``fit_mage`` + ``score_mage`` (the single implementation), so
    there is no duplicated tensor-build / train / score logic.

    Parameters
    ----------
    model : frozen target GNN exposing ``get_emb`` and ``classify``.
    test_list : list[Data] with ``nodes_to_motifs`` and ``smiles`` — the graphs SCORED.
    fit_list : graphs the attention is FIT on (default: ``test_list`` — the legacy
        single-set behaviour). Pass the TRAIN split to avoid train-on-test leakage.
    positive_class : class column of S_cm exposed when ``use_predicted_class`` is
        False (default 1; mutag uses 0).
    use_predicted_class : if True, collapse with each graph's own predicted class.

    Returns
    -------
    {'mean': {motif_id: score}, 'max': {motif_id: score}}. When
    ``return_per_instance`` is set, also returns ``{motif_id: {graph_idx: contrib}}``
    with graph_idx indexing ``test_list``. For a per-split evaluation, call
    ``fit_mage`` once (on train) and ``score_mage`` per split directly.
    """
    empty = {'mean': {}, 'max': {}}
    attn = fit_mage(model, fit_list if fit_list is not None else test_list,
                    vocab, device, task_type, attn_epochs=attn_epochs,
                    attn_lr=attn_lr, max_graphs=max_graphs, verbose=verbose)
    if attn is None:
        return (empty, {}) if return_per_instance else empty
    return score_mage(model, attn, test_list, vocab, device, task_type,
                      positive_class=positive_class,
                      use_predicted_class=use_predicted_class,
                      max_graphs=max_graphs,
                      return_per_instance=return_per_instance, verbose=verbose)
