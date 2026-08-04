from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_add_pool

NodeScoreResult = Dict[str, Dict[int, float]]

def _uses_atom_encoder(model) -> bool:
    for m in (model, getattr(model, 'backbone_net', None)):
        if m is not None and type(getattr(m, 'node_encoder', None)).__name__ == 'AtomEncoder':
            return True
    return False


def _motif_feature_scheme(model, sample) -> str:
    if _uses_atom_encoder(model):
        return 'ogb'
    try:
        from ..data.graph_to_smiles import MUTAG_ATOM_TYPE_MAP
        if sample is not None and int(sample.x.shape[1]) == len(MUTAG_ATOM_TYPE_MAP):
            return 'mutag'
    except Exception:
        pass
    return 'onehot'


# Structural-key prefixes emitted by the FG-first / ring-canonical vocabs, e.g.
# 'ring:c1ccccc1', 'chain:CCC', 'frag:...', 'fg:carbonyl'. RDKit cannot parse these
# keys verbatim, so a naive MolFromSmiles(key) returns None and the motif is dropped
# from MAGE's motif graph. That silently removed EVERY ring motif (incl. the benzene
# cause) on FG-first vocabs, zeroing MAGE's attribution on ring causes. Normalise the
# key to a parseable SMILES/SMARTS first.
_STRUCT_STRIP_PREFIXES = ('ring:', 'chain:', 'frag:', 'linker:', 'group:')
_FG_NAME_TO_SMARTS: Optional[dict] = None


def _fg_name_to_smarts() -> dict:
    """fg NAME -> SMARTS from the FG-first vocab's own pattern tables (lazy). Empty
    dict if the frag module is unavailable — then unknown fg:<name> keys still drop,
    no worse than before."""
    global _FG_NAME_TO_SMARTS
    if _FG_NAME_TO_SMARTS is None:
        try:
            from MotifBreakdown.fg_first_frag import MINIMAL_FG, COMPOSITE_FG
            _FG_NAME_TO_SMARTS = {n: s for n, s in list(MINIMAL_FG) + list(COMPOSITE_FG)}
        except Exception:
            _FG_NAME_TO_SMARTS = {}
    return _FG_NAME_TO_SMARTS


def _normalise_motif_key(key: str) -> Optional[str]:
    """Turn a vocab motif KEY into a parseable SMILES/SMARTS. Strips a leading
    ring:/chain:/frag:/... prefix (the tail is canonical SMILES); maps fg:<name> to
    its SMARTS; passes a bare SMILES/SMARTS through unchanged. None if an fg:<name>
    is unknown (nothing to parse)."""
    s = str(key)
    for p in _STRUCT_STRIP_PREFIXES:
        if s.startswith(p):
            return s[len(p):]
    if s.startswith('fg:'):
        return _fg_name_to_smarts().get(s[len('fg:'):])
    return s


def _motif_graph_from_smarts(smarts: str, scheme: str = 'onehot') -> Optional[Data]:
    from rdkit import Chem
    from rdkit.Chem import RWMol, GetAdjacencyMatrix
    import numpy as np
    from ..data.dataset import _atom_features

    smarts = _normalise_motif_key(smarts)
    if smarts is None:
        return None
    mol = Chem.MolFromSmiles(smarts)
    if mol is None:
        mol = Chem.MolFromSmarts(smarts)
    if mol is None:
        return None

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
        try:
            from ogb.utils.features import atom_to_feature_vector
            feats = [atom_to_feature_vector(a) for a in mol.GetAtoms()]
        except Exception:
            return None
        if not feats:
            return None
        x = torch.tensor(feats, dtype=torch.long)
    elif scheme == 'mutag':
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


def _classify(model: nn.Module, graph_emb: torch.Tensor) -> torch.Tensor:
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


class _MotifAttention(nn.Module):

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
                     use_predicted_class=False, return_per_instance=False,
                     score_mode='attention_mean'):
    """Per-motif model-level importance from the fitted attention.

    score_mode:
      'attention_mean' (DEFAULT) — importance(m) = MEAN over the molecules containing
          the motif of the learned attention: (1/freq[m]) * sum_i alpha_{i,m}. This is
          the published S^cm scoring with ONLY the class-probability term removed
          (S^cm = mean_i(alpha_{i,m} . P[i,c])  ->  mean_i(alpha_{i,m})): it keeps the
          same 1/freq normalization, so it is CLASS-AGNOSTIC (no P[i,c]) AND
          FREQUENCY-NEUTRAL (in [0,1]; a SUM would instead scale with how often the
          motif appears — a frequency bias). It isolates what the attention learned
          and is the right quantity to correlate against the (agnostic, LOO) impact.
      'class_prob' — the published S^cm = S^mm . P: importance(m) =
          mean_i alpha_{i,m} . P[i,c]. positive_class / use_predicted_class pick the
          class column. Kept only to reproduce the paper's scoring; NOT the default.
    """
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
        col = None
        if score_mode == 'attention_mean':
            # MEAN learned attention per motif = (1/freq) * sum_i alpha_{i,m}:
            # class-agnostic (no P[i,c]) and frequency-neutral. Only the
            # class-probability term is dropped from the paper's S^cm.
            score_vec = (torch.zeros(len(motif_ids), device=device).index_add_(
                0, edge_motif_t, alpha) / freq.clamp_min(1.0))
        else:  # 'class_prob' — paper's S^cm = S^mm . P
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
            else:
                col = int(positive_class)
                if col < 0 or col >= n_classes:
                    col = min(1, n_classes - 1)
                score_vec = S_cm[:, col]

    scores = {motif_ids[j]: float(score_vec[j].item()) for j in range(len(motif_ids))}
    result = {'mean': scores, 'max': dict(scores)}
    if not return_per_instance:
        return result

    # per-instance signal: the raw attention alpha (attention_mean) or alpha.P[c].
    per_instance: Dict[int, Dict[int, float]] = {}
    with torch.no_grad():
        if score_mode == 'attention_mean':
            pcol = None
        elif is_reg:
            pcol = P[:, 0]
        elif use_predicted_class:
            c_of_row = P.argmax(dim=1)
            pcol = P.gather(1, c_of_row.unsqueeze(1)).squeeze(1)
        else:
            pcol = P[:, col]
        a_cpu = alpha.cpu()
        pcol_cpu = pcol.cpu() if pcol is not None else None
        em_cpu = edge_mol_t.cpu()
        ek_cpu = edge_motif_t.cpu()
        for e in range(a_cpu.numel()):
            row = int(em_cpu[e].item())
            mid = motif_ids[int(ek_cpu[e].item())]
            gi = kept_gi[row]
            val = (a_cpu[e].item() if pcol_cpu is None
                   else a_cpu[e].item() * pcol_cpu[row].item())
            per_instance.setdefault(mid, {})[gi] = float(val)
    return result, per_instance


def fit_mage(model, train_list, vocab, device, task_type='BinaryClass',
             attn_epochs=300, attn_lr=1e-2, max_graphs=None, verbose=True,
             embed_batch_size=None):
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
               return_per_instance=False, verbose=True, embed_batch_size=None,
               score_mode='attention_mean'):
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
                            return_per_instance=return_per_instance,
                            score_mode=score_mode)


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
    score_mode: str = 'attention_mean',
):
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
                      return_per_instance=return_per_instance, verbose=verbose,
                      score_mode=score_mode)
