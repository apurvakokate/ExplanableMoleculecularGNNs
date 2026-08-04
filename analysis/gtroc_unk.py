"""gtroc_unk.py — masked node-direct GT-ROC for the EXCLUDE-UNK filtered variant.

Restrict the per-graph AUC to a KEEP set of atoms (kept-motif atoms), dropping UNK
atoms from BOTH positives and negatives. Mirrors SharedModules.evaluation.motif_eval's
``dnf_gt_roc_graph`` (instance = best fired clause, global = union of fired clauses) and
the node-level source-GT ROC (single clause), but with a keep-mask. When keep=all-True
(no UNK) it reproduces the pipeline's ``gt_roc_block`` instance/global — verify before use.
"""
import numpy as np


def _auc(scores, pos, neg):
    if not pos or not neg:
        return float('nan')
    from sklearn.metrics import roc_auc_score
    y = [1] * len(pos) + [0] * len(neg)
    s = [float(scores[i]) for i in pos] + [float(scores[i]) for i in neg]
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float('nan')


def _fired(nlc):
    out = []
    if nlc is None or nlc.ndim != 2:
        return out
    for j in range(nlc.shape[1]):
        idx = np.where(nlc[:, j] > 0)[0].tolist()
        if idx:
            out.append(idx)
    return out


def _dnf_masked(att, nlc, keep):
    clauses = _fired(nlc)
    if not clauses:
        return float('nan'), float('nan')
    n = nlc.shape[0]
    U = set().union(*[set(c) for c in clauses])
    neg = [i for i in range(n) if i not in U and keep[i]]           # kept non-causal atoms
    per = [a for a in (_auc(att, [i for i in c if keep[i]], neg) for c in clauses) if a == a]
    inst = max(per) if per else float('nan')
    glob = _auc(att, [i for i in sorted(U) if keep[i]], neg)
    return inst, glob


def _node_masked(att, node_label, keep):
    nl = np.asarray(node_label).astype(bool)
    pos = [i for i in range(len(nl)) if nl[i] and keep[i]]
    neg = [i for i in range(len(nl)) if not nl[i] and keep[i]]
    a = _auc(att, pos, neg)
    return a, a                                                     # single clause: inst==glob


def _to_np(x):
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().view(-1).numpy()
    except Exception:
        pass
    return np.asarray(x, dtype=float).reshape(-1)


def gtroc_masked(att_by_i, gl, keep_all=False):
    """Mean instance/global GT-ROC over graphs, excluding UNK atoms (nodes_to_motifs<0)
    unless keep_all (validation mode). Returns (instance_mean, global_mean, n_graphs)."""
    inst_all, glob_all = [], []
    for i, g in enumerate(gl):
        a = att_by_i.get(i)
        if a is None:
            continue
        att = _to_np(a)
        n2m = getattr(g, 'nodes_to_motifs', None)
        if n2m is None or keep_all:
            keep = np.ones(len(att), dtype=bool)
        else:
            keep = (_to_np(n2m) >= 0)
        nlc = getattr(g, 'node_label_clauses', None)
        if nlc is not None:
            inst, glob = _dnf_masked(att, _to_np2d(nlc), keep)
        else:
            nl = getattr(g, 'node_label', None)
            if nl is None:
                continue
            inst, glob = _node_masked(att, _to_np(nl), keep)
        if inst == inst:
            inst_all.append(inst)
        if glob == glob:
            glob_all.append(glob)
    nan = float('nan')
    return (float(np.mean(inst_all)) if inst_all else nan,
            float(np.mean(glob_all)) if glob_all else nan, len(inst_all))


def _to_np2d(x):
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)
