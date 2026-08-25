#!/usr/bin/env python3
"""evaluate.py — THE single, self-contained source of truth for explainability evaluation.

Fully parameterized; the ONLY internal iteration is folds × backbones × rule-ids (planted):
    --method   gnnexplainer|pgexplainer|mage|motif_occlusion|gsat|motifsat|mose
    --dataset  one dataset
    --vocab    one vocabulary variant (e.g. rbrics ; rbrics_filter selects MoSE's filter model)
    --gt_tier  none | source | planted
    --unk      include | exclude   (exclude drops UNK/below-threshold motifs & atoms via the
               vocab's kept_motif_ids — this IS the filtered view; no separate run/restriction)

TWO regimes, ONE methodology:
  * POST-HOC (GNNExplainer/PGExplainer/MAGE/Motif-Occlusion): stochastic/expensive explanation,
    PRODUCED ONCE upstream (eval_driver_posthoc) and saved. This script READS it — never re-runs
    an explainer. Inputs from --artifacts_root: {stem}_importance/impact_{split}.csv,
    explainer_importances.json (per-node atts), impact_cache_agnostic_{split}.json, summary_splits.json.
  * ANTE-HOC (GSAT/MotifSAT/MoSE): the model's own attention, DETERMINISTIC at eval, so this
    script forwards the saved checkpoint and recomputes (score, own-impact LOO, atts). Per-graph
    node scores are cached to explainer_importances.json (check-exists-else-forward).

ARTIFACT PARITY: it writes the SAME per-run files the retired drivers wrote, into --dest_root
(mirroring the ckpt tree; --unk exclude lands under the sibling `*_filter` path so build_metric_set
harvests it as the filtered variant):
    {stem}_importance_{split}.csv, {stem}_impact_{split}.csv
    {stem}_grouped_corr_pooled_{alltest,testonly}.csv,  {stem}_instance_corr_{split}.csv
    importance_global.csv, impact_global.csv,  summary_splits.json
    explainer_importances.json  (ante-hoc: written; post-hoc: already exists upstream)
Plus a rolled-up metrics CSV at --dest (one row per run) for quick inspection.

All eval MATH is inline (impact LOO, grouped Pearson 4-variant, node GT-ROC, per-instance corr) +
the exact writers. It uses only model/prediction primitives (_model_forward, evaluate_predictions,
model_node_att_fn) — never EvalPipeline / evaluate_*_all_splits / compute_motif_impact, and never
the retired eval_driver_* / rescore_* modules.

summary_splits GT-ROC reproduces the driver's full node-level set (faithful port of gt_roc_block):
whole (gt_roc_node) / fired / spurious / family + DNF instance/global, all per-graph-mean AUC, with
--unk exclude dropping UNK atoms. Only edge-level gt_roc_edge is omitted (no table reads it).

  source .../conda.sh && conda activate l2xgnn
  python analysis/evaluate.py --method mose --dataset BBBP --vocab rbrics --gt_tier none --unk include \
      --ckpt_root final_v2 --dest_root eval_v1 --dest eval_v1/mose_bbbp.csv \
      --data_root <DATA> --vocab_root vocab_final_v2 --processed_root processed_final_v2 --device cuda
  python analysis/evaluate.py --method mage --dataset BBBP --vocab rbrics --gt_tier none --unk include \
      --ckpt_root final_v2 --artifacts_root posthoc_v1 --dest_root eval_v1 --dest eval_v1/mage_bbbp.csv ...
"""
import argparse
import importlib
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from SharedModules.evaluation.metrics import _model_forward, evaluate_predictions
from SharedModules.evaluation.motif_eval import model_node_att_fn, spurious_motif_names

SPLITS = ('train', 'valid', 'test')
UNK_ID = -1
SOURCE_GT = {'Benzene_Verified_GT', 'Fluoride_Carbonyl_Verified_GT',
             'Alkane_Carbonyl_Verified_GT', 'mutag'}
POSTHOC = {'gnnexplainer', 'pgexplainer', 'mage', 'motif_occlusion'}
NATIVE = {'gsat', 'motifsat', 'mose'}
DISK_STEM = {'mage': 'mage_v2'}
FAMILY_DIR = {'gnnexplainer': 'baselines', 'pgexplainer': 'baselines', 'mage': 'baselines',
              'motif_occlusion': 'baselines', 'gsat': 'base_gsat', 'motifsat': 'motifsat',
              'mose': 'mose'}


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING (inlined)
# ═════════════════════════════════════════════════════════════════════════════

def _proc_root(meta, data_root, processed_root, variant):
    from SharedModules.data.dataset_routing import (
        default_processed_base, variant_processed_root)
    if processed_root:
        return variant_processed_root(str(processed_root), variant)
    if meta.get('processed_root'):
        return str(meta['processed_root'])
    return variant_processed_root(default_processed_base(str(data_root), None), variant)


def build_gt_loaders(meta, data_root, vocab_root, processed_root, batch_size=128):
    from SharedModules.data.vocab import load_vocab
    from SharedModules.data.loader import get_loaders, apply_gt_loaders, TASK_TYPE
    ds = meta['dataset']
    fold = int(meta.get('fold', 0))
    variant = meta.get('vocab_variant')
    vocab = load_vocab(str(vocab_root), ds, variant)
    task_type = TASK_TYPE.get(ds, 'BinaryClass')
    loaders, test_ds, dmeta = get_loaders(
        dataset=ds, data_root=str(data_root), fold=fold, vocab=vocab,
        processed_root=_proc_root(meta, data_root, processed_root, variant),
        batch_size=batch_size, normalize=(task_type == 'Regression'))
    if meta.get('use_gt') and meta.get('gt_cache'):
        gt_tier = meta.get('gt_tier')
        gt_vocab = (meta.get('gt_vocab_variant') or variant).replace('_filter', '')
        loaders, test_ds = apply_gt_loaders(
            loaders, test_ds, gt_cache=meta['gt_cache'], dataset=ds, fold=fold,
            vocab_variant=variant, batch_size=batch_size, gt_vocab_variant=gt_vocab,
            gt_relabel_dir=(f'relabel_{gt_tier}' if gt_tier else None),
            refresh_vocab=vocab if gt_vocab != variant else None,
            fold_motif_lookup=getattr(dmeta, 'motif_lookup', None),
            apply_threshold=getattr(dmeta, 'threshold_pct', None) is not None)
    return loaders, vocab, dmeta, task_type


def split_lists_and_gt(loaders, meta):
    split_lists = {s: list(loaders[s].dataset) for s in SPLITS}
    if meta['dataset'] == 'mutag':
        from SharedModules.data.mutag_splits import mutag_gt_eval_graphs
        return split_lists, {s: mutag_gt_eval_graphs(split_lists[s]) for s in SPLITS}

    def _has(graphs):
        return any(getattr(g, 'node_label', None) is not None for g in graphs[:16])
    if meta.get('use_gt') or _has(split_lists['test']):
        return split_lists, {s: split_lists[s] for s in SPLITS}
    return split_lists, {s: None for s in SPLITS}


def _denorm(dmeta, task_type):
    return (dmeta.norm_mean, dmeta.norm_std) if task_type == 'Regression' else None


# ═════════════════════════════════════════════════════════════════════════════
# NATIVE MODEL RELOAD (inlined)
# ═════════════════════════════════════════════════════════════════════════════

_FAM_DIR = {'mose': 'MOSE-GNN', 'motifsat': 'MotifSAT', 'gsat': 'MotifSAT'}
_CLASH = ('config', 'model', 'run', 'train', 'reg_config', 'losses', 'motif_modules')
_loaded = {'fam': None, 'run': None, 'Config': None}


def _family_module(fam):
    want = _FAM_DIR[fam]
    if _loaded['fam'] == fam:
        return _loaded['run'], _loaded['Config']
    for m in _CLASH:
        sys.modules.pop(m, None)
    for d in ('MOSE-GNN', 'MotifSAT'):
        p = str(_REPO / d)
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(_REPO / want))
    run_mod = importlib.import_module('run')
    cfg_mod = importlib.import_module('config')
    Config = cfg_mod.MOSEConfig if fam == 'mose' else cfg_mod.MotifSATConfig
    _loaded.update(fam=fam, run=run_mod, Config=Config)
    return run_mod, Config


def _cfg_from_meta(Config, meta):
    cfg = Config()
    for k, v in meta.items():
        if v is not None and hasattr(cfg, k):
            try:
                setattr(cfg, k, v)
            except Exception:
                pass
    return cfg


def _flatten_mose_scores(ms):
    if isinstance(ms, dict) and ms and isinstance(next(iter(ms.values())), dict):
        flat = {}
        for chan in ms.values():
            for mid, s in chan.items():
                flat[int(mid)] = float(s)
        return flat
    return {int(m): float(s) for m, s in (ms or {}).items()}


def _torch_load_retry(path, *, tries: int = 4, base_delay: float = 1.5, **kw):
    """torch.load with retry on TRANSIENT read failures.

    Checkpoints live on NFS and this driver runs while hundreds of workers hammer the same
    filesystem. A contended read surfaces as ``PytorchStreamReader failed reading zip archive``
    or an OSError, which is indistinguishable from real corruption at the call site -- MEASURED:
    4 cells reported "corrupt", then loaded fine on retry with the bytes unchanged. Without a
    retry those become silent holes in the results tables that look like missing configs.
    Genuine corruption still fails, just after the retries are exhausted.
    """
    import time
    last = None
    for i in range(tries):
        try:
            return torch.load(path, **kw)
        except (RuntimeError, OSError, EOFError) as e:
            last = e
            if i == tries - 1:
                break
            time.sleep(base_delay * (2 ** i))          # 1.5s, 3s, 6s
    raise RuntimeError(
        f"failed to load {path} after {tries} attempts (last: {type(last).__name__}: {last}). "
        f"If this persists the checkpoint is genuinely unreadable.") from last


def load_native_model_and_scores(run_dir, meta, fam, loaders, vocab, dmeta, device, task_type):
    run_mod, Config = _family_module(fam)
    cfg = _cfg_from_meta(Config, meta)
    sd = _torch_load_retry(run_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    sd = sd.get('model_state_dict', sd) if isinstance(sd, dict) else sd
    if fam == 'mose':
        # STRICT per-fold: the model's motif_params are sized by THIS fold's kept set.
        # Never fall back to vocab.kept_motif_ids (fold-0 mining) — that silently mixes
        # a fold-0 filter into a non-fold-0 run. Fail loudly instead.
        kept = getattr(dmeta, 'kept_motif_ids', None)
        if kept is None:
            raise ValueError(
                f"per-fold kept_motif_ids missing from dmeta for "
                f"{meta.get('dataset')} fold {meta.get('fold')} — refusing to fall back "
                f"to fold-0 vocab.kept_motif_ids")
        model = run_mod.build_model(cfg, vocab.num_motifs, task_type, dmeta, kept_motif_ids=kept)
        model.load_state_dict(sd)
        model = model.to(device).eval()
        scores = _flatten_mose_scores(model.get_motif_scores())
    else:
        model = run_mod.build_model(cfg, task_type, dmeta)
        model.load_state_dict(sd)
        model = model.to(device).eval()
        all_graphs = [g for s in SPLITS for g in list(loaders[s].dataset)]
        agg = run_mod._aggregate_att_to_motif(
            model, all_graphs, device,
            learn_edge_att=getattr(cfg, 'learn_edge_att', False), max_graphs=None)
        scores = {int(m): float(s) for m, s in (agg.get('mean') or {}).items()}
    return model, scores


# ═════════════════════════════════════════════════════════════════════════════
# INLINE EVAL MATH
# ═════════════════════════════════════════════════════════════════════════════

def _np1(x):
    return (x.detach().cpu().view(-1).numpy() if torch.is_tensor(x)
            else np.asarray(x, float).reshape(-1))


def _prob(model, data, device, task_type, node_weights=None):
    out = _model_forward(model, data.clone().to(device), node_weights=node_weights)
    v = out.view(-1)[0]
    return float(torch.sigmoid(v)) if task_type == 'BinaryClass' else float(v)


@torch.no_grad()
def own_impact(model, data_list, device, task_type):
    """Ante-hoc y = faithful LOO impact with W = the model's OWN attention. Returns
    {mid: {'impact': mean, 'support': n, 'per': {gi: val}}} (per-graph for instance corr)."""
    att_fn = model_node_att_fn(model, device)
    mask_cache = {}
    for gi, d in enumerate(data_list):
        n2m = getattr(d, 'nodes_to_motifs', None)
        if n2m is None:
            continue
        n2m = n2m.view(-1)
        for mid in torch.unique(n2m).tolist():
            if int(mid) >= 0:
                mask_cache.setdefault(int(mid), {})[gi] = (n2m == int(mid))
    base_W, p_full = {}, {}
    for gi, d in enumerate(data_list):
        W = att_fn(d)
        if W is None:
            continue
        W = W.view(-1).float().to(device)
        base_W[gi] = W
        p_full[gi] = _prob(model, d, device, task_type, node_weights=W)
    out = {}
    for mid, per in mask_cache.items():
        vals = {}
        for gi, gmask in per.items():
            W = base_W.get(gi)
            if W is None or W.numel() != gmask.numel():
                continue
            Wm = W.clone()
            Wm[gmask.to(Wm.device).bool()] = 0.0
            vals[gi] = abs(p_full[gi] - _prob(model, data_list[gi], device, task_type, node_weights=Wm))
        if vals:
            out[mid] = {'impact': float(np.mean(list(vals.values()))),
                        'support': len(vals), 'per': vals}
    return out


def _wpearson(x, y, w):
    x = np.asarray(x, float); y = np.asarray(y, float); w = np.asarray(w, float)
    W = w.sum()
    if len(x) < 3 or W <= 0:
        return float('nan')
    mx, my = (w * x).sum() / W, (w * y).sum() / W
    vx = (w * (x - mx) ** 2).sum() / W
    vy = (w * (y - my) ** 2).sum() / W
    if vx <= 0 or vy <= 0:
        return float('nan')
    return float((w * (x - mx) * (y - my)).sum() / W / np.sqrt(vx * vy))


def _rank(a):
    a = np.asarray(a, float)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    return ((csum - counts) + csum - 1)[inv] / 2.0


def grouped_corr_variants(rows):
    """4 grouped Pearson/Spearman variants — EXACT schema of split_eval.grouped_corr_variants
    (keys pearson_{w,u}_{incl,excl}unk, spearman_..., n_motifs_*). rows: {motif_id,score,impact,support}."""
    out = {}
    for unk_tag, keep in (('inclunk', lambda r: True),
                          ('exclunk', lambda r: int(r['motif_id']) != UNK_ID)):
        sel = [r for r in rows if keep(r)]
        out[f'n_motifs_{unk_tag}'] = len(sel)
        if len(sel) < 3:
            for w_tag in ('w', 'u'):
                out[f'pearson_{w_tag}_{unk_tag}'] = float('nan')
                out[f'spearman_{w_tag}_{unk_tag}'] = float('nan')
            continue
        x = [r['score'] for r in sel]; y = [r['impact'] for r in sel]
        supp = [max(float(r.get('support', 1.0)), 0.0) for r in sel]
        rx, ry = _rank(x), _rank(y)
        for w_tag, w in (('w', supp), ('u', [1.0] * len(sel))):
            out[f'pearson_{w_tag}_{unk_tag}'] = _wpearson(x, y, w)
            out[f'spearman_{w_tag}_{unk_tag}'] = _wpearson(rx, ry, w)
    return out


def instance_corr(score_by_mg, impact_by_mg, kept, unk):
    """Per-instance corr over matched (motif,graph) points. Keys: pearson_instance,
    spearman_instance, n_instances (mirrors per_instance_correlation_from_caches)."""
    xs, ys = [], []
    for mid in set(score_by_mg) & set(impact_by_mg):
        if unk == 'exclude' and kept is not None and mid not in kept:
            continue
        sm, im = score_by_mg[mid], impact_by_mg[mid]
        for gi in set(sm) & set(im):
            s_i, i_i = sm[gi], im[gi]
            if np.isfinite(s_i) and np.isfinite(i_i):
                xs.append(float(s_i)); ys.append(float(i_i))
    if len(xs) < 3:
        return {'pearson_instance': float('nan'), 'spearman_instance': float('nan'),
                'n_instances': len(xs)}
    ones = [1.0] * len(xs)
    return {'pearson_instance': _wpearson(xs, ys, ones),
            'spearman_instance': _wpearson(_rank(xs), _rank(ys), ones),
            'n_instances': len(xs)}


def score_cache_from_atts(atts_by_gi, graphs):
    """{mid: {gi: mean per-node att over motif mid's atoms in graph gi}} from per-node atts."""
    cache = {}
    for gi, a in atts_by_gi.items():
        if gi >= len(graphs):
            continue
        n2m = _np1(getattr(graphs[gi], 'nodes_to_motifs')).astype(int)
        a = _np1(a)
        if len(a) != len(n2m):
            continue
        for mid in np.unique(n2m):
            if mid < 0:
                continue
            vals = a[n2m == mid]
            if len(vals):
                cache.setdefault(int(mid), {})[gi] = float(vals.mean())
    return cache


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


def _graph_auc(att, labels, keep):
    pos = [k for k in range(len(labels)) if labels[k] and keep[k]]
    neg = [k for k in range(len(labels)) if not labels[k] and keep[k]]
    return _auc(att, pos, neg)


def _per_graph_mean_auc(att_by_i, gl, attr, keep_fn, col=None):
    """Mean over graphs of per-graph AUC(atts, graph.<attr>) — the 'auc_mean' the driver's
    compute_gt_roc produces for a single node-label attribute (cause/spurious/family).
    ``col`` selects one column of a 2-D mask (node_label_spurious [N,S]); a 2-D mask without
    ``col`` is a bug, not something to flatten, so it raises."""
    vals = []
    for i, g in enumerate(gl):
        a = att_by_i.get(i)
        lab = getattr(g, attr, None)
        if a is None or lab is None:
            continue
        if getattr(lab, 'ndim', 1) > 1 and lab.shape[-1] != 1:
            if col is None:
                raise ValueError(f"_per_graph_mean_auc: {attr} is 2-D {tuple(lab.shape)}; "
                                 f"pass col= to select one column.")
            lab = lab[:, col]
        v = _graph_auc(_np1(a), _np1(lab).astype(bool), keep_fn(g))
        if v == v:
            vals.append(v)
    return float(np.mean(vals)) if vals else float('nan')


def _contrast_vs_cause(att_by_i, gl, keep_fn, attr, col=None):
    """Per-graph AUC of a diagnostic motif's atoms against THE ANNOTATED CAUSE.

    SpurROC/FamROC ask whether an attribution ranks a competing motif above the cause
    it should have found. That is a contrast between two designated sets, so the
    negatives are the causal atoms -- NOT the molecule's complement (which contains
    the cause, so a correct attribution scores below 0.5 mechanically) and NOT the
    background with the cause removed (which deletes the reference the question is
    about, reducing it to "prefers the motif to inert carbon").

    Negatives are the UNION of all fired clauses, so the score asks whether the
    competitor outranks ANY part of the cause. Scoring against a single clause would
    make the diagnostic depend on which clause the correctness score happened to
    select; the union lets it stand on its own and surfaces partial substitution.

      > 0.5  the competing motif outranks the cause   -> substitution
      = 0.5  no preference between them
      < 0.5  the cause outranks the competitor        -> correct

    Undefined, and skipped, where no clause fired: the contrast has no meaning
    without a cause present.
    """
    vals = []
    for i, g in enumerate(gl):
        a = att_by_i.get(i)
        nlc = getattr(g, 'node_label_clauses', None)
        lab = getattr(g, attr, None)
        if a is None or nlc is None or lab is None:
            continue
        nlc = nlc.detach().cpu().numpy() if torch.is_tensor(nlc) else np.asarray(nlc)
        if getattr(lab, 'ndim', 1) > 1 and lab.shape[-1] != 1:
            if col is None:
                raise ValueError(f'_contrast_vs_cause: {attr} is 2-D; pass col=')
            lab = lab[:, col]
        att, keep = _np1(a), keep_fn(g)

        clauses = [c for c in (np.where(nlc[:, j] > 0)[0].tolist()
                               for j in range(nlc.shape[1])) if c]
        if not clauses:
            continue                                    # no cause -> contrast undefined
        U = set().union(*[set(c) for c in clauses])     # the annotated cause
        neg = [k for k in sorted(U) if keep[k]]
        # fragmentation is atom-disjoint so these should not overlap; enforce it anyway
        pos = [k for k in np.where(_np1(lab) > 0)[0].tolist() if keep[k] and k not in U]
        if not pos or not neg:
            continue                                    # molecule lacks the motif
        v = _auc(att, pos, neg)                         # positives=motif, negatives=cause
        if v == v:
            vals.append(v)
    return float(np.mean(vals)) if vals else float('nan')


def _dnf_gtroc(att_by_i, gl, keep_fn):
    """Per-graph-mean DNF GT-ROC: instance = mean of best fired clause AUC, global = mean of
    union-of-clauses AUC (matches compute_dnf_gt_roc / gtroc_masked — both per-graph means)."""
    inst_l, glob_l = [], []
    for i, g in enumerate(gl):
        a = att_by_i.get(i)
        nlc = getattr(g, 'node_label_clauses', None)
        if a is None or nlc is None:
            continue
        nlc = nlc.detach().cpu().numpy() if torch.is_tensor(nlc) else np.asarray(nlc)
        att = _np1(a); keep = keep_fn(g)
        clauses = [c for c in (np.where(nlc[:, j] > 0)[0].tolist()
                               for j in range(nlc.shape[1])) if c]
        if not clauses:
            continue
        U = set().union(*[set(c) for c in clauses])
        neg = [k for k in range(nlc.shape[0]) if k not in U and keep[k]]
        per = [v for v in (_auc(att, [k for k in c if keep[k]], neg) for c in clauses) if v == v]
        if per:
            inst_l.append(max(per))
        gv = _auc(att, [k for k in sorted(U) if keep[k]], neg)
        if gv == gv:
            glob_l.append(gv)
    return (float(np.mean(inst_l)) if inst_l else float('nan'),
            float(np.mean(glob_l)) if glob_l else float('nan'), len(inst_l))


def gtroc_all(att_by_i, gl, keep_fn):
    """Full node-level GT-ROC set (faithful port of split_eval.gt_roc_block, minus edge-level
    which the tables don't read): whole / fired / spurious / family + DNF instance/global.
    Every AUC is per-graph-mean; keep_fn drops UNK atoms for --unk exclude."""
    out = {}

    # Require a POSITIVE entry, not merely a present tensor. apply_gt attaches
    # node_label_spurious AND node_label_family on EVERY graph, all-zero when the rule has no
    # class-1 correlate / no look-alikes, so an is-not-None guard emits a junk all-NaN column.
    # Mirrors pipeline._has_node_attr, which is what the training-time paths use.
    def _has_pos(attr):
        for g in gl:
            v = getattr(g, attr, None)
            if v is not None and float(_np1(v).sum()) > 0:
                return True
        return False

    # node_label IS the fired-clause cause (2026-08); there is no separate 'fired' metric.
    # It stays on is-not-None DELIBERATELY: it always carries positives on rule-positive graphs,
    # so an all-zero cause mask means the cell is broken and a NaN here is the honest signal.
    if any(getattr(g, 'node_label', None) is not None for g in gl):
        out['gt_roc_node_auc_mean'] = _per_graph_mean_auc(att_by_i, gl, 'node_label', keep_fn)
    # family, unlike the cause, is legitimately empty for many rules -> positive-entry guard.
    # family / spurious are CONTRASTS AGAINST THE CAUSE, not against the complement
    if _has_pos('node_label_family'):
        out['family_roc_node_auc_mean'] = _contrast_vs_cause(
            att_by_i, gl, keep_fn, 'node_label_family')
    # SPURIOUS is [N,S]: one AUC per class-1 correlate, keyed by its motif. Never collapsed —
    # spurious_roc_node_auc_mean is only the max, kept so legacy one-column tables resolve.
    if _has_pos('node_label_spurious'):
        names = spurious_motif_names(gl)
        finite = []
        for j, nm in enumerate(names):
            v = _contrast_vs_cause(att_by_i, gl, keep_fn, 'node_label_spurious', col=j)
            out[f'spurious_roc_node_auc_mean__{nm}'] = v
            if v == v:
                finite.append(v)
        out['spurious_roc_node_auc_mean'] = max(finite) if finite else float('nan')
        out['spurious_motifs'] = list(names)
    inst, glob, n = _dnf_gtroc(att_by_i, gl, keep_fn)
    if n > 0:                                          # planted DNF
        out['instance_gt_roc_node_auc_mean'] = inst
        out['global_gt_roc_node_auc_mean'] = glob
        out['dnf_n_graphs'] = n
    elif 'gt_roc_node_auc_mean' in out:               # source single-clause alias
        out['instance_gt_roc_node_auc_mean'] = out['gt_roc_node_auc_mean']
        out['global_gt_roc_node_auc_mean'] = out['gt_roc_node_auc_mean']
    return out


# ═════════════════════════════════════════════════════════════════════════════
# WRITERS (inline — EXACT schema of split_eval, so build_metric_set reads them)
# ═════════════════════════════════════════════════════════════════════════════

def _w_importance(d, stem, split, rows):
    if rows:
        pd.DataFrame([{'motif_id': r['motif_id'], 'score': r['score'],
                       'support': r.get('support', 0), 'motif_smarts': r.get('motif_smarts', '')}
                      for r in rows]).to_csv(d / f'{stem}_importance_{split}.csv', index=False)


def _w_impact(d, stem, split, rows):
    if rows:
        pd.DataFrame([{'motif_id': r['motif_id'], 'impact': r['impact'],
                       'support': r.get('support', 0), 'motif_smarts': r.get('motif_smarts', '')}
                      for r in rows]).to_csv(d / f'{stem}_impact_{split}.csv', index=False)


def _w_grouped(d, stem, rows_by_split):
    all_rows = [dict(r, split=s) for s in SPLITS for r in rows_by_split.get(s, [])]
    test_rows = [dict(r, split='test') for r in rows_by_split.get('test', [])]
    for tag, rws in (('alltest', all_rows), ('testonly', test_rows)):
        pd.DataFrame([{'method': stem, 'pooling': tag, **grouped_corr_variants(rws)}]).to_csv(
            d / f'{stem}_grouped_corr_pooled_{tag}.csv', index=False)


def _w_instance(d, stem, inst_by_split):
    for s in SPLITS:
        inst = inst_by_split.get(s)
        if inst:
            pd.DataFrame([{'method': stem, 'split': s, **inst}]).to_csv(
                d / f'{stem}_instance_corr_{s}.csv', index=False)


def _w_global(d, stem, rows_by_split):
    for kind, col in (('importance', 'score'), ('impact', 'impact')):
        rows = [{'method': stem, 'split': s, 'motif_id': r['motif_id'], col: r.get(col),
                 'support': r.get('support', 0), 'motif_smarts': r.get('motif_smarts', '')}
                for s in SPLITS for r in rows_by_split.get(s, [])]
        if rows:
            pd.DataFrame(rows).to_csv(d / f'{kind}_global.csv', index=False)


def _w_summary_splits(d, stem, summary):
    p = d / 'summary_splits.json'
    prev = {}
    if p.exists():
        try:
            prev = json.loads(p.read_text())
        except Exception:
            prev = {}
    prev[stem] = summary
    p.write_text(json.dumps(prev, indent=2))


# ═════════════════════════════════════════════════════════════════════════════
# SAVED-ARTIFACT READERS (post-hoc)
# ═════════════════════════════════════════════════════════════════════════════

def read_pooled_rows(run_dir, stem):
    """Per-split per-motif rows {motif_id,score,impact,support,motif_smarts} from the SAVED
    {stem}_importance/impact_{split}.csv. Returns {split: [rows]}."""
    by_split = {}
    for s in SPLITS:
        fi = Path(run_dir) / f'{stem}_importance_{s}.csv'
        fp = Path(run_dir) / f'{stem}_impact_{s}.csv'
        if not (fi.exists() and fp.exists()):
            by_split[s] = []
            continue
        imp = pd.read_csv(fi)
        imc = pd.read_csv(fp)
        sm_col = ['motif_smarts'] if 'motif_smarts' in imp.columns else []
        m = imp[['motif_id', 'score', 'support'] + sm_col].merge(
            imc[['motif_id', 'impact']], on='motif_id', how='inner')
        by_split[s] = [dict(motif_id=int(r['motif_id']), score=float(r['score']),
                            impact=float(r['impact']), support=float(r['support']),
                            motif_smarts=(str(r['motif_smarts']) if sm_col else ''))
                       for _, r in m.iterrows()]
    return by_split


def read_saved_atts(run_dir, stem):
    """explainer_importances.json -> {split: {graph_idx: np atts}} for method `stem`."""
    p = Path(run_dir) / 'explainer_importances.json'
    if not p.exists():
        return {s: {} for s in SPLITS}
    ibs = json.loads(p.read_text()).get('importances_by_split', {})
    return {s: {int(gi): np.asarray(v, float) for gi, v in ((ibs.get(s) or {}).get(stem) or {}).items()}
            for s in SPLITS}


def read_impact_cache(run_dir, split):
    """impact_cache_agnostic_{split}.json -> {mid: {gi: impact}} (post-hoc per-instance y)."""
    p = Path(run_dir) / f'impact_cache_agnostic_{split}.json'
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    return {int(m): {int(g): float(v) for g, v in gm.items()} for m, gm in raw.items()}


def read_producer_pred(run_dir, stem):
    """Predictive scalars per split from the producer's summary_splits.json (post-hoc)."""
    p = Path(run_dir) / 'summary_splits.json'
    if not p.exists():
        return {}
    d = json.loads(p.read_text()).get(stem, {})
    keys = ('auc', 'rmse', 'mae', 'rmse_orig', 'mae_orig')
    return {s: {k: d[s].get(k) for k in keys} for s in SPLITS if isinstance(d.get(s), dict)}


# ═════════════════════════════════════════════════════════════════════════════
# PER-RUN
# ═════════════════════════════════════════════════════════════════════════════

def _keep_fn(kept, unk):
    def fn(g):
        n2m = _np1(getattr(g, 'nodes_to_motifs')).astype(int)
        if unk == 'include' or kept is None:
            return np.ones(len(n2m), dtype=bool)
        return np.array([m in kept for m in n2m], dtype=bool)
    return fn


def _load_vanilla(run_dir, meta, dmeta, device):
    """Rebuild VanillaGNN + load best_model.pt — used ONLY to reconstruct MAGE's mage_v2
    per-node atts (the producer persisted mage_official atts, not mage_v2)."""
    from SharedModules.baselines.vanilla_gnn import VanillaGNN
    from SharedModules.data.loader import NUM_CLASSES
    model = VanillaGNN(
        x_dim=dmeta.x_dim, hidden_dim=int(meta.get('hidden_dim', 64)),
        num_layers=int(meta.get('num_layers', 3)), backbone=meta['backbone'],
        node_encoder=meta.get('node_encoder', 'onehot'),
        apply_layer_norm=bool(meta.get('apply_layer_norm', False)),
        dropout=float(meta.get('dropout', 0.0) or 0.0),
        conv_normalize=meta.get('conv_normalize', 'none'),
        graph_pool=meta.get('graph_pool', 'add'),
        gin_inner_bn=bool(meta.get('gin_inner_bn', True)),
        num_classes=NUM_CLASSES.get(meta['dataset'], 1), deg=getattr(dmeta, 'deg', None))
    model.load_state_dict(_torch_load_retry(Path(run_dir) / 'best_model.pt',
                                            map_location='cpu', weights_only=False))
    return model.to(device).eval()


def mage_reload_atts(run_dir, art_dir, meta, dmeta, vocab, split_lists, device, task_type):
    """mage_v2 per-node atts by RELOADING the saved mage_attention.pt (NO re-fit) and re-scoring
    attention_mean, then broadcasting to nodes. Needed because posthoc_v1 persisted mage_official
    (label-leaking) atts to explainer_importances.json, not mage_v2. {split: {gi: np atts}}."""
    from SharedModules.baselines.mage import _MotifAttention, score_mage
    from SharedModules.evaluation.split_eval import _pi_to_node_atts
    p = Path(art_dir) / 'mage_attention.pt'
    if not p.exists():
        return {s: {} for s in SPLITS}
    sd = _torch_load_retry(p, map_location='cpu', weights_only=False)
    attn = _MotifAttention(sd['W.weight'].shape[0])
    attn.load_state_dict(sd)
    attn = attn.to(device).eval()
    model = _load_vanilla(run_dir, meta, dmeta, device)
    out = {}
    for s in SPLITS:
        sl = split_lists.get(s) or []
        _, pi = score_mage(model, attn, sl, vocab, device, task_type, return_per_instance=True,
                           verbose=False, score_mode='attention_mean', embed_batch_size=256)
        out[s] = {gi: _np1(a) for gi, a in _pi_to_node_atts(pi, sl).items()}
    return out


def _gtroc_summary(att_by_split, gt, split_lists, keep, unk):
    """{split: {instance_gt_roc_node_auc_mean, global_gt_roc_node_auc_mean, gt_roc_node_auc_mean}}.
    att_by_split[s] keyed by split-list index; realign to the GT graph list per split."""
    out = {}
    for s in SPLITS:
        gl = gt.get(s)
        if not gl:
            continue
        asplit = att_by_split.get(s, {})
        pos = {id(g): j for j, g in enumerate(split_lists[s])}
        att_by_i = {}
        for i, g in enumerate(gl):
            j = pos.get(id(g))
            if j is not None and j in asplit:
                att_by_i[i] = asplit[j]
        if not att_by_i:
            continue
        out[s] = gtroc_all(att_by_i, gl, keep)
    return out


def eval_run(method, run_dir, art_dir, meta, args, device, kept, dest_dir):
    stem = DISK_STEM.get(method, method)
    dest_dir.mkdir(parents=True, exist_ok=True)      # before any write (ante-hoc atts cache)
    loaders, vocab, dmeta, tt = build_gt_loaders(
        meta, args.data_root, args.vocab_root, args.processed_root, args.loader_batch_size)
    split_lists, gt = split_lists_and_gt(loaders, meta)
    keep = _keep_fn(kept, args.unk)
    do_gtroc = args.gt_tier in ('source', 'planted')

    rows_by_split, inst_by_split, summary = {}, {}, {}
    motif_list = getattr(vocab, 'motif_list', [])

    if method in NATIVE:
        model, native = load_native_model_and_scores(
            run_dir, meta, method, loaders, vocab, dmeta, device, tt)
        if not native:
            raise RuntimeError('empty motif scores')
        # per-graph node scores: check dest cache, else forward + write (deterministic)
        atts_path = dest_dir / 'explainer_importances.json'
        if atts_path.exists():
            att_by_split = read_saved_atts(dest_dir, stem)
        else:
            base = model_node_att_fn(model, device)
            att_by_split = {s: {gi: _np1(base(g)) for gi, g in enumerate(split_lists[s])
                                if base(g) is not None} for s in SPLITS}
            atts_path.write_text(json.dumps({'importances_by_split': {
                s: {stem: {int(gi): a.tolist() for gi, a in att_by_split[s].items()}}
                for s in SPLITS}}))
        for s in SPLITS:
            sl = split_lists.get(s)
            if not sl:
                continue
            imp = own_impact(model, sl, device, tt)
            rows = []
            for mid, sc in native.items():
                mid = int(mid)
                if mid not in imp:
                    continue
                if args.unk == 'exclude' and kept is not None and mid not in kept:
                    continue
                rows.append(dict(motif_id=mid, score=float(sc), impact=imp[mid]['impact'],
                                 support=imp[mid]['support'],
                                 motif_smarts=(str(motif_list[mid]) if mid < len(motif_list) else '')))
            rows_by_split[s] = rows
            sc_cache = score_cache_from_atts(att_by_split.get(s, {}), sl)
            ic_cache = {mid: imp[mid]['per'] for mid in imp}
            inst_by_split[s] = instance_corr(sc_cache, ic_cache, kept, args.unk)
            pred = evaluate_predictions(model, loaders[s], device, tt, denorm=_denorm(dmeta, tt))
            summary[s] = {'auc': pred.get('auc', pred.get('auc_mean', float('nan'))),
                          'rmse': pred.get('rmse', float('nan')), 'mae': pred.get('mae', float('nan')),
                          'rmse_orig': pred.get('rmse_orig', float('nan')),
                          'mae_orig': pred.get('mae_orig', float('nan'))}
    else:  # POST-HOC — read saved artifacts, no model, no re-run
        by_split = read_pooled_rows(art_dir, stem)
        att_by_split = read_saved_atts(art_dir, stem)
        if method == 'mage' and do_gtroc and not any(att_by_split.get(s) for s in SPLITS):
            # producer saved mage_official atts, not mage_v2 -> reconstruct via saved attention
            att_by_split = mage_reload_atts(run_dir, art_dir, meta, dmeta, vocab,
                                            split_lists, device, tt)
        pred_by_split = read_producer_pred(art_dir, stem)
        for s in SPLITS:
            rows = by_split.get(s, [])
            if args.unk == 'exclude' and kept is not None:
                rows = [r for r in rows if int(r['motif_id']) in kept]
            rows_by_split[s] = rows
            ic_cache = read_impact_cache(art_dir, s)
            if args.unk == 'exclude' and kept is not None:
                ic_cache = {m: gm for m, gm in ic_cache.items() if m in kept}
            sc_cache = score_cache_from_atts(att_by_split.get(s, {}), split_lists.get(s, []))
            inst_by_split[s] = instance_corr(sc_cache, ic_cache, kept, args.unk)
            summary[s] = dict(pred_by_split.get(s, {}))

    if do_gtroc:
        for s, g in _gtroc_summary(att_by_split, gt, split_lists, keep, args.unk).items():
            summary.setdefault(s, {}).update(g)

    # write the full driver artifact set
    dest_dir.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        _w_importance(dest_dir, stem, s, rows_by_split.get(s, []))
        _w_impact(dest_dir, stem, s, rows_by_split.get(s, []))
    _w_grouped(dest_dir, stem, rows_by_split)
    _w_instance(dest_dir, stem, inst_by_split)
    _w_global(dest_dir, stem, rows_by_split)
    _w_summary_splits(dest_dir, stem, summary)

    g_all = grouped_corr_variants([dict(r) for s in SPLITS for r in rows_by_split.get(s, [])])
    tsum = summary.get('test', {})
    # SpurROC / FamROC are surfaced in the ROLLUP as well as summary_splits.json. Without
    # this they existed only inside per-run JSON, so no tabular path carried them: the three
    # summary.json harvesters lost them when pipeline.py stopped emitting the (differently
    # defined) complement-based version, and this rollup never had them. spurious_roc is the
    # MAX over class-1 correlates; the per-motif values stay in summary_splits.json.
    _spur = [v for k, v in tsum.items()
             if k.startswith('spurious_roc_node_auc_mean__') and isinstance(v, float) and v == v]
    return dict(grouped_pearson_u=g_all['pearson_u_exclunk'],
                grouped_pearson_w=g_all['pearson_w_exclunk'],
                n_motifs=g_all['n_motifs_exclunk'],
                gtroc_global=tsum.get('global_gt_roc_node_auc_mean', float('nan')),
                gtroc_instance=tsum.get('instance_gt_roc_node_auc_mean', float('nan')),
                spurious_roc=(max(_spur) if _spur else float('nan')),
                n_spurious=len(_spur),
                family_roc=tsum.get('family_roc_node_auc_mean', float('nan')),
                pred_auc=tsum.get('auc', float('nan')))


def _gt_rule(meta, run_dir=''):
    """Planted DNF rule id. LAYOUT B (ante-hoc) encodes it in vocab_variant
    (`..._relabelled_dnf_kK_rR`); LAYOUT A (baselines) encodes it in the run-id path
    (`..._gt_dnf_kK_rR`). Check both."""
    for s in (str(meta.get('vocab_variant', '')), str(run_dir)):
        m = re.search(r'_(?:relabelled_)?dnf_k(\d+)_r(\d+)', s)
        if m:
            return f'dnf_k{m.group(1)}_r{m.group(2)}'
    return ''


def _run_tier(meta, run_dir=''):
    # planted runs carry use_gt=True (both layouts); source-GT datasets use native node_label.
    if meta.get('use_gt') or _gt_rule(meta, run_dir):
        return 'planted'
    return 'source' if meta['dataset'] in SOURCE_GT else 'none'


def _base_vocab(v):
    v = re.sub(r'_relabelled_dnf_k\d+_r\d+$', '', v)
    return v[:-len('_filter')] if v.endswith('_filter') else v


def _dest_dir(run_dir, args, want_base):
    """Mirror the run path under dest_root; --unk exclude lands under the sibling *_filter
    vocab segment so build_metric_set harvests it as the filtered variant."""
    rel = list(Path(run_dir).relative_to(args.ckpt_root).parts)
    if args.unk == 'exclude':
        for i, p in enumerate(rel):
            if _base_vocab(p) == want_base and not p.endswith('_filter'):
                m = re.search(r'(_relabelled_dnf_k\d+_r\d+)$', p)
                b, suf = (p[:m.start()], m.group(1)) if m else (p, '')
                rel[i] = b + '_filter' + suf
                break
    return Path(args.dest_root).joinpath(*rel)


def _iter_runs(ckpt_root, family_dir, dataset):
    base = Path(ckpt_root) / family_dir
    for ss in sorted(base.rglob('summary.json')):
        rd = ss.parent
        if any(p.startswith(('archive', 'tune')) for p in rd.parts):
            continue
        if not (rd / 'best_model.pt').exists():
            continue
        try:
            meta = json.loads(ss.read_text())
        except Exception:
            continue
        if meta.get('dataset') == dataset:
            yield rd, meta


def _run_backbone(meta, run_dir):
    """A run's backbone (GAT/GCN/GIN/PNA/SAGE). Prefers meta['backbone']; falls back to the
    run-dir name token (native 'GAT_...' or baseline 'bb-GAT_...')."""
    bb = meta.get('backbone')
    if bb:
        return str(bb)
    tok = getattr(run_dir, 'name', str(run_dir)).split('_')[0]
    return tok[3:] if tok.startswith('bb-') else tok


def _run_unk_mode(meta, run_dir):
    """A run's trained UNK mode ('fixed' | 'learnable_shared' | None). Prefers meta,
    falls back to the run-dir name token (unk-fixed / unk-learnable_shared)."""
    m = meta.get('unk_mode')
    if m:
        return str(m)
    name = getattr(run_dir, 'name', str(run_dir))
    if 'unk-learnable_shared' in name:
        return 'learnable_shared'
    if 'unk-fixed' in name:
        return 'fixed'
    return None


def main():
    ap = argparse.ArgumentParser(description='THE single source of truth for explainability eval')
    ap.add_argument('--method', required=True, choices=sorted(POSTHOC | NATIVE))
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--vocab', required=True, help='e.g. rbrics (rbrics_filter selects MoSE-filter)')
    ap.add_argument('--gt_tier', required=True, choices=['none', 'source', 'planted'])
    ap.add_argument('--unk', required=True, choices=['include', 'exclude'])
    ap.add_argument('--unk_mode', default='any', choices=['any', 'fixed', 'learnable_shared'],
                    help='MoSE run selection by trained UNK mode: "fixed" -> mose, '
                         '"learnable_shared" -> mose_U. Default "any" keeps all runs. '
                         'Needed to isolate mose_U in ablated_completely_v1 (which co-locates '
                         'fixed + learnable arms).')
    ap.add_argument('--ckpt_root', required=True, help='checkpoint tree (e.g. final_v2) — enumerated')
    ap.add_argument('--artifacts_root', default=None,
                    help='post-hoc: tree with the SAVED explanation outputs (e.g. posthoc_v1); '
                         'run dir mirrors ckpt_root. Required for post-hoc methods.')
    ap.add_argument('--dest_root', required=True, help='per-run artifact output tree (mirrors ckpt_root)')
    ap.add_argument('--dest', default=None, help='optional rolled-up metrics CSV (one row per run)')
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--vocab_root', required=True)
    ap.add_argument('--processed_root', default=None)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--loader_batch_size', type=int, default=128)
    ap.add_argument('--fold', nargs='*', type=int, default=None,
                    help='restrict to these fold indices (default all) — shard folds across jobs')
    ap.add_argument('--backbone', nargs='*', default=None,
                    help='restrict to these backbones (e.g. GCN GIN PNA SAGE). Used to skip the '
                         'old 4-head GAT in final_v2: run non-GAT here, GAT from final_v2_gath1. '
                         'Planted needs NO filter (planted_v2 GAT is already heads=1, in-place).')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    if args.method in POSTHOC and not args.artifacts_root:
        raise SystemExit('post-hoc methods READ saved atts/CSVs — pass --artifacts_root (e.g. posthoc_v1)')
    device = torch.device('cuda' if (args.device == 'cuda'
                          or (args.device == 'auto' and torch.cuda.is_available())) else 'cpu')
    want_base = _base_vocab(args.vocab)
    want_filter = args.vocab.endswith('_filter')

    print(f'method={args.method} dataset={args.dataset} vocab={args.vocab} '
          f'gt_tier={args.gt_tier} unk={args.unk} device={device}')

    out_rows, ok, failed = [], 0, 0
    # The per-cell write below is append-mode, so clear any prior rollup ONCE up front —
    # otherwise a re-run duplicates every row instead of replacing them.
    if args.dest:
        Path(args.dest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dest).unlink(missing_ok=True)
    for run_dir, meta in _iter_runs(args.ckpt_root, FAMILY_DIR[args.method], args.dataset):

        v = str(meta.get('vocab_variant', ''))
        if _base_vocab(v) != want_base or v.endswith('_filter') != want_filter:
            continue
        if _run_tier(meta, run_dir) != args.gt_tier:
            continue
        if args.fold is not None and meta.get('fold') not in args.fold:
            continue
        if args.unk_mode != 'any' and _run_unk_mode(meta, run_dir) != args.unk_mode:
            continue
        if args.backbone is not None and _run_backbone(meta, run_dir) not in args.backbone:
            continue
        if args.limit and (ok + failed) >= args.limit:
            break
        tag = f"{args.dataset}/{args.method}/{meta.get('backbone')}/fold{meta.get('fold')}/{_gt_rule(meta, run_dir) or '-'}"
        try:
            art_dir = (Path(args.artifacts_root) / run_dir.relative_to(args.ckpt_root)
                       if args.artifacts_root else run_dir)
            dest_dir = _dest_dir(run_dir, args, want_base)

            kept = None
            if args.unk == 'exclude':
                from SharedModules.data.vocab import load_vocab
                from SharedModules.data.fold_threshold import build_fold_annotation
                from SharedModules.data.dataset_schema import DATASET_COLUMN
                _wb = _base_vocab(args.vocab)
                _filt = load_vocab(str(args.vocab_root), args.dataset, _wb + '_filter')
                meta_fold = meta.get('fold')
                if meta_fold is not None:
                    meta_fold_int = int(meta_fold)
                else:
                    raise ValueError(f"no fold in meta")
                _csv = Path(args.data_root) / f"{args.dataset}_{meta_fold_int}.csv"
                _, kept, _, _ = build_fold_annotation(
                    lookup_all=_filt.lookup_all, motif_list=_filt.motif_list,
                    mol_fragment_smarts=_filt.mol_fragment_smarts, csv_path=str(_csv),
                    label_col=DATASET_COLUMN[args.dataset], dataset=args.dataset,
                    variant=_filt.variant or (_wb + '_filter'),
                    vocab_dir=Path(_filt.vocab_dir) if _filt.vocab_dir else Path('.'),
                    apply_threshold=True, threshold_pct=_filt.threshold_pct)
                if kept is None:
                    raise ValueError(f"no per-fold kept for {args.dataset} fold {meta.get('fold')}")
                kept = set(int(x) for x in kept)

            m = eval_run(args.method, run_dir, art_dir, meta, args, device, kept, dest_dir)
            out_rows.append(dict(
                dataset=args.dataset, method=args.method, backbone=meta.get('backbone'),
                fold=meta.get('fold'), vocab=args.vocab, gt_rule=_gt_rule(meta, run_dir),
                gt_tier=args.gt_tier, unk=args.unk, unk_mode=_run_unk_mode(meta, run_dir), **m,
                run_path=str(run_dir.relative_to(args.ckpt_root))))
            print(f'  [ok] {tag}')
            ok += 1
            # INCREMENTAL WRITE: append this cell to the rollup NOW rather than accumulating
            # and writing once at the end. A single evaluate.py run scores hundreds of cells;
            # with an end-only write a preempted or still-running task contributes NOTHING --
            # MEASURED: a MoSE task 86/172 cells in had produced no output at all, which reads
            # as "the method is missing" rather than "the task is halfway". Appending makes the
            # run both OBSERVABLE (partial results are usable) and RESUMABLE-friendly.
            if args.dest:
                _dp = Path(args.dest); _dp.parent.mkdir(parents=True, exist_ok=True)
                _hdr = not _dp.exists() or _dp.stat().st_size == 0
                pd.DataFrame([out_rows[-1]]).to_csv(_dp, mode='a', header=_hdr, index=False)
        except Exception as e:
            print(f'  [FAIL] {tag}: {type(e).__name__}: {e}')
            failed += 1

    print(f'\nDONE ok={ok} failed={failed}  artifacts -> {args.dest_root}'
          + (f'  rollup -> {args.dest}' if args.dest else ''))
    if out_rows:
        df = pd.DataFrame(out_rows)
        print(df[['backbone', 'fold', 'gt_rule', 'n_motifs',
                  'grouped_pearson_u', 'gtroc_global']].to_string(index=False))


if __name__ == '__main__':
    main()
