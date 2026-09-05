"""split_eval.py — per-split (train / valid / test) explainability re-evaluation.

Plan-2 driver-side helper. Reloaded checkpoints are evaluated on EACH split
separately (not just test + pooled-all), for BOTH ante-hoc (MoSE/MotifSAT/GSAT)
and post-hoc (GNNExplainer/PGExplainer/Motif-Occlusion/MAGE) methods, producing:

  importance_{split}.csv         per-motif attribution (x-axis), per split
  impact_{split}.csv             per-motif faithful-LOO impact (y-axis), per split
  grouped_corr_pooled_alltest.csv    grouped (per-motif) Pearson/Spearman pooled
                                     over train+valid+test — 4 variants
  grouped_corr_pooled_testonly.csv   same, test split only — 4 variants
  instance_corr_{split}.csv      per-instance (motif,graph) Pearson/Spearman, 3 files
  summary_splits.json            all GT-ROC (fired-clause cause / per-motif spurious /
                                 family / DNF global / DNF instance) + AUC/RMSE per split
  importance_global.csv / impact_global.csv   pooled over splits AND explainers

DESIGN — Option B (reuse, don't rewrite): every metric comes from the EXISTING
motif_eval / embedding_viz primitives (``compute_motif_impact``,
``score_impact_correlation``, ``per_instance_correlation_from_caches``,
``compute_gt_roc``, ``compute_dnf_gt_roc``). This module only orchestrates them
per split + writes the files; ``run_vanilla.py`` / ``pipeline.py`` are untouched.

The 4 grouped-correlation variants are {weighted, unweighted} × {incl-UNK,
excl-UNK}: weighted uses per-motif support (``n_graphs``) as the weight;
excl-UNK drops the below-fold-threshold motif id (default -1).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .metrics import evaluate_predictions
from .motif_eval import (
    compute_motif_impact,
    compute_gt_roc,
    compute_dnf_gt_roc,
    caches_from_motif_impact,
    per_instance_correlation_from_caches,
    build_graph_mask_cache,
    build_motif_score_cache_from_atts,
    model_node_att_fn,
)
from .pipeline import _has_node_attr

SPLITS = ('train', 'valid', 'test')
UNK_ID = -1                       # below-fold-threshold motif id (excluded when excl-UNK)


# ─────────────────────────────────────────────────────────────────────────────
# Correlation helpers (weighted Pearson/Spearman; the 4 grouped variants)
# ─────────────────────────────────────────────────────────────────────────────

def _wpearson(x, y, w) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float); w = np.asarray(w, float)
    W = w.sum()
    if len(x) < 3 or W <= 0:
        return float('nan')
    mx = (w * x).sum() / W
    my = (w * y).sum() / W
    vx = (w * (x - mx) ** 2).sum() / W
    vy = (w * (y - my) ** 2).sum() / W
    if vx <= 0 or vy <= 0:
        return float('nan')
    cov = (w * (x - mx) * (y - my)).sum() / W
    return float(cov / np.sqrt(vx * vy))


def _rank(a) -> np.ndarray:
    """Average-rank transform (ties → mean rank), for a (weighted) Spearman."""
    a = np.asarray(a, float)
    order = a.argsort()
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(len(a), dtype=float)
    # average tied ranks
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum - 1) / 2.0
    return avg[inv]


def grouped_corr_variants(rows: List[dict]) -> Dict[str, float]:
    """4 grouped Pearson/Spearman variants from per-motif rows.

    ``rows`` : list of {motif_id, score, impact, support}. Returns keys
    ``pearson_{w,u}_{incl,excl}unk`` and ``spearman_...`` plus ``n_motifs_*``.
    """
    out: Dict[str, float] = {}
    for unk_tag, keep in (('inclunk', lambda r: True),
                          ('exclunk', lambda r: int(r['motif_id']) != UNK_ID)):
        sel = [r for r in rows if keep(r)]
        out[f'n_motifs_{unk_tag}'] = len(sel)
        if len(sel) < 3:
            for w_tag in ('w', 'u'):
                out[f'pearson_{w_tag}_{unk_tag}'] = float('nan')
                out[f'spearman_{w_tag}_{unk_tag}'] = float('nan')
            continue
        x = [r['score'] for r in sel]
        y = [r['impact'] for r in sel]
        supp = [max(float(r.get('support', 1.0)), 0.0) for r in sel]
        rx, ry = _rank(x), _rank(y)
        for w_tag, w in (('w', supp), ('u', [1.0] * len(sel))):
            out[f'pearson_{w_tag}_{unk_tag}'] = _wpearson(x, y, w)
            out[f'spearman_{w_tag}_{unk_tag}'] = _wpearson(rx, ry, w)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# GT-ROC block (fired-clause cause / per-motif spurious / family / DNF) for one graph list
# ─────────────────────────────────────────────────────────────────────────────

def gt_roc_block(model, gt_list, device, node_att_fn) -> Dict[str, float]:
    """All GT-ROC scalars for one graph list under a given node-attribution fn.
    Mirrors the ante-hoc/post-hoc GT-ROC set incl. DNF global/instance with the
    source-GT single-clause alias (node_label → global/instance when no clause
    tensor is present). Empty dict when the list carries no GT labels.

    node_label is the FIRED-CLAUSE cause (the whole-rule Mode-1 GT was removed 2026-08), so
    gt_roc_node_auc_mean now means fired-clause recovery — NOT comparable with the same key in
    pre-2026-08 archives. The spurious audit is per-motif: one AUC per class-1 correlate."""
    b: Dict[str, float] = {}
    if not _has_node_attr(gt_list, 'node_label') and not _has_node_attr(gt_list, 'edge_label'):
        return b
    b['gt_roc_node_auc_mean'] = compute_gt_roc(model, gt_list, device,
                                               node_att_fn=node_att_fn, level='node')['auc_mean']
    b['gt_roc_edge_auc_mean'] = compute_gt_roc(model, gt_list, device,
                                               node_att_fn=node_att_fn, level='edge')['auc_mean']
    # SpurROC / FamROC are NOT emitted here. They are contrasts of a competing motif
    # against THE CAUSE (negatives = union of fired clauses), which this training-time
    # path cannot express: compute_gt_roc scores a single mask against its complement,
    # so the causal atoms would sit in the negative set and a correct attribution would
    # score below 0.5 mechanically. Emitting them here under the same names as the
    # phase-6 metric would put two different quantities behind one column. They are
    # produced solely by analysis/evaluate.py::_contrast_vs_cause, which is the single
    # source of truth for every number that reaches a table.
    # DNF clause-decomposed (Instance = any one fired disjunct; Global = union).
    try:
        dnf = compute_dnf_gt_roc(model, gt_list, device, node_att_fn)
        if dnf and dnf.get('n_graphs', 0) > 0:
            b['instance_gt_roc_node_auc_mean'] = dnf['instance_auc_mean']
            b['global_gt_roc_node_auc_mean'] = dnf['global_auc_mean']
            b['dnf_n_graphs'] = dnf['n_graphs']
        elif 'gt_roc_node_auc_mean' in b:      # source-GT single-clause alias
            b['instance_gt_roc_node_auc_mean'] = b['gt_roc_node_auc_mean']
            b['global_gt_roc_node_auc_mean'] = b['gt_roc_node_auc_mean']
    except Exception as e:                     # pragma: no cover
        b['dnf_error'] = str(e)
    return b


# ─────────────────────────────────────────────────────────────────────────────
# Per-split impact + correlation from a per-motif score map + per-graph atts
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: post-hoc GT-ROC grades each explainer's OWN per-node atts node-direct. MAGE /
# Motif-Occlusion produce per-motif scores that ``_pi_to_node_atts`` broadcasts onto atoms
# (full-vocab → every atom in a scored motif, no UNK); GNNExplainer / PGExplainer are
# already per-node. Filtered-vocab natives (MOSE/MotifSAT/GSAT) carry the model's UNK weight
# (0.5 fixed / learnable) and are graded via ``motif_eval.model_node_att_fn`` in the native
# pipeline. There is deliberately no per-motif-score broadcast fallback (it zeroed UNK).


def precompute_agnostic_impact(model, vocab, split_list, device, task_type, cache_path=None):
    """Compute (or LOAD) the AGNOSTIC per-(motif,graph) faithful-LOO impact cache
    ONCE per split: ``{motif_id: {graph_idx: impact}}``.

    This is the post-hoc y-axis. It is model-only (uniform weights, base=ones) →
    IDENTICAL for all 4 explainers, so it is computed a single time and reused,
    instead of the previous per-explainer recompute (4× the LOO forward passes,
    the dominant cost). With ``max_motifs_eval=None`` (the config) this uncapped
    cache is also exactly what the grouped metric needs, so grouped + instance
    both derive from it — one LOO pass per split.

    If ``cache_path`` exists it is LOADED (zero forward passes); otherwise it is
    computed and WRITTEN, so any re-eval of this checkpoint is free. LOO is a model
    masked-forward-pass and cannot be derived from the explainer attributions —
    this caches the result so it is paid once, ever."""
    if cache_path is not None and Path(cache_path).exists():
        try:
            raw = json.loads(Path(cache_path).read_text())
            return {int(m): {int(g): float(v) for g, v in gm.items()}
                    for m, gm in raw.items()}
        except Exception as e:
            print(f'  [warn] impact cache load failed ({e}); recomputing.')
    from .embedding_viz import build_impact_cache_from_eval
    cache = build_impact_cache_from_eval(
        model, split_list, vocab, device, task_type,
        base_att_fn=(lambda d: torch.ones(d.num_nodes)))
    if cache_path is not None:
        try:
            Path(cache_path).write_text(json.dumps(
                {str(m): {str(g): v for g, v in gm.items()} for m, gm in cache.items()}))
        except Exception as e:
            print(f'  [warn] impact cache save failed ({e}).')
    return cache


def _grouped_rows_from_cache(agn_cache, motif_scores, vocab):
    """Per-motif grouped rows {motif_id, score, impact(mean), support} derived from
    the shared agnostic cache — impact=mean over the motif's graphs, support=#graphs
    (identical to compute_motif_impact's entry when uncapped)."""
    motif_list = getattr(vocab, 'motif_list', [])
    rows = []
    for m, gm in agn_cache.items():
        if m in motif_scores and gm:
            vals = list(gm.values())
            rows.append({'motif_id': int(m), 'score': float(motif_scores[m]),
                         'impact': float(np.mean(vals)), 'support': len(vals),
                         'motif_smarts': (str(motif_list[m]) if m < len(motif_list) else '')})
    return rows


def _instance_from_cache(agn_cache, per_graph_atts, split_list):
    """Per-instance correlation using the shared agnostic impact cache as y. Score
    cache x = explainer per-node atts reduced to per-(motif,graph) (GNN/PG/broadcast
    MAGE/Occl); empty atts → no signal → NaN."""
    if per_graph_atts:
        mask_cache = build_graph_mask_cache(split_list)
        score_cache = build_motif_score_cache_from_atts(per_graph_atts, mask_cache)
    else:
        score_cache = {}
    return per_instance_correlation_from_caches(score_cache, agn_cache)


def impact_and_corr_for_split(
    model, vocab, split_list, device, task_type, max_motifs_eval,
    motif_scores: Dict[int, float],
    base_att_fn: Callable,
    per_graph_atts: Optional[Dict[int, torch.Tensor]] = None,
) -> dict:
    """Compute impact (faithful LOO under ``base_att_fn``), the grouped rows, and
    per-instance correlation for one split. Returns a dict with 'impacts',
    'grouped_rows', 'instance'. ``motif_scores`` is the per-motif x (importance).
    ``per_graph_atts`` (GNN/PG per-node masks) supplies the per-instance score
    cache; when absent it falls back to the impact pass's own score cache."""
    from .embedding_viz import build_impact_cache_from_eval

    # GROUPED: per-motif mean impact (+ support) under base_att_fn. max_motifs_eval
    # caps which motifs are ranked, matching the pipeline's grouped metric.
    impacts = compute_motif_impact(
        model, split_list, vocab, device, task_type=task_type,
        max_motifs=max_motifs_eval, base_att_fn=base_att_fn)

    grouped_rows = []
    for mid, e in impacts.items():
        if mid in motif_scores:
            grouped_rows.append({
                'motif_id': int(mid), 'score': float(motif_scores[mid]),
                'impact': float(e.get('impact', float('nan'))),
                'support': int(e.get('n_graphs', 0)),
                'motif_smarts': e.get('motif_smarts', ''),
            })

    # PER-INSTANCE: the impact cache comes from the SAME trusted path
    # pipeline._per_instance / run_vanilla use — build_impact_cache_from_eval,
    # which is UNCAPPED (no max_motifs), so coverage matches the trusted metric
    # exactly (the loo_impact values are identical either way; this only removes the
    # max_motifs coverage discrepancy). Score cache: explainer per-node atts when
    # given (GNN/PG/broadcast MAGE/Occl), else the impact pass's own score cache.
    impact_cache = build_impact_cache_from_eval(
        model, split_list, vocab, device, task_type, base_att_fn=base_att_fn)
    if per_graph_atts is not None:
        mask_cache = build_graph_mask_cache(split_list)
        score_cache = build_motif_score_cache_from_atts(per_graph_atts, mask_cache)
    else:
        score_cache, _ = caches_from_motif_impact(impacts)
    inst = per_instance_correlation_from_caches(score_cache, impact_cache)
    return {'impacts': impacts, 'grouped_rows': grouped_rows, 'instance': inst}


# ─────────────────────────────────────────────────────────────────────────────
# File writers
# ─────────────────────────────────────────────────────────────────────────────

def _write_importance(out_dir: Path, method: str, split: str, rows: List[dict]):
    if not rows:
        return
    pd.DataFrame([{'motif_id': r['motif_id'], 'score': r['score'],
                   'support': r.get('support', 0),
                   'motif_smarts': r.get('motif_smarts', '')} for r in rows]
                 ).to_csv(out_dir / f'{method}_importance_{split}.csv', index=False)


def _write_impact(out_dir: Path, method: str, split: str, rows: List[dict]):
    if not rows:
        return
    pd.DataFrame([{'motif_id': r['motif_id'], 'impact': r['impact'],
                   'support': r.get('support', 0),
                   'motif_smarts': r.get('motif_smarts', '')} for r in rows]
                 ).to_csv(out_dir / f'{method}_impact_{split}.csv', index=False)


def write_grouped_pooled(out_dir: Path, method: str, rows_by_split: Dict[str, List[dict]]):
    """Two grouped-correlation files: pooled over train+valid+test, and test-only.
    Pooling is at the (motif, split) row level so a motif contributes its per-split
    (score, impact, support) point once per split it appears in."""
    all_rows = [dict(r, split=s) for s in SPLITS for r in rows_by_split.get(s, [])]
    test_rows = [dict(r, split='test') for r in rows_by_split.get('test', [])]
    for tag, rws in (('alltest', all_rows), ('testonly', test_rows)):
        v = grouped_corr_variants(rws)
        pd.DataFrame([{'method': method, 'pooling': tag, **v}]).to_csv(
            out_dir / f'{method}_grouped_corr_pooled_{tag}.csv', index=False)


def write_instance(out_dir: Path, method: str, inst_by_split: Dict[str, dict]):
    for s in SPLITS:
        inst = inst_by_split.get(s)
        if inst:
            pd.DataFrame([{'method': method, 'split': s, **inst}]).to_csv(
                out_dir / f'{method}_instance_corr_{s}.csv', index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Scalar flatteners (prediction + GT-ROC) for summary_splits.json
# ─────────────────────────────────────────────────────────────────────────────

def _pred_scalars(pred: dict) -> Dict[str, float]:
    """AUC / RMSE / MAE (normalised + original-unit) from evaluate_predictions."""
    return {
        'auc':       pred.get('auc', pred.get('auc_mean', float('nan'))),
        'rmse':      pred.get('rmse', float('nan')),
        'mae':       pred.get('mae', float('nan')),
        'rmse_orig': pred.get('rmse_orig', float('nan')),
        'mae_orig':  pred.get('mae_orig', float('nan')),
    }


def _gtroc_scalars_from_block(block: dict) -> Dict[str, float]:
    """Flatten the GT-ROC sub-dicts a native ``_compute_explainability`` block
    produces (each ``{'auc_mean':…}``) to flat ``*_auc_mean`` scalars."""
    out: Dict[str, float] = {}
    for src, key in (
        ('gt_roc_node', 'gt_roc_node_auc_mean'),
        ('gt_roc_edge', 'gt_roc_edge_auc_mean'),
        # spurious_roc_node / family_roc_node deliberately absent: see gt_roc_block --
        # those are contrasts against the cause, produced only by analysis/evaluate.py
        ('instance_gt_roc_node', 'instance_gt_roc_node_auc_mean'),
        ('global_gt_roc_node', 'global_gt_roc_node_auc_mean'),
    ):
        d = block.get(src)
        if isinstance(d, dict) and 'auc_mean' in d:
            out[key] = d['auc_mean']
    return out


def _instance_scalars(corr: dict) -> Dict[str, float]:
    """The per-instance correlation fields (own + agnostic) from a correlation dict."""
    return {k: corr.get(k, float('nan')) for k in (
        'pearson_instance', 'spearman_instance',
        'pearson_instance_agnostic', 'spearman_instance_agnostic', 'n_instances')}


def write_summary_splits(out_dir: Path, summary: Dict[str, Dict[str, dict]]):
    """Write summary_splits.json = {method: {split: {metric: value}}}. Merged
    (read-modify-write) so native and post-hoc passes on the same run coexist."""
    path = out_dir / 'summary_splits.json'
    existing: Dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            existing = {}
    for method, by_split in summary.items():
        existing.setdefault(method, {}).update(by_split)
    # ATOMIC write: a kill mid-write must never leave a partial summary_splits.json
    # (which would read as non-empty → falsely "done"). Write to .tmp, then rename.
    import os as _os
    tmp = str(path) + '.tmp'
    with open(tmp, 'w') as _fh:
        _fh.write(json.dumps(existing, indent=2))
    _os.replace(tmp, path)


def write_global_pooled(out_dir: Path, kind: str,
                        rows_by_method_split: Dict[str, Dict[str, List[dict]]]):
    """importance_global.csv / impact_global.csv — pooled over splits AND methods.
    One row per (method, split, motif). ``kind`` in {'importance','impact'}."""
    col = 'score' if kind == 'importance' else 'impact'
    rows = []
    for method, by_split in rows_by_method_split.items():
        for split, rws in by_split.items():
            for r in rws:
                rows.append({'method': method, 'split': split,
                             'motif_id': r['motif_id'], col: r.get(col),
                             'support': r.get('support', 0),
                             'motif_smarts': r.get('motif_smarts', '')})
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / f'{kind}_global.csv', index=False)


# ─────────────────────────────────────────────────────────────────────────────
# ANTE-HOC (native) per-split orchestration
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_native_all_splits(
    model, vocab, loaders, split_lists: Dict[str, list],
    gt_split_lists: Dict[str, Optional[list]],
    device, task_type, denorm, out_dir, max_motifs_eval,
    motif_scores: Dict[int, float], method_name: str,
) -> Dict[str, Dict[str, dict]]:
    """Per-split (train/valid/test) evaluation for a native model (MoSE/MotifSAT/
    GSAT). Drives ``EvalPipeline._compute_explainability`` per split (so it reuses
    the existing native metric logic incl. the DNF add), writes the per-split +
    pooled files, and returns {method: {split: scalars}} for summary_splits.json.

    NOTE the model's per-motif score (``motif_scores``) is GLOBAL, so per-split
    importance files carry the same score set; only impact (per-split LOO) varies.
    """
    from .pipeline import EvalPipeline
    from .motif_eval import model_node_att_fn, caches_from_motif_impact
    out_dir = Path(out_dir)
    rows_by_split: Dict[str, List[dict]] = {}
    inst_by_split: Dict[str, dict] = {}
    summary_by_split: Dict[str, dict] = {}
    # Raw per-node atts per split, persisted so post-training evaluate.py can
    # score GT-ROC artifact-only (no checkpoint reload / re-forward).
    importances_json: Dict[str, dict] = {s: {} for s in SPLITS}

    for split in SPLITS:
        sl = split_lists.get(split)
        if not sl:
            continue
        gl = gt_split_lists.get(split)
        pred = evaluate_predictions(model, loaders[split], device, task_type, denorm=denorm)
        pipe = EvalPipeline(model, vocab, loaders[split], sl, device, task_type,
                            max_motifs_eval=max_motifs_eval, denorm=denorm,
                            gt_eval_list=gl)
        block = pipe._compute_explainability(sl, gl, motif_scores, run_motif_impact=True)

        impacts = block.get('motif_impact', {}) or {}
        grouped_rows = []
        for mid, e in impacts.items():
            if mid in motif_scores:
                grouped_rows.append({
                    'motif_id': int(mid), 'score': float(motif_scores[mid]),
                    'impact': float(e.get('impact', float('nan'))),
                    'support': int(e.get('n_graphs', 0)),
                    'motif_smarts': e.get('motif_smarts', '')})
        rows_by_split[split] = grouped_rows
        inst_by_split[split] = _instance_scalars(block.get('correlation', {}) or {})
        summary_by_split[split] = {**_pred_scalars(pred),
                                   **_gtroc_scalars_from_block(block)}
        _write_importance(out_dir, method_name, split, grouped_rows)
        _write_impact(out_dir, method_name, split, grouped_rows)

        # Persist raw per-node atts (importances) — same family-agnostic extractor
        # GT-ROC uses, keyed by split-local graph index. Lets evaluate.py score
        # GT-ROC from artifacts alone (no re-forward). None => model exposes no
        # node attention (e.g. vanilla); those graphs are simply skipped.
        _natt = model_node_att_fn(model, device)
        _atts = {}
        for _i, _g in enumerate(sl):
            _a = _natt(_g)
            if _a is not None:
                _atts[_i] = _a
        if _atts:
            importances_json[split][method_name] = _atts_to_jsonable(_atts)
        # Persist per-instance OWN impact (free — reuses the motif_impact just
        # computed). Ante-hoc y-axis is the model's OWN weights, NOT the base=ones
        # agnostic cache, so it is written under its own filename.
        _sc, _own_impact = caches_from_motif_impact(impacts)
        (out_dir / f'impact_cache_own_{split}.json').write_text(json.dumps(
            {str(m): {str(g): float(v) for g, v in gm.items()}
             for m, gm in _own_impact.items()}))

    write_grouped_pooled(out_dir, method_name, rows_by_split)
    write_instance(out_dir, method_name, inst_by_split)
    write_global_pooled(out_dir, 'importance', {method_name: rows_by_split})
    write_global_pooled(out_dir, 'impact', {method_name: rows_by_split})
    # explainer_importances.json — merged + atomic, SAME schema/merge as the
    # post-hoc path, so read_saved_atts / evaluate.py consume it identically and a
    # per-method write never clobbers another method's atts.
    import os as _os
    _ei = out_dir / 'explainer_importances.json'
    _prev = {}
    if _ei.exists():
        try:
            _prev = (json.loads(_ei.read_text()) or {}).get('importances_by_split') or {}
        except Exception:
            _prev = {}
    for _s, _by in importances_json.items():
        if _by:
            _prev.setdefault(_s, {}).update(_by)
    _tmp = _ei.with_suffix('.json.tmp')
    _tmp.write_text(json.dumps({'importances_by_split': _prev}))
    _os.replace(_tmp, _ei)
    # summary_splits.json LAST + atomic — the completion marker (poll/skip_done key
    # on it), matching the post-hoc path; a run killed before this is redone.
    summary = {method_name: summary_by_split}
    write_summary_splits(out_dir, summary)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# POST-HOC per-split orchestration
# ─────────────────────────────────────────────────────────────────────────────

def _pi_to_node_atts(per_instance: Dict[int, Dict[int, float]], split_list) -> Dict[int, torch.Tensor]:
    """Broadcast per-(motif, graph) scores onto per-node atts so MAGE / Motif-
    Occlusion share the SAME instance-correlation path as GNN/PG (whose atts are
    already per-node). node_att[i] = score of the motif at node i (0 elsewhere)."""
    atts: Dict[int, torch.Tensor] = {}
    for gi in range(len(split_list)):
        n2m = getattr(split_list[gi], 'nodes_to_motifs', None)
        if n2m is None:
            continue
        n2m = n2m.view(-1).cpu()
        w = torch.zeros(n2m.numel())
        hit = False
        for mid, gmap in per_instance.items():
            val = gmap.get(gi)
            if val is not None:
                w[n2m == int(mid)] = float(val)
                hit = True
        if hit:
            atts[gi] = w
    return atts


def _node_att_fn(gl, att_by_i):
    """Attach each graph's per-node att (aligned by gl position) and return a
    node_att_fn reading it — so post-hoc GT-ROC is NODE-DIRECT (per-node atts, not a
    per-motif-mean broadcast). Ported from validate_posthoc_gtroc (now folded in here).
    Length guard fails loud on misalignment; returns None when no atts were attached."""
    for g in gl:
        g._raw_att = None
    n = 0
    for i, g in enumerate(gl):
        a = att_by_i.get(i)
        if a is None:
            continue
        t = a.float() if torch.is_tensor(a) else torch.tensor(a, dtype=torch.float32)
        if t.numel() != g.num_nodes:
            raise RuntimeError(f'att/nodes mismatch graph {i}: {t.numel()} vs {g.num_nodes}')
        g._raw_att = t
        n += 1

    def fn(d):
        a = getattr(d, '_raw_att', None)
        return a.to(d.x.device) if a is not None else None
    return fn if n else None


def _fill_unk(att_by_i, gl, fill):
    """FILTERED mode: set each graph's per-node att to ``fill`` on atoms whose motif is
    UNK (nodes_to_motifs < 0), so the UNK-atom treatment is identical across methods
    (fill=0.0 zero; fill=0.5 neutral, matching MoSE). Ported from validate_posthoc_gtroc."""
    out = {}
    for i, a in att_by_i.items():
        n2m = getattr(gl[i], 'nodes_to_motifs', None)
        if n2m is None:
            out[i] = a
            continue
        n2m = n2m.view(-1).tolist()
        if torch.is_tensor(a):
            a = a.clone().float()
            for j, m in enumerate(n2m):
                if m < 0 and j < a.numel():
                    a[j] = fill
            out[i] = a
        else:
            out[i] = [fill if (j < len(n2m) and n2m[j] < 0) else v for j, v in enumerate(a)]
    return out


def _gtroc_node_direct(model, gl, sl, device, atts, sc, unk_mode=None):
    """NODE-DIRECT GT-ROC for one explainer on one split: grade the explainer's own
    per-node atts (aligned sl->gl; identity remap for a GT subset like mutag) against the
    GT via gt_roc_block. Replaces validate_posthoc_gtroc so one path/one tree carries the
    full per-node metric set (node/edge/per-motif spurious/family/instance/global).
    ``unk_mode`` = filtered-vocab UNK-atom handling (matches MoSE; folds in the old
    gtroc_filtered_* trees):
      None      -> atts as-is (FULL / unfiltered)
      'zero'    -> UNK atoms set to 0.0      'half' -> UNK atoms set to 0.5
      'exclude' -> UNK atoms dropped from the AUC (instance/global only — no spurious/family)
    ``sc`` (per-motif scores) is unused — kept only for call-site compatibility. It never
    falls back to a per-motif-score broadcast (that silently zeroed UNK and is retired); if
    the explainer produced NO per-node atts it RAISES ValueError (fail loud) rather than
    returning a blank metric set. Callers where a method may legitimately have no node
    attribution (e.g. GNNExplainer on OGB) already skip it upstream / catch around this."""
    att_by_i = {}
    if atts:
        if gl is sl:
            att_by_i = {int(i): a for i, a in atts.items()}
        else:
            pos = {id(g): i for i, g in enumerate(sl)}
            att_by_i = {j: atts[pos[id(g)]] for j, g in enumerate(gl)
                        if id(g) in pos and pos[id(g)] in atts}
    if unk_mode == 'exclude' and att_by_i:          # UNK atoms out of the AUC
        from analysis.gtroc_unk import gtroc_masked
        inst, glob, _ = gtroc_masked(att_by_i, gl)
        return {'instance_gt_roc_node_auc_mean': inst,
                'global_gt_roc_node_auc_mean': glob}
    if unk_mode in ('zero', 'half') and att_by_i:   # fill UNK atoms, then full metric set
        att_by_i = _fill_unk(att_by_i, gl, 0.5 if unk_mode == 'half' else 0.0)
    fn = _node_att_fn(gl, att_by_i) if att_by_i else None
    if fn is None:                                  # no per-node atts -> cannot grade node-direct
        raise ValueError(
            f'_gtroc_node_direct: no per-node attributions to grade against the GT '
            f'({len(gl)} GT graphs, {len(sl)} split graphs, atts={len(atts) if atts else 0}). '
            f"Node-direct GT-ROC needs the explainer's own per-node atts; an empty set means the "
            f'explainer produced none — surface it, do not skip silently.')
    return gt_roc_block(model, gl, device, fn)


def _atts_to_jsonable(atts: Dict[int, torch.Tensor]) -> Dict[int, list]:
    return {int(g): a.detach().cpu().view(-1).tolist() for g, a in atts.items()}


# Explainers whose per-node attribution is a property of the VANILLA model + graph
# only (not the vocabulary), so a run under a different fragmentation can REUSE the
# saved atts verbatim instead of a fresh (stochastic) refit. Motif-Occlusion and MAGE
# are motif-level / vocab-dependent and are always recomputed.
_REUSE_METHODS = ('gnnexplainer', 'pgexplainer')


def _load_reused_atts(reuse_dir, split: str, ex: str, sl) -> Dict[int, torch.Tensor]:
    """Load one explainer's per-node atts for one split from a PREVIOUSLY-SAVED
    ``explainer_importances.json`` — a run on the SAME vanilla checkpoint under a
    DIFFERENT vocabulary. The atts are per-ATOM and vocab-independent, so they are
    reused verbatim and only the motif aggregation (via the current vocab) differs;
    this is what lets the post-hoc explainers be scored on a new fragmentation WITHOUT
    a stochastic refit. Alignment is asserted per graph (atom count must equal the
    current split graph) — a mismatch means the molecule ordering drifted, so we FAIL
    LOUD rather than silently mis-map. Returns {} when the split/method key is absent."""
    p = Path(reuse_dir) / 'explainer_importances.json'
    if not p.exists():
        raise FileNotFoundError(f'reuse_atts: no explainer_importances.json at {p}')
    by_split = (json.loads(p.read_text()).get('importances_by_split') or {})
    raw = (by_split.get(split) or {}).get(ex) or {}
    atts: Dict[int, torch.Tensor] = {}
    for gk, vals in raw.items():
        gi = int(gk)
        if gi >= len(sl):
            raise RuntimeError(
                f'reuse_atts {ex}/{split}: graph index {gi} >= split size {len(sl)} '
                f'(ordering mismatch between the reused run and this vocab).')
        n = int(sl[gi].num_nodes)
        if len(vals) != n:
            raise RuntimeError(
                f'reuse_atts {ex}/{split}: graph {gi} atom count {len(vals)} != {n} '
                f'(molecule ordering drifted between vocabularies; refuse to mis-map).')
        atts[gi] = torch.tensor(vals, dtype=torch.float32)
    return atts


def evaluate_posthoc_all_splits(
    model, vocab, loaders, split_lists: Dict[str, list],
    gt_split_lists: Dict[str, Optional[list]],
    device, task_type, denorm, out_dir, max_motifs_eval, dataset: str,
    method_names=('gnnexplainer', 'motif_occlusion', 'mage_v2', 'pgexplainer'),
    mage_positive_class: Optional[int] = None, save_artifacts: bool = True,
    batch_size: int = 256, unk_mode: Optional[str] = None,
    reuse_atts_dir=None,
) -> Dict[str, Dict[str, dict]]:
    """Per-split (train/valid/test) evaluation for the 4 post-hoc explainers on a
    Vanilla backbone. Trainable explainers are FIT ONCE on TRAIN (MAGE attention;
    PGExplainer with ``ref_list=train`` so edge_size is chosen with zero test
    dependency) and their model artifacts saved; every explainer is then SCORED on
    each split. Correlations use the AGNOSTIC (uniform-weight LOO) impact as the
    y-axis; GT-ROC grades the explainer's own attribution. Writes the per-split +
    pooled files + explainer_importances.json (all splits) and returns the summary.

    ``reuse_atts_dir`` (optional): a run dir holding a saved explainer_importances.json
    from a run on the SAME vanilla checkpoint under a DIFFERENT vocabulary. When set,
    GNNExplainer/PGExplainer are NOT refit — their per-atom atts are loaded from there
    and only re-aggregated to THIS vocab's motifs (see _load_reused_atts). Motif-Occlusion
    (and MAGE, if requested) are still recomputed. This is the mechanism for scoring the
    post-hoc explainers on a new fragmentation without a stochastic refit that would
    confound the vocab comparison.
    """
    from ..baselines.gnn_explainer_batched import run_gnnexplainer_batched  # GPU-batched
    from ..baselines.motif_occlusion import run_motif_occlusion_batched     # GPU-batched
    from ..baselines.mage import fit_mage, score_mage
    from ..baselines.pg_explainer import fit_pgexplainer, score_pgexplainer

    out_dir = Path(out_dir)
    train_list = split_lists['train']
    pc = (mage_positive_class if mage_positive_class is not None
          else (0 if dataset == 'mutag' else 1))

    # ── Fit MAGE upfront; PGExplainer LAZILY so it (fit + score) runs LAST ────
    # GNN / Occlusion / MAGE are GPU-batched and cheap → run first. PGExplainer is
    # sequential + CPU-bound (its training can't batch), so its fit is DEFERRED to
    # the first pgexplainer score call; with 'pgexplainer' ordered last in
    # method_names, PG genuinely goes last within each split.
    mage_attn = (fit_mage(model, train_list, vocab, device, task_type, verbose=False,
                          embed_batch_size=batch_size)
                 if 'mage_v2' in method_names else None)
    if save_artifacts and mage_attn is not None:
        torch.save(mage_attn.state_dict(), out_dir / 'mage_attention.pt')

    _pg = {'fit': None, 'fitted': False}

    def _pg_fit():
        """Fit PGExplainer on TRAIN once, lazily (on first pgexplainer scoring)."""
        if not _pg['fitted']:
            _pg['fitted'] = True
            if 'pgexplainer' in method_names:
                f = fit_pgexplainer(model, loaders, train_list, device, task_type,
                                    verbose=False)                 # ref_list = TRAIN
                _pg['fit'] = f
                if save_artifacts and f is not None and f.get('explainer') is not None:
                    try:
                        torch.save(f['explainer'].algorithm.state_dict(),
                                   out_dir / 'pgexplainer_mlp.pt')
                    except Exception as e:
                        print(f'  [warn] could not save pgexplainer_mlp.pt ({e})')
        return _pg['fit']

    def _score_explainer(ex, sl):
        """(scores dict, per-node atts) for one explainer on one split; empty on
        failure (never raises — a single explainer must not sink the whole run)."""
        try:
            if ex == 'gnnexplainer':
                return run_gnnexplainer_batched(model, sl, vocab, device, task_type,
                                                epochs=200, batch_size=batch_size,
                                                return_node_atts=True)
            if ex == 'pgexplainer':
                fit = _pg_fit()                       # lazy: fits on first call
                if fit is None:
                    return {'mean': {}, 'max': {}}, {}
                return score_pgexplainer(fit, sl, device, return_node_atts=True)
            if ex == 'motif_occlusion':
                sc, pi = run_motif_occlusion_batched(model, sl, vocab, device, task_type,
                                                     batch_size=batch_size,
                                                     return_per_instance=True)
                return sc, _pi_to_node_atts(pi, sl)
            if ex == 'mage_v2':
                if mage_attn is None:
                    return {'mean': {}, 'max': {}}, {}
                # mage_v2 = attention_mean (class-agnostic, freq-normalized); PIN it
                # explicitly so it never silently rides the module default. positive_class
                # is intentionally omitted — attention_mean ignores P[c].
                sc, pi = score_mage(model, mage_attn, sl, vocab, device, task_type,
                                    return_per_instance=True, verbose=False,
                                    embed_batch_size=batch_size, score_mode='attention_mean')
                return sc, _pi_to_node_atts(pi, sl)
        except Exception as e:
            print(f'  [warn] {ex} scoring failed on split ({e}).')
        return {'mean': {}, 'max': {}}, {}

    def _ones(d):
        return torch.ones(d.num_nodes)

    grouped_by_ex = {ex: {} for ex in method_names}
    inst_by_ex = {ex: {} for ex in method_names}
    rows_by_ex = {ex: {} for ex in method_names}       # for global pooled files
    summary = {ex: {} for ex in method_names}
    importances_json = {s: {} for s in SPLITS}

    for split in SPLITS:
        sl = split_lists.get(split)
        if not sl:
            continue
        gl = gt_split_lists.get(split)
        pred_flat = _pred_scalars(
            evaluate_predictions(model, loaders[split], device, task_type, denorm=denorm))
        # AGNOSTIC LOO impact ONCE per split (model-only → shared across all 4
        # explainers; cached to disk so re-evals are free). Replaces the previous
        # per-explainer recompute that was the throughput bottleneck.
        agn_cache = precompute_agnostic_impact(
            model, vocab, sl, device, task_type,
            cache_path=out_dir / f'impact_cache_agnostic_{split}.json')
        for ex in method_names:
            if reuse_atts_dir is not None and ex in _REUSE_METHODS:
                # REUSE path: per-node atts come from a saved run on the SAME vanilla
                # checkpoint under a different vocab (no refit). Motif scores are the
                # per-motif mean of those atts under THIS vocab's mask cache — the same
                # reduction _score_explainer would return, so everything downstream is
                # identical to the fitted path.
                atts = _load_reused_atts(reuse_atts_dir, split, ex, sl)
                if not atts:
                    summary[ex][split] = dict(pred_flat)   # no saved atts for this cell
                    continue
                _sccache = build_motif_score_cache_from_atts(
                    atts, build_graph_mask_cache(sl))
                sc = {int(m): float(np.mean(list(gm.values())))
                      for m, gm in _sccache.items() if gm}
                if not sc:
                    summary[ex][split] = dict(pred_flat)
                    continue
            else:
                scores, atts = _score_explainer(ex, sl)
                if not scores.get('mean'):
                    summary[ex][split] = dict(pred_flat)   # no explainer signal here
                    continue
                sc = scores['mean']
            # grouped + instance derived from the SHARED agnostic cache (no re-LOO).
            grouped_rows = _grouped_rows_from_cache(agn_cache, sc, vocab)
            inst = _instance_from_cache(agn_cache, (atts or None), sl)
            grouped_by_ex[ex][split] = grouped_rows
            inst_by_ex[ex][split] = inst
            rows_by_ex[ex][split] = grouped_rows
            # GT-ROC grades the explainer's OWN attribution NODE-DIRECT: feed the
            # per-node atts it produced (GNN/PG raw per-node; MAGE/occlusion motif-
            # broadcast), NOT the per-motif-mean. Yields the full gt_roc_block set
            # (node/edge/per-motif spurious/family/instance/global) as the correct per-node
            # metric — folds in validate_posthoc_gtroc so there's one path/one tree.
            gtroc = _gtroc_node_direct(model, gl, sl, device, atts, sc, unk_mode) if gl else {}
            summary[ex][split] = {**pred_flat, **gtroc}
            _write_importance(out_dir, ex, split, grouped_rows)
            _write_impact(out_dir, ex, split, grouped_rows)
            if atts:
                importances_json[split][ex] = _atts_to_jsonable(atts)

    for ex in method_names:
        write_grouped_pooled(out_dir, ex, grouped_by_ex[ex])
        write_instance(out_dir, ex, inst_by_ex[ex])
    write_global_pooled(out_dir, 'importance', rows_by_ex)
    write_global_pooled(out_dir, 'impact', rows_by_ex)
    # MERGED (read-modify-write), like write_summary_splits below — NOT a plain
    # overwrite. This one file holds the per-ATOM attributions of EVERY method
    # (split -> method -> graph -> [values]), while every other artifact is
    # per-method by filename. A wholesale write therefore destroyed the other
    # methods' attributions whenever a run scored a subset of method_names, which
    # is exactly what a per-method scheduler does: scoring gnnexplainer on a cell
    # that already had pgexplainer erased pgexplainer's atom-level scores (its
    # CSVs survived, so it looked complete while GT-ROC had nothing to read).
    # Measured: 224 cells lost this way before it was caught.
    _ei = out_dir / 'explainer_importances.json'
    _prev = {}
    if _ei.exists():
        try:
            _prev = (json.loads(_ei.read_text()) or {}).get('importances_by_split') or {}
        except Exception:
            _prev = {}
    for _s, _by_ex in importances_json.items():
        _prev.setdefault(_s, {}).update(_by_ex)
    import os as _os                             # not imported at module scope here
    _tmp = _ei.with_suffix('.json.tmp')           # atomic: a kill must not truncate it
    _tmp.write_text(json.dumps({'importances_by_split': _prev}))
    _os.replace(_tmp, _ei)
    # summary_splits.json is written LAST + atomically — it is the completion
    # marker (poll/skip_done key on it), so it must land only after every other
    # artifact is on disk. A run killed anywhere before this has no summary → redone.
    write_summary_splits(out_dir, summary)
    return summary
