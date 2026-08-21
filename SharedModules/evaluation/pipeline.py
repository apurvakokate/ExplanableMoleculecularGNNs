"""pipeline.py -- EvalPipeline: orchestrates all evaluation for one run.

Usage
-----
    pipeline = EvalPipeline(model, vocab, test_loader, test_list,
                            device, task_type)
    results = pipeline.run(
        motif_scores=model.get_motif_scores(),
        gt_motif_ids={0, 4},    # optional: known GT motif ids
    )
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set

import logging
import pandas as pd
import torch

from ..data.vocab import VocabData
from .metrics import evaluate_predictions
from .motif_eval import (
    compute_motif_impact,
    score_impact_correlation,
    per_instance_correlation_from_caches,
    caches_from_motif_impact,
    top_bottom_motif_eval,
    gt_vs_outside_gt_eval,
    gt_motif_ids_from_labels,
    compute_gt_roc,
    spurious_motif_names,
    compute_dnf_gt_roc,
    model_node_att_fn,
    motif_class_discriminativeness,
    top_motifs_discriminative_check,
    _impact_values,
)


def _has_node_attr(graphs, attr: str) -> bool:
    """True iff at least one graph carries a ``attr`` with a POSITIVE entry. Used to
    gate the Mode-2, spurious, and family ROC so compute_gt_roc is never given a
    missing attr — which would silently fall back to the edge_label-derived (rule)
    mask and mislabel the metric.

    Requiring a positive, not merely a non-None tensor, matters because apply_gt
    initialises node_label_spurious / node_label_family as all-zero tensors on EVERY
    graph and only sets 1.0 where such an atom exists. A rule with no spurious/family
    atoms anywhere therefore has the attr present-but-all-zero; scoring it yields an
    undefined AUC (NaN) that pollutes the summary and the aggregate tables. Skipping
    it instead leaves the metric absent for that rule. The causal label (node_label —
    fired-clause atoms) always carries positives on rule-positive graphs, so this never
    gates it out."""
    for g in graphs:
        v = getattr(g, attr, None)
        if v is None:
            continue
        try:
            if float(v.sum()) > 0:
                return True
        except Exception:
            if bool(v):        # non-tensor truthy attr — presence is enough
                return True
    return False


def explainability_summary_fields(
    results: Dict,
    *,
    scope: str = 'test',
    corr_by_agg: Optional[Dict[str, Dict]] = None,
) -> Dict[str, float]:
    """Flat summary.json fields for pearson / spearman / GT-ROC.

    scope 'test' — headline test-split metrics (``pearson``, ``gt_roc_*``).
    scope 'all'  — pooled train+valid+test metrics (``pearson_all``, …).
    """
    nan = float('nan')
    suffix = '_all' if scope == 'all' else ''
    tag = 'all' if scope == 'all' else ''

    def _corr_block() -> Dict:
        if corr_by_agg is not None:
            _mean = corr_by_agg.get('mean', {})
            out: Dict[str, float] = {
                f'pearson{suffix}':  _mean.get('pearson', nan),
                f'spearman{suffix}': _mean.get('spearman', nan),
                f'pearson_motif{suffix}':  _mean.get('pearson_motif', nan),
                f'spearman_motif{suffix}': _mean.get('spearman_motif', nan),
                # TRUE per-instance (per-graph weight vs per-graph impact):
                # OWN impact (model's own attention LOO) + AGNOSTIC (uniform-W LOO)
                f'pearson_instance{suffix}':  _mean.get('pearson_instance', nan),
                f'spearman_instance{suffix}': _mean.get('spearman_instance', nan),
                f'pearson_instance_agnostic{suffix}':  _mean.get('pearson_instance_agnostic', nan),
                f'spearman_instance_agnostic{suffix}': _mean.get('spearman_instance_agnostic', nan),
            }
            for agg in ('mean', 'max'):
                c = corr_by_agg.get(agg, {})
                out[f'pearson_node_{agg}{suffix}']  = c.get('pearson', nan)
                out[f'spearman_node_{agg}{suffix}'] = c.get('spearman', nan)
            return out
        corr = results.get(f'correlation_{tag}' if tag else 'correlation', {})
        return {
            f'pearson{suffix}':  corr.get('pearson', nan),
            f'spearman{suffix}': corr.get('spearman', nan),
            # motif-level aggregated (grouped) score-vs-impact — Table 3.4 metric
            f'pearson_motif{suffix}':  corr.get('pearson_motif', nan),
            f'spearman_motif{suffix}': corr.get('spearman_motif', nan),
            # TRUE per-instance (per-graph weight vs per-graph impact):
            # OWN impact (model's own attention LOO) + AGNOSTIC (uniform-W LOO)
            f'pearson_instance{suffix}':  corr.get('pearson_instance', nan),
            f'spearman_instance{suffix}': corr.get('spearman_instance', nan),
            f'pearson_instance_agnostic{suffix}':  corr.get('pearson_instance_agnostic', nan),
            f'spearman_instance_agnostic{suffix}': corr.get('spearman_instance_agnostic', nan),
            f'pearson_node_mean{suffix}':  corr.get('pearson', nan),
            f'pearson_node_max{suffix}':   corr.get('pearson', nan),
            f'spearman_node_mean{suffix}': corr.get('spearman', nan),
            f'spearman_node_max{suffix}':  corr.get('spearman', nan),
        }

    gt     = results.get(f'gt_roc_{tag}' if tag else 'gt_roc', {})
    gt_node = results.get(f'gt_roc_node_{tag}' if tag else 'gt_roc_node', {})
    gt_edge = results.get(f'gt_roc_edge_{tag}' if tag else 'gt_roc_edge', {})
    # DNF clause-decomposed AUCs. _compute_explainability has always COMPUTED these into the
    # block, but this flatten never emitted them -- so for ante-hoc families (which write
    # summary.json via this function and no summary_splits.json) instance/global were silently
    # absent from every run, old and new. They are the metrics the paper tables read.
    gt_inst = results.get(f'instance_gt_roc_node_{tag}' if tag else 'instance_gt_roc_node', {})
    gt_glob = results.get(f'global_gt_roc_node_{tag}' if tag else 'global_gt_roc_node', {})
    # Per-motif spurious blocks. BOTH scopes live in ONE results dict: the pooled pass stores
    # every key again with an '_all' suffix (see `results[f'{key}_all']` in the producer). Every
    # other field in this function is fetched by EXACT name, so only this prefix-matched block can
    # cross scopes — and it must exclude the other scope EXPLICITLY. Without that, the test scope
    # absorbs pooled motifs as phantom '<motif>_all' entries AND its max is silently taken over
    # pooled values (measured: test max reported 0.95, the pooled number, instead of 0.60).
    _spur_pref = 'spurious_roc_node__'
    _ALL = '_all'

    def _spur_motif_name(k: str) -> Optional[str]:
        """Motif name if key k belongs to THIS scope, else None. (Motif keys are SMARTS /
        ring:-prefixed fragments and never end in '_all', so the suffix test is unambiguous.)"""
        if not k.startswith(_spur_pref):
            return None
        rest = k[len(_spur_pref):]
        if tag:                                   # pooled scope: take only '_all' keys, strip it
            return rest[:-len(_ALL)] if rest.endswith(_ALL) else None
        return None if rest.endswith(_ALL) else rest   # test scope: reject pooled keys

    spur_per = {}
    for _k, _v in results.items():
        _nm = _spur_motif_name(_k)
        if _nm and isinstance(_v, dict):
            spur_per[_nm] = _v
    fam     = results.get(f'family_roc_node_{tag}' if tag else 'family_roc_node', {})

    fields = _corr_block()
    fields.update({
        f'gt_roc_auc_mean{suffix}':            gt.get('auc_mean', nan),
        f'gt_roc_n_graphs{suffix}':            gt.get('n_graphs', 0),
        # Node GT-ROC vs the planted cause = atoms of the clause(s) that FIRED (node_label).
        # The whole-rule Mode-1 metric is gone; this key now carries fired-clause semantics
        # and is NOT comparable with the same key in pre-2026-08 archives.
        f'gt_roc_node_auc_mean{suffix}':       gt_node.get('auc_mean', nan),
        f'gt_roc_edge_auc_mean{suffix}':       gt_edge.get('auc_mean', nan),
        # Instance = best FIRED disjunct; Global = union of fired disjuncts.
        f'instance_gt_roc_node_auc_mean{suffix}': gt_inst.get('auc_mean', nan),
        f'instance_gt_roc_node_n_graphs{suffix}': gt_inst.get('n_graphs', 0),
        f'global_gt_roc_node_auc_mean{suffix}':   gt_glob.get('auc_mean', nan),
        # Family ROC — explanation vs super/sub-structures of the cause (high = right-chem-wrong-granularity)
        f'family_roc_node_auc_mean{suffix}': fam.get('auc_mean', nan),
        f'family_roc_node_n_graphs{suffix}': fam.get('n_graphs', 0),
        f'within_motif_var{suffix}':  results.get(f'within_motif_var_{tag}' if tag else 'within_motif_var', nan),
        f'between_motif_var{suffix}': results.get(f'between_motif_var_{tag}' if tag else 'between_motif_var', nan),
    })
    # Per-motif spurious-shortcut ROC — ONE field per class-1 correlate, named for its motif.
    # Deliberately not collapsed: 'did the explainer follow THIS shortcut' is a per-motif
    # question, and with an unidentifiable rule (identifiability_sep == 0) the contrast
    # between gt_roc_node and a specific correlate's ROC IS the experiment.
    _finite = []
    for _nm, _d in spur_per.items():
        _v = _d.get('auc_mean', nan)
        fields[f'spurious_roc_node_auc_mean__{_nm}{suffix}'] = _v
        fields[f'spurious_roc_node_n_graphs__{_nm}{suffix}'] = _d.get('n_graphs', 0)
        if _v == _v:
            _finite.append(_v)
    # single comparable scalar (max over motifs) so the legacy one-column tables still resolve
    fields[f'spurious_roc_node_auc_mean{suffix}'] = max(_finite) if _finite else nan
    fields[f'spurious_motifs{suffix}'] = sorted(spur_per)
    return fields


class EvalPipeline:
    """Evaluation pipeline for motif-based GNN explainers.

    Parameters
    ----------
    model : nn.Module
    vocab : VocabData
    test_loader : DataLoader
    test_list : list of Data
        Raw test Data objects.  If the GT cache from ``apply_gt.py`` has been
        loaded, these will already have ``data.node_label`` / ``data.edge_label``
        set.
    device : torch.device
    task_type : str
    max_motifs_eval : int or None
        Limit motif impact evaluation to top-N motifs by frequency.
    top_k : int
        K for top-K vs bottom-K evaluation (default 10).
    correct_pred_threshold : float
        Probability threshold for "correctly predicted" class-1 examples.
    node_att_fn : callable or None
        ``node_att_fn(data) -> Tensor [N]``.  Used for GT ROC when the model's
        forward pass does not return node attention (e.g. VanillaGNN + post-hoc).
        If None, attention is taken from the model's second return value.
    denorm : (mean, std) tuple or None
        Train-split z-score stats for regression targets. When set,
        ``evaluate_predictions`` reports both normalised (``mae``/``rmse``) and
        original-unit (``mae_orig``/``rmse_orig``) metrics in ``prediction``.
    gt_eval_list : list of Data or None
        Subset of test graphs for GT-ROC only (e.g. mutag test mutagens).
        When None, GT-ROC uses the full ``test_list``.
    all_list : list of Data or None
        Train+valid+test graphs for pooled explainability metrics
        (``pearson_all``, ``gt_roc_*_all``). When None, pooled metrics are
        skipped.
    gt_eval_all_list : list of Data or None
        GT-ROC graph subset for ``all_list`` (e.g. mutag mutagens across splits).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        vocab: VocabData,
        test_loader,
        test_list: List,
        device: torch.device,
        task_type: str,
        max_motifs_eval: Optional[int] = None,
        top_k: int = 10,
        correct_pred_threshold: float = 0.5,
        node_att_fn: Optional[Callable] = None,
        gt_level: str = 'node',
        denorm: Optional[tuple] = None,
        gt_eval_list: Optional[List] = None,
        all_list: Optional[List] = None,
        gt_eval_all_list: Optional[List] = None,
    ):
        self.model = model
        self.vocab = vocab
        self.test_loader = test_loader
        self.test_list = test_list
        self.gt_eval_list = gt_eval_list
        self.all_list = all_list
        self.gt_eval_all_list = gt_eval_all_list
        self.device = device
        self.task_type = task_type
        self.max_motifs_eval = max_motifs_eval
        self.top_k = top_k
        self.correct_pred_threshold = correct_pred_threshold
        self.node_att_fn = node_att_fn
        self.gt_level = gt_level
        self.denorm = denorm

    def _gt_graphs(self) -> List:
        """Graphs used for GT-ROC (defaults to full test set)."""
        return self.gt_eval_list if self.gt_eval_list is not None else self.test_list

    def _has_ground_truth(self) -> bool:
        """Check whether GT eval graphs have node_label or edge_label annotations."""
        return self._has_ground_truth_on(self._gt_graphs())

    @staticmethod
    def _has_ground_truth_on(data_list: List) -> bool:
        return any(
            getattr(d, 'edge_label', None) is not None
            or getattr(d, 'node_label', None) is not None
            for d in data_list
        )

    def _per_instance(self, data_list: List, motif_impact: Dict) -> Dict:
        """Per-instance score-vs-impact correlation at BOTH y-axes:
          pearson_instance          — OWN impact (LOO weighted by the model's own
                                       per-node attention; the x is that same
                                       per-instance attention).
          pearson_instance_agnostic — AGNOSTIC impact (LOO with uniform weights;
                                       model-only, the common fair y across
                                       methods). Same x.
        The OWN pair is free (reuses motif_impact); the agnostic adds one
        uniform-weight impact pass."""
        from .embedding_viz import build_impact_cache_from_eval
        score_cache, own_impact = caches_from_motif_impact(motif_impact)
        own = per_instance_correlation_from_caches(score_cache, own_impact)
        out = {
            'pearson_instance':   own['pearson_instance'],
            'spearman_instance':  own['spearman_instance'],
            'n_instances':        own['n_instances'],
        }
        try:
            _ones = lambda d: torch.ones(d.num_nodes)
            agn_impact = build_impact_cache_from_eval(
                self.model, data_list, self.vocab, self.device,
                self.task_type, base_att_fn=_ones)
            agn = per_instance_correlation_from_caches(score_cache, agn_impact)
            out['pearson_instance_agnostic']  = agn['pearson_instance']
            out['spearman_instance_agnostic'] = agn['spearman_instance']
        except Exception as _e:
            print(f'  [warn] agnostic per-instance impact failed ({_e}).')
        return out

    def _compute_explainability(
        self,
        data_list: List,
        gt_eval_list: Optional[List],
        motif_scores: Optional[Dict[int, float]],
        run_motif_impact: bool,
    ) -> Dict:
        """Pearson/Spearman + GT-ROC (+ impacts) on an arbitrary graph list."""
        block: Dict = {}
        gt_graphs = gt_eval_list if gt_eval_list is not None else data_list
        if self._has_ground_truth_on(gt_graphs):
            _is_planted = _has_node_attr(gt_graphs, 'node_label_clauses')
            # Spurious/family shortcut ROC — planted-only, guarded on the attr being present
            # (else compute_gt_roc would silently fall back to the edge_label-derived rule
            # mask and mislabel them). The fired-clause GT-ROC is no longer a separate metric:
            # node_label IS the fired-clause cause since 2026-08.
            if _has_node_attr(gt_graphs, 'node_label_spurious'):
                # [N,S] — one AUC per class-1 correlate, keyed by motif.
                for _j, _nm in enumerate(spurious_motif_names(gt_graphs)):
                    block[f'spurious_roc_node__{_nm}'] = compute_gt_roc(
                        self.model, gt_graphs, self.device,
                        node_att_fn=self.node_att_fn, level='node',
                        gt_attr='node_label_spurious', gt_col=_j)
            if _has_node_attr(gt_graphs, 'node_label_family'):
                block['family_roc_node'] = compute_gt_roc(
                    self.model, gt_graphs, self.device,
                    node_att_fn=self.node_att_fn, level='node',
                    gt_attr='node_label_family')

            if _is_planted:
                # PLANTED / DNF: the paper node metric is the DNF clause-decomposed AUC —
                #   Instance — did the model recover ANY one complete FIRED disjunct (best clause)?
                #   Global   — did it recover the UNION of all fired clauses?
                # PLUS the per-node cause AUC against node_label. Pre-2026-08 that key held the
                # whole-rule Mode-1 mask and was deliberately skipped here as "easily misused";
                # node_label is now FIRED-CLAUSE ONLY, so it is the plain per-node correctness
                # metric and MUST be computed. Omitting it is what left gt_roc_node_auc_mean=nan
                # on planted ante-hoc runs once gt_roc_node_fired was retired — i.e. no
                # correctness number at all for MoSE/GSAT.
                block['gt_roc_node'] = compute_gt_roc(
                    self.model, gt_graphs, self.device,
                    node_att_fn=self.node_att_fn, level='node')
                block['gt_roc_edge'] = compute_gt_roc(
                    self.model, gt_graphs, self.device,
                    node_att_fn=self.node_att_fn, level='edge')
                block['gt_roc'] = (block['gt_roc_node'] if self.gt_level == 'node'
                                   else block['gt_roc_edge'])
                _dnf_fn = self.node_att_fn or model_node_att_fn(self.model, self.device)
                _dnf = compute_dnf_gt_roc(self.model, gt_graphs, self.device, _dnf_fn)
                if _dnf['n_graphs'] > 0:
                    block['instance_gt_roc_node'] = {
                        'auc_mean': _dnf['instance_auc_mean'], 'n_graphs': _dnf['n_graphs']}
                    block['global_gt_roc_node'] = {
                        'auc_mean': _dnf['global_auc_mean'], 'n_graphs': _dnf['n_graphs']}
            else:
                # SOURCE-GT: a SINGLE clause with no clause tensor, so the whole-rule Mode-1
                # node GT-ROC IS the instance metric — computed here and aliased into
                # instance/global. (Original-label runs never reach here: the whole GT block
                # is guarded by _has_ground_truth_on above.)
                block['gt_roc_node'] = compute_gt_roc(
                    self.model, gt_graphs, self.device,
                    node_att_fn=self.node_att_fn, level='node')
                block['gt_roc_edge'] = compute_gt_roc(
                    self.model, gt_graphs, self.device,
                    node_att_fn=self.node_att_fn, level='edge')
                block['gt_roc'] = (block['gt_roc_node']
                                   if self.gt_level == 'node'
                                   else block['gt_roc_edge'])
                block['instance_gt_roc_node'] = dict(block['gt_roc_node'])
                block['global_gt_roc_node'] = dict(block['gt_roc_node'])

        # Motif-attention coherence (within/between-motif variance) — a property of the model's
        # attention, independent of GT. within ≈ 0 for MotifSAT (motif-coherent by construction),
        # > 0 for GSAT (per-node). Logged for every family so the MotifSAT-vs-GSAT table is automatic.
        try:
            from .motif_coherence import motif_attention_coherence
            _catt = self.node_att_fn or model_node_att_fn(self.model, self.device)
            _coh = motif_attention_coherence(self.model, data_list, self.device, _catt)
            block['within_motif_var'] = _coh['within_motif_var']
            block['between_motif_var'] = _coh['between_motif_var']
        except Exception as _e:
            logging.getLogger(__name__).warning("motif_attention_coherence failed: %s", _e)
            raise

        if run_motif_impact and motif_scores is not None:
            impacts = compute_motif_impact(
                self.model, data_list, self.vocab, self.device,
                task_type=self.task_type,
                max_motifs=self.max_motifs_eval,
            )
            if impacts:
                block['motif_impact'] = impacts
                block['correlation'] = score_impact_correlation(
                    motif_scores, impacts)
                # TRUE per-instance correlation (own + agnostic y), merged in.
                block['correlation'].update(self._per_instance(data_list, impacts))
        return block

    def run(
        self,
        motif_scores: Optional[Dict[int, float]] = None,
        motif_scores_all: Optional[Dict[int, float]] = None,
        run_motif_impact: bool = True,
        gt_motif_ids: Optional[Set[int]] = None,
    ) -> Dict:
        """Run full evaluation.

        Parameters
        ----------
        motif_scores : dict[motif_id -> score] or None
            Learned importance scores.  Enables correlation, top/bottom,
            and GT vs outside-GT evaluations.
        motif_scores_all : dict[motif_id -> score] or None
            Scores for pooled train+valid+test correlation (defaults to
            ``motif_scores`` when omitted).
        run_motif_impact : bool
            Set False to skip the (potentially expensive) motif-removal pass.
        gt_motif_ids : set[int] or None
            Vocabulary ids of ground-truth explanatory motifs for the
            GT vs outside-GT analysis.

        Returns
        -------
        dict with keys:
          'prediction'      -- metric dict from evaluate_predictions
          'gt_roc'          -- mean explainer AUC vs edge_label (if available)
          'motif_impact'    -- dict[motif_id -> impact dict]
          'correlation'     -- pearson/spearman score vs impact
          'top_bottom'      -- top-K vs bottom-K impact comparison
          'gt_vs_outside'   -- GT vs non-GT motifs across three subsets
        """
        # Thesis re-eval fast path: when only predictive metrics (train/val/test
        # RMSE/MAE, computed by the caller BEFORE this) are needed, skip the
        # expensive motif-impact pass. RMSE/MAE are unaffected; only the
        # score-vs-impact correlation is dropped (NaN) — not read from these runs.
        import os as _os
        if _os.environ.get('THESIS_SKIP_MOTIF_IMPACT'):
            run_motif_impact = False
        results: Dict = {}

        # 1. Prediction performance (normalised + original units when denorm set)
        results['prediction'] = evaluate_predictions(
            self.model, self.test_loader, self.device, self.task_type,
            denorm=self.denorm,
        )

        # 2. GT explainer ROC — computed at BOTH node and edge level and
        #    reported together. The repo primarily uses node scores, so the
        #    node-level AUC (model attention vs the authoritative node_label)
        #    is the headline; edge-level (vs the AND edge_label) is reported
        #    alongside. results['gt_roc'] stays as the configured primary level
        #    (self.gt_level) for backward compatibility with existing consumers.
        if self._has_ground_truth():
            _gt = self._gt_graphs()
            # Whole-rule Mode-1 GT-ROC (gt_roc_node/edge/gt_roc) is SOURCE-GT ONLY — a single
            # clause, where it IS the instance metric. For planted/DNF it is a different,
            # easily-misused value and is NOT computed here; the correct planted metric is the
            # DNF best-fired-clause instance in summary_splits (evaluate_native_all_splits /
            # _compute_explainability). run() intentionally carries no test-level DNF instance.
            if not _has_node_attr(_gt, 'node_label_clauses'):
                results['gt_roc_node'] = compute_gt_roc(
                    self.model, _gt, self.device,
                    node_att_fn=self.node_att_fn, level='node',
                )
                results['gt_roc_edge'] = compute_gt_roc(
                    self.model, _gt, self.device,
                    node_att_fn=self.node_att_fn, level='edge',
                )
                results['gt_roc'] = (results['gt_roc_node']
                                     if self.gt_level == 'node'
                                     else results['gt_roc_edge'])

            # Per-motif spurious-shortcut ROC, guarded on the attr being present (else
            # compute_gt_roc falls back to the rule mask). node_label is already the
            # fired-clause cause, so there is no separate fired metric any more.
            if _has_node_attr(_gt, 'node_label_spurious'):
                for _j, _nm in enumerate(spurious_motif_names(_gt)):
                    results[f'spurious_roc_node__{_nm}'] = compute_gt_roc(
                        self.model, _gt, self.device,
                        node_att_fn=self.node_att_fn, level='node',
                        gt_attr='node_label_spurious', gt_col=_j)
            if _has_node_attr(_gt, 'node_label_family'):
                results['family_roc_node'] = compute_gt_roc(
                    self.model, _gt, self.device,
                    node_att_fn=self.node_att_fn, level='node',
                    gt_attr='node_label_family')

        # Motif-attention coherence (within/between-motif variance) — model attention structure,
        # independent of GT. within ≈ 0 for MotifSAT (motif-coherent), > 0 for GSAT (per-node).
        try:
            from .motif_coherence import motif_attention_coherence
            _catt = self.node_att_fn or model_node_att_fn(self.model, self.device)
            _coh = motif_attention_coherence(self.model, self.test_list, self.device, _catt)
            results['within_motif_var'] = _coh['within_motif_var']
            results['between_motif_var'] = _coh['between_motif_var']
        except Exception as _e:
            logging.getLogger(__name__).warning("motif_attention_coherence failed: %s", _e)
            raise

        # 3. Motif removal impact
        if run_motif_impact:
            results['motif_impact'] = compute_motif_impact(
                self.model, self.test_list, self.vocab, self.device,
                task_type=self.task_type,
                max_motifs=self.max_motifs_eval,
            )

        has_scores  = motif_scores is not None
        has_impacts = 'motif_impact' in results and results['motif_impact']

        # 4. Score vs impact correlation
        if has_scores and has_impacts:
            results['correlation'] = score_impact_correlation(
                motif_scores, results['motif_impact']
            )
            # TRUE per-instance correlation (own + agnostic y-axis)
            results['correlation'].update(
                self._per_instance(self.test_list, results['motif_impact']))

        # 5. Top-K vs Bottom-K
        if has_scores and has_impacts:
            results['top_bottom'] = top_bottom_motif_eval(
                motif_scores, results['motif_impact'], k=self.top_k
            )

        # 6. GT vs outside-GT
        # Derive GT motif ids from per-node GT labels when not supplied, so
        # gt_vs_outside runs automatically on any GT-annotated run (ante-hoc and
        # vanilla/baselines alike) instead of only when a caller passes them.
        if gt_motif_ids is None:
            _derived = gt_motif_ids_from_labels(self.test_list)
            if _derived:
                gt_motif_ids = _derived
        if has_scores and has_impacts and gt_motif_ids is not None:
            results['gt_vs_outside'] = gt_vs_outside_gt_eval(
                motif_scores=motif_scores,
                motif_impacts=results['motif_impact'],
                gt_motif_ids=gt_motif_ids,
                data_list=self.test_list,
                model=self.model,
                vocab=self.vocab,
                device=self.device,
                task_type=self.task_type,
                threshold=self.correct_pred_threshold,
            )

        # 7. Class-discriminativeness of motifs (label-aware, model-free).
        #    Lets us check whether top-SCORED motifs are actually predictive of
        #    the label, per the concern that explainers can pick up
        #    non-discriminative motifs.
        if self.task_type == 'BinaryClass':
            try:
                disc = motif_class_discriminativeness(
                    self.test_list, self.vocab,
                    max_motifs=self.max_motifs_eval,
                )
                if disc:
                    results['discriminativeness'] = disc
                    if has_scores:
                        results['top_disc_check'] = top_motifs_discriminative_check(
                            motif_scores, disc, k=self.top_k
                        )
            except Exception as e:
                print(f'  [warn] discriminativeness check failed: {e}')

        # 8. Pooled train+valid+test explainability (pearson / spearman / GT-ROC).
        if self.all_list:
            ms_all = (motif_scores_all if motif_scores_all is not None
                      else motif_scores)
            pooled = self._compute_explainability(
                self.all_list, self.gt_eval_all_list, ms_all, run_motif_impact)
            for key, val in pooled.items():
                results[f'{key}_all'] = val

        # Stash scores so to_dataframe can emit the joined score_vs_impact table.
        if motif_scores is not None:
            results['_motif_scores'] = dict(motif_scores)

        return results

    # ── DataFrame conversion ──────────────────────────────────────────────────

    def to_dataframe(self, results: Dict) -> Dict[str, pd.DataFrame]:
        """Convert all result dicts to DataFrames for logging / saving."""
        dfs = {}

        if 'prediction' in results:
            dfs['prediction'] = pd.DataFrame([results['prediction']])

        if 'gt_roc' in results:
            dfs['gt_roc'] = pd.DataFrame([results['gt_roc']])
        if 'gt_roc_node' in results:
            dfs['gt_roc_node'] = pd.DataFrame([results['gt_roc_node']])
        if 'gt_roc_edge' in results:
            dfs['gt_roc_edge'] = pd.DataFrame([results['gt_roc_edge']])

        if 'motif_impact' in results:
            # motif_impact.csv stays one scalar row per motif — strip the
            # per-graph lists (``*_values`` and the aligned ``graph_indices``,
            # used only for instance-level correlation) so they don't get
            # stringified into cells.
            _list_keys = lambda k: k.endswith('_values') or k == 'graph_indices'
            rows = [{'motif_id': mid,
                     **{k: v for k, v in stats.items() if not _list_keys(k)}}
                    for mid, stats in results['motif_impact'].items()]
            if rows:
                dfs['motif_impact'] = (
                    pd.DataFrame(rows)
                    .sort_values('impact', ascending=False)
                    .reset_index(drop=True)
                )
                if any('masking_without_removal' in stats for stats in results['motif_impact'].values()):
                    dfs['masking_without_removal'] = (
                        pd.DataFrame(rows)
                        .sort_values('masking_without_removal', ascending=False)
                        .reset_index(drop=True)
                    )

        if 'correlation' in results:
            dfs['correlation'] = pd.DataFrame([results['correlation']])

        # Joined score↔impact(↔discriminativeness) table — the direct input for
        # score-vs-impact scatter plots. Emitted *instance-level*: one row per
        # (motif, graph), with the motif's score repeated across its graphs and
        # ``impact`` holding that graph's faithful-LOO value (the same per-graph
        # values the instance-level Pearson correlates — masking-without-removal
        # when available, else graph-removal). Matches the original MOSE-GNN,
        # which does NOT average a motif's impact across graphs before plotting.
        motif_scores = results.get('_motif_scores')
        if motif_scores is not None and 'motif_impact' in results and results['motif_impact']:
            imp = results['motif_impact']
            disc = results.get('discriminativeness', {})
            rows = []
            for mid in sorted(set(motif_scores) & set(imp)):
                stats = imp[mid]
                per_graph = _impact_values(stats)   # per-(motif,graph) impacts
                for val in per_graph:
                    rows.append({
                        'motif_id':     mid,
                        'score':        float(motif_scores[mid]),
                        'impact':       val,                       # per-graph
                        'impact_mean':  stats.get('impact'),       # per-motif mean (ref)
                        'masking_without_removal': stats.get('masking_without_removal'),
                        'abs_disc':     disc.get(mid, {}).get('abs_disc'),
                        'presence_auc': disc.get(mid, {}).get('presence_auc'),
                        'motif_smarts': stats.get('motif_smarts'),
                    })
            if rows:
                dfs['score_vs_impact'] = pd.DataFrame(rows)

        if 'discriminativeness' in results:
            rows = [{'motif_id': mid, **stats}
                    for mid, stats in results['discriminativeness'].items()]
            if rows:
                dfs['discriminativeness'] = (
                    pd.DataFrame(rows)
                    .sort_values('abs_disc', ascending=False)
                    .reset_index(drop=True)
                )

        if 'top_disc_check' in results:
            dfs['top_disc_check'] = pd.DataFrame([results['top_disc_check']])

        if 'top_bottom' in results:
            tb = results['top_bottom']
            k  = tb.get('k', len(tb.get('top_k_ids', [])))
            rows = []
            for rank, (tid, bid) in enumerate(
                    zip(tb.get('top_k_ids', []), tb.get('bottom_k_ids', [])), 1):
                rows.append({
                    'rank':           rank,
                    'top_motif_id':   tid,
                    'top_smarts':     tb['top_k_smarts'][rank - 1],
                    'top_score':      tb['top_k_scores'][rank - 1],
                    'top_impact':     tb['top_k_impacts'][rank - 1],
                    'bottom_motif_id': bid,
                    'bottom_smarts':   tb['bottom_k_smarts'][rank - 1],
                    'bottom_score':    tb['bottom_k_scores'][rank - 1],
                    'bottom_impact':   tb['bottom_k_impacts'][rank - 1],
                })
            if rows:
                dfs['top_bottom'] = pd.DataFrame(rows)
            dfs['top_bottom_summary'] = pd.DataFrame([{
                'k':                  k,
                'top_mean_score':     tb.get('top_mean_score'),
                'bottom_mean_score':  tb.get('bottom_mean_score'),
                'top_mean_impact':    tb.get('top_mean_impact'),
                'bottom_mean_impact': tb.get('bottom_mean_impact'),
                'impact_ratio':       tb.get('impact_ratio'),
            }])

        if 'gt_vs_outside' in results:
            rows = []
            for subset_name, subset_stats in results['gt_vs_outside'].items():
                rows.append({'subset': subset_name, **subset_stats})
            dfs['gt_vs_outside'] = pd.DataFrame(rows)

        return dfs

    def print_summary(self, results: Dict) -> None:
        """Print a compact human-readable summary to stdout."""
        print("\nPrediction:")
        for k, v in results.get("prediction", {}).items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        if "gt_roc_node" in results or "gt_roc_edge" in results:
            print("\nExplainer ROC vs ground truth:")
            for _lvl in ("node", "edge"):
                r = results.get(f"gt_roc_{_lvl}")
                if not r:
                    continue
                if r.get("n_graphs", 0) == 0:
                    print(f"  [{_lvl}] no graphs with valid GT labels")
                else:
                    print(f"  [{_lvl}] auc_mean={r['auc_mean']:.4f}  "
                          f"auc_std={r['auc_std']:.4f}  "
                          f"n_graphs={r['n_graphs']}  n_skipped={r['n_skipped']}")
        elif "gt_roc" in results:
            r = results["gt_roc"]
            print("\nExplainer ROC vs ground truth:")
            if r.get("n_graphs", 0) == 0:
                print("  no graphs with valid GT edge labels")
            else:
                print(f"  auc_mean={r['auc_mean']:.4f}  auc_std={r['auc_std']:.4f}"
                      f"  n_graphs={r['n_graphs']}  n_skipped={r['n_skipped']}")

        if "correlation" in results:
            c = results["correlation"]
            print("\nScore-impact correlation:")
            print(f"  Pearson={c['pearson']:.3f}  Spearman={c['spearman']:.3f}")

        if "top_bottom" in results:
            tb = results["top_bottom"]
            k = tb["k"]
            print(f"\nTop-{k} vs Bottom-{k}:")
            print(f"  Top    mean score={tb['top_mean_score']:.3f}  mean impact={tb['top_mean_impact']:.4f}")
            print(f"  Bottom mean score={tb['bottom_mean_score']:.3f}  mean impact={tb['bottom_mean_impact']:.4f}")
            ratio = tb["impact_ratio"]
            ratio_s = f"{ratio:.2f}x" if not (isinstance(ratio, float) and ratio != ratio) else "N/A"
            print(f"  Impact ratio (top/bottom): {ratio_s}")

        if "gt_vs_outside" in results:
            print("\nGT vs non-GT motifs:")
            for subset, stats in results["gt_vs_outside"].items():
                print(f"  [{subset}]  n={stats['n_examples']}  "
                      f"score_auc={stats['score_auc']:.3f}  "
                      f"gt_rank={stats['gt_impact_rank']:.1f}")
                print(f"    GT:     score={stats['gt_mean_score']:.3f}  impact={stats['gt_mean_impact']:.4f}")
                print(f"    non-GT: score={stats['non_gt_mean_score']:.3f}  impact={stats['non_gt_mean_impact']:.4f}")
