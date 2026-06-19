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

import pandas as pd
import torch

from ..data.vocab import VocabData
from .metrics import evaluate_predictions
from .motif_eval import (
    compute_motif_impact,
    score_impact_correlation,
    top_bottom_motif_eval,
    gt_vs_outside_gt_eval,
    compute_gt_roc,
    motif_class_discriminativeness,
    top_motifs_discriminative_check,
)


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
    ):
        self.model = model
        self.vocab = vocab
        self.test_loader = test_loader
        self.test_list = test_list
        self.device = device
        self.task_type = task_type
        self.max_motifs_eval = max_motifs_eval
        self.top_k = top_k
        self.correct_pred_threshold = correct_pred_threshold
        self.node_att_fn = node_att_fn
        self.gt_level = gt_level

    def _has_ground_truth(self) -> bool:
        """Check whether test_list has node_label or edge_label annotations."""
        return any(
            getattr(d, 'edge_label', None) is not None
            or getattr(d, 'node_label', None) is not None
            for d in self.test_list
        )

    def run(
        self,
        motif_scores: Optional[Dict[int, float]] = None,
        run_motif_impact: bool = True,
        gt_motif_ids: Optional[Set[int]] = None,
    ) -> Dict:
        """Run full evaluation.

        Parameters
        ----------
        motif_scores : dict[motif_id -> score] or None
            Learned importance scores.  Enables correlation, top/bottom,
            and GT vs outside-GT evaluations.
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
        results: Dict = {}

        # 1. Prediction performance
        results['prediction'] = evaluate_predictions(
            self.model, self.test_loader, self.device, self.task_type
        )

        # 2. GT explainer ROC — computed at BOTH node and edge level and
        #    reported together. The repo primarily uses node scores, so the
        #    node-level AUC (model attention vs the authoritative node_label)
        #    is the headline; edge-level (vs the AND edge_label) is reported
        #    alongside. results['gt_roc'] stays as the configured primary level
        #    (self.gt_level) for backward compatibility with existing consumers.
        if self._has_ground_truth():
            results['gt_roc_node'] = compute_gt_roc(
                self.model, self.test_list, self.device,
                node_att_fn=self.node_att_fn, level='node',
            )
            results['gt_roc_edge'] = compute_gt_roc(
                self.model, self.test_list, self.device,
                node_att_fn=self.node_att_fn, level='edge',
            )
            results['gt_roc'] = (results['gt_roc_node']
                                 if self.gt_level == 'node'
                                 else results['gt_roc_edge'])




        # 3. Motif removal impact
        if run_motif_impact:
            results['motif_impact'] = compute_motif_impact(
                self.model, self.test_list, self.vocab, self.device,
                split='test', task_type=self.task_type,
                max_motifs=self.max_motifs_eval,
            )

        has_scores  = motif_scores is not None
        has_impacts = 'motif_impact' in results and results['motif_impact']

        # 4. Score vs impact correlation
        if has_scores and has_impacts:
            results['correlation'] = score_impact_correlation(
                motif_scores, results['motif_impact']
            )

        # 5. Top-K vs Bottom-K
        if has_scores and has_impacts:
            results['top_bottom'] = top_bottom_motif_eval(
                motif_scores, results['motif_impact'], k=self.top_k
            )

        # 6. GT vs outside-GT
        if has_scores and has_impacts and gt_motif_ids is not None:
            results['gt_vs_outside'] = gt_vs_outside_gt_eval(
                motif_scores=motif_scores,
                motif_impacts=results['motif_impact'],
                gt_motif_ids=gt_motif_ids,
                data_list=self.test_list,
                model=self.model,
                vocab=self.vocab,
                device=self.device,
                split='test',
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
                    self.test_list, self.vocab, split='test',
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
            rows = [{'motif_id': mid, **stats}
                    for mid, stats in results['motif_impact'].items()]
            if rows:
                dfs['motif_impact'] = (
                    pd.DataFrame(rows)
                    .sort_values('impact', ascending=False)
                    .reset_index(drop=True)
                )

        if 'correlation' in results:
            dfs['correlation'] = pd.DataFrame([results['correlation']])

        # Joined per-motif score↔impact(↔discriminativeness) table — the direct
        # input for score-vs-impact scatter plots.
        motif_scores = results.get('_motif_scores')
        if motif_scores is not None and 'motif_impact' in results and results['motif_impact']:
            imp = results['motif_impact']
            disc = results.get('discriminativeness', {})
            rows = []
            for mid in sorted(set(motif_scores) & set(imp)):
                rows.append({
                    'motif_id':     mid,
                    'score':        float(motif_scores[mid]),
                    'impact':       imp[mid].get('impact'),
                    'impact_std':   imp[mid].get('impact_std'),
                    'abs_disc':     disc.get(mid, {}).get('abs_disc'),
                    'presence_auc': disc.get(mid, {}).get('presence_auc'),
                    'motif_smarts': imp[mid].get('motif_smarts'),
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
            print(f"  {k}: {v:.4f}")

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
