"""multi_explanation.py — Multiple explanation / co-occurrence hypothesis analysis.

Background (from the paper)
----------------------------
Motifs with high global importance but low mean impact across graphs raise
a question: why is the model assigning high importance to something that
barely changes predictions?

Two hypotheses explain this:

  H2 — Instance-specific necessity
        Motif m is locally high-impact in this graph: masking it significantly
        changes the prediction. m is "necessary" in this context.

  H1 — Alternate explanation / co-occurrence
        Motif m is locally low-impact in this graph, but there exists at
        least one OTHER high-impact motif co-occurring in the same graph.
        m is "overshadowed" by a competing explanation.

  H0 — Neither
        m is low-impact and no other motif in the graph is high-impact.

For each graph g and motif m, exactly one of H0/H1/H2 applies.
Aggregating over all graphs that contain m gives:

    ratio_H2(m) = P(m is locally high-impact)
    ratio_H1(m) = P(m is locally low, but co-occurs with a high-impact motif)
    ratio_H0(m) = 1 - ratio_H1 - ratio_H2

Motifs are then categorised by their mean importance and mean impact
into a 2×2 grid:

    HH: high importance, high impact     (important AND locally necessary)
    HL: high importance, low impact      (important but often overshadowed)
    LH: low importance, high impact      (surprise high impact)
    LL: low importance, low impact       (genuinely unimportant)

Key finding: HL motifs tend to have higher ratio_H1 than HH motifs, and HH
motifs tend to have higher ratio_H2. This confirms the "alternate explanation"
hypothesis: HL motifs are important for *some* graphs (raising their global
importance) but are redundant in others.

Input format
------------
The analysis works on a per-graph, per-motif impact table with columns:
  - motif               : motif identity (SMARTS string or id)
  - graph_id            : graph identifier
  - sigmoid_importance  : learned global motif importance (σ(θ_m))
  - impact              : |p(graph) - p(graph with motif masked)|
  - class_label         : true label (float)
  - original_logit      : model logit on unmasked graph

This maps directly to the output of EvalPipeline.run() when motif_impact
and motif_scores are both available.

API
---
    build_per_graph_impact_df(data_list, impacts, scores, task_type)
        → long-form DataFrame, one row per (graph, motif)

    compute_h1_h2_ratios(df, local_filter='p75', importance_thresh='mean')
        → per-motif table with ratio_H1, ratio_H2, ratio_H0, category

    classify_motif_category(df, importance_thr, impact_thr)
        → adds 'category' column (HH/HL/LH/LL)

    MultiExplanationAnalysis  — convenience class wrapping the above
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_ORDER = ["HH", "HL", "LH", "LL"]

LocalFilterType = Literal["global", "p50", "p75", "beat_unk"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — build long-form impact DataFrame from EvalPipeline outputs
# ─────────────────────────────────────────────────────────────────────────────

def build_per_graph_impact_df(
    data_list: list,
    motif_impacts: Dict[int, Dict],
    motif_scores: Dict[int, float],
    motif_list: Optional[List[str]] = None,
    split: str = "test",
) -> pd.DataFrame:
    """Build a long-form DataFrame from EvalPipeline motif_impact output.

    Parameters
    ----------
    data_list : list of PyG Data
        Test graphs with .smiles, .y, .nodes_to_motifs set.
    motif_impacts : dict[motif_id → {impact, n_graphs, motif_smarts, ...}]
        Per-motif global impact dict from compute_motif_impact().
        NOTE: this gives *global average* impact, not per-graph impact.
        For the H1/H2 analysis we need per-graph impacts; see notes below.
    motif_scores : dict[motif_id → float]
        Learned importance scores (sigmoid of θ_m).
    motif_list : list[str] or None
        SMARTS vocabulary — used to look up motif names by id.
    split : str
        Which split these graphs come from (for graph_id labelling).

    Returns
    -------
    DataFrame with columns:
        graph_id, motif, motif_id, sigmoid_importance, impact,
        class_label, smiles

    Note: impact here is the *global average* impact for that motif, repeated
    across all graphs that contain it. For true per-graph impact you need
    to store the per-graph impact values during compute_motif_impact().
    Use build_per_graph_impact_df_from_masks() for the full per-graph version.
    """
    rows = []
    smi_to_data = {d.smiles: d for d in data_list}

    for mid, stats in motif_impacts.items():
        score = motif_scores.get(mid, float("nan"))
        smarts = stats.get("motif_smarts", motif_list[mid] if motif_list and mid < len(motif_list) else str(mid))
        impact = stats.get("impact", float("nan"))

        # Find which graphs contain this motif
        for d in data_list:
            n2m = getattr(d, "nodes_to_motifs", None)
            if n2m is None:
                continue
            if int(mid) not in n2m.tolist():
                continue
            label = float(d.y.view(-1)[0].item()) if d.y is not None else float("nan")
            rows.append({
                "graph_id":          str(d.smiles) + f"_{split}",
                "smiles":            str(d.smiles),
                "motif_id":          int(mid),
                "motif":             smarts,
                "sigmoid_importance": float(score),
                "impact":            float(impact),
                "class_label":       label,
            })

    return pd.DataFrame(rows)


def build_per_graph_impact_df_from_masks(
    model: torch.nn.Module,
    data_list: list,
    vocab,
    device: torch.device,
    motif_scores: Dict[int, float],
    split: str = "test",
    task_type: str = "BinaryClass",
    max_motifs: Optional[int] = None,
    base_att_fn=None,
) -> pd.DataFrame:
    """Build a per-graph, per-motif impact DataFrame with true per-graph impacts.

    This is the correct input for H1/H2 analysis. For each (graph, motif) pair
    the impact is the faithful zero-weight leave-one-out
    ``|p(g;W) - p(g;W\\m)|`` — the graph-removal path is DISABLED (commented out).

    Masks come from the SAME single source as every other eval step
    (``build_graph_mask_cache`` — derived from ``nodes_to_motifs``), and the
    impact reuses the SAME shared helper as ``compute_motif_impact``
    (``build_faithful_loo_baseline`` + ``loo_impact``), so ``impact`` /
    ``masking_without_removal`` here match the pipeline exactly.
    """
    from .motif_eval import (
        build_graph_mask_cache, _get_probs, _single_prob,
        _injection_modes, _ablate_motif,  # kept for the (commented) removal path
        build_faithful_loo_baseline, loo_impact,
    )

    model.eval()
    mask_cache = build_graph_mask_cache(data_list)
    smi_to_data = {d.smiles: d for d in data_list}
    orig_probs = _get_probs(model, data_list, device, task_type)
    # mask_nodes, mask_edges = _injection_modes(model)   # removal impact disabled
    # Shared faithful-LOO baseline (model's own attention, or explainer weights
    # via base_att_fn) — same definition as the pipeline / compute_motif_impact.
    base_W, p_full_W = build_faithful_loo_baseline(
        model, data_list, device, task_type, base_att_fn=base_att_fn)
    motif_ids = sorted(mask_cache.keys())
    if max_motifs is not None:
        motif_ids = motif_ids[:max_motifs]

    rows = []
    n_skipped = 0
    for mid in motif_ids:
        score = motif_scores.get(mid, float("nan"))
        smarts = (vocab.motif_list[mid]
                  if vocab.motif_list and mid < len(vocab.motif_list) else str(mid))
        for smi, graph_mask in mask_cache[mid].items():
            d = smi_to_data.get(smi)
            orig_p = orig_probs.get(smi)
            if d is None or orig_p is None:
                continue
            # ── Removal-based (graph-ablation) impact — COMMENTED OUT ─────────
            # Impact is the zero-weight faithful LOO only (never node/edge removal).
            # masked = _ablate_motif(d, graph_mask, mask_nodes, mask_edges)
            # if masked is None:
            #     n_skipped += 1
            #     continue
            # masked_p = _single_prob(model, masked, device, task_type)

            nw = loo_impact(model, d, graph_mask, base_W, p_full_W, device, task_type)
            if nw is None:
                n_skipped += 1
                continue
            label = float(d.y.view(-1)[0].item()) if d.y is not None else float("nan")
            row = {
                "graph_id":           smi + f"_{split}",
                "smiles":             smi,
                "motif_id":           int(mid),
                "motif":              smarts,
                "sigmoid_importance": float(score),
                # ``impact`` IS the zero-weight faithful LOO now (removal disabled).
                "impact":                  float(nw),
                "masking_without_removal": float(nw),
                "original_logit":     float(torch.logit(torch.tensor(orig_p)).item())
                                      if 0 < orig_p < 1 else float("nan"),
                "class_label":        label,
            }
            rows.append(row)

    if n_skipped:
        print(f"  [multi_explanation] skipped {n_skipped} (graph, motif) rows "
              f"with no node-weight baseline W (model lacks node attention).")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — mark locally high-impact rows
# ─────────────────────────────────────────────────────────────────────────────

def _mark_local_hi(
    df: pd.DataFrame,
    impact_col: str = "impact",
    graph_col: str = "graph_id",
    motif_col: str = "motif",
    impact_thr_global: float = 0.0,
    local_filter: LocalFilterType = "p75",
    unk_token: str = "UNK",
) -> pd.Series:
    """Return a boolean Series: True where a row is 'locally high-impact'.

    Modes
    -----
    global   impact >= global mean across the whole split
    p50      impact >= 50th percentile within the graph
    p75      impact >= 75th percentile within the graph   (paper default)
    beat_unk impact > mean impact of unknown (UNK) motifs in the same graph
    """
    lft = local_filter.lower()

    if lft == "global":
        return df[impact_col] >= impact_thr_global

    if lft in {"p50", "p75"}:
        q = 0.5 if lft == "p50" else 0.75
        per_graph_thr = (
            df.groupby(graph_col)[impact_col]
            .quantile(q)
            .rename("_local_thr")
        )
        thr = df[[graph_col]].join(per_graph_thr, on=graph_col)["_local_thr"].values
        return df[impact_col] >= thr

    if lft == "beat_unk":
        unk_imp = (
            df.loc[df[motif_col] == unk_token]
            .groupby(graph_col)[impact_col]
            .mean()
            .rename("_unk_imp")
        )
        joined = df[[graph_col]].join(unk_imp, on=graph_col)
        fallback = joined["_unk_imp"].fillna(impact_thr_global).values
        return df[impact_col] > fallback

    raise ValueError(f"Unknown local_filter: {local_filter!r}. "
                     f"Choose from 'global', 'p50', 'p75', 'beat_unk'.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — assign H0/H1/H2 flags per (graph, motif) row
# ─────────────────────────────────────────────────────────────────────────────

def assign_hypothesis_flags(
    df: pd.DataFrame,
    impact_col: str = "impact",
    graph_col: str = "graph_id",
    motif_col: str = "motif",
    importance_col: str = "sigmoid_importance",
    local_filter: LocalFilterType = "p75",
    importance_thresh: Union[float, str] = "mean",
    unk_token: str = "UNK",
) -> pd.DataFrame:
    """Add H0, H1, H2 boolean columns to a per-(graph, motif) DataFrame.

    Definition (per the paper)
    --------------------------
    H2  (locally high-impact / instance-specific necessity):
        impact >= per-graph quantile threshold

    H1  (alternate explanation / co-occurrence):
        NOT H2, but the same graph contains at least one motif that IS H2
        AND that motif also has high global importance.

    H0  (unexplained):
        Neither H2 nor H1.

    Parameters
    ----------
    df               : output of build_per_graph_impact_df_from_masks()
    impact_col       : column name for per-graph impact values
    local_filter     : 'global' | 'p50' | 'p75' | 'beat_unk'
    importance_thresh: float or 'mean' — threshold for global importance
    unk_token        : motif label for unknown/UNK motifs (excluded from H1 anchor)

    Returns
    -------
    df with additional columns: is_local_hi, H2, H1, H0
    """
    df = df.copy()

    mean_impact     = float(df[impact_col].mean())
    mean_importance = float(df[importance_col].mean())

    imp_thr = (mean_importance
               if isinstance(importance_thresh, str) and importance_thresh.lower() == "mean"
               else float(importance_thresh))

    # is_local_hi: per-row local high marker
    df["is_local_hi"] = _mark_local_hi(
        df, impact_col, graph_col, motif_col, mean_impact, local_filter, unk_token
    )

    # H2: locally high impact in this graph
    df["H2"] = df["is_local_hi"]

    # Graphs that have at least one H2-and-high-importance motif (the anchor for H1)
    h2_hi_imp = df.loc[
        df["H2"] & (df[importance_col] > imp_thr),
        graph_col,
    ].unique()
    graphs_with_anchor = set(h2_hi_imp)

    # H1: locally low, but another high-importance motif is H2 in the same graph
    df["H1"] = (~df["H2"]) & (df[graph_col].isin(graphs_with_anchor))

    # H0: neither
    df["H0"] = (~df["H2"]) & (~df["H1"])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — aggregate to per-motif ratios
# ─────────────────────────────────────────────────────────────────────────────

def compute_h1_h2_ratios(
    df: pd.DataFrame,
    impact_col: str = "impact",
    graph_col: str = "graph_id",
    motif_col: str = "motif",
    importance_col: str = "sigmoid_importance",
    class_col: str = "class_label",
    local_filter: LocalFilterType = "p75",
    importance_thresh: Union[float, str] = "mean",
    unk_token: str = "UNK",
    min_graphs: int = 5,
) -> pd.DataFrame:
    """Compute per-motif H1/H2/H0 ratios and classify into HH/HL/LH/LL.

    Parameters
    ----------
    df              : per-(graph, motif) DataFrame from build_per_graph_impact_df*
    local_filter    : how to define 'locally high impact' within a graph
    importance_thresh: threshold for high global importance
    min_graphs      : drop motifs appearing in fewer than this many graphs

    Returns
    -------
    DataFrame with one row per motif, columns:
        motif, avg_importance, avg_impact, total_graphs,
        n_H2, n_H1, n_H0, ratio_H2, ratio_H1, ratio_H0,
        category (HH/HL/LH/LL),
        purity_ratio, cooc_avg_impact,
        local_filter, importance_threshold, impact_threshold
    """
    # Ensure deduplicated (one row per motif per graph)
    df = df.drop_duplicates(subset=[motif_col, graph_col], keep="first").copy()

    # Exclude UNK from analysis unless using beat_unk filter
    if local_filter != "beat_unk":
        df = df[df[motif_col] != unk_token].copy()

    # Assign H flags
    df = assign_hypothesis_flags(
        df, impact_col, graph_col, motif_col, importance_col,
        local_filter, importance_thresh, unk_token,
    )

    mean_impact     = float(df[impact_col].mean())
    mean_importance = float(df[importance_col].mean())
    imp_thr = (mean_importance
               if isinstance(importance_thresh, str) and importance_thresh.lower() == "mean"
               else float(importance_thresh))

    # Per-graph co-occurrence: average impact of all OTHER motifs in the same graph
    per_graph = (
        df.groupby(graph_col, as_index=False)
        .agg(_sum_impact=(impact_col, "sum"), _count=(motif_col, "size"))
    )
    df = df.merge(per_graph, on=graph_col, how="left")
    denom = (df["_count"] - 1).clip(lower=1).astype(float)
    df["cooc_avg_impact"] = (df["_sum_impact"] - df[impact_col]) / denom

    # Per-motif aggregation
    agg = (
        df.groupby(motif_col, as_index=False)
        .agg(
            total_graphs=(graph_col, "nunique"),
            n_H2=("H2", "sum"),
            n_H1=("H1", "sum"),
            n_H0=("H0", "sum"),
            avg_importance=(importance_col, "mean"),
            avg_impact=(impact_col, "mean"),
            cooc_avg_impact=("cooc_avg_impact", "mean"),
        )
    )

    # Filter rare motifs
    agg = agg[agg["total_graphs"] >= min_graphs].copy()
    if agg.empty:
        return agg

    # Ratios
    agg["ratio_H2"] = agg["n_H2"] / agg["total_graphs"]
    agg["ratio_H1"] = agg["n_H1"] / agg["total_graphs"]
    agg["ratio_H0"] = agg["n_H0"] / agg["total_graphs"]

    # Category: use dataset-level means as thresholds
    agg = classify_motif_category(agg, imp_thr, mean_impact)

    # Label purity ratio (classification: fraction of dominant class)
    if class_col in df.columns and not df[class_col].isna().all():
        purity = (
            df.groupby(motif_col)
            .agg(
                _total=(graph_col, "nunique"),
                _c1=(class_col, lambda x: (x >= 0.5).sum()),
                _c0=(class_col, lambda x: (x < 0.5).sum()),
            )
            .reset_index()
        )
        purity["purity_ratio"] = purity[["_c1", "_c0"]].max(axis=1) / purity["_total"].clip(lower=1)
        agg = agg.merge(purity[[motif_col, "purity_ratio"]], on=motif_col, how="left")

    # Metadata
    agg["local_filter"]          = local_filter
    agg["importance_threshold"]  = imp_thr
    agg["impact_threshold"]      = mean_impact

    return agg


def classify_motif_category(
    df: pd.DataFrame,
    importance_thr: float,
    impact_thr: float,
    importance_col: str = "avg_importance",
    impact_col: str = "avg_impact",
) -> pd.DataFrame:
    """Add a 'category' column (HH/HL/LH/LL) based on thresholds.

    HH: avg_importance > thr  AND  avg_impact > thr
    HL: avg_importance > thr  AND  avg_impact <= thr
    LH: avg_importance <= thr AND  avg_impact > thr
    LL: avg_importance <= thr AND  avg_impact <= thr
    """
    df = df.copy()
    imp_hi    = df[importance_col] > importance_thr
    impact_hi = df[impact_col]     > impact_thr
    cat = pd.Series("LL", index=df.index, dtype=object)
    cat[imp_hi &  impact_hi] = "HH"
    cat[imp_hi & ~impact_hi] = "HL"
    cat[~imp_hi &  impact_hi] = "LH"
    df["category"] = cat
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — summary statistics comparing categories
# ─────────────────────────────────────────────────────────────────────────────

def category_summary(
    ratios_df: pd.DataFrame,
    categories: List[str] = CATEGORY_ORDER,
) -> pd.DataFrame:
    """Summarise mean H1/H2/H0 ratios per category.

    Parameters
    ----------
    ratios_df : output of compute_h1_h2_ratios()

    Returns
    -------
    DataFrame: one row per category, columns:
        category, n_motifs, mean_ratio_H2, mean_ratio_H1, mean_ratio_H0,
        mean_importance, mean_impact
    """
    if "category" not in ratios_df.columns or ratios_df.empty:
        return pd.DataFrame()

    rows = []
    for cat in categories:
        sub = ratios_df[ratios_df["category"] == cat]
        if sub.empty:
            continue
        rows.append({
            "category":       cat,
            "n_motifs":       len(sub),
            "mean_ratio_H2":  float(sub["ratio_H2"].mean()),
            "mean_ratio_H1":  float(sub["ratio_H1"].mean()),
            "mean_ratio_H0":  float(sub["ratio_H0"].mean()),
            "mean_importance": float(sub["avg_importance"].mean()),
            "mean_impact":    float(sub["avg_impact"].mean()),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

class MultiExplanationAnalysis:
    """Convenience class for the full H1/H2 analysis pipeline.

    Usage
    -----
        analysis = MultiExplanationAnalysis(
            model, vocab, data_list, device,
            motif_scores=model.get_motif_scores(),
        )
        # Build per-graph impact table (requires running masked forward passes)
        analysis.run(local_filter='p75', importance_thresh='mean')
        df = analysis.ratios_df          # per-motif H1/H2 table
        summary = analysis.summary_df    # per-category summary
        analysis.save('output/multi_explanation.csv')

    Parameters
    ----------
    model       : trained GNN model
    vocab       : VocabData
    data_list   : list of test PyG Data objects (with nodes_to_motifs)
    device      : torch.device
    motif_scores: dict[motif_id → float]
    split       : split name for graph_id labelling (default 'test')
    task_type   : 'BinaryClass' | 'Regression' | 'MultiLabel'
    """

    def __init__(
        self,
        model: torch.nn.Module,
        vocab,
        data_list: list,
        device: torch.device,
        motif_scores: Dict[int, float],
        split: str = "test",
        task_type: str = "BinaryClass",
        max_motifs: Optional[int] = None,
        min_graphs: int = 5,
    ):
        self.model        = model
        self.vocab        = vocab
        self.data_list    = data_list
        self.device       = device
        self.motif_scores = motif_scores
        self.split        = split
        self.task_type    = task_type
        self.max_motifs   = max_motifs
        self.min_graphs   = min_graphs

        self._raw_df:     Optional[pd.DataFrame] = None
        self.ratios_df:   Optional[pd.DataFrame] = None
        self.summary_df:  Optional[pd.DataFrame] = None

    def run(
        self,
        local_filter: LocalFilterType = "p75",
        importance_thresh: Union[float, str] = "mean",
        unk_token: str = "UNK",
    ) -> "MultiExplanationAnalysis":
        """Run the full pipeline: build per-graph impacts, compute ratios, summarise.

        This performs |vocab| forward passes with masked node features — may be
        slow for large vocabs. Set max_motifs in the constructor to limit.
        """
        print("Building per-graph impact table ...")
        self._raw_df = build_per_graph_impact_df_from_masks(
            model=self.model,
            data_list=self.data_list,
            vocab=self.vocab,
            device=self.device,
            motif_scores=self.motif_scores,
            split=self.split,
            task_type=self.task_type,
            max_motifs=self.max_motifs,
        )
        n = len(self._raw_df)
        print(f"  {n} (graph, motif) rows for "
              f"{self._raw_df['motif'].nunique()} motifs across "
              f"{self._raw_df['graph_id'].nunique()} graphs")

        print(f"Computing H1/H2 ratios (filter={local_filter}) ...")
        self.ratios_df = compute_h1_h2_ratios(
            self._raw_df,
            local_filter=local_filter,
            importance_thresh=importance_thresh,
            unk_token=unk_token,
            min_graphs=self.min_graphs,
        )

        self.summary_df = category_summary(self.ratios_df)
        print("Category summary:")
        print(self.summary_df.to_string(index=False))
        return self

    def save(self, path: str) -> None:
        """Save ratios_df and summary_df as CSV files."""
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        stem = p.stem
        suffix = p.suffix or ".csv"
        if self.ratios_df is not None:
            self.ratios_df.to_csv(p.parent / f"{stem}_per_motif{suffix}", index=False)
        if self.summary_df is not None:
            self.summary_df.to_csv(p.parent / f"{stem}_category_summary{suffix}", index=False)
        print(f"Saved to {p.parent}/{stem}_*.csv")

    @property
    def hl_motifs(self) -> pd.DataFrame:
        """Subset of ratios_df for HL motifs (high importance, low impact)."""
        if self.ratios_df is None:
            return pd.DataFrame()
        return self.ratios_df[self.ratios_df["category"] == "HL"].copy()

    @property
    def hh_motifs(self) -> pd.DataFrame:
        """Subset for HH motifs."""
        if self.ratios_df is None:
            return pd.DataFrame()
        return self.ratios_df[self.ratios_df["category"] == "HH"].copy()
