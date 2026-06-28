"""Motif support threshold configuration — single source of truth.

Used by MotifBreakdown/generate_vocab_rules.py (fold-0 mining) and
SharedModules/data/fold_threshold.py (per-fold re-application).

Edit CHOSEN_THRESHOLD here after phase-2 coverage review.
"""

from __future__ import annotations

from collections import Counter
from typing import Counter as CounterType, Optional, Set, Union

# Fraction of N_trainval (0.002 → motifs need ≥ 0.2% of train+val fragment hits).
IMBALANCE_MARGIN = 0.6

# Unified table (June 2026) — identical across all *_filter variants.
UNIFIED_FILTER_THRESHOLDS: dict[str, float] = {
    'Mutagenicity':      0.002,
    'Benzene':           0.005,
    'BBBP':              0.006,
    'hERG':              0.005,
    'Alkane_Carbonyl':   0.005,
    'Fluoride_Carbonyl': 0.005,
    'esol':              0.002,
    'Lipophilicity':     0.005,
    'freesolv':          0.005,
    'tox21':             0.005,
    'mutag':             0.005,
    'ogbg-molhiv':       0.004,
    'ogbg-molbace':      0.005,
}

CHOSEN_THRESHOLD: dict[str, dict[str, float]] = {
    'all_fallback_bpe_filter':            dict(UNIFIED_FILTER_THRESHOLDS),
    'rbrics_filter':                      dict(UNIFIED_FILTER_THRESHOLDS),
    'rbrics_with_struct_fallback_filter': dict(UNIFIED_FILTER_THRESHOLDS),
    'rbrics_old_filter':                  dict(UNIFIED_FILTER_THRESHOLDS),
}


def get_chosen_threshold(variant: str, dataset: str) -> float:
    """Return CHOSEN_THRESHOLD[variant][dataset]. Raises KeyError if missing."""
    if variant not in CHOSEN_THRESHOLD:
        raise KeyError(
            f"No CHOSEN_THRESHOLD entry for variant={variant!r}. "
            f"Available: {list(CHOSEN_THRESHOLD.keys())}. "
            f"Add it in SharedModules/data/threshold_config.py."
        )
    if dataset not in CHOSEN_THRESHOLD[variant]:
        raise KeyError(
            f"No CHOSEN_THRESHOLD entry for variant={variant!r}, "
            f"dataset={dataset!r}. "
            f"Add {dataset!r} under CHOSEN_THRESHOLD[{variant!r}] "
            f"in SharedModules/data/threshold_config.py."
        )
    return CHOSEN_THRESHOLD[variant][dataset]


def select_threshold_motifs(
    mol_counts: Union[CounterType[str], dict],
    wt_counts_0: Union[CounterType[str], dict],
    wt_counts_1: Union[CounterType[str], dict],
    *,
    n_tv: int,
    n0_tv: Optional[int],
    n1_tv: Optional[int],
    resolved_pct: float,
    is_regression: bool,
) -> Set[str]:
    """Choose SMARTS strings that pass global (+ optional minority) cutoffs.

    Mirrors generate_vocab_rules.run_dataset and coverage_vs_threshold.py.
    """
    global_cut = int(resolved_pct * n_tv)
    threshold_motifs = {m for m, c in mol_counts.items() if c >= global_cut}

    if is_regression or n0_tv is None or n1_tv is None:
        return threshold_motifs

    r0, r1 = n0_tv / max(n_tv, 1), n1_tv / max(n_tv, 1)
    minority = (
        1 if r0 >= IMBALANCE_MARGIN else
        (0 if r1 >= IMBALANCE_MARGIN else None)
    )
    if minority is not None:
        minority_n = n1_tv if minority == 1 else n0_tv
        mb_cut = int(resolved_pct * minority_n)
        wt = wt_counts_1 if minority == 1 else wt_counts_0
        for m, cnt in wt.items():
            if cnt >= mb_cut:
                threshold_motifs.add(m)

    return threshold_motifs
