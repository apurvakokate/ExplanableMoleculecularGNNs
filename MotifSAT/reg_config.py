"""reg_config.py — per-dataset GSAT IB prior-retention schedule (official GSAT).

Official Graph-COM/GSAT uses fixed Concrete temperature (temp=1) for sampling;
`r` is annealed only for the information-bottleneck prior in info_loss.

Values mirror src/configs/*.yml in https://github.com/Graph-COM/GSAT :
  mutag / ba_2motifs / small-graph benchmarks : final_r=0.5, decay_interval=10
  ogbg-mol* / large graph benchmarks          : final_r=0.7, decay_interval=20

CSV datasets in experiment_config.sh follow the mutag/TUD-style schedule (0.5).
Explicit CLI / YAML values always override this table.
"""
from __future__ import annotations

from typing import Optional, Tuple

# IB prior retention floor (info_loss target Bernoulli(r)).
# TUD / fold-CSV molecular benchmarks (same schedule as GSAT mutag).
_FINAL_R_SMALL = 0.5
# OGB molecular graphs (GIN-ogbg_mol.yml).
_FINAL_R_OGB = 0.7

FINAL_R_BY_DATASET = {
    # TUD + project CSV datasets (experiment_config DATASETS_CSV + mutag)
    "mutag": 0.5,
    "Mutagenicity": _FINAL_R_SMALL,
    "BBBP": _FINAL_R_SMALL,
    "hERG": _FINAL_R_SMALL,
    "Benzene": _FINAL_R_SMALL,
    "Alkane_Carbonyl": _FINAL_R_SMALL,
    "Fluoride_Carbonyl": _FINAL_R_SMALL,
    "Lipophilicity": _FINAL_R_SMALL,
    "esol": _FINAL_R_SMALL,
    "freesolv": _FINAL_R_SMALL,
    "tox21": _FINAL_R_SMALL,
    # OGB molecular (official configs)
    "ogbg-molhiv": _FINAL_R_OGB,
    "ogbg-molbace": _FINAL_R_OGB,
    "ogbg-molbbbp": _FINAL_R_OGB,
    "ogbg-moltox21": _FINAL_R_OGB,
    "ogbg-moltoxcast": _FINAL_R_OGB,
    "ogbg-molesol": _FINAL_R_OGB,
    "ogbg-molfreesolv": _FINAL_R_OGB,
    "ogbg-molclintox": _FINAL_R_OGB,
    "ogbg-molsider": _FINAL_R_OGB,
    "ogbg-mollipo": _FINAL_R_OGB,
}

_DECAY_INTERVAL_SMALL = 10
_DECAY_INTERVAL_OGB = 20

DECAY_INTERVAL_BY_DATASET = {
    # TUD + project CSV datasets
    "mutag": _DECAY_INTERVAL_SMALL,
    "Mutagenicity": _DECAY_INTERVAL_SMALL,
    "BBBP": _DECAY_INTERVAL_SMALL,
    "hERG": _DECAY_INTERVAL_SMALL,
    "Benzene": _DECAY_INTERVAL_SMALL,
    "Alkane_Carbonyl": _DECAY_INTERVAL_SMALL,
    "Fluoride_Carbonyl": _DECAY_INTERVAL_SMALL,
    "Lipophilicity": _DECAY_INTERVAL_SMALL,
    "esol": _DECAY_INTERVAL_SMALL,
    "freesolv": _DECAY_INTERVAL_SMALL,
    "tox21": _DECAY_INTERVAL_SMALL,
    # OGB molecular
    "ogbg-molhiv": _DECAY_INTERVAL_OGB,
    "ogbg-molbace": _DECAY_INTERVAL_OGB,
    "ogbg-molbbbp": _DECAY_INTERVAL_OGB,
    "ogbg-moltox21": _DECAY_INTERVAL_OGB,
    "ogbg-moltoxcast": _DECAY_INTERVAL_OGB,
    "ogbg-molesol": _DECAY_INTERVAL_OGB,
    "ogbg-molfreesolv": _DECAY_INTERVAL_OGB,
    "ogbg-molclintox": _DECAY_INTERVAL_OGB,
    "ogbg-molsider": _DECAY_INTERVAL_OGB,
    "ogbg-mollipo": _DECAY_INTERVAL_OGB,
}

DEFAULT_INIT_R = 0.9
DEFAULT_FINAL_R = _FINAL_R_SMALL
DEFAULT_DECAY_INTERVAL = _DECAY_INTERVAL_SMALL
DEFAULT_DECAY_R = 0.1

# ── Per-dataset information-loss coefficient (motif IB strength) ──────────────
# MotifSAT analogue of MOSE's per-dataset (ent_reg, size_reg). An explicit
# --info_loss_coef always overrides the table.
#
# HISTORY / why this changed 2026-07-09:
#   The old default 0.5 was tuned under the *val-AUC* early-stop regime, which
#   stopped training before the attention converged (undertraining). Under the
#   corrected *smoothed-val-loss* early-stop, coef=0.5 UNDER-regularises: the
#   per-motif attention SATURATES toward 1.0 (mean score_median 0.83; GIN ~0.98
#   -> almost every motif "on"), which flattens score discrimination. A 5-arch x
#   5-config grid on mutag (source GT, the sound explanation metric) showed
#   coef=1.0/final_r=0.5 is the sweet spot:
#       config (coef,final_r)   mean gt_roc_node   pearson_motif   auc
#         (1.0, 0.5)  <-- best        0.791             0.338      0.956
#         (1.0, 0.3)                  0.786             0.269      0.948
#         (2.0, 0.5)  over-compress   0.775             0.238      0.951
#         (0.5, 0.5)  old default     0.774            -0.031      0.961
#         (0.5, 0.3)  worst           0.763             0.061      0.952
#   coef=1.0 rescues the saturating GIN case most (gt_roc 0.711 -> 0.789,
#   score_median 0.96 -> 0.61) at negligible AUC cost (0.961 -> 0.956), and is
#   >= old default for every backbone. coef=2.0 over-compresses; lowering final_r
#   alone hurts. So the binary molecular classification datasets move to 1.0.
#   CAVEAT: GT validation is mutag-only (single fold, 5 archs); the *saturation*
#   itself is reproduced on BBBP/Alkane_Carbonyl/Mutagenicity (all archs, 3 folds).
#   Regression / multitask / OGB stay at 0.5 (untested here; OGB uses r=0.7 floor).
_INFO_LOSS_COEF_DEFAULT = 1.0

INFO_LOSS_COEF_BY_DATASET = {
    # Binary molecular classification: coef=1.0 (mutag-GT validated; saturation
    # reproduced on BBBP/Alkane_Carbonyl/Mutagenicity). Was 0.5 pre-2026-07-09.
    "mutag": 1.0,
    "Mutagenicity": 1.0,
    "BBBP": 1.0,
    "hERG": 1.0,
    "Benzene": 1.0,
    "Alkane_Carbonyl": 1.0,
    "Fluoride_Carbonyl": 1.0,
    # Regression / multitask: unchanged (no GT-ROC signal to tune against here).
    "Lipophilicity": 0.5,
    "esol": 0.5,
    "freesolv": 0.5,
    "tox21": 0.5,
    # OGB molecular graphs already use the wider r=0.7 floor; keep the same IB
    # strength unless a dedicated OGB sweep says otherwise.
    "ogbg-molhiv": 0.5,
    "ogbg-molbace": 0.5,
    "ogbg-molbbbp": 0.5,
    "ogbg-moltox21": 0.5,
    "ogbg-moltoxcast": 0.5,
    "ogbg-molesol": 0.5,
    "ogbg-molfreesolv": 0.5,
    "ogbg-molclintox": 0.5,
    "ogbg-molsider": 0.5,
    "ogbg-mollipo": 0.5,
}


def resolve_info_loss_coef(
    dataset: str,
    info_loss_coef: Optional[float] = None,
) -> Tuple[float, bool]:
    """Resolve the motif-IB info_loss_coef for a run.

    Explicit ``info_loss_coef`` (not None) always wins. Otherwise look up by
    dataset, falling back to ``_INFO_LOSS_COEF_DEFAULT``. Returns
    ``(info_loss_coef, from_table)`` where ``from_table`` is True iff the value
    came from the lookup — mirrors ``resolve_reg`` / ``resolve_gsat_r``.
    """
    if info_loss_coef is not None:
        return float(info_loss_coef), False
    return (float(INFO_LOSS_COEF_BY_DATASET.get(dataset, _INFO_LOSS_COEF_DEFAULT)),
            True)


def resolve_gsat_r(
    dataset: str,
    init_r: Optional[float] = None,
    final_r: Optional[float] = None,
    decay_interval: Optional[int] = None,
    decay_r: Optional[float] = None,
) -> Tuple[float, float, Optional[int], Optional[float], bool]:
    """Resolve GSAT IB schedule for a run.

    Returns (init_r, final_r, decay_interval, decay_r, from_table) where
    from_table is True iff at least one of final_r / decay_interval came
    from the lookup table (not an explicit override).
    """
    used_table = False

    if final_r is None:
        final_r = float(FINAL_R_BY_DATASET.get(dataset, DEFAULT_FINAL_R))
        used_table = True
    if decay_interval is None:
        decay_interval = int(
            DECAY_INTERVAL_BY_DATASET.get(dataset, DEFAULT_DECAY_INTERVAL)
        )
        used_table = True
    if init_r is None:
        init_r = DEFAULT_INIT_R
    if decay_r is None:
        decay_r = DEFAULT_DECAY_R

    return float(init_r), float(final_r), decay_interval, float(decay_r), used_table
