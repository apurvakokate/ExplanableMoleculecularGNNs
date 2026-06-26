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
