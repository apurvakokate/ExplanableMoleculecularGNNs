"""reg_config.py — per (architecture × dataset) regularization coefficients.

Maps (backbone, dataset) -> (ent_reg, size_reg) for MOSE-GNN. PNA is not in the
table by design: it reuses GIN's configuration (per request). Any (backbone,
dataset) not found falls back to DEFAULT_REG.

Used by MOSE-GNN/run.py: when --ent_reg / --size_reg are NOT passed explicitly
on the command line, the values are resolved from this table by the run's
backbone and dataset. Explicit CLI flags always override the table.
"""
from __future__ import annotations

from typing import Optional, Tuple

# (ent_reg, size_reg) per dataset, per architecture.
REG_CONFIG = {
    "GAT": {
        "Mutagenicity":      (0.2, 0.00005),
        "hERG":              (0.2, 0.0),
        "BBBP":              (0.2, 0.0005),
        "Benzene":           (0.2, 0.0005),
        "Alkane_Carbonyl":   (0.2, 0.00005),
        "Fluoride_Carbonyl": (0.2, 0.00005),
        "esol":              (0.2, 0.0005),
        "Lipophilicity":     (0.2, 0.0),
    },
    "GCN": {
        "Mutagenicity":      (0.2, 0.00005),
        "hERG":              (0.1, 0.00005),
        "BBBP":              (0.2, 0.0005),
        "Benzene":           (0.2, 0.0005),
        "Alkane_Carbonyl":   (0.2, 0.0005),
        "Fluoride_Carbonyl": (0.2, 0.00005),
        "esol":              (0.2, 0.0005),
        "Lipophilicity":     (0.2, 0.00005),
    },
    "SAGE": {
        "Mutagenicity":      (0.2, 0.00005),
        "hERG":              (0.1, 0.00005),
        "BBBP":              (0.1, 0.0005),
        "Benzene":           (0.1, 0.0005),
        "Alkane_Carbonyl":   (0.1, 0.00005),
        "Fluoride_Carbonyl": (0.2, 0.00005),
        "esol":              (0.2, 0.0005),
        "Lipophilicity":     (0.1, 0.00005),
    },
    "GIN": {
        "Mutagenicity":      (0.2, 0.00005),
        "hERG":              (0.2, 0.00005),
        "BBBP":              (0.1, 0.0005),
        "Benzene":           (0.2, 0.0005),
        "Alkane_Carbonyl":   (0.2, 0.0),
        "Fluoride_Carbonyl": (0.2, 0.00005),
        "esol":              (0.2, 0.00005),
        "Lipophilicity":     (0.2, 0.00005),
    },
}

# PNA reuses GIN's configuration (per request).
REG_CONFIG["PNA"] = REG_CONFIG["GIN"]

# Verified-GT datasets are corrected-label variants of the same molecules, so
# they share their base dataset's regularization. Added as explicit keys (not a
# silent alias) so resolve_reg stays fail-fast for genuinely unknown datasets.
for _bb_cfg in REG_CONFIG.values():
    for _base in ("Benzene", "Alkane_Carbonyl", "Fluoride_Carbonyl"):
        _bb_cfg.setdefault(f"{_base}_Verified_GT", _bb_cfg[_base])

# Newly-added datasets. mutag IS the Mutagenicity molecules (with source-GT edges),
# so it principledly shares Mutagenicity's regularization. The OGB datasets have no
# tuned analog yet, so they take the table's modal (0.2, 5e-5) as an explicit,
# UNTUNED starting default (kept explicit so resolve_reg stays fail-fast). Override
# per backbone once results are in, or pass --ent_reg/--size_reg.
for _bb_cfg in REG_CONFIG.values():
    _bb_cfg.setdefault("mutag", _bb_cfg["Mutagenicity"])
    _bb_cfg.setdefault("ogbg-molbace", (0.2, 0.00005))
    _bb_cfg.setdefault("ogbg-molhiv", (0.2, 0.00005))

# Referenced only in the resolve_reg error message now (no longer a silent fallback).
DEFAULT_REG: Tuple[float, float] = (0.01, 0.0)


# Per-dataset GNN depth. BBBP uses 2 layers; everything else 3.
NUM_LAYERS_BY_DATASET = {
    "BBBP": 2,
}
DEFAULT_NUM_LAYERS = 3


def resolve_num_layers(dataset: str,
                       num_layers: Optional[int] = None) -> Tuple[int, bool]:
    """Resolve GNN depth for a dataset.

    Explicit num_layers (not None) wins. Otherwise look up by dataset
    (BBBP -> 2, others -> 3). Returns (num_layers, from_table).
    """
    if num_layers is not None:
        return int(num_layers), False
    return int(NUM_LAYERS_BY_DATASET.get(dataset, DEFAULT_NUM_LAYERS)), True


def resolve_reg(backbone: str, dataset: str,
                ent_reg: Optional[float] = None,
                size_reg: Optional[float] = None
                ) -> Tuple[float, float, bool]:
    """Resolve (ent_reg, size_reg) for a run.

    Explicit values (not None) always win. Otherwise look up by
    (backbone, dataset), with PNA->GIN already folded in, and fall back to
    DEFAULT_REG. Returns (ent_reg, size_reg, from_table) where from_table is
    True iff at least one value came from the lookup (not an explicit flag).
    """
    table = REG_CONFIG.get(backbone, {}).get(dataset)
    if (ent_reg is None or size_reg is None) and table is None:
        raise KeyError(
            f"resolve_reg: no (ent_reg, size_reg) configured for "
            f"(backbone={backbone!r}, dataset={dataset!r}). Known backbones: "
            f"{sorted(REG_CONFIG)}; datasets for {backbone!r}: "
            f"{sorted(REG_CONFIG.get(backbone, {}))}. Pass --ent_reg/--size_reg to "
            f"override, or add the pair to REG_CONFIG. Refusing to silently fall "
            f"back to {DEFAULT_REG} (rule: no silent fallback)."
        )
    used_table = False
    if ent_reg is None:
        ent_reg = table[0]
        used_table = True
    if size_reg is None:
        size_reg = table[1]
        used_table = True
    return float(ent_reg), float(size_reg), used_table
