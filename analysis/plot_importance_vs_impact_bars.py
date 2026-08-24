#!/usr/bin/env python3
"""plot_importance_vs_impact_bars.py — rbrics importance-vs-impact bar figures.

Pure post-processing of the SAME per-motif importance / impact CSVs that feed the
grouped-Pearson metric (``{method}_grouped_corr_pooled_alltest.csv``). No model
runs. Reuses helpers from ``analysis/plot_score_vs_impact.py``.

Two deliverables (spec: ``analysis/PLAN_importance_impact_plots.md``):

  1. **MOSE importance vs impact — three box series per bin + motif-count strip.**
     The original single-figure grid (rows = dataset, columns = backbone), kept
     as-is; the only change is a **third box series for MoSE + Learn-Unknown**.
     Per equal-width bin of MOSE motif importance, three whiskered box series show
     the impact distribution of those motifs under: MoSE native LOO (purple), MoSE +
     Learn-Unknown native LOO (teal), and the vanilla model (grey, model-agnostic
     LOO — ``impact_cache_agnostic_{split}.json``). MoSE-LU and vanilla impact are
     joined onto MoSE's motifs by ``motif_smarts``. A motif-count strip sits beneath
     each cell. Sources (rbrics_filter, real tier): MoSE = antehoc_v1 (``unk-fixed``);
     MoSE-LU = ``ablated_completely_v1`` (``unk-learnable_shared``, learn-unknown knob
     only — ``--mose_lu_root``); vanilla = posthoc_v1 agnostic cache.

  2. **All-methods grouped bars (score bin × method), full vs filtered vocab.** One
     ACM-styled grid figure, rows = dataset, columns = backbone, facets sharing x/y.
     Per score bin, one bar per method. Two versions differ ONLY by UNK handling:
     ``inclunk`` (full vocab, keep motif_id == -1) vs ``exclunk`` (drop it). All 6
     methods appear in BOTH. Each method's score is min-max-normalized to [0, 1]
     before binning, so the shared x-axis is meaningful across wildly different score
     scales.

Scope (LOCKED — do not change):
  * rbrics vocabulary ONLY. MoSE → ``rbrics_filter``; GSAT + the 4 post-hoc → ``rbrics``.
  * Real tier by default. Methods: MoSE, GSAT, GNNExplainer, PGExplainer,
    Motif-Occlusion, MAGE (= ``mage_v2``; NOT ``mage_official``; ``motifsat`` ignored).
    GSAT's file stem is ``gsat_`` under ``base_gsat/``.
  * Impact axis is already fair: ``{method}_impact_{split}.csv`` (native own for
    MoSE/GSAT; agnostic for post-hoc). Plot-1 grey bar = the agnostic cache.
  * Bars = UNWEIGHTED means; each bar carries an IQR (Q1–Q3) whisker.
  * Pool ``(motif, split)`` rows over train+valid+test (``alltest``), pool folds.
  * Cross-vocab joins by ``motif_smarts``, never ``motif_id``.

Neither ``antehoc_v1`` nor ``posthoc_v1`` run dirs carry ``summary.json`` (only a
method-keyed ``summary_splits.json``), so metadata is parsed from the run-dir PATH.

Validation (``--validate``): the unweighted, UNK-included Pearson over each setup's
loaded pooled points reproduces ``pearson_u_inclunk`` in that setup's
``{method}_grouped_corr_pooled_alltest.csv``.

Usage
-----
    python analysis/plot_importance_vs_impact_bars.py \\
        --antehoc_root antehoc_v1 --posthoc_root posthoc_v1 \\
        --save_dir analysis/plots_importance_impact --nbins 6 --validate
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib
matplotlib.use('Agg')  # reuse: headless backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# reuse helpers from the sibling script (as mandated by the plan)
from analysis.plot_score_vs_impact import (
    _score_bins_equal, _assign_bins, _ordered, _safe_slug, _FAMILY_TITLES,
)

# ── constants ────────────────────────────────────────────────────────────────
UNK_ID = -1

# Post-hoc methods (all four live side-by-side in one posthoc run dir; they share
# the agnostic cache). MAGE = mage_v2 only.
POSTHOC_METHODS = ('gnnexplainer', 'pgexplainer', 'motif_occlusion', 'mage_v2')

# The 6 methods that appear in Plot 2 (order = grouped-bar / legend order).
# MoSE is placed LAST in each group and visually emphasised (see make_plot2).
PLOT2_METHODS = ('gsat', 'gnnexplainer', 'pgexplainer', 'motif_occlusion',
                 'mage_v2', 'mose')
# The method to emphasise (bold edge + full opacity; others dimmed).
PLOT2_HIGHLIGHT = 'mose'

# method → the rbrics variant it is scored on (standing convention).
METHOD_VARIANT = {
    'mose': 'rbrics_filter',
    'gsat': 'rbrics',
    'gnnexplainer': 'rbrics',
    'pgexplainer': 'rbrics',
    'motif_occlusion': 'rbrics',
    'mage_v2': 'rbrics',
}

# Title map (extends the reused _FAMILY_TITLES, which lacks mage_v2).
_TITLES = dict(_FAMILY_TITLES)
_TITLES.setdefault('mage_v2', 'MAGE')
_TITLES.setdefault('gsat', 'GSAT')


def _title(method: str) -> str:
    return _TITLES.get(method, method.replace('_', ' ').title())


# Maximally-distinct qualitative palette (Trubetskoy): six well-separated hues,
# each also a different grayscale value so bars stay distinguishable in B&W.
METHOD_COLOR = {
    'mose': '#911eb4',            # purple (proposed method; heavier edge)
    'gsat': '#e6194b',            # red
    'gnnexplainer': '#4363d8',    # blue
    'pgexplainer': '#f58231',     # orange
    'motif_occlusion': '#3cb44b', # green
    'mage_v2': '#ffe119',         # yellow
}
_MOSE_COLOR = METHOD_COLOR['mose']
_AGNOSTIC_COLOR = '#9e9e9e'       # grey vanilla-agnostic bar/box (Plot 1)
_MOSE_LU_COLOR = '#1b9e77'        # teal — MoSE + Learn-Unknown series (Plot 1)

# Colour-blind-safe palette per GNN backbone (Plot 2 grouped bars).
BACKBONE_COLOR = {
    'GIN': '#0072B2', 'GCN': '#E69F00', 'SAGE': '#009E73',
    'GraphSAGE': '#009E73', 'GAT': '#CC79A7', 'PNA': '#D55E00',
}

_BACKBONE_ORDER = ('GIN', 'GCN', 'GraphSAGE', 'SAGE', 'GAT', 'PNA')
_DATASET_ORDER = (
    'BBBP', 'Mutagenicity', 'hERG',
    'Benzene_Verified_GT', 'Alkane_Carbonyl_Verified_GT',
    'Fluoride_Carbonyl_Verified_GT',
    'esol', 'Lipophilicity',
    # excluded by default (kept for ordering if ever included):
    'mutag', 'ogbg-molbace', 'ogbg-molhiv',
)

# Plot-2 figure split (requested): GIN gets its own figure (all datasets); the
# remaining backbones are shown for two dataset groups. Each entry is
# (tag, datasets|None=all, backbones|None=all).
# (tag, datasets|None=all, backbones|None=all, ds_grid_ncols|None). When a group
# has a single backbone and ds_grid_ncols is set, its datasets are tiled into a
# grid that many columns wide (GIN → 2×4) instead of a tall single column.
PLOT2_GROUPS = [
    ('gin', None, ['GIN'], 4),
    ('good', ['BBBP', 'Mutagenicity', 'hERG', 'Benzene_Verified_GT'],
     ['GCN', 'SAGE', 'GAT', 'PNA'], None),
    ('medium', ['Alkane_Carbonyl_Verified_GT', 'Fluoride_Carbonyl_Verified_GT',
                'esol', 'Lipophilicity'],
     ['GCN', 'SAGE', 'GAT', 'PNA'], None),
]

# Compact row labels so long dataset names stay legible in a page-width grid.
_DS_DISPLAY = {
    'Benzene_Verified_GT': 'Benzene',
    'Alkane_Carbonyl_Verified_GT': 'Alkane–\nCarbonyl',
    'Fluoride_Carbonyl_Verified_GT': 'Fluoride–\nCarbonyl',
    'esol': 'ESOL',
}


def _ds_label(ds: str) -> str:
    return _DS_DISPLAY.get(ds, ds)


_SPLITS_ALL = ('train', 'valid', 'test')


def _apply_acm_style() -> None:
    """Publication (ACM) styling: sans-serif (Helvetica/Arial), heavier/larger
    text so ticks survive PDF compression, embedded (Type-42) fonts."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'Nimbus Sans',
                            'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans',
        'font.size': 9,
        'font.weight': 'medium',
        'axes.titlesize': 11,
        'axes.titleweight': 'bold',
        'axes.labelsize': 10,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 9,
        'axes.linewidth': 0.9,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 0.9,
        'ytick.major.width': 0.9,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'savefig.dpi': 300,
        'figure.dpi': 150,
        # ACM requires embedded TrueType, never Type-3 bitmaps.
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


# ── metadata: parse the run-dir PATH (no summary.json exists in these trees) ──

def _canon_tier(tier: str) -> str:
    """Underscore-insensitive tier key so posthoc ``gt_dnf_k3_r1`` and antehoc
    ``dnf_k3r1`` compare equal; ``real`` stays ``real``."""
    t = (tier or '').strip().lower()
    if t in ('real', '', 'none'):
        return 'real'
    return re.sub(r'[_\-]', '', t)


def _tier_from_variant(variant_full: str) -> tuple[str, str]:
    """Split an antehoc variant segment into (base_variant, tier).

    ``rbrics``                    → ('rbrics', 'real')
    ``rbrics_filter``             → ('rbrics_filter', 'real')
    ``rbrics_relabelled_dnf_k3r1``→ ('rbrics', 'gt_dnf_k3r1')
    ``rbrics_filter_relabelled_dnf_k2r2`` → ('rbrics_filter', 'gt_dnf_k2r2')
    """
    m = re.search(r'_relabelled_(.+)$', variant_full)
    if not m:
        return variant_full, 'real'
    base = variant_full[:m.start()]
    tail = m.group(1)                       # e.g. 'dnf_k3r1'
    tier = tail if tail.startswith('gt_') else f'gt_{tail}'
    return base, tier


def _setup_posthoc(run_dir: Path):
    """Parse a posthoc run dir. Returns dict or None if not a rbrics leaf.

    Layout A: ``.../baselines/<DATASET>/fold<K>/<VARIANT>/<LEAF>`` where
    LEAF = ``bb-<BB>_enc-<enc>_norm-none_<TIER>`` (enc ∈ {onehot, atom_encoder};
    TIER ∈ {real, gt_dnf_kX_rY}).
    """
    parts = run_dir.parts
    if 'baselines' not in parts:
        return None
    i = parts.index('baselines')
    try:
        dataset = parts[i + 1]
        fold = int(re.match(r'fold(\d+)', parts[i + 2]).group(1))
        variant = parts[i + 3]
        leaf = parts[i + 4]
    except (IndexError, AttributeError):
        return None
    if variant not in ('rbrics', 'rbrics_filter'):
        return None
    if '_norm-none_' not in leaf:
        return None
    left, tier = leaf.split('_norm-none_', 1)
    m = re.match(r'bb-([^_]+)', left)
    if not m:
        return None
    return dict(dataset=dataset, backbone=m.group(1), variant=variant,
                tier=tier, fold=fold)


def _setup_antehoc(run_dir: Path, family_dir: str):
    """Parse an antehoc run dir (``mose`` or ``base_gsat``). Returns dict or None.

    Layout B: ``.../<family_dir>/<VARIANT[_relabelled_...]>/<DATASET>/fold<K>/<LEAF>``
    where LEAF starts with ``<BB>_``.
    """
    parts = run_dir.parts
    if family_dir not in parts:
        return None
    i = parts.index(family_dir)
    try:
        variant_full = parts[i + 1]
        dataset = parts[i + 2]
        fold = int(re.match(r'fold(\d+)', parts[i + 3]).group(1))
        leaf = parts[i + 4]
    except (IndexError, AttributeError):
        return None
    base_variant, tier = _tier_from_variant(variant_full)
    backbone = leaf.split('_')[0]
    return dict(dataset=dataset, backbone=backbone, variant=base_variant,
                tier=tier, fold=fold)


# ── per-motif loading ────────────────────────────────────────────────────────

def _read_pair(run_dir: Path, stem: str, split: str) -> pd.DataFrame | None:
    """Join ``{stem}_importance_{split}.csv`` (x=score) ⨝ ``{stem}_impact_{split}.csv``
    (y=impact) on motif_id → one frame of pooled points for this split."""
    imp = run_dir / f'{stem}_importance_{split}.csv'
    ipa = run_dir / f'{stem}_impact_{split}.csv'
    if not (imp.exists() and ipa.exists()):
        return None
    try:
        di = pd.read_csv(imp)          # motif_id, score, support, motif_smarts
        dp = pd.read_csv(ipa)          # motif_id, impact, support, motif_smarts
    except Exception:
        return None
    if 'score' not in di.columns or 'impact' not in dp.columns:
        return None
    di = di[['motif_id', 'score', 'support', 'motif_smarts']]
    dp = dp[['motif_id', 'impact']]
    m = di.merge(dp, on='motif_id', how='inner')
    if m.empty:
        return None
    m['split'] = split
    return m


def _load_setup_points(run_dir: Path, stem: str, splits) -> pd.DataFrame | None:
    """Pooled (motif, split) points for one setup: concat the per-split joins."""
    frames = [f for f in (_read_pair(run_dir, stem, s) for s in splits)
              if f is not None]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# ── grouped-Pearson math (exact, from split_eval.py) — for --validate ─────────

def _wpearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
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


def _validate_setup(run_dir: Path, stem: str, pts: pd.DataFrame,
                    report: list[dict]) -> None:
    """Check unweighted incl-UNK Pearson over ``pts`` == CSV ``pearson_u_inclunk``."""
    csv = run_dir / f'{stem}_grouped_corr_pooled_alltest.csv'
    if not csv.exists():
        return
    try:
        row = pd.read_csv(csv).iloc[0]
        expected = float(row['pearson_u_inclunk'])
    except Exception:
        return
    got = _wpearson(pts['score'].values, pts['impact'].values,
                    np.ones(len(pts)))
    ok = (np.isnan(expected) and np.isnan(got)) or (
        np.isfinite(expected) and np.isfinite(got)
        and abs(expected - got) < 1e-6)
    report.append(dict(run_dir=str(run_dir), stem=stem,
                       expected=expected, got=got,
                       diff=(float('nan') if not np.isfinite(expected)
                             else abs(expected - got)),
                       ok=bool(ok)))


# ── walkers ──────────────────────────────────────────────────────────────────

def _iter_run_dirs(root: Path, subdir: str):
    base = root / subdir
    if not base.exists():
        return
    for sj in base.rglob('summary_splits.json'):
        yield sj.parent


def _as_roots(x):
    """Normalise a root arg (None | str | Path | list) to a list[Path]."""
    if x is None:
        return []
    if isinstance(x, (str, Path)):
        return [Path(x)]
    return [Path(p) for p in x]


def _iter_run_dirs_multi(roots, subdir: str):
    """`_iter_run_dirs` fanned out over several roots (e.g. eval_real + eval_gath1
    so a single call covers non-GAT and GAT trees from the bundle)."""
    for r in _as_roots(roots):
        yield from _iter_run_dirs(r, subdir)


def load_method_points(antehoc_root, posthoc_root, *,
                       tier: str, pooling: str, datasets: set | None,
                       validate: bool, mose_unk: str | None = None
                       ) -> tuple[pd.DataFrame, list[dict]]:
    """Long frame of pooled per-motif points for all 6 methods, real tier.

    Columns: method, dataset, backbone, fold, motif_id, motif_smarts, split,
    score, impact, support.
    """
    splits = _SPLITS_ALL if pooling == 'alltest' else ('test',)
    want_tier = _canon_tier(tier)
    report: list[dict] = []
    recs: list[pd.DataFrame] = []

    def _keep(meta) -> bool:
        if meta is None:
            return False
        if _canon_tier(meta['tier']) != want_tier:
            return False
        if datasets and meta['dataset'] not in datasets:
            return False
        return True

    # --- ante-hoc MoSE (rbrics_filter) & GSAT (rbrics) ---
    for subdir, stem, family, variant_ok in (
            ('mose', 'mose', 'mose', 'rbrics_filter'),
            ('base_gsat', 'gsat', 'gsat', 'rbrics')):
        for run_dir in _iter_run_dirs_multi(antehoc_root, subdir):
            meta = _setup_antehoc(run_dir, subdir)
            if not _keep(meta) or meta['variant'] != variant_ok:
                continue
            # bundle: eval_real/mose holds BOTH unk-fixed and unk-learnable_shared
            # leaves; keep only the requested MoSE-native variant here.
            if subdir == 'mose' and mose_unk and mose_unk not in run_dir.name:
                continue
            pts = _load_setup_points(run_dir, stem, splits)
            if pts is None:
                continue
            if validate:
                _validate_setup(run_dir, stem, pts, report)
            pts = pts.assign(method=family, dataset=meta['dataset'],
                             backbone=meta['backbone'], fold=meta['fold'])
            recs.append(pts)

    # --- post-hoc (rbrics): 4 methods share one dir ---
    for run_dir in _iter_run_dirs_multi(posthoc_root, 'baselines'):
        meta = _setup_posthoc(run_dir)
        if not _keep(meta) or meta['variant'] != 'rbrics':
            continue
        for method in POSTHOC_METHODS:
            pts = _load_setup_points(run_dir, method, splits)
            if pts is None:
                continue  # e.g. GNNExplainer absent on OGB (by design)
            if validate:
                _validate_setup(run_dir, method, pts, report)
            pts = pts.assign(method=method, dataset=meta['dataset'],
                             backbone=meta['backbone'], fold=meta['fold'])
            recs.append(pts)

    df = (pd.concat(recs, ignore_index=True) if recs
          else pd.DataFrame(columns=['method', 'dataset', 'backbone', 'fold',
                                     'motif_id', 'motif_smarts', 'split',
                                     'score', 'impact', 'support']))
    return df, report


def load_agnostic_points(posthoc_root: Path, *, tier: str, pooling: str,
                         datasets: set | None) -> pd.DataFrame:
    """Vanilla-model agnostic per-motif impact on the ``rbrics_filter`` vocab
    (matches MoSE), keyed by motif_smarts for the cross-vocab join (Plot 1 grey).

    Reduces ``impact_cache_agnostic_{split}.json`` (mean over graphs) and maps
    motif_id → motif_smarts via the co-located importance CSV (same vocab). This
    equals the co-located post-hoc ``{method}_impact`` CSV exactly (verified), but
    we build it from the cache as the plan specifies.

    Columns: dataset, backbone, fold, split, motif_smarts, agn_impact.
    """
    splits = _SPLITS_ALL if pooling == 'alltest' else ('test',)
    want_tier = _canon_tier(tier)
    recs = []
    for run_dir in _iter_run_dirs_multi(posthoc_root, 'baselines'):
        meta = _setup_posthoc(run_dir)
        if meta is None or meta['variant'] != 'rbrics_filter':
            continue
        if _canon_tier(meta['tier']) != want_tier:
            continue
        if datasets and meta['dataset'] not in datasets:
            continue
        for split in splits:
            cache = run_dir / f'impact_cache_agnostic_{split}.json'
            if not cache.exists():
                continue
            # motif_id → motif_smarts map from any co-located importance CSV.
            id2smarts = None
            for method in POSTHOC_METHODS:
                imp = run_dir / f'{method}_importance_{split}.csv'
                if imp.exists():
                    try:
                        d = pd.read_csv(imp)
                        id2smarts = dict(zip(d['motif_id'].astype(int),
                                             d['motif_smarts']))
                        break
                    except Exception:
                        pass
            if not id2smarts:
                continue
            try:
                with open(cache, encoding='utf-8') as f:
                    c = json.load(f)
            except Exception:
                continue
            for mid_str, per_graph in c.items():
                if not per_graph:
                    continue
                mid = int(mid_str)
                smarts = id2smarts.get(mid)
                if smarts is None:
                    continue
                agn = float(np.mean(list(per_graph.values())))
                recs.append(dict(dataset=meta['dataset'],
                                 backbone=meta['backbone'], fold=meta['fold'],
                                 split=split, motif_smarts=smarts,
                                 agn_impact=agn))
    if not recs:
        return pd.DataFrame(columns=['dataset', 'backbone', 'fold', 'split',
                                     'motif_smarts', 'agn_impact'])
    return pd.DataFrame(recs)


def load_mose_lu_points(ablation_root: Path, *, tier: str, pooling: str,
                        datasets: set | None) -> pd.DataFrame:
    """MoSE **Learn-Unknown** per-motif points from the ablation tree.

    The learn-unknown knob is NOT in ``antehoc_v1`` (verified: 0 runs). It lives in
    ``ablated_completely_v1/mose`` as the OFAT cell ``unk-learnable_shared`` — the
    matched partner of the ``antehoc_v1`` baseline (identical run name, same ``hp``
    hash and depth, only ``unk-fixed`` → ``unk-learnable_shared``). We take only the
    *learn-unknown-only* cell (``onehot`` encoder + ``norm-none`` → the other two
    ablation knobs held at baseline), rbrics_filter, real tier, so this isolates the
    single knob. Layout matches ante-hoc (``mose/<variant>/<ds>/fold<K>/<leaf>``), so
    ``_setup_antehoc`` parses it directly.

    Columns: dataset, backbone, fold, motif_id, motif_smarts, split, score, impact,
    support (same schema as the ``mose`` rows from ``load_method_points``).
    """
    splits = _SPLITS_ALL if pooling == 'alltest' else ('test',)
    want_tier = _canon_tier(tier)
    recs: list[pd.DataFrame] = []
    for run_dir in _iter_run_dirs_multi(ablation_root, 'mose'):
        meta = _setup_antehoc(run_dir, 'mose')
        if meta is None or meta['variant'] != 'rbrics_filter':
            continue
        if _canon_tier(meta['tier']) != want_tier:
            continue
        if datasets and meta['dataset'] not in datasets:
            continue
        name = run_dir.name                    # isolate the learn-unknown-only cell
        if ('unk-learnable_shared' not in name or 'onehot' not in name
                or 'norm-none' not in name):
            continue
        pts = _load_setup_points(run_dir, 'mose', splits)
        if pts is None:
            continue
        pts = pts.assign(method='mose_lu', dataset=meta['dataset'],
                         backbone=meta['backbone'], fold=meta['fold'])
        recs.append(pts)
    if not recs:
        return pd.DataFrame(columns=['method', 'dataset', 'backbone', 'fold',
                                     'motif_id', 'motif_smarts', 'split',
                                     'score', 'impact', 'support'])
    return pd.concat(recs, ignore_index=True)


# ── binning helpers ──────────────────────────────────────────────────────────

def _iqr_whisker(vals: np.ndarray, height: float) -> tuple[float, float]:
    """Return (lower_err, upper_err) so [Q1, Q3] maps to a yerr around ``height``."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 0.0, 0.0
    q1, q3 = np.percentile(v, [25, 75])
    return max(0.0, height - q1), max(0.0, q3 - height)


def _bin_means(scores, impacts, edges) -> tuple[np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray]:
    """Per-bin unweighted mean impact + IQR-whisker errs + counts."""
    nbin = len(edges) - 1
    b = _assign_bins(np.asarray(scores, float), edges)
    imp = np.asarray(impacts, float)
    means = np.full(nbin, np.nan)
    lo = np.zeros(nbin)
    hi = np.zeros(nbin)
    cnt = np.zeros(nbin, int)
    for k in range(nbin):
        sel = imp[b == k]
        sel = sel[np.isfinite(sel)]
        cnt[k] = len(sel)
        if len(sel):
            means[k] = float(sel.mean())
            lo[k], hi[k] = _iqr_whisker(sel, means[k])
    return means, lo, hi, cnt


def _minmax(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, float)
    finite = s[np.isfinite(s)]
    if len(finite) == 0:
        return s
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


def _bin_edge_labels(edges) -> list[str]:
    return [f'{edges[i]:.2g}–{edges[i+1]:.2g}' for i in range(len(edges) - 1)]


def _box_series(ax, positions, data, color, width, whisker_col='0.25',
                highlight=False, alpha=None, zorder=None):
    """Draw one coloured box-plot series (whiskers to 1.5·IQR, fliers hidden).

    Line weights are deliberately heavy so edges/medians/whiskers survive PDF
    compression and grayscale printing (ACM guidance). ``highlight`` gives the
    focal method (MoSE) a bold black edge + heavier median. ``alpha`` overrides
    the fill opacity (e.g. translucent so overlaid lines stay visible)."""
    if not data:
        return 0.0
    box_lw = 1.9 if highlight else 0.8
    med_lw = 2.3 if highlight else 1.3
    wh_lw = 1.3 if highlight else 0.9
    wc = 'black' if highlight else whisker_col
    z = zorder if zorder is not None else (7 if highlight else 4)
    bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True,
                    showfliers=False, manage_ticks=False, zorder=z,
                    medianprops=dict(color='black', linewidth=med_lw),
                    whiskerprops=dict(linewidth=wh_lw, color=wc),
                    capprops=dict(linewidth=wh_lw, color=wc),
                    boxprops=dict(linewidth=box_lw, edgecolor='black'))
    import matplotlib.colors as _mcolors
    fa = alpha if alpha is not None else (1.0 if highlight else 0.95)
    for b in bp['boxes']:
        # Translucent FILL but keep the outline crisp/opaque (don't fade edges).
        b.set_facecolor(_mcolors.to_rgba(color, fa))
        b.set_edgecolor('black')
    # Return the tallest whisker cap so callers can set a tight y-limit.
    tops = [w.get_ydata().max() for w in bp['whiskers']]
    return max(tops) if tops else 0.0


def _binned_values(scores, impacts, edges):
    """List (per bin) of finite impact values whose score falls in that bin."""
    b = _assign_bins(np.asarray(scores, float), edges)
    imp = np.asarray(impacts, float)
    out = []
    for k in range(len(edges) - 1):
        v = imp[(b == k) & np.isfinite(imp)]
        out.append(v)
    return out


# ── Plot 1 ───────────────────────────────────────────────────────────────────

def make_plot1(mose_df: pd.DataFrame, lu_df: pd.DataFrame, agn_df: pd.DataFrame,
               *, nbins: int, score_min, score_max, save_dir: Path, tier: str,
               fmt: tuple[str, ...], fig_width=None, fig_height=None) -> Path | None:
    """MOSE importance vs impact — three box series per bin + motif-count strip.

    The original single-figure grid (rows = dataset, columns = backbone), kept
    as-is; the only addition is a **third box series for MoSE + Learn-Unknown**.
    Per equal-width bin of MoSE motif importance, three whiskered box series show
    the impact distribution of those motifs under: MoSE native LOO (purple), MoSE +
    Learn-Unknown native LOO (teal, ``unk-learnable_shared``), and the vanilla model
    (grey, model-agnostic LOO). MoSE-LU and vanilla impact are joined onto MoSE's
    motifs by ``motif_smarts`` (never motif_id). A motif-count strip (unique MoSE
    motifs per bin) sits inverted beneath each cell. MoSE = rbrics_filter."""
    if mose_df.empty:
        print('[plot1] no MoSE points — skipped.')
        return None

    keys = ['dataset', 'backbone', 'fold', 'split', 'motif_smarts']
    # Attach vanilla-agnostic + MoSE-Learn-Unknown impact onto MoSE's motifs
    # (grouped by the join key so a shared SMARTS never multiplies MoSE rows).
    merged = mose_df.copy()
    if agn_df is not None and not agn_df.empty:
        agn = agn_df.groupby(keys, as_index=False)['agn_impact'].mean()
        merged = merged.merge(agn, on=keys, how='left')
    else:
        merged['agn_impact'] = np.nan
    if lu_df is not None and not lu_df.empty:
        lu = (lu_df.groupby(keys, as_index=False)['impact'].mean()
                    .rename(columns={'impact': 'lu_impact'}))
        merged = merged.merge(lu, on=keys, how='left')
    else:
        merged['lu_impact'] = np.nan
    n_lu = int(merged['lu_impact'].notna().sum())
    n_ag = int(merged['agn_impact'].notna().sum())
    if len(merged) - n_ag:
        warnings.warn(f'[plot1] {len(merged) - n_ag}/{len(merged)} MoSE points '
                      f'had no agnostic match (grey box excludes them).')
    print(f'[plot1] {len(merged)} MoSE points | Learn-Unknown match {n_lu} | '
          f'vanilla match {n_ag}.')

    datasets = _ordered(merged['dataset'].unique(), _DATASET_ORDER)
    backbones = _ordered(merged['backbone'].unique(), _BACKBONE_ORDER)

    # One shared MoSE-score range across the whole grid (comparable x-axis).
    s = merged['score'].to_numpy(float)
    s = s[np.isfinite(s)]
    lo = float(score_min) if score_min is not None else (0.0 if (len(s) and s.min() >= 0) else float(s.min()))
    hi = float(score_max) if score_max is not None else (float(s.max()) if len(s) else 1.0)
    edges = _score_bins_equal(lo, hi, nbins)
    nbin = nbins
    # Box width taken from Plot 2 (grouped-box geometry): three boxes per bin,
    # clustered with clear intra-/inter-group whitespace. The GAPS, however, are
    # trimmed from Plot 2's values: a 3-box group is ~half a 6-box group, so
    # Plot 2's gaps (tuned for 6 boxes) leave the panels too wide here. Reduce
    # both slightly — NOT to zero — to restore Plot 2's whitespace proportions.
    ng = 3
    p1_intra, p1_inter = 0.4, 1.3          # vs Plot 2's 0.5 / 2.6
    box_centres, group_centres, xlim, span, pitch = _grouped_positions(
        nbin, ng, p1_intra, p1_inter)
    # Wider than Plot 2's box (0.6) so the 3 boxes read clearly; the intra-group
    # step is 1 + p1_intra = 1.4, so 0.95 still leaves a ~0.45 gap between boxes.
    box_w = 0.95
    inner = pitch - p1_inter               # group extent (for the count bars)
    gc = np.asarray(group_centres)
    # (colour, value-column) per series, in draw order: MoSE, MoSE-LU, vanilla.
    series = [(_MOSE_COLOR, 'impact'), (_MOSE_LU_COLOR, 'lu_impact'),
              (_AGNOSTIC_COLOR, 'agn_impact')]

    nrow, ncol = len(datasets), len(backbones)
    # Match Plot 2's per-panel width so the box width / spacing render at the
    # same proportions (otherwise 0.6-wide boxes in a ~37-unit span vanish).
    per_panel_w = 0.09 * span + 0.7
    fw = fig_width if fig_width else (per_panel_w * ncol + 1.2)
    fh = fig_height if fig_height else (2.35 * nrow + 1.2)
    fig, axes = plt.subplots(nrow, ncol, squeeze=False, figsize=(fw, fh))

    drew_any = False
    for ri, ds in enumerate(datasets):
        for ci, bb in enumerate(backbones):
            ax = axes[ri][ci]
            cell = merged[(merged['dataset'] == ds) & (merged['backbone'] == bb)]

            if cell.empty:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=13, color='0.55')
                if ri == 0:
                    ax.set_title(str(bb), fontsize=20, pad=6)
                if ci == 0:
                    ax.set_ylabel(_ds_label(ds), fontsize=17)
                continue
            drew_any = True

            # Dotted dividers centred in each between-group gap.
            for k in range(1, nbin):
                ax.axvline(k * pitch - p1_inter / 2.0, color='0.6',
                           linewidth=0.7, linestyle=':', zorder=0.5)

            # Three whiskered box series per bin (MoSE / MoSE-LU / vanilla) of the
            # SAME motifs, at Plot-2 box width + grouped spacing.
            ymax = 0.0
            for j, (clr, col) in enumerate(series):
                sub = cell if col == 'impact' else cell.dropna(subset=[col])
                bv = _binned_values(sub['score'], sub[col], edges)
                pos = [box_centres[k][j] for k in range(nbin) if len(bv[k])]
                dat = [v for v in bv if len(v)]
                ymax = max(ymax, _box_series(ax, pos, dat, clr, box_w))
            if ymax <= 0:
                ymax = 1.0

            # Counts (unique MoSE motifs per bin) drawn DOWNWARD below the y=0
            # baseline on the SAME axes — an inverted mini-axis. Counts use their
            # own fixed band, scaled to the fullest bin; integers are annotated.
            b = _assign_bins(cell['score'].to_numpy(float), edges)
            ucnt = np.array([int(cell.loc[b == k, 'motif_id'].nunique())
                             for k in range(nbin)])
            cmax = int(ucnt.max()) if ucnt.max() > 0 else 1
            band = 0.34 * ymax                       # vertical room for counts
            scaled = -band * ucnt / cmax
            ax.axhline(0, color='0.4', linewidth=0.5, zorder=2)
            ax.bar(gc, scaled, inner * 0.9, color='#e8a24c', alpha=0.85,
                   edgecolor='white', linewidth=0.3, zorder=1)
            for xi, cc, sv in zip(gc, ucnt, scaled):
                if cc > 0:
                    ax.text(xi, sv - band * 0.05, str(cc), ha='center',
                            va='top', fontsize=8, color='#8a5a10')

            ax.set_ylim(-band * 1.32, ymax * 1.12)
            ax.set_xlim(*xlim)
            # Keep only non-negative (impact) y ticks; the count band is unitless.
            yt = [t for t in ax.get_yticks() if t >= -1e-9]
            ax.set_yticks(yt)

            ax.set_xticks(group_centres)
            if ri == nrow - 1:   # declutter: bin labels only on the bottom row
                # Plot-2 font size (13); rotated so the 6 range labels don't crowd
                # in these (3-box, narrower) panels.
                ax.set_xticklabels(_bin_edge_labels(edges), rotation=30,
                                   ha='right', rotation_mode='anchor', fontsize=13)
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis='y', labelsize=13, length=3.5)
            ax.tick_params(axis='x', length=4)

            if ri == 0:
                ax.set_title(str(bb), fontsize=20, pad=6)
            if ci == 0:
                ax.set_ylabel(_ds_label(ds), fontsize=17)

    if not drew_any:
        plt.close(fig)
        print('[plot1] no non-empty cells — skipped.')
        return None

    handles = [Patch(facecolor=_MOSE_COLOR, edgecolor='black', label='MOSE impact'),
               Patch(facecolor=_MOSE_LU_COLOR, edgecolor='black',
                     label='MOSE + Learn-Unknown impact'),
               Patch(facecolor=_AGNOSTIC_COLOR, edgecolor='black',
                     label='Vanilla impact'),
               Patch(facecolor='#e8a24c', edgecolor='white', alpha=0.85,
                     label='# unique motifs / bin (inverted, below axis)')]
    # Reserve a top band for the legend + a bottom/left band for the shared
    # axis labels (Plot-2 font sizes throughout).
    fig.tight_layout(rect=(0.03, 0.025, 0.995, 0.93))
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.985), fontsize=17)
    fig.supylabel(r'Impact $= |\Delta\sigma(\mathrm{logit})|$', x=0.006, fontsize=18)
    fig.supxlabel('Motif Importance', y=0.008, fontsize=18)

    out = None
    for ext in fmt:
        p = save_dir / f'importance_vs_impact_mose_{tier}.{ext}'
        fig.savefig(p, bbox_inches='tight')
        print('wrote', p)
        out = out or p
    plt.close(fig)
    return out


# ── Plot 2 ───────────────────────────────────────────────────────────────────

# Grouped-box geometry (Plot 2). Each method occupies one unit "slot"; a bin's
# ``ng`` boxes sit together and consecutive bins are held apart by this many
# empty slots, so the groups read as clearly distinct clusters.
_P2_INTER_GAP = 2.6      # whitespace (in slots) between importance-range groups
_P2_INTRA_GAP = 0.5      # whitespace (in slots) between boxes WITHIN a group
_P2_BOX_W = 0.6          # box width (of a slot) → leaves an intra-group gap


def _grouped_positions(nbin: int, ng: int, intra_gap: float = _P2_INTRA_GAP,
                       inter_gap: float = _P2_INTER_GAP):
    """Slot geometry for grouped boxes.

    Each box centre is spaced ``1 + intra_gap`` apart so there is clear whitespace
    between the methods of a group; consecutive groups are separated by
    ``inter_gap`` further slots. ``intra_gap`` / ``inter_gap`` default to the
    Plot-2 constants (so Plot 2 / Plot 3 are unchanged); Plot 1 passes smaller
    values because a 3-box group needs less between-group whitespace than the
    6-box groups these gaps were tuned for. Returns per-bin lists of box centres,
    the group centres (x-ticks), the x-limits, the total slot span (for auto
    figure width), and the group pitch (for the between-group dividers)."""
    step = 1.0 + intra_gap                      # centre-to-centre within a group
    inner = (ng - 1) * step + 1.0              # first→last box extent (with edges)
    pitch = inner + inter_gap                   # group centre-to-centre
    box_centres = [[k * pitch + j * step + 0.5 for j in range(ng)]
                   for k in range(nbin)]
    group_centres = [k * pitch + inner / 2.0 for k in range(nbin)]
    span = (nbin - 1) * pitch + inner
    xlim = (-0.5, span + 0.5)
    return box_centres, group_centres, xlim, span, pitch


def _quartiles(v):
    """(Q1, median, Q3) of the finite values, or None if empty."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return float(q1), float(med), float(q3)


def _draw_box(ax, xc, width, q1, med, q3, color, *, edge_lw=0.8, dot_size=13,
              zorder=3):
    """One clean box: Q1–Q3 rectangle + median rule + a small dot marking the
    25th and 75th percentile. Deliberately NO whiskers, caps, or fliers — the
    box IS the interquartile range and the dots pin its ends (spec)."""
    from matplotlib.patches import Rectangle
    rect = Rectangle((xc - width / 2.0, q1), width, max(q3 - q1, 0.0),
                     facecolor=color, edgecolor='black', linewidth=edge_lw,
                     joinstyle='miter', zorder=zorder)
    ax.add_patch(rect)
    ax.plot([xc - width / 2.0, xc + width / 2.0], [med, med], color='black',
            linewidth=edge_lw + 0.5, solid_capstyle='butt', zorder=zorder + 1)
    ax.scatter([xc, xc], [q1, q3], s=dot_size, color='black',
               edgecolors='none', zorder=zorder + 2)


def make_plot2(df: pd.DataFrame, *, nbins: int, unk: str, save_dir: Path,
               tier: str, fmt: tuple[str, ...], fig_width=None,
               fig_height=None, panel_groups=None) -> Path | None:
    """All-methods GROUPED BOX plots (wide, ACM-ready).

    Grid: rows = dataset, cols = backbone (a single-backbone group tiles its
    datasets ``ds_grid_ncols`` wide instead). Inside each panel the x-axis is the
    motif-importance range (per-method min-max-normalised into ``nbins`` equal
    bins); at each range the methods sit side by side as boxes, clustered with
    clear whitespace both BETWEEN the boxes of a group and BETWEEN groups. Each
    box is Q1–Q3 + a median rule + small dots at the 25th/75th percentile — no
    whiskers, caps, or fliers. **y auto-scales PER PANEL** so a large-impact
    panel never crushes a small-impact one — the faithfulness signal (impact
    rising across importance bins) stays visible everywhere. Fixed
    ColorBrewer-Dark2 colour per method (MoSE gets a slightly heavier edge as the
    proposed method). ``panel_groups`` = list of
    ``(tag, datasets|None, backbones|None, ds_grid_ncols|None)``. Returns the
    first figure path written."""
    import math
    if df.empty:
        print(f'[plot2:{unk}] no points — skipped.')
        return None

    work = df if unk == 'incl' else df[df['motif_id'] != UNK_ID]
    if work.empty:
        print(f'[plot2:{unk}] empty after UNK filter — skipped.')
        return None

    datasets_all = _ordered(work['dataset'].unique(), _DATASET_ORDER)
    backbones_all = _ordered(work['backbone'].unique(), _BACKBONE_ORDER)
    methods = [m for m in PLOT2_METHODS if m in set(work['method'].unique())]

    edges = np.linspace(0.0, 1.0, nbins + 1)   # per-method normalized [0,1] bins
    nbin = nbins
    ng = len(methods)
    box_centres, group_centres, xlim, span, pitch = _grouped_positions(nbin, ng)
    box_w = _P2_BOX_W
    tag = 'inclunk' if unk == 'incl' else 'exclunk'

    entries = panel_groups if panel_groups else [('', None, None, None)]
    first_out = None
    for entry in entries:
        gtag, gds_sel, gbb_sel = entry[0], entry[1], entry[2]
        ds_ncols = entry[3] if len(entry) > 3 else None
        gds = [d for d in datasets_all if gds_sel is None or d in set(gds_sel)]
        gbbs = [b for b in backbones_all if gbb_sel is None or b in set(gbb_sel)]
        if not gds or not gbbs:
            continue

        # Panel map: (row, col) -> (dataset, backbone). Tiled mode packs a single
        # backbone's datasets into a ds_ncols-wide grid (GIN → 2×4).
        tiled = bool(ds_ncols) and len(gbbs) == 1
        panel_map = {}
        if tiled:
            bb0 = gbbs[0]
            ncol = min(ds_ncols, len(gds))
            nrow = math.ceil(len(gds) / ncol)
            for idx, ds in enumerate(gds):
                panel_map[divmod(idx, ncol)] = (ds, bb0)
        else:
            nrow, ncol = len(gds), len(gbbs)
            for ri, ds in enumerate(gds):
                for ci, bb in enumerate(gbbs):
                    panel_map[(ri, ci)] = (ds, bb)

        # Pass 1 (per panel): each method → per-bin (Q1, median, Q3) of impact,
        # keyed by method; also the panel's max Q3 for the shared per-panel ylim.
        stats_of, ymax_of = {}, {}
        for (r, c), (ds, bb) in panel_map.items():
            cell = work[(work['dataset'] == ds) & (work['backbone'] == bb)]
            per_method, ymax = {}, 0.0
            for m in methods:
                cm = cell[cell['method'] == m]
                if cm.empty:
                    continue
                sn = _minmax(cm['score'].to_numpy(float))
                bv = _binned_values(sn, cm['impact'].to_numpy(float), edges)
                qs = [_quartiles(bv[k]) for k in range(nbin)]
                per_method[m] = qs
                for q in qs:
                    if q is not None:
                        ymax = max(ymax, q[2])
            stats_of[(r, c)] = per_method
            ymax_of[(r, c)] = ymax

        if not any(stats_of.values()):
            continue

        # Broader is better: width follows the total slot span; height is compact.
        per_panel_w = 0.09 * span + 0.7
        fw = fig_width if fig_width else (per_panel_w * ncol + 1.2)
        fh = fig_height if fig_height else (1.5 * nrow + 1.4)
        fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=False,
                                 squeeze=False, figsize=(fw, fh))
        for r in range(nrow):
            for c in range(ncol):
                ax = axes[r][c]
                if (r, c) not in panel_map:
                    ax.set_visible(False)
                    continue
                ds, bb = panel_map[(r, c)]
                per_method = stats_of.get((r, c), {})
                ymax = ymax_of.get((r, c), 0.0) or 1.0
                ax.grid(axis='y', linewidth=0.4, alpha=0.3, zorder=0)
                # Dotted divider centred in each between-group gap so the
                # importance ranges read as clearly separate regions.
                for k in range(1, nbin):
                    ax.axvline(k * pitch - _P2_INTER_GAP / 2.0, color='0.45',
                               linewidth=1.3, linestyle=(0, (2, 2)), zorder=1)
                if not per_method:
                    ax.text(0.5, 0.5, 'n/a', ha='center', va='center',
                            transform=ax.transAxes, fontsize=8, color='0.6')
                else:
                    for j, m in enumerate(methods):
                        qs = per_method.get(m)
                        if qs is None:
                            continue
                        edge_lw = 1.4 if m == PLOT2_HIGHLIGHT else 0.8
                        for k in range(nbin):
                            if qs[k] is None:
                                continue
                            q1, med, q3 = qs[k]
                            _draw_box(ax, box_centres[k][j], box_w, q1, med, q3,
                                      METHOD_COLOR[m], edge_lw=edge_lw)
                # Per-panel y-scale (no crushing) with a few compact ticks.
                ax.set_ylim(0, ymax * 1.15)
                ax.set_xlim(*xlim)
                ax.set_xticks(group_centres)
                ax.yaxis.set_major_locator(MaxNLocator(3))
                is_bottom = (r + 1, c) not in panel_map
                if is_bottom:
                    ax.set_xticklabels(_bin_edge_labels(edges), rotation=0,
                                       ha='center', fontsize=13)
                else:
                    ax.set_xticklabels([])
                ax.tick_params(axis='y', labelsize=13, length=3.5)
                ax.tick_params(axis='x', length=4)
                if tiled:                          # each panel is a dataset
                    ax.set_title(_ds_label(ds).replace('\n', ''),
                                 fontsize=18, pad=5)
                else:
                    if r == 0:                     # column header = backbone
                        ax.set_title(str(bb), fontsize=20, pad=6)
                    if c == 0:                     # row label = dataset
                        ax.set_ylabel(_ds_label(ds), fontsize=17)

        # One colour key per method; MoSE last with its heavier edge.
        def _leg_label(m):
            return 'MOSE-GNN (ante-hoc)' if m == PLOT2_HIGHLIGHT else _title(m)
        handles = [Patch(facecolor=METHOD_COLOR[m], edgecolor='black',
                         linewidth=(1.4 if m == PLOT2_HIGHLIGHT else 0.8),
                         label=_leg_label(m)) for m in methods]
        top = 0.86 if tiled else 0.94
        fig.tight_layout(rect=(0.055, 0.06, 0.995, top))
        # Legend sits below the (tiled) backbone suptitle so they never overlap.
        legend_y = 0.925 if tiled else 0.997
        fig.legend(handles=handles, loc='upper center', ncol=len(methods),
                   frameon=False, bbox_to_anchor=(0.5, legend_y),
                   title='Explainer method', fontsize=17, title_fontsize=18,
                   handlelength=1.6, columnspacing=1.6)
        if tiled:
            fig.suptitle(f'GNN backbone: {gbbs[0]}', y=0.995, fontsize=20,
                         fontweight='bold')
        fig.supylabel('Impact  (Δ prediction if motif removed)\n'
                      r'$|\Delta\sigma(\mathrm{logit})|$', x=0.008, fontsize=18)
        fig.supxlabel('Motif Importance  (normalized 0–1)', y=0.02, fontsize=18)

        stem = (f'importance_vs_impact_allmethods_{gtag}_{tag}_{tier}' if gtag
                else f'importance_vs_impact_allmethods_{tag}_{tier}')
        for ext in fmt:
            p = save_dir / f'{stem}.{ext}'
            fig.savefig(p, bbox_inches='tight')
            print('wrote', p)
            first_out = first_out or p
        plt.close(fig)

    return first_out


# ── Plot 3 — MoSE boxes + baseline median-lines ──────────────────────────────

# Paul Tol "muted" qualitative palette: subdued, colour-blind-safe, print-safe —
# deliberately NOT the saturated Plot-2 set (ACM-appropriate, not garish). The
# MoSE box is a NEUTRAL GREY backdrop so every coloured line contrasts against it
# (a coloured box would wash out any line sharing its hue, e.g. green-on-teal).
_MOSE_BOX_FILL = '#b3b3b3'        # neutral grey (MoSE distribution, backdrop)
_LINE_COLOR = {
    'gsat': '#cc6677',            # rose
    'gnnexplainer': '#332288',    # indigo
    'pgexplainer': '#d55e00',     # vermillion (was amber #ddaa33 — low contrast)
    'motif_occlusion': '#117733', # green
    'mage_v2': '#aa4499',         # purple
}
_LINE_MARKER = {
    'gsat': 'o', 'gnnexplainer': 's', 'pgexplainer': '^',
    'motif_occlusion': 'D', 'mage_v2': 'v',
}
_LINE_ALPHA = 0.7          # opacity of the baseline median-line segments


def _draw_mose_box(ax, xc, width, q1, med, q3, *, alpha=0.5, edge_lw=1.4,
                   zorder=3):
    """Translucent MoSE box (Q1–Q3 + median + quartile dots), no whiskers — the
    baseline median-lines are drawn ON TOP so they stay visible through it."""
    from matplotlib.patches import Rectangle
    import matplotlib.colors as _mc
    rect = Rectangle((xc - width / 2.0, q1), width, max(q3 - q1, 0.0),
                     facecolor=_mc.to_rgba(_MOSE_BOX_FILL, alpha),
                     edgecolor='black', linewidth=edge_lw, joinstyle='miter',
                     zorder=zorder)
    ax.add_patch(rect)
    ax.plot([xc - width / 2.0, xc + width / 2.0], [med, med], color='black',
            linewidth=edge_lw + 0.3, solid_capstyle='butt', zorder=zorder + 1)
    ax.scatter([xc, xc], [q1, q3], s=12, color='black', edgecolors='none',
               zorder=zorder + 2)


def make_plot3(df: pd.DataFrame, *, nbins: int, unk: str, save_dir: Path,
               tier: str, fmt: tuple[str, ...], fig_width=None,
               fig_height=None, panel_groups=None) -> Path | None:
    """MoSE = box per importance range; the 5 baselines = median-lines THROUGH the
    boxes (one marker per range = that explainer's median impact in the range).

    Same grid as Plot 2 (rows = dataset, cols = backbone; single-backbone groups
    tile their datasets ``ds_grid_ncols`` wide). Wide panels, large text, dotted
    vertical separators between ranges, per-panel y-scale. Muted ACM palette.
    Returns the first figure path written. Does NOT touch make_plot2 / its
    helpers."""
    import math
    if df.empty:
        print(f'[plot3:{unk}] no points — skipped.')
        return None
    work = df if unk == 'incl' else df[df['motif_id'] != UNK_ID]
    if work.empty:
        print(f'[plot3:{unk}] empty after UNK filter — skipped.')
        return None

    datasets_all = _ordered(work['dataset'].unique(), _DATASET_ORDER)
    backbones_all = _ordered(work['backbone'].unique(), _BACKBONE_ORDER)
    methods = [m for m in PLOT2_METHODS if m in set(work['method'].unique())]
    base_methods = [m for m in methods if m != PLOT2_HIGHLIGHT]

    edges = np.linspace(0.0, 1.0, nbins + 1)
    nbin = nbins
    x = np.arange(nbin)
    box_w = 0.3          # narrow MoSE box so the median-lines aren't swallowed
    tag = 'inclunk' if unk == 'incl' else 'exclunk'

    entries = panel_groups if panel_groups else [('', None, None, None)]
    first_out = None
    for entry in entries:
        gtag, gds_sel, gbb_sel = entry[0], entry[1], entry[2]
        ds_ncols = entry[3] if len(entry) > 3 else None
        gds = [d for d in datasets_all if gds_sel is None or d in set(gds_sel)]
        gbbs = [b for b in backbones_all if gbb_sel is None or b in set(gbb_sel)]
        if not gds or not gbbs:
            continue

        tiled = bool(ds_ncols) and len(gbbs) == 1
        panel_map = {}
        if tiled:
            bb0 = gbbs[0]
            ncol = min(ds_ncols, len(gds))
            nrow = math.ceil(len(gds) / ncol)
            for idx, ds in enumerate(gds):
                panel_map[divmod(idx, ncol)] = (ds, bb0)
        else:
            nrow, ncol = len(gds), len(gbbs)
            for ri, ds in enumerate(gds):
                for ci, bb in enumerate(gbbs):
                    panel_map[(ri, ci)] = (ds, bb)

        # Pass 1: per panel → MoSE per-range (Q1,med,Q3) + each baseline's per-
        # range median line; also the panel ymax.
        data_of, ymax_of = {}, {}
        for (r, c), (ds, bb) in panel_map.items():
            cell = work[(work['dataset'] == ds) & (work['backbone'] == bb)]
            ymax = 0.0
            mose_q = [None] * nbin
            mc = cell[cell['method'] == PLOT2_HIGHLIGHT]
            if not mc.empty:
                sn = _minmax(mc['score'].to_numpy(float))
                bv = _binned_values(sn, mc['impact'].to_numpy(float), edges)
                for k in range(nbin):
                    q = _quartiles(bv[k])
                    mose_q[k] = q
                    if q is not None:
                        ymax = max(ymax, q[2])
            lines = {}
            for m in base_methods:
                cm = cell[cell['method'] == m]
                if cm.empty:
                    continue
                sn = _minmax(cm['score'].to_numpy(float))
                bv = _binned_values(sn, cm['impact'].to_numpy(float), edges)
                med = np.array([np.median(bv[k]) if len(bv[k]) else np.nan
                                for k in range(nbin)])
                lines[m] = med
                if np.isfinite(med).any():
                    ymax = max(ymax, float(np.nanmax(med)))
            data_of[(r, c)] = {'mose': mose_q, 'lines': lines}
            ymax_of[(r, c)] = ymax
        if not any(data_of.values()):
            continue

        per_panel_w = 0.85 * nbin + 1.0
        fw = fig_width if fig_width else (per_panel_w * ncol + 1.3)
        # Taller panels give the median-lines vertical room to separate.
        fh = fig_height if fig_height else (2.35 * nrow + 1.4)
        fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=False,
                                 squeeze=False, figsize=(fw, fh))
        for r in range(nrow):
            for c in range(ncol):
                ax = axes[r][c]
                if (r, c) not in panel_map:
                    ax.set_visible(False)
                    continue
                ds, bb = panel_map[(r, c)]
                md = data_of.get((r, c), {})
                ymax = ymax_of.get((r, c), 0.0) or 1.0
                ax.grid(axis='y', linewidth=0.4, alpha=0.3, zorder=0)
                # Dotted vertical separators between the importance ranges.
                for k in range(1, nbin):
                    ax.axvline(k - 0.5, color='0.45', linewidth=1.3,
                               linestyle=(0, (2, 2)), zorder=1)
                mose_q = md.get('mose', [None] * nbin)
                lines = md.get('lines', {})
                if not any(q is not None for q in mose_q) and not lines:
                    ax.text(0.5, 0.5, 'n/a', ha='center', va='center',
                            transform=ax.transAxes, fontsize=13, color='0.6')
                else:
                    # MoSE boxes (drawn first, translucent).
                    for k in range(nbin):
                        if mose_q[k] is not None:
                            q1, med, q3 = mose_q[k]
                            _draw_mose_box(ax, k, box_w, q1, med, q3)
                    # Baseline median-lines THROUGH the boxes (on top). The line
                    # segments are translucent so overlapping trends blend rather
                    # than hide each other; the markers stay opaque for legibility.
                    for m in base_methods:
                        med = lines.get(m)
                        if med is None:
                            continue
                        ax.plot(x, med, color=_LINE_COLOR[m], linewidth=2.2,
                                alpha=_LINE_ALPHA, zorder=9, clip_on=True)
                        ax.plot(x, med, color=_LINE_COLOR[m], linestyle='none',
                                marker=_LINE_MARKER[m], markersize=8,
                                markeredgecolor='white', markeredgewidth=0.8,
                                zorder=10, clip_on=True)
                ax.set_ylim(0, ymax * 1.15)
                ax.set_xlim(-0.6, nbin - 0.4)
                ax.set_xticks(x)
                ax.yaxis.set_major_locator(MaxNLocator(3))
                is_bottom = (r + 1, c) not in panel_map
                if is_bottom:
                    ax.set_xticklabels(_bin_edge_labels(edges), rotation=60,
                                       ha='right', rotation_mode='anchor',
                                       fontsize=13)
                else:
                    ax.set_xticklabels([])
                ax.tick_params(axis='y', labelsize=13, length=3.5)
                ax.tick_params(axis='x', length=4)
                if tiled:
                    ax.set_title(_ds_label(ds).replace('\n', ''),
                                 fontsize=18, pad=5)
                else:
                    if r == 0:
                        ax.set_title(str(bb), fontsize=20, pad=6)
                    if c == 0:
                        ax.set_ylabel(_ds_label(ds), fontsize=17)

        # Legend: baselines as line keys, MoSE as a box key (last).
        handles = [Line2D([0], [0], color=_LINE_COLOR[m], marker=_LINE_MARKER[m],
                          markersize=9, markeredgecolor='white',
                          markeredgewidth=0.8, linewidth=2.4, label=_title(m))
                   for m in base_methods]
        handles.append(Patch(facecolor=_MOSE_BOX_FILL, alpha=0.6,
                             edgecolor='black', linewidth=1.4,
                             label='MOSE-GNN (ante-hoc)'))
        top = 0.86 if tiled else 0.94
        # Bring the y-label INWARD (was clipping the page): larger left margin +
        # supylabel x nudged in.
        fig.tight_layout(rect=(0.065, 0.06, 0.995, top))
        legend_y = 0.925 if tiled else 0.997
        fig.legend(handles=handles, loc='upper center', ncol=len(methods),
                   frameon=False, bbox_to_anchor=(0.5, legend_y),
                   title='Explainer method', fontsize=17, title_fontsize=18,
                   handlelength=1.9, columnspacing=1.6)
        if tiled:
            fig.suptitle(f'GNN backbone: {gbbs[0]}', y=0.995, fontsize=20,
                         fontweight='bold')
        fig.supylabel('Impact  (Δ prediction if motif removed)\n'
                      r'$|\Delta\sigma(\mathrm{logit})|$', x=0.024, fontsize=18)
        fig.supxlabel('Motif Importance  (normalized 0–1)', y=0.02, fontsize=18)

        stem = (f'importance_vs_impact_moseline_{gtag}_{tag}_{tier}' if gtag
                else f'importance_vs_impact_moseline_{tag}_{tier}')
        for ext in fmt:
            p = save_dir / f'{stem}.{ext}'
            fig.savefig(p, bbox_inches='tight')
            print('wrote', p)
            first_out = first_out or p
        plt.close(fig)
    return first_out


# ── validation report ────────────────────────────────────────────────────────

def _emit_validation(report: list[dict], save_dir: Path) -> None:
    if not report:
        print('[validate] no setups checked.')
        return
    rep = pd.DataFrame(report)
    checkable = rep[np.isfinite(rep['diff'])]
    n_ok = int(rep['ok'].sum())
    print(f'[validate] {n_ok}/{len(rep)} setups reproduce pearson_u_inclunk '
          f'(tol 1e-6); {len(rep) - len(checkable)} were NaN-vs-NaN.')
    if len(checkable):
        worst = checkable.sort_values('diff', ascending=False).iloc[0]
        print(f'[validate] worst finite diff = {worst["diff"]:.2e} '
              f'({worst["stem"]} @ {worst["run_dir"]})')
    fails = rep[~rep['ok']]
    if len(fails):
        print(f'[validate] {len(fails)} MISMATCHES (showing up to 10):')
        for _, r in fails.head(10).iterrows():
            print(f'    {r["stem"]:14s} exp={r["expected"]:.6f} '
                  f'got={r["got"]:.6f}  {r["run_dir"]}')
    out = save_dir / 'validation_pearson_u_inclunk.csv'
    rep.to_csv(out, index=False)
    print('wrote', out)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _resolve_roots(args) -> tuple[list, list, list]:
    if args.base:
        base = Path(args.base)
        ah = args.antehoc_root or [base / 'antehoc_v1']
        ph = args.posthoc_root or [base / 'posthoc_v1']
        lu = args.mose_lu_root or [base / 'ablated_completely_v1']
    else:
        if not (args.antehoc_root and args.posthoc_root):
            raise SystemExit('Provide --base, or both --antehoc_root and '
                             '--posthoc_root.')
        ah, ph = args.antehoc_root, args.posthoc_root
        lu = args.mose_lu_root or None
    ah, ph, lu = _as_roots(ah), _as_roots(ph), _as_roots(lu)
    for p in ah + ph:
        if not p.exists():
            warnings.warn(f'root does not exist: {p}')
    for p in lu:
        if not p.exists():
            warnings.warn(f'MoSE Learn-Unknown root does not exist: {p} '
                          '(Plot 1 will omit that series)')
    return ah, ph, lu


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', default=None,
                    help='Base dir containing antehoc_v1/ and posthoc_v1/ '
                         '(convenience; overridden by explicit roots).')
    ap.add_argument('--antehoc_root', nargs='*', default=None,
                    help='MoSE-native tree(s). Accepts several roots, e.g. '
                         'eval_real eval_gath1 (non-GAT + GAT from the bundle).')
    ap.add_argument('--posthoc_root', nargs='*', default=None,
                    help='post-hoc / vanilla tree(s) (e.g. posthoc_v1).')
    ap.add_argument('--mose_unk', default=None,
                    help="keep only MoSE-native leaves whose name contains this "
                         "token (e.g. 'unk-fixed') — needed when native and "
                         "learn-unknown leaves share one tree (bundle eval_real/mose).")
    ap.add_argument('--mose_lu_root', nargs='*', default=None,
                    help='ablated_completely_v1 tree (source of the MoSE + '
                         'Learn-Unknown series for Plot 1). Defaults to '
                         '<base>/ablated_completely_v1 when --base is given.')
    ap.add_argument('--save_dir', default=None,
                    help='Output dir (default: analysis/plots_importance_impact).')
    ap.add_argument('--nbins', type=int, default=6,
                    help='Default importance-bin count for both plots.')
    ap.add_argument('--nbins_plot1', type=int, default=None,
                    help='Override bin count for Plot 1 (default: --nbins).')
    ap.add_argument('--nbins_plot2', type=int, default=None,
                    help='Override bin count for Plot 2 (default: --nbins).')
    ap.add_argument('--score_min', type=float, default=None,
                    help='Plot-1 MOSE-importance axis min (default 0).')
    ap.add_argument('--score_max', type=float, default=None,
                    help='Plot-1 MOSE-importance axis max (default data max; '
                         'pass 1 to force [0,1]).')
    ap.add_argument('--dataset', nargs='*', default=None,
                    help='Restrict to these dataset(s).')
    ap.add_argument('--exclude_dataset', nargs='*', default=None,
                    help='Drop these dataset(s), e.g. --exclude_dataset mutag '
                         'ogbg-molbace ogbg-molhiv.')
    ap.add_argument('--plot2_split', action='store_true',
                    help='Split Plot 2 into the requested figure groups '
                         '(PLOT2_GROUPS): GIN alone (all datasets); "good" and '
                         '"medium" dataset groups over GCN/SAGE/GAT/PNA.')
    ap.add_argument('--fig_width', type=float, default=None,
                    help='Figure width in inches (default: auto per column). Set '
                         'to your Overleaf \\textwidth (e.g. 7.0 for a two-column '
                         'ACM figure*) so \\includegraphics[width=\\textwidth] '
                         'scales ~1:1 and fonts stay legible.')
    ap.add_argument('--fig_height', type=float, default=None,
                    help='Figure height in inches (default: auto per row). Cap at '
                         'the page text height (~9) to fit on one page.')
    ap.add_argument('--pooling', default='alltest',
                    choices=['alltest', 'testonly'],
                    help='Pool train+valid+test (default) or test only.')
    ap.add_argument('--tier', default='real',
                    help="Tier filter: 'real' (default) or e.g. 'gt_dnf_k2r1'. "
                         'real and GT are never pooled in one figure.')
    ap.add_argument('--plots', nargs='*', default=['1', '2'],
                    choices=['1', '2', '3'],
                    help='Which deliverables to render (default: 1 2). 3 = MoSE '
                         'boxes + baseline median-lines (shares --nbins_plot2 / '
                         '--plot2_split / --plot2_unk).')
    ap.add_argument('--formats', nargs='*', default=['pdf', 'png'],
                    help='Figure formats (default: pdf png — pdf is vector for ACM).')
    ap.add_argument('--validate', action='store_true',
                    help='Reproduce pearson_u_inclunk from loaded points and report.')
    ap.add_argument('--plot2_unk', default='incl', choices=['incl', 'excl', 'both'],
                    help="Which Plot-2 UNK variant(s) to render (default: incl "
                         'only — exclunk is identical when no UNK rows exist).')
    args = ap.parse_args()

    _apply_acm_style()
    antehoc_root, posthoc_root, mose_lu_root = _resolve_roots(args)
    save_dir = (Path(args.save_dir) if args.save_dir
                else _REPO / 'analysis' / 'plots_importance_impact')
    save_dir.mkdir(parents=True, exist_ok=True)
    datasets = set(args.dataset) if args.dataset else None
    exclude = set(args.exclude_dataset) if args.exclude_dataset else set()
    fmt = tuple(args.formats)

    df, report = load_method_points(
        antehoc_root, posthoc_root, tier=args.tier, pooling=args.pooling,
        datasets=datasets, validate=args.validate, mose_unk=args.mose_unk)
    if exclude:
        df = df[~df['dataset'].isin(exclude)].copy()
    if df.empty:
        print('No per-motif points found. Check roots / tier / dataset filters.')
        raise SystemExit(1)

    print(f'Loaded {len(df)} pooled (motif,split,fold) points across '
          f'{df["method"].nunique()} methods, {df["dataset"].nunique()} datasets, '
          f'{df["backbone"].nunique()} backbones (tier={args.tier}).')
    if args.validate:
        _emit_validation(report, save_dir)

    if '1' in args.plots:
        mose_df = df[df['method'] == 'mose'].copy()
        if mose_lu_root:
            lu_df = load_mose_lu_points(mose_lu_root, tier=args.tier,
                                        pooling=args.pooling, datasets=datasets)
            if exclude and not lu_df.empty:
                lu_df = lu_df[~lu_df['dataset'].isin(exclude)].copy()
        else:
            lu_df = pd.DataFrame(columns=['method', 'dataset', 'backbone', 'fold',
                                          'motif_id', 'motif_smarts', 'split',
                                          'score', 'impact', 'support'])
        agn_df = load_agnostic_points(posthoc_root, tier=args.tier,
                                      pooling=args.pooling, datasets=datasets)
        if exclude and not agn_df.empty:
            agn_df = agn_df[~agn_df['dataset'].isin(exclude)].copy()
        print(f'[plot1] {len(mose_df)} MoSE, {len(lu_df)} MoSE-LU, '
              f'{len(agn_df)} vanilla-agnostic rows.')
        make_plot1(mose_df, lu_df, agn_df, nbins=(args.nbins_plot1 or args.nbins),
                   score_min=args.score_min, score_max=args.score_max,
                   save_dir=save_dir, tier=args.tier, fmt=fmt,
                   fig_width=args.fig_width, fig_height=args.fig_height)

    if '2' in args.plots:
        pgroups = PLOT2_GROUPS if args.plot2_split else None
        unk_modes = {'incl': ('incl',), 'excl': ('excl',),
                     'both': ('incl', 'excl')}[args.plot2_unk]
        for unk in unk_modes:
            make_plot2(df, nbins=(args.nbins_plot2 or args.nbins), unk=unk,
                       save_dir=save_dir, tier=args.tier, fmt=fmt,
                       fig_width=args.fig_width, fig_height=args.fig_height,
                       panel_groups=pgroups)

    if '3' in args.plots:
        pgroups = PLOT2_GROUPS if args.plot2_split else None
        unk_modes = {'incl': ('incl',), 'excl': ('excl',),
                     'both': ('incl', 'excl')}[args.plot2_unk]
        for unk in unk_modes:
            make_plot3(df, nbins=(args.nbins_plot2 or args.nbins), unk=unk,
                       save_dir=save_dir, tier=args.tier, fmt=fmt,
                       fig_width=args.fig_width, fig_height=args.fig_height,
                       panel_groups=pgroups)


if __name__ == '__main__':
    main()
