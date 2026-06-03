#!/usr/bin/env python3
"""
analyse_phase3.py
=================
Phase 3 analysis using vocab artifacts produced by phase1.
Does NOT re-run fragmentation — loads everything from the vocab output.

Usage
-----
    # Minimal — reads data_root and vocab_root from experiment_config.sh
    source ../experiment_config.sh
    python3 analyse_phase3.py \\
        --data_root "$DATA_ROOT" \\
        --vocab_root "$VOCAB_ROOT" \\
        --datasets Mutagenicity BBBP Benzene \\
        --variants rbrics all_fallback_bpe \\
        --fold 0

    # With explicit output path
    python3 analyse_phase3.py \\
        --data_root /nfs/hpc/.../FOLDS \\
        --vocab_root /nfs/hpc/.../vocab_output \\
        --datasets Mutagenicity Benzene \\
        --variants rbrics all_fallback_bpe \\
        --out_json ./results/phase3_analysis.json

Outputs
-------
    {out_json}   — JSON consumed by the Phase 3 widget
"""

import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Make the script importable from its own directory ─────────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

import motif_label_pipeline as pipe

# ─────────────────────────────────────────────────────────────────────────────
# Dataset label column map  (matches generate_vocab_rules.py)
# ─────────────────────────────────────────────────────────────────────────────
DATASET_COLUMN: Dict[str, str] = {
    'Mutagenicity':      'Mutagenicity',
    'BBBP':              'BBBP',
    'hERG':              'hERG',
    'Lipophilicity':     'Lipophilicity',
    'esol':              'esol',
    'tox21':             'tox21',
    'Benzene':           'label',
    'Alkane_Carbonyl':   'label',
    'Fluoride_Carbonyl': 'label',
    'freesolv':          'freesolv',
}


# ─────────────────────────────────────────────────────────────────────────────
# Load vocab artifacts produced by phase1
# ─────────────────────────────────────────────────────────────────────────────

def _load_pickle(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_vocab_artifacts(vocab_root: str, dataset: str, variant: str, fold: int
                          ) -> Tuple[dict, dict, dict, pd.DataFrame]:
    """Load all vocab artifacts for one dataset × variant.

    Returns
    -------
    lookup_train  {smiles: {node_idx: (smarts, motif_id)}}  training split
    lookup_valid  {smiles: {node_idx: (smarts, motif_id)}}  valid split
    cols_df       DataFrame from matrix_columns.csv
    smiles_df     DataFrame from smiles_labels.csv
    """
    vdir = Path(vocab_root) / dataset / variant
    base = str(vdir / f'{dataset}_{variant}')

    if not vdir.exists():
        raise FileNotFoundError(
            f"Vocab directory not found: {vdir}\n"
            f"Run phase1 first:  bash run_experiments.sh phase1"
        )

    lookup_train = _load_pickle(Path(f'{base}_graph_lookup.pickle'))

    valid_path = Path(f'{base}_valid_graph_lookup.pickle')
    lookup_valid = _load_pickle(valid_path) if valid_path.exists() else {}

    cols_df   = pd.read_csv(vdir / 'matrix_columns.csv')
    smiles_df = pd.read_csv(vdir / 'smiles_labels.csv')

    return lookup_train, lookup_valid, cols_df, smiles_df


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruct tv_frags + raw_stats from vocab artifacts
# ─────────────────────────────────────────────────────────────────────────────

def tv_frags_from_lookup(smiles_tv: List[str],
                          lookup_train: dict,
                          lookup_valid: dict) -> List[List[str]]:
    """Reconstruct per-molecule fragment lists from the graph lookup dicts.

    For each molecule returns the unique SMARTS strings assigned to its atoms.
    This matches what fragmentation produced — no re-running needed.

    Unknown (-1) motif_id entries are included as their smarts string so that
    the structural SNR computation (Jaccard on fragment sets) is correct.
    """
    combined = {**lookup_train, **lookup_valid}
    tv_frags = []
    for smi in smiles_tv:
        node_map = combined.get(smi, {})
        frags = list({smarts for smarts, _mid in node_map.values()})
        tv_frags.append(frags)
    return tv_frags


def raw_stats_from_cols(cols_df: pd.DataFrame) -> dict:
    """Build raw_stats dict from matrix_columns.csv.

    matrix_columns.csv has these columns (from generate_vocab_rules.py):
        motif_identity, n_mols, wt_count_0, wt_count_1

    Returns {smarts: {'n': int, 'wt0': float, 'wt1': float}}
    """
    raw = {}
    for _, row in cols_df.iterrows():
        s = str(row['motif_identity'])
        raw[s] = {
            'n':   int(row.get('n_mols',   row.get('freq_count', 0))),
            'wt0': float(row.get('wt_count_0', 0.0)),
            'wt1': float(row.get('wt_count_1', 0.0)),
        }
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Distribution statistics
# ─────────────────────────────────────────────────────────────────────────────

def _frag_distributions(tv_frags: List[List[str]],
                         raw: dict,
                         N_tv: int) -> dict:
    stats = []
    for s, d in raw.items():
        sup = d['n'] / N_tv * 100
        tot = d['wt0'] + d['wt1']
        c1  = d['wt1'] / tot * 100 if tot else 50.0
        stats.append({'s': s, 'sup': round(sup, 2),
                      'n_atoms': pipe.atom_count(s), 'c1': round(c1, 1)})

    lc = Counter(x['n_atoms'] for x in stats)
    length_dist = [{'atoms': k, 'n_motifs': lc[k]} for k in sorted(lc)]

    bins = [(0,1,'<1%'),(1,5,'1-5%'),(5,10,'5-10%'),(10,20,'10-20%'),(20,101,'>20%')]
    supdist = [{'label': lbl,
                'n': sum(1 for x in stats if lo <= x['sup'] < hi)}
               for lo, hi, lbl in bins]

    fpm = [len(f) for f in tv_frags]
    fpm_dist = [{'frags': b if b < 15 else '15+',
                 'n': sum(1 for x in fpm if (x == b if b < 15 else x >= 15))}
                for b in range(1, 16)]

    return {
        'length_dist':  length_dist,
        'support_dist': supdist,
        'fpm_dist':     fpm_dist,
        'n_unique':     len(stats),
        'mean_fpm':     round(float(np.mean(fpm)), 2) if fpm else 0.0,
        'median_fpm':   int(np.median(fpm)) if fpm else 0,
        'n_single':     int(sum(1 for x in fpm if x == 1)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-motif SNR table
# ─────────────────────────────────────────────────────────────────────────────

def _motif_snr_table(selected, presence, raw, tv_labels, synth_labels, N_tv,
                      br_motifs):
    rows = []
    for c in selected[:15]:
        s = c['s']
        if s not in presence:
            continue
        vec = presence[s].astype(bool)
        n   = int(vec.sum())
        if n == 0:
            continue
        d   = raw.get(s, {})
        tot = d.get('wt0', 0) + d.get('wt1', 0)
        orig_c1  = round(d.get('wt1', 0) / tot * 100, 1) if tot else 50.0
        synth_c1 = round((vec & (synth_labels == 1)).sum() / n * 100, 1)
        orig_prec  = max(orig_c1,  100 - orig_c1)  / 100
        synth_prec = max(synth_c1, 100 - synth_c1) / 100
        orig_snr   = round(orig_prec  / max(1 - orig_prec,  1e-6), 2)
        synth_snr  = round(synth_prec / max(1 - synth_prec, 1e-6), 2)
        rows.append({
            's':         s,
            'sup':       c['sup'],
            'n_atoms':   c['n_atoms'],
            'sel_score': c['sel_score'],
            'in_rule':   s in set(br_motifs),
            'n_mols':    n,
            'orig_c1':   orig_c1,
            'synth_c1':  synth_c1,
            'orig_snr':  orig_snr,
            'synth_snr': synth_snr,
            'delta_snr': round(synth_snr - orig_snr, 2),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Example molecule walkthrough (from lookup, no re-fragmentation)
# ─────────────────────────────────────────────────────────────────────────────

def _example_molecule(tv_frags: List[List[str]],
                       tv_smiles: List[str],
                       tv_labels: np.ndarray,
                       synth_labels_arr: np.ndarray,
                       raw: dict,
                       br_motifs: List[str],
                       N_tv: int) -> Optional[dict]:
    """Pick the first molecule with 3–6 fragments that fires the best rule."""
    rule_set = set(br_motifs)
    for rank, (smi, frags_m) in enumerate(zip(tv_smiles, tv_frags)):
        if not (3 <= len(frags_m) <= 6):
            continue
        if not any(s in rule_set for s in frags_m):
            continue

        # Build flat tree: mol root → fragment leaves (no sub-hierarchy needed)
        tree_nodes = [{'id': 'mol',
                       'label': smi[:48] + ('…' if len(smi) > 48 else ''),
                       'type': 'mol'}]
        tree_edges = []

        for fi, fs in enumerate(frags_m):
            fid     = f'f{fi}'
            in_rule = fs in rule_set
            tree_nodes.append({
                'id':      fid,
                'label':   fs,
                'type':    'fragment',
                'in_rule': in_rule,
                'n_atoms': pipe.atom_count(fs),
            })
            tree_edges.append({'from': 'mol', 'to': fid})

        rule_motifs_fired = [s for s in frags_m if s in rule_set]
        return {
            'rank':              rank,
            'smiles':            smi,
            'orig_label':        int(tv_labels[rank]),
            'synth_label':       int(synth_labels_arr[rank]),
            'frags':             frags_m,
            'rule_motifs_fired': rule_motifs_fired,
            'tree_nodes':        tree_nodes,
            'tree_edges':        tree_edges,
        }
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis loop
# ─────────────────────────────────────────────────────────────────────────────

def run_all(data_root: str,
            vocab_root: str,
            datasets: List[str],
            variants: List[str],
            fold: int = 0) -> dict:

    output = {}

    for ds in datasets:
        print(f'\n{"="*60}\n  {ds}\n{"="*60}')

        # ── Load CSV ────────────────────────────────────────────────────────
        label_col = DATASET_COLUMN.get(ds)
        if label_col is None:
            print(f'  [skip] {ds} — not in DATASET_COLUMN')
            continue

        csv_path = Path(data_root) / f'{ds}_{fold}.csv'
        if not csv_path.exists():
            print(f'  [skip] {ds} — CSV not found: {csv_path}')
            continue

        df = pd.read_csv(csv_path)
        if label_col not in df.columns:
            print(f'  [skip] {ds} — label column "{label_col}" not in CSV')
            continue

        smiles_all = df['smiles'].tolist()
        labels_all = df[label_col].astype(float).fillna(0).values
        groups_all = df['group'].tolist() if 'group' in df.columns else ['training'] * len(df)

        tv_mask   = np.array([g in ('training', 'valid') for g in groups_all])
        tv_labels = labels_all[tv_mask].astype(int)
        tv_smiles = [s for s, m in zip(smiles_all, tv_mask) if m]
        N_tv      = int(tv_mask.sum())

        output[ds] = {
            'N':       N_tv,
            'orig_n1': int(tv_labels.sum()),
            'orig_n0': int((tv_labels == 0).sum()),
            'orig_c1': round(tv_labels.mean() * 100, 1),
            'methods': {},
        }

        for variant in variants:
            t0 = time.time()
            print(f'\n  [{variant}]')

            # ── Load vocab artifacts ─────────────────────────────────────────
            try:
                lookup_train, lookup_valid, cols_df, smiles_df = \
                    load_vocab_artifacts(vocab_root, ds, variant, fold)
            except FileNotFoundError as e:
                print(f'  [skip] {e}')
                continue

            # ── Reconstruct tv_frags and raw_stats ──────────────────────────
            tv_frags = tv_frags_from_lookup(tv_smiles, lookup_train, lookup_valid)
            raw      = raw_stats_from_cols(cols_df)

            # Sanity check
            n_frag_mols = sum(1 for f in tv_frags if f)
            print(f'    Loaded {len(raw):,} motifs from matrix_columns.csv')
            print(f'    tv_frags: {N_tv} molecules, {n_frag_mols} with fragments')

            # ── Distributions ────────────────────────────────────────────────
            dists = _frag_distributions(tv_frags, raw, N_tv)
            print(f'    motifs={dists["n_unique"]}  '
                  f'mean_fpm={dists["mean_fpm"]}  '
                  f'single={dists["n_single"]}')

            # ── Phase 3 pipeline ─────────────────────────────────────────────
            result = pipe.run_phase3(dict(raw), tv_frags, tv_labels, N_tv, variant)
            br     = result.get('best_rule', {})
            sm     = result.get('snr_metrics', {})
            elapsed = round(time.time() - t0, 1)
            print(f'    match={br.get("pct_match")}%  '
                  f'bal={br.get("balance")}  '
                  f'sep={br.get("separation")}  '
                  f'n_clauses={result.get("n_clauses")}  '
                  f'({elapsed}s)')
            print(f'    rule: {br.get("motifs")}')

            # ── SNR table ────────────────────────────────────────────────────
            selected, presence = pipe.select_top_motifs(raw, tv_frags, N_tv)
            synth_labels_arr   = np.array(result.get('synth_labels', [0] * N_tv))
            snr_rows = _motif_snr_table(
                selected, presence, raw,
                tv_labels, synth_labels_arr, N_tv,
                br.get('motifs', []))

            # ── Example molecule ─────────────────────────────────────────────
            example = _example_molecule(
                tv_frags, tv_smiles, tv_labels, synth_labels_arr,
                raw, br.get('motifs', []), N_tv)

            # ── Enrich selected motifs ───────────────────────────────────────
            sel_enriched = []
            for c in selected[:15]:
                s   = c['s']
                d   = raw.get(s, {})
                tot = d.get('wt0', 0) + d.get('wt1', 0)
                c1  = round(d.get('wt1', 0) / tot * 100, 1) if tot else 50.0
                snr_row = next((r for r in snr_rows if r['s'] == s), {})
                sel_enriched.append({**c,
                    'orig_c1':  c1,
                    'synth_c1': snr_row.get('synth_c1', 50.0),
                    'in_rule':  s in set(br.get('motifs', [])),
                })

            output[ds]['methods'][variant] = {
                'distributions': dists,
                'selected':      sel_enriched,
                'n_clauses':     result['n_clauses'],
                'n_and_valid':   result['n_and_valid'],
                'best_rule':     br,
                'top10_rules':   result.get('all_rules', [])[:10],
                'snr':           sm,
                'snr_rows':      snr_rows,
                'example':       example,
            }

    return output


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_path(path_str: str, script_dir: Path) -> Path:
    """Resolve a path that may be relative.

    Tries in order:
      1. Absolute path or path relative to CWD (standard behaviour)
      2. Path relative to the script's parent directory (project root)
         — handles the common case of running from a subdirectory like
         MotifBreakdown/ while VOCAB_ROOT=./vocab_output is set relative
         to the project root.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    # Try CWD-relative first
    if p.exists():
        return p.resolve()
    # Fall back: try relative to the project root (script's parent dir)
    candidate = (script_dir.parent / path_str).resolve()
    if candidate.exists():
        print(f'  [info] Resolved "{path_str}" → {candidate}')
        return candidate
    # Neither exists — return the CWD-relative version so error messages
    # show the path the user actually passed.
    return p.resolve()


def main():
    p = argparse.ArgumentParser(
        description='Phase 3 analysis — loads from phase1 vocab artifacts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Use paths from experiment_config.sh
  source ../experiment_config.sh
  python3 analyse_phase3.py \\
      --data_root "$DATA_ROOT" \\
      --vocab_root "$VOCAB_ROOT" \\
      --datasets Mutagenicity BBBP Benzene \\
      --variants rbrics all_fallback_bpe

  # Explicit paths
  python3 analyse_phase3.py \\
      --data_root /nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS \\
      --vocab_root /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/vocab_output \\
      --datasets Mutagenicity Benzene \\
      --variants rbrics all_fallback_bpe \\
      --out_json ./results/phase3_analysis.json
""")
    p.add_argument('--data_root',  required=True,
                   help='Directory containing {dataset}_{fold}.csv files')
    p.add_argument('--vocab_root', required=True,
                   help='Root of phase1 vocab output (contains {dataset}/{variant}/ dirs)')
    p.add_argument('--datasets',   nargs='+', required=True,
                   help='Dataset names (must match DATASET_COLUMN keys)')
    p.add_argument('--variants',   nargs='+',
                   default=['rbrics', 'all_fallback_bpe'],
                   help='Vocab variant names (subdirs under {vocab_root}/{dataset}/)')
    p.add_argument('--fold',       type=int, default=0,
                   help='Fold number (default 0)')
    p.add_argument('--out_json',   default=None,
                   help='Output JSON path (default: ./phase3_analysis.json)')
    args = p.parse_args()

    # Resolve relative paths robustly — works whether run from project root
    # or a subdirectory (e.g. MotifBreakdown/).
    script_dir = Path(__file__).resolve().parent
    vocab_root = str(_resolve_path(args.vocab_root, script_dir))
    data_root  = str(_resolve_path(args.data_root,  script_dir))

    out_path = Path(args.out_json).resolve() if args.out_json else \
               Path(vocab_root) / 'phase3_analysis.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'data_root  = {data_root}')
    print(f'vocab_root = {vocab_root}')
    print(f'datasets   = {args.datasets}')
    print(f'variants   = {args.variants}')
    print(f'fold       = {args.fold}')
    print(f'out_json   = {out_path}')

    out = run_all(
        data_root  = data_root,
        vocab_root = vocab_root,
        datasets   = args.datasets,
        variants   = args.variants,
        fold       = args.fold,
    )

    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    size_kb = out_path.stat().st_size // 1024
    print(f'\nSaved {out_path}  ({size_kb} KB)')

    # Print a compact results table
    print(f'\n{"Dataset":<20} {"Variant":<22} {"match%":>7} {"bal":>6} '
          f'{"sep":>6} {"synth_c1":>9} {"motifs"}')
    print('-' * 100)
    for ds, dv in out.items():
        for var, mv in dv.get('methods', {}).items():
            br = mv.get('best_rule', {})
            sm = mv.get('snr', {})
            print(f'{ds:<20} {var:<22} {br.get("pct_match", "?"):>7} '
                  f'{br.get("balance", "?"):>6} {br.get("separation", "?"):>6} '
                  f'{sm.get("pct_synth_1", "?"):>9} '
                  f'{br.get("motifs", [])}')


if __name__ == '__main__':
    main()
