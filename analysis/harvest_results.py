"""Harvest every summary.json under OUT_ROOT into ONE dated, provenance-stamped,
versioned results CSV — the single source of truth Rule Set 1 scans.

Design (Rule Set 1, decisions #1/#2):
  * one row per summary.json (raw scalar fields + parsed axes + provenance);
  * provenance read from the summary (git_sha/run_timestamp/config_hash, written by the
    producer); if absent (legacy), fall back to the file mtime and git_sha='unknown';
  * output results_YYYYMMDD.csv is append-only + idempotent, keyed by summary_path+git_sha
    so a re-harvest never duplicates and a re-run under new code keeps BOTH versions;
  * count reconciliation: summaries found vs rows written vs skipped;
  * paper-scope filters as functions over the CSV.

Usage:  OUT_ROOT=/path python3 analysis/harvest_results.py [--save_dir DIR] [--out_root DIR]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from SharedModules.evaluation.provenance import PROVENANCE_KEYS  # noqa: E402

AXES = ('dataset', 'backbone', 'fold', 'vocab_variant', 'gt_tier', 'use_gt',
        'model_type', 'variant_tag', 'node_encoder', 'conv_normalize')
KEY = ['summary_path', 'git_sha']


def _scalar_items(d: dict):
    for k, v in d.items():
        if v is None or isinstance(v, (int, float, str, bool)):
            yield k, v


def load_row(path: Path) -> dict | None:
    try:
        d = json.load(open(path))
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    row = dict(_scalar_items(d))
    # provenance: prefer producer-written fields, else mtime + unknown
    if d.get('git_sha'):
        prov = {k: d.get(k) for k in PROVENANCE_KEYS}
    else:
        prov = {'git_sha': 'unknown',
                'run_timestamp': _dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec='seconds'),
                'config_hash': 'unknown'}
    row.update(prov)
    for a in AXES:
        row.setdefault(a, d.get(a))
    row['summary_path'] = str(path)
    return row


def harvest(out_root: str):
    root = Path(out_root)
    paths = sorted(root.rglob('summary.json'))
    rows, skipped = [], 0
    for p in paths:
        r = load_row(p)
        if r is None:
            skipped += 1
        else:
            rows.append(r)
    df = pd.DataFrame(rows)
    return df, len(paths), skipped


def write_dated(df: pd.DataFrame, save_dir: str) -> Path:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = Path(save_dir) / f"results_{_dt.date.today().strftime('%Y%m%d')}.csv"
    if path.exists():                       # append-only + versioned: keep prior rows
        prev = pd.read_csv(path, dtype=str)
        df = pd.concat([prev, df.astype(str)], ignore_index=True)
    key = [k for k in KEY if k in df.columns]
    df = df.drop_duplicates(subset=key, keep='first')   # idempotent per (summary, code version)
    df.to_csv(path, index=False)
    return path


# ── paper-scope views (deterministic filters over the harvested CSV) ───────────
_MOSE_MOTIFSAT = {'MOSE', 'MotifSAT', 'GSAT'}

def scope_mose(df):            # MOSE journal: rbrics only (fg_first → appendix)
    base = df['vocab_variant'].astype(str).str.replace('_filter', '', regex=False)
    return df[base.str.startswith('rbrics')]

def scope_synthetic_labelling(df):   # exclude the self-explaining models
    mt = df.get('model_type', pd.Series(['']*len(df))).astype(str)
    return df[~mt.isin(_MOSE_MOTIFSAT)]

def scope_motifsat(df):        # everything
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', default=os.environ.get('OUT_ROOT', 'results_v2'))
    ap.add_argument('--save_dir', default=os.environ.get('HARVEST_DIR', 'analysis/harvest'))
    a = ap.parse_args()
    df, n_found, n_skipped = harvest(a.out_root)
    out = write_dated(df, a.save_dir) if len(df) else None
    print(f"summaries found : {n_found}")
    print(f"rows harvested  : {len(df)}")
    print(f"skipped (bad)   : {n_skipped}")
    if len(df):
        n_prov = int((df['git_sha'] != 'unknown').sum()) if 'git_sha' in df else 0
        print(f"with provenance : {n_prov}/{len(df)} (rest = legacy, mtime fallback)")
        print(f"reconciliation  : found {n_found} == harvested {len(df)} + skipped {n_skipped} "
              f"→ {'OK' if n_found == len(df) + n_skipped else 'MISMATCH'}")
        print(f"written         : {out}")


if __name__ == '__main__':
    main()
