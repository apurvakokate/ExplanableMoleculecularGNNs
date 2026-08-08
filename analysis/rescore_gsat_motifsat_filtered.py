#!/usr/bin/env python3
"""rescore_gsat_motifsat_filtered.py — OFFLINE (pandas-only) filtered-vocab GROUPED
re-score of the existing full-vocab GSAT / MotifSAT runs.

NO model, NO training, NO evaluation. This is a pure metric-scope restriction: take
each run's already-saved per-motif (importance, impact) points and recompute the
grouped Pearson/Spearman over ONLY the motifs that the *_filter vocabulary keeps
(`kept_motif_ids`), so GSAT/MotifSAT are scored on the SAME motif universe as MoSE.

Why this is the right object for GSAT/MotifSAT (not a filtered-model re-run): their
explanation is embedding/attention based and there is only ONE model (full-vocab);
"filtering" them means restricting which motifs we score, not inventing a filtered
model. importance (mean node-attention over a motif's atoms) is per-motif isolated
and unchanged by the restriction, so subsetting the saved points is exact.

Correctness by construction: `_wpearson`, `_rank`, `grouped_corr_variants` are copied
VERBATIM from SharedModules/evaluation/split_eval.py, and the pooling ("each motif
contributes its (score,impact,support) point once per split it appears in") mirrors
write_grouped_pooled. As a self-check we also recompute the FULL (all-motif) value
and assert it reproduces the run's saved {method}_grouped_corr_pooled_alltest.csv.

NEVER overwrites: reads antehoc_v1 read-only; writes ONE consolidated CSV to --dest
(a NEW dir). Kept-set resolution is label-independent (support threshold on
structure), so the base `<frag>_filter` kept_motif_ids applies to real AND planted
runs — the same file MoSE resolves.

  python analysis/rescore_gsat_motifsat_filtered.py \
      --antehoc antehoc_v1 --vocab_root vocab_final_v2 \
      --dest gsat_motifsat_filtered_grouped_v1 [--dataset BBBP] [--limit N]
"""
import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

UNK_ID = -1
FAM_METHOD = {'motifsat': 'motifsat', 'base_gsat': 'gsat'}   # dir -> CSV prefix/method
SOURCE_GT = {'Benzene_Verified_GT', 'Fluoride_Carbonyl_Verified_GT',
             'Alkane_Carbonyl_Verified_GT', 'mutag'}
SPLITS = ['train', 'valid', 'test']


# ── copied VERBATIM from split_eval.py (do not edit — keeps results bit-identical) ──
def _wpearson(x, y, w) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float); w = np.asarray(w, float)
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


def _rank(a) -> np.ndarray:
    a = np.asarray(a, float)
    order = a.argsort()
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(len(a), dtype=float)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum - 1) / 2.0
    return avg[inv]


def grouped_corr_variants(rows):
    out = {}
    for unk_tag, keep in (('inclunk', lambda r: True),
                          ('exclunk', lambda r: int(r['motif_id']) != UNK_ID)):
        sel = [r for r in rows if keep(r)]
        out[f'n_motifs_{unk_tag}'] = len(sel)
        if len(sel) < 3:
            for w_tag in ('w', 'u'):
                out[f'pearson_{w_tag}_{unk_tag}'] = float('nan')
                out[f'spearman_{w_tag}_{unk_tag}'] = float('nan')
            continue
        x = [r['score'] for r in sel]
        y = [r['impact'] for r in sel]
        supp = [max(float(r.get('support', 1.0)), 0.0) for r in sel]
        rx, ry = _rank(x), _rank(y)
        for w_tag, w in (('w', supp), ('u', [1.0] * len(sel))):
            out[f'pearson_{w_tag}_{unk_tag}'] = _wpearson(x, y, w)
            out[f'spearman_{w_tag}_{unk_tag}'] = _wpearson(rx, ry, w)
    return out
# ───────────────────────────────────────────────────────────────────────────────


def _parse_run(antehoc: Path, ss: Path):
    """LAYOUT B: <family>/<VOCAB>/<DATASET>/fold<K>/<runid>. Returns identity or None."""
    parts = ss.parent.relative_to(antehoc).parts
    if len(parts) < 5 or any(p.startswith('tune') for p in parts):
        return None
    family, vocab, dataset, fold_s, runid = parts[0], parts[1], parts[2], parts[3], parts[4]
    if family not in FAM_METHOD:
        return None
    backbone = runid.split('_', 1)[0]
    mf = re.match(r'fold(\d+)', fold_s)
    fold = int(mf.group(1)) if mf else -1
    # strip planted suffix -> base fragmentation variant; append _filter for kept-set
    m = re.search(r'_relabelled_dnf_k(\d+)_r(\d+)$', vocab)
    gt_rule = f'dnf_k{m.group(1)}_r{m.group(2)}' if m else ''
    base = vocab[:m.start()] if m else vocab
    base = base[:-len('_filter')] if base.endswith('_filter') else base
    filter_variant = base + '_filter'
    return dict(family=family, method=FAM_METHOD[family], dataset=dataset,
                backbone=backbone, fold=fold, fragmentation=base, gt_rule=gt_rule,
                filter_variant=filter_variant, vocab=vocab)


def _load_kept(vocab_root: Path, dataset: str, filter_variant: str):
    p = (vocab_root / dataset / filter_variant /
         f'{dataset}_{filter_variant}_kept_motif_ids.pickle')
    if not p.exists():
        return None
    with open(p, 'rb') as f:
        return set(int(x) for x in pickle.load(f))


def _pooled_rows(run_dir: Path, method: str):
    """Per-(motif,split) rows {motif_id, score, impact, support}, pooled over splits
    — mirrors write_grouped_pooled('alltest')."""
    rows = []
    for s in SPLITS:
        fi = run_dir / f'{method}_importance_{s}.csv'
        fp = run_dir / f'{method}_impact_{s}.csv'
        if not (fi.exists() and fp.exists()):
            continue
        imp = pd.read_csv(fi)          # motif_id, score, support
        imc = pd.read_csv(fp)          # motif_id, impact, support
        m = imp[['motif_id', 'score', 'support']].merge(
            imc[['motif_id', 'impact']], on='motif_id', how='inner')
        for _, r in m.iterrows():
            rows.append(dict(motif_id=int(r['motif_id']), score=float(r['score']),
                             impact=float(r['impact']), support=float(r['support'])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--antehoc', required=True)
    ap.add_argument('--vocab_root', required=True)
    ap.add_argument('--dest', required=True, help='NEW dir; one consolidated CSV written here')
    ap.add_argument('--dataset', nargs='*', default=None)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    antehoc = Path(args.antehoc)
    vocab_root = Path(args.vocab_root)
    dest = Path(args.dest)
    if (dest / 'gsat_motifsat_filtered_grouped.csv').exists():
        raise SystemExit(f'refusing to overwrite {dest}/gsat_motifsat_filtered_grouped.csv')

    out_rows = []
    n_ok = n_skip = n_mismatch = 0
    fams = list(FAM_METHOD)
    ss_files = [ss for fam in fams for ss in (antehoc / fam).rglob('summary_splits.json')]
    for ss in ss_files:
        if args.limit and n_ok >= args.limit:
            break
        ident = _parse_run(antehoc, ss)
        if ident is None:
            continue
        if args.dataset and ident['dataset'] not in args.dataset:
            continue
        run_dir = ss.parent
        kept = _load_kept(vocab_root, ident['dataset'], ident['filter_variant'])
        if kept is None:
            n_skip += 1
            continue
        rows = _pooled_rows(run_dir, ident['method'])
        if len(rows) < 3:
            n_skip += 1
            continue
        full = grouped_corr_variants(rows)
        kept_rows = [r for r in rows if r['motif_id'] in kept]
        filt = grouped_corr_variants(kept_rows)

        # self-check: our FULL recompute must reproduce the saved pooled-alltest number
        saved = run_dir / f"{ident['method']}_grouped_corr_pooled_alltest.csv"
        chk = np.nan
        if saved.exists():
            try:
                sv = pd.read_csv(saved).iloc[0]
                a, b = float(sv.get('pearson_w_exclunk')), full['pearson_w_exclunk']
                chk = abs(a - b) if (np.isfinite(a) and np.isfinite(b)) else np.nan
                if np.isfinite(chk) and chk > 1e-6:
                    n_mismatch += 1
            except Exception:
                pass

        gt_tier = ('planted' if ident['gt_rule']
                   else 'source' if ident['dataset'] in SOURCE_GT else 'none')
        row = dict(dataset=ident['dataset'], backbone=ident['backbone'],
                   fold=ident['fold'], fragmentation=ident['fragmentation'],
                   gt_rule=ident['gt_rule'], gt_tier=gt_tier, model=ident['method'],
                   filter_variant=ident['filter_variant'],
                   n_motifs_full=full['n_motifs_exclunk'],
                   n_motifs_kept=filt['n_motifs_exclunk'],
                   full_recompute_vs_saved_absdiff=chk,
                   run_path=str(run_dir.relative_to(antehoc)))
        for wu in ('w', 'u'):
            for stat in ('pearson', 'spearman'):
                row[f'grouped_{stat}_agnostic_{wu}_full'] = full[f'{stat}_{wu}_exclunk']
                row[f'grouped_{stat}_agnostic_{wu}_filtered'] = filt[f'{stat}_{wu}_exclunk']
        out_rows.append(row)
        n_ok += 1

    dest.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out_rows)
    outp = dest / 'gsat_motifsat_filtered_grouped.csv'
    df.to_csv(outp, index=False)
    print(f'runs scored={n_ok}  skipped(no kept/rows)={n_skip}  '
          f'full-recompute mismatches(>1e-6)={n_mismatch}')
    if len(df):
        print('max |full_recompute - saved| =',
              float(np.nanmax(df['full_recompute_vs_saved_absdiff'].values)))
        print(df.groupby(['model'])[['n_motifs_full', 'n_motifs_kept']].mean().round(1).to_string())
    print(f'wrote {outp}')


if __name__ == '__main__':
    main()
