#!/usr/bin/env python3
"""instance_offline_baselines.py — OFFLINE per-instance (agnostic) Pearson/Spearman
for the post-hoc baselines, FULL and FILTERED, with NO model forward.

Reconstructs the per-(graph, motif) point cloud from saved data only:
  score(g,m)  = mean over motif m's atoms of the explainer's per-node importance
                (explainer_importances.json[split][method][g] = [per-node]),
                node->motif from the run's loaders (nodes_to_motifs).
  impact(g,m) = impact_cache_agnostic[m][g]   (the shared model-only agnostic LOO).
Instance corr = correlation over all (g,m) points; FILTERED = points whose motif is
in kept_motif_ids. This is the same metric-scope restriction as the grouped re-score.

Self-check (no saved agnostic-instance scalar exists): aggregate the reconstructed
per-(g,m) score/impact back to per-motif (mean over graphs) and compare to the saved
{method}_importance/impact CSVs -> validates the node->motif alignment + extraction.

Needs the loaders (for nodes_to_motifs), NO model forward. Reuses build_gt_loaders.
  python analysis/instance_offline_baselines.py --posthoc posthoc_v1 --out_root final_v2 \
      --data_root <DATA> --vocab_root vocab_final_v2 --processed_root processed_final_v2 \
      --dest baselines_instance_filtered_v1 [--dataset BBBP] [--limit N] [--planted_only]
"""
import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.eval_driver_common import build_gt_loaders

SPLITS = ['train', 'valid', 'test']
METHOD_PFX = {'gnnexplainer': 'GNNExplainer', 'pgexplainer': 'PGExplainer',
              'motif_occlusion': 'MotifOcclusion'}   # mage per-node not in this json


def _wpearson(x, y, w):
    x = np.asarray(x, float); y = np.asarray(y, float); w = np.asarray(w, float)
    W = w.sum()
    if len(x) < 3 or W <= 0:
        return float('nan')
    mx = (w * x).sum() / W; my = (w * y).sum() / W
    vx = (w * (x - mx) ** 2).sum() / W; vy = (w * (y - my) ** 2).sum() / W
    if vx <= 0 or vy <= 0:
        return float('nan')
    return float(((w * (x - mx) * (y - my)).sum() / W) / np.sqrt(vx * vy))


def _rank(a):
    a = np.asarray(a, float); order = a.argsort(); r = np.empty(len(a), float)
    r[order] = np.arange(len(a), dtype=float)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(c); return ((cs - c + cs - 1) / 2.0)[inv]


def _load_kept(vocab_root, dataset, filt_variant):
    p = vocab_root / dataset / filt_variant / f'{dataset}_{filt_variant}_kept_motif_ids.pickle'
    if not p.exists():
        return None
    with open(p, 'rb') as f:
        return set(int(x) for x in pickle.load(f))


def _n2m_by_split(loaders):
    """Per split: list over graphs of nodes_to_motifs arrays (graph order == loader order)."""
    out = {}
    for s in SPLITS:
        if s not in loaders:
            continue
        arrs = []
        for d in loaders[s].dataset:
            n2m = getattr(d, 'nodes_to_motifs', None)
            arrs.append(np.asarray(n2m).ravel() if n2m is not None else None)
        out[s] = arrs
    return out


def process_run(run_dir, meta, args):
    loaders, vocab, dmeta, task_type = build_gt_loaders(
        meta, args.data_root, args.vocab_root, args.processed_root, 128)
    n2m = _n2m_by_split(loaders)
    rows = []
    for pfx, model in METHOD_PFX.items():
        # per-node importance + agnostic impact cache
        eip = run_dir / 'explainer_importances.json'
        if not eip.exists():
            continue
        ei = json.loads(eip.read_text()).get('importances_by_split', {})
        pts = []                       # (score, impact, motif_id)
        agg = {}                       # motif -> [scores], [impacts]  (for self-check)
        for s in SPLITS:
            icp = run_dir / f'impact_cache_agnostic_{s}.json'
            if not (icp.exists() and s in ei and pfx in ei[s] and s in n2m):
                continue
            ic = json.loads(icp.read_text())          # {motif_id: {graph_idx: impact}}
            per_node = ei[s][pfx]                      # {graph_idx: [per-node]}
            # graph -> motifs present
            gm = {}
            for m, gd in ic.items():
                for g in gd:
                    gm.setdefault(g, []).append(int(m))
            for g, motifs in gm.items():
                gi = int(g)
                if gi >= len(n2m[s]) or n2m[s][gi] is None or g not in per_node:
                    continue
                arr = n2m[s][gi]; imp = np.asarray(per_node[g], float)
                if len(imp) != len(arr):
                    continue
                for m in motifs:
                    idx = np.where(arr == m)[0]
                    if len(idx) == 0:
                        continue
                    sc = float(imp[idx].mean())
                    im = float(ic[str(m)][g])
                    pts.append((sc, im, m))
                    agg.setdefault(m, ([], []))
                    agg[m][0].append(sc); agg[m][1].append(im)
        if len(pts) < 3:
            continue
        # self-check: per-motif mean(score/impact) vs saved importance/impact CSVs
        import pandas as pd
        chk_s = chk_i = np.nan
        try:
            fi = pd.concat([pd.read_csv(run_dir / f'{pfx}_importance_{s}.csv')
                            for s in SPLITS if (run_dir / f'{pfx}_importance_{s}.csv').exists()])
            fimp = pd.concat([pd.read_csv(run_dir / f'{pfx}_impact_{s}.csv')
                              for s in SPLITS if (run_dir / f'{pfx}_impact_{s}.csv').exists()])
            svs = fi.groupby('motif_id')['score'].mean(); svi = fimp.groupby('motif_id')['impact'].mean()
            ds, di = [], []
            for m, (ss, ii) in agg.items():
                if m in svs.index:
                    ds.append(abs(np.mean(ss) - svs.loc[m]))
                if m in svi.index:
                    di.append(abs(np.mean(ii) - svi.loc[m]))
            chk_s = float(np.nanmax(ds)) if ds else np.nan
            chk_i = float(np.nanmax(di)) if di else np.nan
        except Exception:
            pass
        # instance corr full + filtered
        kept = process_run.kept
        x = [p[0] for p in pts]; y = [p[1] for p in pts]
        xf = [p[0] for p in pts if p[2] in kept]; yf = [p[1] for p in pts if p[2] in kept]
        row = dict(model=model, n_instances_full=len(pts), n_instances_filt=len(xf),
                   selfcheck_score_absdiff=chk_s, selfcheck_impact_absdiff=chk_i)
        row['instance_pearson_agnostic_full'] = _wpearson(x, y, [1.0] * len(x))
        row['instance_spearman_agnostic_full'] = _wpearson(_rank(x), _rank(y), [1.0] * len(x))
        row['instance_pearson_agnostic_filtered'] = _wpearson(xf, yf, [1.0] * len(xf)) if len(xf) >= 3 else np.nan
        row['instance_spearman_agnostic_filtered'] = _wpearson(_rank(xf), _rank(yf), [1.0] * len(xf)) if len(xf) >= 3 else np.nan
        rows.append(row)
    return rows


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument('--posthoc', required=True)
    ap.add_argument('--out_root', required=True, help='final_v2 (best_model.pt + meta for loaders)')
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--vocab_root', required=True)
    ap.add_argument('--processed_root', required=True)
    ap.add_argument('--dest', required=True)
    ap.add_argument('--dataset', nargs='*', default=None)
    ap.add_argument('--planted_only', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    posthoc = Path(args.posthoc); vocab_root = Path(args.vocab_root); dest = Path(args.dest)
    outp = dest / 'baselines_instance_filtered.csv'
    if outp.exists():
        raise SystemExit(f'refusing to overwrite {outp}')

    out_rows = []
    n_ok = n_fail = 0
    for ss in (posthoc / 'baselines').rglob('summary_splits.json'):
        parts = ss.parent.relative_to(posthoc).parts
        if len(parts) < 5:
            continue
        _, dataset, fold_s, vocab, runid = parts[:5]
        mg = re.search(r'_gt_dnf_k(\d+)_r(\d+)$', runid)
        if args.planted_only and not mg:
            continue
        if args.dataset and dataset not in args.dataset:
            continue
        gt_rule = f'dnf_k{mg.group(1)}_r{mg.group(2)}' if mg else ''
        mbb = re.match(r'bb-([A-Za-z]+)_', runid); backbone = mbb.group(1) if mbb else ''
        mf = re.match(r'fold(\d+)', fold_s); fold = int(mf.group(1)) if mf else -1
        base = vocab[:-7] if vocab.endswith('_filter') else vocab
        filt_variant = base + '_filter'                   # kept-set is label-independent
        kept = _load_kept(vocab_root, dataset, filt_variant)
        if kept is None:
            continue
        # minimal meta for the loaders: instance corr needs only nodes_to_motifs,
        # so dataset+vocab_variant+fold suffice (processed_root re-resolves; no GT/model).
        rel = ss.parent.relative_to(posthoc)
        meta = dict(dataset=dataset, vocab_variant=base, fold=fold, gt_cache=None)
        process_run.kept = kept
        try:
            rr = process_run(ss.parent, meta, args)
            for row in rr:
                row.update(dataset=dataset, backbone=backbone, fold=fold, fragmentation=base,
                           gt_rule=gt_rule, gt_tier=('planted' if gt_rule else 'source_or_none'),
                           run_path=str(rel))
                out_rows.append(row)
            n_ok += 1
        except Exception as e:
            print(f'  [FAIL] {rel}: {type(e).__name__}: {e}')
            n_fail += 1
        if args.limit and n_ok >= args.limit:
            break

    dest.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out_rows)
    df.to_csv(outp, index=False)
    print(f'runs ok={n_ok} fail={n_fail}  method-rows={len(df)}')
    if len(df):
        print('self-check max |per-motif agg - saved|: score=%.2e impact=%.2e'
              % (np.nanmax(df.selfcheck_score_absdiff.values), np.nanmax(df.selfcheck_impact_absdiff.values)))
        print(df.groupby('model')[['instance_pearson_agnostic_full',
              'instance_pearson_agnostic_filtered']].mean().round(3).to_string())
    print(f'wrote {outp}')


if __name__ == '__main__':
    main()
