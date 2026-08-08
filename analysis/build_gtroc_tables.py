#!/usr/bin/env python3
"""build_gtroc_tables.py — aggregate the node-direct GT-ROC long-form CSV into tables.

Input : gtroc_long.csv (dataset,backbone,fragmentation,fold,gt_tier,method,split,
        gtroc_instance,gtroc_global)  — one row per (setup, method, split).
Output (into --out-dir):
  gtroc_agg_by_fold.csv  : mean/std OVER FOLDS, per
        (dataset,gt_tier,fragmentation,backbone,method,split) x {instance,global}.
  tables_<split>_<metric>.md : markdown pivots (rows=method, cols=backbone), one block
        per (dataset,gt_tier,fragmentation), value = mean-over-folds. metric in
        {instance,global}; split default test.

  python analysis/build_gtroc_tables.py --csv <gtroc_long.csv> --out-dir <dir> [--split test]
"""
import argparse
from pathlib import Path
import pandas as pd

BACKBONES = ['GIN', 'GCN', 'GAT', 'SAGE', 'PNA']
METHODS = ['gnnexplainer', 'pgexplainer', 'motif_occlusion', 'mage_v2']
MLABEL = {'gnnexplainer': 'GNNExplainer', 'pgexplainer': 'PGExplainer',
          'motif_occlusion': 'MotifOcclusion', 'mage_v2': 'MAGE'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--split', default='test')
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    # 1) mean/std over folds
    keys = ['dataset', 'gt_tier', 'fragmentation', 'backbone', 'method', 'split']
    agg = (df.groupby(keys)[['gtroc_instance', 'gtroc_global']]
             .agg(['mean', 'std', 'count']).reset_index())
    agg.columns = ['_'.join(c).rstrip('_') for c in agg.columns]
    agg.to_csv(out / 'gtroc_agg_by_fold.csv', index=False)
    print(f'wrote {out/"gtroc_agg_by_fold.csv"}  ({len(agg)} rows)')

    # 2) markdown pivots per (dataset,gt_tier,fragmentation), rows=method cols=backbone
    def pivot_tables(metric):
        col = f'gtroc_{metric}'
        sub = df[df.split == args.split]
        lines = [f'# Node-direct GT-ROC — {metric.upper()} (split={args.split}, mean over folds)\n']
        for (ds, tier), g0 in sub.groupby(['dataset', 'gt_tier']):
            lines.append(f'\n## {ds}  ({tier})')
            for frag, g in g0.groupby('fragmentation'):
                piv = (g.groupby(['method', 'backbone'])[col].mean().reset_index()
                        .pivot(index='method', columns='backbone', values=col))
                piv = piv.reindex(index=[m for m in METHODS if m in piv.index],
                                  columns=[b for b in BACKBONES if b in piv.columns])
                piv.index = [MLABEL.get(m, m) for m in piv.index]
                lines.append(f'\n### fragmentation = {frag}\n')
                lines.append('| method | ' + ' | '.join(piv.columns) + ' |')
                lines.append('|' + '---|' * (len(piv.columns) + 1))
                for m, row in piv.iterrows():
                    cells = ['' if pd.isna(v) else f'{v:.3f}' for v in row]
                    lines.append(f'| {m} | ' + ' | '.join(cells) + ' |')
        p = out / f'tables_{args.split}_{metric}.md'
        p.write_text('\n'.join(lines) + '\n')
        print(f'wrote {p}')

    pivot_tables('instance')
    pivot_tables('global')

    # 3) compact headline: per (dataset,tier) mean over folds+fragmentations+backbones
    head = (df[df.split == args.split].groupby(['dataset', 'gt_tier', 'method'])
              [['gtroc_instance', 'gtroc_global']].mean().reset_index())
    head['method'] = head['method'].map(lambda m: MLABEL.get(m, m))
    head.to_csv(out / f'gtroc_headline_{args.split}.csv', index=False)
    print(f'wrote {out/f"gtroc_headline_{args.split}.csv"}')
    print('\n=== headline (test, mean over folds+frag+backbone) ===')
    print(head.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
