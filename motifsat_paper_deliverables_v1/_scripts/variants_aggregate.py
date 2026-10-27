"""variants_aggregate.py — local aggregation of variants_collect.py output.

usage: python3 variants_aggregate.py runs.jsonl r_table.sh OUT_DIR   (paper_v1: r_table_paper.sh)
Checks every run against the campaign definition (grouping recorded = grouping in path; final_r /
decay_interval = r_table.sh, fix_r only for base_gsat_noanneal) and lists violations instead of
averaging them. Writes
  OUT_DIR/cells.csv      one row per (dataset, vocab, preset, grouping, backbone): n folds, mean, std
  OUT_DIR/tables.md      per dataset x metric: rows vocab/preset/grouping, columns backbones,
                         'mean ± std' over folds; '(n=k)' when fewer than 5 folds.
"""
import json
import os
import re
import sys

import pandas as pd

METRICS = [('pred', 'Test prediction (ROC-AUC; RMSE for ESOL / Lipophilicity)'),
           ('gtroc', 'Test node GT-ROC'),
           ('pm_all', 'Grouped Pearson, impact = mask all copies'),
           ('pm_one', 'Grouped Pearson, impact = mask one copy'),
           ('pi_all', 'Instance Pearson, impact = mask all copies'),
           ('pi_one', 'Instance Pearson, impact = mask one copy'),
           ('score_std', 'Motif score std')]
DATASETS = ['Benzene_Verified_GT', 'Alkane_Carbonyl_Verified_GT', 'Fluoride_Carbonyl_Verified_GT',
            'BBBP', 'hERG', 'Mutagenicity', 'esol', 'Lipophilicity',
            # GSAT paper datasets (paper_v1 tree, r_table_paper.sh); MultiLabel pred = mean per-task ROC-AUC
            'mutag', 'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp', 'ogbg-moltox21', 'ogbg-molclintox']
BACKBONES = ['GIN', 'GCN', 'GAT', 'SAGE', 'PNA']
PRESETS = ['base_gsat', 'base_gsat_noanneal', 'intra_consistency', 'inter_loss', 'intra_inter_loss',
           'readout', 'readout_inter', 'readout_layernorm', 'readout_nonorm',
           'motif_emb_multihot_gin_res', 'motif_emb_multihot_chem_res',
           'motif_emb_iddesc_gin_res', 'motif_emb_iddesc_chem_res',
           # KL-level control (2026-10-26): motif gates, KL averaged over atoms
           'readout_layernorm_nodekl', 'readout_nonorm_nodekl', 'motif_emb_iddesc_gin_res_nodekl',
           # motif-identity control (2026-10-26): base GSAT + motif-id features in the predictor
           'base_gsat_motifid']


def main(runs_path, rtable_path, out_dir):
    rows = [json.loads(line) for line in open(runs_path) if line.strip()]
    df = pd.DataFrame(rows)
    rtab = {f'{a}|{b}': (float(r), int(i)) for a, b, r, i in
            re.findall(r'"([^"|]+)\|([^"]+)"\) echo "([\d.]+) (\d+)"', open(rtable_path).read())}
    bad = []
    for r in rows:
        exp_g = 'pooled' if r['grouping'] == 'none' else r['grouping']
        fr, di = rtab.get(f"{r['dataset']}|{r['backbone']}", (None, None))
        if r['motif_grouping'] != exp_g:
            bad.append((r, 'grouping'))
        if r['final_r'] is None or abs(float(r['final_r']) - fr) > 1e-6 or int(r['decay_interval']) != di:
            bad.append((r, f"r {r['final_r']}/{r['decay_interval']} != {fr}/{di}"))
        if bool(r['fix_r']) != (r['preset'] == 'base_gsat_noanneal'):
            bad.append((r, f"fix_r={r['fix_r']}"))
        if r['preset'].startswith('motif_emb_'):           # recorded GNN1 knobs must match the preset
            feat, edge = r['preset'][len('motif_emb_'):].split('_')[:2]  # e.g. iddesc, chem (res[, nodekl])
            want = dict(motif_feat='id_desc' if feat == 'iddesc' else 'multihot', motif_edge_feat=edge == 'chem',
                        motif_edge_chem=edge == 'chem', motif_gnn_layers=2, motif_gnn_norm='layer', motif_gnn_residual=True)
            got = {k: r.get(k) for k in want}
            if got != want:
                bad.append((r, f"motif_emb knobs {got}"))
    os.makedirs(out_dir, exist_ok=True)
    print(f'{len(rows)} runs; {len(bad)} violations')
    for r, why in bad[:20]:
        print('  VIOLATION', why, r['vocab'], r['preset'], r['grouping'], r['dataset'], r['fold'], r['backbone'])
    bad_keys = {(r['vocab'], r['preset'], r['grouping'], r['dataset'], r['fold'], r['backbone']) for r, _ in bad}
    df = df[[tuple(x) not in bad_keys for x in
             df[['vocab', 'preset', 'grouping', 'dataset', 'fold', 'backbone']].itertuples(index=False)]]
    keys = ['dataset', 'vocab', 'preset', 'grouping', 'backbone']
    m = [k for k, _ in METRICS] + ['edge_auc', 'pi_all_multi', 'pi_one_multi', 'score_mean']
    agg = df.groupby(keys)[m].agg(['mean', 'std', 'count'])
    agg.columns = [f'{a}_{b}' for a, b in agg.columns]
    agg.reset_index().to_csv(f'{out_dir}/cells.csv', index=False)

    def cell(g, k):
        v = g[k].dropna()
        if v.empty:
            return '—'
        s = f'{v.mean():.3f} ± {v.std(ddof=1):.3f}' if len(v) > 1 else f'{v.mean():.3f}'
        return s + ('' if len(v) == 5 else f' (n={len(v)})')

    out = ['# MotifSAT variant campaign — test-split tables', '',
           f'{len(df)} complete runs (after {len(bad)} violations removed). Mean ± std over folds.', '']
    for ds in DATASETS:
        d = df[df.dataset == ds]
        if d.empty:
            continue
        for k, title in METRICS:
            if k == 'gtroc' and d[k].isna().all():
                continue
            out += [f'### {ds} — {title}', '', '| Vocab | Preset | Grouping | ' + ' | '.join(BACKBONES) + ' |',
                    '|---|---|---|' + '---|' * len(BACKBONES)]
            for (v, p, g), gg in sorted(d.groupby(['vocab', 'preset', 'grouping']),
                                        key=lambda x: (x[0][0], PRESETS.index(x[0][1]), x[0][2])):
                out.append(f'| {v} | {p} | {g} | ' + ' | '.join(cell(gg[gg.backbone == b], k) for b in BACKBONES) + ' |')
            out.append('')
    open(f'{out_dir}/tables.md', 'w').write('\n'.join(out))
    print(f'wrote {out_dir}/cells.csv, {out_dir}/tables.md')


if __name__ == '__main__':
    main(*sys.argv[1:4])
