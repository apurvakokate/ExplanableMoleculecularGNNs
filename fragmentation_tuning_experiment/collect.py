#!/usr/bin/env python3
"""collect.py — summarize the MoSE HP-tuning runs for one dataset, per backbone.

For each backbone it prints the 16 HP cases sorted by pearson_motif_all (the
target metric for esol/Lipo/hERG regression / no-node-GT), compares the best
against the DEPLOYED final_ertlmdl run for that (backbone, dataset), and states
whether tuning raised or lowered Pearson.

Usage (on HPC, after the array finishes):
    python3 fragmentation_tuning_experiment/collect.py \
        fragmentation_tuning_experiment/runs/conservative_ertl_ring_mdl_filter \
        esol \
        [final_ertlmdl/mose/conservative_ertl_ring_mdl_filter]   # deployed baseline root (optional)
"""
import sys, os, json, glob, math

BACKBONES = ['GIN', 'GCN', 'SAGE', 'GAT', 'PNA']

def num(x):
    try:
        f = float(x)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None

def read_runs(root, ds, bb):
    rows = []
    for sm in glob.glob(os.path.join(root, ds, 'fold0', f'{bb}_*unk-fixed*', 'summary.json')):
        d = json.load(open(sm))
        rows.append(dict(
            ent=num(d.get('ent_reg')), size=num(d.get('size_reg')),
            xlr=num(d.get('explainer_lr')), L=d.get('num_layers'),
            h=d.get('hidden_dim'), glr=num(d.get('gnn_lr')), pat=d.get('patience'),
            rmse=num(d.get('rmse_orig')), auc=num(d.get('auc')),
            pearson=num(d.get('pearson_motif_all')),
            gtroc=num(d.get('gt_roc_node_auc_mean_all')),
        ))
    return rows

def fmt(r, dp=''):
    rm = f"{r['rmse']:.3f}" if r['rmse'] is not None else '  -  '
    au = f"{r['auc']:.3f}" if r['auc'] is not None else '  -  '
    gt = f"{r['gtroc']:.3f}" if r['gtroc'] is not None else '  -  '
    pe = f"{r['pearson']:.3f}" if r['pearson'] is not None else '  -  '
    return (f"{str(r['ent']):>5} {str(r['size']):>7} {str(r['xlr']):>5} {str(r['L']):>2} "
            f"{str(r['h']):>3} {str(r['glr']):>7} {str(r['pat']):>3} | "
            f"{rm:>6} {au:>6} {pe:>8} {gt:>6} {dp:>7}")

def main():
    tune_root = sys.argv[1]
    ds = sys.argv[2]
    prod_root = sys.argv[3] if len(sys.argv) > 3 else None

    summary = []
    for bb in BACKBONES:
        rows = read_runs(tune_root, ds, bb)
        if not rows:
            continue
        rows.sort(key=lambda r: (r['pearson'] if r['pearson'] is not None else -9), reverse=True)
        # deployed baseline for this (bb, ds)
        bp = None
        if prod_root:
            prod = read_runs(prod_root, ds, bb)
            cand = [r['pearson'] for r in prod if r['pearson'] is not None]
            bp = cand[0] if cand else None

        print(f"\n=== {ds} — {bb} — conservative_ertl_ring_mdl_filter (fold 0) ===")
        print(f"{'ent':>5} {'size':>7} {'xlr':>5} {'L':>2} {'h':>3} {'glr':>7} {'pat':>3} | "
              f"{'rmse':>6} {'auc':>6} {'pearson':>8} {'gtroc':>6} {'Δvs.dep':>7}")
        print('-' * 82)
        for r in rows:
            dp = '' if (bp is None or r['pearson'] is None) else f"{r['pearson'] - bp:+.3f}"
            print(fmt(r, dp))
        best = rows[0]
        if bp is not None:
            verdict = 'IMPROVED' if best['pearson'] > bp else 'did NOT improve'
            print(f"deployed pearson={bp:.3f}  best-tuned pearson={best['pearson']:.3f}  "
                  f"-> tuning {verdict} ({best['pearson'] - bp:+.3f})")
            summary.append((bb, bp, best['pearson']))
        else:
            print(f"best-tuned pearson={best['pearson']:.3f}  (no deployed baseline found)")
            summary.append((bb, None, best['pearson']))

    if summary:
        print(f"\n================ {ds}: verdict across backbones ================")
        print(f"{'bb':>5} {'deployed':>9} {'best-tuned':>11} {'Δ':>7}  result")
        for bb, bp, best in summary:
            d = '' if bp is None else f"{best - bp:+.3f}"
            res = '' if bp is None else ('IMPROVED' if best > bp else 'no-improve')
            print(f"{bb:>5} {('%.3f'%bp) if bp else '  -  ':>9} {best:>11.3f} {d:>7}  {res}")

if __name__ == '__main__':
    main()
