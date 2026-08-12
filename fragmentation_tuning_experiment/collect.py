#!/usr/bin/env python3
"""collect.py — directional (OFAT) analysis of the MoSE tuning sweep for one dataset.

Groups the GAT runs by which knob was moved off the fixed center, prints each
knob's sweep (metric vs knob value) so you can read the DIRECTION, and reports
whether the move helps the model (auc / rmse) and the explainer (pearson_motif /
gt_roc) together. Compares against the deployed GAT run when a prod root is given.

Usage (on HPC, after an array finishes):
    python3 fragmentation_tuning_experiment/collect.py \
        fragmentation_tuning_experiment/runs/conservative_ertl_ring_mdl_filter \
        Benzene_Verified_GT \
        [final_ertlmdl/mose/conservative_ertl_ring_mdl_filter]   # deployed baseline (optional)
"""
import sys, os, json, glob, math

BACKBONE = 'GAT'
CENTER = dict(ent=0.2, size=5e-5, xlr=0.01, glr=1e-3, L=3, h=64, pat=30, wd=0.01, es='loss')
KNOBS = ['ent', 'size', 'xlr', 'glr', 'L', 'h', 'pat', 'wd', 'es']

def num(x):
    try:
        f = float(x); return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None

def approx(a, b):
    if isinstance(b, str):
        return (a or 'loss') == b
    if a is None:
        return False
    return abs(a - b) <= 1e-12 + 1e-6 * abs(b)

def getcfg(d):
    return dict(ent=num(d.get('ent_reg')), size=num(d.get('size_reg')),
                xlr=num(d.get('explainer_lr')), glr=num(d.get('gnn_lr')),
                L=d.get('num_layers'), h=d.get('hidden_dim'),
                pat=d.get('patience'), wd=num(d.get('weight_decay')),
                es=(d.get('early_stop_metric') or 'loss'))

def getmet(d):
    return dict(auc=num(d.get('auc')), rmse=num(d.get('rmse_orig')),
                pearson=num(d.get('pearson_motif_all')),
                gtroc=num(d.get('gt_roc_node_auc_mean_all')))

def read_runs(root, ds, bb):
    out = []
    for sm in glob.glob(os.path.join(root, ds, 'fold0', f'{bb}_*unk-fixed*', 'summary.json')):
        d = json.load(open(sm)); out.append((getcfg(d), getmet(d)))
    return out

def which_knobs(cfg):
    return [k for k in KNOBS if not approx(cfg[k], CENTER[k])]

def metstr(m):
    a = f"{m['auc']:.3f}" if m['auc'] is not None else '  -  '
    r = f"{m['rmse']:.3f}" if m['rmse'] is not None else '  -  '
    p = f"{m['pearson']:.3f}" if m['pearson'] is not None else '  -  '
    g = f"{m['gtroc']:.3f}" if m['gtroc'] is not None else '  -  '
    return a, r, p, g

def main():
    tune_root, ds = sys.argv[1], sys.argv[2]
    prod_root = sys.argv[3] if len(sys.argv) > 3 else None
    runs = read_runs(tune_root, ds, BACKBONE)
    if not runs:
        print(f"no {BACKBONE} runs under {tune_root}/{ds}/fold0"); return

    center = None; sweeps = {k: [] for k in KNOBS}
    for cfg, met in runs:
        dk = which_knobs(cfg)
        if not dk:
            center = (cfg, met)
        elif len(dk) == 1:
            sweeps[dk[0]].append((cfg, met))

    dep = None
    if prod_root:
        pr = read_runs(prod_root, ds, BACKBONE)
        if pr:
            dep = pr[0][1]

    print(f"\n################  {ds} — {BACKBONE} — OFAT sweep  ################")
    hdr = f"{'value':>8} | {'auc':>6} {'rmse':>6} {'pearson':>8} {'gtroc':>6}"
    if dep is not None:
        a, r, p, g = metstr(dep)
        print(f"DEPLOYED GAT baseline:            auc={a} rmse={r} pearson={p} gtroc={g}")
    if center is not None:
        a, r, p, g = metstr(center[1])
        print(f"CENTER (ent0.2,size5e-5,xlr0.01,L3,h64,pat30,wd0.01,es-loss): "
              f"auc={a} rmse={r} pearson={p} gtroc={g}")
    cp = center[1]['pearson'] if center else None

    summary = []
    for k in KNOBS:
        rows = sweeps[k]
        if not rows:
            continue
        # sort by the knob value (numeric if possible)
        def keyf(cm):
            v = cm[0][k]
            return (0, v) if isinstance(v, (int, float)) else (1, str(v))
        rows = sorted(rows, key=keyf)
        # include center as the baseline point for this knob
        allpts = rows + ([('CENTER', center[1])] if center else [])
        print(f"\n--- {k}  (center={CENTER[k]}) ---")
        print(hdr)
        best = None
        for cfg, met in rows:
            a, r, p, g = metstr(met)
            print(f"{str(cfg[k]):>8} | {a:>6} {r:>6} {p:>8} {g:>6}")
            if met['pearson'] is not None and (best is None or met['pearson'] > best[1]):
                best = (cfg[k], met['pearson'], met)
        if center is not None:
            a, r, p, g = metstr(center[1])
            print(f"{str(CENTER[k]):>8} | {a:>6} {r:>6} {p:>8} {g:>6}  <- center")
        # direction verdict for this knob
        if best is not None and cp is not None:
            dp = best[1] - cp
            arrow = 'toward ' + ('higher' if (isinstance(best[0], (int, float)) and best[0] > CENTER[k])
                                 else 'lower' if isinstance(best[0], (int, float)) else str(best[0]))
            # did predictive move the right way vs center too?
            m = best[2]
            pred = ''
            if m['auc'] is not None and center[1]['auc'] is not None:
                pred = f"auc {m['auc'] - center[1]['auc']:+.3f}"
            elif m['rmse'] is not None and center[1]['rmse'] is not None:
                pred = f"rmse {m['rmse'] - center[1]['rmse']:+.3f}"
            gt = ''
            if m['gtroc'] is not None and center[1]['gtroc'] is not None:
                gt = f"gtroc {m['gtroc'] - center[1]['gtroc']:+.3f}"
            print(f"  best pearson at {k}={best[0]} ({dp:+.3f} vs center; move {arrow}) [{pred} {gt}]")
            summary.append((k, best[0], dp, pred, gt))

    if summary:
        print(f"\n================ {ds}: knob directions (by pearson_motif) ================")
        print(f"{'knob':>12} {'best@':>10} {'Δpearson':>9}  predictive/gtroc@best")
        for k, bv, dp, pred, gt in sorted(summary, key=lambda x: -x[2]):
            print(f"{k:>12} {str(bv):>10} {dp:>+9.3f}  {pred} {gt}")

if __name__ == '__main__':
    main()
