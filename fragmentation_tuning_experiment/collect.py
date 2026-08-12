#!/usr/bin/env python3
"""collect.py — fold-averaged OFAT analysis of the MoSE tuning sweep for one dataset.

Averages each config across all available folds (mean +/- std), groups by which
knob was moved off the fixed center, and reports the direction with variance
visible. Compares against the fold-averaged DEPLOYED GAT run. Flags configs whose
auc collapses (high pearson on a broken model is not a win).

Usage (on HPC):
    python3 fragmentation_tuning_experiment/collect.py \
        fragmentation_tuning_experiment/runs/conservative_ertl_ring_mdl_filter \
        Benzene_Verified_GT \
        [final_ertlmdl/mose/conservative_ertl_ring_mdl_filter]   # deployed baseline (optional)
"""
import sys, os, json, glob, math
from collections import defaultdict

BACKBONE = 'GAT'
CENTER = dict(ent=0.2, size=5e-5, xlr=0.01, glr=1e-3, L=3, h=64, pat=30, wd=0.01, es='loss')
KNOBS = ['ent', 'size', 'xlr', 'glr', 'L', 'h', 'pat', 'wd', 'es']
AUC_GUARD = 0.05   # flag configs whose mean auc is >this below the deployed/center baseline

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

def cfgkey(d):
    return (round(num(d.get('ent_reg')) or -1, 6), round(num(d.get('size_reg')) or -1, 9),
            round(num(d.get('explainer_lr')) or -1, 6), d.get('num_layers'), d.get('hidden_dim'),
            round(num(d.get('gnn_lr')) or -1, 6), d.get('patience'),
            round(num(d.get('weight_decay')) or -1, 6), d.get('early_stop_metric') or 'loss')

def cfgdict(d):
    return dict(ent=num(d.get('ent_reg')), size=num(d.get('size_reg')), xlr=num(d.get('explainer_lr')),
                L=d.get('num_layers'), h=d.get('hidden_dim'), glr=num(d.get('gnn_lr')),
                pat=d.get('patience'), wd=num(d.get('weight_decay')), es=d.get('early_stop_metric') or 'loss')

def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5 if len(xs) > 1 else 0.0
    return m, sd, len(xs)

def gather(root, ds, bb):
    """Return {cfgkey: {'cfg':dict, 'pearson':[...], 'auc':[...], 'rmse':[...], 'gtroc':[...]}} across folds."""
    g = defaultdict(lambda: dict(cfg=None, pearson=[], auc=[], rmse=[], gtroc=[]))
    for sm in glob.glob(os.path.join(root, ds, 'fold*', f'{bb}_*unk-fixed*', 'summary.json')):
        d = json.load(open(sm)); k = cfgkey(d); e = g[k]
        e['cfg'] = cfgdict(d)
        e['pearson'].append(num(d.get('pearson_motif_all')))
        e['auc'].append(num(d.get('auc')))
        e['rmse'].append(num(d.get('rmse_orig')))
        e['gtroc'].append(num(d.get('gt_roc_node_auc_mean_all')))
    return g

def which_knobs(cfg):
    return [k for k in KNOBS if not approx(cfg[k], CENTER[k])]

def main():
    tune_root, ds = sys.argv[1], sys.argv[2]
    prod_root = sys.argv[3] if len(sys.argv) > 3 else None

    g = gather(tune_root, ds, BACKBONE)
    if not g:
        print(f"no {BACKBONE} runs under {tune_root}/{ds}/fold*"); return

    # deployed baseline (fold-averaged)
    base_p = base_a = None
    if prod_root:
        pg = gather(prod_root, ds, BACKBONE)
        # deployed = the single production config; if several, take the one with most folds
        if pg:
            best = max(pg.values(), key=lambda e: len(e['pearson']))
            base_p = mean_std(best['pearson'])[0]
            base_a = mean_std(best['auc'])[0]

    # classify configs
    center = None
    sweeps = {k: [] for k in KNOBS}
    for k, e in g.items():
        dk = which_knobs(e['cfg'])
        rec = dict(cfg=e['cfg'],
                   pear=mean_std(e['pearson']), auc=mean_std(e['auc']),
                   rmse=mean_std(e['rmse']), gt=mean_std(e['gtroc']))
        if not dk:
            center = rec
        elif len(dk) == 1:
            sweeps[dk[0]].append(rec)

    guard_a = base_a if base_a is not None else (center['auc'][0] if center else None)

    def fmt(rec, knob=None):
        c = rec['cfg']; val = c[knob] if knob else 'center'
        pm, ps, pn = rec['pear']; am = rec['auc'][0]; rm = rec['rmse'][0]; gm = rec['gt'][0]
        pear = f"{pm:.3f}±{ps:.3f}" if pm is not None else '   -   '
        auc = f"{am:.3f}" if am is not None else '  -  '
        rmse = f"{rm:.3f}" if rm is not None else '  -  '
        gt = f"{gm:.3f}" if gm is not None else '  -  '
        flag = ''
        if guard_a is not None and am is not None and am < guard_a - AUC_GUARD:
            flag = ' AUC-COLLAPSE'
        return f"{str(val):>8} | {pear:>12} {auc:>6} {rmse:>6} {gt:>6}  n={pn}{flag}"

    print(f"\n################  {ds} — {BACKBONE} — OFAT sweep (fold-averaged, mean±std)  ################")
    if base_p is not None:
        print(f"DEPLOYED GAT baseline (fold-avg): pearson={base_p:.3f}  auc={base_a:.3f}")
    if center:
        cm = center['pear']; print(f"CENTER: pearson={cm[0]:.3f}±{cm[1]:.3f} (n={cm[2]})  auc={center['auc'][0]:.3f}")
    cp = center['pear'][0] if center else None

    summary = []
    for k in KNOBS:
        recs = sweeps[k]
        if not recs:
            continue
        recs = sorted(recs, key=lambda r: (r['cfg'][k] if isinstance(r['cfg'][k], (int, float)) else 9e9))
        print(f"\n--- {k}  (center={CENTER[k]}) ---")
        print(f"{'value':>8} | {'pearson':>12} {'auc':>6} {'rmse':>6} {'gtroc':>6}")
        for r in recs:
            print(fmt(r, k))
        if center:
            print(fmt(center) + '  <- center')
        # best AUC-SAFE config for this knob
        safe = [r for r in recs if r['auc'][0] is None or guard_a is None or r['auc'][0] >= guard_a - AUC_GUARD]
        cand = [r for r in safe if r['pear'][0] is not None]
        if cand and cp is not None:
            best = max(cand, key=lambda r: r['pear'][0])
            bv = best['cfg'][k]; dp = best['pear'][0] - cp
            summary.append((k, bv, dp, best['pear'][1], best['auc'][0]))

    if summary:
        print(f"\n============ {ds}: AUC-safe knob directions (fold-avg pearson) ============")
        print(f"{'knob':>6} {'best@':>10} {'Δpearson':>9} {'±std':>6} {'auc@best':>9}")
        for k, bv, dp, sd, au in sorted(summary, key=lambda x: -x[2]):
            print(f"{k:>6} {str(bv):>10} {dp:>+9.3f} {sd:>6.3f} {au if au is None else round(au,3):>9}")

if __name__ == '__main__':
    main()
