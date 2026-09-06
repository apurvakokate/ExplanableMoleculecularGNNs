#!/usr/bin/env python3
"""base_status.py — audit + targeted reset for the MotifSAT base-runs campaign.

Run ON THE HPC (needs the output filesystem and `squeue`).

Cell states (authoritative — computed by SCANNING every expected cell, not by
trusting a log, so preempted/orphaned runs are caught):
  VALID     native_complete.py passes                          -> leave alone
  RUNNING   incomplete, claimed, owning SLURM job is ALIVE      -> leave alone
  FAILED    incomplete, claim has a .failed marker             -> reset target
  ORPHANED  incomplete, claimed, owning job is DEAD (preempt)  -> reset target
  MISSING   incomplete, no claim                               -> pull picks it up

Usage:
  base_status.py --report                       # counts + incomplete cells w/ reasons
  base_status.py --report --tsv                 # machine-readable incomplete rows
  base_status.py --reset                        # DRY-RUN: show what reset would clear
  base_status.py --reset --apply                # actually clear claims + partial dirs
  base_status.py --reset --apply --rc 137 --dataset hERG   # target a slice (e.g. OOM)

Reset removes the claim dir AND the partial output dir (so a re-launch reruns clean,
no stale merge into summary_splits.json / explainer_importances.json). Only paths
under <out_root>/base_runs and <out_root>/_dispatch/claims are ever touched.
"""
import os
import sys
import csv
import shutil
import argparse
import subprocess
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import native_complete as nc  # noqa: E402

DEFAULT_OUT = '/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/motifsat_paper_deliverables_v1'
RESET_STATES_DEFAULT = 'FAILED,ORPHANED'


def live_jobids():
    """Set of this user's currently-queued/running SLURM job ids."""
    try:
        out = subprocess.run(
            ['squeue', '-u', os.environ.get('USER', ''), '-h', '-o', '%A'],
            capture_output=True, text=True, timeout=30).stdout
        return {x.strip() for x in out.split() if x.strip()}
    except Exception as e:
        print(f'[warn] squeue failed ({e}); treating all claims as possibly-live '
              f'(no ORPHANED detection).', file=sys.stderr)
        return None  # None => cannot tell; be conservative (never mark ORPHANED)


def claim_jobid(claim_dir):
    try:
        return open(os.path.join(claim_dir, 'info')).read().split('\t')[1].strip()
    except Exception:
        return None


def classify(out_root, live, runs_name='base_runs', dispatch_name='_dispatch'):
    runs = os.path.join(out_root, runs_name)
    claims = os.path.join(out_root, dispatch_name, 'claims')
    rows = []
    for cid, preset, stem, ds, fold, bb, rel in nc.iter_cells():
        d = os.path.join(runs, rel)
        ok, reason = nc.is_complete(d, stem)
        cdir = os.path.join(claims, cid)
        claimed = os.path.isdir(cdir)
        failed = claimed and os.path.exists(os.path.join(cdir, '.failed'))
        if ok:
            state, reason = 'VALID', ''
        elif failed:
            state = 'FAILED'
        elif claimed:
            jid = claim_jobid(cdir)
            if live is None:
                state = 'RUNNING'            # cannot verify liveness -> conservative
            else:
                state = 'RUNNING' if (jid and jid in live) else 'ORPHANED'
        else:
            state = 'MISSING'
        rows.append(dict(cell_id=cid, preset=preset, stem=stem, dataset=ds,
                         fold=str(fold), backbone=bb, state=state, reason=reason,
                         run_dir=d, claim_dir=cdir))
    return rows


def load_last_rc(out_root, dispatch_name='_dispatch'):
    rc = {}
    p = os.path.join(out_root, dispatch_name, 'failures.tsv')
    if os.path.exists(p):
        for r in csv.DictReader(open(p), delimiter='\t'):
            rc[r['cell_id']] = r.get('rc', '')   # last occurrence wins
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out_root', default=os.environ.get('MOTIFSAT_DELIV_OUT', DEFAULT_OUT))
    ap.add_argument('--runs', default='base_runs',
                    help='run tree under out_root (base_runs | sfo_runs | ...)')
    ap.add_argument('--dispatch', default='_dispatch',
                    help='dispatch dir under out_root (_dispatch | _dispatch_sfo | ...)')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--reset', action='store_true')
    ap.add_argument('--apply', action='store_true', help='execute the reset (default: dry-run)')
    ap.add_argument('--states', default=RESET_STATES_DEFAULT,
                    help=f'comma states eligible for reset (default {RESET_STATES_DEFAULT})')
    ap.add_argument('--tsv', action='store_true', help='report incomplete cells as TSV')
    # slice filters (apply to both report and reset)
    ap.add_argument('--preset'); ap.add_argument('--dataset')
    ap.add_argument('--backbone'); ap.add_argument('--fold'); ap.add_argument('--rc')
    a = ap.parse_args()

    live = live_jobids()
    rows = classify(a.out_root, live, a.runs, a.dispatch)
    rc_map = load_last_rc(a.out_root, a.dispatch)

    def match(r):
        if a.preset and r['preset'] != a.preset: return False
        if a.dataset and r['dataset'] != a.dataset: return False
        if a.backbone and r['backbone'] != a.backbone: return False
        if a.fold and r['fold'] != str(a.fold): return False
        if a.rc and rc_map.get(r['cell_id']) != str(a.rc): return False
        return True

    sel = [r for r in rows if match(r)]

    if a.reset:
        states = {s.strip() for s in a.states.split(',') if s.strip()}
        targets = [r for r in sel if r['state'] in states]
        print(f'reset: {len(targets)} cells eligible (states={sorted(states)}, apply={a.apply})')
        base_runs = os.path.join(a.out_root, a.runs)
        base_claims = os.path.join(a.out_root, a.dispatch, 'claims')
        for r in targets:
            tag = 'DELETE' if a.apply else 'would delete'
            print(f'  {tag} {r["cell_id"]} [{r["state"]}] rc={rc_map.get(r["cell_id"], "-")}')
            if not a.apply:
                continue
            d, c = r['run_dir'], r['claim_dir']
            if os.path.isdir(d) and os.path.abspath(d).startswith(os.path.abspath(base_runs)):
                shutil.rmtree(d, ignore_errors=True)
            if os.path.isdir(c) and os.path.abspath(c).startswith(os.path.abspath(base_claims)):
                shutil.rmtree(c, ignore_errors=True)
        if not a.apply:
            print('  (dry-run — add --apply to execute)')
        return 0

    # default: report
    c = Counter(r['state'] for r in sel)
    if a.tsv:
        print('cell_id\tpreset\tstem\tdataset\tfold\tbackbone\tstate\trc\treason')
        for r in sel:
            if r['state'] != 'VALID':
                print('\t'.join([r['cell_id'], r['preset'], r['stem'], r['dataset'],
                                 r['fold'], r['backbone'], r['state'],
                                 rc_map.get(r['cell_id'], ''), r['reason']]))
        return 0
    print(f'=== base-runs status: {len(sel)} cells (out_root={a.out_root}) ===')
    for st in ('VALID', 'RUNNING', 'FAILED', 'ORPHANED', 'MISSING'):
        print(f'  {st:9} {c.get(st, 0)}')
    inc = [r for r in sel if r['state'] != 'VALID']
    if inc:
        print(f'--- {len(inc)} incomplete (first 40) ---')
        for r in inc[:40]:
            print(f'  {r["state"]:9} {r["cell_id"]}  rc={rc_map.get(r["cell_id"], "-")}  {r["reason"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
