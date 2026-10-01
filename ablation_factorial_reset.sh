#!/usr/bin/env bash
# ablation_factorial_reset.sh — clear ORPHANED claim dirs so a relaunch can re-run them.
# An orphan = a claim dir whose cell has NO summary_splits.json and NO .failed marker
# (its worker was preempted/killed mid-run). The claim dir otherwise blocks re-running.
#
# SAFE BY DEFAULT: prints what it would remove and deletes NOTHING. Set APPLY=1 to delete.
# AGE GUARD: only considers claims whose info is older than AGE_MIN (default 20) minutes AND
# whose target run dir has had NO file modified in the last AGE_MIN minutes — so a cell that
# is still being trained by a live worker is NEVER touched.
#
# Usage:
#   bash ablation_factorial_reset.sh              # dry run: list orphans
#   APPLY=1 bash ablation_factorial_reset.sh      # remove the listed orphan claim dirs
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
D=$REPO/ablated_completely_v1/_dispatch_fac; B=$REPO/ablated_completely_v1/mose/rbrics_filter
AGE_MIN="${AGE_MIN:-20}"; APPLY="${APPLY:-0}"
[ -d "$D/claims" ] || { echo "no claims dir at $D/claims"; exit 0; }
python3 - "$B" "$D" "$AGE_MIN" "$APPLY" <<'PYEOF'
import glob,os,sys,time,shutil
B,D,age_min,apply = sys.argv[1], sys.argv[2], int(sys.argv[3]), (sys.argv[4]=="1")
now=time.time(); age_s=age_min*60
def poolsfx(p): return "_pool-mean" if p=="mean" else ""
claims=sorted(os.listdir(D+"/claims"))
orphans=[]; inflight=0; done=0; failed=0
for c in claims:
    if not c.startswith("mosefac__"): continue
    p=c.split("__")
    if len(p)!=8: continue
    _,ds,fold,bb,enc,norm,pool,unk=p; fold=fold[1:]
    cdir=f"{D}/claims/{c}"
    tag=f"{bb}_{enc}_norm-{norm}{poolsfx(pool)}_wf+wr_unk-{unk}_real_ep500_rbrics_filter*"
    hits=glob.glob(f"{B}/{ds}/fold{fold}/{tag}")
    if any(os.path.exists(h+"/summary_splits.json") for h in hits): done+=1; continue
    if os.path.exists(cdir+"/.failed"): failed+=1; continue
    # age guard on the claim marker
    info=cdir+"/info"
    claim_mtime=os.path.getmtime(info) if os.path.exists(info) else os.path.getmtime(cdir)
    if now-claim_mtime < age_s: inflight+=1; continue
    # age guard on any partial output in the target dir
    recent=False
    for h in hits:
        for root,_,files in os.walk(h):
            for f in files:
                if now-os.path.getmtime(os.path.join(root,f)) < age_s: recent=True; break
            if recent: break
        if recent: break
    if recent: inflight+=1; continue
    orphans.append(c)
print(f"claims={len(claims)} done={done} failed-marked={failed} protected(in-flight/<{age_min}min)={inflight} ORPHANS={len(orphans)}")
from collections import Counter
print("orphans by dataset:", dict(Counter(c.split('__')[1] for c in orphans)))
for c in orphans: print(("[DELETE] " if apply else "[dry] ")+c)
if apply:
    n=0
    for c in orphans:
        shutil.rmtree(f"{D}/claims/{c}", ignore_errors=True); n+=1
    print(f"removed {n} orphan claim dirs. Now relaunch workers to re-run them.")
else:
    print("DRY RUN — nothing deleted. Re-run with APPLY=1 to remove the above.")
PYEOF
