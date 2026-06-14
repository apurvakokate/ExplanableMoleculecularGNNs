#!/usr/bin/env python3
"""Merge stage v4: SEPARATE learning from application.

PHASE 1 (learn): global MDL agglomeration over the whole forest discovers and
ranks merge rules. A rule is keyed on (parent_key, sorted child_keys). Output is
a frozen, priority-ordered RULEBOOK — no tokenization is committed here.

PHASE 2 (apply): each molecule is tokenized INDEPENDENTLY by a deterministic
bottom-up tree-rewrite: collapse a cascade-tree node iff its (parent,children)
rule is in the rulebook. Because every node has exactly one parent and the
rulebook is fixed, a node's fate depends ONLY on its own subtree + the rulebook,
never on cross-molecule frontier state. => same chemistry tokenizes identically
everywhere (consistency by construction).

No chemical guard, no potential frequency."""
import math, argparse
from collections import Counter, defaultdict
import chemfrag as C
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')

def all_nodes(node):
    yield node
    for c in node['children']:
        yield from all_nodes(c)

_ATOM={}
def natoms_key(k):
    if k not in _ATOM:
        m=Chem.MolFromSmiles(k); _ATOM[k]=sum(1 for a in m.GetAtoms() if a.GetAtomicNum()>0) if m else 1
    return _ATOM[k]

def mdl_L(counts):
    tot=sum(counts.values())
    if tot==0: return 0.0
    V=list(counts)
    model=sum(natoms_key(m)*math.log2(20) for m in V)+len(V)*math.log2(max(2,len(V)))
    data=sum(c*(-math.log2(c/tot)) for c in counts.values() if c>0)
    return model+data

def build_forest(smis):
    forest=[]
    for smi in smis:
        cs=C.canon_mol(smi)
        if cs is None: continue
        mol=Chem.MolFromSmiles(cs); tree=C.cascade_tree(mol)
        for nd in all_nodes(tree):
            nd['key']=C.frag_key(mol,nd['atomset'])
        forest.append((mol,tree))
    return forest

# ── PHASE 1: learn the rulebook (global MDL; identical mechanics to before, but
#    we record the ORDER of accepted rules and do NOT keep the tokenization) ──
def learn_rulebook(forest, verbose=False):
    NB={}; active={}; parent={}; child_ids={}
    for mi,(mol,tree) in enumerate(forest):
        active[mi]=set()
        for nd in all_nodes(tree):
            NB[(mi,id(nd))]=nd; child_ids[(mi,id(nd))]=[id(c) for c in nd['children']]
            for c in nd['children']: parent[(mi,id(c))]=id(nd)
            if not nd['children']: active[mi].add(id(nd))
    def counts_now():
        c=Counter()
        for mi,ids in active.items():
            for nid in ids: c[NB[(mi,nid)]['key']]+=1
        return c
    def candidates():
        rules=defaultdict(list)
        for mi,ids in active.items():
            checked=set()
            for nid in ids:
                p=parent.get((mi,nid))
                if p is None or p in checked: continue
                checked.add(p)
                ch=child_ids[(mi,p)]
                if ch and all(c in ids for c in ch):
                    sig=(NB[(mi,p)]['key'], tuple(sorted(NB[(mi,c)]['key'] for c in ch)))
                    rules[sig].append((mi,p))
        return rules
    cur=counts_now(); L=mdl_L(cur); rulebook=[]
    while True:
        rules=candidates()
        if not rules: break
        best=None; best_dL=1e-9; best_tie=None; best_sig=None; best_trial=None
        for sig,occ in rules.items():
            pkey,ckeys=sig
            dcount=Counter()
            for ck in ckeys: dcount[ck]-=len(occ)
            dcount[pkey]+=len(occ)
            trial=cur.copy(); trial.update(dcount); trial=Counter({k:v for k,v in trial.items() if v>0})
            dL=mdl_L(trial)-L
            if dL>1e-9: continue
            tie=(natoms_key(pkey), pkey, ckeys)  # total order → run-to-run determinism
            better=(dL<best_dL-1e-12) or (abs(dL-best_dL)<=1e-9 and best_tie is not None and
                    (tie[0]>best_tie[0] or (tie[0]==best_tie[0] and tie[1]<best_tie[1]) or
                     (tie[0]==best_tie[0] and tie[1]==best_tie[1] and tie[2]<best_tie[2])))
            if best_sig is None or better:
                best=occ; best_dL=dL; best_tie=tie; best_sig=sig; best_trial=trial
        if best_sig is None: break
        for (mi,pid) in best:
            for c in child_ids[(mi,pid)]: active[mi].discard(c)
            active[mi].add(pid)
        cur=best_trial; L=L+best_dL
        rulebook.append(best_sig)           # record rule in discovery (priority) order
        if verbose and len(rulebook)%50==0: print(f"  learned {len(rulebook)} rules")
    return rulebook

# ── PHASE 2: apply the frozen rulebook to each molecule independently ──
def apply_rulebook(mol, tree, ruleset):
    """Deterministic bottom-up tree-rewrite. Collapse a node iff its
    (parent,children) signature is in ruleset. Pure function of (tree, ruleset).
    Returns list of (key, atomset) tokens."""
    # ensure every node has a 'key' (self-contained: compute from mol if absent,
    # so the function does not silently depend on build_forest having run).
    for nd in all_nodes(tree):
        if 'key' not in nd: nd['key']=C.frag_key(mol, nd['atomset'])
    # active = set of node ids currently tokenizing; start at leaves
    active=set(); NB={}; child_ids={}
    for nd in all_nodes(tree):
        NB[id(nd)]=nd; child_ids[id(nd)]=[id(c) for c in nd['children']]
        if not nd['children']: active.add(id(nd))
    # process nodes bottom-up (post-order): a node may collapse once all its
    # children are active (i.e. they themselves already resolved). Iterate to
    # fixpoint; deterministic because it depends only on tree+ruleset.
    changed=True
    while changed:
        changed=False
        for nd in all_nodes(tree):
            nid=id(nd); ch=child_ids[nid]
            if not ch or nid in active: continue
            if all(c in active for c in ch):
                sig=(NB[nid]['key'], tuple(sorted(NB[c]['key'] for c in ch)))
                if sig in ruleset:
                    for c in ch: active.discard(c)
                    active.add(nid); changed=True
    return [(NB[d]['key'], NB[d]['atomset']) for d in active]

def merge(forest, verbose=False):
    """v4: learn rulebook globally, then apply deterministically per tree."""
    rulebook=learn_rulebook(forest, verbose=verbose)
    ruleset=set(rulebook)
    permol=[]; vocab=Counter()
    for mol,tree in forest:
        toks=apply_rulebook(mol,tree,ruleset)
        permol.append((mol,toks))
        for k,_ in toks: vocab[k]+=1
    return permol, rulebook, vocab

def merge_v3(forest, verbose=False):
    """v3: single-pass global rewrite — each accepted rule is applied to the live
    frontier as it is learned (learning and application conflated). Kept for
    comparison against v4."""
    NB={}; active={}; parent={}; child_ids={}
    for mi,(mol,tree) in enumerate(forest):
        active[mi]=set()
        for nd in all_nodes(tree):
            NB[(mi,id(nd))]=nd; child_ids[(mi,id(nd))]=[id(c) for c in nd['children']]
            for c in nd['children']: parent[(mi,id(c))]=id(nd)
            if not nd['children']: active[mi].add(id(nd))
    def counts_now():
        c=Counter()
        for mi,ids in active.items():
            for nid in ids: c[NB[(mi,nid)]['key']]+=1
        return c
    def candidates():
        rules=defaultdict(list)
        for mi,ids in active.items():
            checked=set()
            for nid in ids:
                p=parent.get((mi,nid))
                if p is None or p in checked: continue
                checked.add(p)
                ch=child_ids[(mi,p)]
                if ch and all(c in ids for c in ch):
                    sig=(NB[(mi,p)]['key'], tuple(sorted(NB[(mi,c)]['key'] for c in ch)))
                    rules[sig].append((mi,p))
        return rules
    cur=counts_now(); L=mdl_L(cur); rulebook=[]
    while True:
        rules=candidates()
        if not rules: break
        best=None; best_dL=1e-9; best_tie=None; best_sig=None; best_trial=None
        for sig,occ in rules.items():
            pkey,ckeys=sig
            dcount=Counter()
            for ck in ckeys: dcount[ck]-=len(occ)
            dcount[pkey]+=len(occ)
            trial=cur.copy(); trial.update(dcount); trial=Counter({k:v for k,v in trial.items() if v>0})
            dL=mdl_L(trial)-L
            if dL>1e-9: continue
            tie=(natoms_key(pkey), pkey)
            better=(dL<best_dL-1e-12) or (abs(dL-best_dL)<=1e-9 and best_tie is not None and
                    (tie[0]>best_tie[0] or (tie[0]==best_tie[0] and tie[1]<best_tie[1])))
            if best_sig is None or better:
                best=occ; best_dL=dL; best_tie=tie; best_sig=sig; best_trial=trial
        if best_sig is None: break
        for (mi,pid) in best:
            for c in child_ids[(mi,pid)]: active[mi].discard(c)
            active[mi].add(pid)
        cur=best_trial; L=L+best_dL; rulebook.append(best_sig)
    permol=[]; vocab=Counter()
    for mi,(mol,tree) in enumerate(forest):
        toks=[(NB[(mi,d)]['key'],NB[(mi,d)]['atomset']) for d in active[mi]]
        permol.append((mol,toks))
        for k,_ in toks: vocab[k]+=1
    return permol, rulebook, vocab

def run(csv, cap=None, verbose=False):
    import pandas as pd
    df=pd.read_csv(csv); smis=[str(s) for s in df['smiles']]
    if cap: smis=smis[:cap]
    forest=build_forest(smis)
    return (forest,)+merge(forest,verbose=verbose)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True); ap.add_argument('--cap',type=int,default=None)
    a=ap.parse_args()
    forest,permol,rulebook,vocab=run(a.csv,cap=a.cap,verbose=True)
    print(f"\nmolecules={len(permol)} rules={len(rulebook)} vocab={len(vocab)}")
    okp=sum(1 for mol,toks in permol if set().union(*[s for _,s in toks])==set(range(mol.GetNumAtoms())) and sum(len(s) for _,s in toks)==mol.GetNumAtoms())
    print(f"atom conservation: {okp}/{len(permol)}")
