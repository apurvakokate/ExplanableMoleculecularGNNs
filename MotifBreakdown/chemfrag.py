#!/usr/bin/env python3
"""chemfrag — per-algorithm partitions + RECAP best-path tree + sequential
cascade. NO MERGING. Each algorithm yields a complete, non-overlapping partition;
RECAP resolves its alternative splits by a max-min-size best-path selector. The
cascade RECAP->BRICS->rBRICS->Murcko(+sidechains)->structural refines within each
fragment, preserving non-overlap throughout."""
import re, json, argparse
from collections import Counter, defaultdict
from rdkit import Chem
from rdkit.Chem import BRICS, Draw, Recap
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
# BRICS / rBRICS bond discovery lives in the shared module so the v4 cascade and
# the legacy engine identify the same chemistry identically.
import brics_rbrics as BR

# Shared tautomer canonicalizer (created once; reused). Normalizes tautomers so
# the same compound in different tautomeric forms maps to ONE canonical key,
# closing the ~13% tautomer-ambiguity hole found in the canonicalization audit.
_TAUT = rdMolStandardize.TautomerEnumerator()

# ── canonicalization ──
def canon_mol(s):
    m = Chem.MolFromSmiles(str(s))
    if m is None: return None
    try: Chem.SanitizeMol(m)
    except Exception: return None
    # Best-effort tautomer normalization: map all tautomers to one canonical form.
    # Never drops a molecule — falls back to the plain canonical form on failure.
    try:
        mt = _TAUT.Canonicalize(m)
        if mt is not None: m = mt
    except Exception:
        pass
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)

def _strip(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return re.sub(r'\[\d+\*\]', '[*]', smi)
    for a in m.GetAtoms():
        if a.GetAtomicNum() == 0: a.SetIsotope(0); a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)

def frag_key(mol, atomset):
    aset=set(atomset); N=mol.GetNumAtoms()
    cross=[b.GetIdx() for b in mol.GetBonds() if (b.GetBeginAtomIdx() in aset)!=(b.GetEndAtomIdx() in aset)]
    if not cross: return _strip(Chem.MolToSmiles(mol, isomericSmiles=False))
    fr=Chem.FragmentOnBonds(mol,cross,addDummies=True)
    for pa,pm in zip(Chem.GetMolFrags(fr,asMols=False),Chem.GetMolFrags(fr,asMols=True,sanitizeFrags=False)):
        if {a for a in pa if a<N}==aset:
            try: Chem.SanitizeMol(pm)
            except: pass
            return _strip(Chem.MolToSmiles(pm, isomericSmiles=False))
    return None

# ── partition primitives ──
def comps_in(mol, atomset, bonds):
    broken={tuple(sorted((mol.GetBondWithIdx(i).GetBeginAtomIdx(),mol.GetBondWithIdx(i).GetEndAtomIdx()))) for i in bonds}
    adj=defaultdict(set)
    for b in mol.GetBonds():
        u,v=b.GetBeginAtomIdx(),b.GetEndAtomIdx()
        if u in atomset and v in atomset and tuple(sorted((u,v))) not in broken:
            adj[u].add(v); adj[v].add(u)
    seen=set(); out=[]
    for a in atomset:
        if a in seen: continue
        st=[a]; c=set()
        while st:
            x=st.pop()
            if x in seen: continue
            seen.add(x); c.add(x); st.extend(adj[x]-seen)
        out.append(frozenset(c))
    return out

def components_after_break(mol, bonds):
    return comps_in(mol, frozenset(range(mol.GetNumAtoms())), bonds)

def submol_dummies(mol, atomset):
    aset=set(atomset); N=mol.GetNumAtoms()
    cross=[b.GetIdx() for b in mol.GetBonds() if (b.GetBeginAtomIdx() in aset)!=(b.GetEndAtomIdx() in aset)]
    if not cross: return mol,{i:i for i in range(N)}
    fr=Chem.FragmentOnBonds(mol,cross,addDummies=True)
    for pa,pm in zip(Chem.GetMolFrags(fr,asMols=False),Chem.GetMolFrags(fr,asMols=True,sanitizeFrags=False)):
        if {a for a in pa if a<N}==aset:
            try: Chem.SanitizeMol(pm)
            except: pass
            return pm,{newi:pa[newi] for newi in range(len(pa))}
    return None,None

def _nonring_pairs_to_bonds(mol, pairs, within=None):
    # Thin alias kept for readability at call sites; shared implementation.
    return BR.nonring_bond_indices(mol, pairs, within)

# ── RECAP single-reaction splits + best-path tree ──
def recap_split_options(mol, atomset):
    """List of distinct single-reaction RECAP cut-bond sets within atomset."""
    sm,mp=submol_dummies(mol,atomset)
    if sm is None: return []
    N=mol.GetNumAtoms(); apps=set()
    for rxn in Recap.reactions:
        try: products=rxn.RunReactants((sm,))
        except Exception: continue
        for prodset in products:
            a2g={}
            for gi,p in enumerate(prodset):
                for a in p.GetAtoms():
                    if a.HasProp('react_atom_idx'): a2g[int(a.GetProp('react_atom_idx'))]=gi
            matched=set()
            for mt in sm.GetSubstructMatches(rxn.GetReactantTemplate(0)): matched.update(mt)
            cuts=set()
            for b in sm.GetBonds():
                if b.IsInRing(): continue
                u,v=b.GetBeginAtomIdx(),b.GetEndAtomIdx()
                if u not in matched or v not in matched: continue
                gu,gv=a2g.get(u),a2g.get(v)
                if (gu!=gv or gu is None) and (gu is not None or gv is not None):
                    ou,ov=mp.get(u),mp.get(v)
                    if ou is not None and ov is not None and ou<N and ov<N:
                        bd=mol.GetBondBetweenAtoms(ou,ov)
                        if bd is not None and not bd.IsInRing(): cuts.add(bd.GetIdx())
            if cuts: apps.add(frozenset(cuts))
    return list(apps)

_RECAP_FREQ = None   # None -> max-min size selector; dict(key->freq) -> frequency-consistent

def recap_best_split(mol, atomset):
    """Pick the single best split. Default objective: maximize min-fragment-size,
    tie-break by min size-difference, then smallest SMILES. If _RECAP_FREQ is set
    (frequency-consistent mode), the PRIMARY objective becomes: maximize the
    minimum global frequency of the resulting child fragments (so the same node
    is split toward globally-common, hence consistent, pieces), with size/SMILES
    as tie-breaks. Returns child atomsets or None."""
    best=None; best_key=None
    for cutset in recap_split_options(mol, atomset):
        kids=comps_in(mol, atomset, cutset)
        if len(kids)<2: continue
        sizes=sorted(len(k) for k in kids)
        smi_min=min((frag_key(mol,k) or '~') for k in kids)
        if _RECAP_FREQ is None:
            key=(sizes[0], -(sizes[-1]-sizes[0]))          # max min-size, min spread
        else:
            fmin=min(_RECAP_FREQ.get(frag_key(mol,k),0) for k in kids)
            key=(fmin, sizes[0], -(sizes[-1]-sizes[0]))    # max min-freq, then size
        if best_key is None or key>best_key or (key==best_key and smi_min<best_smi):
            best=kids; best_key=key; best_smi=smi_min
    return best

def recap_tree(mol, atomset):
    """Best-path RECAP hierarchy (recursive binary splits). Node: atomset, children."""
    kids=recap_best_split(mol, atomset)
    if not kids: return {'atomset':atomset,'children':[],'cut_by':None}
    return {'atomset':atomset,'cut_by':'recap',
            'children':[recap_tree(mol,k) for k in kids]}

def tree_leaves(n): return [n['atomset']] if not n['children'] else [x for c in n['children'] for x in tree_leaves(c)]

# ── the five independent partition algorithms ──
def part_recap(mol):
    return tree_leaves(recap_tree(mol, frozenset(range(mol.GetNumAtoms()))))

def part_brics(mol):
    return components_after_break(mol,_nonring_pairs_to_bonds(mol,BR.brics_bonds(mol)))

def part_rbrics(mol):
    # Full rBRICS = rBRICS environments + reBRICS long-chain breaks (shared with
    # the legacy engine's method='rbrics').
    return components_after_break(mol,_nonring_pairs_to_bonds(mol,BR.rbrics_full_bonds(mol)))

def part_murcko(mol):
    try: scaf=MurckoScaffold.GetScaffoldForMol(mol)
    except Exception: scaf=None
    if scaf is None or scaf.GetNumAtoms()==0:
        return [frozenset(range(mol.GetNumAtoms()))]
    match=mol.GetSubstructMatch(scaf)
    if not match:
        match=tuple(a.GetIdx() for a in mol.GetAtoms() if a.IsInRing())
        if not match: return [frozenset(range(mol.GetNumAtoms()))]
    sa=frozenset(match)
    cut=[b.GetIdx() for b in mol.GetBonds() if (b.GetBeginAtomIdx() in sa)!=(b.GetEndAtomIdx() in sa)]
    return components_after_break(mol,set(cut))

def part_structural(mol):
    idx=[]
    for b in mol.GetBonds():
        if b.IsInRing() or b.GetBondType()!=Chem.BondType.SINGLE: continue
        a1,a2=b.GetBeginAtom(),b.GetEndAtom()
        if a1.GetAtomicNum()==1 or a2.GetAtomicNum()==1: continue
        if a1.GetDegree()<2 or a2.GetDegree()<2: continue
        idx.append(b.GetIdx())
    return components_after_break(mol,set(idx))

ALGOS={'recap':part_recap,'brics':part_brics,'rbrics':part_rbrics,
       'murcko':part_murcko,'structural':part_structural}

# ── the cascade (stage cutters, applied in order, refining within fragments) ──
def _bonds_within(mol, atomset, pairs_fn):
    """Shared within-atomset helper: run a bond-discovery primitive on the
    dummy-capped submol, then map the cut atom pairs back to ORIGINAL non-ring
    bond indices. Used for both BRICS and rBRICS cascade stages."""
    sm,mp=submol_dummies(mol,atomset); N=mol.GetNumAtoms(); out=set()
    if sm is None: return out
    for (a,b) in pairs_fn(sm):
        oa,ob=mp.get(a),mp.get(b)
        if oa is None or ob is None or oa>=N or ob>=N: continue
        bd=mol.GetBondBetweenAtoms(oa,ob)
        if bd is not None and not bd.IsInRing(): out.add(bd.GetIdx())
    return out

def _brics_within(mol, atomset):
    return _bonds_within(mol, atomset, BR.brics_bonds)

def _rbrics_within(mol, atomset):
    # Full rBRICS = rBRICS environments + reBRICS long-chain breaks, the SAME
    # definition used by the legacy engine's method='rbrics'.
    return _bonds_within(mol, atomset, BR.rbrics_full_bonds)

def _murcko_within(mol, atomset):
    ring=[a for a in atomset if mol.GetAtomWithIdx(a).IsInRing()]
    if not ring or len(ring)==len(atomset): return set()
    sa=set(ring)
    # cut bonds between ring(scaffold) and non-ring(sidechain), but ONLY single,
    # non-ring bonds where both atoms have degree>=2 (never sever a double bond
    # like a carbonyl C=O, never cut off a terminal heteroatom).
    cut=set()
    for b in mol.GetBonds():
        if b.IsInRing() or b.GetBondType()!=Chem.BondType.SINGLE: continue
        u,v=b.GetBeginAtomIdx(),b.GetEndAtomIdx()
        if u not in atomset or v not in atomset: continue
        if (u in sa)==(v in sa): continue
        if b.GetBeginAtom().GetDegree()<2 or b.GetEndAtom().GetDegree()<2: continue
        cut.add(b.GetIdx())
    return cut

def _structural_within(mol, atomset):
    cut=set()
    for b in mol.GetBonds():
        if b.IsInRing() or b.GetBondType()!=Chem.BondType.SINGLE: continue
        u,v=b.GetBeginAtomIdx(),b.GetEndAtomIdx()
        if u not in atomset or v not in atomset: continue
        a1,a2=b.GetBeginAtom(),b.GetEndAtom()
        if a1.GetDegree()<2 or a2.GetDegree()<2: continue
        cut.add(b.GetIdx())
    return cut

def _stage_split(stage, mol, atomset):
    if stage=='recap': return recap_best_split(mol, atomset)
    if stage=='brics': c=_brics_within(mol,atomset)
    elif stage=='rbrics': c=_rbrics_within(mol,atomset)
    elif stage=='murcko': c=_murcko_within(mol,atomset)
    else: c=_structural_within(mol,atomset)
    if not c: return None
    kids=comps_in(mol,atomset,c)
    return kids if len(kids)>=2 else None

CASCADE=['recap','brics','rbrics','murcko','structural']

def cascade_tree(mol, atomset=None, stage=0):
    if atomset is None: atomset=frozenset(range(mol.GetNumAtoms()))
    if stage>=len(CASCADE): return {'atomset':atomset,'children':[],'cut_by':None}
    kids=_stage_split(CASCADE[stage], mol, atomset)
    if not kids:
        return cascade_tree(mol, atomset, stage+1)   # this stage can't cut; advance
    return {'atomset':atomset,'cut_by':CASCADE[stage],
            'children':[cascade_tree(mol,k,stage) for k in kids]}  # exhaust stage, then advance

# ── verification ──
def verify_partition(mol, parts):
    N=mol.GetNumAtoms()
    cov=set().union(*parts) if parts else set()
    ok=cov==set(range(N)) and sum(len(p) for p in parts)==N
    par=ref=True
    for p in parts:
        k=frag_key(mol,p); fm=Chem.MolFromSmiles(k) if k else None
        if fm is None: par=ref=False; continue
        try: BRICS.FindBRICSBonds(fm)
        except Exception: ref=False
    return ok,par,ref

# ── driver ──
def run(csv, smiles_col='smiles', cap=None, do_cascade=True):
    import pandas as pd
    df=pd.read_csv(csv); smis=[str(s) for s in df[smiles_col]]
    if cap: smis=smis[:cap]
    records=[]; vstats={a:Counter() for a in ALGOS}
    motif={a:Counter() for a in ALGOS}; disagree=Counter(); ptot=Counter()
    for smi in smis:
        cs=canon_mol(smi)
        if cs is None: continue
        mol=Chem.MolFromSmiles(cs)
        parts={}; keys={}
        for a,fn in ALGOS.items():
            try: p=fn(mol)
            except Exception: p=[frozenset(range(mol.GetNumAtoms()))]
            ok,par,ref=verify_partition(mol,p)
            vstats[a]['total']+=1; vstats[a]['partition']+=ok; vstats[a]['parse']+=par; vstats[a]['refrag']+=ref
            parts[a]=p; keys[a]=[frag_key(mol,fr) for fr in p]
            for k in keys[a]: motif[a][k]+=1
        for A in ALGOS:
            for B in ALGOS:
                if A==B: continue
                for fa in parts[A]:
                    ptot[(A,B)]+=1
                    if not any(fa<=fb for fb in parts[B]): disagree[(A,B)]+=1
        rec={'smiles':cs,'mol':mol,'parts':parts,'keys':keys}
        if do_cascade: rec['cascade']=cascade_tree(mol)
        records.append(rec)
    per_algo={a:{'vocab':len(motif[a]),
                 'motifs':[{'key':k,'occ':c} for k,c in motif[a].most_common()],
                 'verification':dict(vstats[a])} for a in ALGOS}
    disagreement=[{'A':A,'B':B,'split':disagree[(A,B)],'total':ptot[(A,B)],
                   'pct':round(100*disagree[(A,B)]/max(1,ptot[(A,B)]),1)} for (A,B) in ptot]
    return records,{'n':len(records),'per_algo':per_algo,'disagreement':disagreement}

def svg_for_key(key, size=120):
    m=Chem.MolFromSmiles(key)
    if m is None: return ''
    d=Draw.rdMolDraw2D.MolDraw2DSVG(size,size); d.drawOptions().clearBackground=False
    try: d.DrawMolecule(m); d.FinishDrawing(); return d.GetDrawingText()
    except Exception: return ''

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True); ap.add_argument('--cap',type=int,default=None)
    a=ap.parse_args()
    recs,an=run(a.csv,cap=a.cap)
    print(f"molecules: {an['n']}")
    print(f"{'algo':>11} {'vocab':>7} {'partition':>11} {'refrag':>9}")
    for alg,d in an['per_algo'].items():
        v=d['verification']; print(f"{alg:>11} {d['vocab']:>7} {v['partition']:>5}/{v['total']:<5} {v['refrag']:>5}/{v['total']:<5}")
    print("\ncross-algorithm disagreement (% of A's frags B splits):")
    for r in sorted(an['disagreement'],key=lambda x:-x['pct'])[:12]:
        print(f"  {r['A']:>10} vs {r['B']:<10} {r['pct']:>5}%")
