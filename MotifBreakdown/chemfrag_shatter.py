"""Maximal-floor cascade variants for the over-fragmentation experiment.
Both keep rings whole (Guard 1: IsInRing) and protect double/triple bonds
(Guard 1: SINGLE only). Only the terminal-atom degree guard (Guard 2) is removed.

  MILD : v4 cascade as-is, but the trailing structural stage drops Guard 2
         (shatters only acyclic leftovers the chemistry stages didn't cut).
  FULL : after the normal v4 cascade, additionally cut EVERY acyclic single
         bond (Guard 2 removed) on each leaf -> maximal acyclic floor regardless
         of what chemistry fired.
Both feed the SAME v4 MDL merge afterward.
"""
import chemfrag as C
from rdkit import Chem

def _structural_noguard(mol, atomset):
    """Acyclic single-bond cut with the terminal-atom guard REMOVED.
    Rings + multiple bonds still protected."""
    cut=set()
    for b in mol.GetBonds():
        if b.IsInRing() or b.GetBondType()!=Chem.BondType.SINGLE: continue
        u,v=b.GetBeginAtomIdx(),b.GetEndAtomIdx()
        if u not in atomset or v not in atomset: continue
        cut.add(b.GetIdx())
    return cut

# ---------- MILD: monkeypatch the trailing structural stage ----------
def cascade_tree_mild(mol, atomset=None, stage=0):
    """Same as C.cascade_tree but the 'structural' stage uses the no-guard cut."""
    if atomset is None: atomset=frozenset(range(mol.GetNumAtoms()))
    if stage>=len(C.CASCADE): return {'atomset':atomset,'children':[],'cut_by':None}
    name=C.CASCADE[stage]
    if name=='structural':
        c=_structural_noguard(mol,atomset)
        kids=C.comps_in(mol,atomset,c) if c else None
        kids=kids if (kids and len(kids)>=2) else None
    else:
        kids=C._stage_split(name, mol, atomset)
    if not kids:
        return cascade_tree_mild(mol, atomset, stage+1)
    return {'atomset':atomset,'cut_by':name,
            'children':[cascade_tree_mild(mol,k,stage) for k in kids]}

# ---------- FULL: normal cascade, then shatter every acyclic single bond ----------
def cascade_tree_full(mol, atomset=None):
    """Run the standard v4 cascade, then push EVERY leaf down to the acyclic
    floor by cutting all remaining acyclic single bonds (no terminal guard)."""
    base=C.cascade_tree(mol)
    def shatter(node):
        if node['children']:
            for ch in node['children']: shatter(ch)
            return
        # leaf: cut all acyclic single bonds within it (no guard)
        c=_structural_noguard(mol, node['atomset'])
        if not c: return
        kids=C.comps_in(mol, node['atomset'], c)
        if len(kids)<2: return
        # recurse: keep shattering until no acyclic single bond remains
        node['children']=[{'atomset':k,'children':[],'cut_by':'shatter'} for k in kids]
        for ch in node['children']: shatter(ch)
    shatter(base)
    return base
