"""Canonical SMILES for a fragment written out of a parent molecule (2026-10-18 duplicate-key fix).

MolFragmentToSmiles / FragmentOnBonds write the fragment with RDKit's canonical atom order computed on
the PARENT molecule, so the order depends on parent context (degree, H count, a chiral tag on an atom
that is no longer a stereocentre). The same ring therefore got several keys, e.g. piperidine
'C1CCNCC1' vs 'C1CCCCN1' (the second from piperidinone rings whose C=O was cut off), and with
isomeric keys '*[C@H](*)C' vs '*[C@@H](*)C' once the attachment isotopes are cleared. Re-parsing the
string as a standalone molecule and writing it again makes the key depend on the fragment alone.
"""
from rdkit import Chem


def canonical_fragment_smiles(smi: str, isomeric: bool) -> str:
    """Re-parse `smi` as a standalone molecule and return its canonical SMILES.

    Ring systems cut out of a larger aromatic system may not kekulize (e.g. 'c1ccnc1' from an
    N-substituted pyrrole); those are parsed unsanitized, which is still canonical for the written
    graph (stable under atom reordering). A string RDKit cannot parse at all raises."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        m = Chem.MolFromSmiles(smi, sanitize=False)
        if m is None:
            raise ValueError(f'canonical_fragment_smiles: RDKit cannot parse fragment SMILES {smi!r}')
        m.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(m)
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=isomeric)
