"""chemfrag_v4_adapter.py — v4 cascade + MDL-merge tokenizer adapter.

Replaces the legacy fragmentation engine (molfragbpe5.py: first-match-wins
cascade + prevalence BPE) used by generate_vocab_rules.py.

The ONLY contract the downstream vocab build depends on is, per molecule:
        List[Tuple[str, Set[int]]]      # (fragment_smarts, {original_atom_idx})
covering every atom exactly once, with atom indices in the order of
Chem.MolFromSmiles(orig_csv_smiles)  — because the GNN DataLoader re-parses the
same SMILES and indexes atoms identically.

v4 differs from the legacy engine on every axis that matters (see V4_VS_LEGACY.md):
  * cascade RECAP->BRICS->rBRICS->Murcko->structural is SEQUENTIAL/refining
    (each stage refines within the previous stage's fragments) instead of
    first-match-wins (legacy stops at the first algorithm that cuts);
  * RECAP is the canonical non-overlapping best-path tree (max-min split),
    not a hand-written SMARTS list;
  * the merge is a global MDL agglomeration with SEPARATED learn/apply
    (learn one frozen rulebook over the corpus; apply it per-molecule as a
    deterministic tree rewrite) keyed on (parent, children) — consistency by
    construction, instead of prevalence BPE which tiles greedily and can label
    identical chemistry differently across molecules;
  * tautomer canonicalization is folded into the motif key.

Atom-index note: v4's chemfrag.canon_mol() applies tautomer canonicalization and
REORDERS atoms, so we DO NOT use it for indices. We fragment the RAW mol
(Chem.MolFromSmiles(orig_smi)) so atomsets are raw indices; canonicalization is
used only inside frag_key() to produce a consistent motif string per fragment.
"""
from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')

import chemfrag as C
import merge as M
import chemfrag_shatter as _S

# ── cascade selector ─────────────────────────────────────────────────────────
# SHATTER=False -> standard v4 cascade (chemistry-floored).
# SHATTER=True  -> mild-shatter floor: the trailing structural stage drops the
#   terminal-atom guard, so every acyclic single bond is cut (rings + double
#   bonds still protected), giving the MDL merge a finer floor to rebuild from.
#   Measured to lower vocabulary ~20% and match-or-improve env-consistency.
SHATTER = False

def _tree(mol, shatter=None):
    # Explicit `shatter` arg takes precedence; None falls back to the module
    # default SHATTER. Passing it explicitly avoids relying on global state in
    # long-lived / programmatic multi-run processes (the CLI still uses the
    # global, set once from --shatter).
    use = SHATTER if shatter is None else shatter
    return _S.cascade_tree_mild(mol) if use else C.cascade_tree(mol)


# ── PHASE 1: learn the MDL rulebook once over the whole corpus ──────────────
def learn_corpus_rulebook(smiles_list: List[str],
                          use_merge: bool = True,
                          verbose: bool = False,
                          shatter: bool = None
                          ) -> Tuple[set, Dict[str, tuple]]:
    """Fragment every (raw) molecule into its cascade tree and learn the global
    MDL rulebook. Returns (ruleset, index) where index maps
    raw_smiles -> (raw_mol, cascade_tree) so PHASE 2 can tokenize without
    re-fragmenting. If use_merge is False, an empty ruleset is returned (the
    tokenization is then exactly the cascade leaves)."""
    forest = []
    index: Dict[str, tuple] = {}
    for smi in smiles_list:
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            continue
        try:
            Chem.SanitizeMol(m)
        except Exception:
            continue
        tree = _tree(m, shatter)
        for nd in M.all_nodes(tree):
            nd['key'] = C.frag_key(m, nd['atomset'])
        forest.append((m, tree))
        index[str(smi)] = (m, tree)
    ruleset = set(M.learn_rulebook(forest, verbose=verbose)) if use_merge else set()
    return ruleset, index


# ── PHASE 2: tokenize ONE molecule deterministically ────────────────────────
def fragment_tracked_v4(orig_smi: str,
                        ruleset: set,
                        index: Optional[Dict[str, tuple]] = None,
                        shatter: bool = None
                        ) -> List[Tuple[str, Set[int]]]:
    """Return [(fragment_smarts, {original_atom_indices})] for one molecule,
    covering every atom exactly once (raw atom-index order).

    `ruleset` is the frozen MDL rulebook from learn_corpus_rulebook. `index`
    (optional) lets us reuse the already-built cascade tree; if the molecule is
    not in the index it is fragmented on the fly and the same ruleset applied."""
    entry = None if index is None else index.get(str(orig_smi))
    if entry is None:
        m = Chem.MolFromSmiles(str(orig_smi))
        if m is None:
            raise ValueError(f"Invalid SMILES for v4 fragmentation: {orig_smi!r}")
        try:
            Chem.SanitizeMol(m)
        except Exception as exc:
            raise ValueError(
                f"Unsanitizable SMILES for v4 fragmentation: {orig_smi!r}"
            ) from exc
        tree = _tree(m, shatter)
        for nd in M.all_nodes(tree):
            nd['key'] = C.frag_key(m, nd['atomset'])
    else:
        m, tree = entry

    n = m.GetNumAtoms()
    tokens = M.apply_rulebook(m, tree, ruleset)        # [(key, atomset)]
    out: List[Tuple[str, Set[int]]] = []
    covered: Set[int] = set()
    overlap: Set[int] = set()
    for key, atomset in tokens:
        atoms = {int(a) for a in atomset}
        if not atoms:
            continue
        overlap |= (atoms & covered)
        out.append((key, atoms))
        covered |= atoms

    # FAIL FAST on any partition violation. v4 conserves atoms by construction;
    # if this fires it means the tokenizer is corrupt (a real bug), and silently
    # patching it (the old "absorb missing atoms into token 0" safety net) would
    # hide that corruption and feed wrong atom→motif assignments downstream.
    missing = set(range(n)) - covered
    if missing or overlap:
        raise ValueError(
            f"v4 tokenizer produced an invalid partition for {orig_smi!r} "
            f"({n} atoms): missing={sorted(missing)} overlap={sorted(overlap)}. "
            f"This indicates a fragmentation/merge bug — refusing to emit a "
            f"silently-corrected lookup. Tokens={[(k, sorted(a)) for k,a in out]}")
    return out


# ── Convenience: tokenize a whole corpus -> mol_frags_tracked ────────────────
def fragment_corpus_v4(smiles_list: List[str],
                       use_merge: bool = True,
                       verbose: bool = False
                       ) -> List[List[Tuple[str, Set[int]]]]:
    """End-to-end: learn rulebook on `smiles_list`, then tokenize each molecule.
    Returns mol_frags_tracked in the exact legacy contract."""
    ruleset, index = learn_corpus_rulebook(smiles_list, use_merge=use_merge, verbose=verbose)
    return [fragment_tracked_v4(smi, ruleset, index) for smi in smiles_list]
