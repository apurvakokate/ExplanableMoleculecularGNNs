"""fm_bpe.py — FM-BPE (Frequency-MDL BPE): greedy agglomerative merge under a fixed-b MDL criterion.

Starts from the FINEST blocks consistent with every fragmentation approach in the motif pool
(rings kept whole — no approach ever cuts inside a ring; every other atom its own block, only
single acyclic bonds cut so C=O/C=N pairs stay together). NOTHING else is pre-grouped: FG heads
are NOT frozen, so the FM-BPE criterion alone decides every merge (amide/ester/... re-form only
if the data pays for them). Rings may GROW outward but never split.

Merge criterion (FM-BPE, fixed-b form). For an adjacent fragment-TYPE pair (M1, M2) with corpus
count f12 and heavy-atom sizes |M1|, |M2|. f12 = number of merges that would ACTUALLY happen
(node-disjoint adjacencies; see _local_pairs), NOT raw adjacencies — so each counted co-occurrence
removes exactly one token and N_after = N - f12 holds exactly:

    dC(M1,M2) = lambda * (|M1| + |M2|)  -  f12 * b          merge iff dC < 0

derived from the two-part cost  lambda*cost(vocab) + cost(D|vocab)  under a FIXED b bits per token
reference (NOT log|V|): adding the merged type costs its size |M1|+|M2| in the vocab; each of the
f12 co-occurrences then uses ONE reference instead of two, saving b bits apiece. The alone-terms
(M1, M2 used elsewhere) and the rest-of-vocab term cancel under fixed b (see the derivation — the
log|V| variant's +(f1+f2+T)*log(|V+1|/|V|) inflation vanishes when b is constant).

ONLY the ratio theta = lambda/b matters:  dC < 0  <=>  f12 > theta * (|M1|+|M2|).  So this module
fixes b = 1.0 and sweeps lambda == theta = "required co-occurrences per unit merged-motif size".
Small theta -> coarse (aggressive merging); large theta -> fine. theta is the single granularity
knob (the analogue of SFO's size ramp).

FULL-MDL form (cost='full_mdl'): uniform log|V| code instead of fixed b,

    dC = lambda*(|M1|+|M2|) + (f1+f2+T)*log2((V+1)/V) + f12*log2((V+1)/V^2)
       = lambda*(|M1|+|M2|) + N*log2((V+1)/V) - f12*log2(V+1)        (f1+f2+T = N - 2*f12)

with N = total fragment occurrences and V = number of distinct fragment types present. Here lambda
is NOT a pure ratio (it trades vocabulary size against code length), and the bar depends on N and
V, so lambda is dataset-specific; see fm_bpe_lambda_star.py for how lambda* is chosen.

Greedy loop: each round score every adjacent type-pair, apply the most-negative-dC merge across the
whole corpus (BPE: merge that type-pair everywhere), recount, stop when no pair has dC < 0.

Reuses the vetted _MolGraph BPE engine from cascade_bpe_linker.py unchanged (same finest='all_bonds'
starting cut, same frag_key re-keying on merge, same connectivity-respecting apply_merge).

Top-level: build(smiles_all, lam, b=1.0, max_atoms=None) -> per-mol [(key, set(atoms))] aligned to
smiles_all, IDENTICAL format to ring_mdl.build / mdl_linker.build so the vocab call site is unchanged.
head_source / linker_method / break_fused_rings are accepted for call-compatibility and ignored.

NO FALLBACK on keying/cover errors (fails loudly, like ring_mdl / mdl_only). Unparseable SMILES,
a missing/mismatched `groups`, an empty train+val set, or a selected merge that applies nowhere
all RAISE — nothing is skipped or given an empty cover.
"""
from __future__ import annotations

import copy
import math
import sys
from collections import Counter
from typing import Dict, List, Sequence, Set, Tuple

from rdkit import Chem, RDLogger

import ring_mdl
from cascade_bpe_linker import _MolGraph

RDLogger.DisableLog('rdApp.*')


# ───────────────────────────── finest starting state ───────────────────────────
def _union_cut_bonds(mol: Chem.Mol):
    """SFO's EXACT union-of-fragmenters cut set — the same blocks SFO's exact-cover starts from —
    via xcover.exp1_candidate_inventory.segment_to_fixpoint (BRICS + rBRICS + RECAP + CRUSH, unioned
    and refined to a fixpoint). Using SFO's own block builder (not a separate scheme set) makes the
    union arm a clean criterion-vs-SFO comparison: identical admissible boundaries, only the merge
    rule differs. These schemes never cut ring bonds. (Deps xcover + crush_smirks_rules are the ones
    SFO itself uses, already deployed — no extra module needed.)"""
    import os, sys
    _mb = os.path.dirname(os.path.abspath(__file__))
    _xc = os.path.join(_mb, 'xcover')
    for _p in (_mb, _xc):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import exp1_candidate_inventory as _X
    rules = _X._ensure_rules(_mb)
    cuts, _base, _rounds = _X.segment_to_fixpoint(mol, rules)
    out = set()
    for bi in cuts:                                     # SFO union-of-fragmenters bonds (fixpoint)
        b = mol.GetBondWithIdx(bi)
        out.add(frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
    return out


def _finest_graph(mol: Chem.Mol, freeze_rings: bool = True, start: str = 'atomic') -> _MolGraph:
    """Starting blocks for the FM-BPE merge. Ring systems are kept whole (ring_mdl partition with
    fused_only=True, freeze='rings'): a ring motif is one fused/spiro system; a biaryl bond between two
    ring systems is a boundary between two frozen ring motifs (never merged, like any ring-ring pair).

    start='atomic' (default): every non-ring atom is its own block (finest='all_bonds' cuts every
      single acyclic bond) — the absolute finest; the merge can reach ANY partition.
    start='union' — RINGS EXPLICITLY SEPARATED (used by the completed fmbpe_v2 runs; unchanged): fused ring
      systems are frozen bare blocks first, so EVERY non-ring bond touching a ring atom (ring–substituent and
      ring–ring) is a boundary; among non-ring atoms, blocks merge across every SINGLE bond no fragmenter
      cuts (union of BRICS + rBRICS + RECAP + CRUSH, to a fixpoint). Verified rule, 37,850/37,850 molecules.
    start='union_frag' — RINGS CUT ONLY WHERE THE FRAGMENTER UNION CUTS (= SFO's starting blocks, added
      2026-10-16): blocks merge across EVERY bond no fragmenter cuts, ring–substituent bonds included, so a
      halogen / exocyclic =O (never cut by any fragmenter) joins its ring's block while ring–CH3/–NH2/–OH
      usually stay separate. Ring bonds are never cut. Like 'union', FM-BPE never cuts a double/triple bond
      (C=O, C=N, exocyclic C=C) — the one deliberate difference from SFO's raw blocks. Verified identical to
      SFO's blocks with SFO's double-bond cuts ignored, every train+val molecule of 7 datasets.
    In both union starts a block containing ring atoms is FROZEN (never merged by FM-BPE); a bare ring
    system keeps the substituent-agnostic `ring:` key, a decorated ring gets the attachment-aware frag_key.

    freeze_rings=True (default): ring blocks stay frozen ("rings don't grow"). freeze_rings=False: every
      block is un-frozen after the start is built, so ring blocks take part in the MDL merge like any other
      block ("rings grow": ring+substituent and ring+ring merges become candidates)."""
    if start not in ('atomic', 'union', 'union_frag'):
        raise ValueError(f"unknown start {start!r}; expected 'atomic', 'union' or 'union_frag'")
    if mol is None:
        raise ValueError("_finest_graph got mol=None (unparseable SMILES)")
    # fused_only=True: a ring motif is a FUSED/spiro system; rings joined by a non-ring bond (biaryl)
    # are separate ring motifs. (ring_mdl's default groups whole ring ASSEMBLIES, which made every
    # biaryl/teraryl combination its own rare frozen motif — e.g. hERG kept-atom coverage 59.7% vs
    # 84.4% fused-only at no merges.)
    g = _MolGraph(mol, finest='all_bonds', freeze='rings',
                  fg_partition=ring_mdl.make_partition(fused_only=True))
    if start == 'union':
        # ORIGINAL loop (completed runs): merge linker atoms across every NON-voted single acyclic bond;
        # frozen ring blocks are never touched, so every ring boundary bond stays a boundary.
        U = _union_cut_bonds(mol)
        changed = True
        while changed:
            changed = False
            a2n = {a: ni for ni, at in enumerate(g.atoms) for a in at}
            for b in mol.GetBonds():
                if b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE:
                    continue                            # rings/double bonds already within one block
                ai, aj = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                i, j = a2n[ai], a2n[aj]
                if i == j or g.frozen[i] or g.frozen[j]:
                    continue                            # skip ring (frozen) nodes during pre-merge
                if frozenset((ai, aj)) in U:
                    continue                            # union boundary -> keep this cut
                g._merge_nodes(i, j); changed = True; break   # indices shift -> restart scan
    elif start == 'union_frag':
        U = _union_cut_bonds(mol)
        changed = True
        while changed:                                  # merge blocks across every bond NO fragmenter cuts
            changed = False
            a2n = {a: ni for ni, at in enumerate(g.atoms) for a in at}
            for b in mol.GetBonds():
                ai, aj = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                i, j = a2n[ai], a2n[aj]
                if i == j:
                    continue                            # already one block
                if frozenset((ai, aj)) in U and b.GetBondType() == Chem.BondType.SINGLE:
                    continue                            # a fragmenter cuts this SINGLE bond -> keep the boundary
                # (double/triple bonds are never boundaries in FM-BPE, even where a fragmenter cuts one)
                if g.frozen[j] and not g.frozen[i]:
                    i, j = j, i                         # merge INTO the ring block so it stays frozen
                g._merge_nodes(i, j); changed = True; break   # indices shift -> restart scan
    if not freeze_rings:
        g.frozen = [False] * len(g.frozen)              # un-freeze -> rings grow (take part in the merge)
    return g


# ───────────────────────────── greedy advisor-MDL merge ────────────────────────
def _local_pairs(g: _MolGraph, type_size: Dict[str, int]) -> Counter:
    """Counter of this graph's REALIZABLE merges per fragment-TYPE pair (sorted); also records each
    type's size.

    f12 counts the merges apply_merge(pair) would ACTUALLY perform in this graph, not raw
    adjacencies. Adjacencies of one pair type overlap when they share a node (CF3: three *C-*F bonds
    share the carbon; a *C*-*C*-*C* chain: two bonds share the middle): merging one turns the shared
    node into a NEW, larger type, so the other adjacencies no longer match the pair. apply_merge scans
    adjacencies in bond order, merges the first match and rescans, and a merged node can never match
    the same pair again (its key has more atoms than either part). Its result is therefore exactly a
    greedy node-disjoint matching over that pair's adjacencies in bond order — computed here.
    learn_fm_bpe asserts applied == counted every step."""
    by_type: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    for i, j, _ in g.linker_adjacencies():                   # bond order, same as apply_merge
        a, c = g.ident[i], g.ident[j]
        type_size.setdefault(a, len(g.atoms[i]))
        type_size.setdefault(c, len(g.atoms[j]))
        by_type.setdefault(tuple(sorted((a, c))), []).append((i, j))
    lp: Counter = Counter()
    for p, edges in by_type.items():
        used: Set[int] = set()
        for i, j in edges:
            if i not in used and j not in used:
                used.add(i); used.add(j); lp[p] += 1
    return lp


def learn_fm_bpe(smiles: Sequence[str], lam: float, b: float = 1.0,
                  max_atoms: int | None = None, max_merges: int | None = None, verbose: bool = False,
                  freeze_rings: bool = True, start: str = 'atomic', cost: str = 'fixed_b',
                  start_graphs: Sequence[_MolGraph] | None = None):
    """Greedy BPE. Returns (rules, graphs) where graphs are the merged per-molecule covers and
    rules = [(pair_type, dC, f12, merged_size, n_mols), ...] in merge order; n_mols = number of
    molecules the merge touched (= the pair's molecule support at the moment it fired).

    cost='fixed_b'  : dC = lam*(|M1|+|M2|) - f12*b                                   (default)
    cost='full_mdl' : dC = lam*(|M1|+|M2|) + N*log2((V+1)/V) - f12*log2(V+1)
                      N = total fragment occurrences, V = distinct fragment types present.
    freeze_rings / start are passed to _finest_graph (see there).

    start_graphs (optional, speed only): the _finest_graph(...) of each molecule, pre-built by the
    caller with the SAME freeze_rings/start. They are deep-copied (never mutated), so repeated calls
    (e.g. a lambda search) skip the expensive union-start rebuild; the result is identical to building
    them here. Must align 1:1 with `smiles` (checked: count and per-molecule atom coverage).

    Incremental: the global adjacency-pair count (and, for full_mdl, the per-type occurrence count)
    is maintained exactly, re-scanning ONLY the graphs a merge actually touches. Output is identical
    to a full recount."""
    if cost not in ('fixed_b', 'full_mdl'):
        raise ValueError(f"unknown cost {cost!r}; expected 'fixed_b' or 'full_mdl'")
    full = cost == 'full_mdl'
    graphs: List[_MolGraph] = []
    if start_graphs is not None:
        if len(start_graphs) != len(smiles):
            raise ValueError(f"learn_fm_bpe: {len(start_graphs)} start_graphs for {len(smiles)} SMILES")
        for k, (s, g0) in enumerate(zip(smiles, start_graphs)):
            m = Chem.MolFromSmiles(s)
            if m is None:
                raise ValueError(f"learn_fm_bpe: unparseable SMILES {s!r} (no silent skip)")
            if (m.GetNumAtoms() != g0.mol.GetNumAtoms()
                    or sorted(a for at in g0.atoms for a in at) != list(range(m.GetNumAtoms()))):
                raise ValueError(f"learn_fm_bpe: start_graphs[{k}] does not match SMILES {s!r}")
            graphs.append(copy.deepcopy(g0))
    else:
        for s in smiles:
            m = Chem.MolFromSmiles(s)
            if m is None:
                raise ValueError(f"learn_fm_bpe: unparseable SMILES {s!r} (no silent skip)")
            graphs.append(_finest_graph(m, freeze_rings=freeze_rings, start=start))
    if not graphs:
        raise ValueError("learn_fm_bpe: no molecules to learn from")

    type_size: Dict[str, int] = {}
    gpairs: List[Counter] = [_local_pairs(g, type_size) for g in graphs]
    f12: Counter = Counter()
    for lp in gpairs:
        f12.update(lp)

    # full_mdl needs N (total occurrences) and V (distinct types) of the CURRENT cover.
    gtypes: List[Counter] = []
    tcount: Counter = Counter()
    if full:
        for g in graphs:
            t = Counter(k for k, _ in g.fragments())
            gtypes.append(t)
            tcount.update(t)

    rules: List[Tuple[Tuple[str, str], float, int, int, int]] = []
    # max_merges=None => uncapped: the greedy loop stops naturally when no pair has dC<0
    # (the ``best_p is None`` break below).
    for step in range(max_merges if max_merges is not None else 10**9):
        if full:
            V = sum(1 for v in tcount.values() if v > 0)
            N = sum(tcount.values())
            if V < 2:
                break
            growth = N * math.log2((V + 1) / V)              # every token's code lengthens
            perc = math.log2(V + 1)                          # bits saved per merged co-occurrence
        best_p, best_dC, best_msz = None, 0.0, 0             # accept only dC < 0, pick most negative
        for p, c in f12.items():
            if c <= 0:
                continue
            msz = type_size[p[0]] + type_size[p[1]]          # |M1|+|M2| (heavy atoms)
            if max_atoms is not None and msz > max_atoms:
                continue
            dC = (lam * msz + growth - c * perc) if full else (lam * msz - c * b)
            if dC < best_dC:
                best_p, best_dC, best_msz = p, dC, msz
        if best_p is None:
            break

        cnt = f12[best_p]
        applied = 0
        n_mols = 0
        for gi, g in enumerate(graphs):                      # only graphs containing best_p change
            exp_g = gpairs[gi].get(best_p, 0)
            if exp_g == 0:
                continue
            n_mols += 1
            f12.subtract(gpairs[gi])                         # remove this graph's old contribution
            if full:
                tcount.subtract(gtypes[gi])
            got_g = g.apply_merge(best_p)                    # BPE: merge best_p everywhere in g
            if got_g != exp_g:
                raise RuntimeError(f"learn_fm_bpe: step {step} pair {best_p} graph {gi}: counted "
                                   f"{exp_g} realizable merges but apply_merge did {got_g}")
            applied += got_g
            gpairs[gi] = _local_pairs(g, type_size)          # rescan ONLY this graph
            f12.update(gpairs[gi])                           # add its new contribution
            if full:
                gtypes[gi] = Counter(k for k, _ in g.fragments())
                tcount.update(gtypes[gi])
        if applied != cnt:
            # counted f12 and the merges actually performed disagree -> bookkeeping bug: fail loudly.
            raise RuntimeError(f"learn_fm_bpe: merge {best_p} has f12={cnt} but applied {applied} "
                               f"(touched {n_mols} graphs) at step {step}")
        rules.append((best_p, best_dC, cnt, best_msz, n_mols))
        if verbose and (step < 15 or step % 25 == 0):
            print(f"  merge {step:3d}: {best_p[0]:>14s} + {best_p[1]:<14s} "
                  f"f12={cnt:5d} mols={n_mols:5d} size={best_msz:2d} dC={best_dC:9.1f}", flush=True)

    return rules, graphs


# ───────────────────────────── vocab-pipeline entry point ──────────────────────
_TRAINVAL = {'training', 'train', 'valid', 'validation', 'val'}


def build(smiles_all: Sequence[str], groups=None, head_source: str = 'none', linker_method: str = 'mdl',
          break_fused_rings: bool = False, lam: float = 1.0, b: float = 1.0,
          max_atoms: int | None = None, max_merges: int | None = None, verbose: bool = False,
          freeze_rings: bool = True, start: str = 'atomic', cost: str = 'fixed_b',
          return_rules: bool = False):
    """Per-mol [(key, set(atoms))] aligned to smiles_all. head_source/linker_method/break_fused_rings
    accepted for call-compatibility and ignored. lam == theta (b fixed to 1.0 is the natural choice).
    freeze_rings (default True) -> bare ring tokens; start 'atomic' (default) or 'union' (SFO pool).
    cost: 'fixed_b' (default) or 'full_mdl' (see learn_fm_bpe). return_rules=True returns
    (covers, rules) so the caller can record the learned merges.

    HELD-OUT TEST (proper per-fold CV): merge rules are LEARNED on train+val only (`groups` in
    {training,valid}) and then REPLAYED per-molecule onto the whole set, so test co-occurrences never
    influence which merges happen — the vocabulary is genuinely fold-specific and matches SFO's
    train+val-driven selection. `groups` is REQUIRED: there is no learn-on-all mode (it would let
    test molecules shape the vocabulary). Fails loudly on unparseable SMILES or an empty train+val."""
    if groups is None:
        raise ValueError("fm_bpe.build: groups is required (merges must be learned on train+val only)")
    if len(groups) != len(smiles_all):
        raise ValueError(f"fm_bpe.build: len(groups)={len(groups)} != len(smiles_all)={len(smiles_all)}")
    valid: List[Tuple[int, str]] = []
    for i, s in enumerate(smiles_all):
        if Chem.MolFromSmiles(s) is None:
            raise ValueError(f"fm_bpe.build: unparseable SMILES at row {i}: {s!r} (no silent skip)")
        valid.append((i, s))

    learn_smi = [s for (i, s) in valid if str(groups[i]) in _TRAINVAL]
    if not learn_smi:
        raise ValueError(f"fm_bpe.build: no train+val molecules (groups seen: {sorted(set(map(str, groups)))})")
    if verbose:
        print(f"[fm_bpe] learning merges on {len(learn_smi)} train+val mols of {len(valid)} total; "
              f"start={start} freeze_rings={freeze_rings} cost={cost} lam={lam}", flush=True)

    rules, _ = learn_fm_bpe(learn_smi, lam=lam, b=b, max_atoms=max_atoms, max_merges=max_merges,
                            verbose=verbose, freeze_rings=freeze_rings, start=start, cost=cost)
    rule_pairs = [r[0] for r in rules]                     # (M1,M2) type-pairs in learned order

    # REPLAY the learned merges onto EVERY molecule (train/val/test), per-molecule, deterministically.
    covers: List[List[Tuple[str, Set[int]]]] = [[] for _ in smiles_all]
    for i, s in valid:
        g = _finest_graph(Chem.MolFromSmiles(s), freeze_rings=freeze_rings, start=start)
        for p in rule_pairs:
            g.apply_merge(p)
        covers[i] = [(k, set(at)) for k, at in g.fragments()]
    return (covers, rules) if return_rules else covers
