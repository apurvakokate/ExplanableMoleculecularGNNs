#!/usr/bin/env python3
"""wl_ground_truth.py — INDEPENDENT, structure-driven ground truth via Weisfeiler-Leman.

This is the "OGX method, our data" tool. It does NOT depend on OpenGraphXAI or the TU
collection; it runs WL colouring on our own molecular datasets (SMILES) and plants a
ground truth defined by a WL structural signature — a cause that neither rbrics nor SFO
invented, so a fragmentation comparison against it is non-circular. It writes the result
in the SAME on-disk format as the planted / source Verified-GT datasets so it plugs into
the existing loaders/eval unchanged.

Two independent capabilities (subcommands):

  gt     Build a WL-planted dataset. Pick a WL colour at a chosen hop; y = 1 iff the
         molecule contains that colour; node ground truth = the atoms carrying it.
         Writes, matching MotifBreakdown/export_planted_gt.py:
            {out_root}/{name}_{fold}.csv         columns: smiles,label,group
            {gt_out}/{name}_planted_gt.npz       smiles -> node_imp [n_atoms x J]
         (J = number of disjoint colour instances in the molecule, one column each;
          grade max-over-columns for instance, union for global — like the Verified-GT
          sets. Negatives get an all-zero [n_atoms x 1] column.)

  stats  Hop/IoU diagnostic. For a given fragmentation vocab, sweep WL hops and report,
         per hop, how well each atom's k-hop neighbourhood aligns with the motif that
         contains it — mean IoU and the fraction of motifs "fully consumed" (motif
         subset of the k-hop neighbourhood). Tells us at what hop the WL scale matches
         the fragmentation's motif scale (i.e. the hop to plant at for a fair ceiling).

Design notes
  * WL colours are CONTENT-hashed (canonical string -> md5) so the same k-hop
    neighbourhood gets the same colour ACROSS molecules — required for planting a
    colour that recurs corpus-wide.
  * No new dependencies: rdkit + numpy + pandas only (all already in the env).
  * Read-only w.r.t. the repo; all outputs go to --out_root / --gt_out you pass.

Usage
  python analysis/wl_ground_truth.py stats --dataset BBBP --fold 0 \
      --data_root <FOLDS> --vocab_root vocab_final_v2 --variant rbrics --hops 0-6 \
      --out <SCRATCH>/hop_iou_BBBP_rbrics.csv

  python analysis/wl_ground_truth.py gt --dataset BBBP --fold 0 --hop 3 \
      --data_root <FOLDS> --name BBBP_WL_h3 \
      --out_root <SCRATCH>/wl --gt_out <SCRATCH>/wl_gt
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ─────────────────────────────────────────────────────────────────────────────
# RDKit helpers
# ─────────────────────────────────────────────────────────────────────────────
def _mol(smiles):
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    return Chem.MolFromSmiles(smiles)


def _adj(mol):
    """Adjacency list {atom_idx: [(nbr_idx, bond_type_str), ...]}."""
    adj = defaultdict(list)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = str(b.GetBondType())
        adj[i].append((j, bt))
        adj[j].append((i, bt))
    return adj


def _atom_invariant(atom):
    """Stable initial WL colour for an atom (its own type, ignoring neighbours)."""
    return (
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
    )


def _hash(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:16]


def wl_colours(mol, hops):
    """Per-atom WL colour at EACH hop 0..hops. Returns list `colours` where
    colours[k] is a dict {atom_idx: colour_str} — content-hashed so colours are
    comparable across molecules."""
    adj = _adj(mol)
    # hop 0: the atom's own invariant
    cur = {a: _hash(str(_atom_invariant(mol.GetAtomWithIdx(a))))
           for a in range(mol.GetNumAtoms())}
    out = [dict(cur)]
    for _ in range(hops):
        nxt = {}
        for a in range(mol.GetNumAtoms()):
            nbr = sorted((bt, cur[j]) for j, bt in adj[a])       # sorted multiset of (bond, nbr-colour)
            nxt[a] = _hash(cur[a] + '|' + '|'.join(f'{bt}:{c}' for bt, c in nbr))
        cur = nxt
        out.append(dict(cur))
    return out                                                    # length hops+1


def k_hop_atoms(adj, root, k):
    """Set of atoms within k bonds of `root` (BFS, inclusive)."""
    seen = {root}
    frontier = deque([(root, 0)])
    while frontier:
        a, d = frontier.popleft()
        if d == k:
            continue
        for j, _ in adj[a]:
            if j not in seen:
                seen.add(j)
                frontier.append((j, d + 1))
    return seen


def _connected_components(atoms, adj):
    """Split a set of atoms into connected components (in the molecular graph)."""
    atoms = set(atoms)
    comps, seen = [], set()
    for a in atoms:
        if a in seen:
            continue
        comp, stack = [], [a]
        seen.add(a)
        while stack:
            x = stack.pop()
            comp.append(x)
            for j, _ in adj[x]:
                if j in atoms and j not in seen:
                    seen.add(j)
                    stack.append(j)
        comps.append(sorted(comp))
    return comps


# ─────────────────────────────────────────────────────────────────────────────
# Dataset IO (matches the {data_root}/{dataset}_{fold}.csv convention)
# ─────────────────────────────────────────────────────────────────────────────
def load_fold_csv(data_root, dataset, fold, smiles_col, group_col):
    p = Path(data_root) / f'{dataset}_{fold}.csv'
    if not p.exists():
        raise SystemExit(f'fold CSV not found: {p}')
    df = pd.read_csv(p)
    if smiles_col not in df.columns:
        raise SystemExit(f'{p}: no smiles column {smiles_col!r} (have {list(df.columns)})')
    if group_col not in df.columns:
        # not fatal for gt build (we can synthesise a single pool), but warn loudly
        print(f'  [warn] {p}: no group column {group_col!r}; writing group="training" for all rows')
        df = df.copy()
        df[group_col] = 'training'
    return df, p


# ─────────────────────────────────────────────────────────────────────────────
# gt: build a WL-planted dataset
# ─────────────────────────────────────────────────────────────────────────────
def select_colour(df, smiles_col, hop, cov_lo, cov_hi, min_size, max_size, colour):
    """Return (colour_str, per_smiles_atom_colours, stats). Auto-select a balanced
    colour at `hop` unless an explicit --colour is given."""
    # colour presence per molecule + instance sizes
    mol_has = defaultdict(set)               # colour -> set of row indices containing it
    size_acc = defaultdict(list)             # colour -> instance sizes
    atom_colours = {}                        # row idx -> {atom: colour at hop}
    n_ok = 0
    for i, smi in enumerate(df[smiles_col]):
        mol = _mol(smi)
        if mol is None:
            atom_colours[i] = None
            continue
        n_ok += 1
        cols = wl_colours(mol, hop)[hop]
        atom_colours[i] = cols
        adj = _adj(mol)
        byc = defaultdict(list)
        for a, c in cols.items():
            byc[c].append(a)
        for c, atoms in byc.items():
            mol_has[c].add(i)
            for comp in _connected_components(atoms, adj):
                size_acc[c].append(len(comp))

    if colour is not None:
        chosen = colour
        if chosen not in mol_has:          # explicit colour absent at this hop -> fail loud,
            raise SystemExit(              # rather than silently emitting an all-negative dataset
                f'--colour {chosen!r} not found at hop {hop} in any of the {n_ok} parsed '
                f'molecules. Check the hop, or drop --colour to auto-select.')
    else:
        cands = []
        for c, rows in mol_has.items():
            cov = len(rows) / max(n_ok, 1)
            msz = float(np.mean(size_acc[c])) if size_acc[c] else 0.0
            if cov_lo <= cov <= cov_hi and min_size <= msz <= max_size:
                cands.append((abs(cov - 0.5), c, cov, msz, len(rows)))
        if not cands:
            raise SystemExit(f'no WL colour at hop {hop} in coverage [{cov_lo},{cov_hi}] '
                             f'and instance size [{min_size},{max_size}]. '
                             f'Loosen the bands or change --hop.')
        cands.sort()
        chosen = cands[0][1]
        print(f'  [select] {len(cands)} candidate colours; chose {chosen} '
              f'(coverage={cands[0][2]:.3f}, mean_inst_size={cands[0][3]:.2f}, '
              f'n_mols={cands[0][4]})')

    cov = len(mol_has[chosen]) / max(n_ok, 1)
    stats = dict(colour=chosen, hop=hop, coverage=round(cov, 4),
                 n_pos=len(mol_has[chosen]), n_mols=n_ok,
                 mean_inst_size=round(float(np.mean(size_acc[chosen])), 3) if size_acc[chosen] else 0.0)
    return chosen, atom_colours, stats


def build_gt(df, smiles_col, group_col, hop, chosen, atom_colours):
    """Relabel by presence of `chosen` colour; node_imp = per-instance atom masks."""
    rows_out, node_imp = [], {}
    n_skipped = 0
    seen = Counter()
    for i, smi in enumerate(df[smiles_col]):
        cols = atom_colours.get(i)
        grp = df.iloc[i][group_col]
        if cols is None:
            # (#2) Unparsable SMILES: no graph, so no node mask can exist. Drop from BOTH the CSV
            # and the npz so they stay in lockstep (a row without a mask KeyErrors / is silently
            # dropped downstream). The docstring's "negatives get a zero column" is for PARSED negs.
            n_skipped += 1
            continue
        seen[smi] += 1
        mol = _mol(smi)
        n = mol.GetNumAtoms()
        hit_atoms = [a for a, c in cols.items() if c == chosen]
        y = 1 if hit_atoms else 0
        rows_out.append(dict(smiles=smi, label=y, group=grp))
        if y:
            adj = _adj(mol)
            # (#1) GT = union of the hop-radius ego-graphs of the colour-carrying atoms, i.e. the
            # substructure that actually DETERMINES the WL colour and the GNN's receptive field —
            # paper Eq.10  V_{M_G} = ⋃_{v∈Ψ_G(c̄)} S_G^(ℓ)(v)  (Fig.3: "ℓ-radius ego-graphs of
            # nodes with WL colour"). NOT just the carrier atoms. Instances = connected components.
            region = set()
            for a in hit_atoms:
                region |= k_hop_atoms(adj, a, hop)
            comps = _connected_components(region, adj)           # one column per instance
            arr = np.zeros((n, len(comps)), dtype=np.float32)
            for j, comp in enumerate(comps):
                arr[comp, j] = 1.0
            node_imp[smi] = arr
        else:
            node_imp[smi] = np.zeros((n, 1), dtype=np.float32)   # negatives: no attribution
    if n_skipped:                                                 # (#2)
        print(f'  [gt] skipped {n_skipped} unparsable SMILES (dropped from both CSV and npz)')
    dups = sum(1 for c in seen.values() if c > 1)                 # (#5)
    if dups:
        print(f'  [gt] WARNING: {dups} SMILES occur more than once; their npz masks collapse to a '
              f'single entry, so npz key count < CSV row count (harmless only if truly identical).')
    return pd.DataFrame(rows_out), node_imp


def write_gt(out_root, gt_out, name, fold, df_out, node_imp):
    Path(out_root).mkdir(parents=True, exist_ok=True)
    Path(gt_out).mkdir(parents=True, exist_ok=True)
    csv_p = Path(out_root) / f'{name}_{fold}.csv'
    df_out.to_csv(csv_p, index=False)
    bad = [s for s in node_imp if '/' in s or '\\' in s]         # (#5) stereo-bond SMILES
    if bad:
        print(f'  [gt] WARNING: {len(bad)} SMILES keys contain "/" or "\\" (stereo bonds); '
              f'npz archive names with slashes can fail to round-trip through np.load — verify '
              f'they load back by key (e.g. {bad[0]!r}).')
    npz_p = Path(gt_out) / f'{name}_planted_gt.npz'
    np.savez_compressed(npz_p, **{smi: arr for smi, arr in node_imp.items()})
    return csv_p, npz_p


# ─────────────────────────────────────────────────────────────────────────────
# stats: hop vs motif-IoU diagnostic
# ─────────────────────────────────────────────────────────────────────────────
def _motif_of_atom(smi, vocab):
    """{atom_idx: motif_id} for one molecule from the vocab's per-atom lookup.
    Returns None if the SMILES isn't in the lookup or atom counts disagree."""
    lut = getattr(vocab, 'lookup_all', None)
    if lut is None or smi not in lut:
        return None
    entry = lut[smi]                                             # {atom: (smarts, motif_id)}
    return {int(a): int(v[1]) for a, v in entry.items()}


def hop_iou_stats(df, smiles_col, vocab, hops):
    """Per hop k: how well each atom's k-hop neighbourhood matches the motif that
    contains it. Returns a DataFrame(hop, n_atoms, mean_iou, median_iou,
    pct_consumed, pct_exact)."""
    per_hop = {k: {'iou': [], 'consumed': 0, 'exact': 0, 'n': 0} for k in hops}
    for smi in df[smiles_col]:
        mol = _mol(smi)
        if mol is None:
            continue
        a2m = _motif_of_atom(smi, vocab)
        if not a2m:
            continue
        adj = _adj(mol)
        # motif instance of each atom = connected component of same-motif-id atoms
        byid = defaultdict(list)
        for a, m in a2m.items():
            if m >= 0:
                byid[m].append(a)
        atom_motif = {}                                          # atom -> frozenset(instance atoms)
        for m, atoms in byid.items():
            for comp in _connected_components(atoms, adj):
                fs = frozenset(comp)
                for a in comp:
                    atom_motif[a] = fs
        for a, M in atom_motif.items():
            for k in hops:
                N = k_hop_atoms(adj, a, k)
                inter = len(N & M)
                union = len(N | M)
                iou = inter / union if union else 0.0
                d = per_hop[k]
                d['iou'].append(iou)
                d['n'] += 1
                if M <= N:
                    d['consumed'] += 1
                if N == M:
                    d['exact'] += 1
    rows = []
    for k in hops:
        d = per_hop[k]
        n = max(d['n'], 1)
        rows.append(dict(
            hop=k, n_atoms=d['n'],
            mean_iou=round(float(np.mean(d['iou'])), 4) if d['iou'] else float('nan'),
            median_iou=round(float(np.median(d['iou'])), 4) if d['iou'] else float('nan'),
            pct_consumed=round(d['consumed'] / n, 4),
            pct_exact=round(d['exact'] / n, 4)))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_hops(s):
    if '-' in s:
        lo, hi = s.split('-')
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(',')]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    g = sub.add_parser('gt', help='build a WL-planted dataset (planted/source format)')
    g.add_argument('--dataset', required=True)
    g.add_argument('--fold', type=int, required=True)
    g.add_argument('--data_root', required=True)
    g.add_argument('--name', required=True, help='output dataset name (e.g. BBBP_WL_h3)')
    g.add_argument('--out_root', required=True, help='dir for {name}_{fold}.csv')
    g.add_argument('--gt_out', required=True, help='dir for {name}_planted_gt.npz')
    g.add_argument('--hop', type=int, required=True)
    g.add_argument('--colour', default=None, help='specific WL colour to plant (default: auto)')
    g.add_argument('--cov_lo', type=float, default=0.2)
    g.add_argument('--cov_hi', type=float, default=0.8)
    g.add_argument('--min_size', type=float, default=3.0, help='min mean instance atoms')
    g.add_argument('--max_size', type=float, default=15.0)
    g.add_argument('--smiles_col', default='smiles')
    g.add_argument('--group_col', default='group')

    s = sub.add_parser('stats', help='hop vs motif-IoU diagnostic for a fragmentation')
    s.add_argument('--dataset', required=True)
    s.add_argument('--fold', type=int, required=True)
    s.add_argument('--data_root', required=True)
    s.add_argument('--vocab_root', required=True)
    s.add_argument('--variant', required=True, help='fragmentation vocab variant, e.g. rbrics')
    s.add_argument('--hops', default='0-6', help='e.g. 0-6 or 0,2,4')
    s.add_argument('--out', default=None, help='write the per-hop table to this CSV')
    s.add_argument('--limit', type=int, default=None, help='cap #molecules (quick test)')
    s.add_argument('--smiles_col', default='smiles')
    s.add_argument('--group_col', default='group')

    args = ap.parse_args()

    if args.cmd == 'gt':
        df, _ = load_fold_csv(args.data_root, args.dataset, args.fold,
                              args.smiles_col, args.group_col)
        chosen, atom_colours, st = select_colour(
            df, args.smiles_col, args.hop, args.cov_lo, args.cov_hi,
            args.min_size, args.max_size, args.colour)
        print(f'  [gt] {st}')
        df_out, node_imp = build_gt(df, args.smiles_col, args.group_col,
                                    args.hop, chosen, atom_colours)
        csv_p, npz_p = write_gt(args.out_root, args.gt_out, args.name, args.fold,
                                df_out, node_imp)
        pos = int((df_out['label'] == 1).sum())
        print(f'  [gt] wrote {csv_p}  ({pos}/{len(df_out)} positive)')
        print(f'  [gt] wrote {npz_p}  ({len(node_imp)} masks)')

    elif args.cmd == 'stats':
        from SharedModules.data.vocab import load_vocab
        df, _ = load_fold_csv(args.data_root, args.dataset, args.fold,
                              args.smiles_col, args.group_col)
        if args.limit:
            df = df.head(args.limit)
        vocab = load_vocab(str(args.vocab_root), args.dataset, args.variant)
        hops = _parse_hops(args.hops)
        tab = hop_iou_stats(df, args.smiles_col, vocab, hops)
        print(tab.to_string(index=False))
        # report the hop where motifs are mostly consumed
        consumed = tab[tab['pct_consumed'] >= 0.9]
        if len(consumed):
            print(f"\n  motifs mostly consumed (>=90%) at hop {int(consumed.iloc[0]['hop'])}")
        peak = tab.loc[tab['mean_iou'].idxmax()]
        print(f"  peak mean-IoU at hop {int(peak['hop'])} (IoU={peak['mean_iou']})")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            tab.to_csv(args.out, index=False)
            print(f'  wrote {args.out}')


if __name__ == '__main__':
    main()
