#!/usr/bin/env python3
"""apply_gt.py — Phase 4: apply synthetic GT relabelling and cache results.

Reads rules.json from the vocab output (written by phase3 / generate_vocab_rules.py),
picks the rule at --rule_index, then for every split of the dataset annotates each
PyG Data object:

  data.y          replaced with 1/0 based on whether the rule fires
  data.node_label float [N] — 1.0 for atoms that belong to a rule-motif, 0.0
                  otherwise.  The authoritative node-level explanation target.
  data.edge_label float [E] — 1.0 for edges whose BOTH endpoints are rule-motif
                  atoms (AND), 0.0 otherwise.  Used by the edge-level explainer
                  ROC; AND (not OR) so motif-boundary edges don't penalise a
                  correctly motif-focused explainer and so the edge GT matches
                  the att[src]*att[dst] edge score used in evaluation.

Output structure:
  {out_dir}/{dataset}/fold{fold}/{variant}/relabel1/
      train_with_gt.pt
      valid_with_gt.pt
      test_with_gt.pt
      selected_rule.json

Usage:
    python apply_gt.py \\
        --dataset Mutagenicity --fold 0 \\
        --vocab_root ./vocab_output \\
        --variant all_fallback_bpe \\
        --rule_index 0 \\
        --data_root ./datasets/FOLDS \\
        --out_dir ./results/gt_cache
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch

# Make the project importable from any working directory
_HERE = Path(__file__).resolve()
_PROJECT = _HERE.parents[2]
for _p in [str(_PROJECT), str(_HERE.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from SharedModules.data.loader import get_loaders, TASK_TYPE
from SharedModules.data.vocab import load_vocab
from SharedModules.utils import set_seed

# ─────────────────────────────────────────────────────────────────────────────
# Rule loading
# ─────────────────────────────────────────────────────────────────────────────

def load_best_rule(vocab_root: str, dataset: str, variant: str,
                   rule_index: int) -> dict:
    """Load the rule at rule_index from rules.json.

    rules.json is a list of rule dicts sorted by score descending (best first).
    Each rule dict has at minimum: {'motifs': [smarts, ...], 'score': float, ...}

    Returns the selected rule dict, or raises if the file is missing / index
    is out of range.
    """
    rules_path = Path(vocab_root) / dataset / variant / 'rules.json'
    if not rules_path.exists():
        raise FileNotFoundError(
            f"rules.json not found: {rules_path}\n"
            f"Run phase3 first: bash run_experiments.sh phase3"
        )

    with open(rules_path) as f:
        raw = json.load(f)

    # Handle both formats: plain list or {'all_rules': [...]}
    if isinstance(raw, list):
        rules = raw
    elif isinstance(raw, dict) and 'all_rules' in raw:
        rules = raw['all_rules']
    else:
        rules = []

    if not rules:
        raise ValueError(f"rules.json is empty: {rules_path}")

    if rule_index >= len(rules):
        raise IndexError(
            f"rule_index={rule_index} out of range — "
            f"only {len(rules)} rules in {rules_path}"
        )

    rule = rules[rule_index]
    # Normalise: rules may have 'clauses' (DNF format from extract_rules)
    # or a flat 'motifs' list (format from run_phase3).
    # Always store as 'clauses' internally — a list of motif-sets.
    # DNF evaluation: rule fires if ANY clause fires (OR of ANDs).
    rule = dict(rule)
    if 'clauses' in rule:
        # Legacy format: clauses is a list of {motifs: [...], k: int}
        rule['_clauses'] = [set(cl.get('motifs', [])) for cl in rule['clauses']]
    elif 'motifs' in rule:
        # Flat format from run_phase3: treat as single AND-clause
        rule['_clauses'] = [set(rule['motifs'])]
    else:
        raise ValueError(f"Rule has no 'motifs' or 'clauses' key: {list(rule.keys())}")

    return rule


# ─────────────────────────────────────────────────────────────────────────────
# Graph annotation
# ─────────────────────────────────────────────────────────────────────────────

def annotate_split(data_list: List,
                   rule_clauses: List[Set[str]],
                   graph_lookup: Dict[str, Dict[int, Tuple[str, int]]],
                   relabel: bool = True) -> Tuple[List, Dict]:
    """Annotate Data objects with GT labels and edge labels.

    rule_clauses: list of sets — DNF rule fires when ANY clause fires.
                  A clause fires when ALL its motifs are present (AND logic).
    graph_lookup: {smiles: {node_idx: (smarts, motif_id)}}

    Sets data.node_label [N] (1.0 = rule-motif atom) and data.edge_label [E]
    (1.0 = BOTH endpoints are rule-motif atoms; AND).

    Returns (annotated_data_list, stats_dict).
    """
    n_rule_pos = 0
    n_relabelled = 0
    n_pos_edges = 0
    n_total_edges = 0
    n_pos_nodes = 0
    n_total_nodes = 0

    out = []
    for data in data_list:
        smi = getattr(data, 'smiles', None)
        data = data.clone()

        node_map = graph_lookup.get(smi, {}) if smi else {}

        # Fragment set for this molecule: unique SMARTS strings present
        frag_set = {smarts for smarts, _mid in node_map.values()}

        # DNF: rule fires if ANY clause fires; a clause fires if ALL its
        # motifs are present in this molecule (OR of ANDs).
        rule_fires = any(cl.issubset(frag_set) for cl in rule_clauses if cl)

        gt_y = 1.0 if rule_fires else 0.0
        if rule_fires:
            n_rule_pos += 1

        # Relabel data.y
        old_y = float(data.y.view(-1)[0].item()) if data.y is not None else -1
        if relabel:
            data.y = torch.tensor([gt_y], dtype=torch.float32)
            if old_y != gt_y:
                n_relabelled += 1

        # Build node-level GT (authoritative target) and edge-level GT from it.
        #   node_label[i] = 1.0  iff atom i belongs to a rule-clause motif
        #   edge_label[e] = 1.0  iff BOTH endpoints are rule-motif atoms (AND)
        n_nodes = data.x.size(0)
        n_edges = data.edge_index.size(1)
        n_total_nodes += n_nodes
        n_total_edges += n_edges
        node_label = torch.zeros(n_nodes, dtype=torch.float32)
        edge_label = torch.zeros(n_edges, dtype=torch.float32)

        if rule_fires and node_map:
            # Nodes whose fragment belongs to any motif in any rule clause
            all_rule_motifs = {m for cl in rule_clauses for m in cl}
            rule_nodes = [idx for idx, (smarts, _mid) in node_map.items()
                          if smarts in all_rule_motifs]
            if rule_nodes:
                active = torch.zeros(n_nodes, dtype=torch.bool)
                idx_t = torch.tensor(rule_nodes, dtype=torch.long).clamp(0, n_nodes - 1)
                active[idx_t] = True
                node_label[active] = 1.0
                n_pos_nodes += int(active.sum().item())
                if n_edges > 0:
                    src, dst = data.edge_index
                    pos = active[src] & active[dst]   # AND of both endpoints
                    edge_label[pos] = 1.0
                    n_pos_edges += int(pos.sum().item())

        data.node_label = node_label
        data.edge_label = edge_label
        out.append(data)

    stats = {
        'n_graphs':       len(out),
        'n_rule_pos':     n_rule_pos,
        'n_relabelled':   n_relabelled,
        'n_pos_nodes':    n_pos_nodes,
        'n_total_nodes':  n_total_nodes,
        'node_pos_frac':  (n_pos_nodes / n_total_nodes
                           if n_total_nodes > 0 else 0.0),
        'n_pos_edges':    n_pos_edges,
        'n_total_edges':  n_total_edges,
        'edge_pos_frac':  (n_pos_edges / n_total_edges
                           if n_total_edges > 0 else 0.0),
    }
    return out, stats


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Phase 4: apply GT relabelling from rules.json',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  export RULE_INDEX=0
  source ../../experiment_config.sh
  python apply_gt.py \\
      --dataset Mutagenicity --fold 0 \\
      --vocab_root "$VOCAB_ROOT" \\
      --variant all_fallback_bpe \\
      --rule_index "$RULE_INDEX" \\
      --data_root "$DATA_ROOT" \\
      --out_dir "$OUT_ROOT/gt_cache"
""")
    parser.add_argument('--dataset',     required=True)
    parser.add_argument('--fold',        type=int, default=0)
    parser.add_argument('--vocab_root',  required=True)
    parser.add_argument('--variant',     required=True,
                        help='Vocab variant, e.g. all_fallback_bpe')
    parser.add_argument('--rule_index',  type=int, default=0,
                        help='Index into rules.json (0 = best rule)')
    parser.add_argument('--data_root',   required=True)
    parser.add_argument('--out_dir',     required=True,
                        help='Root of gt_cache output directory')
    parser.add_argument('--no_relabel',  action='store_true',
                        help='Attach edge_label but keep original data.y')
    parser.add_argument('--batch_size',  type=int, default=128)
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    relabel = not args.no_relabel

    # Guard: synthetic relabelling only makes sense for the classification
    # datasets that have a motif-rule GT pipeline. Regression sets (esol,
    # Lipophilicity) have no rule-fires label, and mutag ships SOURCE GT and must
    # NOT be synthetically relabelled. Fail fast rather than silently writing a
    # bogus 0/1 cache.
    from SharedModules.data.ground_truth import GT_SUPPORTED_DATASETS
    if args.dataset not in GT_SUPPORTED_DATASETS:
        raise ValueError(
            f"--dataset {args.dataset!r} is not in GT_SUPPORTED_DATASETS "
            f"({sorted(GT_SUPPORTED_DATASETS)}). Synthetic GT relabelling is only "
            f"defined for these classification datasets. Regression datasets and "
            f"mutag (which has source GT) must not be relabelled.")

    print(f'\n{"="*60}')
    print(f'  Phase 4 — GT annotation')
    print(f'  dataset    = {args.dataset}')
    print(f'  variant    = {args.variant}')
    print(f'  rule_index = {args.rule_index}')
    print(f'  out_dir    = {args.out_dir}')
    print(f'{"="*60}')

    # ── Load rule from rules.json ─────────────────────────────────────────────
    rule = load_best_rule(args.vocab_root, args.dataset, args.variant,
                          args.rule_index)
    rule_clauses: List[Set[str]] = rule.get('_clauses', [])
    all_motifs = {m for cl in rule_clauses for m in cl}

    print(f'\n  Rule #{args.rule_index} (DNF — OR of {len(rule_clauses)} clause(s)):')
    for i, cl in enumerate(rule_clauses):
        print(f'    clause {i}   = {sorted(cl)}')
    # rules.json uses pct1/n1/n0 (legacy extract_rules pipeline)
    match_pct = rule.get("pct1", rule.get("pct_match", "?"))
    n_match   = rule.get("n1",   rule.get("n_match",   "?"))
    print(f'    match%     = {match_pct}%')
    print(f'    n_match    = {n_match}')
    print(f'    n_clauses  = {rule.get("n_clauses", len(rule.get("clauses", [])))}' )
    for i, cl in enumerate(rule.get("clauses", [])):
        print(f'    clause {i}   = {cl.get("motifs", [])}' )

    if not rule_clauses or not all_motifs:
        print('\n  [error] Rule has no motifs — check rules.json')
        sys.exit(1)

    # ── Load vocab (for graph_lookup) ─────────────────────────────────────────
    print('\n  Loading vocabulary...')
    vocab = load_vocab(args.vocab_root, args.dataset, args.variant)
    print(f'    {vocab.num_motifs} motifs')

    # Merge train + valid lookups (both are needed for the three splits)
    lookup_train = vocab.lookup_train   # training molecules
    lookup_valid = vocab.lookup_valid   # valid molecules
    lookup_test  = vocab.lookup_test    # test molecules

    # ── Load data loaders to get Data objects ─────────────────────────────────
    print('  Loading data loaders...')
    task_type = TASK_TYPE.get(args.dataset, 'BinaryClass')
    # processed_root MUST be variant-specific: the cached .pt bakes in
    # nodes_to_motifs from THIS variant's vocab lookup. Sharing a path across
    # variants (as the old non-variant path did) makes the second variant reuse
    # the first variant's motif annotations → wrong edge_label / relabels.
    loaders, test_ds, meta = get_loaders(
        dataset=args.dataset,
        data_root=args.data_root,
        fold=args.fold,
        vocab=vocab,
        processed_root=(f'/tmp/apply_gt_processed/'
                        f'{args.dataset}_fold{args.fold}/{args.variant}'),
        batch_size=args.batch_size,
    )
    print(f'    train={len(loaders["train"].dataset)}  '
          f'valid={len(loaders["valid"].dataset)}  '
          f'test={len(test_ds)}')

    # ── Annotate each split ───────────────────────────────────────────────────
    split_configs = [
        ('train', loaders['train'].dataset, lookup_train),
        ('valid', loaders['valid'].dataset, lookup_valid),
        ('test',  test_ds,                  lookup_test),
    ]

    out_base = (Path(args.out_dir) / args.dataset
                / f'fold{args.fold}' / args.variant
                / ('relabel1' if relabel else 'relabel0'))
    out_base.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for split_name, ds, lookup in split_configs:
        data_list = [ds[i] for i in range(len(ds))]
        annotated, stats = annotate_split(data_list, rule_clauses, lookup,
                                          relabel=relabel)
        pt_path = out_base / f'{split_name}_with_gt.pt'
        torch.save(annotated, pt_path)

        all_stats[split_name] = stats
        print(f'\n  [{split_name}]'
              f'  n={stats["n_graphs"]}'
              f'  rule_pos={stats["n_rule_pos"]}'
              f'  relabelled={stats["n_relabelled"]}'
              f'  node_pos_frac={stats["node_pos_frac"]:.4f}'
              f'  edge_pos_frac={stats["edge_pos_frac"]:.4f}'
              f'  → {pt_path.name}')

    # ── Save selected rule JSON ───────────────────────────────────────────────
    rule_out = {
        'dataset':     args.dataset,
        'variant':     args.variant,
        'rule_index':  args.rule_index,
        'motifs':      sorted(all_motifs),
        'clauses':     [sorted(cl) for cl in rule_clauses],
        'score':       rule.get('score'),
        'pct_match':   rule.get('pct_match'),
        'balance':     rule.get('balance'),
        'separation':  rule.get('separation'),
    }
    rule_path = out_base / 'selected_rule.json'
    with open(rule_path, 'w') as f:
        json.dump(rule_out, f, indent=2)

    print(f'\n  Saved to: {out_base}/')
    for split_name, _, _ in split_configs:
        p = out_base / f'{split_name}_with_gt.pt'
        kb = p.stat().st_size // 1024
        print(f'    {split_name}_with_gt.pt  ({kb} KB)')
    print(f'    selected_rule.json')
    print(f'\n  Phase 4 complete.')


if __name__ == '__main__':
    main()
