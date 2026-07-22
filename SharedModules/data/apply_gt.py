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
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

# Make the project importable from any working directory
_HERE = Path(__file__).resolve()
_PROJECT = _HERE.parents[2]
for _p in [str(_PROJECT), str(_HERE.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from SharedModules.data.loader import get_loaders
from SharedModules.data.vocab import load_vocab
from SharedModules.utils import set_seed

# ─────────────────────────────────────────────────────────────────────────────
# Rule loading
# ─────────────────────────────────────────────────────────────────────────────

def load_tier_rule(vocab_root: str, dataset: str, variant: str,
                   tier: str) -> dict:
    """Load one difficulty tier's rule from rule_tiers.json (written by
    generate_vocab_rules.py --rule_tiers). Returns an apply_gt-normalised rule
    dict with '_clauses' = list of motif-key sets. Raises if the file or tier
    is missing."""
    tiers_path = Path(vocab_root) / dataset / variant / 'rule_tiers.json'
    if not tiers_path.exists():
        raise FileNotFoundError(
            f"rule_tiers.json not found: {tiers_path}\n"
            f"Regenerate the vocab with --rule_tiers: "
            f"bash run_experiments.sh phase1  (RULE_TIERS=1)"
        )
    with open(tiers_path) as f:
        tiers = json.load(f)
    if tier not in tiers:
        raise KeyError(
            f"tier {tier!r} not in {tiers_path} — available: {sorted(tiers)}"
        )
    rule = dict(tiers[tier])
    clauses = rule.get('clauses')
    if not clauses:
        raise ValueError(f"tier {tier!r} rule has no 'clauses': {list(rule.keys())}")
    rule['_clauses'] = [set(cl.get('motifs', [])) for cl in clauses]
    return rule


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

def _validate_rule_nodes(rule_nodes: list, n_nodes: int, smi: str) -> list:
    """Ensure lookup node indices align with graph atom count."""
    bad = [i for i in rule_nodes if i < 0 or i >= n_nodes]
    if bad:
        raise ValueError(
            f"Rule node indices {bad} out of range [0, {n_nodes}) for "
            f"smiles={smi!r}. Lookup/graph atom count mismatch — re-export "
            "vocab or check index_maps."
        )
    return rule_nodes


def validate_rule_threshold_consistency(
    rule_clauses: List[Set[str]],
    motif_list: List[str],
    kept_motif_ids: Optional[List[int]],
    apply_threshold: bool,
    *,
    dataset: str,
    variant: str,
    fold: int,
) -> None:
    """Fail if any rule-motif SMARTS would be ``-1`` under this fold's threshold.

    Rules are mined from high-support motifs on the filtered vocab, so this
    should not trigger in normal operation. When it does, training would see
    unknown (-1) nodes for motifs the GT treats as present.
    """
    if not apply_threshold or kept_motif_ids is None:
        return

    rule_motifs = {m for cl in rule_clauses for m in cl}
    if not rule_motifs:
        return

    kept_smarts = {
        motif_list[i] for i in kept_motif_ids
        if 0 <= i < len(motif_list)
    }
    below_threshold = sorted(rule_motifs - kept_smarts)
    if below_threshold:
        raise ValueError(
            f"Rule references motif(s) below fold-{fold} threshold for "
            f"{dataset}/{variant!r} — they would be motif_id=-1 during "
            f"training but are used for GT rule presence:\n"
            f"  {below_threshold}\n"
            f"kept={len(kept_motif_ids)}/{len(motif_list)} motifs on this fold.\n"
            f"Pick a different RULE_INDEX, re-mine rules on the filtered vocab, "
            f"or adjust CHOSEN_THRESHOLD."
        )


def validate_rule_fires_on_trainable_motifs(
    data_list: List,
    rule_clauses: List[Set[str]],
    graph_lookup: Dict[str, Dict[int, Tuple[str, int]]],
    *,
    dataset: str,
    variant: str,
    fold: int,
    split: str,
) -> None:
    """Fail if a rule fires on pre-threshold fragments but rule atoms are -1 on graphs."""
    all_rule_motifs = {m for cl in rule_clauses for m in cl}
    violations: List[Tuple[str, str, int]] = []

    for data in data_list:
        smi = getattr(data, 'smiles', None)
        if not smi:
            continue
        node_map = graph_lookup.get(smi, {})
        if not node_map:
            continue

        frag_set = {smarts for smarts, _mid in node_map.values()}
        rule_fires = any(cl.issubset(frag_set) for cl in rule_clauses if cl)
        if not rule_fires:
            continue

        n2m = getattr(data, 'nodes_to_motifs', None)
        if n2m is None:
            continue

        for idx, (smarts, _mid) in node_map.items():
            if smarts not in all_rule_motifs:
                continue
            if int(n2m[idx].item()) < 0:
                violations.append((smi, smarts, idx))

    if violations:
        sample = violations[:5]
        detail = '\n'.join(
            f"  smiles={s!r} motif={m!r} atom={i}" for s, m, i in sample
        )
        extra = (
            f"\n  … and {len(violations) - len(sample)} more"
            if len(violations) > len(sample) else ''
        )
        raise ValueError(
            f"Rule fired on pre-threshold fragments but {len(violations)} "
            f"rule-motif atom(s) are motif_id=-1 on fold-{fold} {split} graphs "
            f"for {dataset}/{variant!r}:\n{detail}{extra}\n"
            f"GT rule presence uses pre-threshold lookup_all; training uses "
            f"fold-specific threshold. This mismatch is not allowed."
        )


def choose_spurious_motif(data_lists: List[List],
                          rule_clauses: List[Set[str]],
                          graph_lookup: Dict[str, Dict[int, Tuple[str, int]]],
                          min_support: int = 10) -> Tuple[Optional[str], float]:
    """Pick the strongest SPURIOUS (non-GT) motif: the motif — not part of any
    rule clause — whose per-molecule presence most POSITIVELY correlates with the
    synthetic label (the rule firing). This is the shortcut that competes with the
    true cause; a fooled explainer attributes to it. Returns (smarts, corr) or
    (None, nan) if no eligible motif clears ``min_support``.

    Computed corpus-wide (all splits) for a stable choice, then the same motif is
    annotated across every split so spurious_roc is comparable across folds/splits.
    """
    rule_motifs = {m for cl in rule_clauses for m in cl}
    ys: List[float] = []
    present: Dict[str, List[int]] = {}
    row = 0
    for dl in data_lists:
        for data in dl:
            smi = getattr(data, 'smiles', None)
            node_map = graph_lookup.get(smi, {}) if smi else {}
            frag_set = {smarts for smarts, _mid in node_map.values()}
            fires = any(cl.issubset(frag_set) for cl in rule_clauses if cl)
            ys.append(1.0 if fires else 0.0)
            for smarts in frag_set:
                if smarts in rule_motifs:
                    continue
                present.setdefault(smarts, []).append(row)
            row += 1
    if not ys:
        return None, float('nan')
    y = np.asarray(ys, dtype=float)
    n = len(y)
    if y.std() == 0:
        return None, float('nan')
    best_m, best_corr = None, -2.0
    for smarts, rows in present.items():
        if len(rows) < min_support or n - len(rows) < min_support:
            continue
        p = np.zeros(n, dtype=float)
        p[rows] = 1.0
        if p.std() == 0:
            continue
        corr = float(np.corrcoef(p, y)[0, 1])
        if corr > best_corr:
            best_corr, best_m = corr, smarts
    if best_m is None:
        return None, float('nan')
    return best_m, round(best_corr, 4)


def annotate_split(data_list: List,
                   rule_clauses: List[Set[str]],
                   graph_lookup: Dict[str, Dict[int, Tuple[str, int]]],
                   relabel: bool = True,
                   spurious_motif: Optional[str] = None) -> Tuple[List, Dict]:
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
    n_pos_spurious_nodes = 0
    n_graphs_with_spurious = 0

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
        node_label = torch.zeros(n_nodes, dtype=torch.float32)         # Mode 1: whole-rule
        node_label_fired = torch.zeros(n_nodes, dtype=torch.float32)   # Mode 2: fired-clause only
        node_label_spurious = torch.zeros(n_nodes, dtype=torch.float32)  # strongest non-GT shortcut
        edge_label = torch.zeros(n_edges, dtype=torch.float32)

        # Spurious GT: atoms of the chosen shortcut motif, marked on the SAME positive
        # graphs as node_label so spurious_roc and gt_roc share a graph population. A
        # fooled explainer ranks these atoms high → high spurious_roc.
        if spurious_motif is not None and rule_fires and node_map:
            spur_nodes = [idx for idx, (smarts, _mid) in node_map.items()
                          if smarts == spurious_motif]
            if spur_nodes:
                spur_nodes = _validate_rule_nodes(spur_nodes, n_nodes, smi or '?')
                node_label_spurious[torch.tensor(spur_nodes, dtype=torch.long)] = 1.0
                n_pos_spurious_nodes += len(spur_nodes)
                n_graphs_with_spurious += 1

        if rule_fires and node_map:
            # TWO GT views (both attached; the evaluator picks one via compute_gt_roc(gt_attr=...)):
            #  Mode 1 — whole-rule / instance-agnostic: atoms of ANY motif in ANY clause.
            all_rule_motifs = {m for cl in rule_clauses for m in cl}
            #  Mode 2 — per-instance / OR-aware: atoms of motifs in the clause(s) that ACTUALLY FIRED
            #  here (a clause fires when ALL its motifs are present). A DNF needs only one fired
            #  clause, not all — so this excludes present-but-not-fired clauses' motifs.
            fired_motifs = {m for cl in rule_clauses
                            if cl and cl.issubset(frag_set) for m in cl}
            rule_nodes = [idx for idx, (smarts, _mid) in node_map.items()
                          if smarts in all_rule_motifs]
            fired_nodes = [idx for idx, (smarts, _mid) in node_map.items()
                           if smarts in fired_motifs]
            if rule_nodes:
                rule_nodes = _validate_rule_nodes(rule_nodes, n_nodes, smi or '?')
                active = torch.zeros(n_nodes, dtype=torch.bool)
                active[torch.tensor(rule_nodes, dtype=torch.long)] = True
                node_label[active] = 1.0
                n_pos_nodes += int(active.sum().item())
                if n_edges > 0:
                    src, dst = data.edge_index
                    pos = active[src] & active[dst]   # AND of both endpoints
                    edge_label[pos] = 1.0
                    n_pos_edges += int(pos.sum().item())
            if fired_nodes:
                fired_nodes = _validate_rule_nodes(fired_nodes, n_nodes, smi or '?')
                node_label_fired[torch.tensor(fired_nodes, dtype=torch.long)] = 1.0

        data.node_label = node_label               # Mode 1 (whole-rule) — the default GT
        data.node_label_fired = node_label_fired   # Mode 2 (per-instance fired-clause / OR-aware)
        data.node_label_spurious = node_label_spurious  # strongest non-GT shortcut motif
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
        'n_pos_spurious_nodes':    n_pos_spurious_nodes,
        'n_graphs_with_spurious':  n_graphs_with_spurious,
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
    parser.add_argument('--tier',        default=None,
                        choices=['easy', 'medium', 'hard'],
                        help='Relabel using a difficulty tier from rule_tiers.json '
                             'instead of rules.json[--rule_index]. Output goes to '
                             'relabel_<tier>/ (not relabel1/).')
    parser.add_argument('--data_root',   required=True)
    parser.add_argument('--out_dir',     required=True,
                        help='Root of gt_cache output directory')
    parser.add_argument('--processed_root', default=os.environ.get('PROCESSED_ROOT'),
                        help='Base PyG cache root ($PROCESSED_ROOT). '
                             'Variant cache goes under {root}/apply_gt/{variant}.')
    parser.add_argument('--no_relabel',  action='store_true',
                        help='Attach edge_label but keep original data.y')
    parser.add_argument('--batch_size',  type=int, default=128)
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    relabel = not args.no_relabel

    # Difficulty tiers are the CURRENT recipe. The single-best-rule path (rules.json
    # + --rule_index, → relabel1/) is deprecated: it plants ONE rule with no graded
    # difficulty and predates the tier design. Fail fast so no run silently uses it.
    if not args.tier:
        raise ValueError(
            "Single-best-rule relabelling (--rule_index / relabel1/) is DEPRECATED — "
            "difficulty tiers are required. Pass --tier {easy,medium,hard} (needs "
            "rule_tiers.json from `generate_vocab_rules.py --rule_tiers`; the pipeline "
            "builds it when RULE_TIERS=1).")

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
    if args.tier:
        print(f'  tier       = {args.tier}  (from rule_tiers.json)')
    else:
        print(f'  rule_index = {args.rule_index}')
    print(f'  out_dir    = {args.out_dir}')
    print(f'{"="*60}')

    # ── Load rule: difficulty tier (rule_tiers.json) or best rule (rules.json) ──
    if args.tier:
        rule = load_tier_rule(args.vocab_root, args.dataset, args.variant, args.tier)
    else:
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
    # Duplicate of _clauses print above (raw rules.json layout) — kept for debugging:
    # for i, cl in enumerate(rule.get("clauses", [])):
    #     print(f'    clause {i}   = {cl.get("motifs", [])}' )

    if not rule_clauses or not all_motifs:
        print('\n  [error] Rule has no motifs — check rules.json')
        sys.exit(1)

    # ── Load vocab (for graph_lookup) ─────────────────────────────────────────
    print('\n  Loading vocabulary...')
    vocab = load_vocab(args.vocab_root, args.dataset, args.variant)
    print(f'    {vocab.num_motifs} motifs')

    # Pre-threshold fragmentation for rule presence (threshold is per-fold on y/n2m).
    lookup = vocab.annotation_lookup

    # ── Load data loaders to get Data objects ─────────────────────────────────
    print('  Loading data loaders...')
    from SharedModules.data.dataset_routing import (
        default_processed_base,
        variant_processed_root,
    )
    base_proc = default_processed_base(args.data_root, args.processed_root)
    apply_gt_base = f'{base_proc.rstrip("/")}/apply_gt'
    proc_root = variant_processed_root(apply_gt_base, args.variant)
    loaders, test_ds, meta = get_loaders(
        dataset=args.dataset,
        data_root=args.data_root,
        fold=args.fold,
        vocab=vocab,
        processed_root=proc_root,
        batch_size=args.batch_size,
    )
    print(f'    train={len(loaders["train"].dataset)}  '
          f'valid={len(loaders["valid"].dataset)}  '
          f'test={len(test_ds)}')

    validate_rule_threshold_consistency(
        rule_clauses, vocab.motif_list, meta.kept_motif_ids,
        vocab.apply_threshold,
        dataset=args.dataset, variant=args.variant, fold=args.fold,
    )

    # ── Annotate each split ───────────────────────────────────────────────────
    split_configs = [
        ('train', loaders['train'].dataset, lookup),
        ('valid', loaders['valid'].dataset, lookup),
        ('test',  test_ds,                  lookup),
    ]

    for split_name, ds, _lookup in split_configs:
        data_list = [ds[i] for i in range(len(ds))]
        validate_rule_fires_on_trainable_motifs(
            data_list, rule_clauses, lookup,
            dataset=args.dataset, variant=args.variant,
            fold=args.fold, split=split_name,
        )

    # relabel_<tier>/ for a difficulty tier; relabel1//relabel0/ for a rules.json rule.
    if args.tier:
        _relabel_dir = f'relabel_{args.tier}' if relabel else f'relabel0_{args.tier}'
    else:
        _relabel_dir = 'relabel1' if relabel else 'relabel0'
    out_base = (Path(args.out_dir) / args.dataset
                / f'fold{args.fold}' / args.variant
                / _relabel_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # Choose the strongest spurious (non-GT) shortcut motif ONCE over all splits so
    # node_label_spurious (→ eval spurious_roc) marks the same motif everywhere.
    _all_split_lists = [[ds[i] for i in range(len(ds))]
                        for _sn, ds, _lk in split_configs]
    spurious_motif, spurious_corr = choose_spurious_motif(
        _all_split_lists, rule_clauses, lookup)
    print(f'\n  Spurious shortcut motif: {spurious_motif!r} '
          f'(corr with label = {spurious_corr})')

    all_stats = {}
    for (split_name, ds, lookup), data_list in zip(split_configs, _all_split_lists):
        annotated, stats = annotate_split(data_list, rule_clauses, lookup,
                                          relabel=relabel,
                                          spurious_motif=spurious_motif)
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
        'tier':        args.tier,                      # None for a rules.json rule
        'rule_index':  None if args.tier else args.rule_index,
        'motifs':      sorted(all_motifs),
        'clauses':     [sorted(cl) for cl in rule_clauses],
        'score':       rule.get('score'),
        'pct_match':   rule.get('pct_match'),
        'balance':     rule.get('balance'),
        'separation':  rule.get('separation'),
        # tier-only difficulty metadata (present when --tier):
        'tier_grader':    rule.get('grader'),           # 'lr' (default) or 'gnn'
        'tier_band':      rule.get('tier_band'),
        'foolability_auc': rule.get('foolability_auc'), # LR grader: shortcut availability
        'learnable_auc':  rule.get('learnable_auc'),    # LR grader: composition learnability
        'P2':          rule.get('P2'),                  # GNN grader only
        'P4':          rule.get('P4'),                  # GNN grader only
        'cov':         rule.get('cov'),
        'foolability': rule.get('foolability'),
        # spurious shortcut motif whose atoms → node_label_spurious → eval spurious_roc
        'spurious_motif':      spurious_motif,
        'spurious_motif_corr': spurious_corr,
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
