#!/usr/bin/env python3
"""run_multi_explanation.py — post-hoc H0/H1/H2 analysis on trained checkpoints.

Compatible with MOSE-GNN, MotifSAT (readout), and base GSAT (node attention,
``learn_edge_att=False``). Skips vanilla/baselines (no learned motif scores)
and edge-attention GSAT runs.

Usage
-----
    # All ante-hoc runs under an output tree
    python3 analysis/run_multi_explanation.py \\
        --out_root results --data_root $DATA_ROOT --vocab_root $VOCAB_ROOT

    # Single run directory
    python3 analysis/run_multi_explanation.py \\
        --run_dir results/mose/rbrics_filter/Mutagenicity/fold0/<tag> \\
        --data_root $DATA_ROOT --vocab_root $VOCAB_ROOT
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.aggregate_experiments import resolve_family, dataset_allowed
from analysis.probe_masked_nodes import _load_model_and_data, _is_probeable_run
from SharedModules.evaluation.multi_explanation_posthoc import run_multi_explanation_posthoc


def _att_aggregate_fn():
    """Lazy import of MotifSAT attention aggregation (avoids import at module load)."""
    msat = str(REPO / 'MotifSAT')
    if msat not in sys.path:
        sys.path.insert(0, msat)
    from run import _aggregate_att_to_motif  # noqa: E402
    return _aggregate_att_to_motif


def _run_one(run_dir: Path, data_root: str, vocab_root: str, device, local_filter: str) -> bool:
    model, test_list, status = _load_model_and_data(run_dir, data_root, vocab_root, device)
    if status != 'ok':
        print(f'  [skip] {run_dir}: {status}')
        return False

    sj = run_dir / 'summary.json'
    with open(sj, encoding='utf-8') as f:
        meta = json.load(f)

    learn_edge_att = bool(meta.get('learn_edge_att', False))
    fam = resolve_family(meta, str(run_dir))
    if fam in ('vanilla', 'baselines'):
        print(f'  [skip] {run_dir}: post-hoc explainers have no global motif scores')
        return False

    from SharedModules.data.loader import TASK_TYPE
    task_type = TASK_TYPE.get(meta.get('dataset', ''), 'BinaryClass')

    from SharedModules.data.vocab import load_vocab
    vocab = load_vocab(vocab_root, meta['dataset'], meta.get('vocab_variant', ''))

    # Mutag needs SMILES→graph index maps to remap motif masks (explicit-H
    # atoms make masks shorter than the graph); without them every mutag
    # ablation would be skipped.
    _index_maps = None
    if meta.get('dataset') == 'mutag':
        from SharedModules.data.dataset_routing import load_mutag_eval_index_maps
        _index_maps = load_mutag_eval_index_maps(
            data_root, int(meta.get('fold', 0)))

    agg_fn = _att_aggregate_fn() if fam in ('gsat', 'motifsat') else None
    ok = run_multi_explanation_posthoc(
        model, vocab, test_list, device, task_type, run_dir,
        learn_edge_att=learn_edge_att,
        att_aggregate_fn=agg_fn,
        max_motifs=meta.get('max_motifs_eval'),
        local_filter=local_filter,
        index_maps=_index_maps,
    )
    if ok:
        meta['run_multi_explanation'] = True
        meta['multi_explanation_posthoc'] = True
        with open(sj, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
    return ok


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run_dir', default=None,
                    help='Single run directory (summary.json + best_model.pt).')
    ap.add_argument('--out_root', default=None,
                    help='Walk this tree for MOSE/MotifSAT/GSAT checkpoints.')
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--vocab_root', required=True)
    ap.add_argument('--local_filter', default='p75',
                    choices=['global', 'p50', 'p75', 'beat_unk'])
    ap.add_argument('--dataset', nargs='*', default=None,
                    help='only process runs for these dataset(s), e.g. --dataset mutag')
    ap.add_argument('--families', nargs='*', default=['mose', 'motifsat', 'gsat'],
                    help='Only process these families (default: ante-hoc models).')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    allowed = set(args.families)
    datasets = set(args.dataset) if args.dataset else None

    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    elif args.out_root:
        run_dirs = []
        for p in Path(args.out_root).rglob('summary.json'):
            if not _is_probeable_run(p):
                continue
            if datasets and not dataset_allowed(p, datasets):
                continue
            try:
                with open(p, encoding='utf-8') as f:
                    meta = json.load(f)
                fam = resolve_family(meta, str(p.parent))
            except Exception:
                continue
            if fam not in allowed and not (fam == 'gsat' and 'gsat' in allowed):
                continue
            run_dirs.append(p.parent)
    else:
        raise SystemExit('provide --run_dir or --out_root')

    ok = fail = skip = 0
    for rd in sorted(set(run_dirs)):
        print(f'## {rd}')
        try:
            if _run_one(rd, args.data_root, args.vocab_root, device, args.local_filter):
                ok += 1
            else:
                skip += 1
        except Exception as e:
            print(f'  [fail] {rd}: {e}')
            fail += 1

    print(f'\nDone. succeeded={ok} skipped={skip} failed={fail}')


if __name__ == '__main__':
    main()
