"""eval_driver_common.py — shared reload/loader plumbing for the two split-eval
drivers (eval_driver_antehoc.py, eval_driver_posthoc.py).

Both drivers reload finished checkpoints and run SharedModules.evaluation.split_eval
per split; only the model-reload + score-source differ (post-hoc explainers make
their own scores; ante-hoc uses the model's own attention). Everything up to that
point — GT-loader building, the train/valid/test split lists, run iteration — is
identical and lives here.
"""
from __future__ import annotations

import json
from pathlib import Path

SPLITS = ('train', 'valid', 'test')


def _proc_root(meta, data_root, processed_root, variant):
    from SharedModules.data.dataset_routing import (
        default_processed_base, variant_processed_root)
    if processed_root:
        return variant_processed_root(str(processed_root), variant)
    if meta.get('processed_root'):
        return str(meta['processed_root'])
    return variant_processed_root(default_processed_base(str(data_root), None), variant)


def build_gt_loaders(meta, data_root, vocab_root, processed_root, batch_size=128):
    """(loaders {train,valid,test}, vocab, dmeta, task_type). When the run used GT,
    apply_gt_loaders replaces ALL THREE splits with GT-relabelled graphs carrying
    node_label(_*). Mirrors MOSE-GNN/run.py + eval_antehoc_dnf_gtroc."""
    from SharedModules.data.vocab import load_vocab
    from SharedModules.data.loader import get_loaders, apply_gt_loaders, TASK_TYPE
    ds = meta['dataset']
    fold = int(meta.get('fold', 0))
    variant = meta.get('vocab_variant')
    vocab = load_vocab(str(vocab_root), ds, variant)
    task_type = TASK_TYPE.get(ds, 'BinaryClass')
    loaders, test_ds, dmeta = get_loaders(
        dataset=ds, data_root=str(data_root), fold=fold, vocab=vocab,
        processed_root=_proc_root(meta, data_root, processed_root, variant),
        batch_size=batch_size, normalize=(task_type == 'Regression'))
    if meta.get('use_gt') and meta.get('gt_cache'):
        gt_tier = meta.get('gt_tier')
        gt_vocab = (meta.get('gt_vocab_variant') or variant).replace('_filter', '')
        loaders, test_ds = apply_gt_loaders(
            loaders, test_ds, gt_cache=meta['gt_cache'], dataset=ds, fold=fold,
            vocab_variant=variant, batch_size=batch_size, gt_vocab_variant=gt_vocab,
            gt_relabel_dir=(f'relabel_{gt_tier}' if gt_tier else None),
            refresh_vocab=vocab if gt_vocab != variant else None,
            fold_motif_lookup=getattr(dmeta, 'motif_lookup', None),
            apply_threshold=getattr(dmeta, 'threshold_pct', None) is not None)
    return loaders, vocab, dmeta, task_type


def split_lists_and_gt(loaders, meta):
    """{split: [Data]} + the GT-eval subset per split (full split for most datasets;
    the mutagen subset for mutag; None when the run has no GT)."""
    split_lists = {s: list(loaders[s].dataset) for s in SPLITS}
    if not meta.get('use_gt'):
        return split_lists, {s: None for s in SPLITS}
    if meta['dataset'] == 'mutag':
        from SharedModules.data.mutag_splits import mutag_gt_eval_graphs
        gt = {s: mutag_gt_eval_graphs(split_lists[s]) for s in SPLITS}
    else:
        gt = {s: split_lists[s] for s in SPLITS}
    return split_lists, gt


def denorm_of(dmeta, task_type):
    return (dmeta.norm_mean, dmeta.norm_std) if task_type == 'Regression' else None


def iter_runs(out_root, allowed_families, datasets, shard):
    """Yield (run_dir, meta, family) for finished checkpoints under out_root, filtered
    by family + dataset + shard. Skips archives and summary-less / corrupt runs."""
    from analysis.aggregate_experiments import (
        family_of, dataset_allowed, ARCHIVE_PREFIXES)
    out_root = Path(out_root)
    runs = sorted({p.parent for p in out_root.rglob('best_model.pt')})
    runs = [rd for rd in runs if not any(
        part.startswith(ARCHIVE_PREFIXES) for part in rd.relative_to(out_root).parts)]
    if datasets:
        runs = [rd for rd in runs if dataset_allowed(rd, set(datasets))]
    if shard:
        i, n = (int(x) for x in shard.split('/'))
        runs = runs[i::n]
    for rd in runs:
        sj = rd / 'summary.json'
        if not sj.exists():
            continue
        try:
            meta = json.load(open(sj, encoding='utf-8'))
            fam = family_of(meta)
        except Exception:
            continue
        if fam in allowed_families:
            yield rd, meta, fam


def out_dir_for(run_dir, out_root, dest_root):
    d = (Path(dest_root) / Path(run_dir).relative_to(out_root)
         if dest_root else Path(run_dir))
    d.mkdir(parents=True, exist_ok=True)
    return d
