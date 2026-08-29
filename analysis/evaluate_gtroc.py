#!/usr/bin/env python3
"""evaluate_gtroc.py — INSTANCE GT-ROC only, from the SAME node scores evaluate.py uses,
computed for train / valid / test / all splits, in FULL and FILTERED conditions.

WHAT IT IS
  A focused sibling of analysis/evaluate.py. It reuses evaluate.py's exact loaders,
  ground-truth attachment, node-attribution extraction, and AUC primitives — it does NOT
  re-derive any of them — and changes ONLY what is reported:

    * metric      : INSTANCE GT-ROC only (per-graph AUC, then mean over graphs). For planted
                    (DNF) datasets this is `_dnf_gtroc`'s instance = the MAX over the graph's
                    FIRED clauses (disjuncts); for source-GT it is the single-cause node AUC.
                    Global GT-ROC is deliberately not computed.
    * splits      : train, valid, test, AND `all` = ONE AUC recomputed over the COMBINED
                    train+valid+test graphs (not a pool of per-split means).
    * conditions  : FULL  (--unk include : every node's score vs GT on every node), and
                    FILTERED (--unk exclude : UNK-motif nodes are dropped from BOTH the score
                    and the GT target; UNK = a node whose motif is not in the per-fold kept
                    list). One run -> both conditions from the same node scores + one keep mask.

NODE SCORES (node-direct, as evaluate.py grades them; never a re-broadcast):
    * native (mose / mose_U / gsat / motifsat) : model's own per-node attention via
      model_node_att_fn (forward pass, deterministic). NOTE MoSE/Occlusion are motif-gated,
      so every atom in a motif shares that motif's score — that is the model's real output.
    * gnnexplainer / pgexplainer / motif_occlusion : the saved per-node atts from
      explainer_importances.json under --artifacts_root (NOT recomputed — post-hoc atts are
      stochastic; we read exactly what produced the paper).
    * mage : mage_v2 is reconstructed from the saved mage_attention.pt (the json only holds
      mage_official). Reload attention -> score_mage(attention_mean) -> _pi_to_node_atts.

OUTPUT: one CSV row per (tier, dataset, gt_rule, method, backbone, fold, unk_mode) with
    gtroc_full_{train,valid,test,all} and gtroc_filt_{train,valid,test,all}.

It writes NOTHING back into the run trees, changes NO existing file layout, and RE-RUNS NO
models (except MAGE's forward pass, which is inherent to reconstructing mage_v2). CLI mirrors
evaluate.py so the same (--method/--vocab/--gt_tier/--ckpt_root/--artifacts_root/...) select
the same runs.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Reuse evaluate.py's machinery verbatim — do not re-implement loaders / AUC / alignment.
from analysis.evaluate import (
    build_gt_loaders, split_lists_and_gt,                      # loaders + GT
    read_saved_atts, load_native_model_and_scores, _load_vanilla,  # atts sources
    _keep_fn, gtroc_all, _np1,                                 # AUC primitives
    _iter_runs, FAMILY_DIR, POSTHOC, NATIVE, DISK_STEM,        # run enumeration
    _base_vocab, _run_backbone, _run_unk_mode, _run_tier, _gt_rule,
)
from SharedModules.evaluation.motif_eval import model_node_att_fn

SPLITS = ('train', 'valid', 'test')
INSTANCE_KEY = 'instance_gt_roc_node_auc_mean'   # gtroc_all sets this from _dnf_gtroc (planted)
                                                 # or the single-cause alias (source-GT)
READ_POSTHOC = {'gnnexplainer', 'pgexplainer', 'motif_occlusion'}


# ── node-attribution extraction, per method (mirrors evaluate.py eval_run) ──────────────────
def _atts_native(run_dir, meta, method, loaders, vocab, dmeta, device, tt, split_lists):
    """Native per-node atts via the model's own attention (forward pass, deterministic)."""
    model, _scores = load_native_model_and_scores(
        run_dir, meta, method, loaders, vocab, dmeta, device, tt)
    base = model_node_att_fn(model, device)
    out = {}
    for s in SPLITS:
        d = {}
        for gi, g in enumerate(split_lists[s]):
            a = base(g)
            if a is not None:
                d[gi] = _np1(a)
        out[s] = d
    return out


def _atts_read(art_dir, method):
    """Post-hoc per-node atts read from the saved explainer_importances.json (no recompute)."""
    return read_saved_atts(art_dir, DISK_STEM.get(method, method))


def _atts_mage(run_dir, art_dir, meta, dmeta, vocab, split_lists, device, tt):
    """mage_v2 per-node atts: reload mage_attention.pt, re-score attention_mean, broadcast to
    nodes. (The json stores mage_official, not mage_v2 — so we reconstruct, exactly as the
    now-removed evaluate.py.mage_reload_atts did.)"""
    from SharedModules.baselines.mage import _MotifAttention, score_mage
    from SharedModules.evaluation.split_eval import _pi_to_node_atts
    p = Path(art_dir) / 'mage_attention.pt'
    if not p.exists():
        return {s: {} for s in SPLITS}
    sd = torch.load(p, map_location='cpu', weights_only=False)
    attn = _MotifAttention(sd['W.weight'].shape[0])
    attn.load_state_dict(sd)
    attn = attn.to(device).eval()
    model = _load_vanilla(run_dir, meta, dmeta, device)
    out = {}
    for s in SPLITS:
        sl = split_lists.get(s) or []
        _sc, pi = score_mage(model, attn, sl, vocab, device, tt, return_per_instance=True,
                             verbose=False, score_mode='attention_mean', embed_batch_size=256)
        out[s] = {gi: _np1(a) for gi, a in _pi_to_node_atts(pi, sl).items()}
    return out


# ── per-fold kept motif ids (defines UNK for the FILTERED condition) ────────────────────────
def _kept_for_fold(dataset, fold, vocab_root, data_root, base_vocab):
    from SharedModules.data.vocab import load_vocab
    from SharedModules.data.fold_threshold import build_fold_annotation
    from SharedModules.data.dataset_schema import DATASET_COLUMN
    filt = load_vocab(str(vocab_root), dataset, base_vocab + '_filter')
    csv = Path(data_root) / f'{dataset}_{int(fold)}.csv'
    _, kept, _, _ = build_fold_annotation(
        lookup_all=filt.lookup_all, motif_list=filt.motif_list,
        mol_fragment_smarts=filt.mol_fragment_smarts, csv_path=str(csv),
        label_col=DATASET_COLUMN[dataset], dataset=dataset,
        variant=filt.variant or (base_vocab + '_filter'),
        vocab_dir=Path(filt.vocab_dir) if filt.vocab_dir else Path('.'),
        apply_threshold=True, threshold_pct=filt.threshold_pct)
    return None if kept is None else set(int(x) for x in kept)


# ── alignment: atts keyed by split-list position -> the GT graph list (a subset) ─────────────
def _att_by_i(atts_split, gl, sl):
    pos = {id(g): j for j, g in enumerate(sl)}
    return {i: atts_split[pos[id(g)]] for i, g in enumerate(gl)
            if id(g) in pos and pos[id(g)] in atts_split}


def _instance(att_by_i, gl, keep_fn):
    # No atts / no GT graphs on this split is legitimate (e.g. a method that produced nothing
    # here) -> NaN. But if there ARE graphs and gtroc_all returns no instance key at all, the
    # graphs carry no node_label / node_label_clauses -> a mis-pointed tier or unattached GT.
    # Fail loud rather than emit a silent blank that reads as "computed, got nothing".
    if not att_by_i or not gl:
        return float('nan')
    res = gtroc_all(att_by_i, gl, keep_fn)
    if INSTANCE_KEY not in res:
        raise ValueError(
            f'{INSTANCE_KEY} absent for {len(gl)} graphs: they carry no node_label / '
            f'node_label_clauses. Wrong --gt_tier, or GT not attached for this dataset.')
    return res[INSTANCE_KEY]


def _combined(gt, atts, split_lists):
    """Build the COMBINED train+valid+test graph list + its att_by_i (re-indexed) so `all`
    is one AUC over all graphs, not a pool of per-split means."""
    gl_all, att_all, off = [], {}, 0
    for s in SPLITS:
        gl = gt.get(s) or []
        abi = _att_by_i(atts.get(s, {}), gl, split_lists[s]) if gl else {}
        for i, g in enumerate(gl):
            gl_all.append(g)
            if i in abi:
                att_all[off + i] = abi[i]
        off += len(gl)
    return gl_all, att_all


def eval_one(method, run_dir, art_dir, meta, args, device, kept):
    loaders, vocab, dmeta, tt = build_gt_loaders(
        meta, args.data_root, args.vocab_root, args.processed_root, args.loader_batch_size)
    split_lists, gt = split_lists_and_gt(loaders, meta)
    if all(gt.get(s) is None for s in SPLITS):
        raise ValueError('no node-GT on any split (nothing to grade)')

    # Read the persisted per-node atts FIRST for every method (mose/gsat under their eval/artifacts
    # subtree, mage_v2 + post-hoc under posthoc_gtroc). A forward pass runs only as a fallback when
    # nothing is on disk -> no needless re-embedding when the scores already exist.
    atts = _atts_read(art_dir, method)
    if any(atts.get(s) for s in SPLITS):
        src = 'read'
    elif method in NATIVE:
        atts = _atts_native(run_dir, meta, method, loaders, vocab, dmeta, device, tt, split_lists)
        src = 'recompute'
    elif method == 'mage':
        atts = _atts_mage(run_dir, art_dir, meta, dmeta, vocab, split_lists, device, tt)
        src = 'recompute'
    elif method in READ_POSTHOC:
        src = 'read'   # post-hoc has no recompute path; empty atts -> downstream NaN / fail-loud
    else:
        raise SystemExit(f'unsupported method {method}')
    eval_one.last_src = src

    row = {}
    for cond, unk in (('full', 'include'), ('filt', 'exclude')):
        keep_fn = _keep_fn(kept if cond == 'filt' else None, unk)
        for s in SPLITS:
            gl = gt.get(s)
            abi = _att_by_i(atts.get(s, {}), gl, split_lists[s]) if gl else {}
            row[f'gtroc_{cond}_{s}'] = _instance(abi, gl, keep_fn)
        gl_all, abi_all = _combined(gt, atts, split_lists)
        row[f'gtroc_{cond}_all'] = _instance(abi_all, gl_all, keep_fn)
    return row


def main():
    ap = argparse.ArgumentParser(description='Instance GT-ROC (full+filtered, all 4 splits) '
                                             'from saved node scores — no re-training.')
    ap.add_argument('--method', required=True, choices=sorted(NATIVE | READ_POSTHOC | {'mage'}))
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--vocab', required=True, help='eval vocab, e.g. rbrics / rbrics_filter / '
                                                   'size_frequency_optimization')
    ap.add_argument('--gt_tier', required=True, choices=['source', 'planted'])
    ap.add_argument('--ckpt_root', required=True, help='model tree to enumerate (e.g. final_v2, '
                                                       'final_v2_gath1, planted_v2/<ds>/<rule>)')
    ap.add_argument('--artifacts_root', default=None,
                    help='post-hoc atts tree (posthoc_v1 / final_v2_gath1 / <planted>/posthoc_gtroc); '
                         'required for gnn/pg/occlusion/mage')
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--vocab_root', required=True)
    ap.add_argument('--processed_root', default=None)
    ap.add_argument('--dest', required=True, help='rollup CSV (append; one row per cell)')
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'auto'])
    ap.add_argument('--unk_mode', default='any', choices=['any', 'fixed', 'learnable_shared'])
    ap.add_argument('--fold', nargs='*', type=int, default=None)
    ap.add_argument('--backbone', nargs='*', default=None)
    ap.add_argument('--loader_batch_size', type=int, default=128)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--weight_vocab', default=None,
                    help='NATIVE reuse: the vocab the MODEL was TRAINED under, when it differs '
                         'from --vocab (the EVAL vocab). Run discovery + weight loading use '
                         '--weight_vocab; loaders / kept / GT use --vocab. For SFO GSAT (a '
                         'rBRICS-trained, vocab-independent GSAT scored under SFO). Refused for '
                         'MoSE (its motif-gated head is sized by the vocab).')
    ap.add_argument('--posthoc_vocab_override', default=None,
                    help='POST-HOC atts path only: when composing art_dir, swap the eval-vocab '
                         'path segment for this value. For SFO gnn/pg whose vocab-independent '
                         'atts live under the rBRICS path inside --artifacts_root '
                         '(e.g. --posthoc_vocab_override rbrics).')
    args = ap.parse_args()

    if args.method in ({'mage'} | READ_POSTHOC) and not args.artifacts_root:
        raise SystemExit('post-hoc / mage need saved atts — pass --artifacts_root')
    if args.weight_vocab and args.method == 'mose':
        raise SystemExit('--weight_vocab is not valid for MoSE: its motif-gated head is sized by '
                         'the vocabulary, so a model trained on one fragmentation cannot be scored '
                         'on another. Vocab-independent natives (GSAT) only.')
    device = torch.device('cuda' if (args.device == 'cuda'
                          or (args.device == 'auto' and torch.cuda.is_available())) else 'cpu')
    want_base = _base_vocab(args.vocab)               # EVAL vocab (loaders / kept / GT / dest)
    want_filter = args.vocab.endswith('_filter')
    # Discovery + weight loading key on the WEIGHT vocab (defaults to eval vocab -> unchanged).
    _wt = args.weight_vocab or args.vocab
    want_base_wt = _base_vocab(_wt)
    want_filter_wt = _wt.endswith('_filter')

    dest = Path(args.dest); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)                       # append-mode rollup: clear once
    out_rows, ok, failed = [], 0, 0
    print(f'method={args.method} dataset={args.dataset} vocab={args.vocab} tier={args.gt_tier} '
          f'device={device}')

    for run_dir, meta in _iter_runs(args.ckpt_root, FAMILY_DIR[args.method], args.dataset):
        v = str(meta.get('vocab_variant', ''))
        if _base_vocab(v) != want_base_wt or v.endswith('_filter') != want_filter_wt:
            continue
        if _run_tier(meta, run_dir) != args.gt_tier:
            continue
        if args.fold is not None and meta.get('fold') not in args.fold:
            continue
        if args.unk_mode != 'any' and _run_unk_mode(meta, run_dir) != args.unk_mode:
            continue
        if args.backbone is not None and _run_backbone(meta, run_dir) not in args.backbone:
            continue
        if args.limit and (ok + failed) >= args.limit:
            break
        bb = _run_backbone(meta, run_dir)
        rule = _gt_rule(meta, run_dir)
        fold = meta.get('fold')
        tag = f'{args.dataset}/{args.method}/{bb}/fold{fold}/{rule or "-"}'
        try:
            rel = run_dir.relative_to(args.ckpt_root)
            if args.posthoc_vocab_override:
                # the run path carries the eval-vocab segment (e.g. size_frequency_optimization);
                # the vocab-independent atts live under a different vocab segment in artifacts_root
                # (e.g. rbrics). Swap that one path part so art_dir points at the real atts.
                rel = Path(*[args.posthoc_vocab_override if p == want_base else p
                             for p in rel.parts])
            art_dir = (Path(args.artifacts_root) / rel if args.artifacts_root else run_dir)
            # NATIVE reuse: model discovered under the weight vocab, but loaders / GT / motif
            # aggregation must use the eval vocab -> override the vocab keys on a meta copy.
            eval_meta = meta
            if args.weight_vocab:
                eval_meta = {**meta, 'vocab_variant': want_base, 'gt_vocab_variant': want_base}
            kept = _kept_for_fold(args.dataset, fold, args.vocab_root, args.data_root, want_base)
            m = eval_one(args.method, run_dir, art_dir, eval_meta, args, device, kept)
            src = getattr(eval_one, 'last_src', '?')
            out_rows.append(dict(
                tier=args.gt_tier, dataset=args.dataset, gt_rule=(rule or '-'),
                method=args.method, backbone=bb, fold=fold,
                unk_mode=_run_unk_mode(meta, run_dir), **m,
                run_path=str(run_dir.relative_to(args.ckpt_root))))
            print(f'  [ok] {tag} atts={src}')
            ok += 1
            _hdr = not dest.exists() or dest.stat().st_size == 0
            pd.DataFrame([out_rows[-1]]).to_csv(dest, mode='a', header=_hdr, index=False)
        except Exception as e:
            print(f'  [FAIL] {tag}: {type(e).__name__}: {e}')
            failed += 1

    print(f'\nDONE ok={ok} failed={failed} -> {args.dest}')
    if out_rows:
        cols = ['backbone', 'fold', 'gt_rule',
                'gtroc_full_test', 'gtroc_full_all', 'gtroc_filt_test', 'gtroc_filt_all']
        print(pd.DataFrame(out_rows)[cols].to_string(index=False))


if __name__ == '__main__':
    main()
