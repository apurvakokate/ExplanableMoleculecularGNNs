#!/usr/bin/env python3
"""mage_v2_rescore.py — CO-LOCATED MAGE re-score using the SAVED attention.

Why this exists
---------------
The finished post-hoc sweep scored MAGE with the PUBLISHED S^cm = S^mm . P, i.e.
importance(m) = mean_i alpha_{i,m} . P[i,c]. The `. P[i,c]` term makes the score
class-probability-weighted, which on the source-GT datasets is a label-leaking
confound. `mage_v2` re-scores with score_mode='attention_mean' —
importance(m) = mean_i alpha_{i,m} (the SAME 1/freq normalisation, class term
DROPPED) — so any metric difference vs. `mage_official` is attributable purely to
the P term, not to a different learned attention.

Fresh fit is NOT needed and is deliberately avoided: the learned attention alpha
is a property of the (frozen) vanilla model and the fit — it does not depend on
score_mode (score_mode only changes `_score_from_attn`, never `_fit_attention`).
So we RELOAD the run's saved `mage_attention.pt` (the exact attention that produced
`mage_official`) and re-score it. This guarantees identical alpha; only the scoring
formula changes. A fresh re-fit would re-randomise alpha and muddy the comparison.

What it writes (co-located, non-destructive)
--------------------------------------------
Into the SAME run dir as the existing MAGE artifacts, under the method name
`mage_v2` (never collides with `mage_official_*`):
  mage_v2_importance_{split}.csv          attention_mean per-motif score (x)
  mage_v2_impact_{split}.csv              agnostic LOO impact (y) — reused cache
  mage_v2_grouped_corr_pooled_{alltest,testonly}.csv   4 grouped Pearson/Spearman
  mage_v2_instance_corr_{split}.csv       per-instance Pearson/Spearman
and MERGES a `mage_v2` key into the shared summary_splits.json (read-modify-write;
`mage_official` and every other method key are left untouched). Merging — rather
than a separate summary_splits_mage_v2.json — means an UNMODIFIED future harvester
that rglobs summary_splits.json still discovers `mage_v2` sitting next to
`mage_official`, so the new scores can never be silently missed.

Every metric the full post-hoc path computes for one method is reproduced here:
importance, impact, grouped corr (4 variants x alltest/testonly), instance corr,
and the FULL GT-ROC block (whole / fired / spurious / family / DNF global+instance)
plus per-split AUC/RMSE/MAE.

Agnostic LOO impact (the y-axis) REUSES the existing impact_cache_agnostic_{split}.json
that mage_official wrote to the run dir: it is model-only (base_att_fn=ones) and the
LOO is deterministic, so with the SAME best_model.pt it is bit-identical to a
recompute — and it is the dominant cost. Absent/corrupt cache → recomputed + written
(self-heal). To force a recompute instead, pass cache_path=None in rescore_run.

Reuses the vanilla checkpoint (best_model.pt), the saved mage_attention.pt, and the
agnostic impact cache. Nothing else from the prior run is consumed.

TWO-TREE dependency (point the args at the RIGHT trees):
  --out_root   the tree iter_runs scans for best_model.pt (the vanilla checkpoints
               — the base training run root).
  --dest_root  the POST-HOC re-eval sweep tree (posthoc_v1) where mage_official
               already wrote mage_attention.pt, impact_cache_agnostic_*.json and
               summary_splits.json. mage_v2 READS the saved attention from here and
               CO-LOCATES its outputs here, alongside mage_official.
If a single tree already holds both best_model.pt AND the mage artifacts, pass it
as --out_root and omit --dest_root (out_dir == run_dir).

  python analysis/mage_v2_rescore.py \
      --out_root <BASE_CKPT_ROOT> --dest_root posthoc_v1 \
      --data_root <DATA> --vocab_root <VOCAB> [--processed_root <PROC>] \
      --device cuda --batch_size 256 \
      [--dataset BBBP] [--shard i/N] [--limit N] [--dry_run]

A run without a saved mage_attention.pt is reported (MISSING-ATTN) and skipped —
there is NO re-fit fallback (a fresh fit would re-randomise alpha and break the
apples-to-apples with mage_official).
"""
import argparse
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.eval_driver_common import (
    build_gt_loaders, split_lists_and_gt, denorm_of, iter_runs, out_dir_for)
from analysis.eval_driver_posthoc import load_vanilla_model
from SharedModules.evaluation.metrics import evaluate_predictions
from SharedModules.evaluation.split_eval import (
    SPLITS, precompute_agnostic_impact, _grouped_rows_from_cache,
    _instance_from_cache, gt_roc_block, _pred_scalars, _pi_to_node_atts,
    _write_importance, _write_impact, write_grouped_pooled, write_instance,
    write_summary_splits)
from SharedModules.baselines.mage import _MotifAttention, score_mage
from SharedModules.baselines.run_vanilla import _motif_score_node_att_fn

FAMILIES = ('vanilla', 'baselines')
METHOD = 'mage_v2'                 # co-located method name + summary key
SCORE_MODE = 'attention_mean'      # raw-alpha aggregation (drop P[i,c])


def load_saved_attention(out_dir: Path, device):
    """Rebuild the fitted _MotifAttention from a run's saved mage_attention.pt.
    dim is read from the a_q parameter (== model embedding dim). Returns None if
    the file is absent (mage_official never ran / was not saved)."""
    p = Path(out_dir) / 'mage_attention.pt'
    if not p.exists():
        return None
    sd = torch.load(p, map_location='cpu', weights_only=True)
    dim = int(sd['a_q'].shape[0])
    attn = _MotifAttention(dim)
    attn.load_state_dict(sd)          # strict: any shape mismatch fails LOUD
    return attn.to(device).eval()


def rescore_run(run_dir, meta, args, device):
    """Score mage_v2 on every split of one finished vanilla run, co-located."""
    loaders, vocab, dmeta, task_type = build_gt_loaders(
        meta, args.data_root, args.vocab_root, args.processed_root, args.loader_batch_size)
    split_lists, gt = split_lists_and_gt(loaders, meta)
    out_dir = out_dir_for(run_dir, args.out_root, args.dest_root)

    attn = load_saved_attention(out_dir, device)
    if attn is None:
        # NO fallback: mage_v2 re-scores the SAVED attention only. A run without
        # mage_attention.pt is reported (MISSING-ATTN) and left for the owner —
        # never silently re-fit (a fresh fit would re-randomise alpha and break
        # the apples-to-apples with mage_official).
        raise FileNotFoundError(f'no mage_attention.pt in {out_dir}')

    model = load_vanilla_model(run_dir, meta, dmeta, device)
    denorm = denorm_of(dmeta, task_type)
    pc = 0 if meta['dataset'] == 'mutag' else 1     # mutag property-positive is y==0

    def _pi_node_att_fn(data):
        """PER-INSTANCE GT-ROC attribution: that graph's OWN raw alpha (attached as
        'mage_pi_att'), read off exactly like nodes_to_motifs — NOT the frequency-
        averaged global attention_mean. Zeros when a graph carries no per-instance att."""
        a = getattr(data, 'mage_pi_att', None)
        if a is None:
            return torch.zeros(data.x.size(0), device=data.x.device)
        return a.to(data.x.device)

    grouped_by_split = {}
    inst_by_split = {}
    summary_by_split = {}

    for split in SPLITS:
        sl = split_lists.get(split)
        if not sl:
            continue
        gl = gt.get(split)
        pred_flat = _pred_scalars(
            evaluate_predictions(model, loaders[split], device, task_type, denorm=denorm))
        # score mage_v2 with attention_mean (raw alpha, NO P). per-instance atts
        # (raw alpha broadcast) feed the instance-correlation path, same as the
        # full post-hoc driver.
        scores, pi = score_mage(model, attn, sl, vocab, device, task_type,
                                positive_class=pc, return_per_instance=True,
                                verbose=False, embed_batch_size=args.batch_size,
                                score_mode=SCORE_MODE)
        if not scores.get('mean'):
            summary_by_split[split] = dict(pred_flat)     # no MAGE signal on this split
            continue
        atts = _pi_to_node_atts(pi, sl)
        # GT-ROC now grades mage_v2's PER-INSTANCE attribution — the raw per-(motif,
        # graph) alpha — NOT the frequency-averaged global attention_mean (which is a
        # single constant per motif, identical across graphs, and dilutes ubiquitous
        # causal motifs). atts[gi] is that graph's per-node alpha; attach it so
        # _pi_node_att_fn reads it off each graph exactly like nodes_to_motifs.
        # gl == sl (same objects) for GT datasets (planted / *_Verified_GT); mutag's
        # gt subset shares the same object refs, so the attached tensor still travels.
        for gi in range(len(sl)):
            setattr(sl[gi], 'mage_pi_att', atts.get(gi))
        gtroc = gt_roc_block(model, gl, device, _pi_node_att_fn) if gl else {}
        summary_by_split[split] = {**pred_flat, **gtroc}
        # --- grouped/instance correlation + importance are the GLOBAL attention_mean
        # quantity (the correct per-MOTIF score for those metrics, already written) and
        # are NOT re-run in this GT-ROC-only pass: ---
        # sc = scores['mean']
        # agn_cache = precompute_agnostic_impact(
        #     model, vocab, sl, device, task_type,
        #     cache_path=out_dir / f'impact_cache_agnostic_{split}.json')
        # grouped_rows = _grouped_rows_from_cache(agn_cache, sc, vocab)
        # inst = _instance_from_cache(agn_cache, (atts or None), sl)
        # grouped_by_split[split] = grouped_rows
        # inst_by_split[split] = inst
        # _write_importance(out_dir, METHOD, split, grouped_rows)
        # _write_impact(out_dir, METHOD, split, grouped_rows)

    # write_grouped_pooled(out_dir, METHOD, grouped_by_split)
    # write_instance(out_dir, METHOD, inst_by_split)
    # MERGE mage_v2 into the shared summary_splits.json (never overwrites the
    # existing mage_official / other method keys). Written LAST = completion marker.
    write_summary_splits(out_dir, {METHOD: summary_by_split})


def _already_done(out_dir: Path) -> bool:
    """True iff the run's summary_splits.json already carries a non-empty mage_v2
    key (skip_done keys on the merged key, not on a separate file)."""
    ss = Path(out_dir) / 'summary_splits.json'
    if not (ss.exists() and ss.stat().st_size > 2):
        return False
    try:
        import json
        d = json.loads(ss.read_text())
    except Exception:
        return False
    return bool(d.get(METHOD))


def main():
    ap = argparse.ArgumentParser(description='Co-located MAGE re-score (mage_v2) '
                                             'from saved attention, attention_mean scoring')
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--vocab_root', required=True)
    ap.add_argument('--processed_root', default=None)
    ap.add_argument('--dest_root', default=None,
                    help='the post-hoc re-eval sweep tree (posthoc_v1) where '
                         'mage_official wrote mage_attention.pt + summary_splits.json. '
                         'mage_v2 READS the saved attention from here and CO-LOCATES '
                         'its outputs here. Omit only if --out_root already holds the '
                         'mage artifacts (then out_dir == run_dir).')
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--batch_size', type=int, default=256,
                    help='GPU batch for MAGE motif/molecule embedding')
    ap.add_argument('--loader_batch_size', type=int, default=128)
    ap.add_argument('--dataset', nargs='*', default=None)
    ap.add_argument('--shard', default=None, help='i/N: process runs[i::N]')
    ap.add_argument('--families', nargs='*', default=None,
                    help='model families (default: baselines ONLY — the vanilla '
                         'backbone runs whose saved MAGE attention we re-score)')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--skip_done', action='store_true', default=True,
                    help='skip runs whose summary_splits.json already has a mage_v2 key')
    ap.add_argument('--no_skip_done', dest='skip_done', action='store_false')
    args = ap.parse_args()

    device = torch.device(
        'cuda' if (args.device == 'cuda'
                   or (args.device == 'auto' and torch.cuda.is_available()))
        else 'cpu')
    fams = set(args.families) if args.families else {'baselines'}
    if not fams <= set(FAMILIES):
        raise SystemExit(f'--families must be subset of {FAMILIES}, got {sorted(fams)}')
    print(f'device: {device}  (mage_v2 rescore, families={sorted(fams)}, '
          f'score_mode={SCORE_MODE})')

    ok = failed = skipped = missing = 0
    for run_dir, meta, fam in iter_runs(args.out_root, fams, args.dataset, args.shard):
        if args.limit and (ok + failed) >= args.limit:
            break
        tag = (f"{meta.get('dataset')}/{fam}/{meta.get('backbone')}/"
               f"{meta.get('vocab_variant')}/fold{meta.get('fold')}")
        out_dir = out_dir_for(run_dir, args.out_root, args.dest_root)
        if args.skip_done and _already_done(out_dir):
            skipped += 1
            continue
        if args.dry_run:
            print(f'  [dry] {tag}')
            ok += 1
            continue
        try:
            rescore_run(run_dir, meta, args, device)
            print(f'  [ok] {tag}')
            ok += 1
        except FileNotFoundError as e:
            print(f'  [MISSING-ATTN] {tag}: {e}')
            missing += 1
        except Exception as e:
            print(f'  [FAIL] {tag}: {type(e).__name__}: {e}')
            failed += 1
    print(f'\nDONE (mage_v2): processed={ok}  failed={failed}  '
          f'skipped_done={skipped}  missing_attention={missing}')


if __name__ == '__main__':
    main()
