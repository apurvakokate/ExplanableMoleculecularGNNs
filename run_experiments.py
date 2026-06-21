#!/usr/bin/env python3
"""run_experiments.py — unified driver for all ChemIntuit training experiments.

ONE entry point for: vanilla baselines, post-hoc/antehoc baseline explainers,
MOSE, MotifSAT, and any of them on synthetic-relabelled (GT) data. Every run is
written to a UNIQUE results folder derived from its config, so nothing is ever
overwritten.

The knobs you asked to vary easily (all sweepable as comma-lists):
  --features        onehot | linear            (1-hot vs. linear node encoder)
  --layer_norm      none | l2 | layernorm       (norm BETWEEN conv layers)
  --encoder_norm    off | on                    (LayerNorm AFTER the linear encoder;
                                                 vanilla only, via --apply_layer_norm)
  --injection       111 | 101 | ...             (w_feat / w_message / w_readout bits)
  --epochs          e.g. 100                    (sweepable: 50,100,200)
  --synthetic       off | on                    (train on phase-4 GT relabelled data)

Feature/norm coupling shortcut (your phrase "1hot+no norm / linear+norm"):
  --preset onehot_nonorm   == --features onehot --layer_norm none --encoder_norm off
  --preset linear_norm     == --features linear --layer_norm layernorm --encoder_norm on
(an explicit --features/--layer_norm/--encoder_norm overrides the preset.)

Examples
--------
  # MOSE, both injections, linear+norm vs onehot+nonorm, 100 epochs, BBBP fold 0
  python3 run_experiments.py --experiments mose \
      --datasets BBBP --folds 0 --epochs 100 \
      --injection 111,101 --preset onehot_nonorm,linear_norm \
      --out_root ./RESULTS

  # everything, synthetic + real, one command (a full grid)
  python3 run_experiments.py --experiments vanilla,baselines,mose,motifsat \
      --datasets BBBP,Mutagenicity --folds 0,1 --epochs 100 \
      --injection 111,101 --features onehot,linear --layer_norm none,l2,layernorm \
      --synthetic off,on --out_root ./RESULTS --dry_run
"""
from __future__ import annotations
import argparse, itertools, json, os, subprocess, sys, shlex
from pathlib import Path

# Repo root on sys.path so SharedModules imports work when invoked as a script.
_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from SharedModules.data.dataset_routing import (
    effective_fold,
    is_single_fold_dataset,
    mutag_artifact_paths,
    default_processed_base,
    resolve_data_root,
    resolve_node_encoder_for_dataset,
)

# ── trainer entry points (relative to PROJECT root) ─────────────────────────
TRAINERS = {
    'vanilla':   'SharedModules/baselines/run_vanilla.py',
    'baselines': 'SharedModules/baselines/run_vanilla.py',   # epochs=0 → load+explain
    'mose':      'MOSE-GNN/run.py',
    'motifsat':  'MotifSAT/run.py',
    'gsat':      'MotifSAT/run.py',                           # motif_method=none
}

from SharedModules.data.ground_truth import GT_SUPPORTED_DATASETS

PRESETS = {
    'onehot_nonorm': dict(features='onehot', layer_norm='none',      encoder_norm='off'),
    'linear_norm':   dict(features='linear', layer_norm='layernorm', encoder_norm='on'),
}

def parse_injection(bits: str):
    """'111' -> (w_feat, w_message, w_readout) booleans."""
    bits = bits.strip()
    if len(bits) != 3 or any(c not in '01' for c in bits):
        raise SystemExit(f"--injection must be a 3-bit string like 111 or 101, got {bits!r}")
    return bits[0] == '1', bits[1] == '1', bits[2] == '1'

def csv(s): return [x for x in s.split(',') if x != '']


def _env_on(name: str, default: str = '1') -> bool:
    """True unless env *name* is 0/false/no/off."""
    return os.environ.get(name, default).strip().lower() not in (
        '0', 'false', 'no', 'off',
    )


def mose_run_multi_explanation_enabled(args) -> bool:
    """Match run_experiments.sh / experiment_config.sh MOSE_RUN_MULTI_EXPLANATION."""
    if args.mose_run_multi_explanation is not None:
        return args.mose_run_multi_explanation == '1'
    return _env_on('MOSE_RUN_MULTI_EXPLANATION')


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--experiments', default='mose',
                   help='comma list of: vanilla,baselines,mose,motifsat,gsat')
    p.add_argument('--datasets', default='BBBP,Mutagenicity,mutag,ogbg-molhiv')
    p.add_argument('--folds', default='0')
    p.add_argument('--vocab_variants', default='all_fallback_bpe',
                   help='comma list of vocab variant dirs under --vocab_root')

    # the requested knobs (each a comma-list → swept)
    p.add_argument('--features', default=None, help='onehot,linear')
    p.add_argument('--layer_norm', default=None, help='none,l2,layernorm (between conv layers)')
    p.add_argument('--encoder_norm', default=None, help='off,on (LayerNorm after linear encoder; vanilla only)')
    p.add_argument('--injection', default='111', help='3-bit w_feat/w_message/w_readout, e.g. 111,101')
    p.add_argument('--epochs', default='100', help='e.g. 100 or 50,100,200')
    p.add_argument('--synthetic', default='off', help='off,on (train on phase-4 GT relabelled data)')
    p.add_argument('--preset', default=None,
                   help='comma list of presets: onehot_nonorm,linear_norm (overridden by explicit knobs)')

    # fixed-ish model/training
    p.add_argument('--backbone', default='GIN')
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--num_layers', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)

    # paths
    p.add_argument('--project', default='.', help='project root (contains trainers)')
    p.add_argument('--data_root', default='./datasets/FOLDS')
    p.add_argument('--mutag_data_root', default=None,
                   help='data root for mutag TUDataset exports (mutag_<fold>.csv, …)')
    p.add_argument('--ogb_data_root', default=None,
                   help='OGB cache root (defaults to --data_root)')
    p.add_argument('--processed_root', default=os.environ.get('PROCESSED_ROOT'),
                   help='PyG processed .pt cache root (per-vocab subdir appended)')
    p.add_argument('--vocab_root', default='./vocab_output')
    p.add_argument('--gt_cache', default='./RESULTS/gt_cache',
                   help='phase-4 GT cache root (used when --synthetic on)')
    p.add_argument('--out_root', default='./RESULTS')
    # --weights_root was unused (baselines resolve weights via vanilla_weights_dir).
    # p.add_argument('--weights_root', default=None,
    #                help='Root of trained vanilla checkpoints for baselines/'
    #                     'explainers (epochs=0). Defaults to <out_root>/vanilla.')
    p.add_argument('--share_filter_weights', action='store_true',
                   help='Let a *_filter baseline load the UNFILTERED vanilla '
                        'checkpoint (vanilla weights are motif-agnostic, so one '
                        'training is shared across the filter/no-filter pair).')

    # MotifSAT-specific
    p.add_argument('--motif_method', default='readout', help='readout|none (none ⇒ base GSAT)')

    # MOSE-specific (default on — same as MOSE_RUN_MULTI_EXPLANATION=1 in experiment_config.sh)
    p.add_argument('--mose_run_multi_explanation', choices=('0', '1'), default=None,
                   help='MOSE H0/H1/H2 analysis via --run_multi_explanation '
                        '(default: MOSE_RUN_MULTI_EXPLANATION env or 1)')

    # extra passthrough + control
    p.add_argument('--extra', default='', help='extra flags appended verbatim to every trainer call')
    p.add_argument('--dry_run', action='store_true', help='print commands, do not execute')
    p.add_argument('--skip_existing', action='store_true',
                   help='skip any run whose out_dir already has a summary.json '
                        '(resume a partially-completed sweep without redoing work)')
    p.add_argument('--continue_on_error', action='store_true')
    return p

def resolve_norm_feature(preset, feat_override, ln_override, en_override):
    """Combine a preset (if any) with explicit overrides."""
    base = dict(features='onehot', layer_norm='l2', encoder_norm='off')
    if preset:
        if preset not in PRESETS:
            raise SystemExit(
                f"Unknown preset {preset!r}; choose from {sorted(PRESETS)}"
            )
        base.update(PRESETS[preset])
    if feat_override: base['features'] = feat_override
    if ln_override:   base['layer_norm'] = ln_override
    if en_override:   base['encoder_norm'] = en_override
    return base

# Families whose checkpoint/identity does NOT depend on injection or epochs:
# the vanilla GNN (its weights are motif-agnostic) and the post-hoc baseline
# explainers that load those weights. Their config slug omits inj/ep so a single
# vanilla checkpoint is reused across injection and epoch sweeps.
INJECTION_AGNOSTIC = ('vanilla', 'baselines')


def _cfg_slug(feat, ln, en, inj, epochs, syn, include_inj_ep, backbone):
    """Leaf config folder name (the part below <exp>/<ds>/fold/<variant>)."""
    nrm = f"norm-{ln}" + ("+encLN" if en == 'on' else "")
    parts = [f"bb-{backbone}", f"enc-{feat}", nrm]
    if include_inj_ep:
        parts += [f"inj{inj}", f"ep{epochs}"]
    parts.append('gt' if syn == 'on' else 'real')
    return '_'.join(parts)


def config_tag(exp, ds, fold, variant, feat, ln, en, inj, epochs, syn, backbone):
    """Canonical results path (relative to --out_root) for one config.

    Layout (a SINGLE dataset/fold level — the trainers are invoked with
    --final_out_dir so they do not re-append <ds>/fold/<tag>):

        <exp>/<ds>/fold<fold>/<variant>/<cfg_slug>

    vanilla/baselines omit the inj/ep tokens from <cfg_slug> so their (motif-
    agnostic) weights are shared across injection and epoch sweeps.
    """
    include_inj_ep = exp not in INJECTION_AGNOSTIC
    slug = _cfg_slug(feat, ln, en, inj, epochs, syn, include_inj_ep, backbone)
    return f"{exp}/{ds}/fold{fold}/{variant}/{slug}"


def vanilla_weights_dir(args, ds, fold, weight_variant, feat, ln, en, syn):
    """FINAL dir of the trained vanilla run a baseline should load weights from."""
    slug = _cfg_slug(feat, ln, en, inj=None, epochs=None, syn=syn,
                     include_inj_ep=False, backbone=args.backbone)
    return (Path(args.out_root) /
            f"vanilla/{ds}/fold{fold}/{weight_variant}/{slug}")


def canonical_config(exp, args, ds, fold, variant, cfg, inj, epochs, syn,
                     weight_variant=None):
    """Explicit, machine-readable axis record written as config.json into the
    run dir. analysis/run_analysis.py collect merges these so all_results.csv
    carries real axis columns instead of path tokens."""
    threshold = 'on' if variant.endswith('_filter') else 'off'
    fragmentation = variant[:-len('_filter')] if variant.endswith('_filter') else variant
    # injection only applies to ante-hoc motif models; baselines do not train.
    inj_val = 'na' if exp in INJECTION_AGNOSTIC else inj
    ep_val = 0 if exp == 'baselines' else int(epochs)
    rec = {
        'schema':        'chemintuit/v1',
        'family':        exp,
        'dataset':       ds,
        'fold':          int(fold),
        'backbone':      args.backbone,
        'vocab_variant': variant,
        'fragmentation': fragmentation,
        'threshold':     threshold,
        'features':      cfg['features'],
        'norm':          cfg['layer_norm'],
        'encoder_norm':  cfg['encoder_norm'],
        'injection':     inj_val,
        'epochs':        ep_val,
        'synthetic':     'gt' if syn == 'on' else 'real',
        'seed':          args.seed,
    }
    if exp == 'baselines':
        rec['weight_vocab_variant'] = weight_variant or variant
        rec['weights_dir'] = str(vanilla_weights_dir(
            args, ds, fold, weight_variant or variant,
            cfg['features'], cfg['layer_norm'], cfg['encoder_norm'], syn))
    if exp == 'mose':
        rec['run_multi_explanation'] = mose_run_multi_explanation_enabled(args)
    return rec

def _trainer_paths(args, ds: str):
    """Resolve data_root and base processed_root (trainers append vocab variant)."""
    dr = resolve_data_root(
        ds, args.data_root,
        mutag_data_root=args.mutag_data_root,
        ogb_data_root=args.ogb_data_root,
    )
    return dr, default_processed_base(dr, args.processed_root)


def _mutag_cli(ds: str, data_root: str, fold: int) -> list:
    if ds != 'mutag':
        return []
    paths = mutag_artifact_paths(data_root, fold)
    return [
        '--mutag_index_maps_path', paths['mutag_index_maps_path'],
        '--mutag_smiles_csv_path', paths['mutag_smiles_csv_path'],
        '--mutag_splits_path', paths['mutag_splits_path'],
        '--mutag_seed', '42',
    ]


def make_command(exp, args, ds, fold, variant, cfg, inj, epochs, syn):
    feat = cfg['features']; ln = cfg['layer_norm']; en = cfg['encoder_norm']
    w_feat, w_msg, w_read = parse_injection(inj)
    eff_fold = effective_fold(ds, int(fold))
    ds_root, proc_root = _trainer_paths(args, ds)
    node_enc = resolve_node_encoder_for_dataset(ds, feat)
    out_dir = Path(args.out_root) / config_tag(
        exp, ds, fold, variant, feat, ln, en, inj, epochs, syn, args.backbone)
    script = Path(args.project) / TRAINERS[exp]
    cmd = [sys.executable, str(script),
           '--dataset', ds, '--fold', str(eff_fold),
           '--backbone', args.backbone, '--node_encoder', node_enc,
           '--hidden_dim', str(args.hidden_dim), '--num_layers', str(args.num_layers),
           '--conv_normalize', ln,
           '--data_root', ds_root, '--vocab_root', args.vocab_root,
           '--vocab_variant', variant, '--out_dir', str(out_dir),
           '--processed_root', proc_root,
           '--final_out_dir', '--seed', str(args.seed)]
    cmd += _mutag_cli(ds, ds_root, eff_fold)

    if exp == 'vanilla':
        cmd += ['--epochs', str(epochs), '--lr', str(args.lr)]
        if en == 'on': cmd += ['--apply_layer_norm']
        if syn == 'on':
            cmd += ['--use_gt', '--gt_cache', args.gt_cache]
    elif exp == 'baselines':
        # Post-hoc explainers: load the trained vanilla weights, no training.
        # Resolve the EXACT vanilla run dir (final layout) for this config so the
        # checkpoint is found deterministically. Vanilla weights are motif-
        # agnostic, so with --share_filter_weights a *_filter baseline reuses the
        # unfiltered vanilla checkpoint (one training shared across thresholds).
        weight_variant = variant
        if args.share_filter_weights and variant.endswith('_filter'):
            weight_variant = variant[:-len('_filter')]
        vdir = vanilla_weights_dir(args, ds, fold, weight_variant, feat, ln, en, syn)
        cmd += ['--epochs', '0',
                '--load_weights_from', str(vdir),
                '--weight_vocab_variant', weight_variant]
        if en == 'on': cmd += ['--apply_layer_norm']
        if syn == 'on':
            # GT-backed test set so the post-hoc explainers get GT-ROC; loads the
            # GT-trained vanilla checkpoint (weights dir already encodes syn).
            cmd += ['--use_gt', '--gt_cache', args.gt_cache]
    elif exp in ('mose',):
        cmd += ['--epochs', str(epochs)]
        if w_feat: cmd += ['--w_feat']
        if w_msg:  cmd += ['--w_message']
        if w_read: cmd += ['--w_readout']
        if mose_run_multi_explanation_enabled(args):
            cmd += ['--run_multi_explanation']
        if syn == 'on':
            cmd += ['--use_gt', '--gt_cache', args.gt_cache]
    elif exp == 'gsat':
        # GSAT baseline: edge-attention + node IB, no MOSE injection.
        cmd += ['--epochs', str(epochs), '--lr', str(args.lr),
                '--motif_method', 'none',
                '--learn_edge_att', '--noise', 'node',
                '--info_loss_level', 'node', '--info_loss_coef', '1.0']
        if syn == 'on':
            cmd += ['--use_gt', '--gt_cache', args.gt_cache]
    elif exp == 'motifsat':
        cmd += ['--epochs', str(epochs), '--lr', str(args.lr),
                '--motif_method', args.motif_method,
                '--noise', 'none', '--info_loss_level', 'none', '--info_loss_coef', '0.0']
        if w_feat: cmd += ['--w_feat']
        if w_msg:  cmd += ['--w_message']
        if w_read: cmd += ['--w_readout']
        if syn == 'on':
            cmd += ['--use_gt', '--gt_cache', args.gt_cache]

    if args.extra:
        cmd += shlex.split(args.extra)
    return cmd, out_dir

def main():
    args = build_arg_parser().parse_args()
    if not args.processed_root:
        args.processed_root = default_processed_base(
            str(Path(args.project).resolve()), None)
    exps      = csv(args.experiments)
    datasets  = csv(args.datasets)
    folds     = csv(args.folds)
    variants  = csv(args.vocab_variants)
    injections= csv(args.injection)
    epochs_l  = csv(args.epochs)
    synth_l   = csv(args.synthetic)
    presets   = csv(args.preset) if args.preset else [None]
    feats     = csv(args.features) if args.features else [None]
    lns       = csv(args.layer_norm) if args.layer_norm else [None]
    ens       = csv(args.encoder_norm) if args.encoder_norm else [None]

    # Build the (feature,norm) config list: either from presets, or from the
    # explicit feature/norm sweeps (cartesian), or the default single config.
    cfgs = []
    if args.preset:
        for pre in presets:
            # explicit knobs (if given) override within each preset
            for f, l, e in itertools.product(feats, lns, ens):
                cfgs.append((pre, f, l, e))
    else:
        for f, l, e in itertools.product(feats, lns, ens):
            cfgs.append((None, f, l, e))
    # de-dup resolved configs
    seen = set(); resolved_cfgs = []
    for pre, f, l, e in cfgs:
        c = resolve_norm_feature(pre, f, l, e)
        key = (c['features'], c['layer_norm'], c['encoder_norm'])
        if key in seen: continue
        seen.add(key); resolved_cfgs.append(c)

    runs = list(itertools.product(exps, datasets, folds, variants,
                                  resolved_cfgs, injections, epochs_l, synth_l))
    planned = []           # de-duped (vanilla/baselines collapse inj/ep sweeps)
    seen_dirs = set()
    for exp, ds, fold, variant, cfg, inj, epochs, syn in runs:
        # OGB/mutag only have fold-0 artifacts; skip redundant fold sweeps.
        if is_single_fold_dataset(ds) and int(fold) != 0:
            continue
        # Non-GT datasets (mutag source GT, regression, OGB) never relabel:
        # force the synthetic axis off so out_dir/config/cmd stay consistent and
        # we don't request a GT cache that phase-4 intentionally skips.
        if syn == 'on' and ds not in GT_SUPPORTED_DATASETS:
            syn = 'off'
        cmd, out_dir = make_command(exp, args, ds, fold, variant, cfg, inj, epochs, syn)
        if str(out_dir) in seen_dirs:
            # vanilla/baseline slug drops inj/ep, so an inj/epoch sweep maps many
            # run tuples to one dir — run it once, not N times.
            continue
        seen_dirs.add(str(out_dir))
        wv = (variant[:-len('_filter')]
              if (exp == 'baselines' and args.share_filter_weights
                  and variant.endswith('_filter')) else variant)
        config = canonical_config(exp, args, ds, fold, variant, cfg, inj,
                                  epochs, syn, weight_variant=wv)
        planned.append((exp, ds, fold, variant, cfg, inj, epochs, syn,
                        cmd, out_dir, config))

    print(f"# {len(planned)} run(s) planned"
          + (f" ({len(runs)} before de-dup)" if len(planned) != len(runs) else "")
          + "\n")
    skipped_existing = dry_run_only = attempted = failed = 0
    for (exp, ds, fold, variant, cfg, inj, epochs, syn,
         cmd, out_dir, config) in planned:
        if args.skip_existing and (out_dir / 'summary.json').exists():
            print(f"## [skip existing] {out_dir}\n")
            skipped_existing += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        printable = ' '.join(shlex.quote(c) for c in cmd)
        print(f"## {out_dir}")
        print(printable + "\n")
        # persist the exact command + canonical axis record for provenance
        (out_dir / 'run_command.sh').write_text("#!/bin/bash\n" + printable + "\n")
        (out_dir / 'config.json').write_text(json.dumps(config, indent=2) + "\n")
        if args.dry_run:
            dry_run_only += 1
            continue
        attempted += 1
        env = dict(os.environ)
        env.setdefault('PROCESSED_ROOT', args.processed_root)
        try:
            r = subprocess.run(cmd, env=env,
                               cwd=str(Path(args.project).resolve()))
            if r.returncode != 0:
                failed += 1
                print(f"!! exit {r.returncode} for {out_dir}", file=sys.stderr)
                if not args.continue_on_error:
                    sys.exit(r.returncode)
        except Exception as e:
            failed += 1
            print(f"!! {e} for {out_dir}", file=sys.stderr)
            if not args.continue_on_error:
                raise
    succeeded = attempted - failed
    print(f"\n# done. planned={len(planned)} succeeded={succeeded} failed={failed} "
          f"skipped_existing={skipped_existing} dry_run={dry_run_only}")

if __name__ == '__main__':
    main()
