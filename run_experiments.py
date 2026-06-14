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
import argparse, itertools, os, subprocess, sys, shlex
from pathlib import Path

# ── trainer entry points (relative to PROJECT root) ─────────────────────────
TRAINERS = {
    'vanilla':   'SharedModules/baselines/run_vanilla.py',
    'baselines': 'SharedModules/baselines/run_vanilla.py',   # epochs=0 → load+explain
    'mose':      'MOSE-GNN/run.py',
    'motifsat':  'MotifSAT/run.py',
    'gsat':      'MotifSAT/run.py',                           # motif_method=none
}

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

def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--experiments', default='mose',
                   help='comma list of: vanilla,baselines,mose,motifsat,gsat')
    p.add_argument('--datasets', default='BBBP')
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
    p.add_argument('--vocab_root', default='./vocab_output')
    p.add_argument('--gt_cache', default='./RESULTS/gt_cache',
                   help='phase-4 GT cache root (used when --synthetic on)')
    p.add_argument('--out_root', default='./RESULTS')
    p.add_argument('--weights_root', default=None,
                   help='Root of trained vanilla checkpoints for baselines/'
                        'explainers (epochs=0). Defaults to <out_root>/vanilla.')

    # MotifSAT-specific
    p.add_argument('--motif_method', default='readout', help='readout|none (none ⇒ base GSAT)')

    # extra passthrough + control
    p.add_argument('--extra', default='', help='extra flags appended verbatim to every trainer call')
    p.add_argument('--dry_run', action='store_true', help='print commands, do not execute')
    p.add_argument('--continue_on_error', action='store_true')
    return p

def resolve_norm_feature(args, preset, feat_override, ln_override, en_override):
    """Combine a preset (if any) with explicit overrides."""
    base = dict(features='onehot', layer_norm='l2', encoder_norm='off')
    if preset and preset in PRESETS:
        base.update(PRESETS[preset])
    if feat_override: base['features'] = feat_override
    if ln_override:   base['layer_norm'] = ln_override
    if en_override:   base['encoder_norm'] = en_override
    return base

def config_tag(exp, ds, fold, variant, feat, ln, en, inj, epochs, syn):
    """Unique, human-readable folder name for this config."""
    enc = f"enc-{feat}"
    nrm = f"norm-{ln}" + ("+encLN" if en == 'on' else "")
    return (f"{exp}/{ds}/fold{fold}/{variant}/"
            f"{enc}_{nrm}_inj{inj}_ep{epochs}_{'gt' if syn=='on' else 'real'}")

def make_command(exp, args, ds, fold, variant, cfg, inj, epochs, syn):
    feat = cfg['features']; ln = cfg['layer_norm']; en = cfg['encoder_norm']
    w_feat, w_msg, w_read = parse_injection(inj)
    out_dir = Path(args.out_root) / config_tag(exp, ds, fold, variant, feat, ln, en, inj, epochs, syn)
    script = Path(args.project) / TRAINERS[exp]
    cmd = [sys.executable, str(script),
           '--dataset', ds, '--fold', str(fold),
           '--backbone', args.backbone, '--node_encoder', feat,
           '--hidden_dim', str(args.hidden_dim), '--num_layers', str(args.num_layers),
           '--conv_normalize', ln,
           '--data_root', args.data_root, '--vocab_root', args.vocab_root,
           '--vocab_variant', variant, '--out_dir', str(out_dir),
           '--seed', str(args.seed)]

    if exp == 'vanilla':
        cmd += ['--epochs', str(epochs), '--lr', str(args.lr)]
        if en == 'on': cmd += ['--apply_layer_norm']
    elif exp == 'baselines':
        # post-hoc / antehoc explainers: load trained vanilla weights, no training.
        # Point the trainer at the weights root so it can resolve the checkpoint;
        # with the fail-fast fix, a missing checkpoint now errors instead of
        # silently explaining random weights.
        cmd += ['--epochs', '0',
                '--load_weights_from', args.weights_root]
        if en == 'on': cmd += ['--apply_layer_norm']
    elif exp in ('mose',):
        cmd += ['--epochs', str(epochs)]
        if w_feat: cmd += ['--w_feat']
        if w_msg:  cmd += ['--w_message']
        if w_read: cmd += ['--w_readout']
        if syn == 'on':
            cmd += ['--use_gt', '--gt_cache', args.gt_cache]
    elif exp in ('motifsat', 'gsat'):
        method = 'none' if exp == 'gsat' else args.motif_method
        cmd += ['--epochs', str(epochs), '--lr', str(args.lr),
                '--motif_method', method,
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
    if args.weights_root is None:
        args.weights_root = str(Path(args.out_root) / 'vanilla')
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
        c = resolve_norm_feature(args, pre, f, l, e)
        key = (c['features'], c['layer_norm'], c['encoder_norm'])
        if key in seen: continue
        seen.add(key); resolved_cfgs.append(c)

    runs = list(itertools.product(exps, datasets, folds, variants,
                                  resolved_cfgs, injections, epochs_l, synth_l))
    print(f"# {len(runs)} run(s) planned\n")
    failures = 0
    for exp, ds, fold, variant, cfg, inj, epochs, syn in runs:
        cmd, out_dir = make_command(exp, args, ds, fold, variant, cfg, inj, epochs, syn)
        out_dir.mkdir(parents=True, exist_ok=True)
        printable = ' '.join(shlex.quote(c) for c in cmd)
        print(f"## {out_dir}")
        print(printable + "\n")
        # persist the exact command for provenance
        (out_dir / 'run_command.sh').write_text("#!/bin/bash\n" + printable + "\n")
        if args.dry_run:
            continue
        env = dict(os.environ)
        try:
            r = subprocess.run(cmd, env=env)
            if r.returncode != 0:
                failures += 1
                print(f"!! exit {r.returncode} for {out_dir}", file=sys.stderr)
                if not args.continue_on_error:
                    sys.exit(r.returncode)
        except Exception as e:
            failures += 1
            print(f"!! {e} for {out_dir}", file=sys.stderr)
            if not args.continue_on_error:
                raise
    print(f"\n# done. {len(runs)-failures}/{len(runs)} succeeded"
          + (f", {failures} failed" if failures else ""))

if __name__ == '__main__':
    main()
