#!/usr/bin/env python3
"""gnn1_grid.py — single source of truth for the GNN1-design ablation.

Mechanism ③ (motif_method='motif_emb') GNN1 is an HIMP-style GIN/GINE fragment-graph
encoder. This ablation is a FULL factorial over four GNN1-design factors (depth is
FIXED at 2 — user decision to cut experiments):

    feat      {multihot, id_desc}         fragment-node features F_m
    edge      {none, mult, chem}          junction edge feature: none=GIN (no edge),
                                          mult=GINE + scalar multiplicity D,
                                          chem=GINE + 17-dim D-chemistry (multiplicity
                                          + bond-order + source/target attach element).
                                          The ONLY conv variation — GNN1 conv is GIN/GINE.
    norm      {none, layer}               per-layer norm (LayerNorm only; M=1-safe)
    residual  {off, on}                   per-layer skip connection

=> 2 x 3 x 2 x 2 = 24 configs. Grid = 24 configs x 8 datasets x 5 folds x 5 backbones = 4800.

All 32 share the base preset MotifSAT/configs/motif_emb_base.yaml; each cell differs
only in the CLI flags emitted by ``config_flags`` below. motif_emb runs are written
by run.py with method stem 'gsat' (see run.py `_method`), so completeness is checked
with native_complete.is_complete(dir, 'gsat').

Distinct, analyzable folders: each cell lives at
    <out_root>/gnn1_ablation_runs/<config_id>/<dataset>/fold<f>/<backbone>/
where config_id encodes all five knobs (see ``config_id``). This is what makes the
factorial recoverable post-hoc; it also means --final_out_dir is REQUIRED (the path,
not variant_tag, carries the knobs).

Stdlib only. Mirrors the shape of native_complete.py so gnn1_status.py can reuse it:
``iter_cells`` yields 7-tuples (cid, config_id, stem, dataset, fold, backbone, rel).
"""
import os
import sys
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import native_complete as nc  # noqa: E402  (DATASETS/FOLDS/BACKBONES/SPLITS + is_complete)

BASE_PRESET = 'MotifSAT/configs/motif_emb_base.yaml'
STEM = 'gsat'                       # motif_emb -> run.py _method == 'gsat'

# ── Factors (ordered). Depth is FIXED at 2 (user decision — cut experiments), so the
# grid is feat{2} x edge{3} x norm{2} x residual{2} = 24 configs. The edge factor is
# 3-level: 'none' (GIN, no edge feature), 'mult' (GINE + scalar junction multiplicity
# D), 'chem' (GINE + 17-dim D-chemistry: multiplicity + bond-order + source/target
# attachment element). 'chem' inherently implies the edge feature is on, so all 24
# cells are valid.
FEAT = ['multihot', 'id_desc']
EDGE = ['none', 'mult', 'chem']
NORM = ['none', 'layer']
RESID = [False, True]
LAYERS_FIXED = 2                   # depth fixed at 2 (not a factor)

# edge_mode -> config_id token
_EDGE_TAG = {'none': 'gin', 'mult': 'gine', 'chem': 'ginechem'}

DATASETS = nc.DATASETS
FOLDS = nc.FOLDS
BACKBONES = nc.BACKBONES
SPLITS = nc.SPLITS


def all_configs():
    """The 24 factorial configs, each a dict of the four factors (+ fixed depth)."""
    out = []
    for feat, edge, norm, resid in itertools.product(FEAT, EDGE, NORM, RESID):
        out.append(dict(feat=feat, edge=edge, norm=norm,
                        residual=resid, layers=LAYERS_FIXED))
    return out


CONFIGS = all_configs()


def config_id(c):
    """Path-safe id encoding all factors (so the factorial is recoverable)."""
    return (f"mf-{c['feat']}"
            f"__{_EDGE_TAG[c['edge']]}"
            f"__n-{c['norm']}"
            f"__r-{'on' if c['residual'] else 'off'}"
            f"__L{c['layers']}")


def config_flags(c):
    """The EXACT CLI flags that realize config `c` over the base preset.

    Contract with run.py argparse:
      * --motif_feat is always passed (explicit).
      * edge 'none'  -> --no_motif_edge_feat (store_false; turns the edge feat off).
        edge 'mult'  -> nothing (edge feat on by default -> scalar multiplicity D).
        edge 'chem'  -> --motif_edge_chem (edge feat stays on; 17-dim D-chemistry).
        Presence/absence is what run.py's _cli_provided_dests keys on.
      * --motif_gnn_norm is always passed (explicit).
      * --motif_gnn_residual (store_true) is passed ONLY for residual-on.
      * --motif_gnn_layers is always passed (explicit; fixed at 2).
    """
    f = ['--motif_feat', c['feat']]
    if c['edge'] == 'none':
        f += ['--no_motif_edge_feat']
    elif c['edge'] == 'chem':
        f += ['--motif_edge_chem']
    f += ['--motif_gnn_norm', c['norm']]
    if c['residual']:
        f += ['--motif_gnn_residual']
    f += ['--motif_gnn_layers', str(c['layers'])]
    return f


def iter_cells():
    """Yield (cid, config_id, stem, dataset, fold, backbone, rel) for the full
    expected grid (32 configs x 8 ds x 5 folds x 5 bb = 6400). The `config_id` sits
    in the slot native_complete/base_status call `preset`, so gnn1_status reuses it
    and its --preset slice filter becomes a per-config filter."""
    for c in CONFIGS:
        cfgid = config_id(c)
        for ds in DATASETS:
            for fold in FOLDS:
                for bb in BACKBONES:
                    cid = f'{cfgid}__{ds}__f{fold}__{bb}'
                    rel = f'{cfgid}/{ds}/fold{fold}/{bb}'
                    yield cid, cfgid, STEM, ds, fold, bb, rel


# ── native_complete-compatible surface (so gnn1_status can import gnn1_grid as nc)
def is_complete(cell_dir, stem=STEM):
    return nc.is_complete(cell_dir, stem)


def required_files(stem=STEM):
    return nc.required_files(stem)


def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cell_dir', nargs='?', help='a run dir to check (stem=gsat)')
    ap.add_argument('--list-configs', action='store_true',
                    help='print "<config_id>\\t<space-joined flags>" per line '
                         '(consumed by gnn1_worker.sh)')
    ap.add_argument('--list-cells', action='store_true',
                    help='print the full 6400-cell grid as TSV')
    ap.add_argument('--quiet', action='store_true', help='no stdout, exit code only')
    a = ap.parse_args(argv)

    if a.list_configs:
        for c in CONFIGS:
            print(f'{config_id(c)}\t{" ".join(config_flags(c))}')
        return 0
    if a.list_cells:
        for cid, cfgid, stem, ds, fold, bb, rel in iter_cells():
            print(f'{cid}\t{cfgid}\t{stem}\t{ds}\t{fold}\t{bb}\t{rel}')
        return 0
    if not a.cell_dir:
        print('usage: gnn1_grid.py <cell_dir> | --list-configs | --list-cells',
              file=sys.stderr)
        return 2
    ok, reason = is_complete(a.cell_dir)
    if not a.quiet:
        print('VALID' if ok else f'INCOMPLETE: {reason}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(_main(sys.argv[1:]))
