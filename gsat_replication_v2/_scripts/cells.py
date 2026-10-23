#!/usr/bin/env python
"""cells.py — GSAT paper replication with the CURRENT MotifSAT package (base_gsat).

Each cell = (backbone, dataset, split) with the official GSAT config for that
backbone/dataset (Graph-COM/GSAT src/configs/*.yml); every run uses the GSAT-parity
knobs (arch_style=gsat, ckpt_rule=gsat, no LR scheduler, no grad clipping, plain BCE,
weight_decay 0, beta=1, Concrete temp 1, no early stopping) and no motif vocabulary.

Paper targets are x100 as printed (GSAT, not GSAT*); metric key = summary.json field.
  T1 explanation ROC-AUC pooled over test edges  -> gsat_edge_auc_test
  T2/T3 test ROC-AUC (OGB evaluator convention)   -> test_auc / test_auc_mean

Rows 0-109: arch_style=gsat (the replication). Rows 110-219: the same cells with
arch_style=ours (backbone-effect control; later --arch_style on the CLI wins).
Rows 220-259: our PNA on ogbg-molhiv + one GSAT piece (noscalers / bnres / meanpool) or all three.

CLI:  cells.py --n | --list | --args <idx> [--check]
"""
import sys

SEEDS = list(range(10))
COMMON = ['--config', 'MotifSAT/configs/base_gsat.yaml', '--arch_style', 'gsat',
          '--pna_scalers', 'false', '--info_loss_coef', '1', '--init_r', '0.9',
          '--decay_r', '0.1', '--ckpt_rule', 'gsat', '--lr_scheduler', 'none',
          '--pos_weight', 'none', '--clip_grad', '0', '--weight_decay', '0',
          '--dropout', '0.3', '--lr', '1e-3', '--patience', '100000',
          '--vocab_variant', 'none']
GIN = ['--backbone', 'GIN', '--hidden_dim', '64', '--num_layers', '2']
PNA = ['--backbone', 'PNA', '--hidden_dim', '80', '--num_layers', '4']

# name -> (args, paper targets {table: (metric, mean, std)}, folds)
CELLS = {
    'GIN_mutag_gsatsplit': (GIN + ['--dataset', 'mutag', '--mutag_split', 'gsat',
                                   '--epochs', '100', '--decay_interval', '10',
                                   '--final_r', '0.5', '--ckpt_metric', 'acc'],
                            {'T1': ('gsat_edge_auc_test', 99.60, 0.51)}, [0]),
    'PNA_mutag_gsatsplit': (PNA + ['--dataset', 'mutag', '--mutag_split', 'gsat',
                                   '--epochs', '50', '--decay_interval', '5',
                                   '--final_r', '0.5', '--ckpt_metric', 'acc',
                                   '--pna_aggregators', 'mean,min,max,std,sum'],
                            {'T1': ('gsat_edge_auc_test', 99.07, 0.50)}, [0]),
    # same configs on our DISJOINT mutag folds (5 folds x 2 seeds) — not in the paper
    'GIN_mutag_folds': (GIN + ['--dataset', 'mutag', '--mutag_split', 'folds',
                               '--epochs', '100', '--decay_interval', '10',
                               '--final_r', '0.5', '--ckpt_metric', 'acc'], {}, [0, 1, 2, 3, 4]),
    'PNA_mutag_folds': (PNA + ['--dataset', 'mutag', '--mutag_split', 'folds',
                               '--epochs', '50', '--decay_interval', '5',
                               '--final_r', '0.5', '--ckpt_metric', 'acc',
                               '--pna_aggregators', 'mean,min,max,std,sum'], {}, [0, 1, 2, 3, 4]),
    'GIN_ogbg-molhiv': (GIN + ['--dataset', 'ogbg-molhiv', '--epochs', '200',
                               '--decay_interval', '20', '--final_r', '0.7'],
                        {'T2': ('test_auc', 76.47, 1.53)}, [0]),
}
_PNA_OGB = ['--epochs', '100', '--decay_interval', '10', '--final_r', '0.7',
            '--pna_aggregators', 'mean,min,max,std']
for ds, tgt in [('ogbg-molhiv', {'T2': ('test_auc', 80.24, 0.73)}),
                ('ogbg-molbace', {'T3': ('test_auc', 77.41, 2.42)}),
                ('ogbg-molbbbp', {'T3': ('test_auc', 69.17, 1.12)}),
                ('ogbg-molclintox', {'T3': ('test_auc_mean', 87.80, 2.36)}),
                ('ogbg-moltox21', {'T3': ('test_auc_mean', 74.96, 0.66)}),
                ('ogbg-molsider', {'T3': ('test_auc_mean', 57.58, 1.23)})]:
    CELLS[f'PNA_{ds}'] = (PNA + ['--dataset', ds] + _PNA_OGB, tgt, [0])


# Backbone-effect control: every cell again with OUR backbone (arch_style=ours: BaseGNN
# GIN with learned eps / PNA without BN+residual, our heads and readout), all other
# settings identical. The gap to the gsat-arch cell = effect of the backbone alone.
# Appended AFTER the gsat-arch rows so their array indices (0..109) never change.
OURS_ARCH = {f'{name}__oursarch': (args + ['--arch_style', 'ours'], tgt, folds)
             for name, (args, tgt, folds) in CELLS.items()}
CELLS.update(OURS_ARCH)

# PNA-piece test (rows 220-259): OUR PNA on ogbg-molhiv with ONE GSAT piece added at a time,
# then all three. Baseline = PNA_ogbg-molhiv__oursarch; target = PNA_ogbg-molhiv (GSAT backbone).
_base_args, _base_tgt, _base_folds = CELLS['PNA_ogbg-molhiv__oursarch']
for _tag, _extra in [('noscalers', ['--pna_pieces', 'noscalers']),
                     ('bnres', ['--pna_pieces', 'bnres']),
                     ('meanpool', ['--graph_pool', 'mean']),
                     ('allthree', ['--pna_pieces', 'noscalers,bnres', '--graph_pool', 'mean'])]:
    CELLS[f'PNA_ogbg-molhiv__oursarch__{_tag}'] = (_base_args + _extra, _base_tgt, _base_folds)


def runs():
    """(cell, fold, seed): 10 seeds for single-fold cells, 5 folds x 2 seeds otherwise.
    gsat-arch cells first, then their __oursarch copies (dict insertion order)."""
    out = []
    for name, (_, _, folds) in CELLS.items():
        seeds = SEEDS if len(folds) == 1 else [0, 1]
        out += [(name, f, s) for f in folds for s in seeds]
    return out


RUNS = runs()


def args_for(idx, check=False):
    name, fold, seed = RUNS[idx]
    a = COMMON + CELLS[name][0] + ['--fold', str(fold), '--seed', str(seed)]
    if check:                                   # 2-epoch smoke of every code path
        i = a.index('--epochs'); a[i + 1] = '2'
    return name, fold, seed, a


if __name__ == '__main__':
    if '--n' in sys.argv:
        print(len(RUNS))
    elif '--list' in sys.argv:
        for i, r in enumerate(RUNS):
            print(i, *r)
    elif '--args' in sys.argv:
        name, fold, seed, a = args_for(int(sys.argv[sys.argv.index('--args') + 1]),
                                       check='--check' in sys.argv)
        print(name, fold, seed, ' '.join(a))
    else:
        print(__doc__)
