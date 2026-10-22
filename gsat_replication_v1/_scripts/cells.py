#!/usr/bin/env python
"""cells.py — single source of truth for the faithful GSAT replication (E1).

Grid = the GSAT (from-scratch, not GSAT*) rows of Miao et al. ICML'22 Tables 1-3,
run with the UNMODIFIED upstream code (Graph-COM/GSAT @ UPSTREAM_COMMIT) and its
own per-(backbone, dataset) yml configs. 10 seeds (= upstream global_config).

Paper targets are x100 as printed; `metric` is the upstream metric_dict key:
  best_x_roc_test : explanation ROC-AUC pooled over all test edges (Table 1)
  best_clf_test   : test acc, or test ROC-AUC for ogbg_* (Tables 2, 3)
Both are read at the epoch with best validation prediction (upstream rule).

CLI:  cells.py --list      -> "<idx> <backbone> <dataset> <seed>" per run
      cells.py --n         -> number of runs
"""
import sys

UPSTREAM_REPO = 'https://github.com/Graph-COM/GSAT.git'
UPSTREAM_COMMIT = '6812f5c'
SEEDS = list(range(10))

# (backbone, dataset) -> {table: (metric_key, paper_mean, paper_std)}
TARGETS = {
    ('GIN', 'ba_2motifs'):      {'T1': ('best_x_roc_test', 98.74, 0.55)},
    ('GIN', 'mutag'):           {'T1': ('best_x_roc_test', 99.60, 0.51)},
    ('GIN', 'mnist'):           {'T1': ('best_x_roc_test', 83.36, 1.02),
                                 'T2': ('best_clf_test',   96.24, 0.17)},
    ('GIN', 'spmotif_0.5'):     {'T1': ('best_x_roc_test', 78.45, 3.12),
                                 'T2': ('best_clf_test',   52.74, 4.08)},
    ('GIN', 'spmotif_0.7'):     {'T1': ('best_x_roc_test', 74.07, 5.28),
                                 'T2': ('best_clf_test',   49.12, 3.29)},
    ('GIN', 'spmotif_0.9'):     {'T1': ('best_x_roc_test', 71.97, 4.41),
                                 'T2': ('best_clf_test',   44.22, 5.57)},
    ('GIN', 'ogbg_molhiv'):     {'T2': ('best_clf_test',   76.47, 1.53)},
    ('GIN', 'Graph-SST2'):      {'T2': ('best_clf_test',   82.95, 0.58)},
    ('PNA', 'ba_2motifs'):      {'T1': ('best_x_roc_test', 93.77, 3.90)},
    ('PNA', 'mutag'):           {'T1': ('best_x_roc_test', 99.07, 0.50)},
    ('PNA', 'mnist'):           {'T1': ('best_x_roc_test', 84.68, 1.06),
                                 'T2': ('best_clf_test',   93.96, 0.92)},
    ('PNA', 'spmotif_0.5'):     {'T1': ('best_x_roc_test', 83.34, 2.17),
                                 'T2': ('best_clf_test',   68.74, 2.24)},
    ('PNA', 'spmotif_0.7'):     {'T1': ('best_x_roc_test', 86.94, 4.05),
                                 'T2': ('best_clf_test',   64.38, 3.20)},
    ('PNA', 'spmotif_0.9'):     {'T1': ('best_x_roc_test', 88.66, 2.44),
                                 'T2': ('best_clf_test',   57.01, 2.95)},
    ('PNA', 'ogbg_molhiv'):     {'T2': ('best_clf_test',   80.24, 0.73)},
    ('PNA', 'Graph-SST2'):      {'T2': ('best_clf_test',   80.92, 0.66)},
    ('PNA', 'ogbg_molbace'):    {'T3': ('best_clf_test',   77.41, 2.42)},
    ('PNA', 'ogbg_molbbbp'):    {'T3': ('best_clf_test',   69.17, 1.12)},
    ('PNA', 'ogbg_molclintox'): {'T3': ('best_clf_test',   87.80, 2.36)},
    ('PNA', 'ogbg_moltox21'):   {'T3': ('best_clf_test',   74.96, 0.66)},
    ('PNA', 'ogbg_molsider'):   {'T3': ('best_clf_test',   57.58, 1.23)},
}

# Paper datasets that are molecules (fragmentable); tracked for later vocab work.
MOLECULAR = ['mutag', 'ogbg_molhiv', 'ogbg_molbace', 'ogbg_molbbbp',
             'ogbg_molclintox', 'ogbg_moltox21', 'ogbg_molsider']

CELLS = sorted(TARGETS)                                   # 21 (backbone, dataset)
RUNS = [(bb, ds, s) for (bb, ds) in CELLS for s in SEEDS]  # 210


if __name__ == '__main__':
    if '--n' in sys.argv:
        print(len(RUNS))
    elif '--list' in sys.argv:
        for i, (bb, ds, s) in enumerate(RUNS):
            print(i, bb, ds, s)
    else:
        print(__doc__)
