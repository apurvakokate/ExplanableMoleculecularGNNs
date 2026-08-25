#!/usr/bin/env python3
"""Assemble the two grouped-Pearson rollup inputs for the FOLD-CORRECTED Table 5.

The fold bug lived ONLY in the exclude-unk (Filt) kept-motif filter, so:
  * Filt (unk=exclude) rows  <- the V2 re-eval shards            (fold-CORRECT)
  * Full (unk=include) rows  <- the sealed `mose_replication` bundle rollups
                                (the fold bug never touched include-unk)

These are kept STRICTLY separate: Filt comes only from V2, Full only from the
bundle's include rows — so a Filt cell can never be contaminated by the old
bundle's (buggy) exclude rows. Rows are split GAT vs non-GAT because
gen_grouped_pearson_table.py routes --gat_rollups -> backbone=='GAT' and
--other_rollups -> backbone!='GAT'. MoSE-fixed vs MoSE_U is left to that script's
method_key(), which keys off `run_path` containing 'unk-learnable_shared'
(verified present in the V2 rollups).

Schema note: V2 shards carry one extra column (`unk_mode`) vs the bundle rollups;
pandas.concat aligns by name, so bundle rows simply get NaN there — harmless, the
generator ignores it.

Outputs: <OUT>/gat.csv and <OUT>/other.csv  (default OUT under V2/paper_deliverables).
Then run gen_grouped_pearson_table.py --gat_rollups <OUT>/gat.csv --other_rollups <OUT>/other.csv.
"""
import glob
import os

import pandas as pd

P = os.environ.get('ROOT', '/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor')
V2 = os.environ.get('V2', f'{P}/mose_replication_v2')
B = os.environ.get('BUNDLE', f'{P}/mose_replication')
OUT = os.environ.get('OUT', f'{V2}/paper_deliverables/_gp_inputs')
os.makedirs(OUT, exist_ok=True)


def cat(patterns):
    fs = [pd.read_csv(f) for p in patterns for f in sorted(glob.glob(p))]
    return pd.concat(fs, ignore_index=True) if fs else pd.DataFrame()


# --- Filt (exclude-unk) from the V2 re-eval — fold-correct ---
v2_gat = cat([f'{V2}/rollups/none/*/GAT/metrics_*_unk-exclude.csv',
              f'{V2}/rollups/source/*/GAT/metrics_*_unk-exclude.csv'])
v2_oth = cat([f'{V2}/rollups/none/*/nonGAT/metrics_*_unk-exclude.csv',
              f'{V2}/rollups/source/*/nonGAT/metrics_*_unk-exclude.csv'])
assert len(v2_gat) and len(v2_oth), 'no V2 exclude shards found — check V2 rollups path'
assert (v2_gat['unk'] == 'exclude').all() and (v2_oth['unk'] == 'exclude').all(), \
    'V2 shard carried a non-exclude row'

# --- Full (include-unk) from the sealed bundle — keep ONLY include rows ---
b_gat = cat([f'{B}/eval_gath1/gath1_rollup_all.csv',
             f'{B}/eval_gath1/gath1_rollup_include.csv'])
b_gat = b_gat[b_gat['unk'] == 'include'].drop_duplicates()
b_oth = cat([f'{B}/eval_real/eval_real_rollup_all.csv'])
b_oth = b_oth[b_oth['unk'] == 'include'].drop_duplicates()

gat = pd.concat([v2_gat, b_gat], ignore_index=True)
oth = pd.concat([v2_oth, b_oth], ignore_index=True)
gat.to_csv(f'{OUT}/gat.csv', index=False)
oth.to_csv(f'{OUT}/other.csv', index=False)

for nm, d in (('gat', gat), ('other', oth)):
    filt = int((d['unk'] == 'exclude').sum())
    full = int((d['unk'] == 'include').sum())
    print(f'wrote {OUT}/{nm}.csv  rows={len(d)}  Filt/V2={filt}  Full/bundle={full}')
    print(f'   backbones={sorted(d.backbone.dropna().unique())}')
    print(f'   methods={sorted(d.method.dropna().unique())}')
