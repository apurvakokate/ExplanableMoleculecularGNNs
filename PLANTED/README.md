# PLANTED — source-verified planted-ground-truth datasets

This directory exists to **guarantee that Benzene, Fluoride_Carbonyl, and Alkane_Carbonyl are the
correct datasets, taken directly from their authoritative source** — not the earlier, broken local
copies. It holds only the verification artifacts; nothing derived (vocabularies, processed graphs,
training outputs) belongs here.

## Contents

```
gt/Benzene_Verified_GT_planted_gt.npz            atom-level planted GT, all explanation columns
gt/Fluoride_Carbonyl_Verified_GT_planted_gt.npz
gt/Alkane_Carbonyl_Verified_GT_planted_gt.npz
```

Each `.npz` maps `smiles -> node_imp [n_atoms x J]` (J = every non-empty subset of matching fragments;
grade **max-over-columns**, never against their union). SMILES are stored **verbatim from source** — do
not canonicalize, or the attribution atom indices desync.

Generator / verifier: `MotifBreakdown/export_planted_gt.py`. It fetches from the authoritative source
(Sanchez-Lengeling et al., *Evaluating Attribution for GNNs*, NeurIPS 2020 —
`github.com/google-research/graph-attribution`, tasks `benzene / logic7 / logic8`) and **fails loudly**
(`assert n_fp == 0 and n_fn == 0`) unless the labels and atom attributions reproduce the rule exactly.

| dataset (`*_Verified_GT`) | source task | planted rule |
|---|---|---|
| Benzene | `benzene` | `c1ccccc1` |
| Fluoride_Carbonyl | `logic7` | `[FX1]` **AND** `[CX3]=O` |
| Alkane_Carbonyl | `logic8` | `[R0;D2,D1][R0;D2][R0;D2,D1]` **AND** `[CX3]=O` |

## What was wrong with the previous (legacy) datasets

Three defects, all in the local legacy copies, all fixed by regenerating from source with
`export_planted_gt.py`:

1. **The names were SWAPPED.** Per the source rules, `logic7` is *fluoride ∧ carbonyl* and `logic8` is
   *unbranched-alkane ∧ carbonyl*. The legacy local files had `Alkane_Carbonyl` pointing at logic7
   (4,326 mols) and `Fluoride_Carbonyl` at logic8 (8,671 mols) — **backwards**. Any legacy result keyed
   by dataset name (vocab / results / caches / `skip_existing`) could silently report logic7 numbers as
   logic8. The `_Verified_GT` suffix makes those name collisions impossible.

2. **The label column was JUNK.** The source CSV's `label` column disagrees with the true rule
   (agreement 0.735 / 0.733 for logic7/8 — near chance). The upstream data-gen never reads it; it
   recomputes `y` from the rule. Legacy exports used that junk column, so a large fraction of labels were
   wrong. The `_Verified_GT` labels come from `y_true.npz` (the rule), not the CSV column.

3. **Alkane_Carbonyl's "alkane" SMARTS is element-agnostic** (`[R0;D2,D1][R0;D2][R0;D2,D1]` — acyclic +
   degree, no element constraint), so C–C–O, C–N–C, C–S–C all match; only ~24.4% of its "alkane" ground
   truth is pure carbon. The data faithfully implements the source pattern, but the pattern is chemically
   loose — **report this caveat whenever Alkane_Carbonyl results are shown.**

## Rule (project standard)

Always use the `*_Verified_GT` datasets. Treat any legacy result on bare `Benzene` /
`Alkane_Carbonyl` / `Fluoride_Carbonyl` as void (wrong labels and/or swapped names).

## Regenerating

```
python3 MotifBreakdown/export_planted_gt.py --out_root <FOLDS_dir> --gt_out PLANTED/gt
```

The fold CSVs (`{Dataset}_{fold}.csv`) are written to `<FOLDS_dir>` (currently kept outside this repo at
`../_PLANTED_csv_moved_20260720/`); only the GT `.npz` live here. Stale experiment output that used to
sit under `PLANTED/` was moved to `../_PLANTED_experiment_output_moved_20260721/` on 2026-07-21 — delete
it once you've confirmed nothing needs it.
