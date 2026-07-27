# Rule Set 1 — Pipeline Validation

**Status:** authoritative. Survives across chats. Revisit and edit freely.

**How to invoke:** say *"invoke Rule Set 1"* (optionally scoped: *"…on the BBBP/fg_first/GIN run"* or *"…on results_YYYYMMDD.csv"*). On invocation I read this file and execute the checks below **as validation over statistics and outputs**, reporting **pass/fail + the exact offending cells** — never a summary adjective.

---

## 0. Operating principles (apply to every invocation)

1. **Statistics are the verdict; code reading is only a hypothesis.** A green code review proves nothing about 5,000 cells. An observable number that violates an invariant proves an error.
2. **Every check runs *through* the hook/seam, never around it.** A test that hand-feeds past the point where the bug can occur (e.g. passing SMARTS directly instead of driving from the dataset name through `get_loaders`) is theater. Drive from the real entry point.
2a. **Test as HPC invokes it — CLI/env/shell, not just direct method calls.** The outermost seam is `env vars → run_experiments.sh/run_full_pipeline.sh → argparse → script`. Many bugs live only there (`RULE_ENGINE` default, argparse `choices` rejecting dnf keys, `_gt_tier_list`, `SKIP_SELF_EXPLAINING`, `--mage_official_positive_class`, the source-GT hook). Every stage therefore also gets a **CLI/end-to-end smoke** that runs the real script (via subprocess) and, where relevant, the shell phase, on a tiny fixture, asserting the produced artifacts — not internal return values.
3. **Hunt loopholes / bypasses / fallbacks explicitly.** Silent fallbacks (rBRICS→BRICS, missing FG file → all-leftover, absent GT-attr → rule-mask fallback, stale cache) are the dangerous class because they produce plausible-looking results.
4. **A test only counts if it maps to a real failure mode.** Design tests from the failure taxonomy (one per mode, at its seam), not one per function. No decorative suites.
5. **Cheap gates before expensive compute.** Run Stages 1–2 (+ GT) for the whole matrix *before* training; a data/fragmentation error must cost minutes, not a multi-day sweep.
6. **Banned language / behavior:** never claim "full picture", "everything's healthy", "verified" as a blanket. Every claim states its **scope** and its **evidence**. Distinguish "I ran X, it showed Y" (evidence) from "I think Z" (inference to confirm with a statistic).
7. **Cross-branch principle:** an error may live in *one* branch (one backbone, one fragmentation, one dataset type), not all. Check each branch; do not generalize one branch's pass to the others.

## Execution protocol

Run stages in order 1→5, cheapest first. Stages 1–2 (+GT) gate 3–5. Output per stage: `PASS` or `FAIL` with the offending `(dataset, fragmentation, backbone, model, fold, label, tier, explainer)` cells listed. The dated results CSV is the single source of truth; these checks are scans over it plus targeted recomputation.

**Path = one branch chosen at each stage.** Branch dimensions:

| Stage | Branch dimension |
|---|---|
| 1 Load | CSV-classification · CSV-regression · source-GT-CSV (`_Verified_GT`) · mutag-TU · OGB |
| 2 Fragment | rbrics · fg_first · ertl_first · rdkit_fg_first · (all_fallback_bpe) · ±merge · ±threshold-filter |
| 3 Train | Vanilla · GSAT · MotifSAT · MOSE × {GIN,GCN,SAGE,GAT,PNA} × {real, synthetic-GT} |
| 4 Explain+Eval | GNNExplainer·PGExplainer·MotifOcclusion·MAGE / ante-hoc native × {synthetic, source} GT × metric family |
| 5 Save+Aggregate | summary.json → dated CSV (axes, metric-map, provenance) |

---

## Stage 1 — Load

**1.A CSV-classification** (BBBP, Mutagenicity, hERG)
1. Label col = `DATASET_COLUMN[ds]` and present in CSV (not a KeyError-fallback, not `group`). *LH: wrong target trained.*
2. `TASK_TYPE=='BinaryClass'`, `num_classes==1`. *LH: regression/multilabel misrouted.*
3. Splits `{training,valid,test}` disjoint, cover all rows. *LH: fold file is a full re-split → cross-fold test overlap.*
4. Every `x` is one-hot (`NUM_ATOM_TYPES`=51). *LH: dim mismatch silently projected.*
5. Invalid SMILES dropped **and counted**, never a 0-node graph. *LH: empty graph inflates negatives.*
6. **Seam→frag/GT:** `MolFromSmiles(data.smiles)` atom order == `x` rows. *LH: re-canonicalization → motif/GT on wrong atoms.*
7. Determinism + `force_reprocess` actually rebuilds. *LH: stale `.pt` after code change.*

**1.B CSV-regression** (esol, Lipophilicity)
1. `Regression`; normalization mean/std from **train only**, passed to val/test. *LH: leakage.*
2. Rule mining / synthetic GT refused. *LH: planting on continuous labels.*
3. Metric RMSE; no `gt_roc`/`node_label` columns. *LH: AUC on regression, or empty GT scored.*

**1.C source-GT-CSV** (`Benzene_Verified_GT`, `Fluoride_Carbonyl_Verified_GT`, `Alkane_Carbonyl_Verified_GT`)
1. **Hook driven by name:** `get_loaders(<name>)` → graphs carry `node_label`, via the `SOURCE_GT_SMARTS[name]` lookup, not hand-fed SMARTS. *LH: name typo → lookup None → no GT, no error (today's bug).*
2. **Attribution:** positive `node_label` == independent SMARTS match on `data.smiles`, per molecule. *LH: atom-order drift.*
3. **Fold-consistent:** a molecule in multiple folds has identical `node_label`. *LH: fold-dependent GT.*
4. **Label↔GT agreement:** positives have GT atoms, negatives none, %positive ≈ known prevalence. *LH: label and substructure disagree.*
5. **No double-GT:** `∉ GT_SUPPORTED_DATASETS`; `--use_gt` raises. *LH: source + synthetic GT stacked.*

**1.D mutag-TU** — pre-baked `node_label`/`edge_label`; single fold; 14-dim; **positive class `y==0`** (GT-ROC must not use a flipped label); known ~57.7% loader-drop surfaced as a number (see mutag investigation note).

**1.E OGB** — single fold; scaffold split; 9-dim raw → AtomEncoder (not the 51-dim one-hot); export bridge ran; molhiv scale flagged (run last). *LH: encoder/dim mismatch.*

**Cross-branch invariant:** every Stage-1 statistic is **identical across all downstream cells of the same dataset** (load is model/frag-independent).

## Stage 2 — Fragment

**2·shared** (every method, per-molecule, over sample + oracle molecules)
1. Exact partition: `∪ motif_atoms == {0..N-1}` **and** pairwise disjoint. *LH: coverage 100% while an atom is double-owned/dropped.*
2. No `owner==-1` before threshold filter (UNK only from 2.filter). *LH: silent −1 as a motif id.*
3. Connectedness per motif. *LH: the `O.O.O` class.*
4. Whole-FG (no subset claim). *LH: the pre-fix bug.*
5. No `fg:`/`chain:`/`frag:` key leaks after rekey. *LH: internal label mismatches rule keys.*
6. Determinism of `(key, sorted(atoms))`. *LH: vocab differs across backbones.*
7. Label truthfulness: induced submol canonical == key structure; `ring:` atoms actually in a ring. *LH: benzene key over 5C+O.*
8. **Seam←load:** `nodes_to_motifs` indexed in `data.smiles` atom order. *LH: off-by-one.*
9. **Seam→GT/rule:** emitted keys share the rule/`apply_gt` namespace. *LH: rule references K′, never fires, GT empty.*

**2.A rbrics** — BRICS fallback **observable/counted** (*LH: silent method mislabel*); coverage reported, unclaimed handled; trivial fragments keyed consistently.
**2.B fg_first** — curated FG dict loaded non-empty (*LH: mis-path → all-leftover, coverage still 100%*); rings-first (no FG carves a ring atom); leftovers → connected chains.
**2.C ertl_first** — connectedness/whole-FG; **vocab must differ from rdkit_fg** (*LH: one silently ran the other*); Ertl FG matches spec on oracles.
**2.D rdkit_fg_first** — `FunctionalGroups.txt` present/loaded (pattern count>0) (*LH: missing file → all-leftover silently*); semantic FG match.
**2.E all_fallback_bpe** — deterministic BPE merges; still exact partition; flag if it collapses distinct FGs.
**2.merge (cascade_bpe_linker)** — rings frozen (*LH: merge dissolves a ring*); MDL-improving + capped (*LH: runaway giant motifs*); learn/apply replay deterministic (*LH: split-dependent vocab*); re-run 2·shared **after** merge.
**2.filter** — UNK only from filtering, model never uses `-1` as a real index (*LH: crash/garbage vector*); filtered variant reuses **base** rules (*LH: re-mines its own*); coverage-after-filter reported; excessive-UNK guard fires.

**Cross-branch invariant:** vocab + `nodes_to_motifs` **identical across all backbones/models** for a `(dataset, method)`; the three FG methods yield **different** vocabs; rBRICS fallback **counted, never silent**.

## Stage 3 — Train

**3·shared**
1. Fragmentation reaches the model: `nodes_to_motifs` non-degenerate (not all −1) for motif-using models; vocab matches rules/GT. *LH: motif machinery a silent no-op.*
2. Correct label: real→CSV/source label, synthetic-GT→relabelled `y` from `gt_cache`. *LH: gt run trains on real label.*
3. Declared losses actually backpropped (nonzero coef ⇒ contributes to `total` and moves params). *LH: computed-but-detached loss.*
4. Best-model/early-stop on the intended metric; saved checkpoint is that epoch. *LH: undertraining — task-AUC saturates before explainer sharpens.*
5. Checkpoint round-trip + **correct reuse** (`load_weights_from` loads the matching `(ds,frag,bb,tier,label)`). *LH: GT baselines explain a different model.*
6. `conv_normalize` honored (none/L2/LayerNorm). *LH: silent norm regression (broke MOSE-rbrics).*
7. Determinism/seed recorded. *LH: cross-backbone gaps become noise.*

**3.A Vanilla** — no injection, no aux losses; `nodes_to_motifs` carried but doesn't alter forward pass (*LH: accidental injection invalidates post-hoc comparison*); learned above chance.
**3.B GSAT** — injection 010; IB loss executed + backpropped; stochastic mask genuinely sampled, not collapsed. *LH: IB detached / mask collapse.*
**3.C MotifSAT** — `motif_method readout`, `noise motif`, `info_loss_level motif`, injection 111 applied (*LH: silent node-level fallback*); motif losses do what coefs say — effective within = `motif_loss_coef×within_node_coef`, between = `motif_loss_coef×between_motif_coef`; if 0, recorded **inert** not "set but gated" (*LH: the exact trap*); real IB present, `info_loss_coef` avoids score saturation at 1.0.
**3.D MOSE** — injection 101; **native impact only** ("agnostic" disables the bottleneck, meaningless — never compare to baselines' agnostic); no real IB → score-freeze ~0.5 flagged; `conv_normalize` at intended value; multi-explanation runs here.
**3.backbone** — PNA degree histogram from train (*LH: missing → PNA degraded*); GAT heads + `edge_attr` when ablation says so; injection points + loss stack identical across all 5 (*LH: one branch skips a term*); no lone collapse (all 5 in a plausible band).
**3.label** — real/gt each own checkpoint, GT-eval reuses GT checkpoint (*LH: crossover*); source-GT trained on source label, scored vs SMARTS `node_label`, never synthetically relabelled.
**3.ablation** — {no-norm|L2|LayerNorm} × {onehot|linear} × {add|mean readout} × {edge|no-edge}: **each toggle must demonstrably change the trained model/forward pass** (*LH: a no-op toggle makes the ablation measure nothing*).

**Cross-branch invariant:** inputs received (label set, `nodes_to_motifs`) identical across backbones/models for a `(dataset,frag)`; and the loss stack **claimed** == **executed** == **moved gradients**.

## Stage 4 — Explain + Eval

**4·shared**
1. Correct GT attr, **no fallback** to rule/`edge_label` mask when the attr is absent (skip instead). *LH: `_has_node_attr` fallback mislabels metric.*
2. Attribution ↔ node-order alignment. *LH: GT-ROC on misaligned atoms.*
3. Explicit node→motif reduction (mean AND max = the reduction, applied once).
4. Non-degenerate: `score_std>0`; collapse flagged not reported. *LH: PGExplainer 0.5 reported as real.*
5. Correct graph population; `#graphs-scored` recorded.
6. Correct split: post-hoc test-only, ante-hoc its list. *LH: explaining train graphs.*
7. Every metric finite or carries a NaN reason.

**4.A GNNExplainer** — `node_mask_type='object'`, edge None; frozen model, no contamination. *LH: wrong mask → near-zero scores.*
**4.B PGExplainer** — mask-collapse retry **observable**; all-`edge_size` collapse ⇒ NaN by design, not silent 0.5.
**4.C MotifOcclusion** — the real occlusion baseline (not old fake "MAGE"); deterministic; add/mean readout a no-op.
**4.D MAGE** — freq-normed `S_cm=S_mm@P`; `--mage_official_positive_class`/`--mage_predicted_class` match polarity (mutag `y==0`); recon CE drops; known signature (weak on ubiquitous causes, high family / low GT, near model-invariant) is **expected**, not a bug.
**4.E ante-hoc native** — own attribution (not post-hoc re-explanation); MOSE native-only; `*_instance_all` present for ante-hoc.

**4.metric** (as defined)
1. **Global GT-ROC** ("recovered **all** clauses") = per graph, GT⁺ = union of all fired disjuncts' atoms; per-graph-mean over rule-positive graphs.
2. **Instance GT-ROC** ("recovered **any** clause completely") = per graph, max over fired disjuncts of AUC(that disjunct's atoms), other fired disjuncts excluded from the negative set; mean ± std. When one disjunct fires, == Global.
3. **Grouped Pearson/Spearman** = per-motif-averaged mean\|Δprob\|, UNK excluded, **no freq filter**.
4. **Instance Pearson/Spearman** = per-instance, no groupby-motif.
5. GT-ROC is the fragmentation correctness target; **Pearson-on-real is not the headline**.
6. **Top-10 vs Bottom-10**: importance, impact, discriminativeness, node-mask probe (gated vs raw).
7. **Multi-explanation (MOSE)** = distinct motifs per concept, not redundant variants.

**4.save** — every `{ex}_{agg}_{metric}` key named exactly as the aggregator reads; channel symmetry (gt/fired/spurious/family per explainer); write survives a crashed explainer; NaN with reason.

**Cross-branch invariant:** for a fixed model, all explainers scored against the same GT attr, population, alignment; deterministic explainers reproduce, stochastic vary within a band; metric **claimed** == **computed** == GT-attr **used**.

## Stage 5 — Save + Aggregate

**5·shared** — 1:1 summary→row completeness; no silent metric drop; each identifier its own column; no cross-axis averaging; **provenance stamped** (engine + code-version + config hash); dated, append-only, idempotent; NaN-with-reason; correct units.
**5.axis** — tier/rule axis from authoritative field, admits `dnf_k*_r*[_rare]`/easy-medium-hard/source, never coerced to `none`; real/gt its own column, tables computed **per** (real|gt); `vocab_variant` split tier-aware; `{ex}_{agg}` de-multiplexed without collision.
**5.dtype** — regression RMSE-only; source-GT tier=source; MOSE native-only column; multi-explanation its own table.
**5.papers** — deterministic scope filters: MOSE journal = rbrics(+fg_first appendix); Synthetic-labelling = all frags, exclude MOSE/MotifSAT; MotifSAT = all frags/models. Row in a view **iff** it matches scope.
**Final invariants** — round-trip (sample rows == source summary.json); count reconciliation (`#rows == #summaries == expected cells` = the inventory).

---

## Policy decisions

**RULED:**
1. **Provenance marker** — stamp every row with **commit SHA + timestamp + config/env hash**.
2. **Re-run policy** — **keep both rows, versioned; never overwrite.** A path is re-run ONLY when the code affecting *that path* changed since the row's stamped SHA; otherwise the stale row stands and is identified by its timestamp/SHA. (Provenance drives the re-run decision — no unnecessary recompute.)
5. **Pearson-on-real** — **include as a diagnostic** in the synthetic-labelling paper (never as the correctness headline; GT-ROC is the headline).

3. **Undertraining** — **keep early-stop on val task metric unchanged.** The explanation must not be improved at the cost of prediction. (Fixed-budget is a worthwhile *future* experiment, not now.) Record the stop epoch per cell; the undertraining caveat stands as a known limitation, not a code change.
6. **`--mage_official_positive_class`** — **per-dataset check: `1` for all, `0` for mutag** (property-active polarity). Regression: MAGE not run.
4. **GT-ROC granularity (final):** question asked — *did the explainer recover any clause completely (Instance) vs all clauses (Global)?*
   - **Instance GT-ROC** = per graph, **max over fired disjuncts** of AUC(that disjunct's atoms) → recovered **any one** clause completely; mean ± std over rule-positive graphs.
   - **Global GT-ROC** = per graph, GT⁺ = **union of all fired disjuncts' atoms** → recovered **all** clauses; **per-graph-mean** over rule-positive graphs (same aggregation as Instance, so they differ only in any-vs-all).
   - Per-disjunct AUC negative set **excludes the other fired disjuncts' atoms** (never penalize surfacing another genuine cause).
   - When exactly one disjunct fires, Instance == Global.
7. **Hard-fail vs flag** — for `_Verified` cross-fold overlap, rBRICS→BRICS fallback, all-leftover molecule, mostly-UNK-after-filter: **flag+count when present; hard-fail only when the dataset-level magnitude crosses a threshold** (the fingerprint of an actual bug — unloaded FG dict, misconfigured filter). Keep mostly-UNK/all-leftover graphs in training, record the fraction.

**INVESTIGATED — root cause MEASURED (a prior code-trace guess was refuted by measurement):**
- **mutag drop (57.7%) — MEASURED CAUSE:** a **deliberate selection rule** in `datasets/mutag.py` drops mutagen graphs (class 0) that lack an annotated NO2/NH2 motif edge. Measured on `data/mutag/raw`: 2,401 mutagens, only 1,015 (42.3%) carry the motif → **1,386 (57.7%) dropped** → remaining mutagens are 100% motif-bearing ⇒ **toxicophore recall = 100% by construction** (Faber et al. dataset-bias pitfall; PGExplainer-inherited convention). GT is also label-dependent (341 nonmutagens have NO2/NH2 but GT zeroed because y==1) — circular.
- **REFUTED guess (recorded as a caution):** the SMILES reconstruction (`graph_to_smiles.py`, all bonds forced `SINGLE`) was hypothesized to cause the drop. **Measurement: all-single fails sanitization on only 17/4337 (0.4%)** — RDKit fills implicit Hs, so it's fine. Threading the real bond orders (`Mutagenicity_edge_labels.txt`, valence 0/1/2) instead makes it **worse** — 785/4337 (18.1%) fail, 27.3% of mutagens — so it is NOT a fix. The reconstruction is not the problem; the selection rule is. *(Lesson: this ruleset exists because that code-trace story was confident and wrong; only measuring caught it.)*
  - **RULED (b):** keep mutag as source/calibration only; **flag + caveat** the recall-by-construction (report the drop rule and the 42.3%→100% inflation); never present mutag GT-ROC as evidence an explainer "found the cause." (a) remains a future clean-up option.

## Change log
- v1 (draft, this chat): Stages 1–5 + operating principles + open decisions. Awaiting approval.

## Implemented components (this session)
- **Instance vs Global GT-ROC** — `motif_eval.dnf_gt_roc_graph`/`compute_dnf_gt_roc`, `apply_gt` `node_label_clauses [N,K]`, per-explainer emission in `run_vanilla`, metric_map in `aggregate_experiments`. Instance=max-over-fired-clauses (sufficiency), Global=union (completeness); negatives exclude other fired clauses; per-graph-mean.
- **Provenance stamping** — `SharedModules/evaluation/provenance.py` (git_sha/run_timestamp/config_hash); `run_vanilla` writes it into summary.json.
- **Harvester** — `analysis/harvest_results.py`: one dated, provenance-stamped, append-only+idempotent CSV (`results_YYYYMMDD.csv`); paper-scope filters; count reconciliation. Verified on 3530 local summaries.
- **Test spine** — Stage 1–5 + CLI/E2E unit tests (49 tests + 12 subtests green): test_stage1_load / test_stage2_fragment / test_stage3_train / test_stage4_eval / test_stage5_aggregate / test_cli_e2e.
