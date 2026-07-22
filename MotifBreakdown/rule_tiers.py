"""rule_tiers.py — select easy / medium / hard planted rules by TRAINED-P2 difficulty.

Why trained P2 and not a data-side score
----------------------------------------
The project already established (calibrated against source-GT Benzene / Fluoride /
Alkane) that NO data-side statistic defines rule difficulty: spurious / separation
reject rules a trained model recovers cleanly and miss rules it cannot. Difficulty
is EMPIRICAL — the clause-aware GT-ROC of a reference motif-occluder on a model
trained on the rule. So the difficulty axis here is **P2** (occluder GT-ROC),
measured by a small GIN screen, exactly as the validated ``gates_p2.py`` harness.

  * P4  (test AUC)     — VALIDITY gate: the rule must be learnable (>= TAU_P4).
  * cov (label balance)— VALIDITY gate: non-degenerate, cov in [COV_LO, COV_HI].
  * P2  (occluder GT-ROC) — the DIFFICULTY axis; absolute bands map to easy/med/hard.
  * foolability (data-side LR shortcut AUC over distractor motifs) — REPORTED per
    clause, never a gate (it measures shortcut AVAILABILITY, not the model's use).

This phase-1 grade is a PRIOR (GIN, few folds). The AUTHORITATIVE ordering is the
post-phase5 confirm against the real trained backbones (Mode-2 clause GT-ROC); if
that ordering disagrees, relabel easy/medium/hard by the confirmed P2 (as gates_p2
does). Keying is the production motif key (rings canonical, no fallback); a literal
is one motif key, a clause an AND of keys, a rule a DNF (OR of clauses).
"""
from __future__ import annotations

import itertools
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# ── difficulty / validity constants (mirror gates_p2.py) ─────────────────────────
COV_LO, COV_HI = 0.15, 0.50          # non-degenerate label balance
TAU_P4 = 0.75                        # learnable (GNN grader only)
BANDS = [('easy', 0.85, 1.01), ('medium', 0.70, 0.85), ('hard', -0.01, 0.70)]  # GNN-P2 bands
MIN_SUP = 0.01                       # motif must appear in >=1% of molecules to be a literal
LIT_COV_LO, LIT_COV_HI = 0.03, 0.60  # a literal's own coverage band (pool candidates)
POOL_SINGLES = 40                    # cap balanced singles (top by |cov-0.3|)
POOL_C, POOL_D = 12, 3               # max conjunctions / DNFs

# grading budget (GNN grader only — the cyclic/expensive prior). Overridable.
SCREEN_EPOCHS = 25
CONFIRM_EPOCHS = 35
NOCC = 60                            # max positive graphs occluded per (backbone,fold)

# ── DEFAULT grader: fast LR proxy (no GNN, no cyclic dependency) ───────────────────
# Difficulty for the EXPLANATION task is driven by SHORTCUT AVAILABILITY: if non-rule
# motifs linearly predict the rule label, a model/explainer can latch onto the shortcut
# instead of the true cause → low GT-ROC → HARD. So difficulty = foolability = AUC of
# LogReg(non-rule motif occurrence -> rule label). This is a data-side PROXY (fast, no
# GNN); the authoritative difficulty is still the phase-5 models' measured GT-ROC, which
# can relabel the tiers post-hoc. `learnable_auc` (LogReg on atom composition) is
# reported as the composition-vs-structure axis + a non-degeneracy gate.
FOOL_BANDS = [('easy', -0.01, 0.62), ('medium', 0.62, 0.78), ('hard', 0.78, 1.01)]
LEARN_MIN = 0.60                     # rule must be at least composition-learnable (LR AUC)
LR_CV_FOLDS = 3


# ─────────────────────────────────────────────────────────────────────────────
# Featurization — DEPLOYED 51-dim atom-type one-hot (SharedModules/data/dataset.ATOMS).
# No aromatic / in-ring / degree flags: those leak the very structure the model must
# learn (measured: benzene P2 0.99 -> 0.62 once removed).
# ─────────────────────────────────────────────────────────────────────────────

def _featurizer():
    import torch
    import torch.nn.functional as F
    from rdkit import Chem
    from SharedModules.data.dataset import ATOMS, NUM_ATOM_TYPES

    def feat(smi: str):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        idx = [ATOMS.get(a.GetSymbol(), 0) for a in m.GetAtoms()]
        x = F.one_hot(torch.tensor(idx), NUM_ATOM_TYPES).float()
        rows, cols = [], []
        for b in m.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            rows += [i, j]; cols += [j, i]
        ei = torch.tensor([rows, cols], dtype=torch.long) if rows else torch.zeros((2, 0), dtype=torch.long)
        return x, ei, m.GetNumAtoms()
    return feat


def _make_gnn(kind: str, din: int, h: int = 64, L: int = 3):
    import torch.nn as nn
    from torch_geometric.nn import GINConv, GCNConv, SAGEConv, global_mean_pool

    def conv(d, o):
        if kind == 'GIN':
            return GINConv(nn.Sequential(nn.Linear(d, o), nn.ReLU(), nn.Linear(o, o)))
        if kind == 'GCN':
            return GCNConv(d, o)
        return SAGEConv(d, o)

    import torch

    class G(nn.Module):
        def __init__(s):
            super().__init__()
            s.c = nn.ModuleList([conv(din if l == 0 else h, h) for l in range(L)])
            s.o = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

        def forward(s, x, ei, b=None):
            if b is None:
                b = torch.zeros(x.size(0), dtype=torch.long)
            for c in s.c:
                x = torch.relu(c(x, ei))
            return s.o(global_mean_pool(x, b)).squeeze(-1)

    return G()


# ─────────────────────────────────────────────────────────────────────────────
# Train + motif-occlusion grade one rule  (port of gates_p2.train_grade)
# ─────────────────────────────────────────────────────────────────────────────

def _train_grade(cls: List[Tuple[str, ...]],
                 base: List,                       # per-mol (x, ei, n_atoms) or None
                 mf: List[List[Tuple[str, frozenset]]],
                 backbones: Sequence[str],
                 folds: int,
                 epochs: int) -> dict:
    """Return {P4, P2, P5drop, P2_std, pos} averaged over backbones x folds.

    P2 = clause-aware occluder GT-ROC: for each positive graph, occlude each motif
    instance (zero its atom features), score |Δp|; ROC of that saliency vs the atoms
    of the FIRED clause(s). Best clause AUC per graph, mean over graphs.
    """
    import torch
    import torch.nn as nn
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from sklearn.metrics import roc_auc_score

    def clause_fires(cl, i):
        ms = {k for k, _ in mf[i]}
        return all(k in ms for k in cl)

    def clause_gt_atoms(cl, i):
        at: Set[int] = set()
        for k, ats in mf[i]:
            if k in cl:
                at |= set(ats)
        return at

    graphs, meta = [], []
    for i, b in enumerate(base):
        if b is None:
            continue
        x, ei, na = b
        fired = [cl for cl in cls if clause_fires(cl, i)]
        graphs.append(Data(x=x, edge_index=ei,
                           y=torch.tensor([1.0 if fired else 0.0])))
        meta.append((i, fired, na))
    if len(graphs) < 20:
        return dict(P4=float('nan'), P2=float('nan'), P5drop=float('nan'),
                    P2_std=0.0, pos=float('nan'))

    rows = []
    for fold in range(folds):
        rng = np.random.RandomState(fold)
        idx = rng.permutation(len(graphs))
        a, bb = int(.7 * len(graphs)), int(.85 * len(graphs))
        tr = [graphs[j] for j in idx[:a]]
        va = [graphs[j] for j in idx[a:bb]]
        te = [graphs[j] for j in idx[bb:]]
        tem = [meta[j] for j in idx[bb:]]
        for kind in backbones:
            m = _make_gnn(kind, tr[0].x.size(1))
            opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
            bce = nn.BCEWithLogitsLoss()
            best, bs = -1.0, None
            for _ in range(epochs):
                m.train()
                for bt in DataLoader(tr, batch_size=256, shuffle=True):
                    opt.zero_grad()
                    bce(m(bt.x, bt.edge_index, bt.batch), bt.y).backward()
                    opt.step()
                m.eval(); ys, ps = [], []
                with torch.no_grad():
                    for bt in DataLoader(va, batch_size=256):
                        ps.append(torch.sigmoid(m(bt.x, bt.edge_index, bt.batch)).numpy())
                        ys.append(bt.y.numpy())
                yv = np.concatenate(ys)
                v = roc_auc_score(yv, np.concatenate(ps)) if len(set(yv)) > 1 else .5
                if v > best:
                    best = v; bs = {k: t.clone() for k, t in m.state_dict().items()}
            if bs:
                m.load_state_dict(bs)
            m.eval(); ys, ps = [], []
            with torch.no_grad():
                for bt in DataLoader(te, batch_size=256):
                    ps.append(torch.sigmoid(m(bt.x, bt.edge_index, bt.batch)).numpy())
                    ys.append(bt.y.numpy())
            yt = np.concatenate(ys)
            p4 = roc_auc_score(yt, np.concatenate(ps)) if len(set(yt)) > 1 else np.nan
            p2s, drops = [], []
            for d, (i, fired, na) in zip(te, tem):
                if d.y.item() < 0.5 or not fired:
                    continue
                z = torch.zeros(na, dtype=torch.long)
                sc = np.zeros(na)
                with torch.no_grad():
                    b0 = torch.sigmoid(m(d.x, d.edge_index, z)).item()
                for k, ats in mf[i]:
                    xx = d.x.clone(); xx[list(ats)] = 0
                    with torch.no_grad():
                        sc[list(ats)] = b0 - torch.sigmoid(m(xx, d.edge_index, z)).item()
                best_auc = np.nan
                for cl in fired:
                    gt = np.zeros(na, bool); gt[list(clause_gt_atoms(cl, i))] = True
                    if gt.any() and not gt.all():
                        try:
                            a2 = roc_auc_score(gt, sc)
                            best_auc = a2 if np.isnan(best_auc) else max(best_auc, a2)
                        except Exception:
                            pass
                if not np.isnan(best_auc):
                    p2s.append(best_auc)
                gtall: Set[int] = set()
                for cl in fired:
                    gtall |= clause_gt_atoms(cl, i)
                xx = d.x.clone(); xx[list(gtall)] = 0
                with torch.no_grad():
                    abl = torch.sigmoid(m(xx, d.edge_index, z)).item()
                drops.append(b0 - abl)
                if len(p2s) >= NOCC:
                    break
            rows.append(dict(P4=float(p4),
                             P5drop=float(np.mean(drops)) if drops else np.nan,
                             P2=float(np.mean(p2s)) if p2s else np.nan))
    P4 = float(np.nanmean([r['P4'] for r in rows]))
    P5 = float(np.nanmean([r['P5drop'] for r in rows]))
    p2v = [r['P2'] for r in rows if not np.isnan(r['P2'])]
    return dict(P4=P4, P5drop=P5,
                P2=float(np.mean(p2v)) if p2v else np.nan,
                P2_std=float(np.std(p2v)) if len(p2v) > 1 else 0.0,
                pos=float(np.mean([g.y.item() for g in graphs])))


# ─────────────────────────────────────────────────────────────────────────────
# Foolability (data-side) — reported, never a gate  (port of gates_p2.spurious_for)
# ─────────────────────────────────────────────────────────────────────────────

def _foolability(cls, cand_keys, pres, motset, n, thr=0.15) -> List[dict]:
    """Per-clause data-side shortcut report: shortcut_auc = LR(non-rule motif
    occurrence -> clause label) AUC (held-out) + the top positively-correlated
    distractor motifs. (The old label-permutation `spur_z` null was removed: it was
    deprecated as misleading — flags anything above chance at large n — and cost 25
    extra LR fits per clause. shortcut_auc is the meaningful exploitability measure.)"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rule_keys = {k for cl in cls for k in cl}
    distract = [k for k in cand_keys if k not in rule_keys]
    X = np.stack([pres[k] for k in distract], 1).astype(float) if distract else np.zeros((n, 0))
    rng = np.random.RandomState(0)
    tr = rng.rand(n) < 0.7
    rep = []
    for cl in cls:
        yc = np.array([1.0 if all(k in motset[i] for k in cl) else 0.0 for i in range(n)])
        corr = []
        for j, k in enumerate(distract):
            p = X[:, j]
            if p.std() == 0 or yc.std() == 0:
                continue
            r = float(np.corrcoef(p, yc)[0, 1])
            if r >= thr:
                corr.append((k, round(r, 2)))
        corr.sort(key=lambda t: -t[1])
        if X.shape[1] == 0 or len(set(yc)) < 2 or len(set(yc[tr])) < 2 or len(set(yc[~tr])) < 2:
            obs = 0.5
        else:
            try:
                lr = LogisticRegression(max_iter=200)
                lr.fit(X[tr], yc[tr])
                obs = float(roc_auc_score(yc[~tr], lr.predict_proba(X[~tr])[:, 1]))
            except Exception:
                obs = 0.5
        rep.append(dict(clause='&'.join(cl), shortcut_auc=round(obs, 3),
                        top_distractors=corr[:5]))
    return rep


# ─────────────────────────────────────────────────────────────────────────────
# Fast LR grader (default) — no GNN, no cyclic dependency
# ─────────────────────────────────────────────────────────────────────────────

def _lr_cv_auc(X: np.ndarray, y: np.ndarray, folds: int = LR_CV_FOLDS,
               seed: int = 0) -> float:
    """Mean stratified-CV ROC-AUC of LogisticRegression(X -> y). 0.5 if degenerate."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    if X.shape[1] == 0 or len(set(y.tolist())) < 2:
        return 0.5
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    aucs = []
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te, assume_unique=False)
        if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
            continue
        try:
            lr = LogisticRegression(max_iter=200)
            lr.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], lr.predict_proba(X[te])[:, 1]))
        except Exception:
            continue
    return float(np.mean(aucs)) if aucs else 0.5


def _atom_hist_matrix(base: List, n: int) -> np.ndarray:
    """Per-molecule atom-type histogram (sum of the 51-dim one-hot over atoms).
    A composition-only feature (no structure): LogReg on it measures how learnable a
    rule is WITHOUT message passing."""
    dim = None
    for b in base:
        if b is not None:
            dim = b[0].shape[1]
            break
    dim = dim or 1
    H = np.zeros((n, dim), dtype=float)
    for i, b in enumerate(base):
        if b is not None:
            H[i] = b[0].sum(0).cpu().numpy()
    return H


def _lr_grade_rule(cls, cand, pres, motset, H, n):
    """Fast LR difficulty grade for ONE rule (list of clauses of motif keys):
    foolability = LR(non-rule motif occurrence -> rule label) AUC (shortcut availability),
    learnable   = LR(atom composition -> rule label) AUC (composition learnability)."""
    y = np.array([1.0 if any(all(k in motset[i] for k in cl) for cl in cls) else 0.0
                  for i in range(n)])
    rule_keys = {k for cl in cls for k in cl}
    distract = [k for k in cand if k not in rule_keys]
    Xd = np.stack([pres[k] for k in distract], 1).astype(float) if distract else np.zeros((n, 0))
    fool = _lr_cv_auc(Xd, y)
    learn = _lr_cv_auc(H, y)
    return float(fool), float(learn), y


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def select_tiers(smiles: List[str],
                 mol_frags: List[List[Tuple[str, Set[int]]]],
                 *,
                 grader: str = 'lr',
                 backbones: Sequence[str] = ('GIN',),
                 confirm_backbones: Sequence[str] = ('GIN', 'GCN', 'SAGE'),
                 folds: int = 1,
                 confirm_folds: int = 2,
                 screen_epochs: int = SCREEN_EPOCHS,
                 confirm_epochs: int = CONFIRM_EPOCHS,
                 log: Callable[[str], None] = print) -> Dict[str, dict]:
    """Select one representative rule per difficulty band (easy/medium/hard).

    smiles     : one SMILES per molecule.
    mol_frags  : per molecule, list of (motif_key, atom_index_set) — the production
                 tracked fragmentation (mol_frags_tracked). Keys must be the SAME
                 strings used in rules.json / annotation_lookup so apply_gt matches.

    Returns {tier: rule_dict} where rule_dict is apply_gt-compatible:
        {'clauses': [{'motifs': [key, ...]}, ...], 'rule_str', 'P2', 'P4', 'cov',
         'P2_std', 'tier_band', 'foolability': [ per-clause ... ]}
    Missing bands are simply absent from the returned dict.
    """
    n = len(smiles)
    feat = _featurizer()
    base = [feat(s) for s in smiles]                       # None for unparseable

    motset = [{k for k, _ in mf} for mf in mol_frags]
    from collections import Counter
    cov = Counter()
    for s in motset:
        cov.update(s)
    kept = {k for k, c in cov.items() if c >= MIN_SUP * n}
    # literal candidates: kept motifs whose OWN coverage is in the literal band
    cand = [k for k in kept if LIT_COV_LO <= cov[k] / n <= LIT_COV_HI]
    pres = {k: np.array([k in motset[i] for i in range(n)]) for k in cand}

    def rule_cov(cls):
        fires = np.zeros(n, bool)
        for cl in cls:
            f = np.ones(n, bool)
            for k in cl:
                f &= pres[k]
            fires |= f
        return float(fires.mean())

    bal = lambda p: COV_LO <= p.mean() <= COV_HI
    jc = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)

    # ── pool: balanced singles + genuine conjunctions + a few disjoint DNFs ──────
    singles = [(k,) for k in cand if bal(pres[k])]
    singles.sort(key=lambda c: abs(cov[c[0]] / n - 0.3))
    singles = singles[:POOL_SINGLES]

    def genuine(a, b):
        mm = pres[a] & pres[b]
        pa, pb = pres[a].mean(), pres[b].mean()
        return bool(pa) and bool(pb) and mm.mean() > 0.03 and \
            mm.mean() / pa <= 0.7 and mm.mean() / pb <= 0.7
    conj_all = [(a, b) for a, b in itertools.combinations(cand, 2)
                if genuine(a, b) and bal(pres[a] & pres[b])]

    def spread(items, key, k):
        s = sorted(items, key=key)
        if len(s) <= k:
            return s
        return [s[round(j * (len(s) - 1) / (k - 1))] for j in range(k)]
    conjs = spread(conj_all, lambda ab: (pres[ab[0]] & pres[ab[1]]).mean(), POOL_C)

    dnfs = []
    chosen, mask = [], np.zeros(n, bool)
    dp = [(ab, pres[ab[0]] & pres[ab[1]]) for ab in conj_all]
    while len(chosen) < 3 and dp:
        cs = [(ab, mm) for ab, mm in dp if all(jc(mm, cm) <= 0.2 for _, cm in chosen)]
        if not cs:
            break
        ab, mm = max(cs, key=lambda x: (x[1] & ~mask).sum())
        chosen.append((ab, mm)); mask |= mm
        dp = [b for b in dp if b[0] != ab]
    if len(chosen) >= 2:
        dnfs.append([list(c[0]) for c in chosen])
    topc = sorted(conj_all, key=lambda ab: -(pres[ab[0]] & pres[ab[1]]).mean())
    for a, b in itertools.combinations(topc[:6], 2):
        if jc(pres[a[0]] & pres[a[1]], pres[b[0]] & pres[b[1]]) <= 0.2:
            dnfs.append([list(a), list(b)])
        if len(dnfs) >= POOL_D:
            break

    pool = ([('single', [list(c)]) for c in singles]
            + [('conj', [list(c)]) for c in conjs]
            + [('dnf', d) for d in dnfs])
    log(f"    [tiers] pool: {len(singles)} singles + {len(conjs)} conj + {len(dnfs)} dnf")

    def rule_str(cls):
        return ' ∨ '.join('(' + ' ∧ '.join(cl) + ')' for cl in cls)

    # ══ FAST LR GRADER (default) — no GNN, no cyclic dependency ═══════════════════
    if grader == 'lr':
        H = _atom_hist_matrix(base, n)
        graded = []
        for form, cls in pool:
            cvg = rule_cov(cls)
            if not (COV_LO <= cvg <= COV_HI):
                continue
            fool, learn, _ = _lr_grade_rule(cls, cand, pres, motset, H, n)
            ok = learn >= LEARN_MIN                       # non-degenerate (composition-learnable)
            graded.append(dict(form=form, cls=cls, cov=cvg,
                               foolability_auc=fool, learnable_auc=learn, valid=ok))
            log(f"    [tiers-lr] {form:6} cov={cvg:.2f} fool={fool:.3f} "
                f"learn={learn:.3f} {'OK' if ok else 'reject'}  {rule_str(cls)[:48]}")
        valids = [g for g in graded if g['valid']]
        log(f"    [tiers-lr] {len(valids)}/{len(graded)} usable "
            f"(cov∈[{COV_LO},{COV_HI}], learnable≥{LEARN_MIN})")
        # TERCILE banding on foolability (shortcut availability): the 3 planted rules
        # SPAN the available difficulty range (easy = least foolable third, hard = most).
        # Relative-within-dataset by design — the ABSOLUTE difficulty is the phase-5
        # models' measured GT-ROC, which can relabel the tiers post-hoc. Robust to the
        # data-dependent foolability distribution (fixed thresholds can leave bands empty).
        vs = sorted(valids, key=lambda g: g['foolability_auc'])
        m = len(vs)
        if m >= 3:
            t = m // 3
            groups = {'easy': vs[:t], 'medium': vs[t:m - t], 'hard': vs[m - t:]}
        elif m == 2:
            groups = {'easy': [vs[0]], 'hard': [vs[1]]}
        elif m == 1:
            groups = {'medium': [vs[0]]}
        else:
            groups = {}
        out = {}
        for lvl in ('easy', 'medium', 'hard'):
            inband = groups.get(lvl) or []
            if not inband:
                log(f"    [tiers-lr] {lvl:6}: (no rule in this tercile)")
                continue
            # representative: most solidly learnable, mid-coverage
            rep = max(inband, key=lambda g: (g['learnable_auc'], -abs(g['cov'] - 0.3)))
            out[lvl] = dict(
                clauses=[{'motifs': list(cl)} for cl in rep['cls']],
                rule_str=rule_str(rep['cls']), form=rep['form'],
                cov=round(rep['cov'], 4),
                foolability_auc=round(rep['foolability_auc'], 4),
                learnable_auc=round(rep['learnable_auc'], 4),
                grader='lr', tier_band=lvl,
                # per-clause data-side shortcut detail (same LR family), for inspection
                foolability=_foolability([tuple(cl) for cl in rep['cls']],
                                         cand, pres, motset, n),
            )
            log(f"    [tiers-lr] {lvl:6}: {rep['form']} fool={rep['foolability_auc']:.3f} "
                f"learn={rep['learnable_auc']:.3f} cov={rep['cov']:.2f}  {rule_str(rep['cls'])[:48]}")
        return out

    # ══ GNN GRADER (opt-in: grader='gnn') — the trained-P2 prior (cyclic, slow) ════
    # ── STAGE A: screen whole pool (cheap GIN), gate on cov & P4 ──────────────────
    screened = []
    for form, cls in pool:
        cvg = rule_cov(cls)
        if not (COV_LO <= cvg <= COV_HI):
            continue
        agg = _train_grade([tuple(cl) for cl in cls], base, mol_frags,
                           backbones, folds, screen_epochs)
        valid = (agg['P4'] >= TAU_P4) and not np.isnan(agg['P2'])
        screened.append(dict(form=form, cls=cls, cov=cvg, valid=valid, **agg))
        log(f"    [tiers] {form:6} cov={cvg:.2f} P4={agg['P4']:.3f} "
            f"P2={agg['P2']:.3f} {'OK' if valid else 'reject'}  {rule_str(cls)[:52]}")

    valids = [s for s in screened if s['valid']]
    log(f"    [tiers] {len(valids)}/{len(screened)} learnable "
        f"(cov∈[{COV_LO},{COV_HI}], P4≥{TAU_P4})")

    selected = {}
    for lvl, lo, hi in BANDS:
        inband = [s for s in valids if lo <= s['P2'] < hi]
        if not inband:
            log(f"    [tiers] {lvl:6}: (no valid rule with P2∈[{lo:.2f},{hi:.2f}))")
            continue
        rep = max(inband, key=lambda s: (s['P4'], -abs(s['cov'] - 0.3)))
        selected[lvl] = rep
        log(f"    [tiers] {lvl:6}: {rep['form']} P2={rep['P2']:.3f} "
            f"cov={rep['cov']:.2f}  {rule_str(rep['cls'])[:56]}")

    # ── STAGE B: confirm selected on more backbones/folds + foolability ──────────
    out = {}
    for lvl, rep in selected.items():
        cls = rep['cls']
        agg = _train_grade([tuple(cl) for cl in cls], base, mol_frags,
                           confirm_backbones, confirm_folds, confirm_epochs)
        fool = _foolability([tuple(cl) for cl in cls], cand, pres, motset, n)
        out[lvl] = dict(
            clauses=[{'motifs': list(cl)} for cl in cls],
            rule_str=rule_str(cls),
            form=rep['form'],
            grader='gnn',
            cov=round(rep['cov'], 4),
            P4=round(agg['P4'], 4),
            P2=round(agg['P2'], 4) if not np.isnan(agg['P2']) else None,
            P2_std=round(agg['P2_std'], 4),
            P5drop=round(agg['P5drop'], 4) if not np.isnan(agg['P5drop']) else None,
            foolability=fool,
        )
        log(f"    [tiers] confirm {lvl:6} P4={agg['P4']:.3f} P2={agg['P2']:.3f} "
            f"(spread={agg['P2_std']:.3f})")

    # RELABEL by the CONFIRMED multi-backbone P2 (authoritative over the 1-fold screen
    # that assigned the bands) — otherwise labels can violate their own P2 ordering.
    if len(out) >= 2:
        order = sorted(out, key=lambda l: -(out[l]['P2'] if out[l]['P2'] is not None else -1))
        names = ['easy', 'medium', 'hard'][:len(order)]
        out = {nm: {**out[ol], 'tier_band': nm} for nm, ol in zip(names, order)}
    else:
        for lvl in out:
            out[lvl]['tier_band'] = lvl
    return out
