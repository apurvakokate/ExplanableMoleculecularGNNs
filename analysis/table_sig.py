#!/usr/bin/env python3
"""Significance highlighting for the paper tables (Welch's t-test from mean/std/n).

Goal: in each comparison group, mark the single BEST cell, and every other cell
that is NOT significantly worse than the best (a statistical tie — "could have
reached that performance"). Best -> \\textbf{}, tie -> \\underline{} (default).

Test: Welch's two-sample t-test (unequal variances) using only (mean, std_sample,
n). std is the SAMPLE std (ddof=1) — exactly what the generators report. This is
the correct summary-only test; it does NOT use fold pairing (documented tradeoff).
Multiple competitors vs the best are Holm-Bonferroni corrected within the group.

Pure Python (t-distribution via regularized incomplete beta) — no scipy needed.
"""
import math

# ── t-distribution tail via regularized incomplete beta (Numerical Recipes) ──
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 1e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN: d = FPMIN
    d = 1.0 / d; h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d;  d = FPMIN if abs(d) < FPMIN else d
        c = 1.0 + aa / c;  c = FPMIN if abs(c) < FPMIN else c
        d = 1.0 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d;  d = FPMIN if abs(d) < FPMIN else d
        c = 1.0 + aa / c;  c = FPMIN if abs(c) < FPMIN else c
        d = 1.0 / d; de = d * c; h *= de
        if abs(de - 1.0) < EPS: break
    return h

def _betai(a, b, x):
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b

def _t_sf(t, df):                     # one-sided P(T > |t|)
    t = abs(t)
    return 0.5 * _betai(df / 2.0, 0.5, df / (df + t * t))

def welch_p(m1, s1, n1, m2, s2, n2):
    """Two-sided Welch p-value. NaN if untestable (n<2 or zero pooled variance)."""
    if None in (m1, s1, n1, m2, s2, n2): return float('nan')
    if any(v != v for v in (m1, s1, m2, s2)): return float('nan')   # NaN guard
    if n1 < 2 or n2 < 2: return float('nan')
    v1, v2 = s1 * s1 / n1, s2 * s2 / n2
    denom = math.sqrt(v1 + v2)
    if denom == 0.0:
        return 1.0 if m1 == m2 else 0.0
    t = (m1 - m2) / denom
    df = (v1 + v2) ** 2 / (v1 * v1 / (n1 - 1) + v2 * v2 / (n2 - 1))
    return 2.0 * _t_sf(t, df)

# ── Holm-Bonferroni: which competitors are SIGNIFICANTLY different from best ──
def _holm_reject(pvals, alpha):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals); rej = [False] * m
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            rej[i] = True
        else:
            break                      # Holm stops at first non-rejection
    return rej

# ── decide tags for one comparison group ──
def decide(cells, higher_better=True, alpha=0.05):
    """cells: list of dict(key, mean, std, n). Returns {key: 'best'|'tie'|'plain'}.
    'tie'  = not significantly worse than best (could have reached it).
    'plain'= significantly worse, OR untestable (n<2 / missing std -> not highlighted).
    """
    valid = [c for c in cells if c['mean'] == c['mean']]           # drop NaN means
    if not valid: return {}
    best = (max if higher_better else min)(valid, key=lambda c: c['mean'])
    tag = {best['key']: 'best'}
    testable = [c for c in valid if c is not best
                and welch_p(best['mean'], best['std'], best['n'],
                            c['mean'], c['std'], c['n']) == welch_p(best['mean'], best['std'], best['n'],
                            c['mean'], c['std'], c['n'])]           # p not NaN
    ps = [welch_p(best['mean'], best['std'], best['n'], c['mean'], c['std'], c['n']) for c in testable]
    rej = _holm_reject(ps, alpha) if ps else []
    for c, r in zip(testable, rej):
        tag[c['key']] = 'plain' if r else 'tie'
    for c in valid:
        tag.setdefault(c['key'], 'plain')                          # untestable competitors
    return tag

def wrap(s, tag, style='bold_underline'):
    """Wrap a formatted cell string per tag. style: 'bold_underline' (best bold,
    tie underline) or 'bold_all' (best+tie bold)."""
    if s is None or s.strip() in ('--', ''):
        return s
    if tag == 'best':
        return r'\textbf{' + s + '}'
    if tag == 'tie':
        return (r'\textbf{' + s + '}') if style == 'bold_all' else (r'\underline{' + s + '}')
    return s

CAPTION = ("Best result per group in \\textbf{bold}; results not significantly "
           "worse than the best (Welch's two-sided $t$-test on the per-fold "
           "mean/std, $\\alpha={alpha}$, Holm-corrected) \\underline{underlined}. "
           "Cells with $<2$ folds are not tie-tested.")
