# -*- coding: utf-8 -*-
"""Exact day-level relabelling tests and test-inverted intervals for fig6_model_contrast.py. perm_generic, _one_sided and ratio_ci
are copies of the functions in rates_data.py (which runs its whole computation on import). ratio_ci keeps r when both exact one-sided
p values of the adjusted data (differentiated-day values divided by r, or log values shifted by log r) exceed 0.025; ratio_of_ratios_ci
inverts the test of a ratio of two differentiated/same factors."""
from itertools import combinations
import numpy as np, pandas as pd


def perm_generic(v, n_same, stat='mean'):
    v = np.asarray(v, float)
    if stat == 'rank':
        v = pd.Series(v).rank().values
    k = len(v); obs = v[n_same:].mean() - v[:n_same].mean(); null = []
    for S_ in combinations(range(k), n_same):
        m = np.ones(k, bool); m[list(S_)] = False
        null.append(v[m].mean() - v[~m].mean())
    null = np.array(null)
    return float(np.mean(np.abs(null) >= abs(obs) - 1e-12)), float(np.mean(null >= obs - 1e-12)), len(null)


def _one_sided(v, n_same):
    k = len(v); obs = v[n_same:].mean() - v[:n_same].mean(); null = []
    for S_ in combinations(range(k), n_same):
        m = np.ones(k, bool); m[list(S_)] = False
        null.append(v[m].mean() - v[~m].mean())
    null = np.array(null)
    return float(np.mean(null >= obs - 1e-12)), float(np.mean(null <= obs + 1e-12))


def ratio_ci(v, n_same, log=False, grid=np.exp(np.linspace(np.log(0.02), np.log(50), 1200))):
    v = np.asarray(v, float); ok = []
    for r in grid:
        w = v.copy()
        if log:
            w[n_same:] = w[n_same:] - np.log(r)
        else:
            w[n_same:] = w[n_same:] / r
        pu, pl = _one_sided(w, n_same)
        ok.append(pu > 0.025 and pl > 0.025)
    ok = np.array(ok)
    est = np.exp(v[n_same:].mean() - v[:n_same].mean()) if log else v[n_same:].mean() / v[:n_same].mean()
    i0 = int(np.argmin(np.abs(np.log(grid) - np.log(est))))
    if not ok[i0]:
        return np.nan, np.nan
    lo = i0
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    hi = i0
    while hi < len(grid) - 1 and ok[hi + 1]:
        hi += 1
    return (grid[lo] if lo > 0 else 0.0), (grid[hi] if hi < len(grid) - 1 else np.inf)


def ratio_of_ratios_ci(free, lead, n_same, grid=np.exp(np.linspace(np.log(0.02), np.log(50), 1200))):
    """Sensitivity: test inversion for the ratio-of-group-means contrast T = log[(mean_D f / mean_S f) / (mean_D l / mean_S l)].
    For each r the differentiated-day unconstrained rates are divided by r and the exact one-sided p values of T over the
    C(k, n_same) relabellings are computed; r is kept when both exceed 0.025."""
    f0 = np.asarray(free, float); l = np.asarray(lead, float); k = len(f0)
    labs = []
    for S_ in combinations(range(k), n_same):
        m = np.ones(k, bool); m[list(S_)] = False; labs.append(m)
    def T(f, m):
        return np.log(f[m].mean() / f[~m].mean()) - np.log(l[m].mean() / l[~m].mean())
    obs_m = np.r_[np.zeros(n_same, bool), np.ones(k - n_same, bool)]
    ok = []
    for r in grid:
        f = f0.copy(); f[n_same:] = f[n_same:] / r
        t0 = T(f, obs_m); null = np.array([T(f, m) for m in labs])
        ok.append(np.mean(null >= t0 - 1e-12) > 0.025 and np.mean(null <= t0 + 1e-12) > 0.025)
    ok = np.array(ok)
    est = np.exp(T(f0, obs_m)); i0 = int(np.argmin(np.abs(np.log(grid) - np.log(est))))
    if not ok[i0]:
        return est, np.nan, np.nan
    lo = i0
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    hi = i0
    while hi < len(grid) - 1 and ok[hi + 1]:
        hi += 1
    null = np.array([T(f0, m) for m in labs]); t0 = T(f0, obs_m)
    return est, (grid[lo] if lo > 0 else 0.0), (grid[hi] if hi < len(grid) - 1 else np.inf), float(np.mean(null >= t0 - 1e-12)), float(np.mean(np.abs(null) >= abs(t0) - 1e-12))
