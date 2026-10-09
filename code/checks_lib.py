# -*- coding: utf-8 -*-
"""Day-level helpers for the checks_ and versions_ scripts: exact relabelling test over the 56 labellings of the eight days (difference of
the group means, also on ranks), 95 % interval of the ratio of group means by inverting the two one-sided exact tests (contrast_tests),
Garwood intervals, the day groups (DAYS8, NS8) and the recording order of the days; out() writes a table to results/final."""
import os
from itertools import combinations, permutations
import numpy as np, pandas as pd
from scipy import stats
from plot_style import *          # noqa  (DATF, DAY_META, ORDER8, perm8 ...)

OUTD = DATF


def out(df, name):
    p = os.path.join(OUTD, name)
    df.to_csv(p, index=False)
    return p


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


def garwood(n, km):
    lo = stats.chi2.ppf(0.025, 2 * n) / 2 / km if n > 0 else 0.0
    hi = stats.chi2.ppf(0.975, 2 * (n + 1)) / 2 / km
    return lo, hi


DEL = DAY_META.set_index('day').delta.to_dict()
CFG = DAY_META.set_index('day').cfg.to_dict()
VIN = DAY_META.set_index('day').vin.to_dict()
VOUT = DAY_META.set_index('day').vout.to_dict()


def group_days(days_main, drop=None):
    d = [x for x in days_main if x != drop]
    same = [x for x in d if DEL[x] == 0]; diff = [x for x in d if DEL[x] > 0]
    return same + diff, len(same)


def contrast_tests(vals_by_day, days, n_same, log=False):
    v = np.array([vals_by_day[x] for x in days], float)
    p2, p1, nperm = perm_generic(v, n_same)
    pr = perm_generic(v, n_same, 'rank')[0]
    if log:
        ratio = float(np.exp(v[n_same:].mean() - v[:n_same].mean()))
    else:
        ratio = float(v[n_same:].mean() / v[:n_same].mean())
    lo, hi = ratio_ci(v, n_same, log=log)
    return dict(ratio=ratio, ci_lo=lo, ci_hi=hi, p2=p2, p1=p1, p2_rank=pr, nperm=nperm,
                same_mean=float(np.exp(v[:n_same].mean()) if log else v[:n_same].mean()),
                diff_mean=float(np.exp(v[n_same:].mean()) if log else v[n_same:].mean()),
                complete_sep=int(v[n_same:].min() > v[:n_same].max()))


DAYS8, NS8 = group_days(ORDER8)

# recording (calendar) order of the eight days
REC_ORDER = ['scen_2022-07-28_100-80', 'scen_2022-07-29_80-80', 'scen_2022-08-01_80-60', 'scen_2022-08-02_80-70',
             'scen_2022-08-03_90-70', 'scen_2022-08-05_100-70', 'scen_2022-08-08_60-60', 'scen_2022-08-11_100-100']
DATE = {d: pd.Timestamp(d[5:15]) for d in ORDER8}
