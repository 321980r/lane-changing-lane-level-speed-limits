# -*- coding: utf-8 -*-
"""Independent day rates of unconstrained and leader-driven inward changes of cars per car-km in that situation, with exact day-level
tests and a slope on Delta (the functions are also used by check_lane_shift.py).
Inputs: _cache/work/check_sit_state.pkl, check_sit_events.pkl. Outputs: results/check_sit_day_rates.csv, check_sit_tests.csv."""
import os, sys, itertools, numpy as np, pandas as pd
from lc_common import *

S = pd.read_pickle(os.path.join(CACHE, 'check_sit_state.pkl'))
E = pd.read_pickle(os.path.join(CACHE, 'check_sit_events.pkl'))
E = E[E.cls == 'car'].copy()
names = [n for n, _, _ in DAYS]
D8 = names
dl = {n: vi - vo for n, vi, vo in DAYS}


def exact_same_diff(v, nsame=3):
    """mean-diff and rank-sum exact tests (two-sided, one-sided)"""
    from itertools import combinations
    from scipy.stats import rankdata
    v = np.asarray(v, float); k = len(v); rk = rankdata(v)
    obs = v[nsame:].mean() - v[:nsame].mean(); obsr = rk[:nsame].sum()
    nm, nr = [], []
    for Sg in combinations(range(k), nsame):
        m = np.zeros(k, bool); m[list(Sg)] = True
        nm.append(v[~m].mean() - v[m].mean()); nr.append(rk[m].sum())
    nm = np.array(nm); nr = np.array(nr); mu = nsame * (k + 1) / 2
    return dict(obs=obs, p2=np.mean(np.abs(nm) >= abs(obs) - 1e-12), p1=np.mean(nm >= obs - 1e-12),
                p2_rank=np.mean(np.abs(nr - mu) >= abs(obsr - mu) - 1e-12), p1_rank=np.mean(nr <= obsr + 1e-12), nperm=len(nm))


def slope_perm(v, x):
    v = np.asarray(v, float); x = np.asarray(x, float)
    b = np.polyfit(x, v, 1)[0]
    null = np.array([np.polyfit(np.array(p), v, 1)[0] for p in itertools.permutations(x)])
    return b, np.mean(np.abs(null) >= abs(b) - 1e-12), np.mean(null >= b - 1e-12)


def rates(look=300.0, lag=3, xlo=900.0, xhi=None, require_lane2=True, hw_c=4.0, gap_c=150.0, dv_c=3.0, ev=None, st=None):
    """returns per-day table of free and close-leader events / km"""
    xhi = (X1 - look) if xhi is None else xhi
    st = S if st is None else st; ev = E if ev is None else ev
    s = st[(st.PositionX >= xlo) & (st.PositionX <= xhi)]
    km = s.Smoothed_VelocityX * 0.1 / 1000.0
    free = s.gap.isna() | (s.gap >= look)
    hw = s.gap / s.Smoothed_VelocityX.clip(lower=0.1)
    close = (~s.gap.isna()) & ((hw < hw_c) | ((s.gap < gap_c) & ((s.Smoothed_VelocityX - s.lead_v) * 3.6 > dv_c)))
    ex = pd.DataFrame({'km_free': km[free].groupby(s.day[free]).sum(), 'km_close': km[close].groupby(s.day[close]).sum(),
                       'km_all': km.groupby(s.day).sum()}).reindex(names)
    e = ev[(ev[f'ok{lag}'] == 1)].copy()
    e = e[(e[f'x{lag}'] >= xlo) & (e[f'x{lag}'] <= xhi)]
    if require_lane2:
        e = e[e[f'lane{lag}'] == 2]
    g = e[f'gap{lag}']; v = e[f'v{lag}']
    e_free = g.isna() | (g >= look)
    e_close = g.notna() & ((g / v.clip(lower=0.1) < hw_c) | ((g < gap_c) & ((v - e[f'lv{lag}']) * 3.6 > dv_c)))
    ex['n_free'] = e[e_free].groupby('day').size().reindex(names).fillna(0)
    ex['n_close'] = e[e_close].groupby('day').size().reindex(names).fillna(0)
    ex['r_free'] = ex.n_free / ex.km_free; ex['r_close'] = ex.n_close / ex.km_close
    ex['delta'] = [dl[n] for n in names]
    return ex


if __name__ == '__main__':
    T = rates()
    T.to_csv(os.path.join(RES, 'check_sit_day_rates.csv'))
    print(T[['n_free', 'km_free', 'r_free', 'n_close', 'km_close', 'r_close']].round(3).to_string())
    rows = []
    for col in ('r_free', 'r_close'):
        v8 = T.loc[D8, col].values
        t8 = exact_same_diff(v8)
        b, ps2, ps1 = slope_perm(v8, T.loc[D8, 'delta'].values)
        rows.append(dict(rate=col, same_mean=v8[:3].mean(), diff_mean=v8[3:].mean(), ratio=v8[3:].mean() / v8[:3].mean(),
                         p2_mean8=t8['p2'], p1_mean8=t8['p1'], p2_rank8=t8['p2_rank'],
                         slope_per10=b * 10, slope_p2=ps2, slope_p1=ps1, min_diff_over_max_same=v8[3:].min() / v8[:3].max()))
        # pooled counts
    R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, 'check_sit_tests.csv'), index=False)
    print(R.round(4).to_string())
    print('pooled free: same', T.loc[D8[:3], 'n_free'].sum(), round(T.loc[D8[:3], 'km_free'].sum(), 1), ' diff', T.loc[D8[3:], 'n_free'].sum(), round(T.loc[D8[3:], 'km_free'].sum(), 1))
    print('pooled close: same', T.loc[D8[:3], 'n_close'].sum(), round(T.loc[D8[:3], 'km_close'].sum(), 1), ' diff', T.loc[D8[3:], 'n_close'].sum(), round(T.loc[D8[3:], 'km_close'].sum(), 1))
