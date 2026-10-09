# -*- coding: utf-8 -*-
"""Helpers for the versions_ scripts: get_carin() car-inward decision points of the eight days with the M0 / M1 / M1p MOBIL terms (causal
desired-speed proxy, x = 1,000-1,800 m; cached in _cache/work/versions_carin.pkl); design_m2f() the M2F design (M0 terms + F + F x D) with
options (without the desired-speed term, day constants, bias linear in Delta); lodo() leave-one-day-out fits, log-likelihood and
held-out hazards; day_factors() differentiated/same factors of the day rates at unconstrained decision points (F = 1) and at decision
points with a leader within 300 m (F = 0), each change assigned to the step closest to 3 s before its recorded time."""
import gc, os, time
import numpy as np, pandas as pd
from model_lib import *          # noqa  (steps, features, design, ich_*, event_ids, M, MAIN, SAME, FIN, CACHE, DAY_META, perm8 ...)

CARIN = os.path.join(CACHE, 'versions_carin.pkl')
KEEP = ['day', 'GID', 'GlobalID', 'Timestamp', 'PositionX', 'v', 'truck', 'free', 'olgap', 'y', 'y5', 'dt_next', 't_next', 'vdes_use',
        'vd10', 'entered_by_lc', 'delta', 'd10', 'diff_day', 'vin', 'vout', 'drop5']


def get_carin():
    if os.path.exists(CARIN):
        return pd.read_pickle(CARIN)
    t0 = time.time()
    D = steps('causal', 0)
    D = D[D.day.isin(MAIN) & (D.cls == 'car') & (D.dirn == 'in')].reset_index(drop=True); gc.collect()
    G = M.gaps(D)
    out = D[KEEP].copy()
    for spec in ('M0', 'M1', 'M1p'):
        Fm = M.mobil_terms(D, G, spec)
        for c in ('dA_ego', 'dA_n', 'dA_o', 'unsafe'):
            out[f'{spec}_{c}'] = Fm[c].values.astype(float)
    del D, G; gc.collect()
    out.to_pickle(CARIN)
    print('built', CARIN, len(out), 'points', round(time.time() - t0, 1), 's', flush=True)
    return out


def terms(D, spec='M0'):
    return pd.DataFrame({c: D[f'{spec}_{c}'].values for c in ('dA_ego', 'dA_n', 'dA_o', 'unsafe')})


def design_m2f(D, spec='M0', free=True, bias='step', drop_v=False, dayfe=False, z=None):
    """columns: const | dA_ego dA_n dA_o unsafe [vd10] state | [free] [bias]; dayfe replaces const by 8 day dummies (MAIN order).
    bias: 'step' = F x D, 'lin' = F x Delta/10, None; z (optional) = relabelled D or Delta/10 per point"""
    F = terms(D, spec)
    cols, X = [], []
    if dayfe:
        for d in MAIN:
            X.append((D.day.values == d).astype(float)); cols.append(f'day_{d[10:15]}')
    else:
        X.append(np.ones(len(D))); cols.append('const')
    X += [F.dA_ego.values, F.dA_n.values, F.dA_o.values, F.unsafe.values]; cols += ['dA_ego', 'dA_n', 'dA_o', 'unsafe']
    if not drop_v:
        X.append(D.vd10.values); cols.append('vd10')
    X.append(D.entered_by_lc.values.astype(float)); cols.append('state')
    fr = D.free.values.astype(float)
    if free:
        X.append(fr); cols.append('free')
    if bias is not None:
        if z is None:
            z = D.diff_day.values.astype(float) if bias == 'step' else D.d10.values.astype(float)
        X.append(fr * z); cols.append('free_x_diff' if bias == 'step' else 'free_x_d10')
    return np.column_stack(X), cols


def b0_for(cols):
    b = np.zeros(len(cols))
    for i, c in enumerate(cols):
        if c == 'const' or c.startswith('day_'):
            b[i] = -5.0
    return b


def ratio_ci_delta(b, V, i_num, i_den):
    r = b[i_num] / b[i_den]
    g = np.zeros(len(b)); g[i_num] = 1 / b[i_den]; g[i_den] = -b[i_num] / b[i_den] ** 2
    s = float(np.sqrt(max(g @ V @ g, 0)))
    return r, r - 1.96 * s, r + 1.96 * s, s


def lodo(D, X, cols, ev, b_all):
    """leave-one-day-out over the 8 days. With day dummies the held-out day gets the mean of the 7 training-day intercepts."""
    h = np.full(len(D), np.nan); ll_by_day = {}; fits = {}
    dayc = [i for i, c in enumerate(cols) if c.startswith('day_')]
    for d in MAIN:
        sel = (D.day == d).values; tr = ~sel
        if dayc:
            jd = cols.index(f'day_{d[10:15]}')
            keep = [i for i in range(len(cols)) if i != jd]
            b, _ = ich_fit(X[tr][:, keep], ev[tr], b0=b_all[keep])
            bf = np.zeros(len(cols)); bf[keep] = b
            bf[jd] = np.mean([bf[i] for i in dayc if i != jd])
        else:
            bf, _ = ich_fit(X[tr], ev[tr], b0=b_all)
        fits[d] = bf
        ll_by_day[d] = ich_ll(bf, X[sel], ev[sel])
        h[sel] = expit(X[sel] @ bf)
    return sum(ll_by_day.values()), h, ll_by_day, fits


def event_state(D, ev):
    """free state of each event at the step closest to 3 s before the recorded stamp"""
    w = pd.DataFrame({'ev': ev, 'day': D.day.values, 'free': D.free.values, 'd3': (D.dt_next.values - 3.0)})
    w = w[w.ev >= 0].copy(); w['d3'] = w.d3.abs()
    return w.sort_values(['ev', 'd3'], kind='mergesort').groupby('ev').first().reset_index()


def day_rates(D, ev, h=None):
    """per day: km, observed events and (optionally) predicted rate, for F = 1 and F = 0"""
    km = D.v.values / 3600.0
    es = event_state(D, ev)
    rows = []
    for d in MAIN:
        sd = (D.day == d).values
        for st, fv in (('free', 1), ('leader', 0)):
            s = sd & (D.free.values == fv)
            n = int(((es.day == d) & (es.free == fv)).sum())
            r = dict(day=d, state=st, km=float(km[s].sum()), n_obs=n)
            r['rate_obs'] = n / r['km']
            if h is not None:
                r['rate_pred'] = float(h[s].sum() / r['km'])
            rows.append(r)
    return pd.DataFrame(rows)


def factors(R, col):
    """differentiated/same factor (ratio of day means, 8 days) for F = 1 and F = 0, and their ratio"""
    out = {}
    for st in ('free', 'leader'):
        g = R[R.state == st].set_index('day').reindex(MAIN)
        s = np.array([d in SAME for d in MAIN])
        out[st] = g[col].values[~s].mean() / g[col].values[s].mean()
    out['contrast'] = out['free'] / out['leader']
    return out
