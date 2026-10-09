# -*- coding: utf-8 -*-
"""Situation definitions and day-level tests: state() loads the per-frame state tables of the eight days (situations_state.py); events()
attaches to each lane change the state of the vehicle 1, 3 and 5 s before it; lane2_state() and lane1_state() classify the situations in
the outer lane (C leader-driven, F unconstrained, M intermediate, N look-ahead truncated) and in the inner lane (R return, Y yield to a
follower, B blocked, K keep-right); perm_label(), perm_tests(), perm_slope() and quasi_poisson_ratio() are the day-level tests."""
import os
from itertools import combinations, permutations
from lc_common import *

ORDER8 = [d for d, _, _ in DAYS]
CFG = {d: f'{a}/{b}' for d, a, b in DAYS}
DEL = {d: a - b for d, a, b in DAYS}
VIN = {d: a for d, a, b in DAYS}
VOUT = {d: b for d, a, b in DAYS}
_S = None


def state():
    global _S
    if _S is None:
        _S = pd.concat([pd.read_pickle(os.path.join(CACHE, f'sit_state_{n}.pkl')) for n in ORDER8], ignore_index=True)
    return _S


def events():
    E = pd.read_pickle(os.path.join(CACHE, 'core_events.pkl')).copy()
    S = state()
    cols = ['day', 'GlobalID', 'Timestamp', 'PositionX', 'Lane', 'v', 'meas', 'hour', 'lead_gap', 'lead_dv', 'lead_truck', 'lag_gap', 'lag_dv',
            'oth_lead_gap', 'oth_lead_dv', 't_since_pass', 't_since_lc']
    for k in (3, 5, 1):
        E['_ts'] = E.t - k * 1000
        m = E[['day', 'GlobalID', '_ts']].merge(S[cols].rename(columns={'Timestamp': '_ts'}), on=['day', 'GlobalID', '_ts'], how='left')
        for c in cols[3:]:
            E[f's{k}_{c}'] = m[c].values
    E['hour_ev'] = C.hours(E.t)
    return E.drop(columns=['_ts'])


# ------------------------------------------------------------------------------------------------ state definitions
def lane2_state(x, gap, dv, v, L=300.0, H=4.0, G=150.0, DV=3.0, X1=2100.0):
    """state of a vehicle in lane 2 (candidate for an inward change).
    C = constrained by a slower/close leader (hw < H s, or gap < G m and closing > DV km/h)
    F = free: no leader within L m and the L m ahead lie inside the window (a leader would be visible)
    M = leader within L m but not constraining; N = no leader seen but view truncated by the window end."""
    x = np.asarray(x, float); gap = np.asarray(gap, float); dv = np.asarray(dv, float); v = np.asarray(v, float)
    hw = gap / np.maximum(v / 3.6, 0.1)
    has = np.isfinite(gap)
    C_ = has & ((hw < H) | ((gap < G) & (dv > DV)))
    near = has & (gap < L)
    view = x + L <= X1
    out = np.where(C_, 'C', np.where(near, 'M', np.where(view, 'F', 'N')))
    return out


def lane1_state(gap, dv, v, lag_gap, lag_dv, t_pass, t_lc, TP=30.0, TL=60.0):
    """state of a vehicle in lane 1 (candidate for an outward change), priority R > Y > B > K.
    R = return-eligible: made a cross-lane pass in the last TP s or entered lane 1 by a lane change in the last TL s
    Y = pressed by an own-lane follower (< 100 m, approaching > 5 km/h); B = blocked (hw < 3 s and closing); K = settled (keep-right motive only)"""
    gap = np.asarray(gap, float); dv = np.asarray(dv, float); v = np.asarray(v, float)
    hw = gap / np.maximum(v / 3.6, 0.1)
    R_ = (np.asarray(t_pass, float) <= TP) | (np.asarray(t_lc, float) <= TL)
    Y_ = (np.asarray(lag_gap, float) < 100) & (np.asarray(lag_dv, float) > 5)
    B_ = np.isfinite(gap) & (hw < 3) & (dv > 0)
    return np.where(R_, 'R', np.where(Y_, 'Y', np.where(B_, 'B', 'K')))


def event_states(E, k=3, **kw):
    """state of each event at t-k from the state table (NaN state -> 'U' unclassifiable)."""
    p = f's{k}_'
    st = np.full(len(E), 'U', dtype=object)
    ok = E[p + 'Lane'].notna().values
    inw = (E.direction == 'in').values
    L2 = {kk: v for kk, v in kw.items() if kk in ('L', 'H', 'G', 'DV')}
    L1 = {kk: v for kk, v in kw.items() if kk in ('TP', 'TL')}
    a = ok & inw
    st[a] = lane2_state(E.loc[a, p + 'PositionX'], E.loc[a, p + 'lead_gap'], E.loc[a, p + 'lead_dv'], E.loc[a, p + 'v'], **L2)
    b = ok & ~inw
    st[b] = lane1_state(E.loc[b, p + 'lead_gap'], E.loc[b, p + 'lead_dv'], E.loc[b, p + 'v'], E.loc[b, p + 'lag_gap'], E.loc[b, p + 'lag_dv'],
                        E.loc[b, p + 't_since_pass'], E.loc[b, p + 't_since_lc'], **L1)
    return st


def frame_states(S, **kw):
    st = np.full(len(S), '', dtype=object)
    L2 = {kk: v for kk, v in kw.items() if kk in ('L', 'H', 'G', 'DV')}
    L1 = {kk: v for kk, v in kw.items() if kk in ('TP', 'TL')}
    a = (S.Lane == 2).values
    st[a] = lane2_state(S.PositionX.values[a], S.lead_gap.values[a], S.lead_dv.values[a], S.v.values[a], **L2)
    b = ~a
    st[b] = lane1_state(S.lead_gap.values[b], S.lead_dv.values[b], S.v.values[b], S.lag_gap.values[b], S.lag_dv.values[b],
                        S.t_since_pass.values[b], S.t_since_lc.values[b], **L1)
    return st


# ------------------------------------------------------------------------------------------------ day-level tests
def perm_label(vals, n_same):
    """exact relabelling test, first n_same values = same-limit days. returns (diff-same, ratio, p_two, p_one, n_perm)."""
    v = np.asarray(vals, float); k = len(v)
    obs = v[n_same:].mean() - v[:n_same].mean()
    null = []
    for S in combinations(range(k), n_same):
        m = np.ones(k, bool); m[list(S)] = False
        null.append(v[m].mean() - v[~m].mean())
    null = np.array(null)
    ratio = v[n_same:].mean() / v[:n_same].mean() if v[:n_same].mean() > 0 else np.nan
    return obs, ratio, float(np.mean(np.abs(null) >= abs(obs) - 1e-12)), float(np.mean(null >= obs - 1e-12)), len(null)


def perm_tests(day_vals):
    """day_vals: dict day -> value for the 8 days. returns dict with the exact 56-relabelling test."""
    v8 = [day_vals[d] for d in ORDER8]
    o = perm_label(v8, 3)
    return dict(same=np.mean(v8[:3]), diff=np.mean(v8[3:]), diff_minus_same=o[0], ratio=o[1], p2_8=o[2], p1_8=o[3], nperm8=o[4])


def perm_slope(day_vals, xkey='delta', exact=True, ndraw=5000, seed=1):
    """slope of day value on a day covariate (delta, vout, vin), permuting the 8 (V_in, V_out) pairs across days.
    returns slope, p_two."""
    global _PERM
    y = np.array([day_vals[d] for d in ORDER8], float)
    x = np.array([{'delta': VIN[d] - VOUT[d], 'vout': VOUT[d], 'vin': VIN[d]}[xkey] for d in ORDER8], float)
    if _PERM is None:
        _PERM = np.array(list(permutations(range(8))))
    X = x[_PERM]                                  # 40320 x 8: covariate re-assigned to days
    Xc = X - X.mean(1, keepdims=True); yc = y - y.mean()
    slopes = (Xc * yc).sum(1) / (Xc ** 2).sum(1)
    xc = x - x.mean(); b_obs = (xc * yc).sum() / (xc ** 2).sum()
    return float(b_obs), float(np.mean(np.abs(slopes) >= abs(b_obs) - 1e-12))


_PERM = None


def quasi_poisson_ratio(counts, km, diff):
    """day-level quasi-Poisson GLM log(E count) = log km + b0 + b1*diff, dispersion from Pearson chi2 of the day counts.
    returns rate ratio and 95% CI (t with n-2 df)."""
    import statsmodels.api as sm
    from scipy import stats
    X = sm.add_constant(np.asarray(diff, float))
    m = sm.GLM(np.asarray(counts, float), X, family=sm.families.Poisson(), offset=np.log(np.asarray(km, float))).fit(scale='X2')
    b, se = m.params[1], m.bse[1]
    tq = stats.t.ppf(0.975, len(counts) - 2)
    return float(np.exp(b)), float(np.exp(b - tq * se)), float(np.exp(b + tq * se)), float(m.scale)
