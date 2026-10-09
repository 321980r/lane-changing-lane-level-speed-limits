# -*- coding: utf-8 -*-
"""Library of the lane-change model: interval-censored 1-s logit hazard (ich_nll, ich_fit, ich_ll and the cluster-robust ich_vcov;
the step that triggered a change is unknown within the 5 s before its recorded time), the 1 Hz decision table steps() with the
desired-speed proxy ('orig': running free-driving mean up to t, else the whole-track value; 'causal': running mean, else the day x class
median; 'lag3': 'causal' with all covariates taken 3 s earlier), MOBIL terms per specification (features(), via mobil_model.py), design
matrices, subsets, leave-one-day-out hazards and a vehicle-cluster bootstrap."""
import os, sys, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from plot_style import *          # noqa  (paths, DAYS, DAY_META, perm8 ...)
import mobil_model as M

M.AMIN = -4.0                      # as in decisions_models (accelerations floored at -4 m/s2)
MAIN = [n for n, _, _ in DAYS]
SAME = MAIN[:3]
GROUPS = (('car', 'in'), ('car', 'out'), ('truck', 'in'), ('truck', 'out'))
BASECOLS = ['dA_ego', 'dA_n', 'dA_o', 'unsafe', 'vd10', 'state']
FIN = os.path.join(RES, 'final')


# ------------------------------------------------------------------------------------------------ likelihood (own implementation)
def _prep_ev(ev):
    ev = np.asarray(ev, int); m = ev >= 0
    evl = np.full(len(ev), -1, int)
    if m.any():
        _, inv = np.unique(ev[m], return_inverse=True); evl[m] = inv
    return evl, int(evl.max() + 1) if m.any() else 0


def ich_nll(b, X, evl, nev):
    eta = X @ b
    sp = np.logaddexp(0.0, eta)                  # -log(1-h)
    h = expit(eta)
    out = evl < 0
    nll = sp[out].sum()
    grad = (X[out] * h[out, None]).sum(0)
    if nev:
        inn = ~out
        S = np.bincount(evl[inn], weights=sp[inn], minlength=nev)       # -log P(no trigger in window)
        S = np.maximum(S, 1e-300)
        nll -= np.log(-np.expm1(-S)).sum()                               # log P(>=1 trigger in window)
        w = h[inn] / np.expm1(S[evl[inn]])
        grad -= (X[inn] * w[:, None]).sum(0)
    return nll, grad


def ich_fit(X, ev, b0=None):
    X = np.asarray(X, float); evl, nev = _prep_ev(ev)
    if b0 is None:
        b0 = np.zeros(X.shape[1]); b0[0] = -5.0
    r = minimize(ich_nll, b0, args=(X, evl, nev), jac=True, method='BFGS', options={'gtol': 1e-6, 'maxiter': 5000})
    if not r.success:
        r2 = minimize(ich_nll, r.x, args=(X, evl, nev), jac=True, method='L-BFGS-B', options={'maxiter': 5000, 'gtol': 1e-8})
        if r2.fun <= r.fun:
            r = r2
    return r.x, -float(r.fun)


def ich_ll(b, X, ev):
    evl, nev = _prep_ev(ev)
    return -ich_nll(b, np.asarray(X, float), evl, nev)[0]


def ich_vcov(b, X, ev, groups, eps=1e-5):
    """cluster-robust sandwich (clusters = vehicles)"""
    X = np.asarray(X, float); evl, nev = _prep_ev(ev); k = len(b)
    H = np.zeros((k, k))
    for i in range(k):
        e = np.zeros(k); e[i] = eps
        H[:, i] = (ich_nll(b + e, X, evl, nev)[1] - ich_nll(b - e, X, evl, nev)[1]) / (2 * eps)
    H = (H + H.T) / 2; Hi = np.linalg.pinv(H)
    eta = X @ b; h = expit(eta); sp = np.logaddexp(0, eta); out = evl < 0
    sc = np.zeros_like(X); sc[out] = -X[out] * h[out, None]
    if nev:
        S = np.maximum(np.bincount(evl[~out], weights=sp[~out], minlength=nev), 1e-300)
        sc[~out] = X[~out] * (h[~out] / np.expm1(S[evl[~out]]))[:, None]
    Sg = pd.DataFrame(sc).groupby(np.asarray(groups)).sum().values; G = Sg.shape[0]
    V = Hi @ (Sg.T @ Sg) @ Hi * G / (G - 1)
    return V


def event_ids(S, H=5):
    """event id per step: steps with y5 = 1 grouped by vehicle and recorded stamp (t_next)"""
    flag = S[f'y{H}'].values == 1
    key = np.where(flag, S.GID.astype(str).values + '|' + S.t_next.fillna(-1).astype(np.int64).astype(str).values, '')
    ev = np.full(len(S), -1, int)
    if flag.any():
        _, inv = np.unique(key[flag], return_inverse=True); ev[flag] = inv
    return ev


# ------------------------------------------------------------------------------------------------ decision table
def steps(vsrc='causal', lag=0, x_lo=1000.0, x_hi=1800.0, H=5):
    """1 Hz decision points (lanes 1/2, cars and heavies) with MOBIL inputs. vsrc: 'orig' | 'causal'; lag: covariates k s earlier"""
    D = pd.read_pickle(os.path.join(CACHE, 'decision_points.pkl'))
    keep = ['GlobalID', 'Timestamp', 'PositionX', 'Lane', 'v', 'truck', 'ol_PositionX', 'ol_v', 'ol_truck', 'of_PositionX', 'of_v', 'of_truck',
            'tl_PositionX', 'tl_v', 'tl_truck', 'tf_PositionX', 'tf_v', 'tf_truck', 'vdes_pre', 'sv1', 'n1', 'sv2', 'n2', 'vdes_all',
            'entered_by_lc', 't_next', 'y3', 'y5', 'drop5', 'dt_next', 'day', 'vin', 'vout', 'delta', 'diff_day', 'hour', 'frame_source']
    D = D[keep]
    D = D[D.truck.isin([0, 1])]
    D = D.sort_values(['day', 'GlobalID', 'Timestamp'], ignore_index=True)
    D['GID'] = D.day.str[10:15] + '_' + D.GlobalID.astype(str)
    D['cls'] = np.where(D.truck == 1, 'truck', 'car')
    if vsrc == 'orig':
        D['vdes_use'] = D.vdes_pre.fillna(D.vdes_all).fillna(D.v).values
        D['vdes_fb'] = (D.vdes_pre.isna() & D.vdes_all.notna()).astype(int)
    else:
        med = D.groupby(['day', 'truck']).vdes_pre.transform('median')
        D['vdes_use'] = D.vdes_pre.fillna(med).values
        D['vdes_fb'] = D.vdes_pre.isna().astype(int)
    # speed factor: running free-flow mean of v / V_lane; fallback vdes_use / V_own_lane (as mobil_model.sf_from)
    D['sf_use'] = M.sf_from(D, D.vin.values.astype(float), D.vout.values.astype(float))
    if lag:
        cov = ['PositionX', 'v', 'ol_PositionX', 'ol_v', 'ol_truck', 'of_PositionX', 'of_v', 'of_truck', 'tl_PositionX', 'tl_v', 'tl_truck',
               'tf_PositionX', 'tf_v', 'tf_truck', 'vdes_use', 'sf_use', 'sv1', 'n1', 'sv2', 'n2', 'entered_by_lc', 'Lane']
        L = D.groupby('GID')[cov + ['Timestamp']].shift(lag)
        ok = ((D.Timestamp - L.Timestamp) == lag * 1000) & (L.Lane == D.Lane)
        xnow = D.PositionX.values.copy()
        for c in cov:
            D[c] = L[c]
        D['x_now'] = xnow
        D = D[ok.values].copy()
        D['Lane'] = D.Lane.astype(int)
        xsel = D.x_now
    else:
        xsel = D.PositionX
    D = D[(xsel >= x_lo) & (xsel < x_hi) & (D.drop5 == 0)].copy()
    D['y'] = D[f'y{H}']
    D['dirn'] = np.where(D.Lane == 2, 'in', 'out')
    D['vd10'] = (D.vdes_use - 80) / 10
    D['d10'] = D.delta / 10
    D['olgap'] = D.ol_PositionX - D.PositionX
    D['free'] = (D.olgap.isna() | (D.olgap > 300)).astype(int)
    return D.reset_index(drop=True)


def features(D, specs=('M0', 'M1', 'M1p'), vin=None, vout=None):
    """MOBIL terms per channel; optional (permuted) lane limits"""
    if vin is not None:
        D = D.copy(); D['vin'] = vin; D['vout'] = vout
        D['sf_use'] = M.sf_from(D, D.vin.values.astype(float), D.vout.values.astype(float))
    G = M.gaps(D)
    return {s: M.mobil_terms(D, G, s) for s in specs}


def design(S, F, bias=None, dayfe=False):
    X = [np.ones(len(S)), F.dA_ego.values, F.dA_n.values, F.dA_o.values, F.unsafe.values, S.vd10.values, S.entered_by_lc.values.astype(float)]
    cols = ['const'] + BASECOLS
    if bias == 'step':
        X.append(S.diff_day.values.astype(float)); cols.append('bias_step')
    elif bias == 'lin':
        X.append(S.d10.values.astype(float)); cols.append('bias_per10')
    return np.column_stack(X), cols


def subset(D, F, cls, dirn, days=None):
    m = ((D.cls == cls) & (D.dirn == dirn)).values
    if days is not None:
        m &= D.day.isin(days).values
    S = D[m].reset_index(drop=True)
    return S, {k: v[m].reset_index(drop=True) for k, v in F.items()}


def expected_triggers(b, X, S, cut=1.5):
    """per-step expected triggers; inside an event window the steps within `cut` s of the stamp count as already changed"""
    h = expit(np.asarray(X, float) @ b)
    atrisk = ~((S.y.values == 1) & (S.dt_next.values <= cut))
    return h * atrisk


def lodo_hazards(S, X, days=MAIN, all_days=None):
    """leave-one-day-out hazards for every step of S: for a day in `days`, fit on the other days of `days`; a day outside `days`
    gets the fit on all of `days`"""
    ev = event_ids(S); h = np.full(len(S), np.nan); fits = {}
    main = S.day.isin(days).values
    b_all, _ = ich_fit(X[main], ev[main])
    for d in S.day.unique():
        sel = (S.day == d).values
        if d in days:
            tr = main & ~sel
            b, _ = ich_fit(X[tr], ev[tr], b0=b_all)
        else:
            b = b_all
        fits[d] = b
        h[sel] = expit(X[sel] @ b)
    return h, fits, b_all


def boot_ci(num, den, gid, B=1000, seed=1):
    """vehicle-cluster bootstrap of sum(num)/sum(den)"""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'g': gid, 'n': num, 'd': den}).groupby('g')[['n', 'd']].sum()
    n = df.n.values; d = df.d.values; k = len(df)
    idx = rng.integers(0, k, (B, k))
    r = n[idx].sum(1) / np.maximum(d[idx].sum(1), 1e-12)
    return np.percentile(r, 2.5), np.percentile(r, 97.5)
