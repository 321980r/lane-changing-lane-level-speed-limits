# -*- coding: utf-8 -*-
"""Interval-censored discrete-time hazard (ICH) for late-stamped lane changes (library).

Per 1-s step j the lane-change hazard is h_j = logistic(X_j b). A recorded change at t_rec is known only to have been triggered
in the window W = {steps with 0 < t_rec - t_j <= L} (L = 5 s: the stamp is late by up to ~5 s in 90 % of cases). Likelihood:
  steps outside any window: log(1 - h_j);  each event: log(1 - prod_{j in W}(1 - h_j)).
This is the exact first-trigger likelihood when the trigger step is unknown within W (steps before W are 'no trigger').
The per-step hazard is directly usable as a stochastic MOBIL rule in a 1-s simulation step."""
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


def _sp(x):
    return np.logaddexp(0, x)


def ich_negll(b, X, ev, nev):
    """ev: event index per step (-1 = not in a window); returns (negLL, grad)"""
    eta = X @ b; h = expit(eta); sp = _sp(eta)
    out = ev < 0
    ll = -sp[out].sum(); g = -(X[out] * h[out, None]).sum(0)
    if nev:
        inn = ~out
        S = np.bincount(ev[inn], weights=sp[inn], minlength=nev)
        S = np.maximum(S, 1e-12)
        ll += np.sum(np.log(-np.expm1(-S)))
        wj = h[inn] / np.expm1(S)[ev[inn]]
        g += (X[inn] * wj[:, None]).sum(0)
    return -ll, -g


def ich_fit(X, ev, b0=None):
    X = np.asarray(X, float); ev = np.asarray(ev, int); nev = int(ev.max() + 1) if (ev >= 0).any() else 0
    if b0 is None:
        b0 = np.zeros(X.shape[1]); b0[0] = -5.0
    r = minimize(ich_negll, b0, args=(X, ev, nev), jac=True, method='L-BFGS-B', options={'maxiter': 2000, 'gtol': 1e-7})
    return r.x, -r.fun


def ich_ll(b, X, ev):
    X = np.asarray(X, float); ev = np.asarray(ev, int)
    # re-index events locally
    evl = ev.copy(); m = ev >= 0
    if m.any():
        u, inv = np.unique(ev[m], return_inverse=True); evl[m] = inv; nev = len(u)
    else:
        nev = 0
    return -ich_negll(b, X, evl, nev)[0]


def ich_scores(b, X, ev, groups):
    """per-cluster score vectors (for the sandwich)"""
    X = np.asarray(X, float); ev = np.asarray(ev, int)
    eta = X @ b; h = expit(eta); sp = _sp(eta)
    out = ev < 0
    sc = np.zeros_like(X)
    sc[out] = -X[out] * h[out, None]
    if (~out).any():
        nev = ev.max() + 1
        S = np.maximum(np.bincount(ev[~out], weights=sp[~out], minlength=nev), 1e-12)
        sc[~out] = X[~out] * (h[~out] / np.expm1(S)[ev[~out]])[:, None]
    return pd.DataFrame(sc).groupby(np.asarray(groups)).sum().values


def ich_se(b, X, ev, groups, eps=1e-5):
    X = np.asarray(X, float); ev = np.asarray(ev, int); nev = int(ev.max() + 1) if (ev >= 0).any() else 0
    k = len(b); Hm = np.zeros((k, k))
    for i in range(k):
        e = np.zeros(k); e[i] = eps
        g1 = ich_negll(b + e, X, ev, nev)[1]; g0 = ich_negll(b - e, X, ev, nev)[1]
        Hm[:, i] = (g1 - g0) / (2 * eps)
    Hm = (Hm + Hm.T) / 2
    Hi = np.linalg.pinv(Hm)
    S = ich_scores(b, X, ev, groups); G = S.shape[0]
    V = Hi @ (S.T @ S) @ Hi * G / (G - 1)
    return np.sqrt(np.clip(np.diag(V), 0, None)), V, np.sqrt(np.clip(np.diag(Hi), 0, None))


def event_index(S, H=5):
    """event id per step: steps flagged y{H}=1 grouped by vehicle and the recorded change time"""
    flag = S[f'y{H}'].values == 1
    key = np.where(flag, S.GID.astype(str).values + '|' + (S.Timestamp.values + np.round(S.dt_next.fillna(0).values * 1000)).astype(np.int64).astype(str), '')
    ev = np.full(len(S), -1, int)
    if flag.any():
        u, inv = np.unique(key[flag], return_inverse=True); ev[flag] = inv
    return ev


def expected_events(b, X, S, cut=1.5):
    """martingale check: expected number of changes = sum of step hazards over the time at risk; inside an event window the
    steps within `cut` s of the recorded stamp are treated as already changed (median stamp delay 1.5 s)"""
    h = expit(np.asarray(X, float) @ b)
    atrisk = ~((S.y5.values == 1) & (S.dt_next.values <= cut))
    return h * atrisk
