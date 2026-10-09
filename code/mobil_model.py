# -*- coding: utf-8 -*-
"""IDM accelerations and MOBIL incentive terms at the decision points (library).

IDM (Treiber et al. 2000): a_IDM = a [1 - (v/v0)^delta - (s*/s)^2],  s* = s0 + max(0, v T + v dv / (2 sqrt(a b))).
Free-road term for v > v0: bounded form of the improved IDM (Treiber & Kesting 2013): -b [1 - (v0/v)^(a delta / b)]
('bounded'); the plain IDM term a[1-(v/v0)^delta] is kept as a sensitivity ('plain'). Accelerations clipped to [-9, a].
Gap s = leader reference position - ego reference position - leader length (car 5 m, heavy 12 m); leaders farther than 400 m
or not visible count as absent.

MOBIL (Kesting et al. 2007): ego gain dA_ego = a~_c - a_c (target minus current lane), new follower dA_n = a~_n - a_n (<= 0),
old follower dA_o = a~_o - a_o (>= 0). Followers keep their lane, so only the interaction part of their IDM changes.
The ego's desired speed in each lane (v0_cur, v0_tgt) is where a lane-specific posted limit enters the model:
  M0  v0 = vdes (limits ignored)             M1  v0 = min(vdes, V_lane)  (hard cap)
  M1s v0 = min(vdes, V_lane + theta)          M1p v0 = sf * V_lane        (SUMO: speedFactor x lane speed limit)"""
import numpy as np, pandas as pd
from lc_common import *

BASE = {'a': 1.2, 'b': 2.1, 's0': 2.5, 'T': 1.5, 'delta': 4.0, 'tau': 5.0, 'pass': True}   # tau: anticipation (s); pass: anticipated completion of a pass
LEN = {1: 12.0, 0: 5.0, -1: 5.0}
LOOK = 400.0
AMIN = -9.0


def free_term(v, v0, P, bounded=True):
    v0 = np.maximum(v0, 1.0)
    fr = P['a'] * (1 - (v / v0) ** P['delta'])
    if bounded:
        over = v > v0
        fb = -P['b'] * (1 - (v0 / np.maximum(v, 0.1)) ** (P['a'] * P['delta'] / P['b']))
        fr = np.where(over, fb, fr)
    return fr


def int_term(v, s, vl, P):
    """interaction part; s NaN (no leader) -> 0"""
    ss = s0 = P['s0'] + np.maximum(0.0, v * P['T'] + v * (v - vl) / (2 * np.sqrt(P['a'] * P['b'])))
    out = -P['a'] * (ss / np.maximum(s, 0.5)) ** 2
    return np.where(np.isnan(s) | np.isnan(vl), 0.0, out)


def clipa(x, P):
    return np.clip(x, AMIN, P['a'])


def gaps(D):
    """bumper gaps (m) and speeds (m/s) needed for MOBIL, leaders beyond LOOK dropped"""
    x = D.PositionX.values.astype(float); v = D.v.values / 3.6
    L = lambda t: np.where(pd.isna(t), 5.0, np.where(t == 1, 12.0, 5.0))
    Lego = np.where(D.truck.values == 1, 12.0, 5.0)
    G = {}
    G['v'] = v
    s_ol = D.ol_PositionX.values - x - L(D.ol_truck.values); s_ol = np.where(D.ol_PositionX.values - x > LOOK, np.nan, s_ol)
    s_tl = D.tl_PositionX.values - x - L(D.tl_truck.values); s_tl = np.where(D.tl_PositionX.values - x > LOOK, np.nan, s_tl)
    G['s_ol'], G['v_ol'] = s_ol, D.ol_v.values / 3.6
    G['s_tl'], G['v_tl'] = s_tl, D.tl_v.values / 3.6
    G['L_tl'] = L(D.tl_truck.values); G['L_ego'] = Lego
    # new follower (target lane, behind ego)
    xn = D.tf_PositionX.values; vn = D.tf_v.values / 3.6
    near_n = (x - xn) <= LOOK
    G['v_n'] = np.where(near_n, vn, np.nan)
    G['s_n_after'] = np.where(near_n, x - xn - Lego, np.nan)                                   # new follower -> ego
    G['s_n_before'] = np.where(near_n & ~np.isnan(D.tl_PositionX.values), D.tl_PositionX.values - xn - L(D.tl_truck.values), np.nan)
    G['s_n_before'] = np.where(G['s_n_before'] > LOOK, np.nan, G['s_n_before'])
    # old follower (own lane, behind ego)
    xo = D.of_PositionX.values; vo = D.of_v.values / 3.6
    near_o = (x - xo) <= LOOK
    G['v_o'] = np.where(near_o, vo, np.nan)
    G['s_o_before'] = np.where(near_o, x - xo - Lego, np.nan)                                   # old follower -> ego
    G['s_o_after'] = np.where(near_o & ~np.isnan(D.ol_PositionX.values), D.ol_PositionX.values - xo - L(D.ol_truck.values), np.nan)
    G['s_o_after'] = np.where(G['s_o_after'] > LOOK, np.nan, G['s_o_after'])
    return G


def v0_lanes(D, spec, vdes, theta=0.0):
    """desired speeds (km/h) of the ego in its current and in the target lane"""
    Vc = np.where(D.Lane.values == 1, D.vin.values, D.vout.values).astype(float)
    Vt = np.where(D.Lane.values == 1, D.vout.values, D.vin.values).astype(float)
    if spec == 'M0':
        return vdes, vdes
    if spec == 'M1':
        return np.minimum(vdes, Vc), np.minimum(vdes, Vt)
    if spec == 'M1s':
        return np.minimum(vdes, Vc + theta), np.minimum(vdes, Vt + theta)
    if spec == 'M1p':
        sf = D.sf_use.values
        return sf * Vc, sf * Vt
    raise ValueError(spec)


def mobil_terms(D, G, spec, P=BASE, bounded=True, theta=0.0):
    vdes = D.vdes_use.values
    v0c, v0t = v0_lanes(D, spec, vdes, theta)
    v = G['v']
    tau = P.get('tau', 0.0)            # anticipation (s): leader gaps extrapolated tau s ahead at constant speeds
    s_ol = G['s_ol'] + (G['v_ol'] - v) * tau if tau else G['s_ol']
    s_tl = G['s_tl'] + (G['v_tl'] - v) * tau if tau else G['s_tl']
    v_tl = G['v_tl']; v_n = G['v_n']; s_n_after = G['s_n_after']; s_n_before = G['s_n_before']
    if tau and P.get('pass', False):
        # anticipated completion of a pass: a target-lane leader that the ego will have passed within tau s is treated as the
        # new follower (anticipated bumper gap), and the target lane ahead as free
        dx = G['s_tl'] + G['L_tl'] + (G['v_tl'] - v) * tau
        passed = ~np.isnan(dx) & (dx < 0)
        s_n_after = np.where(passed, np.maximum(-dx - G['L_ego'], 0.5), s_n_after); v_n = np.where(passed, G['v_tl'], v_n)
        s_n_before = np.where(passed, np.nan, s_n_before)
        s_tl = np.where(passed, np.nan, s_tl); v_tl = np.where(passed, np.nan, v_tl)
    a_c = clipa(free_term(v, v0c / 3.6, P, bounded) + int_term(v, s_ol, G['v_ol'], P), P)
    at_c = clipa(free_term(v, v0t / 3.6, P, bounded) + int_term(v, s_tl, v_tl, P), P)
    n_after = int_term(np.nan_to_num(v_n), s_n_after, v, P); n_before = int_term(np.nan_to_num(v_n), s_n_before, v_tl, P)
    o_before = int_term(np.nan_to_num(G['v_o']), G['s_o_before'], v, P); o_after = int_term(np.nan_to_num(G['v_o']), G['s_o_after'], G['v_ol'], P)
    has_n = ~np.isnan(v_n); has_o = ~np.isnan(G['v_o'])
    dA_n = np.where(has_n, np.clip(n_after, AMIN, 0) - np.clip(n_before, AMIN, 0), 0.0)
    dA_o = np.where(has_o, np.clip(o_after, AMIN, 0) - np.clip(o_before, AMIN, 0), 0.0)
    unsafe = has_n & (n_after < -4.0)
    return pd.DataFrame({'a_c': a_c, 'at_c': at_c, 'dA_ego': at_c - a_c, 'dA_n': dA_n, 'dA_o': dA_o, 'unsafe': unsafe.astype(int)}, index=D.index)


def sf_from(D, vin, vout):
    """SUMO-style speed factor: running mean over free-flow frames of v / V_lane (lanes 1 and 2), for given lane limits;
    fallback vdes_use / V_own_lane"""
    n = D.n1.values + D.n2.values
    sf = (D.sv1.values / vin + D.sv2.values / vout) / np.maximum(n, 1)
    Vown = np.where(D.Lane.values == 1, vin, vout)
    return np.where(n >= 30, sf, D.vdes_use.values / Vown)


def load_sample(x_lo=1000.0, x_hi=1800.0, H=5, vdes_src='pre'):
    """estimation sample: lanes 1/2, cars and heavies, x in [x_lo, x_hi), no non-analysis-grade change within 5 s"""
    D = pd.read_pickle(os.path.join(CACHE, 'decision_points.pkl'))
    D = D[(D.PositionX >= x_lo) & (D.PositionX < x_hi) & (D.drop5 == 0) & D.truck.isin([0, 1])].copy()
    D['y'] = D[f'y{H}']
    if vdes_src == 'pre':
        vd = D.vdes_pre.fillna(D.vdes_all).fillna(D.v)
    elif vdes_src == 'all':
        vd = D.vdes_all.fillna(D.v)
    elif vdes_src == 'L2':
        vd = D.vdes_pre_L2.fillna(D.vdes_pre).fillna(D.vdes_all).fillna(D.v)
    else:
        vd = D.v
    D['vdes_use'] = vd.values
    D['sf_use'] = sf_from(D, D.vin.values, D.vout.values)
    D['dirn'] = np.where(D.Lane == 2, 'in', 'out')
    D['cls'] = np.where(D.truck == 1, 'truck', 'car')
    D['GID'] = D.day.str[10:15] + '_' + D.GlobalID.astype(str)
    return D.reset_index(drop=True)
