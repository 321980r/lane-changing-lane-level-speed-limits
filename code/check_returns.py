# -*- coding: utf-8 -*-
"""Return to the outer lane after an inward change of a car (stint in lane 1 from the change; event = return; censored at the end of
the track) under definitions V0-V6 (all changes; lateral-confirmed; return also lateral-confirmed; stint ends at any lane change; start at
1,050-2,000 m; leader-driven only; unconstrained only). Kaplan-Meier probability of a return within 300 / 600 / 2,000 m and returns per km,
per day, with exact tests over the 56 labellings and a slope on Delta over all 8! reassignments. The functions are also used by the other
check_*.py scripts.
Inputs: _cache/work/core_events.pkl, check_stint_lat_events.pkl, core_runs.pkl. Outputs: results/check_stint_return_day.csv, check_stint_return_tests.csv."""
import os, itertools, numpy as np, pandas as pd
from lc_common import *

E = pd.read_pickle(os.path.join(CACHE, 'core_events.pkl'))
L = pd.read_pickle(os.path.join(CACHE, 'check_stint_lat_events.pkl'))
R = pd.read_pickle(os.path.join(CACHE, 'core_runs.pkl'))
E = E.merge(L[['day', 'GlobalID', 't', 'maxdepth_m5_10']], on=['day', 'GlobalID', 't'], how='left')
E['lat_ok'] = (E.maxdepth_m5_10 >= 0.8).astype(int)
E = E.sort_values(['day', 'GlobalID', 't']).reset_index(drop=True)
g = E.groupby(['day', 'GlobalID'])
E['next_lat_ok'] = g.lat_ok.shift(-1)
DAYORDER = [n for n, _, _ in DAYS]


def km_at(dist, ev, H):
    """KM probability of the event by distance H (1 - S(H))."""
    d = np.asarray(dist, float); e = np.asarray(ev, int)
    o = np.argsort(d, kind='mergesort'); d = d[o]; e = e[o]
    S = 1.0; nrisk = len(d)
    for u in np.unique(d):
        if u > H:
            break
        m = d == u
        k = e[m].sum()
        if nrisk > 0:
            S *= 1 - k / nrisk
        nrisk -= m.sum()
    return 1 - S


def stints(variant):
    s = E[(E.direction == 'in') & (E.truck == 0)].copy()
    if variant != 'V0':
        s = s[s.lat_ok == 1]
    s['dist'] = s.stay_m.clip(lower=0.1)
    s['ev'] = ((s.next_dir == 'out')).astype(int)
    if variant == 'V2':
        s['ev'] = ((s.next_dir == 'out') & (s.next_lat_ok == 1)).astype(int)
    if variant == 'V3':
        Rr = R[['day', 'GlobalID', 't_start', 't_end', 'x_start', 'x_end', 'end_type']]
        s = s.merge(Rr, left_on=['day', 'GlobalID', 't'], right_on=['day', 'GlobalID', 't_start'], how='left')
        s['dist'] = (s.x_end - s.x).clip(lower=0.1)
        s['ev'] = (s.end_type == 'lc').astype(int)
    if variant == 'V4':
        s = s[(s.x >= 1050) & (s.x < 2000)]
    if variant == 'V5':
        s = s[s.mtype0.str.startswith('in_overtake')]
    if variant == 'V6':
        s = s[~s.mtype0.str.startswith('in_overtake')]
    return s


def slope_perm(vals8, deltas8):
    v = np.asarray(vals8, float); dl = np.asarray(deltas8, float)
    b = np.polyfit(dl, v, 1)[0]
    null = np.array([np.polyfit(dl[list(p)], v, 1)[0] for p in itertools.permutations(range(8))])
    return b, np.mean(np.abs(null) >= abs(b) - 1e-12)


if __name__ == '__main__':
    rows_d, rows_t = [], []
    for var in ('V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'):
        s = stints(var)
        for H in (300, 600, 2000):
            dv = []
            for n in DAYORDER:
                q = s[s.day == n]
                val = km_at(q.dist, q.ev, H) if len(q) else np.nan
                rate = q.ev.sum() / (q.dist.sum() / 1000) if len(q) else np.nan
                dv.append(val)
                rows_d.append(dict(variant=var, H=H, day=n, cfg=label(n), delta=delta(n), n_stints=len(q), n_ret=int(q.ev.sum()),
                                   n_ret_withinH=int(((q.ev == 1) & (q.dist <= H)).sum()), km=round(q.dist.sum() / 1000, 3), p_ret=val, rate_per_km=rate))
            dv = np.array(dv)
            pooled_same = km_at(s[s.day.isin(DAYORDER[:3])].dist, s[s.day.isin(DAYORDER[:3])].ev, H)
            pooled_diff = km_at(s[s.day.isin(DAYORDER[3:8])].dist, s[s.day.isin(DAYORDER[3:8])].ev, H)
            ok = ~np.isnan(dv[:8])
            obs, p2, p1, _ = perm_same_vs_diff(dv[:8]) if ok.all() else (np.nan,) * 4
            b, pb = slope_perm(dv[:8], [delta(n) for n in DAYORDER[:8]]) if (ok.all() and H == 600) else (np.nan, np.nan)
            ss = s[s.day.isin(DAYORDER[:3])]; sd = s[s.day.isin(DAYORDER[3:8])]
            rows_t.append(dict(variant=var, H=H, n_same=len(ss), ret_same=int(ss.ev.sum()), n_diff=len(sd), ret_diff=int(sd.ev.sum()),
                               pooled_same=pooled_same, pooled_diff=pooled_diff,
                               daymean_same=dv[:3].mean(), daymean_diff=dv[3:8].mean(), p2_8=p2, p1_8=p1,
                               slope_per10=b * 10 if pd.notna(b) else np.nan, p_slope=pb))
    D = pd.DataFrame(rows_d); T = pd.DataFrame(rows_t)
    D.to_csv(os.path.join(RES, 'check_stint_return_day.csv'), index=False); T.to_csv(os.path.join(RES, 'check_stint_return_tests.csv'), index=False)
    pd.set_option('display.width', 250)
    print(T.round(3).to_string(index=False))
    print(D[(D.variant == 'V1') & (D.H == 600)][['cfg', 'n_stints', 'n_ret', 'n_ret_withinH', 'km', 'p_ret', 'rate_per_km']].round(3).to_string(index=False))
