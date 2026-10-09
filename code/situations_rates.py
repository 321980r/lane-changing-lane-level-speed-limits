# -*- coding: utf-8 -*-
"""Lane changes by the situation 3 s before the change and the distance driven in each situation; day rates and day-level tests.
Outer lane: C leader-driven (headway < 4 s, or gap < 150 m and closing > 3 km/h), F unconstrained (no leader within 300 m, view
available), M leader within 300 m but not constraining, N view truncated. Inner lane: R pass in the last 30 s or lane entered by a change
in the last 60 s, Y follower < 100 m approaching > 5 km/h, B headway < 3 s and closing, K settled.
Rate of type T on a day = changes in situation T / vehicle-km driven in situation T. The functions classify_events, exposure and
day_table are also used by other scripts.
Inputs: _cache/work/core_events.pkl, sit_state_<day>.pkl. Outputs: results/sit_typology_counts.csv, sit_day_rates.csv, sit_tests_primary.csv;
_cache/work/sit_events_typed.pkl, sit_vehicle_state_km.pkl, sit_vehicle_events.pkl."""
import os, sys
from situations_lib import *

TYPES = {('in', 'C'): 'IN_overtake', ('in', 'F'): 'IN_free', ('in', 'M'): 'IN_mid', ('in', 'N'): 'IN_noview',
         ('out', 'R'): 'OUT_return', ('out', 'Y'): 'OUT_yield', ('out', 'B'): 'OUT_blocked', ('out', 'K'): 'OUT_keepright'}
STATE_OF = {v: k[1] for k, v in TYPES.items()}
LANE_OF = {v: (2 if k[0] == 'in' else 1) for k, v in TYPES.items()}


def classify_events(E, k=3, **kw):
    st = event_states(E, k=k, **kw)
    return np.array([TYPES.get((d, s), 'U') for d, s in zip(E.direction.values, st)])


def exposure(S, **kw):
    st = frame_states(S, **kw)
    return st


def day_table(E, S, typ, st_frames, k=3, xmin=None, hours=None, ev_mask=None):
    """counts and km per day x class x type."""
    Sx = S.assign(st=st_frames)
    Ex = E.assign(typ=typ)
    if ev_mask is not None:
        Ex = Ex[ev_mask]
    if xmin is not None:
        Sx = Sx[Sx.PositionX >= xmin]; Ex = Ex[Ex[f's{k}_PositionX'] >= xmin]
    if hours is not None:
        Sx = Sx[(Sx.hour >= hours[0]) & (Sx.hour < hours[1])]; Ex = Ex[(Ex[f's{k}_hour'] >= hours[0]) & (Ex[f's{k}_hour'] < hours[1])]
    rows = []
    kmt = Sx.groupby(['day', 'truck', 'Lane', 'st']).km.sum()
    cnt = Ex.groupby(['day', 'truck', 'typ']).size()
    for d in ORDER8:
        for tr in (0, 1):
            for T in TYPES.values():
                km = kmt.get((d, tr, LANE_OF[T], STATE_OF[T]), 0.0)
                n = int(cnt.get((d, tr, T), 0))
                rows.append(dict(day=d, cfg=CFG[d], delta=DEL[d], truck=tr, type=T, n=n, km=float(km), rate=n / km if km > 0 else np.nan))
    return pd.DataFrame(rows)


def tests_for(D, types=('IN_free', 'IN_overtake', 'IN_mid', 'OUT_return', 'OUT_keepright', 'OUT_blocked'), trucks=(0, 1), tag=''):
    out = []
    for tr in trucks:
        for T in types:
            d = D[(D.truck == tr) & (D.type == T)].set_index('day')
            vals = d.rate.to_dict()
            if any(not np.isfinite(vals[x]) for x in ORDER8):
                continue
            r = perm_tests(vals)
            m8 = d.loc[ORDER8]
            try:
                rr, lo, hi, disp = quasi_poisson_ratio(m8.n.values, m8.km.values, (m8.delta > 0).astype(int).values)
            except Exception:
                rr = lo = hi = disp = np.nan
            b, pb = perm_slope(vals, 'delta')
            r.update(spec=tag, truck=tr, type=T, n_same=int(m8.n.values[:3].sum()), km_same=m8.km.values[:3].sum(), n_diff=int(m8.n.values[3:].sum()),
                     km_diff=m8.km.values[3:].sum(), qp_ratio=rr, qp_lo=lo, qp_hi=hi, qp_disp=disp, slope_delta_per10=b * 10, p_slope_delta=pb,
                     min_diff_gt_max_same=int(min(vals[x] for x in ORDER8[3:]) > max(vals[x] for x in ORDER8[:3])))
            out.append(r)
    return pd.DataFrame(out)


if __name__ == '__main__':
    E = events(); S = state()
    typ = classify_events(E, k=3)
    E['typ'] = typ
    st = exposure(S)
    # --- counts per type (all 567 events), by class and group
    E['grp'] = np.where(E.delta > 0, 'diff', 'same')
    ct = E.groupby(['truck', 'direction', 'typ', 'grp']).size().unstack('grp').fillna(0).astype(int)
    ct.to_csv(os.path.join(RES, 'sit_typology_counts.csv'))
    print(ct)
    # --- cross-tab with core mtype0 (sanity)
    print(pd.crosstab(E.typ, E.mtype0).loc[:, lambda x: x.sum() > 0].to_string())
    D = day_table(E, S, typ, st)
    D.to_csv(os.path.join(RES, 'sit_day_rates.csv'), index=False)
    T = tests_for(D, tag='primary')
    T.to_csv(os.path.join(RES, 'sit_tests_primary.csv'), index=False)
    pd.set_option('display.width', 250)
    print(T[['truck', 'type', 'n_same', 'n_diff', 'same', 'diff', 'ratio', 'p2_8', 'p1_8', 'qp_ratio', 'qp_lo', 'qp_hi', 'slope_delta_per10', 'p_slope_delta', 'min_diff_gt_max_same']].round(3).to_string(index=False))
    print(D[(D.truck == 0) & D.type.isin(['IN_free', 'IN_overtake', 'OUT_return', 'OUT_keepright'])].pivot_table(index=['day', 'cfg'], columns='type', values='rate').round(3))
    # vehicle x state km and events (for bootstrap)
    Sx = S.assign(st=st)
    VK = Sx.groupby(['day', 'GlobalID', 'truck', 'Lane', 'st']).km.sum().reset_index()
    VE = E.groupby(['day', 'GlobalID', 'truck', 'typ']).size().rename('n').reset_index()
    VK.to_pickle(os.path.join(CACHE, 'sit_vehicle_state_km.pkl')); VE.to_pickle(os.path.join(CACHE, 'sit_vehicle_events.pkl'))
    E.to_pickle(os.path.join(CACHE, 'sit_events_typed.pkl'))
