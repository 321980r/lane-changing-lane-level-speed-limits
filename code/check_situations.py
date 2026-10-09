# -*- coding: utf-8 -*-
"""Independent computation of the outer-lane situation of cars from the frames and events: for every car frame in lane 2 the nearest
lane-2 vehicle ahead at the same time, and the same for every car inward change 3 s and 5 s before it.
Inputs: session frames and events. Outputs: _cache/work/check_sit_state.pkl, check_sit_events.pkl."""
import os, sys, numpy as np, pandas as pd
from lc_common import *

OUTC = os.path.join(CACHE, 'check_sit_state.pkl'); OUTE = os.path.join(CACHE, 'check_sit_events.pkl')


def lane2_state(f):
    """nearest vehicle ahead in lane 2 for every lane-2 frame (all classes as leaders)."""
    a = f[f.Lane == 2][['GlobalID', 'Timestamp', 'PositionX', 'Smoothed_VelocityX', 'truck', 'frame_source']].sort_values(['Timestamp', 'PositionX']).copy()
    nxt_t = a.Timestamp.shift(-1); same = nxt_t.values == a.Timestamp.values
    a['gap'] = np.where(same, a.PositionX.shift(-1) - a.PositionX, np.nan)
    a['lead_v'] = np.where(same, a.Smoothed_VelocityX.shift(-1), np.nan)
    a['lead_truck'] = np.where(same, a.truck.shift(-1), np.nan)
    return a


def snap_gap(f_by_t, gid, tau, lane):
    s = f_by_t.get(tau)
    if s is None:
        return np.nan, np.nan, np.nan
    me = s[s.GlobalID == gid]
    if len(me) == 0:
        return np.nan, np.nan, np.nan
    x = float(me.PositionX.iloc[0]); v = float(me.Smoothed_VelocityX.iloc[0])
    o = s[(s.GlobalID != gid) & (s.Lane == lane) & (s.PositionX > x)]
    if len(o) == 0:
        return x, v, np.nan
    r = o.loc[o.PositionX.idxmin()]
    return x, v, (float(r.PositionX) - x, float(r.Smoothed_VelocityX))


if __name__ == '__main__':
    S, EV = [], []
    for n, vi, vo in DAYS:
        f = frames(n)
        f = f[(f.in_monitored_session == 1) & (f.in_disturbed_period == 0) & f.Lane.isin([1, 2])].copy()
        a = lane2_state(f)
        a['day'] = n
        S.append(a[a.truck == 0].drop(columns=['truck']))
        e = lc_events(n)
        e = e[(e.from_lane == 2) & (e.to_lane == 1)]
        byT = {t: s for t, s in f[['GlobalID', 'Timestamp', 'PositionX', 'Lane', 'Smoothed_VelocityX', 'frame_source']].groupby('Timestamp')}
        for r in e.itertuples():
            d = dict(day=n, GlobalID=r.GlobalID, t=int(r.timestamp_ms), x=r.position_x_m, cls=r.vehicle_class, resolved=int(pd.notna(r.transition_s)))
            for k in (3, 5):
                tau = int(r.timestamp_ms) - k * 1000
                s = byT.get(tau)
                me = s[s.GlobalID == r.GlobalID] if s is not None else []
                if s is None or len(me) == 0:
                    d[f'ok{k}'] = 0; continue
                x = float(me.PositionX.iloc[0]); v = float(me.Smoothed_VelocityX.iloc[0])
                d[f'ok{k}'] = 1; d[f'x{k}'] = x; d[f'v{k}'] = v; d[f'lane{k}'] = int(me.Lane.iloc[0]); d[f'meas{k}'] = int(me.frame_source.iloc[0] <= 1)
                o = s[(s.GlobalID != r.GlobalID) & (s.Lane == 2) & (s.PositionX > x)]
                if len(o):
                    rr = o.loc[o.PositionX.idxmin()]
                    d[f'gap{k}'] = float(rr.PositionX) - x; d[f'lv{k}'] = float(rr.Smoothed_VelocityX)
                else:
                    d[f'gap{k}'] = np.nan; d[f'lv{k}'] = np.nan
            EV.append(d)
        print(n, 'car lane2 frames', int((a.truck == 0).sum()), 'inward ev', len(e), flush=True)
    pd.concat(S, ignore_index=True).to_pickle(OUTC)
    pd.DataFrame(EV).to_pickle(OUTE)
    print('saved')
