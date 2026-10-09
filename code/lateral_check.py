# -*- coding: utf-8 -*-
"""Lateral confirmation of each analysis-grade lane change: lat_ok = 1 if the lateral position of the vehicle reaches at least 0.8 m
past the lane line (3.8 m) into the new lane (inward: < 3.0 m; outward: > 4.6 m) within its first 10 s in the new lane; also the extremes
in the old lane in the 5 s before and the share of measured frames in the new lane.
Inputs: session frames; _cache/work/core_events.pkl. Outputs: _cache/work/lateral_check.pkl, results/lateral_check.csv."""
import numpy as np, pandas as pd
from lc_common import *

E = pd.read_pickle(os.path.join(CACHE, 'core_events.pkl'))
rows = []
for n, vi, vo in DAYS:
    f = frames(n)
    e = E[E.day == n]
    byV = {g: d for g, d in f[f.GlobalID.isin(e.GlobalID.unique())].groupby('GlobalID')}
    for r in e.itertuples():
        d = byV[r.GlobalID]
        t_next = r.t + min(r.stay_s, 10.0) * 1000
        new = d[(d.Timestamp >= r.t) & (d.Timestamp < t_next) & (d.Lane == r.to_lane)]
        old = d[(d.Timestamp >= r.t - 5000) & (d.Timestamp < r.t) & (d.Lane == r.from_lane)]
        lat_new_ext = new.lateral_position_m.min() if r.to_lane == 1 else new.lateral_position_m.max()
        lat_old_ext = old.lateral_position_m.max() if r.to_lane == 1 else old.lateral_position_m.min()
        ok = (lat_new_ext < 3.0) if r.to_lane == 1 else (lat_new_ext > 4.6)
        ok_old = (lat_old_ext > 4.6) if r.to_lane == 1 else (lat_old_ext < 3.0)
        rows.append(dict(day=n, GlobalID=r.GlobalID, t=r.t, direction=r.direction, truck=r.truck, x=r.x, lat_new_ext=lat_new_ext, lat_old_ext=lat_old_ext,
                         meas_new=float((new.frame_source <= 1).mean()) if len(new) else np.nan, n_meas_new=int((new.frame_source <= 1).sum()),
                         lat_ok=int(bool(ok)), lat_ok_old=int(bool(ok_old))))
L = pd.DataFrame(rows)
L['lat_ok_both'] = L.lat_ok * L.lat_ok_old
L.to_pickle(os.path.join(CACHE, 'lateral_check.pkl')); L.to_csv(os.path.join(RES, 'lateral_check.csv'), index=False)
pd.set_option('display.width', 200)
print('events', len(L), 'lat_ok', L.lat_ok.mean().round(3), 'lat_ok_old', L.lat_ok_old.mean().round(3), 'both', L.lat_ok_both.mean().round(3))
print(L.groupby(['direction', 'truck']).lat_ok_both.agg(['size', 'mean']).round(3))
print(L.groupby(pd.cut(L.x, np.arange(900, 2101, 100)), observed=True).lat_ok_both.agg(['size', 'mean']).round(2).T)
print('lat_ok vs measured frames in new lane:'); print(L.groupby(L.n_meas_new > 0).lat_ok_both.agg(['size', 'mean']).round(3))
