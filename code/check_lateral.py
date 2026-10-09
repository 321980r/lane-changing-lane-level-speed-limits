# -*- coding: utf-8 -*-
"""Lateral excursion of every analysis-grade lane change around its recorded time: whether the lateral position goes past the lane line
(3.75 m) by at least a threshold on the new-lane side in several windows; measured-frame share and duration of the new-lane stint; passes
made or received during it (an order flip with a vehicle in the old lane requires a lane change).
Inputs: session frames; _cache/work/core_events.pkl, core_passes.pkl, core_runs.pkl.
Outputs: _cache/work/check_stint_lat_events.pkl; results/check_stint_lat_events.csv, check_stint_lat_summary.csv."""
import os, numpy as np, pandas as pd
from lc_common import *

LINE = 3.75
E = pd.read_pickle(os.path.join(CACHE, 'core_events.pkl'))
P = pd.read_pickle(os.path.join(CACHE, 'core_passes.pkl'))
R = pd.read_pickle(os.path.join(CACHE, 'core_runs.pkl'))
cache = os.path.join(CACHE, 'check_stint_lat_events.pkl')

rows = []
for n, vi, vo in DAYS:
    f = frames(n)
    e = E[E.day == n]
    g = f[f.GlobalID.isin(e.GlobalID.unique())].groupby('GlobalID')
    ego_all = {k: d.sort_values('Timestamp') for k, d in g}
    Pn = P[P.day == n]; Rn = R[R.day == n]
    for r in e.itertuples():
        d = ego_all[r.GlobalID]
        ts = d.Timestamp.values; lat = d.lateral_position_m.values; ln = d.Lane.values; fs = d.frame_source.values
        new = r.to_lane; old = r.from_lane
        side = (lat < LINE) if new == 1 else (lat > LINE)          # on the new-lane side of the line
        depth = (LINE - lat) if new == 1 else (lat - LINE)          # how far past the line into the new lane (m)
        out = dict(day=n, GlobalID=r.GlobalID, t=r.t, direction=r.direction, truck=r.truck, x=r.x, resolved=r.resolved)
        for nm, a, b in (('a0_10', 0, 10), ('m5_10', -5, 10), ('m10_10', -10, 10), ('a0_5', 0, 5)):
            m = (ts >= r.t + a * 1000) & (ts <= r.t + b * 1000)
            out[f'maxdepth_{nm}'] = float(np.nanmax(depth[m])) if m.any() else np.nan
        # new-lane stint (filtered run starting at the change)
        run = Rn[(Rn.GlobalID == r.GlobalID) & (Rn.t_start == r.t)]
        if len(run):
            ru = run.iloc[0]
            ms = (ts >= ru.t_start) & (ts <= ru.t_end)
            out['stint_s'] = float(ru.dur_s); out['stint_end'] = ru.end_type
            out['stint_meas'] = float((fs[ms] <= 1).mean())
            out['stint_side_share'] = float(side[ms].mean())
            out['stint_maxdepth'] = float(np.nanmax(depth[ms]))
            # cross-lane passes during the stint: ego (in new lane) passes / is passed by a vehicle in the old lane
            pm = Pn[(Pn.passer == r.GlobalID) & (Pn.t >= ru.t_start) & (Pn.t <= ru.t_end) & (Pn.lane_passer == new) & (Pn.lane_passed == old)]
            pr = Pn[(Pn.passed == r.GlobalID) & (Pn.t >= ru.t_start) & (Pn.t <= ru.t_end) & (Pn.lane_passed == new) & (Pn.lane_passer == old)]
            out['stint_pass_made'] = len(pm); out['stint_passed_by'] = len(pr)
        # measured share before the change (5 s)
        mb = (ts >= r.t - 5000) & (ts < r.t)
        out['meas_before5'] = float((fs[mb] <= 1).mean()) if mb.any() else np.nan
        rows.append(out)
    print(n, len(e), flush=True)

L = pd.DataFrame(rows)
L.to_pickle(cache)
L.to_csv(os.path.join(RES, 'check_stint_lat_events.csv'), index=False)

# ---- summary: share of events that never go >= thr past the line, by window / threshold
S = []
for w in ('a0_10', 'm5_10', 'm10_10', 'a0_5'):
    for thr in (0.2, 0.5, 0.8, 1.2):
        bad = L[f'maxdepth_{w}'] < thr
        S.append(dict(window=w, thr=thr, n=len(L), n_pseudo=int(bad.sum()), share=bad.mean(),
                      car_in=f"{int(bad[(L.truck==0)&(L.direction=='in')].sum())}/{int(((L.truck==0)&(L.direction=='in')).sum())}",
                      truck_in=f"{int(bad[(L.truck==1)&(L.direction=='in')].sum())}/{int(((L.truck==1)&(L.direction=='in')).sum())}",
                      car_out=f"{int(bad[(L.truck==0)&(L.direction=='out')].sum())}/{int(((L.truck==0)&(L.direction=='out')).sum())}",
                      truck_out=f"{int(bad[(L.truck==1)&(L.direction=='out')].sum())}/{int(((L.truck==1)&(L.direction=='out')).sum())}"))
S = pd.DataFrame(S)
S.to_csv(os.path.join(RES, 'check_stint_lat_summary.csv'), index=False)
print(S[S.thr == 0.8].to_string(index=False))
