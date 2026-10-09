# -*- coding: utf-8 -*-
"""Day-level checks of lane use: (a) entry and exit lane of full-traverse cars and car-km share in lane 1; (b) cars already in lane 1
at entry: Kaplan-Meier probability of leaving lane 1 within 600 m and leaving rate per km; (c) returns per km after an inward change of a
car (definition V2 of check_returns.py), also leaving out one day at a time; (d) morning-only and morning/afternoon comparisons.
Inputs: results/check_stint_seq_runs_lat.csv; _cache/work/core_vehicles.pkl. Output: results/check_misc.csv."""
import os, itertools, numpy as np, pandas as pd
from lc_common import *
from check_returns import km_at, slope_perm, DAYORDER, stints

R = pd.read_csv(os.path.join(RES, 'check_stint_seq_runs_lat.csv'))
V = pd.read_pickle(os.path.join(CACHE, 'core_vehicles.pkl'))
out = []


def tests(name, v, extra=''):
    v = np.asarray(v, float)
    o, p2, p1, _ = perm_same_vs_diff(v[:8])
    out.append(dict(check=name, detail=extra, same_mean=v[:3].mean(), diff_mean=v[3:8].mean(), p2_8=p2, p1_8=p1,
                    day_values=';'.join(f'{x:.3f}' for x in v)))

# (a) entry lane composition
cars = V[(V.truck == 0) & (V.full_traverse == 1)]
tests('car_entry_lane1_share', [(cars[cars.day == n].lane_entry == 1).mean() for n in DAYORDER], 'full-traverse cars, modal lane 1000-1100 m')
tests('car_exit_lane1_share', [(cars[cars.day == n].lane_exit == 1).mean() for n in DAYORDER], 'full-traverse cars, modal lane 1900-2000 m')
allc = V[(V.truck == 0)]
tests('car_km_share_lane1', [allc[allc.day == n].dist_lane1_m.sum() / allc[allc.day == n].dist_m.sum() for n in DAYORDER], 'all cars')

# (b) cars in lane 1 at first detection (entry runs), leave = next filtered change to lane 2 that is not pseudo
R = R.sort_values(['day', 'GlobalID', 't_start']).reset_index(drop=True)
R['next_pseudo'] = R.groupby(['day', 'GlobalID']).pseudo.shift(-1)
R = R.merge(V[['day', 'GlobalID', 'truck']], on=['day', 'GlobalID'])
er = R[(R.lane == 1) & (R.start_type == 'entry') & (R.truck == 0)].copy()
er['dist'] = (er.x_end - er.x_start).clip(lower=0.1)
er['ev'] = ((er.end_type == 'lc') & (er.next_pseudo != 1)).astype(int)
tests('entry_run_leave_KM600', [km_at(er[er.day == n].dist, er[er.day == n].ev, 600) for n in DAYORDER], f'n={len(er)}, leaves={int(er.ev.sum())}')
tests('entry_run_leave_per_km', [er[er.day == n].ev.sum() / er[er.day == n].dist.sum() * 1000 for n in DAYORDER])

# (c) inward-LC return per stint-km (V2: both changes lateral-confirmed)
s = stints('V2')
rate = np.array([s[s.day == n].ev.sum() / s[s.day == n].dist.sum() * 1000 for n in DAYORDER])
tests('inLC_return_per_km_V2', rate, f'n={len(s)}, returns={int(s.ev.sum())}')
kmv = np.array([km_at(s[s.day == n].dist, s[s.day == n].ev, 600) for n in DAYORDER])
for i in range(8):
    keep = [j for j in range(8) if j != i]
    v = kmv[keep]; ns = sum(1 for j in keep if j < 3)
    obs = v[ns:].mean() - v[:ns].mean()
    null = [v[[k for k in range(7) if k not in c]].mean() - v[list(c)].mean() for c in itertools.combinations(range(7), ns)]
    out.append(dict(check='inLC_return_KM600_leave_one_out', detail='drop ' + label(DAYORDER[i]), same_mean=v[:ns].mean(), diff_mean=v[ns:].mean(),
                    p2_8=np.mean(np.abs(null) >= abs(obs) - 1e-12)))

# (d) morning sessions only: stints whose LC happened before 10:30 local
s['hour'] = C.hours(s.t)
sm = s[s.hour < 10.5]
kmm = np.array([km_at(sm[sm.day == n].dist, sm[sm.day == n].ev, 600) if (sm.day == n).sum() >= 3 else np.nan for n in DAYORDER])
out.append(dict(check='inLC_return_KM600_morning_only', detail='n per day=' + ','.join(str(int((sm.day == n).sum())) for n in DAYORDER),
                day_values=';'.join(f'{x:.3f}' for x in kmm), same_mean=np.nanmean(kmm[:3]), diff_mean=np.nanmean(kmm[3:8])))
# afternoon vs morning within the long sessions (0808, 0811, 0805)
for n in ('scen_2022-08-08_60-60', 'scen_2022-08-11_100-100', 'scen_2022-08-05_100-70'):
    q = s[s.day == n]
    out.append(dict(check='inLC_return_am_vs_pm', detail=label(n), same_mean=km_at(q[q.hour < 12].dist, q[q.hour < 12].ev, 600),
                    diff_mean=km_at(q[q.hour >= 12].dist, q[q.hour >= 12].ev, 600),
                    day_values=f'am n={int((q.hour < 12).sum())} ret={int(q[q.hour < 12].ev.sum())}; pm n={int((q.hour >= 12).sum())} ret={int(q[q.hour >= 12].ev.sum())}'))
O = pd.DataFrame(out); O.to_csv(os.path.join(RES, 'check_misc.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
print(O.round(3).to_string(index=False))
