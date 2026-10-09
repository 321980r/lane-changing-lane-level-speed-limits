# -*- coding: utf-8 -*-
"""Return to the outer lane after the last pass made from the inner lane: lane-1 runs (core_runs) in which the vehicle passed at least
one lane-2 vehicle; the clock starts at the last such pass; event = the run ends with a change to lane 2; censored at the end of the
track. Runs opened by an inward change must be lateral-confirmed. Variants: all runs, runs opened by an inward change, entry runs, pass at
1,050-1,900 m; any or lateral-confirmed return. Kaplan-Meier by class with day-level tests, and a heavy/car Poisson rate ratio with day
effects.
Inputs: _cache/work/core_runs.pkl, core_passes.pkl, core_vehicles.pkl. Outputs: results/check_stint_postpass_units.csv, check_stint_postpass_tests.csv."""
import os, numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from lc_common import *
from check_returns import km_at, slope_perm, DAYORDER, E as EV

R = pd.read_pickle(os.path.join(CACHE, 'core_runs.pkl'))
P = pd.read_pickle(os.path.join(CACHE, 'core_passes.pkl'))
V = pd.read_pickle(os.path.join(CACHE, 'core_vehicles.pkl'))
P = P[(P.lane_passer == 1) & (P.lane_passed == 2)]
R = R.sort_values(['day', 'GlobalID', 't_start']).reset_index(drop=True)
R['next_lane'] = R.groupby(['day', 'GlobalID']).lane.shift(-1)
R['next_t'] = R.groupby(['day', 'GlobalID']).t_start.shift(-1)
R1 = R[R.lane == 1].copy()
# lateral confirmation of lc-started runs and of the ending lc (analysis-grade events only have a lat check)
ev = EV[['day', 'GlobalID', 't', 'lat_ok', 'truck']]
R1 = R1.merge(ev.rename(columns={'t': 't_start', 'lat_ok': 'start_lat_ok', 'truck': 'truck_ev'}), on=['day', 'GlobalID', 't_start'], how='left')
R1 = R1.merge(ev[['day', 'GlobalID', 't', 'lat_ok']].rename(columns={'t': 'next_t', 'lat_ok': 'end_lat_ok'}), on=['day', 'GlobalID', 'next_t'], how='left')
R1 = R1.merge(V[['day', 'GlobalID', 'truck', 'v_des']], on=['day', 'GlobalID'], how='left')
units = []
for r in R1.itertuples():
    pp = P[(P.day == r.day) & (P.passer == r.GlobalID) & (P.t >= r.t_start) & (P.t <= r.t_end)]
    if len(pp) == 0:
        continue
    last = pp.loc[pp.t.idxmax()]
    units.append(dict(day=r.day, GlobalID=r.GlobalID, truck=r.truck, v_des=r.v_des, start_type=r.start_type, start_lat_ok=r.start_lat_ok,
                      end_type=r.end_type, next_lane=r.next_lane, end_lat_ok=r.end_lat_ok, n_pass=len(pp), x_pass=last.x, x_end=r.x_end,
                      passed_truck=last.truck_passed, dist=max(r.x_end - last.x, 0.1)))
U = pd.DataFrame(units)
U['ev_any'] = ((U.end_type == 'lc') & (U.next_lane == 2)).astype(int)
U['ev_strict'] = (U.ev_any == 1) & (U.end_lat_ok == 1)
U['ev_strict'] = U.ev_strict.astype(int)
U['lc_ok'] = ~((U.start_type == 'lc') & (U.start_lat_ok != 1))
U.to_csv(os.path.join(RES, 'check_stint_postpass_units.csv'), index=False)

rows = []
def summarise(name, u, evcol, H=500):
    out = dict(subset=name, event=evcol, H=H)
    for cls, t in (('car', 0), ('truck', 1)):
        q = u[u.truck == t]
        out[f'{cls}_n'] = len(q); out[f'{cls}_ret'] = int(q[evcol].sum()); out[f'{cls}_km'] = km_at(q.dist, q[evcol], H)
        dv = np.array([km_at(q[q.day == n].dist, q[q.day == n][evcol], H) if (q.day == n).any() else np.nan for n in DAYORDER])
        out[f'{cls}_same_pooled'] = km_at(q[q.day.isin(DAYORDER[:3])].dist, q[q.day.isin(DAYORDER[:3])][evcol], H)
        out[f'{cls}_diff_pooled'] = km_at(q[q.day.isin(DAYORDER[3:8])].dist, q[q.day.isin(DAYORDER[3:8])][evcol], H)
        out[f'{cls}_n_same'] = int(q.day.isin(DAYORDER[:3]).sum()); out[f'{cls}_n_diff'] = int(q.day.isin(DAYORDER[3:8]).sum())
        out[f'{cls}_ret_same'] = int(q[q.day.isin(DAYORDER[:3])][evcol].sum()); out[f'{cls}_ret_diff'] = int(q[q.day.isin(DAYORDER[3:8])][evcol].sum())
        if not np.isnan(dv[:8]).any():
            o, p2, p1, _ = perm_same_vs_diff(dv[:8]); out[f'{cls}_daymean_same'] = dv[:3].mean(); out[f'{cls}_daymean_diff'] = dv[3:8].mean(); out[f'{cls}_p2_8'] = p2
    rows.append(out)

base = U[U.lc_ok]
summarise('all_runs', base, 'ev_any'); summarise('all_runs', base, 'ev_strict')
summarise('lc_started', base[base.start_type == 'lc'], 'ev_strict'); summarise('entry_runs', base[base.start_type == 'entry'], 'ev_strict')
summarise('pass_x_1050_1900', base[(base.x_pass >= 1050) & (base.x_pass < 1900)], 'ev_strict')
summarise('all_runs_H300', base, 'ev_strict', 300)
summarise('passed_a_truck', base[base.passed_truck == 1], 'ev_strict')
T = pd.DataFrame(rows)

# Poisson RR truck vs car, day FE, offset, cluster by vehicle
b = base.copy(); b['lkm'] = np.log(b.dist / 1000); b = b[b.truck >= 0]
b['gid'] = b.day + '_' + b.GlobalID.astype(str); b['day'] = b['day'].astype('category')
for evcol in ('ev_any', 'ev_strict'):
    m = smf.glm(f'{evcol} ~ truck + day', data=b, family=sm.families.Poisson(), offset=b.lkm).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(b.gid)[0]})
    ci = np.exp(m.conf_int().loc['truck'])
    T.loc[len(T)] = {'subset': 'poisson_RR_truck_vs_car', 'event': evcol, 'H': np.nan, 'car_n': int((b.truck == 0).sum()), 'truck_n': int((b.truck == 1).sum()),
                     'car_km': np.exp(m.params['truck']), 'truck_km': f'{ci.iloc[0]:.2f}-{ci.iloc[1]:.2f}'}
T.to_csv(os.path.join(RES, 'check_stint_postpass_tests.csv'), index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40)
print(T.round(3).T.to_string())
