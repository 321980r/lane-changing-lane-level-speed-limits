# -*- coding: utf-8 -*-
"""Observed lane-change numbers per day, the reference for the simulation comparisons: car inward and outward rates at x = 1,000-1,800 m
with vehicle-bootstrap intervals and the leave-one-day-out model predictions, heavy-vehicle share of lane-1 km, change in the inner-lane
share of full-traverse cars from 1,000-1,100 m to 1,900-2,000 m, and the share of cars entering in lane 2 that move in before 1,800 m.
Inputs: results/final/model_lodo_day.csv; _cache/work/model_lodo_steps.pkl, core_vehicles.pkl, core_events.pkl; results/check_sit_lane_shift.csv.
Output: results/final/sim_observed_by_day.csv (simulation_summary.py adds intervals in sim_observed_by_day_ci.csv)."""
from plot_style import *

Mt = DAY_META.set_index('day')
# ---------------- observed
L = pd.read_csv(os.path.join(DATF, 'model_lodo_day.csv'))
P = pd.read_pickle(os.path.join(CACHE, 'model_lodo_steps.pkl'))
P['km'] = P.v / 3600.0
rng = np.random.default_rng(5)
obs = []
for d in ORDER8:
    r = dict(day=d, cfg=Mt.cfg[d], delta=int(Mt.delta[d]), label=Mt.label[d])
    for dirn, nm in (('in', 'car_rate_in'), ('out', 'car_rate_out')):
        x = L[(L.cls == 'car') & (L.dirn == dirn) & (L.state == 'all') & (L.day == d)].iloc[0]
        r[nm], r[nm + '_lo'], r[nm + '_hi'], r[nm.replace('rate', 'n')], r[nm.replace('rate', 'km')] = x.rate_obs, x.rate_obs_lo, x.rate_obs_hi, x.n_obs, x.km
        for m in ('M0', 'M1', 'M1p', 'M2L', 'M2F', 'M2Fl'):
            r[f'{nm}_mobil_{m}'] = x[f'rate_pred_{m}']
        xf = L[(L.cls == 'car') & (L.dirn == dirn) & (L.state == 'free') & (L.day == d)].iloc[0]
        r[nm + '_free'] = xf.rate_obs
    t = P[(P.cls == 'truck') & (P.day == d)]
    g = t.groupby(['GID', 'dirn']).km.sum().unstack(fill_value=0.0)
    for c in ('in', 'out'):
        if c not in g:
            g[c] = 0.0
    a, b = g['out'].values, (g['in'] + g['out']).values
    r['heavy_share_km_L1'] = a.sum() / b.sum()
    idx = rng.integers(0, len(g), (2000, len(g)))
    bs = a[idx].sum(1) / b[idx].sum(1)
    r['heavy_share_km_L1_lo'], r['heavy_share_km_L1_hi'] = np.percentile(bs, [2.5, 97.5])
    r['heavy_km'] = b.sum()
    # stochastic MOBIL (replay) heavy lane-1 km share is not defined (open loop); left out
    obs.append(r)
O = pd.DataFrame(obs).set_index('day')
S1 = pd.read_csv(os.path.join(RES, 'check_sit_lane_shift.csv')).set_index('day')
O['car_shift_pp'] = S1.gain_pp.reindex(O.index); O['car_entry_inner'] = S1.in_entry.reindex(O.index); O['n_full_cars'] = S1.n_cars.reindex(O.index)
V = pd.read_pickle(os.path.join(CACHE, 'core_vehicles.pkl')); E = pd.read_pickle(os.path.join(CACHE, 'core_events.pkl'))
for d in ORDER8:
    v = V[(V.day == d) & (V.truck == 0) & (V.lane_entry == 2)]
    e = E[(E.day == d) & (E.truck == 0) & (E.direction == 'in') & (E.x >= 1000) & (E.x < 1800)]
    O.loc[d, 'car_n_entry_L2'] = len(v); O.loc[d, 'car_p_in_by1800'] = v.GlobalID.isin(e.GlobalID).mean()
O.reset_index().to_csv(os.path.join(DATF, 'sim_observed_by_day.csv'), index=False)
pd.set_option('display.width', 250)
print(O[['cfg', 'car_rate_in', 'car_rate_in_lo', 'car_rate_in_hi', 'car_rate_out', 'heavy_share_km_L1', 'car_shift_pp', 'car_p_in_by1800', 'car_rate_in_mobil_M1p', 'car_rate_in_mobil_M2F']].round(3).to_string())
