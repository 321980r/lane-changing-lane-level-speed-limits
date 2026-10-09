# -*- coding: utf-8 -*-
"""Simulation inputs.
(a) Desired-speed distributions from the free-driving speeds of the vehicles (core_vehicles.v_des) on the three same-limit days: speed
    factor relative to the posted limit (REL: sf = v_des / V, normal) and absolute desired speed with a cap (ABS: v_des = min(vmax,
    k V_lane), vmax normal; mean, sd and k fitted to the deciles).
(b) Replay table: every observed vehicle of the eight days with class, insertion time at x = 900 m, lane at entry, speed at first
    detection, own free-driving speed and speed factor.
Inputs: _cache/work/core_vehicles.pkl, decision_points.pkl. Outputs: results/final/model_desired_speed_fits.csv, model_demand_by_day.csv;
_cache/work/model_replay.pkl."""
from plot_style import *
from scipy.stats import norm

V = pd.read_pickle(os.path.join(CACHE, 'core_vehicles.pkl'))
V = V[V.truck.isin([0, 1, -1])].copy()
Mt = DAY_META.set_index('day')
V['vin'] = V.day.map(Mt.vin); V['vout'] = V.day.map(Mt.vout)
SAME = ORDER8[:3]
rows = []
Q = np.arange(0.1, 0.91, 0.1)
for tr, cls in ((0, 'car'), (1, 'heavy')):
    S = V[V.day.isin(SAME) & (V.truck == tr) & V.v_des.notna()]
    sf = S.v_des / S.vin
    rows.append(dict(cls=cls, variant='REL', param='speedFactor mean', value=sf.mean()))
    rows.append(dict(cls=cls, variant='REL', param='speedFactor sd', value=sf.std()))
    rows.append(dict(cls=cls, variant='REL', param='speedFactor q01', value=sf.quantile(0.01)))
    rows.append(dict(cls=cls, variant='REL', param='speedFactor q99', value=sf.quantile(0.99)))
    for d in SAME:
        s = S[S.day == d]
        rows.append(dict(cls=cls, variant='obs', param=f'mean sf {Mt.cfg[d]}', value=(s.v_des / s.vin).mean()))
        rows.append(dict(cls=cls, variant='obs', param=f'mean free speed {Mt.cfg[d]} (km/h)', value=s.v_des.mean()))
    # ABS fit by grid search on deciles
    obsq = {d: S[S.day == d].v_des.quantile(Q).values for d in SAME}
    best = None
    ks = np.arange(0.9, 2.001, 0.01)
    for mu in np.arange(55, 100.1, 0.5):
        for sg in np.arange(3, 20.1, 0.5):
            base = mu + sg * norm.ppf(Q)
            sse = sum(((np.minimum(base[None, :], ks[:, None] * Mt.vin[d]) - obsq[d][None, :]) ** 2).sum(1) for d in SAME)
            j = int(np.argmin(sse))
            if best is None or sse[j] < best[0]:
                best = (float(sse[j]), mu, sg, float(ks[j]))
    sse, mu, sg, k = best
    rows += [dict(cls=cls, variant='ABS', param='vmax mean (km/h)', value=mu), dict(cls=cls, variant='ABS', param='vmax sd (km/h)', value=sg),
             dict(cls=cls, variant='ABS', param='tolerance k (cap = k x lane limit)', value=k), dict(cls=cls, variant='ABS', param='decile rmse (km/h)', value=np.sqrt(sse / (3 * len(Q))))]
    # REL fit quality on deciles for comparison: sf ~ N(m, s) -> v = sf * V
    m, s_ = sf.mean(), sf.std()
    sse_rel = sum(((Mt.vin[d] * (m + s_ * norm.ppf(Q)) - obsq[d]) ** 2).sum() for d in SAME)
    rows.append(dict(cls=cls, variant='REL', param='decile rmse (km/h)', value=np.sqrt(sse_rel / (3 * len(Q)))))
    rows.append(dict(cls=cls, variant='obs', param='n vehicles (3 same-limit days)', value=len(S)))
F = pd.DataFrame(rows); F.to_csv(os.path.join(DATF, 'model_desired_speed_fits.csv'), index=False)
print(F.round(3).to_string(index=False))

# ---------------- replay table
D = pd.read_pickle(os.path.join(CACHE, 'decision_points.pkl'))[['day', 'GlobalID', 'Timestamp', 'PositionX', 'Lane', 'v', 'sv1', 'n1', 'sv2', 'n2']]
D = D.sort_values(['day', 'GlobalID', 'Timestamp'])
first = D.groupby(['day', 'GlobalID']).first()
last = D.groupby(['day', 'GlobalID']).last()
R = V.set_index(['day', 'GlobalID'])[['truck', 't0', 'x0', 'lane_entry', 'lane_exit', 'v_des', 'full_traverse', 'vin', 'vout']].join(
    first[['Timestamp', 'PositionX', 'Lane', 'v']].rename(columns={'Timestamp': 't_first', 'PositionX': 'x_first', 'Lane': 'lane_first', 'v': 'v_first'})).join(
    last[['sv1', 'n1', 'sv2', 'n2']])
R = R.reset_index()
nfree = R.n1.fillna(0) + R.n2.fillna(0)
R['sf_own'] = np.where(nfree >= 20, (R.sv1 / R.vin + R.sv2 / R.vout) / nfree.clip(lower=1), np.nan)
med_v = R.groupby(['day', 'truck']).v_des.transform('median')
R['vdes_own'] = R.v_des.fillna(med_v)
R['sf_own'] = R.sf_own.fillna(R.vdes_own / np.where(R.lane_entry == 1, R.vin, R.vout))
R['v_first'] = R.v_first.fillna(R.vdes_own)
R['lane_ins'] = R.lane_entry.fillna(R.lane_first).fillna(2).clip(1, 2).astype(int)
# insertion time at x = 900 m (first detection is at ~900 m for > 99 % of vehicles)
t_first = R.t_first.fillna(R.t0)
x_first = R.x_first.fillna(R.x0)
R['t_ins'] = (t_first - 1000.0 * (x_first - 900.0) / (R.v_first / 3.6).clip(lower=5)) / 1000.0
R['t_ins'] = R.t_ins - R.groupby('day').t_ins.transform('min')
R = R.sort_values(['day', 't_ins']).reset_index(drop=True)
R.to_pickle(os.path.join(CACHE, 'model_replay.pkl'))
dem = R.groupby('day').agg(n=('GlobalID', 'size'), n_car=('truck', lambda s: int((s == 0).sum())), n_heavy=('truck', lambda s: int((s == 1).sum())),
                           span_h=('t_ins', lambda s: (s.max() - s.min()) / 3600),
                           car_entry_inner=('lane_ins', lambda s: np.nan), )
for d, g in R.groupby('day'):
    dem.loc[d, 'car_entry_inner'] = (g[g.truck == 0].lane_ins == 1).mean()
    dem.loc[d, 'heavy_entry_inner'] = (g[g.truck == 1].lane_ins == 1).mean()
dem['flow_veh_h'] = dem.n / dem.span_h; dem['heavy_share'] = dem.n_heavy / dem.n
dem = dem.join(Mt[['cfg', 'delta']])
dem.to_csv(os.path.join(DATF, 'model_demand_by_day.csv'))
print(dem.round(3).to_string())
