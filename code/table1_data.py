# -*- coding: utf-8 -*-
"""Study-design numbers: share of measured radar frames by 50 m bin (pooled over the eight days and by day), observed session bounds and
the per-day design table behind Table 1.
Inputs: session frames and events; results/model_reproduction.csv (model_reproduction.py).
Outputs: results/final/site_data.csv, site_coverage_by_day.csv, design_by_day.csv, Table1_days_values.csv."""
import os
from plot_style import *

B = np.arange(900, 2101, 50)


def local_hhmm(ms):
    h = C.hours(np.array([ms]))[0]
    tot = int(np.floor(h * 60 + 1e-6))          # floor to the minute
    return f'{tot // 60:02d}:{tot % 60:02d}', h


rows_cov, rows_day = [], []
for n in ORDER8:
    f = frames(n)
    f = f[(f.in_monitored_session == 1) & (f.in_disturbed_period == 0)]
    x = f.PositionX.astype(float)
    b = pd.cut(x, B, right=False, labels=B[:-1])
    meas = (f.frame_source <= 1).astype(int)
    g = pd.DataFrame({'bin': b, 'lane': f.Lane.values, 'meas': meas.values}).dropna(subset=['bin'])
    for (bb, ln), s in g.groupby(['bin', 'lane'], observed=True):
        rows_cov.append(dict(day=n, bin_start_m=int(bb), lane=int(ln), frames=len(s), meas_frames=int(s.meas.sum())))
    # distance driven by class (car / heavy / unknown), vehicle-km in the window
    d = f.Smoothed_VelocityX.astype(float) * 0.1
    km = d.groupby(f.truck).sum() / 1000.0
    t0, h0 = local_hhmm(f.Timestamp.min()); t1, h1 = local_hhmm(f.Timestamp.max())
    rows_day.append(dict(day=n, car_km=km.get(0, 0.0), heavy_km=km.get(1, 0.0), unknown_km=km.get(-1, 0.0),
                         obs_first=t0, obs_last=t1, obs_first_h=h0, obs_last_h=h1))

CV = pd.DataFrame(rows_cov)
M = CV[CV.day.isin(ORDER8)]
pool = M.groupby('bin_start_m')[['frames', 'meas_frames']].sum()
pool['meas_share_all'] = pool.meas_frames / pool.frames
for ln in (1, 2):
    s = M[M.lane == ln].groupby('bin_start_m')[['frames', 'meas_frames']].sum()
    pool[f'meas_share_lane{ln}'] = s.meas_frames / s.frames
    pool[f'frames_lane{ln}'] = s.frames
# day-to-day range of the pooled-lane share
byday = CV.groupby(['day', 'bin_start_m'])[['frames', 'meas_frames']].sum()
byday['share'] = byday.meas_frames / byday.frames
rng = byday.share.unstack('day')[ORDER8]
pool['share_min_day'] = rng.min(axis=1); pool['share_max_day'] = rng.max(axis=1)
pool = pool.reset_index()
pool.insert(1, 'bin_end_m', pool.bin_start_m + 50)
pool.to_csv(os.path.join(DATF, 'site_data.csv'), index=False, float_format='%.4f')
rng.reset_index().to_csv(os.path.join(DATF, 'site_coverage_by_day.csv'), index=False, float_format='%.4f')
print(pool[['bin_start_m', 'frames', 'meas_share_all', 'meas_share_lane1', 'meas_share_lane2', 'share_min_day', 'share_max_day']].round(3).to_string(index=False))
print('overall measured share, 8 days:', round(M.meas_frames.sum() / M.frames.sum(), 3))

# ------------------------------------------------------------------ per-day design table
D = pd.DataFrame(rows_day)
P0 = pd.read_csv(os.path.join(RES, 'model_reproduction.csv'))
T = DAY_META.merge(P0[['day', 'vehicles', 'truck_share', 'vkm', 'lc', 'lc_car', 'lc_heavy', 'inward_car', 'inward_heavy', 'inward_all']], on='day').merge(D, on='day')
# car inward share recomputed from the events (count form, for the table)
inn = []
for n in T.day:
    e = lc_events(n); e = e[e.from_lane.isin([1, 2]) & e.to_lane.isin([1, 2])]
    c = e[e.vehicle_class == 'car']
    inn.append(dict(day=n, car_in=int((c.to_lane == 1).sum()), car_out=int((c.to_lane == 2).sum())))
T = T.merge(pd.DataFrame(inn), on='day')
assert (T.car_in + T.car_out == T.lc_car).all()
T['date_iso'] = pd.to_datetime(T.day.str.slice(5, 15))
T['weekday'] = T.date_iso.dt.day_name()
# session bounds used in the analysis (logbook; 0803 truncated at the start of the disturbed period)
SESSION_LOG = {'scen_2022-08-03_90-70': '07:53-11:55'}
T['session_logbook'] = [SESSION_LOG.get(n, s) for n, s in zip(T.day, T.session)]
T['session_analysed'] = T.session
_h = lambda v: int(v[:2]) + int(v[3:5]) / 60
T['session_hours'] = [_h(s_.split('-')[1]) - _h(s_.split('-')[0]) for s_ in T.session_analysed]
T['flow_vph'] = T.vehicles / T.session_hours
NOTES = {
    'scen_2022-07-29_80-80': 'Rebuilt from the leader-augmented export; vehicles with nobody else in range are missing (isolated vehicles under-represented).',
    'scen_2022-08-03_90-70': 'Outer lane temporarily occupied later in the day; only the undisturbed part 07:53-10:30 is analysed.',
    'scen_2022-08-08_60-60': '',
    'scen_2022-08-11_100-100': 'Third gantry (downstream of the window) showed 60/60.',
}
T['notes'] = T.day.map(NOTES).fillna('')
T['group'] = np.where(T.delta == 0, 'same-limit', 'differentiated')
T['session_part'] = np.where(T.obs_last_h <= 12.05, 'morning', 'late morning-afternoon')
out = pd.DataFrame({
    'date': T.date_iso.dt.strftime('%Y-%m-%d'), 'weekday': T.weekday, 'group': T.group,
    'configuration_inner_outer_kmh': T.cfg, 'delta_kmh': T.delta,
    'session_logbook': T.session_logbook, 'session_analysed': T.session_analysed, 'session_part': T.session_part,
    'session_hours': T.session_hours.round(3), 'vehicles': T.vehicles, 'flow_vph': T.flow_vph.round(1), 'heavy_share': T.truck_share.round(3),
    'vehicle_km': T.vkm.round(1), 'car_km': T.car_km.round(1), 'heavy_km': T.heavy_km.round(1), 'unknown_class_km': T.unknown_km.round(1),
    'lc_analysis_grade': T.lc, 'lc_car': T.lc_car, 'lc_heavy': T.lc_heavy,
    'car_inward': T.car_in, 'car_outward': T.car_out, 'inward_share_cars': (T.car_in / T.lc_car).round(2),
    'notes': T.notes})
out['_o'] = pd.to_datetime(out.date); out = out.sort_values('_o').drop(columns='_o')
out.to_csv(os.path.join(DATF, 'Table1_days_values.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 40)
print(out.drop(columns='notes').to_string(index=False))
print(T[['day', 'session', 'obs_first', 'obs_last']].to_string(index=False))

# per-day design table (sessions, configuration, vehicles, lane changes)
F2 = T[['day', 'date_iso', 'cfg', 'delta', 'group', 'session_logbook', 'session_analysed', 'obs_first', 'obs_last', 'vehicles', 'truck_share', 'lc_car', 'lc_heavy']].copy()
F2['date'] = F2.date_iso.dt.strftime('%Y-%m-%d'); F2 = F2.drop(columns='date_iso').sort_values('date')
F2.to_csv(os.path.join(DATF, 'design_by_day.csv'), index=False)
