# -*- coding: utf-8 -*-
"""Simulation against observation (Section 5.5 and the simulation blocks of sim_parameters.csv).
Observed per day (sim_observed_by_day.csv), with bootstrap intervals for the change in the inner-lane share of cars (multinomial over cars)
and for the share of outer-lane cars that move in before 1,800 m (binomial). Simulated: mean and range over seeds of the LC2013 runs
(default, grid, class-specific) and of the closed-loop runs, per day. Loss of each single LC2013 setting over the eight days: mean squared
log ratio sim/obs of the car inward rate plus that of the car outward rate (plus the heavy lane-1 share for the class-specific settings).
Geometric-mean sim/obs ratios on same-limit and differentiated days.
Inputs: results/final/sim_observed_by_day.csv, sim_runs_default.csv, sim_runs_grid.csv, sim_runs_classlc2.csv,
sim_closedloop_runs.csv; results/check_sit_lane_shift.csv.
Outputs: results/final/sim_observed_by_day_ci.csv, sim_vs_observed.csv, sim_grid_loss.csv, sim_classlc_loss.csv,
sim_ratio_summary.csv."""
from plot_style import *

Mt = DAY_META.set_index('day')
MAIN = ORDER8
rng = np.random.default_rng(21)

# ---------------- observed
O = pd.read_csv(os.path.join(DATF, 'sim_observed_by_day.csv')).set_index('day').reindex(ORDER8)
S1 = pd.read_csv(os.path.join(RES, 'check_sit_lane_shift.csv')).set_index('day')
for d in ORDER8:
    n, a, b = int(S1.n_cars[d]), int(S1.n_21[d]), int(S1.n_12[d])
    draws = rng.multinomial(n, [a / n, b / n, 1 - (a + b) / n], size=4000)
    bs = 100 * (draws[:, 0] - draws[:, 1]) / n
    O.loc[d, 'car_shift_pp_net'] = 100 * (a - b) / n
    O.loc[d, 'car_shift_pp_lo'], O.loc[d, 'car_shift_pp_hi'] = np.percentile(bs, [2.5, 97.5])
    O.loc[d, 'car_shift_n_cars'] = n
    nn = int(O.car_n_entry_L2[d]); k = int(round(O.car_p_in_by1800[d] * nn))
    bp = rng.binomial(nn, k / nn, 4000) / nn
    O.loc[d, 'car_p_in_by1800_lo'], O.loc[d, 'car_p_in_by1800_hi'] = np.percentile(bp, [2.5, 97.5])
    O.loc[d, 'car_n_in_by1800'] = k
O.reset_index().to_csv(os.path.join(DATF, 'sim_observed_by_day_ci.csv'), index=False)

# ---------------- simulated
MET = ['car_rate_in', 'car_rate_out', 'heavy_share_km_L1', 'car_shift_pp', 'car_p_in_by1800', 'n_inserted', 'n_planned']
Dd = pd.read_csv(os.path.join(DATF, 'sim_runs_default.csv')); Dd['lcSpeedGain'] = 1.0; Dd['lcKeepRight'] = 1.0
Gr = pd.read_csv(os.path.join(DATF, 'sim_runs_grid.csv'))
A = pd.concat([Dd, Gr], ignore_index=True)
A = A.drop_duplicates(['day', 'variant', 'seed', 'lcSpeedGain', 'lcKeepRight'])
A['series'] = 'LC2013_' + A.variant + '_sg' + A.lcSpeedGain.map('{:g}'.format) + '_kr' + A.lcKeepRight.map('{:g}'.format)
parts = [A]
fc = os.path.join(DATF, 'sim_runs_classlc2.csv')
if os.path.exists(fc):
    C = pd.read_csv(fc)
    C['series'] = ('LC2013_' + C.variant + '_sg' + C.lcSpeedGain.map('{:g}'.format) + '_kr' + C.lcKeepRight.map('{:g}'.format) +
                   '_hvsg' + C.hvSpeedGain.map('{:g}'.format) + '_hvkr' + C.hvKeepRight.map('{:g}'.format))
    parts.append(C)
CL = pd.read_csv(os.path.join(DATF, 'sim_closedloop_runs.csv'))
CL['series'] = 'MOBIL_' + CL.variant
parts.append(CL)
ALL = pd.concat(parts, ignore_index=True)
cnt = ALL.groupby(['series', 'day']).size()
agg = ALL.groupby(['series', 'day'])[MET].agg(['mean', 'min', 'max'])
agg.columns = [f'{a}_{b}' for a, b in agg.columns]
agg = agg.reset_index()
agg['n_seeds'] = cnt.reindex(pd.MultiIndex.from_frame(agg[['series', 'day']])).values
for m in ('car_rate_in', 'car_rate_out', 'heavy_share_km_L1', 'car_p_in_by1800', 'car_shift_pp'):
    agg[f'{m}_obs'] = agg.day.map(O[m])
    if m != 'car_shift_pp':
        agg[f'{m}_ratio'] = agg[f'{m}_mean'] / agg[f'{m}_obs']
agg['cfg'] = agg.day.map(Mt.cfg); agg['delta'] = agg.day.map(Mt.delta)
agg.to_csv(os.path.join(DATF, 'sim_vs_observed.csv'), index=False)

# ---------------- grid loss (LC2013 single sets, 8 days)
eps = 1e-3
rows = []
for s, g in agg[agg.series.str.startswith('LC2013_') & ~agg.series.str.contains('_hv')].groupby('series'):
    g = g.set_index('day').reindex(ORDER8)
    li = np.log((g.car_rate_in_mean + eps) / g.car_rate_in_obs); lo = np.log((g.car_rate_out_mean + eps) / g.car_rate_out_obs)
    v, sg, kr = s.split('_')[1], float(s.split('_sg')[1].split('_')[0]), float(s.split('_kr')[1])
    rows.append(dict(series=s, variant=v, lcSpeedGain=sg, lcKeepRight=kr, loss=np.mean(li ** 2) + np.mean(lo ** 2), loss_in=np.mean(li ** 2), loss_out=np.mean(lo ** 2),
                     n_seeds_min=int(g.n_seeds.min())))
GL = pd.DataFrame(rows).sort_values('loss').reset_index(drop=True)
GL.to_csv(os.path.join(DATF, 'sim_grid_loss.csv'), index=False)
# class-specific sets: loss adds the heavy lane-1 km share
rows = []
for s, g in agg[agg.series.str.contains('_hv')].groupby('series'):
    g = g.set_index('day').reindex(ORDER8)
    li = np.log((g.car_rate_in_mean + eps) / g.car_rate_in_obs); lo = np.log((g.car_rate_out_mean + eps) / g.car_rate_out_obs)
    lh = np.log((g.heavy_share_km_L1_mean + eps) / g.heavy_share_km_L1_obs)
    rows.append(dict(series=s, hvSpeedGain=float(s.split('_hvsg')[1].split('_')[0]), hvKeepRight=float(s.split('_hvkr')[1]),
                     loss3=np.mean(li ** 2) + np.mean(lo ** 2) + np.mean(lh ** 2), loss_heavy=np.mean(lh ** 2), n_seeds_min=int(g.n_seeds.min())))
GLC = pd.DataFrame(rows).sort_values('loss3').reset_index(drop=True) if rows else pd.DataFrame()
GLC.to_csv(os.path.join(DATF, 'sim_classlc_loss.csv'), index=False)

# ---------------- ratio summary per series (8 days)
same = np.array([Mt.delta[d] == 0 for d in MAIN])
rows = []
for s, g in agg.groupby('series'):
    g = g.set_index('day').reindex(MAIN)
    r = dict(series=s, n_seeds_min=int(g.n_seeds.min()) if g.n_seeds.notna().all() else 0)
    for m in ('car_rate_in', 'car_rate_out', 'heavy_share_km_L1', 'car_p_in_by1800'):
        lr = np.log(np.maximum(g[f'{m}_mean'].values, 1e-6) / g[f'{m}_obs'].values)
        r[f'{m}_gm_same'] = np.exp(lr[same].mean()); r[f'{m}_gm_diff'] = np.exp(lr[~same].mean())
        r[f'{m}_day_min'] = np.exp(lr.min()); r[f'{m}_day_max'] = np.exp(lr.max())
        r[f'{m}_diff_day_min'] = np.exp(lr[~same].min()); r[f'{m}_diff_day_max'] = np.exp(lr[~same].max())
        r[f'{m}_sim_same'] = g[f'{m}_mean'].values[same].mean(); r[f'{m}_sim_diff'] = g[f'{m}_mean'].values[~same].mean()
    r['car_shift_sim_same'] = g.car_shift_pp_mean.values[same].mean(); r['car_shift_sim_diff'] = g.car_shift_pp_mean.values[~same].mean()
    r['car_shift_sim_min'] = g.car_shift_pp_mean.min(); r['car_shift_sim_max'] = g.car_shift_pp_mean.max()
    r['car_shift_mae_pp'] = np.mean(np.abs(g.car_shift_pp_mean.values - g.car_shift_pp_obs.values))
    r['inserted_share_min'] = (g.n_inserted_mean / g.n_planned_mean).min()
    rows.append(r)
R = pd.DataFrame(rows)
ob = O.reindex(MAIN)
ref = dict(series='observed', car_rate_in_sim_same=ob.car_rate_in.values[same].mean(), car_rate_in_sim_diff=ob.car_rate_in.values[~same].mean(),
           car_rate_out_sim_same=ob.car_rate_out.values[same].mean(), car_rate_out_sim_diff=ob.car_rate_out.values[~same].mean(),
           heavy_share_km_L1_sim_same=ob.heavy_share_km_L1.values[same].mean(), heavy_share_km_L1_sim_diff=ob.heavy_share_km_L1.values[~same].mean(),
           car_p_in_by1800_sim_same=ob.car_p_in_by1800.values[same].mean(), car_p_in_by1800_sim_diff=ob.car_p_in_by1800.values[~same].mean(),
           car_shift_sim_same=ob.car_shift_pp.values[same].mean(), car_shift_sim_diff=ob.car_shift_pp.values[~same].mean())
R = pd.concat([pd.DataFrame([ref]), R], ignore_index=True)
R.to_csv(os.path.join(DATF, 'sim_ratio_summary.csv'), index=False)

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(GL.head(12).round(3).to_string(index=False))
print(GLC.round(3).to_string(index=False))
key = ['observed', 'LC2013_REL_sg1_kr1', 'LC2013_ABS_sg1_kr1', GL.series.iloc[0], 'MOBIL_CL_M2F', 'MOBIL_CL_M1p'] + ([GLC.series.iloc[0]] if len(GLC) else [])
cols = ['series', 'car_rate_in_gm_same', 'car_rate_in_gm_diff', 'car_rate_in_diff_day_min', 'car_rate_in_diff_day_max', 'car_p_in_by1800_gm_same', 'car_p_in_by1800_gm_diff',
        'car_rate_out_gm_same', 'car_rate_out_gm_diff', 'heavy_share_km_L1_gm_same', 'heavy_share_km_L1_gm_diff', 'car_shift_sim_same', 'car_shift_sim_diff', 'car_shift_mae_pp',
        'inserted_share_min']
print(R[R.series.isin(key)][cols].round(3).to_string(index=False))
print(O[['cfg', 'car_shift_pp', 'car_shift_pp_net', 'car_shift_pp_lo', 'car_shift_pp_hi', 'car_p_in_by1800', 'car_p_in_by1800_lo', 'car_p_in_by1800_hi']].round(3).to_string())
