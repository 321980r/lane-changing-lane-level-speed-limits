# -*- coding: utf-8 -*-
"""How each simulation produces the response to the configuration (eight days, mean over 5 seeds): share of car inward changes made by
unconstrained cars and car outward rate per car-km in lane 1, on same-limit and differentiated days, for the closed-loop runs, the default
LC2013 runs and the best LC2013 setting, and observed.
Inputs: results/final/sim_closedloop_runs.csv, sim_runs_default.csv, sim_runs_grid.csv, sim_grid_loss.csv,
model_rec_lodo_day.csv, sim_observed_by_day_ci.csv. Output: results/final/sim_mechanism_check.csv."""
from plot_style import *

MAIN = ORDER8
C = pd.read_csv(os.path.join(DATF, 'sim_closedloop_runs.csv')); C['series'] = 'MOBIL_' + C.variant
D = pd.read_csv(os.path.join(DATF, 'sim_runs_default.csv')); D['series'] = 'LC2013_' + D.variant + '_sg1_kr1'
GL = pd.read_csv(os.path.join(DATF, 'sim_grid_loss.csv')); b = GL.iloc[0]
G = pd.read_csv(os.path.join(DATF, 'sim_runs_grid.csv'))
G = G[(G.variant == b.variant) & (G.lcSpeedGain == b.lcSpeedGain) & (G.lcKeepRight == b.lcKeepRight)].copy(); G['series'] = b.series
A = pd.concat([C, D, G], ignore_index=True); A = A[A.day.isin(MAIN)]
g = A.groupby(['series', 'day'])[['car_n_in', 'car_n_in_free', 'car_n_out', 'car_km_L1']].mean().reset_index()
g['diff'] = g.day.map(DAY_META.set_index('day').delta) > 0
g['out_rate'] = g.car_n_out / g.car_km_L1
rows = []
for s, x in g.groupby('series'):
    r = dict(series=s)
    for lab, m in (('same', ~x['diff']), ('diff', x['diff'])):
        r[f'free_share_{lab}'] = x[m].car_n_in_free.sum() / x[m].car_n_in.sum()
        r[f'n_in_per_day_{lab}'] = x[m].car_n_in.mean()
        r[f'out_rate_{lab}'] = x[m].out_rate.mean()
    r['out_rate_diff_over_same'] = r['out_rate_diff'] / r['out_rate_same']
    rows.append(r)
L = pd.read_csv(os.path.join(DATF, 'model_rec_lodo_day.csv')); L = L[L.day.isin(MAIN)]
p = L.pivot(index='day', columns='state', values='n_obs'); p['diff'] = L.groupby('day').delta.first() > 0
O = pd.read_csv(os.path.join(DATF, 'sim_observed_by_day_ci.csv')).set_index('day').reindex(MAIN)
o = dict(series='observed')
for lab, m in (('same', ~p['diff']), ('diff', p['diff'])):
    o[f'free_share_{lab}'] = p[m].free.sum() / p[m]['all'].sum(); o[f'n_in_per_day_{lab}'] = p[m]['all'].mean()
od = (O.delta > 0).values
o['out_rate_same'] = O.car_rate_out.values[~od].mean(); o['out_rate_diff'] = O.car_rate_out.values[od].mean()
o['out_rate_diff_over_same'] = o['out_rate_diff'] / o['out_rate_same']
_, o['out_rate_p_two'], _ = perm8(O.car_rate_out.values)
R = pd.concat([pd.DataFrame([o]), pd.DataFrame(rows)], ignore_index=True)
R.to_csv(os.path.join(DATF, 'sim_mechanism_check.csv'), index=False)
pd.set_option('display.width', 250)
print(R.round(3).to_string(index=False))
