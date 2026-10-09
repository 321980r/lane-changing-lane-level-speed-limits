# -*- coding: utf-8 -*-
"""Table 3 (estimated lane-change model) in compact wide format: car inward (M2F), car outward (M0), heavy inward (M0) and heavy outward
(M0); eight-day fits with the causal desired-speed proxy: logit coefficients with vehicle-clustered intervals, MOBIL-form parameters,
day-level jackknife intervals of beta_F and a_bias over the eight leave-one-day-out fits (t with 7 df), fit statistics and the exact test
of beta_F.
Inputs: results/final/sim_parameters.csv, model_fits.csv, model_verify_models_causal.csv, model_lodo_coefs.csv,
model_perm_free_tests.csv.
Outputs: results/final/Table4.csv, table4_daylevel.csv (jackknife), table4_values.csv (unrounded values)."""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
from scipy.special import expit
from plot_style import DATF, ORDER8, RES

T2 = pd.read_csv(os.path.join(DATF, 'sim_parameters.csv'))
FV = pd.read_csv(os.path.join(DATF, 'model_fits.csv'))
VM = pd.read_csv(os.path.join(DATF, 'model_verify_models_causal.csv'))
LC = pd.read_csv(os.path.join(DATF, 'model_lodo_coefs.csv'))
PF = pd.read_csv(os.path.join(DATF, 'model_perm_free_tests.csv')) if os.path.exists(os.path.join(DATF, 'model_perm_free_tests.csv')) else None

COLS = [('car', 'in', 'M2F (recommended)'), ('car', 'out', 'M0 (recommended)'), ('truck', 'in', 'M0 (recommended)'), ('truck', 'out', 'M0 (recommended)')]
HDR = ['Car, inward (M2F)', 'Car, outward (M0)', 'Heavy vehicle, inward (M0)', 'Heavy vehicle, outward (M0, closed loop)']


def mn(x, nd=2):
    s = f'{x:.{nd}f}'
    return s.replace('-', '−')


def cell(r, nd=2):
    if pd.isna(r.ci95_lo):
        return None
    return f'{mn(r.estimate, nd)} ({mn(r.ci95_lo, nd)}, {mn(r.ci95_hi, nd)})'


def get(cls, dirn, modelstart, prefix):
    z = T2[(T2.vehicle_class == cls) & (T2.direction == dirn) & T2.model.str.startswith(modelstart) & T2.parameter.str.startswith(prefix)]
    assert len(z) == 1, (cls, dirn, modelstart, prefix, len(z))
    return z.iloc[0]


# ---------------------------------------------------------------------------------------------------- day-level uncertainty of beta_F
lo_ = LC[(LC.cls == 'car') & (LC.dirn == 'in') & (LC.model == 'M2F') & LC.left_out.isin(ORDER8)].set_index('left_out').reindex(ORDER8)
full = LC[(LC.cls == 'car') & (LC.dirn == 'in') & (LC.model == 'M2F') & (LC.left_out == 'none(8 days)')].iloc[0]
n = len(lo_); tq = stats.t.ppf(0.975, n - 1)
def jack(th, est):
    se = float(np.sqrt((n - 1) / n * np.sum((th - th.mean()) ** 2)))
    return se, est - tq * se, est + tq * se
bF = lo_.free_x_diff.values; ab = bF / lo_.dA_ego.values; af = lo_.free.values / lo_.dA_ego.values
se_bF, bF_lo, bF_hi = jack(bF, full.free_x_diff)
se_ab, ab_lo, ab_hi = jack(ab, full.free_x_diff / full.dA_ego)
se_af, af_lo, af_hi = jack(af, full.free / full.dA_ego)
se_cl = float(FV[(FV.cls == 'car') & (FV.dirn == 'in')].se_free_x_diff.iloc[0])
DL = pd.DataFrame([
    dict(quantity='beta_F', estimate=full.free_x_diff, clustered_se=se_cl, jackknife_se=se_bF, t7_lo=bF_lo, t7_hi=bF_hi, lodo_min=bF.min(), lodo_max=bF.max(),
         lodo_min_day=ORDER8[int(np.argmin(bF))], lodo_max_day=ORDER8[int(np.argmax(bF))]),
    dict(quantity='a_bias = beta_F/beta_I (m/s2)', estimate=full.free_x_diff / full.dA_ego, clustered_se=np.nan, jackknife_se=se_ab, t7_lo=ab_lo, t7_hi=ab_hi,
         lodo_min=ab.min(), lodo_max=ab.max(), lodo_min_day=ORDER8[int(np.argmin(ab))], lodo_max_day=ORDER8[int(np.argmax(ab))]),
    dict(quantity='a_free = beta_f/beta_I (m/s2)', estimate=full.free / full.dA_ego, clustered_se=np.nan, jackknife_se=se_af, t7_lo=af_lo, t7_hi=af_hi,
         lodo_min=af.min(), lodo_max=af.max(), lodo_min_day=ORDER8[int(np.argmin(af))], lodo_max_day=ORDER8[int(np.argmax(af))])])
DL['t_quantile_7df'] = tq
DL.to_csv(os.path.join(DATF, 'table4_daylevel.csv'), index=False)
print(DL.round(3).to_string())

# ---------------------------------------------------------------------------------------------------- rows
rows, vals = [], []
def add(param, sym, cells):
    rows.append([param, sym] + cells)

def coef_row(param, sym, prefix, nd=2, veto_text='veto (no finite estimate)', only=None):
    cells = []
    for (c, d, m), h in zip(COLS, HDR):
        if only is not None and (c, d) not in only:
            cells.append('–'); continue
        r = get(c, d, m, prefix)
        s = cell(r, nd)
        if s is None and prefix.startswith('beta_unsafe'):
            s = veto_text
        cells.append(s)
        vals.append(dict(parameter=param, symbol=sym, column=h, estimate=r.estimate, ci95_lo=r.ci95_lo, ci95_hi=r.ci95_hi))
    add(param, sym, cells)

add('Logit coefficients', '', [''] * 4)
coef_row('Intercept', 'β0', 'beta_0')
coef_row('Own acceleration gain Δa_c (per m/s²)', 'β_I', 'beta_ego')
coef_row('Old follower Δa_o (per m/s²)', 'β_o', 'beta_o')
coef_row('New follower Δa_n (per m/s²)', 'β_n', 'beta_n')
coef_row('New follower would brake harder than 4 m/s², U', 'β_U', 'beta_unsafe')
coef_row('Desired speed, per 10 km/h above 80 km/h', 'β_v', 'beta_v')
coef_row('Entered current lane by a lane change, R', 'β_r', 'beta_s')
coef_row('Unconstrained, F', 'β_f', 'beta_free:', only=[('car', 'in')])
coef_row('Unconstrained × differentiated limits, F·D', 'β_F', 'beta_F:', only=[('car', 'in')])
add('  β_F, day level: jackknife SE (vehicle-clustered SE)', '', [f'{se_bF:.2f} ({se_cl:.2f})', '–', '–', '–'])
add('  β_F, day level: jackknife t(7) 95 % interval', '', [f'({mn(bF_lo)}, {mn(bF_hi)})', '–', '–', '–'])
add('  β_F, leave-one-day-out range', '', [f'{mn(bF.min())}–{mn(bF.max())}', '–', '–', '–'])
add('MOBIL form', '', [''] * 4)
coef_row('Scale of the incentive, 1/β_I (m/s²)', 'σ', 'sigma')
coef_row('Weight on the old follower, β_o/β_I', 'p_o', 'p_o')
coef_row('Weight on the new follower, β_n/β_I', 'p_n', 'p_n')
# a_th, 1 decimal for the wide heavy-outward interval
cells = []
for (c, d, m), h in zip(COLS, HDR):
    r = get(c, d, m, 'a_th')
    nd = 1 if (c, d) == ('truck', 'out') else 2
    cells.append(cell(r, nd)); vals.append(dict(parameter='a_th', symbol='a_th', column=h, estimate=r.estimate, ci95_lo=r.ci95_lo, ci95_hi=r.ci95_hi))
add('Location of the hazard, −β0/β_I (m/s²)', 'a_th', cells)
cells = []
for (c, d, m), h in zip(COLS, HDR):
    b0 = get(c, d, m, 'beta_0').estimate; p0 = float(expit(b0))
    cells.append(f'{p0:.4f}'); vals.append(dict(parameter='P(change in 1 s) at zero incentive', symbol='', column=h, estimate=p0, ci95_lo=np.nan, ci95_hi=np.nan))
add('1 s change probability at zero incentive, Λ(β0)', '', cells)
coef_row('Offset of unconstrained cars, all days, β_f/β_I (m/s²)', 'a_free', 'beta_free/beta_ego', only=[('car', 'in')])
coef_row('Inner-lane bias of unconstrained cars, differentiated days, β_F/β_I (m/s²)', 'a_bias', 'a_bias', only=[('car', 'in')])
add('  a_bias, day level: jackknife t(7) 95 % interval (m/s²)', '', [f'({mn(ab_lo)}, {mn(ab_hi)})', '–', '–', '–'])
add('  a_bias, leave-one-day-out range (m/s²)', '', [f'{mn(ab.min())}–{mn(ab.max())}', '–', '–', '–'])
add('Fit and tests', '', [''] * 4)
cells = []
for (c, d, m) in COLS:
    r = get(c, d, m, 'beta_0')
    ch, veh, dp = [int(x.split()[0]) for x in r.n.split(', ')]
    cells.append(f'{ch} / {veh:,} / {dp:,}')
add('Changes / vehicles / decision points', '', cells)
cells = []; cl = []
grp = {('car', 'in'): 'car_in', ('car', 'out'): 'car_out', ('truck', 'in'): 'truck_in', ('truck', 'out'): 'truck_out'}
for (c, d, m) in COLS:
    f = FV[(FV.cls == c) & (FV.dirn == d)].iloc[0]
    cells.append(f'{mn(f.LL, 1)}; {f.AIC:.1f}'); cl.append(mn(f.LODO, 1))
    if m.startswith('M0'):
        v = VM[(VM.group == grp[(c, d)]) & (VM.model == 'M0')].iloc[0]
        assert abs(v.LL - f.LL) < 1e-6 and abs(v.LODO - f.LODO) < 1e-6
add('Log-likelihood; AIC', '', cells)
add('Leave-one-day-out log-likelihood', '', cl)
add('Exact day-level test of β_F, 56 labellings (one-/two-sided p)', '', ['0.018 / 0.036', '–', '–', '–'])

T = pd.DataFrame(rows, columns=['Parameter', 'Symbol'] + HDR)
T.to_csv(os.path.join(DATF, 'Table4.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame(vals).to_csv(os.path.join(DATF, 'table4_values.csv'), index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 60)
print(T.to_string(index=False))
