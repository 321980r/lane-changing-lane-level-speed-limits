# -*- coding: utf-8 -*-
"""Calibration of the recommended car-inward model M2F on days left out of the fit (Section 5.4).
Bins of the predicted probability of a change within 5 s (p5 = 1 - (1 - h)^5, h = leave-one-day-out 1-s hazard; cut points at the pooled
percentiles 0, 20, 40, 60, 80, 90, 95, 99, 100): observed share of decision points followed by a change within 5 s, with vehicle-cluster
bootstrap intervals, for the three same-limit and the five differentiated days; observed / predicted ratios (O/E) for all eight days and
for each group, also as recorded changes / sum of the 1-s hazards. The inward rows of model_calibration_data.csv are first
reproduced as a check.
Inputs: _cache/work/model_lodo_steps.pkl; results/final/model_calibration_data.csv, model_calibration_summary.csv, model_lodo_day.csv.
Outputs: results/final/calibration_check_data.csv (binned observed and predicted), calibration_check_OE.csv (O/E)."""
import os, sys
import statsmodels.api as sm
from plot_style import *

NAME = 'calibration_check'

PCT = [0, 20, 40, 60, 80, 90, 95, 99, 100]
DIRN, MODEL = 'in', 'M2F'
P = pd.read_pickle(os.path.join(CACHE, 'model_lodo_steps.pkl'))
g = P[(P.cls == 'car') & (P.dirn == DIRN)].copy()
del P
g['p5'] = 1 - (1 - g[f'h_{MODEL}']) ** 5
g['grp'] = np.where(g.delta > 0, 'differentiated', 'same-limit')
cuts = np.percentile(g.p5, PCT); cuts[0] = -1; cuts[-1] = 2          # bin edges exactly as in model_calibration_data.py
g['bin'] = np.digitize(g.p5, cuts[1:-1])                             # 0 .. 7


def cell(s, rng, B=2000):
    a = s.groupby('GID').agg(y=('y', 'sum'), n=('y', 'size'))
    idx = rng.integers(0, len(a), (B, len(a)))
    bs = a.y.values[idx].sum(1) / a.n.values[idx].sum(1)
    return dict(pred=s.p5.mean(), obs=s.y.mean(), obs_lo=np.percentile(bs, 2.5), obs_hi=np.percentile(bs, 97.5), n_steps=len(s), n_pos=int(s.y.sum()),
                n_events=int(s.loc[s.y == 1, 'ev'].nunique()), n_veh=s.GID.nunique(), pred_min=s.p5.min(), pred_max=s.p5.max())


# ---- 1) check: the inward rows of model_calibration_data.csv are reproduced exactly (same data, bins and random stream)
rng = np.random.default_rng(8)
rows0 = []
for (b, grp), s in g.groupby(['bin', 'grp']):
    rows0.append(dict(dirn=DIRN, model=MODEL, bin=int(b) + 1, pct_lo=PCT[b], pct_hi=PCT[b + 1], group=grp, **cell(s, rng)))
for b, s in g.groupby('bin'):
    rows0.append(dict(dirn=DIRN, model=MODEL, bin=int(b) + 1, pct_lo=PCT[b], pct_hi=PCT[b + 1], group='pooled', **cell(s, rng)))
R0 = pd.DataFrame(rows0)
C = pd.read_csv(os.path.join(DATF, 'model_calibration_data.csv'))
Cin = C[C.dirn == DIRN].reset_index(drop=True)
assert list(R0.group) == list(Cin.group) and list(R0.bin) == list(Cin.bin)
num = ['pred', 'obs', 'obs_lo', 'obs_hi', 'n_steps', 'n_pos', 'n_events', 'n_veh', 'pred_min', 'pred_max']
assert np.allclose(R0[num].values.astype(float), Cin[num].values.astype(float), rtol=1e-12, atol=1e-15)
print('check passed: inward rows of model_calibration_data.csv reproduced exactly')

# ---- 2) same-limit and differentiated bins, same bins and bootstrap
g8 = g.copy()
cuts8 = np.percentile(g8.p5, PCT); cuts8[0] = -1; cuts8[-1] = 2
g8['bin'] = np.digitize(g8.p5, cuts8[1:-1])
assert g8.day.nunique() == 8
s3 = g8[g8.grp == 'same-limit']
assert sorted(s3.day.unique()) == sorted(ORDER8[:3])
rng3 = np.random.default_rng(8)
rows3 = [dict(dirn=DIRN, model=MODEL, bin=int(b) + 1, pct_lo=PCT[b], pct_hi=PCT[b + 1], group='same-limit', **cell(s, rng3))
         for b, s in s3.groupby('bin')]
S3 = pd.DataFrame(rows3)
assert len(S3) == 8 and (S3.obs > 0).all(), 'a same-limit bin without changes cannot be drawn on the log axes'
rng5 = np.random.default_rng(8)                                     # differentiated markers on the eight-day bins
D5 = pd.DataFrame([dict(dirn=DIRN, model=MODEL, bin=int(b) + 1, pct_lo=PCT[b], pct_hi=PCT[b + 1], group='differentiated', **cell(s, rng5))
                   for b, s in g8[g8.grp == 'differentiated'].groupby('bin')])
assert set(g8[g8.grp == 'differentiated'].day) == set(ORDER8[3:])
PL = pd.concat([D5, S3], ignore_index=True)
PL['days'] = np.where(PL.group == 'differentiated', '5 differentiated days', '3 same-limit days')
PL.to_csv(os.path.join(DATF, f'{NAME}_data.csv'), index=False)
pd.set_option('display.width', 250)

# ---- 3) observed / predicted (O/E = sum y / sum p5), as in model_calibration_summary.csv
REF = pd.read_csv(os.path.join(DATF, 'model_calibration_summary.csv'))
REF = REF[(REF.dirn == DIRN) & (REF.model == MODEL)].set_index('group')
oe = []
for days_lab, gg in (('8 days', g8),):
    for grp, s in list(gg.groupby('grp')) + [('pooled', gg)]:
        lp = np.log(s.p5 / (1 - s.p5)).values
        fit = sm.GLM(s.y.values, sm.add_constant(lp), family=sm.families.Binomial()).fit()
        oe.append(dict(days=days_lab, group=grp, n_days=s.day.nunique(), n_steps=len(s), n_veh=s.GID.nunique(),
                       n_events=int(s.loc[s.ev >= 0, 'ev'].nunique()), sum_y=int(s.y.sum()), sum_p5=s.p5.sum(), OE=s.y.sum() / s.p5.sum(),
                       calib_slope=fit.params[1]))
OE = pd.DataFrame(oe)
for grp in ('differentiated', 'same-limit', 'pooled'):          # the rows reproduce model_calibration_summary.csv
    r = OE[OE.group == grp].iloc[0]
    assert abs(r.OE - REF.OE[grp]) < 1e-12 and abs(r.calib_slope - REF.calib_slope[grp]) < 1e-9 and r.n_steps == REF.n_steps[grp], grp
# event-count version: recorded changes / sum of the LODO 1-s hazards over the decision points (model_lodo_day.csv, state 'all')
L = pd.read_csv(os.path.join(DATF, 'model_lodo_day.csv'))
L = L[(L.cls == 'car') & (L.dirn == DIRN) & (L.state == 'all')].set_index('day')
cnt = []
for days_lab, dl in (('8 days', ORDER8),):
    for grp, dd in (('differentiated', [d for d in dl if DAY_META.set_index('day').delta[d] > 0]),
                    ('same-limit', [d for d in dl if DAY_META.set_index('day').delta[d] == 0]), ('pooled', dl)):
        cnt.append(dict(days=days_lab, group=grp, n_days=len(dd), n_obs=L.n_obs[dd].sum(), n_pred=L.n_pred_M2F[dd].sum(),
                        obs_over_pred_counts=L.n_obs[dd].sum() / L.n_pred_M2F[dd].sum()))
CNT = pd.DataFrame(cnt)
OUT = OE.merge(CNT, on=['days', 'group', 'n_days'], how='left')
OUT.to_csv(os.path.join(DATF, 'calibration_check_OE.csv'), index=False)
print(OUT.round(4).to_string(index=False))
