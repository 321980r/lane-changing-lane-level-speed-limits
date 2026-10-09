# -*- coding: utf-8 -*-
"""Calibration of the recommended car models on days left out of the fit: M2F for inward changes and M0 for outward changes (hazards
h_M2F and h_M0 of _cache/work/model_lodo_steps.pkl). p5 = 1 - (1 - h)^5 is the predicted probability of a change within 5 s; observed = share of
decision points followed by an analysis-grade change within 5 s. Bins at the pooled percentiles 0, 20, 40, 60, 80, 90, 95, 99, 100 of p5
within each direction, for same-limit days, differentiated days and pooled, with vehicle-cluster bootstrap intervals (B = 2000);
summaries: O/E, logistic calibration slope, Spearman correlation over the deciles 1-9.
Outputs: results/final/model_calibration_data.csv, model_calibration_summary.csv."""
import statsmodels.api as sm
from scipy.stats import spearmanr
from plot_style import *

P = pd.read_pickle(os.path.join(CACHE, 'model_lodo_steps.pkl'))
P = P[P.cls == 'car'].copy()
rng = np.random.default_rng(8)
PCT = [0, 20, 40, 60, 80, 90, 95, 99, 100]
MOD = {'in': 'M2F', 'out': 'M0'}


def cell(s, B=2000):
    a = s.groupby('GID').agg(y=('y', 'sum'), n=('y', 'size'))
    idx = rng.integers(0, len(a), (B, len(a)))
    bs = a.y.values[idx].sum(1) / a.n.values[idx].sum(1)
    return dict(pred=s.p5.mean(), obs=s.y.mean(), obs_lo=np.percentile(bs, 2.5), obs_hi=np.percentile(bs, 97.5), n_steps=len(s), n_pos=int(s.y.sum()),
                n_events=int(s.loc[s.y == 1, 'ev'].nunique()), n_veh=s.GID.nunique(), pred_min=s.p5.min(), pred_max=s.p5.max())


rows, summ = [], []
for dirn, m in MOD.items():
    g = P[P.dirn == dirn].copy()
    g['p5'] = 1 - (1 - g[f'h_{m}']) ** 5
    g['grp'] = np.where(g.delta > 0, 'differentiated', 'same-limit')
    cuts = np.percentile(g.p5, PCT); cuts[0] = -1; cuts[-1] = 2
    g['bin'] = np.digitize(g.p5, cuts[1:-1])          # 0 .. 7
    for (b, grp), s in g.groupby(['bin', 'grp']):
        rows.append(dict(dirn=dirn, model=m, bin=int(b) + 1, pct_lo=PCT[b], pct_hi=PCT[b + 1], group=grp, **cell(s)))
    for b, s in g.groupby('bin'):
        rows.append(dict(dirn=dirn, model=m, bin=int(b) + 1, pct_lo=PCT[b], pct_hi=PCT[b + 1], group='pooled', **cell(s)))
    # descriptive summaries
    g['dec'] = pd.qcut(g.p5.rank(method='first'), 10, labels=False)
    dec = g.groupby('dec').agg(pred=('p5', 'mean'), obs=('y', 'mean'))
    rho = spearmanr(dec.pred.values[:9], dec.obs.values[:9]).correlation
    for grp, s in list(g.groupby('grp')) + [('pooled', g)]:
        lp = np.log(s.p5 / (1 - s.p5)).values
        fit = sm.GLM(s.y.values, sm.add_constant(lp), family=sm.families.Binomial()).fit()
        summ.append(dict(dirn=dirn, model=m, group=grp, n_days=s.day.nunique(), n_steps=len(s), n_veh=s.GID.nunique(), n_events=int(s.loc[s.ev >= 0, 'ev'].nunique()),
                         sum_y=int(s.y.sum()), sum_p5=s.p5.sum(), OE=s.y.sum() / s.p5.sum(), calib_slope=fit.params[1],
                         spearman_dec1_9_pooled=rho if grp == 'pooled' else np.nan,
                         lowest_decile_pred=dec.pred.iloc[0] if grp == 'pooled' else np.nan, lowest_decile_obs=dec.obs.iloc[0] if grp == 'pooled' else np.nan,
                         lowest_decile_events=int(g.loc[(g.dec == 0) & (g.y == 1), 'ev'].nunique()) if grp == 'pooled' else np.nan))
C = pd.DataFrame(rows)
C.to_csv(os.path.join(DATF, 'model_calibration_data.csv'), index=False)
S = pd.DataFrame(summ); S.to_csv(os.path.join(DATF, 'model_calibration_summary.csv'), index=False)
pd.set_option('display.width', 250)
print(S.round(3).to_string(index=False))
print(C.round(4).to_string(index=False))
