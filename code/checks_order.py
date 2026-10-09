# -*- coding: utf-8 -*-
"""Difference between same-limit and differentiated days adjusted for the recording order (the differentiated days were recorded at
positions 1, 3, 4, 5, 6 and the same-limit days at 2, 7, 8): for the unconstrained rate, ln(unconstrained / leader-driven) and the
leader-driven rate, the OLS coefficient of the group indicator in y ~ 1 + diff + order, recomputed for all 56 labellings with the order
held fixed; exact two-sided p = share of labellings with |coefficient| >= observed.
Input: results/final/checks_order_check_days.csv (checks_day_level.py). Output: results/final/checks_order_adjusted.csv."""
from checks_lib import *

OD = pd.read_csv(os.path.join(DATF, 'checks_order_check_days.csv'))
rows = []
d = OD.copy()
order = d.rec_order.values.astype(float)
same_obs = (d.delta.values == 0)
k = len(d); n_same = int(same_obs.sum())
for col, nm in (('rate_free', 'unconstrained rate'), ('ln_contrast', 'ln(unconstrained / leader-driven)'), ('rate_leader', 'leader-driven rate')):
    y = d[col].values.astype(float)

    def coef(same_mask):
        X = np.column_stack([np.ones(k), (~same_mask).astype(float), order])
        return np.linalg.lstsq(X, y, rcond=None)[0][1]
    b_obs = coef(same_obs)
    null = []
    for S_ in combinations(range(k), n_same):
        m = np.zeros(k, bool); m[list(S_)] = True
        null.append(coef(m))
    null = np.array(null)
    X = np.column_stack([np.ones(k), (~same_obs).astype(float), order]); bo = np.linalg.lstsq(X, y, rcond=None)[0]
    rows.append(dict(series=nm, n_days=k, b_diff_unadjusted=y[~same_obs].mean() - y[same_obs].mean(), b_diff_order_adjusted=b_obs,
                     b_order_per_day_position=bo[2], p_two_exact=float(np.mean(np.abs(null) >= abs(b_obs) - 1e-12)),
                     p_one_exact=float(np.mean(null >= b_obs - 1e-12)), n_labellings=len(null),
                     diff_over_same_unadjusted=(y[~same_obs].mean() / y[same_obs].mean()) if col != 'ln_contrast' else np.exp(y[~same_obs].mean() - y[same_obs].mean()),
                     diff_over_same_order_adjusted_at_mean_order=((bo[0] + bo[1] + bo[2] * order.mean()) / (bo[0] + bo[2] * order.mean())) if col != 'ln_contrast' else np.exp(b_obs)))
R = pd.DataFrame(rows); p = out(R, 'checks_order_adjusted.csv')
pd.set_option('display.width', 250)
print(R.round(4).to_string(index=False)); print('saved', p)
