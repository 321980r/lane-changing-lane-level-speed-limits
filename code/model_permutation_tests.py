# -*- coding: utf-8 -*-
"""Permutation p-values of the lane-limit terms (cars): observed statistics against the draws of decisions_permutation.py (desired-speed proxy with
whole-track fallback, 2,000 draws) and of model_permutation.py / model_permutation_resume.py (causal proxy, 1,025 draws). p_one = share of draws
with a statistic at least as large as the observed one.
Inputs: results/permutation/dec_perm_draws_c0..c3.csv, model_perm_causal_c0.csv, model_perm_causal_c1.csv, model_perm_causal_r2.csv,
model_perm_causal_r3.csv. Output: results/final/model_perm_tests.csv."""
import glob
from plot_style import *

PERM = os.path.join(RES, 'permutation')
rows = []
S = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(PERM, 'dec_perm_draws_c*.csv')))], ignore_index=True)
for d in ('in', 'out'):
    S[f'car_{d}_dLL_M1p_M1'] = S[f'car_{d}_dLL_M1p'] - S[f'car_{d}_dLL_M1']
    S[f'car_{d}_b_M2L'] = S[f'car_{d}_bias_lin']
o = S[S.draw == 0].iloc[0]; N = S[S.draw > 0]
stats = [f'car_{d}_{s}' for d in ('in', 'out') for s in ('dLL_M1', 'dLL_M1p', 'dLL_M1p_M1', 'dLL_M2L', 'b_M2L')]
for c in stats:
    v = N[c].values
    rows.append(dict(proxy='S2 (whole-track fallback)', stat=c, observed=o[c], null_mean=v.mean(), null_q95=np.quantile(v, 0.95),
                     p_one_upper=np.mean(v >= o[c] - 1e-9), p_two_abs=np.mean(np.abs(v) >= abs(o[c]) - 1e-9), n_draws=len(v), n_unique=N.perm.nunique()))
c0 = pd.read_csv(os.path.join(PERM, 'model_perm_causal_c0.csv')); o = c0[c0.draw == 0].iloc[0]
N = pd.concat([pd.read_csv(os.path.join(PERM, 'model_perm_causal_c1.csv')).query('draw > 0')] +
              [pd.read_csv(os.path.join(PERM, f'model_perm_causal_r{c}.csv')) for c in (2, 3) if os.path.exists(os.path.join(PERM, f'model_perm_causal_r{c}.csv'))], ignore_index=True)
N = N[N.perm != '01234567']
for c in stats:
    v = N[c].values
    p1 = np.mean(v >= o[c] - 1e-9)
    rows.append(dict(proxy='causal (G5)', stat=c, observed=o[c], null_mean=v.mean(), null_q95=np.quantile(v, 0.95),
                     p_one_upper=p1, p_two_abs=np.mean(np.abs(v) >= abs(o[c]) - 1e-9), n_draws=len(v), n_unique=N.perm.nunique(),
                     mc_se_p=np.sqrt(p1 * (1 - p1) / len(v))))
R = pd.DataFrame(rows); R.to_csv(os.path.join(DATF, 'model_perm_tests.csv'), index=False)
pd.set_option('display.width', 200)
print(R.round(3).to_string(index=False))
