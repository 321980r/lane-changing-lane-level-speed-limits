# -*- coding: utf-8 -*-
"""Versions of the car-inward model with a bias of unconstrained cars (no outer-lane vehicle within 300 m ahead): M0, M1p, M0 + F,
M0 + F + F x D (M2F), M0 + F + F x Delta/10, M0 + F + D, separate biases with and without a leader, and M1p or M1 with the bias.
Log-likelihood, AIC and leave-one-day-out log-likelihood on the eight days (causal desired-speed proxy).
Input: _cache/work/decision_points.pkl. Output: results/final/model_bias_checks.csv."""
import gc
from model_lib import *
D = steps('causal', 0); D = D[(D.cls == 'car') & (D.dirn == 'in')].reset_index(drop=True); gc.collect()
F = features(D)
S, Fs = subset(D, F, 'car', 'in', MAIN)
ev = event_ids(S)
fr = S.free.values.astype(float); dd = S.diff_day.values.astype(float); d10 = S.d10.values.astype(float)
rows = []
def add(name, X):
    b, ll = ich_fit(X, ev)
    lodo = 0.0
    for d in MAIN:
        a = (S.day != d).values
        bb, _ = ich_fit(X[a], ev[a], b0=b); lodo += ich_ll(bb, X[~a], ev[~a])
    V = ich_vcov(b, X, ev, S.GID.values); se = np.sqrt(np.clip(np.diag(V), 0, None))
    rows.append(dict(model=name, k=len(b), LL=ll, AIC=2 * len(b) - 2 * ll, LODO=lodo, b_last=b[-1], se_last=se[-1], b_ego=b[1]))
    print(rows[-1], flush=True)
X0, _ = design(S, Fs['M0']); Xp, _ = design(S, Fs['M1p']); X1, _ = design(S, Fs['M1'])
add('M0', X0); add('M1p', Xp)
add('M0+free', np.column_stack([X0, fr]))
add('M0+free+free:diff', np.column_stack([X0, fr, fr * dd]))
add('M0+free+free:d10', np.column_stack([X0, fr, fr * d10]))
add('M0+free+diff', np.column_stack([X0, fr, dd]))
add('M0+free+lead:diff+free:diff', np.column_stack([X0, fr, (1 - fr) * dd, fr * dd]))
add('M1p+free', np.column_stack([Xp, fr]))
add('M1p+free+free:diff', np.column_stack([Xp, fr, fr * dd]))
add('M1+free+free:diff', np.column_stack([X1, fr, fr * dd]))
pd.DataFrame(rows).to_csv(os.path.join(FIN, 'model_bias_checks.csv'), index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
