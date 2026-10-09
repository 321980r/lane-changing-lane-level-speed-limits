# -*- coding: utf-8 -*-
"""Version M1pF of the car-inward model: M1p (lane desired speed = speed factor x lane limit) plus the bias of unconstrained cars on
differentiated days (F + F x D). Leave-one-day-out hazards at every car-inward decision point (added to those of model_lodo.py), day
rates observed and predicted by all versions in the unconstrained and leader situations with differentiated/same ratios, and the exact
56-labelling test of the bias (M1p terms with the true limits).
Inputs: _cache/work/decision_points.pkl, model_lodo_steps.pkl. Outputs: _cache/work/model_lodo_steps_rec.pkl; results/final/model_rec_lodo_day.csv,
model_rec_ratios.csv, model_rec_exact.csv, model_rec_summary.csv."""
import gc, time
from itertools import combinations
from model_lib import *

t0 = time.time()
D = steps('causal', 0); D = D[(D.cls == 'car') & (D.dirn == 'in')].reset_index(drop=True); gc.collect()
F = features(D, specs=('M1p',))
S, Fs = subset(D, F, 'car', 'in', None)
ev = event_ids(S)
Xp, cols = design(S, Fs['M1p'])
fr = S.free.values.astype(float); dd = S.diff_day.values.astype(float)
X = np.column_stack([Xp, fr, fr * dd]); cols = cols + ['free', 'free_x_diff']
h, fits, b_all = lodo_hazards(S, X)
main = S.day.isin(MAIN).values
V = ich_vcov(b_all, X[main], ev[main], S.GID.values[main]); se = np.sqrt(np.clip(np.diag(V), 0, None))
print('fit', dict(zip(cols, np.round(b_all, 3))), 'se', np.round(se, 3), round(time.time() - t0, 1), 's', flush=True)

P = pd.read_pickle(os.path.join(CACHE, 'model_lodo_steps.pkl'))
P = P[(P.cls == 'car') & (P.dirn == 'in')].reset_index(drop=True)
assert len(P) == len(S) and (P.GID.values == S.GID.values).all() and (P.Timestamp.values == S.Timestamp.values).all()
P['h_M1pF'] = h; P['hin_M1pF'] = expit(X @ b_all)
P.to_pickle(os.path.join(CACHE, 'model_lodo_steps_rec.pkl'))

# LODO log-likelihood over the 8 days (each day scored with the fit that left it out)
lodo = 0.0
for d in MAIN:
    s = (S.day == d).values
    lodo += ich_ll(fits[d], X[s], ev[s])
_, ll = ich_fit(X[main], ev[main], b0=b_all)

# day-level rates, event state at the step closest to 3 s before the stamp (as model_lodo.py)
P['km'] = P.v / 3600.0
w = P[P.ev >= 0].copy(); w['d3'] = (w.dt_next - 3.0).abs()
es = w.sort_values('d3').groupby('ev').first()[['day', 'free']].reset_index()
rows = []
for day, g in P.groupby('day'):
    for st, gs, e in (('all', g, es[es.day == day]), ('free', g[g.free == 1], es[(es.day == day) & (es.free == 1)]),
                      ('leader', g[g.free == 0], es[(es.day == day) & (es.free == 0)])):
        r = dict(day=day, cfg=DAY_META.set_index('day').cfg[day], delta=int(DAY_META.set_index('day').delta[day]), state=st, km=gs.km.sum(), n_obs=len(e))
        r['rate_obs'] = r['n_obs'] / r['km']
        for m in ('M0', 'M1', 'M1p', 'M2F', 'M1pF'):
            r[f'rate_pred_{m}'] = gs[f'h_{m}'].sum() / r['km']
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv(os.path.join(FIN, 'model_rec_lodo_day.csv'), index=False)
summ = []
for st in ('free', 'leader', 'all'):
    g = R[R.state == st].set_index('day').reindex(MAIN)
    s3 = g.delta.values == 0
    for c in ['rate_obs'] + [f'rate_pred_{m}' for m in ('M0', 'M1', 'M1p', 'M2F', 'M1pF')]:
        v = g[c].values
        o, p2, p1 = perm8(v)
        summ.append(dict(state=st, series=c, same=v[s3].mean(), diff=v[~s3].mean(), ratio=v[~s3].mean() / v[s3].mean(), p_two=p2, p_one=p1))
Sm = pd.DataFrame(summ)
print(Sm.round(3).to_string(index=False), flush=True)

# exact 56-labelling test of the free-regime bias with M1p terms
rows = []
Xb = np.column_stack([Xp, fr])[main]; evm = ev[main]; frm = fr[main]; daym = S.day.values[main]
bb, llb = ich_fit(Xb, evm)
for same in combinations(range(8), 3):
    lab = {MAIN[i]: (0 if i in same else 1) for i in range(8)}
    z = pd.Series(daym).map(lab).values.astype(float)
    bF, llF = ich_fit(np.column_stack([Xb, frm * z]), evm, b0=np.r_[bb, 0.0])
    bL, llL = ich_fit(np.column_stack([Xb, (1 - frm) * z]), evm, b0=np.r_[bb, 0.0])
    rows.append(dict(same=''.join(map(str, same)), observed=int(same == (0, 1, 2)), b_free_diff=bF[-1], dLL_free_diff=llF - llb,
                     b_lead_diff=bL[-1], dLL_lead_diff=llL - llb))
E = pd.DataFrame(rows); E.to_csv(os.path.join(FIN, 'model_rec_exact.csv'), index=False)
o = E[E.observed == 1].iloc[0]
tests = {}
for st in ('b_free_diff', 'dLL_free_diff', 'b_lead_diff', 'dLL_lead_diff'):
    v = E[st].values
    tests[st] = dict(obs=o[st], p_one=float(np.mean(v >= o[st] - 1e-9)), p_two=float(np.mean(np.abs(v) >= abs(o[st]) - 1e-9)),
                     rank=int((v > o[st] + 1e-9).sum() + 1))
print(tests)
out = dict(model='M1pF', k=len(b_all), LL=ll, AIC=2 * len(b_all) - 2 * ll, LODO=lodo, **{f'b_{c}': b for c, b in zip(cols, b_all)},
           **{f'se_{c}': s for c, s in zip(cols, se)}, **{f'{k}_{kk}': vv for k, t in tests.items() for kk, vv in t.items()})
pd.DataFrame([out]).to_csv(os.path.join(FIN, 'model_rec_summary.csv'), index=False)
Sm.to_csv(os.path.join(FIN, 'model_rec_ratios.csv'), index=False)
print('done', round(time.time() - t0, 1), 's')
