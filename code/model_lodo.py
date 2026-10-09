# -*- coding: utf-8 -*-
"""Leave-one-day-out predictions of the stochastic MOBIL at the observed 1 Hz decision points (x = 1,000-1,800 m, cars and heavy
vehicles, causal desired-speed proxy). Versions: M0 (limits not used), M1 (desired speed capped at the lane limit), M1p (speed factor x
lane limit), M2L (M0 + bias linear in Delta), M2F (M0 + F + F x D), M2Fl (M0 + F + F x Delta/10); F = no own-lane leader within 300 m.
For each day the coefficients come from the other seven days.
Input: _cache/work/decision_points.pkl. Outputs: _cache/work/model_lodo_steps.pkl (hazards per decision point); results/final/model_lodo_day.csv (observed
and predicted changes and rates per day, class, direction and situation), model_lodo_coefs.csv (coefficients)."""
import time, gc
from model_lib import *

t0 = time.time()
MODELS = [('M0', 'M0', None), ('M1', 'M1', None), ('M1p', 'M1p', None), ('M2L', 'M0', 'lin'), ('M2F', 'M0', 'free_step'), ('M2Fl', 'M0', 'free_lin')]


def design2(S, Fm, bias):
    if bias in ('free_step', 'free_lin'):
        X, cols = design(S, Fm)
        fr = S.free.values.astype(float)
        z = S.diff_day.values.astype(float) if bias == 'free_step' else S.d10.values.astype(float)
        return np.column_stack([X, fr, fr * z]), cols + ['free', 'free_x_' + ('diff' if bias == 'free_step' else 'd10')]
    return design(S, Fm, bias)

D = steps('causal', 0); gc.collect()
F = features(D)
print('steps', len(D), round(time.time() - t0, 1), 's', flush=True)
parts, coefs = [], []
for cls, dirn in GROUPS:
    S, Fs = subset(D, F, cls, dirn, None)
    ev = event_ids(S)
    P = S[['day', 'GID', 'cls', 'dirn', 'Timestamp', 'PositionX', 'v', 'free', 'olgap', 'y', 'dt_next', 't_next', 'vdes_use', 'delta', 'diff_day']].copy()
    P['ev'] = ev
    for name, spec, bias in MODELS:
        X, cols = design2(S, Fs[spec], bias)
        h, fits, b_all = lodo_hazards(S, X)
        P[f'h_{name}'] = h
        # in-sample (8 days) hazard for reference
        main = S.day.isin(MAIN).values
        P[f'hin_{name}'] = expit(X @ b_all)
        for d, b in fits.items():
            coefs.append(dict(cls=cls, dirn=dirn, model=name, left_out=d, **{c: x for c, x in zip(cols, b)}))
        coefs.append(dict(cls=cls, dirn=dirn, model=name, left_out='none(8 days)', **{c: x for c, x in zip(cols, b_all)}))
        print(cls, dirn, name, round(time.time() - t0, 1), 's', flush=True)
    parts.append(P)
    del S, Fs; gc.collect()
P = pd.concat(parts, ignore_index=True)
P.to_pickle(os.path.join(CACHE, 'model_lodo_steps.pkl'))
pd.DataFrame(coefs).to_csv(os.path.join(FIN, 'model_lodo_coefs.csv'), index=False)

# ---------------- day-level summaries
# free state of an event: state of its step closest to 3 s before the recorded stamp (situation 3 s before the change)
P['km'] = P.v / 3600.0
w = P[P.ev >= 0].copy(); w['d3'] = (w.dt_next - 3.0).abs()
ev_state = w.sort_values('d3').groupby(['cls', 'dirn', 'ev']).first()[['day', 'GID', 'free']].reset_index()
rows = []
for (cls, dirn, day), g in P.groupby(['cls', 'dirn', 'day']):
    e = ev_state[(ev_state.cls == cls) & (ev_state.dirn == dirn) & (ev_state.day == day)]
    for st, gs, es in (('all', g, e), ('free', g[g.free == 1], e[e.free == 1]), ('leader', g[g.free == 0], e[e.free == 0])):
        r = dict(cls=cls, dirn=dirn, day=day, cfg=DAY_META.set_index('day').cfg[day], delta=int(DAY_META.set_index('day').delta[day]), state=st,
                 n_steps=len(gs), n_veh=gs.GID.nunique(), km=gs.km.sum(), n_obs=len(es))
        r['rate_obs'] = r['n_obs'] / r['km'] if r['km'] > 0 else np.nan
        # vehicle-cluster bootstrap CI of the observed rate
        if r['km'] > 0 and len(gs):
            num = pd.Series(1.0, index=es.GID).groupby(level=0).sum().reindex(gs.GID.unique()).fillna(0)
            den = gs.groupby('GID').km.sum().reindex(num.index)
            lo, hi = boot_ci(num.values, den.values, num.index.values, B=2000, seed=11)
            r['rate_obs_lo'], r['rate_obs_hi'] = lo, hi
        for name, _, _ in MODELS:
            r[f'n_pred_{name}'] = gs[f'h_{name}'].sum()
            r[f'rate_pred_{name}'] = r[f'n_pred_{name}'] / r['km'] if r['km'] > 0 else np.nan
            r[f'n_fit_{name}'] = gs[f'hin_{name}'].sum()
        rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(FIN, 'model_lodo_day.csv'), index=False)
pd.set_option('display.width', 250)
show = R[(R.cls == 'car') & (R.state.isin(['free', 'all']))][['dirn', 'state', 'cfg', 'n_obs', 'km', 'rate_obs', 'rate_pred_M0', 'rate_pred_M1', 'rate_pred_M1p', 'rate_pred_M2L', 'rate_pred_M2F', 'rate_pred_M2Fl', 'n_fit_M1p']]
print(show.round(3).to_string(index=False))
print('done', round(time.time() - t0, 1), 's')
