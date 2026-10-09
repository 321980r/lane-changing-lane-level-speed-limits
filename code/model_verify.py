# -*- coding: utf-8 -*-
"""Fits of the stochastic MOBIL (interval-censored 1-s hazard) for car inward, car outward, heavy inward and heavy outward changes on the
eight days, models M0, M1, M1p, M2L, M2s and M3pL, with log-likelihood, AIC, leave-one-day-out log-likelihood and vehicle-clustered
standard errors, for one desired-speed proxy ('orig', 'causal' or 'lag3', see model_lib.steps). With 'orig' it also checks the event
windows and recomputes the IDM/MOBIL terms by hand for random rows.
usage: python model_verify.py orig|causal|lag3
Inputs: _cache/work/decision_points.pkl, core_events.pkl. Outputs: results/final/model_verify_models_<proxy>.csv, model_verify_checks_<proxy>.csv,
model_verify_checks_window.csv ('orig' only)."""
import time
from model_lib import *

import gc
t0 = time.time()
rows, checks = [], []
VARIANTS = {'orig': ('orig', 0), 'causal': ('causal', 0), 'lag3': ('causal', 3)}
which = sys.argv[1] if len(sys.argv) > 1 else 'orig'
MODELS = [('M0', 'M0', None), ('M1', 'M1', None), ('M1p', 'M1p', None), ('M2L', 'M0', 'lin'), ('M2s', 'M0', 'step'), ('M3pL', 'M1p', 'lin')]
for vsrc, lag in [VARIANTS[which]]:
    D = steps(vsrc, lag); gc.collect()
    F = features(D)
    tag = vsrc + (f'_lag{lag}' if lag else '')
    for cls, dirn in GROUPS:
        S, Fs = subset(D, F, cls, dirn, MAIN)
        ev = event_ids(S)
        nev = int(ev.max() + 1)
        evf = pd.Series(ev[ev >= 0])
        # event-level fallback share (any step of the window uses the fallback)
        fb_ev = pd.Series(S.vdes_fb.values[ev >= 0]).groupby(evf.values).max().mean()
        checks.append(dict(variant=tag, group=f'{cls}_{dirn}', n_steps=len(S), n_events=nev, n_veh=S.GID.nunique(),
                           share_steps_fallback=S.vdes_fb.mean(), share_events_fallback=fb_ev,
                           steps_per_window_mean=evf.value_counts().mean(), windows_with_5_steps=(evf.value_counts() == 5).mean(),
                           windows_lt3_steps=(evf.value_counts() < 3).mean()))
        for name, spec, bias in MODELS:
            X, cols = design(S, Fs[spec], bias)
            b, ll = ich_fit(X, ev)
            lodo = 0.0; per = {}
            for d in MAIN:
                a = (S.day != d).values
                bb, _ = ich_fit(X[a], ev[a], b0=b)
                per[d] = ich_ll(bb, X[~a], ev[~a]); lodo += per[d]
            V = ich_vcov(b, X, ev, S.GID.values); se = np.sqrt(np.clip(np.diag(V), 0, None))
            r = dict(variant=tag, group=f'{cls}_{dirn}', model=name, k=len(b), LL=ll, AIC=2 * len(b) - 2 * ll, LODO=lodo, n_events=nev, n_steps=len(S))
            r.update({f'b_{c}': x for c, x in zip(cols, b)}); r.update({f'se_{c}': s for c, s in zip(cols, se)})
            r.update({f'lodo_{DAY_META.set_index("day").cfg[d]}': v for d, v in per.items()})
            rows.append(r)
        print(tag, cls, dirn, 'events', nev, round(time.time() - t0, 1), 's', flush=True)
        R = pd.DataFrame(rows); print(R[(R.variant == tag) & (R.group == f'{cls}_{dirn}')][['model', 'LL', 'AIC', 'LODO']].round(2).to_string(index=False), flush=True)

pd.DataFrame(rows).to_csv(os.path.join(FIN, f'model_verify_models_{which}.csv'), index=False)
pd.DataFrame(checks).to_csv(os.path.join(FIN, f'model_verify_checks_{which}.csv'), index=False)
if which != 'orig':
    sys.exit(0)
del D, F; gc.collect(); checks = []

# ---------------- (3) window construction and (4) hand check of the MOBIL terms (orig variant)
D = steps('orig', 0); F = features(D)
S, Fs = subset(D, F, 'car', 'in', MAIN)
ev = event_ids(S)
# steps of each window must lie within 5 s before the stamp and be consecutive seconds of the same vehicle
w = S[ev >= 0].assign(ev=ev[ev >= 0])
dts = (w.t_next - w.Timestamp) / 1000
checks.append(dict(variant='orig', group='car_in', check='window dt range (s)', value=f'{dts.min():.2f}-{dts.max():.2f}'))
# events recorded in the zone for which no step in the zone exists
E = pd.read_pickle(os.path.join(CACHE, 'core_events.pkl'))
e = E[(E.day.isin(MAIN)) & (E.truck == 0) & (E.direction == 'in') & (E.x >= 1000) & (E.x < 1800)]
checks.append(dict(variant='orig', group='car_in', check='analysis-grade car inward stamps in [1000,1800) (core_events)', value=len(e)))
checks.append(dict(variant='orig', group='car_in', check='event windows in the estimation sample', value=int(ev.max() + 1)))
# hand check: free-road IDM with bounded term, interaction term and MOBIL difference for 5 random rows (M1p)
rng = np.random.default_rng(3); P = M.BASE
for i in rng.choice(len(S), 5, replace=False):
    r = S.iloc[i]; v = r.v / 3.6
    v0c = r.sf_use * r.vout / 3.6; v0t = r.sf_use * r.vin / 3.6
    def acc(v0, gap_lead, v_lead):
        fr = P['a'] * (1 - (v / v0) ** 4) if v <= v0 else -P['b'] * (1 - (v0 / v) ** (P['a'] * 4 / P['b']))
        if np.isnan(gap_lead):
            it = 0.0
        else:
            ss = P['s0'] + max(0.0, v * P['T'] + v * (v - v_lead) / (2 * np.sqrt(P['a'] * P['b'])))
            it = -P['a'] * (ss / max(gap_lead, 0.5)) ** 2
        return float(np.clip(fr + it, -4.0, P['a']))
    L = lambda t: 12.0 if t == 1 else 5.0
    tau = P['tau']
    s_ol = (r.ol_PositionX - r.PositionX - L(r.ol_truck)) if (pd.notna(r.ol_PositionX) and r.ol_PositionX - r.PositionX <= 400) else np.nan
    s_tl = (r.tl_PositionX - r.PositionX - L(r.tl_truck)) if (pd.notna(r.tl_PositionX) and r.tl_PositionX - r.PositionX <= 400) else np.nan
    s_ol_a = s_ol + (r.ol_v / 3.6 - v) * tau if not np.isnan(s_ol) else np.nan
    s_tl_a = s_tl + (r.tl_v / 3.6 - v) * tau if not np.isnan(s_tl) else np.nan
    passed = (not np.isnan(s_tl)) and (s_tl + L(r.tl_truck) + (r.tl_v / 3.6 - v) * tau < 0)
    if passed:
        s_tl_a = np.nan
    a_c = acc(v0c, s_ol_a, (r.ol_v / 3.6) if pd.notna(r.ol_v) else np.nan)
    a_t = acc(v0t, s_tl_a, (r.tl_v / 3.6) if pd.notna(r.tl_v) else np.nan)
    checks.append(dict(variant='orig', group='car_in', check=f'hand dA_ego row {i}', value=f'{a_t - a_c:.4f} vs mobil_model {Fs["M1p"].dA_ego.iloc[i]:.4f}'))
C = pd.DataFrame(checks); C.to_csv(os.path.join(FIN, 'model_verify_checks_window.csv'), index=False)
print(C.to_string(index=False))
print('done', round(time.time() - t0, 1), 's')
