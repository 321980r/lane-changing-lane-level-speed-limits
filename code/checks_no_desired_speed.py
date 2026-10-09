# -*- coding: utf-8 -*-
"""Bias of unconstrained cars (beta_F of M2F, car inward changes) without the desired-speed term, and with a desired-speed proxy
measured only upstream of gantry 1 (Section 5.4).
M2F: eta = b0 + b_ego dA_ego + b_n dA_n + b_o dA_o + b_U U + b_v (v_des - 80)/10 + b_s S + b_free F + b_F F D (M0 incentive terms,
causal desired-speed proxy, interval-censored likelihood, eight days, x = 1,000-1,800 m).
Variants: A M2F as in Table 3; B without the b_v term; C as B, with the IDM desired speed inside dA_ego set to 100 km/h; D with v_des =
running free-driving mean over the vehicle's frames upstream of gantry 1 (x < 1,250 m), MOBIL terms recomputed.
For each variant: coefficients with vehicle-clustered standard errors, a_bias = b_F / b_ego (delta method), exp(b_F), and the exact test
of b_F over the 56 labellings.
Input: _cache/work/decision_points.pkl. Outputs: results/final/checks_novdes_fits.csv, checks_novdes_exact.csv, checks_novdes_summary.csv."""
import gc, time
from itertools import combinations
from model_lib import *
from checks_lib import out

t0 = time.time()
D = steps('causal', 0)
D = D[D.day.isin(MAIN) & (D.cls == 'car') & (D.dirn == 'in')].reset_index(drop=True); gc.collect()
print('car inward decision points', len(D), 'events', D.y.sum(), round(time.time() - t0, 1), 's', flush=True)


# ------------------------------------------------------------------------------------------------ upstream-of-gantry-1 proxy
def vdes_upstream(D, xg=1250.0):
    parts = []
    for n in MAIN:
        f = C.frames(n)
        f = f[f.Lane.isin([1, 2])][['GlobalID', 'Timestamp', 'PositionX', 'Smoothed_VelocityX', 'leader_id', 'Headway']].copy()
        f = f.sort_values(['GlobalID', 'Timestamp']).reset_index(drop=True)
        v = f.Smoothed_VelocityX.astype(float).values * 3.6
        free = ((f.leader_id.isna()) | (f.Headway > 5)).values & (f.PositionX.values < xg)
        df = pd.DataFrame({'g': f.GlobalID.values, 'fv': free * v, 'fn': free.astype(float)})
        cs = df.groupby('g')[['fv', 'fn']].cumsum()
        f['vdes_up'] = np.where(cs.fn >= 30, cs.fv / cs.fn.clip(lower=1), np.nan)
        f['nfree_up'] = cs.fn.values
        s = f[f.Timestamp % 1000 == 0][['GlobalID', 'Timestamp', 'vdes_up', 'nfree_up']].copy(); s['day'] = n
        parts.append(s)
    U = pd.concat(parts, ignore_index=True)
    m = D[['day', 'GlobalID', 'Timestamp', 'truck']].merge(U, on=['day', 'GlobalID', 'Timestamp'], how='left')
    assert len(m) == len(D)
    return m.vdes_up.values, m.nfree_up.values


vup, nup = vdes_upstream(D)
D['vdes_up_raw'] = vup
med = D.groupby(['day', 'truck']).vdes_up_raw.transform('median')
print('upstream proxy known at', round(np.isfinite(vup).mean(), 3), 'of decision points; corr with causal proxy',
      round(pd.Series(vup).corr(D.vdes_use), 3), flush=True)


def fit_variant(Dv, drop_v=False, v0_common=None):
    Dm = Dv.copy()
    if v0_common is not None:
        Dm['vdes_use'] = v0_common
    Fm = M.mobil_terms(Dm, M.gaps(Dm), 'M0')
    X, cols = design(Dv, Fm)                            # vd10 column uses Dv.vdes_use (original or upstream)
    fr = Dv.free.values.astype(float); dd = Dv.diff_day.values.astype(float)
    X = np.column_stack([X, fr, fr * dd]); cols = cols + ['free', 'free_x_diff']
    if drop_v:
        k = cols.index('vd10'); X = np.delete(X, k, axis=1); cols = cols[:k] + cols[k + 1:]
    return X, cols, Fm


VARIANTS = [('A', 'M2F (Table 3)', dict()),
            ('B', 'M2F without b_v', dict(drop_v=True)),
            ('C', 'M2F without b_v, IDM v0 = 100 km/h in both lanes', dict(drop_v=True, v0_common=100.0)),
            ('D', 'M2F, v_des upstream of gantry 1', dict(upstream=True)),
            ('E', 'M2F without b_v, v_des upstream of gantry 1 in dA terms', dict(upstream=True, drop_v=True))]
ev = event_ids(D); gid = D.GID.values; daym = D.day.values
fits, exact = [], []
for key, name, opt in VARIANTS:
    Dv = D
    if opt.get('upstream'):
        Dv = D.copy(); Dv['vdes_use'] = D.vdes_up_raw.fillna(med).values; Dv['vd10'] = (Dv.vdes_use - 80) / 10
    X, cols, Fm = fit_variant(Dv, drop_v=opt.get('drop_v', False), v0_common=opt.get('v0_common'))
    b, ll = ich_fit(X, ev)
    V = ich_vcov(b, X, ev, gid); se = np.sqrt(np.clip(np.diag(V), 0, None))
    iF = cols.index('free_x_diff'); iE = cols.index('dA_ego')
    bF, sF = b[iF], se[iF]
    ab = bF / b[iE]; g = np.zeros(len(b)); g[iF] = 1 / b[iE]; g[iE] = -bF / b[iE] ** 2
    sab = float(np.sqrt(g @ V @ g))
    # exact 56-labelling test of b_F
    Xb = np.delete(X, iF, axis=1); bb, llb = ich_fit(Xb, ev, b0=np.delete(b, iF))
    fr = D.free.values.astype(float)
    null = []
    for same in combinations(range(8), 3):
        lab = {MAIN[i]: (0 if i in same else 1) for i in range(8)}
        z = pd.Series(daym).map(lab).values.astype(float)
        bz, llz = ich_fit(np.column_stack([Xb, fr * z]), ev, b0=np.r_[bb, 0.0])
        null.append(bz[-1])
        exact.append(dict(variant=key, same=''.join(map(str, same)), observed=int(same == (0, 1, 2)), b_F=bz[-1], dLL=llz - llb))
    null = np.array(null); obs = null[0]
    assert abs(obs - bF) < 1e-3, (obs, bF)
    r = dict(variant=key, model=name, k=len(b), LL=ll, AIC=2 * len(b) - 2 * ll, n_points=len(D), n_events=int(ev.max() + 1), n_vehicles=D.GID.nunique(),
             b_F=bF, se_F=sF, b_F_lo=bF - 1.96 * sF, b_F_hi=bF + 1.96 * sF, OR_F=np.exp(bF), OR_F_lo=np.exp(bF - 1.96 * sF), OR_F_hi=np.exp(bF + 1.96 * sF),
             a_bias=ab, a_bias_lo=ab - 1.96 * sab, a_bias_hi=ab + 1.96 * sab,
             p_two_exact56=float(np.mean(np.abs(null) >= abs(obs) - 1e-9)), p_one_exact56=float(np.mean(null >= obs - 1e-9)),
             rank_of_56=int((null > obs + 1e-9).sum() + 1),
             **{f'b_{c}': x for c, x in zip(cols, b)}, **{f'se_{c}': x for c, x in zip(cols, se)})
    if opt.get('upstream'):
        r['share_proxy_fallback'] = float(D.vdes_up_raw.isna().mean())
    fits.append(r)
    print(key, name, 'LL', round(ll, 2), 'b_F', round(bF, 3), 'se', round(sF, 3), 'a_bias', round(ab, 2), 'p56', round(r['p_two_exact56'], 3),
          round(time.time() - t0, 1), 's', flush=True)
Fi = pd.DataFrame(fits); p1 = out(Fi, 'checks_novdes_fits.csv'); p2 = out(pd.DataFrame(exact), 'checks_novdes_exact.csv')
cols_s = ['variant', 'model', 'k', 'LL', 'AIC', 'b_F', 'b_F_lo', 'b_F_hi', 'OR_F', 'OR_F_lo', 'OR_F_hi', 'a_bias', 'a_bias_lo', 'a_bias_hi',
          'p_two_exact56', 'p_one_exact56', 'rank_of_56', 'b_dA_ego', 'b_free', 'b_vd10']
Sm = Fi[[c for c in cols_s if c in Fi.columns]]; p3 = out(Sm, 'checks_novdes_summary.csv')
pd.set_option('display.width', 250)
print(Sm.round(4).to_string(index=False)); print('saved', p1, p2, p3, round(time.time() - t0, 1), 's')
