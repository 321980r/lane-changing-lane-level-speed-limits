# -*- coding: utf-8 -*-
"""beta_F of the car-inward M2F without the desired-speed term b_v (v_des - 80)/10 (Section 5.4).
M2F: eta = b0 + b_I dA_c + b_n dA_n + b_o dA_o + b_U U + b_v (v_des - 80)/10 + b_r R + b_f F + b_F F D (M0 incentive terms, causal
desired speed, interval-censored 1-s hazard, eight days, x = 1,000-1,800 m); the variant 'M2F without b_v' drops the b_v term.
For each variant: coefficients with vehicle-clustered errors, a_bias = beta_F / beta_I (delta method), log-likelihood, AIC,
leave-one-day-out log-likelihood and differentiated/same factors, and the exact test of beta_F over the 56 labellings.
Inputs: versions_lib.get_carin(); results/final/model_bias_checks.csv (check).
Outputs: results/final/versions_bias_no_desired_speed.csv, versions_betaF_novdes_exact.csv; _cache/work/versions_lodo_novdes.pkl (leave-one-day-out hazards)."""
import time
from itertools import combinations
from versions_lib import *

t0 = time.time()
D = get_carin(); ev = event_ids(D); gid = D.GID.values; daym = D.day.values; fr = D.free.values.astype(float)
print('car inward', len(D), 'points', int(ev.max() + 1), 'changes', D.GID.nunique(), 'vehicles', flush=True)
Robs = day_rates(D, ev)
fo = factors(Robs, 'rate_obs')
print('observed factors', {k: round(v, 3) for k, v in fo.items()}, flush=True)

rows, exact, cache = [], [], {}
for key, name, drop_v in (('M2F', 'M2F (Table 3)', False), ('M2F_novdes', 'M2F without the desired-speed term', True)):
    X, cols = design_m2f(D, 'M0', bias='step', drop_v=drop_v)
    b, ll = ich_fit(X, ev, b0=b0_for(cols))
    V = ich_vcov(b, X, ev, gid); se = np.sqrt(np.clip(np.diag(V), 0, None))
    iF, iI, if_ = cols.index('free_x_diff'), cols.index('dA_ego'), cols.index('free')
    a, alo, ahi, _ = ratio_ci_delta(b, V, iF, iI)
    af, aflo, afhi, _ = ratio_ci_delta(b, V, if_, iI)
    lodo_ll, h, llday, fits = lodo(D, X, cols, ev, b)
    R = day_rates(D, ev, h); fp = factors(R, 'rate_pred')
    cache[key] = h
    # exact 56-labelling test of beta_F
    Xb = np.delete(X, iF, axis=1); bb, llb = ich_fit(Xb, ev, b0=np.delete(b, iF))
    null, dll = [], []
    for same in combinations(range(8), 3):
        lab = {MAIN[i]: (0 if i in same else 1) for i in range(8)}
        z = pd.Series(daym).map(lab).values.astype(float)
        bz, llz = ich_fit(np.column_stack([Xb, fr * z]), ev, b0=np.r_[bb, 0.0])
        null.append(bz[-1]); dll.append(llz - llb)
        exact.append(dict(variant=key, same=''.join(map(str, same)), observed=int(same == (0, 1, 2)), b_F=bz[-1], dLL=llz - llb))
    null = np.array(null); obs = null[0]
    assert abs(obs - b[iF]) < 1e-3, (obs, b[iF])
    r = dict(variant=key, model=name, k=len(b), LL=ll, AIC=2 * len(b) - 2 * ll, LODO_LL=lodo_ll,
             n_points=len(D), n_changes=int(ev.max() + 1), n_vehicles=D.GID.nunique(),
             b_F=b[iF], se_F=se[iF], b_F_lo=b[iF] - 1.96 * se[iF], b_F_hi=b[iF] + 1.96 * se[iF],
             OR_F=np.exp(b[iF]), OR_F_lo=np.exp(b[iF] - 1.96 * se[iF]), OR_F_hi=np.exp(b[iF] + 1.96 * se[iF]),
             b_I=b[iI], se_I=se[iI], a_bias=a, a_bias_lo=alo, a_bias_hi=ahi, a_free=af, a_free_lo=aflo, a_free_hi=afhi,
             p_one_exact56=float(np.mean(null >= obs - 1e-9)), p_two_exact56=float(np.mean(np.abs(null) >= abs(obs) - 1e-9)),
             rank_of_56=int((null > obs + 1e-9).sum() + 1), null_b_F_max_other=float(np.max(null[1:])),
             lodo_factor_unconstrained=fp['free'], lodo_factor_leader300=fp['leader'], lodo_contrast=fp['contrast'],
             obs_factor_unconstrained=fo['free'], obs_factor_leader300=fo['leader'], obs_contrast=fo['contrast'],
             lodo_b_F_min=min(f[iF] for f in fits.values()), lodo_b_F_max=max(f[iF] for f in fits.values()),
             **{f'lodo_ll_{d[-6:]}': v for d, v in llday.items()},
             **{f'b_{c}': x for c, x in zip(cols, b)}, **{f'se_{c}': x for c, x in zip(cols, se)})
    rows.append(r)
    print(key, 'LL', round(ll, 3), 'LODO', round(lodo_ll, 3), 'b_F', round(b[iF], 4), '(', round(r['b_F_lo'], 3), round(r['b_F_hi'], 3), ')',
          'a_bias', round(a, 3), '(', round(alo, 3), round(ahi, 3), ')', 'p56 one/two', round(r['p_one_exact56'], 4), round(r['p_two_exact56'], 4),
          'factors', round(fp['free'], 3), round(fp['leader'], 3), round(fp['contrast'], 3), round(time.time() - t0, 1), 's', flush=True)

Fi = pd.DataFrame(rows)
ref = pd.read_csv(os.path.join(FIN, 'model_bias_checks.csv')).set_index('model').loc['M0+free+free:diff']
assert abs(Fi.LL.iloc[0] - ref.LL) < 1e-3 and abs(Fi.LODO_LL.iloc[0] - ref.LODO) < 1e-2, (Fi.LL.iloc[0], Fi.LODO_LL.iloc[0], ref.LL, ref.LODO)
p1 = os.path.join(FIN, 'versions_bias_no_desired_speed.csv'); Fi.to_csv(p1, index=False)
p2 = os.path.join(FIN, 'versions_betaF_novdes_exact.csv'); pd.DataFrame(exact).to_csv(p2, index=False)
pd.to_pickle({'GID': D.GID.values, 'Timestamp': D.Timestamp.values, **cache}, os.path.join(CACHE, 'versions_lodo_novdes.pkl'))
print('saved', p1, p2, round(time.time() - t0, 1), 's')
