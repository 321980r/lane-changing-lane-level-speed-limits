# -*- coding: utf-8 -*-
"""Further versions of the car-inward model on the eight days, for the comparison of model versions:
  M2F_dayFE  M2F with a constant per day: beta_F with vehicle-clustered interval, a_bias, leave-one-day-out log-likelihood (held-out
             constant = mean of the seven others) and factors, exact 56-labelling test of beta_F with the day constants refitted;
  M2Fl       bias linear in Delta (b_Fl F Delta/10): a_bias per 10 km/h (delta method), exact test over the 1,120 distinct reassignments
             of the eight Delta values to the days (cached in _cache/work/versions_m2fl_exact.pkl, resumable);
  M1pF       refit for the delta-method interval of a_bias.
Inputs: versions_lib.get_carin(); results/final/model_bias_checks.csv, model_rec_summary.csv (checks).
Outputs: results/final/model_versions.csv, versions_m2fl_exact.csv, versions_dayfe_exact.csv."""
import time
from itertools import combinations, permutations
from versions_lib import *

t0 = time.time()
D = get_carin(); ev = event_ids(D); gid = D.GID.values; daym = D.day.values; fr = D.free.values.astype(float)
Robs = day_rates(D, ev); fo = factors(Robs, 'rate_obs')
rows = []


def summarise(key, name, X, cols, b, ll, bias_col, lodo_res=None, exact=None):
    V = ich_vcov(b, X, ev, gid); se = np.sqrt(np.clip(np.diag(V), 0, None))
    iF, iI, if_ = cols.index(bias_col), cols.index('dA_ego'), cols.index('free')
    a, alo, ahi, _ = ratio_ci_delta(b, V, iF, iI)
    r = dict(model=key, description=name, k=len(b), LL=ll, AIC=2 * len(b) - 2 * ll, bias_term=bias_col,
             b_F=b[iF], se_F=se[iF], b_F_lo=b[iF] - 1.96 * se[iF], b_F_hi=b[iF] + 1.96 * se[iF], OR_F=np.exp(b[iF]),
             b_I=b[iI], a_bias=a, a_bias_lo=alo, a_bias_hi=ahi, b_free=b[if_], se_free=se[if_])
    if lodo_res is not None:
        lodo_ll, h, llday, fits = lodo_res
        fp = factors(day_rates(D, ev, h), 'rate_pred')
        r.update(LODO_LL=lodo_ll, lodo_factor_unconstrained=fp['free'], lodo_factor_leader300=fp['leader'], lodo_contrast=fp['contrast'],
                 lodo_b_F_min=min(f[iF] for f in fits.values()), lodo_b_F_max=max(f[iF] for f in fits.values()))
    if exact is not None:
        r.update(exact)
    r.update({f'b_{c}': x for c, x in zip(cols, b)}, **{f'se_{c}': x for c, x in zip(cols, se)})
    return r


# ------------------------------------------------------------------------------------------------ M2F with day fixed effects
X, cols = design_m2f(D, 'M0', bias='step', dayfe=True)
b, ll = ich_fit(X, ev, b0=b0_for(cols))
lo = lodo(D, X, cols, ev, b)
iF = cols.index('free_x_diff'); Xb = np.delete(X, iF, axis=1); bb, llb = ich_fit(Xb, ev, b0=np.delete(b, iF))
ex = []
for same in combinations(range(8), 3):
    lab = {MAIN[i]: (0 if i in same else 1) for i in range(8)}
    z = pd.Series(daym).map(lab).values.astype(float)
    bz, llz = ich_fit(np.column_stack([Xb, fr * z]), ev, b0=np.r_[bb, 0.0])
    ex.append(dict(same=''.join(map(str, same)), observed=int(same == (0, 1, 2)), b_F=bz[-1], dLL=llz - llb))
EX = pd.DataFrame(ex); EX.to_csv(os.path.join(FIN, 'versions_dayfe_exact.csv'), index=False)
nul = EX.b_F.values; obs = EX[EX.observed == 1].b_F.iloc[0]
assert abs(obs - b[iF]) < 1e-3
exd = dict(p_one=float(np.mean(nul >= obs - 1e-9)), p_two=float(np.mean(np.abs(nul) >= abs(obs) - 1e-9)), test='exact, 56 labellings (day FE refitted)',
           n_perm=len(nul), rank=int((nul > obs + 1e-9).sum() + 1))
r = summarise('M2F_dayFE', 'M2F with day fixed effects (8 day intercepts)', X, cols, b, ll, 'free_x_diff', lo, exd)
r['lodo_note'] = 'held-out day intercept = mean of the 7 training-day intercepts'
rows.append(r)
print('M2F day FE: LL', round(ll, 3), 'b_F', round(r['b_F'], 4), '(', round(r['b_F_lo'], 3), round(r['b_F_hi'], 3), ') a', round(r['a_bias'], 3),
      'LODO', round(r['LODO_LL'], 3), 'p56', round(exd['p_one'], 4), round(exd['p_two'], 4), 'factors', round(r['lodo_factor_unconstrained'], 3),
      round(r['lodo_factor_leader300'], 3), round(time.time() - t0, 1), 's', flush=True)
print('  day intercepts', dict(zip([c for c in cols if c.startswith('day_')], np.round(b[:8], 3))), flush=True)

# ------------------------------------------------------------------------------------------------ M2F linear in Delta
X, cols = design_m2f(D, 'M0', bias='lin')
b, ll = ich_fit(X, ev, b0=b0_for(cols))
FB = pd.read_csv(os.path.join(FIN, 'model_bias_checks.csv')).set_index('model').loc['M0+free+free:d10']
assert abs(ll - FB.LL) < 1e-3 and abs(b[-1] - FB.b_last) < 1e-3, (ll, FB.LL, b[-1], FB.b_last)
lo = lodo(D, X, cols, ev, b)
assert abs(lo[0] - FB.LODO) < 1e-2, (lo[0], FB.LODO)
iF = cols.index('free_x_d10'); Xb = np.delete(X, iF, axis=1); bb, llb = ich_fit(Xb, ev, b0=np.delete(b, iF))
dlt = {n: int(DAY_META.set_index('day').delta[n]) for n in MAIN}
obs_vec = tuple(dlt[n] for n in MAIN)
assign = sorted(set(permutations(obs_vec)))
ck = os.path.join(CACHE, 'versions_m2fl_exact.pkl')
done = pd.read_pickle(ck) if os.path.exists(ck) else []
seen = {r_['assignment'] for r_ in done}
print('M2Fl: distinct Delta assignments', len(assign), 'cached', len(done), flush=True)
for j, a_ in enumerate(assign):
    key = '/'.join(map(str, a_))
    if key in seen:
        continue
    z = pd.Series(daym).map(dict(zip(MAIN, a_))).values.astype(float) / 10
    bz, llz = ich_fit(np.column_stack([Xb, fr * z]), ev, b0=np.r_[bb, 0.0])
    done.append(dict(assignment=key, observed=int(a_ == obs_vec), b_Fl=bz[-1], dLL=llz - llb))
    if len(done) % 200 == 0:
        pd.to_pickle(done, ck); print('  ', len(done), round(time.time() - t0, 1), 's', flush=True)
pd.to_pickle(done, ck)
EL = pd.DataFrame(done); EL.to_csv(os.path.join(FIN, 'versions_m2fl_exact.csv'), index=False)
nul = EL.b_Fl.values; obs = EL[EL.observed == 1].b_Fl.iloc[0]
assert abs(obs - b[iF]) < 1e-3
exl = dict(p_one=float(np.mean(nul >= obs - 1e-9)), p_two=float(np.mean(np.abs(nul) >= abs(obs) - 1e-9)),
           test='exact, 1,120 reassignments of Delta to days', n_perm=len(nul), rank=int((nul > obs + 1e-9).sum() + 1),
           p_one_dLL=float(np.mean(EL.dLL.values >= EL[EL.observed == 1].dLL.iloc[0] - 1e-9)))
r = summarise('M2Fl', 'M2F with the bias linear in Delta (per 10 km/h)', X, cols, b, ll, 'free_x_d10', lo, exl)
rows.append(r)
print('M2Fl: LL', round(ll, 3), 'b', round(r['b_F'], 4), '(', round(r['b_F_lo'], 3), round(r['b_F_hi'], 3), ') a/10', round(r['a_bias'], 3),
      '(', round(r['a_bias_lo'], 3), round(r['a_bias_hi'], 3), ') LODO', round(r['LODO_LL'], 3), 'p one/two', round(exl['p_one'], 4), round(exl['p_two'], 4),
      'rank', exl['rank'], 'factors', round(r['lodo_factor_unconstrained'], 3), round(r['lodo_factor_leader300'], 3), round(time.time() - t0, 1), 's', flush=True)

# ------------------------------------------------------------------------------------------------ M1pF (interval of a_bias only)
X, cols = design_m2f(D, 'M1p', bias='step')
b, ll = ich_fit(X, ev, b0=b0_for(cols))
RS = pd.read_csv(os.path.join(FIN, 'model_rec_summary.csv')).iloc[0]
assert abs(ll - RS.LL) < 1e-3 and abs(b[-1] - RS.b_free_x_diff) < 1e-3, (ll, RS.LL, b[-1], RS.b_free_x_diff)
r = summarise('M1pF', 'M1p + F + F x D', X, cols, b, ll, 'free_x_diff', None,
              dict(p_one=float(RS.b_free_diff_p_one), p_two=float(RS.b_free_diff_p_two), test='exact, 56 labellings (model_rec_summary.csv)', n_perm=56,
                   rank=int(RS.b_free_diff_rank)))
r['LODO_LL'] = float(RS.LODO)
rows.append(r)
print('M1pF: a_bias', round(r['a_bias'], 3), '(', round(r['a_bias_lo'], 3), round(r['a_bias_hi'], 3), ')', flush=True)

Out = pd.DataFrame(rows)
Out['obs_factor_unconstrained'] = fo['free']; Out['obs_factor_leader300'] = fo['leader']; Out['obs_contrast'] = fo['contrast']
p = os.path.join(FIN, 'model_versions.csv'); Out.to_csv(p, index=False)
print('saved', p, round(time.time() - t0, 1), 's')
