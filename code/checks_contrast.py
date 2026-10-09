# -*- coding: utf-8 -*-
"""Further day-level checks of the car inward rates (Section 5.2). Situations as in rates_data.py (at t-3 s: unconstrained = no
outer-lane vehicle within 300 m ahead; leader-driven = state C); exact tests over the 56 labellings of the eight days.
 1. within-day contrast C_d = ln(unconstrained rate / leader-driven rate): (a) adjusted for the day flow of Table 1, by permuting the labels
    on the residuals of C_d ~ flow and by permuting the labels in C_d ~ D + flow (with a test-inverted interval of exp(b_D)); (b) one change
    more or fewer on one day, on two days and on all days in the adverse direction; (c) exact interval of each day's contrast;
 2. log-scale test of the unconstrained rate and Holm adjustment over the six types of lane change;
 3. the part of each session from 10:00 on (days with >= 5 unconstrained car-km);
 4. opposite changes of other vehicles within +-3 s and +-50 m of car inward changes.
Inputs: state tables and typed events; results/final/fig4_day_rates_data.csv, rates_by_type_tests.csv, Table1_days_numbers.csv;
_cache/work/lateral_check.pkl; session frames and events.
Outputs (results/final): contrast_days.csv, contrast_tests.csv, contrast_perturbation.csv, contrast_perturbation_summary.csv,
contrast_log_holm.csv, contrast_from10_days.csv, contrast_from10_tests.csv, contrast_mirror_events.csv, contrast_mirror_summary.csv,
contrast_mirror_tests.csv."""
import os
from itertools import combinations
import numpy as np, pandas as pd
from scipy import stats
from plot_style import *          # noqa  (DATF, DAY_META, ORDER8, frames, lc_events ...)
from situations_lib import events, state, event_states, CFG, DEL
from situations_rates import classify_events as _classify_s1, exposure, day_table

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
out = lambda df, nm: df.to_csv(os.path.join(DATF, nm), index=False)
SAME, DIFF = ORDER8[:3], ORDER8[3:]
ALL56 = [np.array([i not in S_ for i in range(8)]) for S_ in combinations(range(8), 3)]   # True = differentiated


def classify_events(E_, k=3, **kw):
    """identical to rates_data.classify_events (S1 typology + origin-lane consistency at t-k)"""
    typ = _classify_s1(E_, k=k, **kw)
    org = np.where(E_.direction.values == 'in', 2, 1)
    bad = E_[f's{k}_Lane'].notna().values & (E_[f's{k}_Lane'].values != org)
    typ = typ.copy(); typ[bad] = 'U'
    return typ


# ------------------------------------------------------------------------------------------------ exact day-level tests
def perm(v, n_same=3, rank=False):
    """exact relabelling test; first n_same values are same-limit days. statistic mean(diff) - mean(same).
    returns (obs, p two-sided |T|, p one-sided diff > same, n labellings)"""
    v = np.asarray(v, float)
    if rank:
        v = pd.Series(v).rank().values
    k = len(v); obs = v[n_same:].mean() - v[:n_same].mean(); null = []
    for S_ in combinations(range(k), n_same):
        m = np.ones(k, bool); m[list(S_)] = False
        null.append(v[m].mean() - v[~m].mean())
    null = np.array(null)
    return obs, float(np.mean(np.abs(null) >= abs(obs) - 1e-12)), float(np.mean(null >= obs - 1e-12)), len(null)


def _one_sided_both(v, n_same):
    k = len(v); obs = v[n_same:].mean() - v[:n_same].mean(); null = []
    for S_ in combinations(range(k), n_same):
        m = np.ones(k, bool); m[list(S_)] = False
        null.append(v[m].mean() - v[~m].mean())
    null = np.array(null)
    return float(np.mean(null >= obs - 1e-12)), float(np.mean(null <= obs + 1e-12))


GRID = np.exp(np.linspace(np.log(0.02), np.log(50), 1200))


def ratio_ci(v, n_same, log=False):
    """equal-tailed test-inverted 95 % CI (same construction as rates_data.ratio_ci): keep r when both exact one-sided p of the
    shifted data exceed 0.025 (differentiated values divided by r, or log values shifted by log r)."""
    v = np.asarray(v, float); ok = []
    for r in GRID:
        w = v.copy()
        if log:
            w[n_same:] = w[n_same:] - np.log(r)
        else:
            w[n_same:] = w[n_same:] / r
        pu, pl = _one_sided_both(w, n_same)
        ok.append(pu > 0.025 and pl > 0.025)
    ok = np.array(ok)
    est = np.exp(v[n_same:].mean() - v[:n_same].mean()) if log else v[n_same:].mean() / v[:n_same].mean()
    return _connected(ok, est)


def _connected(ok, est):
    i0 = int(np.argmin(np.abs(np.log(GRID) - np.log(est))))
    if not ok[i0]:
        return np.nan, np.nan
    lo = i0
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    hi = i0
    while hi < len(GRID) - 1 and ok[hi + 1]:
        hi += 1
    return (GRID[lo] if lo > 0 else 0.0), (GRID[hi] if hi < len(GRID) - 1 else np.inf)


def full_test(v, n_same=3, log=False):
    """ratio (of arithmetic means, or of geometric means if log=True where v are logs), CI, p two/one-sided, rank p, separation"""
    v = np.asarray(v, float)
    obs, p2, p1, nperm = perm(v, n_same)
    pr = perm(v, n_same, rank=True)[1]
    ratio = float(np.exp(obs)) if log else float(v[n_same:].mean() / v[:n_same].mean())
    lo, hi = ratio_ci(v, n_same, log=log)
    return dict(ratio=ratio, ci_lo=lo, ci_hi=hi, p2=p2, p1=p1, p2_rank=pr, nperm=nperm, complete_sep=int(v[n_same:].min() > v[:n_same].max()))


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for j, i in enumerate(o):
        run = max(run, min(1.0, (m - j) * p[i])); adj[i] = run
    return adj


def garwood(n, km):
    lo = stats.chi2.ppf(0.025, 2 * n) / 2 / km if n > 0 else 0.0
    hi = stats.chi2.ppf(0.975, 2 * (n + 1)) / 2 / km
    return lo, hi


def ratio_exact(n1, k1, n2, k2, alpha=0.05):
    """exact CI for the ratio of two Poisson rates (n1/k1)/(n2/k2): Clopper-Pearson interval for p = n1/(n1+n2) conditional on the
    total, transformed by R = p/(1-p) * k2/k1. Also the exact two-sided conditional binomial p for R = 1."""
    N = n1 + n2
    pl = stats.beta.ppf(alpha / 2, n1, N - n1 + 1) if n1 > 0 else 0.0
    pu = stats.beta.ppf(1 - alpha / 2, n1 + 1, N - n1) if n1 < N else 1.0
    lo = pl / (1 - pl) * k2 / k1
    hi = pu / (1 - pu) * k2 / k1 if pu < 1 else np.inf
    p0 = k1 / (k1 + k2)
    pv = stats.binomtest(int(n1), int(N), p0).pvalue if N > 0 else np.nan
    return (n1 / k1) / (n2 / k2) if n2 > 0 else np.inf, lo, hi, pv


# ------------------------------------------------------------------------------------------------ load and type (primary spec)
E = events(); S = state()
E['typ'] = classify_events(E, k=3)
ST = exposure(S)
D = day_table(E, S, E.typ.values, ST)

# reproduction check vs fig4_day_rates_data.csv
F4 = pd.read_csv(os.path.join(DATF, 'fig4_day_rates_data.csv'))
chk = []
for reg, T in (('free', 'IN_free'), ('leader-driven', 'IN_overtake')):
    a = F4[F4.regime == reg].set_index('day'); b = D[(D.truck == 0) & (D.type == T)].set_index('day')
    chk.append(dict(regime=reg, max_abs_n_diff=int((a.n - b.n.reindex(a.index)).abs().max()),
                    max_abs_km_diff=float((a.car_km - b.km.reindex(a.index)).abs().max())))
print('reproduction vs fig4_day_rates_data.csv:', chk)
assert all(c['max_abs_n_diff'] == 0 and c['max_abs_km_diff'] < 1e-6 for c in chk)


def rates(Dx, tr, T):
    z = Dx[(Dx.truck == tr) & (Dx.type == T)].set_index('day')
    return z.n.to_dict(), z.km.to_dict(), z.rate.to_dict()


fn, fk, fr = rates(D, 0, 'IN_free'); cn, ck, cr = rates(D, 0, 'IN_overtake')
T1 = pd.read_csv(os.path.join(DATF, "Table1_days_numbers.csv")); T1 = T1[T1.day.isin(ORDER8)].set_index("day")
FLOW = T1.flow_vph.to_dict()                                   # exact flows (veh/h)
FLOW_R = {d: round(v) for d, v in FLOW.items()}                # rounded as printed in Table 1

# ================================================================================================ 1. within-day contrast
C_d = {x: np.log(fr[x] / cr[x]) for x in ORDER8}
rows = []
g_same = np.mean([C_d[x] for x in SAME]); g_diff = np.mean([C_d[x] for x in DIFF])
for x in ORDER8:
    R_, lo, hi, _ = ratio_exact(fn[x], fk[x], cn[x], ck[x])
    rows.append(dict(day=x, cfg=CFG[x], label=DAY_META.set_index('day').label[x], delta=DEL[x],
                     group='same-limit' if DEL[x] == 0 else 'differentiated', flow_vph=FLOW[x],
                     n_unc=int(fn[x]), km_unc=fk[x], rate_unc=fr[x], n_lead=int(cn[x]), km_lead=ck[x], rate_lead=cr[x],
                     ratio_unc_lead=R_, ratio_ci_lo=lo, ratio_ci_hi=hi, C_d=np.log(R_), C_d_lo=np.log(lo) if lo > 0 else -np.inf,
                     C_d_hi=np.log(hi), rel_to_same_geomean=R_ / np.exp(g_same), rel_ci_lo=lo / np.exp(g_same), rel_ci_hi=hi / np.exp(g_same),
                     ci_method='exact conditional binomial (Clopper-Pearson) on n_unc | n_unc + n_lead'))
CD = pd.DataFrame(rows); out(CD, 'contrast_days.csv')
print(CD[['cfg', 'group', 'flow_vph', 'n_unc', 'km_unc', 'n_lead', 'km_lead', 'ratio_unc_lead', 'ratio_ci_lo', 'ratio_ci_hi', 'rel_to_same_geomean']].round(3).to_string(index=False))

# ---- (a) flow adjustment
y = np.array([C_d[x] for x in ORDER8]); Dv = np.r_[np.zeros(3), np.ones(5)]
tests = []


def flow_adjusted(y, flow, tag):
    f = np.array([flow[x] for x in ORDER8], float)
    X1 = np.c_[np.ones(8), f]; b1 = np.linalg.lstsq(X1, y, rcond=None)[0]; res = y - X1 @ b1
    # residual version: permutation of the labels on the flow residuals (statistic mean diff of residuals)
    o_r, p2_r, p1_r, _ = perm(res)
    # design-based version: label permutation of b_D in y ~ D + flow
    def bD(yv, lab):
        X = np.c_[np.ones(8), lab.astype(float), f]; return np.linalg.lstsq(X, yv, rcond=None)[0][1]
    b_obs = bD(y, Dv.astype(bool)); null = np.array([bD(y, m) for m in ALL56])
    p2_b = float(np.mean(np.abs(null) >= abs(b_obs) - 1e-12)); p1_b = float(np.mean(null >= b_obs - 1e-12))
    # test-inverted 95 % CI of exp(b_D): H0 b_D = log r  <=>  y - log(r) D has no label effect given flow
    ok = []
    for r in GRID:
        yr = y - np.log(r) * Dv
        bo = bD(yr, Dv.astype(bool)); nl = np.array([bD(yr, m) for m in ALL56])
        ok.append(np.mean(nl >= bo - 1e-12) > 0.025 and np.mean(nl <= bo + 1e-12) > 0.025)
    lo, hi = _connected(np.array(ok), np.exp(b_obs))
    X2 = np.c_[np.ones(8), Dv, f]; b2 = np.linalg.lstsq(X2, y, rcond=None)[0]
    return dict(test=f'within-day contrast, flow-adjusted ({tag})', days=8, ratio=float(np.exp(b_obs)), ci_lo=lo, ci_hi=hi,
                p2=p2_b, p1=p1_b, p2_resid_perm=p2_r, p1_resid_perm=p1_r, ratio_resid_perm=float(np.exp(o_r)), nperm=56,
                complete_sep=int(res[3:].min() > res[:3].max()), flow_coef_per10vph=float(10 * b2[2]),
                note='ratio = exp(b_D) in OLS C_d ~ D + flow; p2/p1 = label permutation of b_D (56); *_resid_perm = labels permuted on residuals '
                     'of C_d ~ flow (residual version); CI inverts the label-permutation test; complete_sep refers to the flow residuals')


t_unadj = full_test(y, log=True)
tests.append(dict(test='within-day contrast, unadjusted', days=8, **t_unadj, note='ratio of geometric means of unc/lead ratios'))
tests.append(flow_adjusted(y, FLOW, 'exact flows'))
tests.append(flow_adjusted(y, FLOW_R, 'rounded Table 1 flows'))
fv = np.array([FLOW[x] for x in ORDER8])
cor = dict(r_lead_flow=np.corrcoef([cr[x] for x in ORDER8], fv)[0, 1], r_unc_flow=np.corrcoef([fr[x] for x in ORDER8], fv)[0, 1],
           r_Cd_flow=np.corrcoef(y, fv)[0, 1], r_loglead_flow=np.corrcoef(np.log([cr[x] for x in ORDER8]), fv)[0, 1],
           r_logunc_flow=np.corrcoef(np.log([fr[x] for x in ORDER8]), fv)[0, 1])
# same adjustment for the log unconstrained rate itself (for completeness)
ylf = np.log([fr[x] for x in ORDER8])
tmp = flow_adjusted(ylf, FLOW, 'exact flows'); tmp['test'] = 'log unconstrained rate, flow-adjusted (exact flows)'; tests.append(tmp)
TT = pd.DataFrame(tests)
for k_, v_ in cor.items():
    TT[k_] = v_
print(TT.drop(columns=['note']).round(4).to_string(index=False))

# ---- (b) +-1-event perturbation
prow = []
base_n = {x: (fn[x], cn[x]) for x in ORDER8}


def contrast_vec(nmod):
    return np.array([np.log((nmod[x][0] / fk[x]) / (nmod[x][1] / ck[x])) for x in ORDER8])


for x in ORDER8:
    for df_ in (-1, 0, 1):
        for dl in (-1, 0, 1):
            nf, nl = fn[x] + df_, cn[x] + dl
            if nf <= 0 or nl <= 0:          # log contrast undefined (80/60 has 2 leader-driven changes: -1 still allowed)
                continue
            nm = dict(base_n); nm[x] = (nf, nl)
            v = contrast_vec(nm); o, p2, p1, _ = perm(v)
            prow.append(dict(scenario='one day', day=x, cfg=CFG[x], d_unc=df_, d_lead=dl, C_d=float(np.log((nf / fk[x]) / (nl / ck[x]))),
                             ratio_geo=float(np.exp(o)), p2=p2, p1=p1, complete_sep=int(v[3:].min() > v[:3].max()),
                             margin=float(v[3:].min() - v[:3].max())))
P1 = pd.DataFrame(prow)
# adverse direction: same-limit days pushed up (+1 unc, -1 lead), differentiated days pushed down (-1 unc, +1 lead)
adv = {x: ((fn[x] + 1, cn[x] - 1) if x in SAME else (fn[x] - 1, cn[x] + 1)) for x in ORDER8}
pair_rows = []
for a in ORDER8:
    for b in ORDER8:
        if a >= b:
            continue
        nm = dict(base_n); nm[a] = adv[a]; nm[b] = adv[b]
        v = contrast_vec(nm); o, p2, p1, _ = perm(v)
        pair_rows.append(dict(scenario='two days, adverse', day=a + ' & ' + b, cfg=CFG[a] + ' & ' + CFG[b], ratio_geo=float(np.exp(o)), p2=p2, p1=p1,
                              complete_sep=int(v[3:].min() > v[:3].max()), margin=float(v[3:].min() - v[:3].max())))
P2 = pd.DataFrame(pair_rows)
v = contrast_vec(adv); o, p2, p1, _ = perm(v)
P3 = pd.DataFrame([dict(scenario='all eight days, adverse', day='all', cfg='all', ratio_geo=float(np.exp(o)), p2=p2, p1=p1,
                        complete_sep=int(v[3:].min() > v[:3].max()), margin=float(v[3:].min() - v[:3].max()))])
PP = pd.concat([P1, P2, P3], ignore_index=True); out(PP, 'contrast_perturbation.csv')
PS = pd.DataFrame([
    dict(item='one day at a time, any of 8 (+-1, +-1) changes on one day (%d scenarios)' % len(P1),
         separation_kept=f'{int(P1.complete_sep.sum())} of {len(P1)}', max_p2=P1.p2.max(), max_p1=P1.p1.max(),
         min_ratio_geo=P1.ratio_geo.min(), max_ratio_geo=P1.ratio_geo.max()),
    dict(item='two days at once, adverse direction (%d pairs)' % len(P2), separation_kept=f'{int(P2.complete_sep.sum())} of {len(P2)}',
         max_p2=P2.p2.max(), max_p1=P2.p1.max(), min_ratio_geo=P2.ratio_geo.min(), max_ratio_geo=P2.ratio_geo.max(),
         pairs_breaking=';'.join(P2[P2.complete_sep == 0].cfg)),
    dict(item='all eight days at once, adverse direction', separation_kept=f'{int(P3.complete_sep.iloc[0])} of 1', max_p2=P3.p2.iloc[0],
         max_p1=P3.p1.iloc[0], min_ratio_geo=P3.ratio_geo.iloc[0], max_ratio_geo=P3.ratio_geo.iloc[0])])
out(PS, 'contrast_perturbation_summary.csv'); print(PS.round(4).to_string(index=False))

# ================================================================================================ 2. log scale, Holm adjustment
F5T = pd.read_csv(os.path.join(DATF, 'rates_by_type_tests.csv'))
TYPES5 = [(0, 'IN_free', 'car', 'Inward, unconstrained'), (0, 'IN_overtake', 'car', 'Inward, leader-driven'), (0, 'OUT_return', 'car', 'Outward, return'),
          (0, 'OUT_keepright', 'car', 'Outward, keep-right'), (1, 'IN_free', 'heavy', 'Inward, unconstrained'),
          (1, 'IN_overtake', 'heavy', 'Inward, leader-driven')]
hrows = []
for tr, T, cls, nm in TYPES5:
    n_, k_, r_ = rates(D, tr, T)
    v = np.array([r_[x] for x in ORDER8]); assert (v > 0).all()
    ta = full_test(v); tl = full_test(np.log(v), log=True)
    ref = F5T[(F5T.cls == ('car' if tr == 0 else 'truck')) & (F5T.type == T)].iloc[0]
    assert abs(ref.ratio - ta['ratio']) < 1e-9 and abs(ref.p2 - ta['p2']) < 1e-9
    hrows.append(dict(cls=cls, type=T, name=nm, n_same=int(sum(n_[x] for x in SAME)), n_diff=int(sum(n_[x] for x in DIFF)),
                      ratio_arith=ta['ratio'], ci_arith_lo=ta['ci_lo'], ci_arith_hi=ta['ci_hi'], p2_arith=ta['p2'], p1_arith=ta['p1'],
                      p2_rank=ta['p2_rank'], ratio_geo=tl['ratio'], ci_geo_lo=tl['ci_lo'], ci_geo_hi=tl['ci_hi'], p2_log=tl['p2'], p1_log=tl['p1'],
                      complete_sep=ta['complete_sep']))
H = pd.DataFrame(hrows)
for c in ('p2_arith', 'p1_arith', 'p2_log', 'p1_log', 'p2_rank'):
    H[c + '_holm6'] = holm(H[c].values)
H['note'] = ('Holm over the six type comparisons of Fig. 5 (the within-day contrast not included); one-sided = differentiated > same for every '
             'type; arith = ratio of arithmetic means of day rates (Fig. 5); log = difference of mean log day rates (ratio of geometric means)')
out(H, 'contrast_log_holm.csv')
print(H.drop(columns=['note']).round(4).to_string(index=False))
TT = pd.concat([TT, pd.DataFrame([dict(test='log unconstrained rate, unadjusted', days=8, **full_test(np.log([fr[x] for x in ORDER8]), log=True))])],
               ignore_index=True)
out(TT, 'contrast_tests.csv')

# ================================================================================================ 3. from 10:00 on
DH = day_table(E, S, E.typ.values, ST, hours=(10.0, 24.0))
an, ak, ar = rates(DH, 0, 'IN_free'); bn, bk, br = rates(DH, 0, 'IN_overtake')
arows = []
for x in ORDER8:
    lo, hi = garwood(an[x], ak[x])
    Cx = np.log((an[x] / ak[x]) / (bn[x] / bk[x])) if an[x] > 0 and bn[x] > 0 else np.nan
    R_, rlo, rhi, _ = ratio_exact(an[x], ak[x], bn[x], bk[x]) if bn[x] > 0 else (np.inf, np.nan, np.nan, np.nan)
    arows.append(dict(day=x, cfg=CFG[x], group='same-limit' if DEL[x] == 0 else 'differentiated',
                      session=DAY_META.set_index('day').session[x], n_unc=int(an[x]), km_unc=ak[x], rate_unc=ar[x], rate_unc_lo=lo, rate_unc_hi=hi,
                      n_lead=int(bn[x]), km_lead=bk[x], rate_lead=br[x], C_d=Cx, ratio_unc_lead=R_, ratio_ci_lo=rlo, ratio_ci_hi=rhi,
                      full_session_rate_unc=fr[x], full_session_C_d=C_d[x], eligible=int(ak[x] >= 5.0)))
AD = pd.DataFrame(arows); out(AD, 'contrast_from10_days.csv')
print(AD.drop(columns=['session']).round(3).to_string(index=False))
el = [x for x in ORDER8 if AD.set_index('day').eligible[x] == 1]
es, ed = [x for x in el if x in SAME], [x for x in el if x in DIFF]
atests = []
ADi = AD.set_index('day')
Cinf = {x: (ADi.C_d[x] if np.isfinite(ADi.C_d[x]) else (np.inf if (bn[x] == 0 and an[x] > 0) else np.nan)) for x in ORDER8}
Chalf = {x: np.log(((an[x] + 0.5) / ak[x]) / ((bn[x] + 0.5) / bk[x])) for x in ORDER8}
# rank test for the contrast: a day with unconstrained changes and no leader-driven change ranks highest (C_d = +inf)
vr = np.array([Cinf[x] for x in es + ed], float)
if not np.isnan(vr).any():
    _, p2r, p1r, npr = perm(np.where(np.isinf(vr), 1e9, vr), len(es), rank=True)
    atests.append(dict(measure='within-day contrast C_d, rank test (80/60: 3 unconstrained, 0 leader-driven changes -> ranked highest)',
                       days=len(el), n_same_days=len(es), days_used=';'.join(CFG[x] for x in es + ed), p2_rank=p2r, p1=np.nan,
                       p1_rank=perm(np.where(np.isinf(vr), 1e9, vr), len(es), rank=True)[2], nperm=npr,
                       complete_sep=int(vr[len(es):].min() > vr[:len(es)].max())))
# days with both counts > 0 only
fin = [x for x in es + ed if np.isfinite(ADi.C_d[x])]
fs, fd = [x for x in fin if x in SAME], [x for x in fin if x in DIFF]
for meas, dct, log, dlist, ns_ in (('unconstrained rate', ar, False, es + ed, len(es)), ('log unconstrained rate', {x: np.log(ar[x]) for x in el}, True, es + ed, len(es)),
                                    ('within-day contrast C_d, days with both counts > 0', {x: ADi.C_d[x] for x in fin}, True, fs + fd, len(fs)),
                                    ('within-day contrast C_d, +0.5 added to both counts on every day', Chalf, True, es + ed, len(es))):
    vals = np.array([dct[x] for x in dlist], float)
    t_ = full_test(vals, ns_, log=log)
    es_, ed_ = dlist[:ns_], dlist[ns_:]
    atests.append(dict(measure=meas, days=len(dlist), n_same_days=ns_, days_used=';'.join(CFG[x] for x in dlist), **t_,
                       same_mean=np.mean([ar[x] for x in es_]) if meas == 'unconstrained rate' else np.exp(np.mean(vals[:ns_])),
                       diff_mean=np.mean([ar[x] for x in ed_]) if meas == 'unconstrained rate' else np.exp(np.mean(vals[ns_:])),
                       n_unc_same=int(sum(an[x] for x in es_)), km_unc_same=sum(ak[x] for x in es_), n_unc_diff=int(sum(an[x] for x in ed_)),
                       km_unc_diff=sum(ak[x] for x in ed_), n_lead_same=int(sum(bn[x] for x in es_)), n_lead_diff=int(sum(bn[x] for x in ed_))))
AT = pd.DataFrame(atests)
AT['note'] = ('from 10:00 local time: exposure frames with hour >= 10 and changes whose t-3 s state is at or after 10:00; eligible days have '
              '>= 5 unconstrained car-km (90/70 excluded: 1.8 km); with 35 labellings the smallest one-sided p is 1/35 = 0.029, so a 95 % '
              'test-inverted interval cannot be bounded (ci 0-inf)')
out(AT, 'contrast_from10_tests.csv')
print(AT.drop(columns=['note']).round(4).to_string(index=False))

# ================================================================================================ 4. opposite (mirror) changes
ci = E[(E.truck >= 0) & (E.direction == 'in') & E.typ.isin(['IN_free', 'IN_overtake'])].copy()   # heavy rows = check of the detector
L = pd.read_pickle(os.path.join(CACHE, 'lateral_check.pkl'))
ci = ci.merge(L[['day', 'GlobalID', 't', 'lat_ok']], on=['day', 'GlobalID', 't'], how='left')
mrows = []
for x in ORDER8:
    ea = lc_events(x, grade=False)
    opp = ea[(ea.from_lane == 1) & (ea.to_lane == 2)]
    f = frames(x)[['GlobalID', 'Timestamp', 'PositionX', 'Lane', 'truck']]
    for r in ci[ci.day == x].itertuples():
        m = opp[(opp.GlobalID != r.GlobalID) & ((opp.timestamp_ms - r.t).abs() <= 3000) & ((opp.position_x_m - r.x).abs() <= 50)]
        mag = m[m.analysis_grade == 1]
        snap = f[(f.Timestamp == r.t) & (f.GlobalID != r.GlobalID) & (f.Lane == 1)]
        along = snap[(snap.PositionX - r.x).abs() <= 30]
        mrows.append(dict(day=x, cfg=CFG[x], group='same-limit' if DEL[x] == 0 else 'differentiated',
                          cls='heavy' if r.truck == 1 else 'car', GlobalID=r.GlobalID, t=r.t, x=r.x, typ=r.typ, lat_ok=r.lat_ok, n_mirror_any=len(m), n_mirror_ag=len(mag),
                          mirror_ids=';'.join(str(g) for g in m.GlobalID), mirror_dt_s=';'.join(f'{(tt - r.t) / 1000:.1f}' for tt in m.timestamp_ms),
                          mirror_dx_m=';'.join(f'{xx - r.x:.0f}' for xx in m.position_x_m),
                          mirror_truck=';'.join(str(int(v_)) for v_ in m.GlobalID.map(f.groupby('GlobalID').truck.first())),
                          n_lane1_alongside30_at_t=len(along)))
M = pd.DataFrame(mrows); out(M, 'contrast_mirror_events.csv')   # cars and heavy vehicles
srows = []
MH = M[M.cls == 'heavy']
for typ_, nm in (('IN_free', 'unconstrained'), ('IN_overtake', 'leader-driven')):
    z = MH[MH.typ == typ_]
    srows.append(dict(cls='heavy (check of the detector)', regime=nm, group='all eight days', n_changes=len(z), n_mirror_any=int((z.n_mirror_any > 0).sum()),
                      share_mirror_any=(z.n_mirror_any > 0).mean(), n_mirror_ag=int((z.n_mirror_ag > 0).sum()), share_mirror_ag=(z.n_mirror_ag > 0).mean(),
                      n_lane1_alongside=int((z.n_lane1_alongside30_at_t > 0).sum()), n_mirror_any_lat_ok=int(((z.n_mirror_any > 0) & (z.lat_ok == 1)).sum())))
M = M[M.cls == 'car']
for typ_, nm in (('IN_free', 'unconstrained'), ('IN_overtake', 'leader-driven')):
    for grp in ('same-limit', 'differentiated', 'eight days'):
        z = M[(M.typ == typ_) & ((M.group == grp) | (grp == 'eight days'))]
        srows.append(dict(cls='car', regime=nm, group=grp, n_changes=len(z), n_mirror_any=int((z.n_mirror_any > 0).sum()), share_mirror_any=(z.n_mirror_any > 0).mean(),
                          n_mirror_ag=int((z.n_mirror_ag > 0).sum()), share_mirror_ag=(z.n_mirror_ag > 0).mean(),
                          n_lane1_alongside=int((z.n_lane1_alongside30_at_t > 0).sum()),
                          n_mirror_any_lat_ok=int(((z.n_mirror_any > 0) & (z.lat_ok == 1)).sum())))
MS = pd.DataFrame(srows)
# day-level test of the mirror share among unconstrained changes
zf = M[M.typ == 'IN_free']
sh = (zf.n_mirror_any > 0).groupby(zf.day).mean()
o_, p2_, p1_, _ = perm([sh.get(x, np.nan) for x in ORDER8])
MS['p2_day_share_any_unconstrained'] = p2_
out(MS, 'contrast_mirror_summary.csv'); print(MS.round(3).to_string(index=False))
# headline without mirror changes
mt = []
for flag, lab in (('n_mirror_any', 'any recorded opposite change'), ('n_mirror_ag', 'analysis-grade opposite change')):
    drop = M[(M.typ == 'IN_free') & (M[flag] > 0)].groupby('day').size()
    drop_l = M[(M.typ == 'IN_overtake') & (M[flag] > 0)].groupby('day').size()
    fn2 = {x: fn[x] - int(drop.get(x, 0)) for x in ORDER8}; cn2 = {x: cn[x] - int(drop_l.get(x, 0)) for x in ORDER8}
    fr2 = {x: fn2[x] / fk[x] for x in ORDER8}
    v = np.array([fr2[x] for x in ORDER8])
    ta = full_test(v); tl = full_test(np.log(v), log=True)
    mt.append(dict(exclusion=lab, measure='unconstrained rate (ratio of day means)', n_removed_same=int(sum(drop.get(x, 0) for x in SAME)),
                   n_removed_diff=int(sum(drop.get(x, 0) for x in DIFF)),
                   same_mean=v[:3].mean(), diff_mean=v[3:].mean(), **ta,
                   day_rates=';'.join(f'{fr2[x]:.3f}' for x in ORDER8)))
    mt.append(dict(exclusion=lab, measure='log unconstrained rate (ratio of geometric means)', **tl))
    # contrast with mirror changes removed from both regimes
    if all(cn2[x] > 0 for x in ORDER8):
        cc = np.array([np.log((fn2[x] / fk[x]) / (cn2[x] / ck[x])) for x in ORDER8])
        mt.append(dict(exclusion=lab + ' (removed from both regimes)', measure='within-day contrast', n_removed_lead=int(sum(drop_l.get(x, 0) for x in ORDER8)),
                       **full_test(cc, log=True)))
MT = pd.DataFrame(mt); out(MT, 'contrast_mirror_tests.csv'); print(MT.round(4).to_string(index=False))
print('done')
