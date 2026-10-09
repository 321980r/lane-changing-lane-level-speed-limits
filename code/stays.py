# -*- coding: utf-8 -*-
"""Inner-lane stints of the eight days (definitions of overtaking_cycle.py; Section 5.3): Kaplan-Meier stays by set, numbers at risk,
rate ratios of returns (Poisson with day effects, vehicle-clustered errors), cars already in the inner lane at entry per day with exact
tests, and flags of heavy stints opened by an unconstrained inward change (from stays_checks.py). The entry group is left-truncated:
its curve is the stay given presence in the inner lane at 900 m, not a stay distribution from a lane change.
Inputs: _cache/work/stints.pkl, lateral_check.pkl, core_events.pkl; results/check_misc.csv; results/final/stays_checks_heavy_free.csv.
Outputs (results/final): stays_stint_sets.csv, stays_atrisk.csv, stays_rate_ratios.csv, stays_entry_days.csv,
stays_entry_tests.csv, stays_heavy_unconstrained_flags.csv."""
import os, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from plot_style import *
warnings.filterwarnings('ignore')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
out = lambda df, nm: df.to_csv(os.path.join(DATF, nm), index=False)

S = pd.read_pickle(os.path.join(CACHE, 'stints.pkl'))
L = pd.read_pickle(os.path.join(CACHE, 'lateral_check.pkl')).set_index(['day', 'GlobalID', 't'])
E = pd.read_pickle(os.path.join(CACHE, 'core_events.pkl')).set_index(['day', 'GlobalID', 't'])
S = S[S.truck >= 0].copy()
S['in_ok'] = [L.lat_ok.get((r.day, r.GlobalID, r.t_start), np.nan) if r.start == 'lc' else 1 for r in S.itertuples()]
S['out_ok'] = [L.lat_ok.get((r.day, r.GlobalID, r.t_end), np.nan) if r.ret == 1 else np.nan for r in S.itertuples()]
S['ret_c'] = ((S.ret == 1) & (S.out_ok == 1)).astype(int)
S['d'] = S.dist.clip(lower=1.0); S['km'] = S.d / 1000
S['grp'] = np.where(S.delta > 0, 'diff', 'same')
S['mtype0'] = [E.mtype0.get((r.day, r.GlobalID, r.t_start), 'NA') if r.start == 'lc' else 'entry' for r in S.itertuples()]
S['regime'] = np.where(S.mtype0.str.startswith('in_overtake'), 'leader', np.where(S.mtype0.str.startswith('in_free'), 'free', S.start))
SC = S[S.in_ok == 1].copy()
LC8 = SC[SC.start == 'lc']; SC8 = SC
SETS8 = {'heavy_ld': LC8[(LC8.truck == 1) & (LC8.regime == 'leader')], 'car_ld': LC8[(LC8.truck == 0) & (LC8.regime == 'leader')],
         'car_un': LC8[(LC8.truck == 0) & (LC8.regime == 'free')], 'car_entry': SC8[(SC8.truck == 0) & (SC8.start == 'entry')],
         'heavy_un': LC8[(LC8.truck == 1) & (LC8.regime == 'free')], 'heavy_lc_all': LC8[LC8.truck == 1], 'car_lc_all': LC8[LC8.truck == 0]}
for g in ('same', 'diff'):
    SETS8[f'car_entry_{g}'] = SETS8['car_entry'][SETS8['car_entry'].grp == g]
    SETS8[f'car_un_{g}'] = SETS8['car_un'][SETS8['car_un'].grp == g]
    SETS8[f'car_ld_{g}'] = SETS8['car_ld'][SETS8['car_ld'].grp == g]
    SETS8[f'heavy_ld_{g}'] = SETS8['heavy_ld'][SETS8['heavy_ld'].grp == g]


def km_curve(d, e):
    d = np.asarray(d, float); e = np.asarray(e, int)
    o = np.argsort(d, kind='mergesort'); d, e = d[o], e[o]
    S_, gw = 1.0, 0.0
    rows = [(0.0, 1.0, 1.0, 1.0, len(d))]
    for u in np.unique(d[e == 1]):
        n = int((d >= u).sum()); k = int(((d == u) & (e == 1)).sum())
        S_ *= 1 - k / n
        gw += k / (n * (n - k)) if n > k else 0.0
        if 0 < S_ < 1:
            se = np.sqrt(gw) / abs(np.log(S_))
            lo, hi = S_ ** np.exp(1.96 * se), S_ ** np.exp(-1.96 * se)
        else:
            lo, hi = S_, S_
        rows.append((u, S_, lo, hi, n))
    return pd.DataFrame(rows, columns=['dist', 'S', 'lo', 'hi', 'n_risk'])


def km_at(T, x):
    i = np.searchsorted(T.dist.values, x, side='right') - 1
    return T.S.values[i], T.lo.values[i], T.hi.values[i]


def rr_fit(q, y='ret_c'):
    q = q.assign(lkm=np.log(q.km), gid=q.day + '_' + q.GlobalID.astype(str))
    m = smf.glm(f'{y} ~ truck + day', data=q, family=sm.families.Poisson(), offset=q.lkm).fit(
        cov_type='cluster', cov_kwds={'groups': pd.factorize(q.gid)[0]})
    return float(np.exp(m.params['truck'])), *[float(v) for v in np.exp(m.conf_int().loc['truck'].values)], float(m.pvalues['truck'])


GRID = [0, 200, 400, 600, 800]
rows, arows = [], []
for nm, D in SETS8.items():
    T = km_curve(D.d, D.ret_c)
    r = dict(set=nm, days='8', n_stints=len(D), n_days=D.day.nunique(), returns=int(D.ret_c.sum()), km=D.km.sum(),
             returns_per_km=D.ret_c.sum() / D.km.sum(), share_censored=1 - D.ret_c.mean(), median_dist_followed=D.dist.median())
    for h in (200, 400, 600):
        s, lo, hi = km_at(T, h)
        r[f'P_back_{h}'] = 1 - s; r[f'P_back_{h}_lo'] = 1 - hi; r[f'P_back_{h}_hi'] = 1 - lo
    r['last_dist_with_10_at_risk'] = float(np.sort(D.d.values)[-10]) if len(D) >= 10 else np.nan
    rows.append(r)
    for x in GRID:
        arows.append(dict(set=nm, dist=x, n_risk=int((D.d >= x).sum())))
ST = pd.DataFrame(rows); out(ST, 'stays_stint_sets.csv')
AR = pd.DataFrame(arows).pivot(index='set', columns='dist', values='n_risk').reset_index(); out(AR, 'stays_atrisk.csv')
print(ST.round(3).to_string(index=False)); print(AR.to_string(index=False))

# rate ratios (Poisson, day fixed effects, offset log km in the inner lane, clusters = vehicle-day); 8 days
RR = []
for tag, lc, sc in (('8 days', LC8, SC8),):
    hl = lc[(lc.truck == 1) & (lc.regime == 'leader')]; cl = lc[(lc.truck == 0) & (lc.regime == 'leader')]
    cu = lc[(lc.truck == 0) & (lc.regime == 'free')]; ce = sc[(sc.truck == 0) & (sc.start == 'entry')]
    for nm, q in (('heavy / car, leader-driven stints', pd.concat([hl, cl])),
                  ('heavy / car, all lateral-confirmed inward changes', lc),
                  ('car leader-driven / car unconstrained', lc[lc.truck == 0].assign(truck=lambda x: (x.regime == 'leader').astype(int))),
                  ('heavy unconstrained / heavy leader-driven', lc[lc.truck == 1].assign(truck=lambda x: (x.regime == 'free').astype(int))),
                  ('car unconstrained / car at entry', pd.concat([cu.assign(truck=1), ce.assign(truck=0)])),
                  ('car leader-driven / car at entry', pd.concat([cl.assign(truck=1), ce.assign(truck=0)]))):
        est, lo, hi, p = rr_fit(q)
        RR.append(dict(comparison=nm, days=tag, rate_ratio=est, ci_lo=lo, ci_hi=hi, p_wald=p, n_stints=len(q), returns=int(q.ret_c.sum()),
                       note='Poisson GLM ret ~ group + day, offset log km, vehicle-day clustered SE'))
# at-entry cars: differentiated vs same-limit (pooled Poisson with the group indicator; no day FE possible)
q = SETS8['car_entry'].assign(truck=lambda x: (x.grp == 'diff').astype(int), lkm=lambda x: np.log(x.km))
m = smf.glm('ret_c ~ truck', data=q, family=sm.families.Poisson(), offset=q.lkm).fit(
    cov_type='cluster', cov_kwds={'groups': pd.factorize(q.day)[0]})
RR.append(dict(comparison='car at entry, differentiated / same-limit (pooled Poisson, day-clustered SE, 8 clusters: indicative only)',
               days='8 days', rate_ratio=float(np.exp(m.params['truck'])), ci_lo=float(np.exp(m.conf_int().loc['truck'][0])),
               ci_hi=float(np.exp(m.conf_int().loc['truck'][1])), p_wald=float(m.pvalues['truck']), n_stints=len(q), returns=int(q.ret_c.sum())))
RRd = pd.DataFrame(RR); out(RRd, 'stays_rate_ratios.csv'); print(RRd.round(4).to_string(index=False))

# cars already in the inner lane at entry: day values and exact day-level tests
ent = SC[(SC.truck == 0) & (SC.start == 'entry')]
drow = []
for x in ORDER8:
    b = ent[ent.day == x]; T = km_curve(b.d, b.ret_c); s, lo, hi = km_at(T, 600)
    drow.append(dict(day=x, cfg=DAY_META.set_index('day').label[x], delta=DAY_META.set_index('day').delta[x], n_stints=len(b),
                     leaves=int(b.ret_c.sum()), km=b.km.sum(), leave_rate_per_km=b.ret_c.sum() / b.km.sum(), P_leave_600=1 - s,
                     P_leave_600_lo=1 - hi, P_leave_600_hi=1 - lo, n_risk_600=int((b.d >= 600).sum())))
EDY = pd.DataFrame(drow); out(EDY, 'stays_entry_days.csv'); print(EDY.round(4).to_string(index=False))
trow = []
for meas in ('leave_rate_per_km', 'P_leave_600'):
    v8 = EDY.set_index('day').reindex(ORDER8)[meas].values
    obs, p2, p1 = perm8(v8)
    trow.append(dict(measure=meas, same_mean_of_days=v8[:3].mean(), diff_mean_of_days=v8[3:8].mean(), diff_minus_same=obs,
                     ratio_of_means=v8[3:8].mean() / v8[:3].mean(), p2_8days=p2, p1_8days=p1, p2_rank_8days=perm_rank8(v8),
                     day_values=';'.join(f'{v:.4f}' for v in v8)))
for g, dd in (('same-limit', ORDER8[:3]), ('differentiated', ORDER8[3:])):
    b = ent[ent.day.isin(dd)]; T = km_curve(b.d, b.ret_c); s, lo, hi = km_at(T, 600)
    trow.append(dict(measure=f'pooled {g}', n_stints=len(b), leaves=int(b.ret_c.sum()), km=b.km.sum(), pooled_rate_per_km=b.ret_c.sum() / b.km.sum(),
                     KM_P_leave_600=1 - s, KM_P_leave_600_lo=1 - hi, KM_P_leave_600_hi=1 - lo, n_risk_600=int((b.d >= 600).sum())))
b = ent[ent.day.isin(ORDER8)]; T = km_curve(b.d, b.ret_c); s, lo, hi = km_at(T, 600)
trow.append(dict(measure='pooled eight days', n_stints=len(b), leaves=int(b.ret_c.sum()), km=b.km.sum(), pooled_rate_per_km=b.ret_c.sum() / b.km.sum(),
                 KM_P_leave_600=1 - s, KM_P_leave_600_lo=1 - hi, KM_P_leave_600_hi=1 - lo, n_risk_600=int((b.d >= 600).sum())))
# the same numbers as defined in check_misc.py (check_misc.csv: entry runs from check_stint_seq_runs_lat.csv, 1,172 runs, 95 leaves)
MI = pd.read_csv(os.path.join(RES, 'check_misc.csv')).set_index('check')
for k in ('entry_run_leave_per_km', 'entry_run_leave_KM600'):
    r = MI.loc[k]
    trow.append(dict(measure=f'check_misc.csv {k} (source of the manuscript values)', same_mean_of_days=r.same_mean, diff_mean_of_days=r.diff_mean,
                     p2_8days=r.p2_8, p1_8days=r.p1_8, day_values=r.day_values,
                     note='verifier definition: entry runs of check_stint_seq_runs_lat.csv, leave = next change to lane 2 that is not pseudo'))
ET = pd.DataFrame(trow)
ET['left_truncation'] = 'clock starts at window entry (x about 900 m); vehicles already in the inner lane upstream: delayed entry, length-biased'
out(ET, 'stays_entry_tests.csv'); print(ET.drop(columns=['left_truncation']).round(4).to_string(index=False))

# heavy unconstrained stints on the eight days: label-swap flags from stays_checks.py
HF = pd.read_csv(os.path.join(DATF, 'stays_checks_heavy_free.csv'))
hf = HF[HF.regime == 'free']
flag = (hf.n_mirror_change > 0) | ((hf.n_lane2_alongside30 > 0) & (hf.n_pass == 0))
hl = HF[HF.regime == 'leader']
HFS = pd.DataFrame([dict(set='heavy unconstrained stints, 8 days', n=len(hf), mirror_change=int((hf.n_mirror_change > 0).sum()),
                         flagged_mirror_or_alongside_without_pass=int(flag.sum()), with_pass_during_stint=int((hf.n_pass > 0).sum())),
                    dict(set='heavy leader-driven stints, 8 days', n=len(hl), mirror_change=int((hl.n_mirror_change > 0).sum()),
                         flagged_mirror_or_alongside_without_pass=int(((hl.n_mirror_change > 0) | ((hl.n_lane2_alongside30 > 0) & (hl.n_pass == 0))).sum()),
                         with_pass_during_stint=int((hl.n_pass > 0).sum()))])
out(HFS, 'stays_heavy_unconstrained_flags.csv'); print(HFS.to_string(index=False))
