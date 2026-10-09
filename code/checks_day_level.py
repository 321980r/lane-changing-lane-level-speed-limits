# -*- coding: utf-8 -*-
"""Day-level checks of the car inward rates (Section 5.2):
1. recording order: correlation of the unconstrained rate, the within-day contrast ln(unconstrained / leader-driven) and the leader-driven
   rate with the recording order (and the calendar day) of the eight days; exact p over all 8! = 40,320 orderings;
2. days that share the inner-lane or the outer-lane limit: unconstrained rates with Garwood intervals and conditional-binomial intervals of
   two-day ratios; slopes on V_in, V_out and Delta with the exact pair permutation;
3. intermediate situation (a leader within 300 m that does not constrain): day rates, exact test and test-inverted interval.
Inputs: state tables and typed events (situations_lib, situations_rates); results/final/fig4_day_rates_data.csv (check).
Outputs (results/final): checks_order_check.csv, checks_order_check_days.csv, checks_matched_limits.csv, checks_matched_limits_slopes.csv,
checks_intermediate_days.csv, checks_intermediate_tests.csv."""
import sys
from scipy import stats
from checks_lib import *
from situations_lib import events, state
from situations_rates import classify_events as _classify_s1, exposure, day_table

pd.set_option('display.width', 250)

# ------------------------------------------------------------------------------------------------ rates (fig4 state machine)
E = events(); S = state()
typ = _classify_s1(E, k=3)
org = np.where(E.direction.values == 'in', 2, 1)
bad = E['s3_Lane'].notna().values & (E['s3_Lane'].values != org)
typ = typ.copy(); typ[bad] = 'U'                     # lane-consistency rule of rates_data (no-op at t-3)
ST = exposure(S)
D = day_table(E, S, typ, ST)
car = D[D.truck == 0]


def rd(T):
    z = car[car.type == T].set_index('day')
    return z.rate.to_dict(), z.n.to_dict(), z.km.to_dict()


fr, fn, fk = rd('IN_free'); cr, cn, ck = rd('IN_overtake'); mr, mn, mk = rd('IN_mid')
# reproduce fig4 numbers
F4 = pd.read_csv(os.path.join(DATF, 'fig4_day_rates_data.csv'))
f4 = F4[F4.regime == 'free'].set_index('day').rate
assert max(abs(f4[x] - fr[x]) for x in ORDER8) < 1e-9, 'free rates differ from fig4'
l4 = F4[F4.regime == 'leader-driven'].set_index('day').rate
assert max(abs(l4[x] - cr[x]) for x in ORDER8) < 1e-9
logc = {x: np.log(fr[x] / cr[x]) for x in ORDER8}

# ================================================================================================ 1. ORDER CHECK
PERM8 = np.array(list(permutations(range(8))), dtype=np.int8)


def corr_perm(x, y, P):
    """Pearson r of (x, y) and exact two-sided / one-sided p over all orderings of x (rows of P index x)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    X = x[P]; Xc = X - X.mean(1, keepdims=True); yc = y - y.mean()
    rr = (Xc * yc).sum(1) / np.sqrt((Xc ** 2).sum(1) * (yc ** 2).sum())
    xc = x - x.mean(); r = float((xc * yc).sum() / np.sqrt((xc ** 2).sum() * (yc ** 2).sum()))
    return r, float(np.mean(np.abs(rr) >= abs(r) - 1e-12)), float(np.mean(rr <= r + 1e-12)), float(np.mean(rr >= r - 1e-12))


rank8 = np.arange(1, 9)
cal8 = np.array([(DATE[d] - DATE[REC_ORDER[0]]).days for d in REC_ORDER], float)
grp8 = np.array([int(DEL[d] > 0) for d in REC_ORDER])
rows, drows = [], []
series = {'unconstrained rate': fr, 'ln(unconstrained / leader-driven)': logc, 'leader-driven rate': cr}
for name, vals in series.items():
    y = np.array([vals[d] for d in REC_ORDER], float)
    # deviation from own group mean (same-limit / differentiated)
    ydev = y - np.where(grp8 == 1, y[grp8 == 1].mean(), y[grp8 == 0].mean())
    for xname, x in (('recording order (1-8)', rank8), ('calendar day (days since 28 Jul)', cal8)):
        r, p2, plo, phi = corr_perm(x, y, PERM8)
        rs = stats.spearmanr(x, y).correlation
        rS, p2S, _, _ = corr_perm(pd.Series(x).rank().values, pd.Series(y).rank().values, PERM8)
        rd_, p2d, _, _ = corr_perm(x, ydev, PERM8)
        rows.append(dict(series=name, order_measure=xname, n_days=8, pearson_r=r, p_two_exact=p2, p_one_neg=plo, p_one_pos=phi,
                         spearman_rho=rs, spearman_p_two_exact=p2S, r_within_group=rd_, r_within_group_p_two=p2d, n_orderings=len(PERM8)))
OC = pd.DataFrame(rows); p_oc = out(OC, 'checks_order_check.csv')
for i, d in enumerate(REC_ORDER):
    drows.append(dict(rec_order=i + 1, day=d, date=DATE[d].strftime('%d %b'),
                      cfg=CFG[d], delta=DEL[d], group=('differentiated' if DEL[d] > 0 else 'same-limit'),
                      n_free=int(fn[d]), km_free=fk[d], rate_free=fr[d], n_leader=int(cn[d]), km_leader=ck[d], rate_leader=cr[d], ln_contrast=logc[d]))
OD = pd.DataFrame(drows); p_od = out(OD, 'checks_order_check_days.csv')
print(OC.round(3).to_string(index=False)); print(OD.round(3).to_string(index=False))

# ================================================================================================ 2. MATCHED-LIMIT COMPARISONS


def pois_ratio_ci(n1, k1, n0, k0):
    """ratio of two Poisson rates (n1/k1)/(n0/k0) with the exact conditional (Clopper-Pearson on n1/(n1+n0)) 95 % interval"""
    N = n1 + n0
    if N == 0:
        return np.nan, np.nan, np.nan
    lo = stats.beta.ppf(0.025, n1, N - n1 + 1) if n1 > 0 else 0.0
    hi = stats.beta.ppf(0.975, n1 + 1, N - n1) if n1 < N else 1.0
    f = k0 / k1
    r = (n1 / k1) / (n0 / k0) if n0 > 0 else np.inf
    return r, lo / (1 - lo) * f, (hi / (1 - hi) * f if hi < 1 else np.inf)


D_ = {c: d for d, c in CFG.items()}
comps = [('shared inner-lane limit 100', '100/100', ['100/80', '100/70']),
         ('shared inner-lane limit 80', '80/80', ['80/70', '80/60']),
         ('shared outer-lane limit 60', '60/60', ['80/60']),
         ('shared outer-lane limit 80', '80/80', ['100/80']),
         ('shared outer-lane limit 70 (no same-limit day)', '80/70', ['90/70', '100/70'])]
rows = []
for nm, ref, others in comps:
    dref = D_[ref]
    for o in others:
        do = D_[o]
        r, lo, hi = pois_ratio_ci(fn[do], fk[do], fn[dref], fk[dref])
        a = garwood(fn[dref], fk[dref]); b = garwood(fn[do], fk[do])
        rows.append(dict(comparison=nm, reference=CFG[dref], ref_delta=DEL[dref], ref_n=int(fn[dref]), ref_km=fk[dref],
                         ref_rate=fr[dref], ref_lo=a[0], ref_hi=a[1], other=o, other_delta=DEL[do], other_n=int(fn[do]), other_km=fk[do], other_rate=fr[do],
                         other_lo=b[0], other_hi=b[1], ratio=r, ratio_pois_lo=lo, ratio_pois_hi=hi,
                         ref_contrast=np.exp(logc[dref]), other_contrast=np.exp(logc[do])))
MC = pd.DataFrame(rows); p_mc = out(MC, 'checks_matched_limits.csv')
print(MC.round(3).to_string(index=False))

# slopes over the 8 days with the exact pair permutation
y = np.array([fr[d] for d in ORDER8]); yl = np.array([logc[d] for d in ORDER8])
rows = []
for cov in ('vin', 'vout', 'delta', 'diff'):
    x = np.array([{'vin': VIN[d], 'vout': VOUT[d], 'delta': DEL[d], 'diff': int(DEL[d] > 0)}[cov] for d in ORDER8], float)
    for yname, yy in (('unconstrained rate', y), ('ln(unconstrained / leader-driven)', yl)):
        r, p2, _, _ = corr_perm(x, yy, PERM8)
        b = np.polyfit(x, yy, 1)[0]
        rows.append(dict(series=yname, covariate=cov, slope=b, pearson_r=r, r2=r ** 2, p_two_pair_perm=p2, n_perm=len(PERM8),
                         note='(V_in, V_out) pairs reassigned to days (8! orderings); identical to relabelling for the diff indicator up to ties'))
# two-covariate check: rate ~ V_in + Delta (is there a V_in effect once Delta is held?)
X = np.column_stack([np.ones(8), [VIN[d] for d in ORDER8], [DEL[d] for d in ORDER8]]).astype(float)
bb = np.linalg.lstsq(X, y, rcond=None)[0]
rows.append(dict(series='unconstrained rate', covariate='vin | delta (OLS, 8 days)', slope=bb[1], pearson_r=np.nan, r2=np.nan, p_two_pair_perm=np.nan,
                 n_perm=np.nan, note=f'delta slope {bb[2]:.4f} per km/h; descriptive'))
X2 = np.column_stack([np.ones(8), [VOUT[d] for d in ORDER8], [int(DEL[d] > 0) for d in ORDER8]]).astype(float)
bb2 = np.linalg.lstsq(X2, y, rcond=None)[0]
rows.append(dict(series='unconstrained rate', covariate='vout | diff (OLS, 8 days)', slope=bb2[1], pearson_r=np.nan, r2=np.nan, p_two_pair_perm=np.nan,
                 n_perm=np.nan, note=f'diff step {bb2[2]:.4f} per km; descriptive'))
MS = pd.DataFrame(rows); p_ms = out(MS, 'checks_matched_limits_slopes.csv')
print(MS.round(4).to_string(index=False))

# ================================================================================================ 3. INTERMEDIATE REGIME
lead_r = {x: (mn[x] + cn[x]) / (mk[x] + ck[x]) for x in ORDER8}
rows = []
for x in ORDER8:
    lo, hi = garwood(mn[x], mk[x])
    rows.append(dict(day=x, cfg=CFG[x], delta=DEL[x], n_intermediate=int(mn[x]), km_intermediate=mk[x], rate_intermediate=mr[x],
                     ci_lo=lo, ci_hi=hi, n_leader_driven=int(cn[x]), km_leader_driven=ck[x], rate_leader_driven=cr[x],
                     n_leader_within300=int(mn[x] + cn[x]), km_leader_within300=mk[x] + ck[x], rate_leader_within300=lead_r[x],
                     n_unconstrained=int(fn[x]), km_unconstrained=fk[x], rate_unconstrained=fr[x]))
MI = pd.DataFrame(rows); p_mi = out(MI, 'checks_intermediate_days.csv')
rows = []
for nm, vals, nd, kd in (('intermediate (M)', mr, mn, mk), ('leader within 300 m (C + M)', lead_r, {x: mn[x] + cn[x] for x in ORDER8}, {x: mk[x] + ck[x] for x in ORDER8}),
                         ('leader-driven (C)', cr, cn, ck), ('unconstrained (F)', fr, fn, fk)):
    t8 = contrast_tests(vals, DAYS8, NS8)
    rows.append(dict(regime=nm, n_same=int(sum(nd[x] for x in ORDER8[:3])), km_same=sum(kd[x] for x in ORDER8[:3]),
                     n_diff=int(sum(nd[x] for x in ORDER8[3:])), km_diff=sum(kd[x] for x in ORDER8[3:]),
                     same_mean=t8['same_mean'], diff_mean=t8['diff_mean'], ratio=t8['ratio'], ci_lo=t8['ci_lo'], ci_hi=t8['ci_hi'],
                     p2_56=t8['p2'], p1_56=t8['p1'], p2_rank_56=t8['p2_rank']))
MT = pd.DataFrame(rows); p_mt = out(MT, 'checks_intermediate_tests.csv')
print(MI.round(3).to_string(index=False)); print(MT.round(3).to_string(index=False))
print('saved', p_oc, p_od, p_mc, p_ms, p_mi, p_mt)
