# -*- coding: utf-8 -*-
"""Lane-change rates of cars by situation and day, and the day-level comparisons of Figures 3 and 4 and Section 5.2.
Situations 3 s before the change (situations_state.py, situations_lib.py, situations_rates.py): in the outer lane C = leader-driven (headway < 4 s, or gap < 150 m
and closing > 3 km/h), F = unconstrained (no outer-lane vehicle within 300 m ahead, look-ahead inside the section), M = leader within
300 m but not constraining, N = look-ahead truncated; in the inner lane R = return, Y = yield, B = blocked, K = keep-right. Rate =
changes in situation T / distance driven in situation T; a change is typed only if the vehicle is in its origin lane at t-k.
The day is the unit: exact tests over the 56 labellings of three of the eight days as same-limit, and 95 % intervals of the
differentiated/same ratio by inverting them.
Parts: rates by distance to the leader (Figure 3); unconstrained and leader-driven rates with Garwood intervals and the within-day
contrast (Figure 4); all types of lane change; alternative specifications (look-ahead 150/200/500 m, t-5 s, lateral-confirmed changes,
x >= 1,050 m, hour of day, each day left out); situation of the car inward changes and share of heavy leaders.
Inputs: state tables and typed events; results/sit_day_rates.csv, results/lateral_check.csv.
Outputs (results/final): rates_reproduction_check.csv, fig3_rates_by_distance_data.csv, fig3_rates_by_distance_days.csv, fig4_day_rates_data.csv,
fig4_day_rates_tests.csv, rates_by_type_data.csv, rates_by_type_tests.csv, rates_robustness_data.csv, rates_robustness_days.csv,
rates_inward_context_data.csv, rates_heavy_leader_days.csv."""
import os, sys
from itertools import combinations
from scipy import stats
from plot_style import *          # noqa  (DATF, DAY_META, perm8, perm_rank8 ...)
from situations_lib import (events, state, event_states, frame_states, perm_label, perm_tests, quasi_poisson_ratio, ORDER8 as S1_O8,
                    CFG, DEL)
from situations_rates import classify_events as _classify_s1, exposure, day_table

assert S1_O8 == ORDER8


def classify_events(E_, k=3, **kw):
    """S1 typology plus a lane-consistency rule: an event is typed only if the vehicle is in its origin lane at t-k
    (lane 2 for inward, lane 1 for outward); otherwise 'U'. No-op at t-3 (all events are in the origin lane); at t-5 it removes
    one car inward event (80/70) that was still in lane 1 at t-5 and was typed 'free' by S1 (as check_rates.py, which requires lane 2)."""
    typ = _classify_s1(E_, k=k, **kw)
    org = np.where(E_.direction.values == 'in', 2, 1)
    bad = E_[f's{k}_Lane'].notna().values & (E_[f's{k}_Lane'].values != org)
    typ = typ.copy(); typ[bad] = 'U'
    return typ
pd.set_option('display.width', 250)
out = lambda df, nm: df.to_csv(os.path.join(DATF, nm), index=False)


# ------------------------------------------------------------------------------------------------ generic day-level helpers
def perm_generic(v, n_same, stat='mean'):
    """exact relabelling test; first n_same values are same-limit days. stat 'mean' = mean(diff) - mean(same); 'rank' = same on ranks."""
    v = np.asarray(v, float)
    if stat == 'rank':
        v = pd.Series(v).rank().values
    k = len(v); obs = v[n_same:].mean() - v[:n_same].mean(); null = []
    for S_ in combinations(range(k), n_same):
        m = np.ones(k, bool); m[list(S_)] = False
        null.append(v[m].mean() - v[~m].mean())
    null = np.array(null)
    return float(np.mean(np.abs(null) >= abs(obs) - 1e-12)), float(np.mean(null >= obs - 1e-12)), len(null)


def _one_sided(v, n_same):
    k = len(v); obs = v[n_same:].mean() - v[:n_same].mean(); null = []
    for S_ in combinations(range(k), n_same):
        m = np.ones(k, bool); m[list(S_)] = False
        null.append(v[m].mean() - v[~m].mean())
    null = np.array(null)
    return float(np.mean(null >= obs - 1e-12)), float(np.mean(null <= obs + 1e-12))


def ratio_ci(v, n_same, log=False, grid=np.exp(np.linspace(np.log(0.02), np.log(50), 1200))):
    """equal-tailed test-inverted 95% interval: r is kept when both exact one-sided p-values of the adjusted data exceed 0.025
    (differentiated-day values divided by r, or log values shifted by log r). The connected set around the estimate is returned.
    (The |mean difference| two-sided test is not monotone in r with 3 vs 5 days, hence the two one-sided tests.)"""
    v = np.asarray(v, float); ok = []
    for r in grid:
        w = v.copy()
        if log:
            w[n_same:] = w[n_same:] - np.log(r)
        else:
            w[n_same:] = w[n_same:] / r
        pu, pl = _one_sided(w, n_same)
        ok.append(pu > 0.025 and pl > 0.025)
    ok = np.array(ok)
    est = np.exp(v[n_same:].mean() - v[:n_same].mean()) if log else v[n_same:].mean() / v[:n_same].mean()
    i0 = int(np.argmin(np.abs(np.log(grid) - np.log(est))))
    if not ok[i0]:
        return np.nan, np.nan
    lo = i0
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    hi = i0
    while hi < len(grid) - 1 and ok[hi + 1]:
        hi += 1
    return (grid[lo] if lo > 0 else 0.0), (grid[hi] if hi < len(grid) - 1 else np.inf)


def garwood(n, km):
    lo = stats.chi2.ppf(0.025, 2 * n) / 2 / km if n > 0 else 0.0
    hi = stats.chi2.ppf(0.975, 2 * (n + 1)) / 2 / km
    return lo, hi


def group_days(days_main, drop=None):
    """ordered day list (same-limit days first) and n_same for a variant."""
    d = [x for x in days_main if x != drop]
    same = [x for x in d if DEL[x] == 0]; diff = [x for x in d if DEL[x] > 0]
    return same + diff, len(same)


def contrast_tests(vals_by_day, days, n_same, log=False):
    v = np.array([vals_by_day[x] for x in days], float)
    p2, p1, nperm = perm_generic(v, n_same)
    pr = perm_generic(v, n_same, 'rank')[0]
    if log:
        ratio = float(np.exp(v[n_same:].mean() - v[:n_same].mean()))
    else:
        ratio = float(v[n_same:].mean() / v[:n_same].mean())
    lo, hi = ratio_ci(v, n_same, log=log)
    return dict(ratio=ratio, ci_lo=lo, ci_hi=hi, p2=p2, p1=p1, p2_rank=pr, nperm=nperm,
                complete_sep=int(v[n_same:].min() > v[:n_same].max()))


# ------------------------------------------------------------------------------------------------ load and type (primary)
E = events(); S = state()
E['typ'] = classify_events(E, k=3)
E['st3'] = event_states(E, k=3)
ST = exposure(S)
D = day_table(E, S, E.typ.values, ST)
car = D[D.truck == 0]

# ------------------------------------------------------------------------------------------------ reproduction check vs S1 CSVs
ref = pd.read_csv(os.path.join(RES, 'sit_day_rates.csv'))
mm = D.merge(ref, on=['day', 'truck', 'type'], suffixes=('', '_ref'))
dev_n = int((mm.n != mm.n_ref).sum()); dev_km = float(np.nanmax(np.abs(mm.km - mm.km_ref)))
chk = [dict(item='day_table n mismatches vs sit_day_rates.csv', value=dev_n, reference=0),
       dict(item='day_table max |km diff| vs sit_day_rates.csv', value=dev_km, reference=0)]


def rate_dict(tr, T, Dx=D):
    z = Dx[(Dx.truck == tr) & (Dx.type == T)].set_index('day')
    return z.rate.to_dict(), z.n.to_dict(), z.km.to_dict()


fr, fn, fk = rate_dict(0, 'IN_free'); cr, cn, ck = rate_dict(0, 'IN_overtake')
days8, ns8 = group_days(ORDER8)
F1 = contrast_tests(fr, days8, ns8)
F3 = contrast_tests(cr, days8, ns8)
logc = {x: np.log(fr[x] / cr[x]) for x in ORDER8}
F2 = contrast_tests(logc, days8, ns8, log=True)
chk += [dict(item='F1 free same mean', value=np.mean([fr[x] for x in ORDER8[:3]]), reference=0.107),
        dict(item='F1 free diff mean', value=np.mean([fr[x] for x in ORDER8[3:]]), reference=0.269),
        dict(item='F1 ratio', value=F1['ratio'], reference=2.51), dict(item='F1 p two-sided', value=F1['p2'], reference=0.071),
        dict(item='F1 p rank', value=F1['p2_rank'], reference=0.036), dict(item='F1 p one-sided', value=F1['p1'], reference=0.018),
        dict(item='F2 within-day log(free/leader-driven) p (S1 state C)', value=F2['p2'], reference=0.018),
        dict(item='F3 leader-driven same mean', value=np.mean([cr[x] for x in ORDER8[:3]]), reference=1.04),
        dict(item='F3 leader-driven diff mean', value=np.mean([cr[x] for x in ORDER8[3:]]), reference=1.01),
        dict(item='F3 ratio', value=F3['ratio'], reference=0.97), dict(item='F3 p', value=F3['p2'], reference=0.93)]
for x in ORDER8:
    chk.append(dict(item=f'F1 free rate {CFG[x]}', value=fr[x], reference=np.nan))
CHK = pd.DataFrame(chk); out(CHK, "rates_reproduction_check.csv"); print(CHK.round(4).to_string(index=False))

# ================================================================================================ fig4: day rates of the two regimes
rows = []
for x in ORDER8:
    for T, rd_, nd, kd in (('free', fr, fn, fk), ('leader-driven', cr, cn, ck)):
        lo, hi = garwood(nd[x], kd[x])
        rows.append(dict(day=x, cfg=CFG[x], label=DAY_META.set_index('day').label[x], delta=DEL[x], regime=T,
                         n=int(nd[x]), car_km=kd[x], rate=rd_[x], ci_lo=lo, ci_hi=hi))
F4 = pd.DataFrame(rows); out(F4, 'fig4_day_rates_data.csv')
T4 = pd.DataFrame([dict(test='free inward rate', days=8, **F1),
                   dict(test='leader-driven inward rate', days=8, **F3),
                   dict(test='within-day log(free/leader-driven)', days=8, **F2)])
for g, dd in (('same', ORDER8[:3]), ('diff', ORDER8[3:])):
    T4[f'free_mean_{g}'] = np.mean([fr[x] for x in dd]); T4[f'leader_mean_{g}'] = np.mean([cr[x] for x in dd])
out(T4, 'fig4_day_rates_tests.csv'); print(T4.round(4).to_string(index=False))

# ================================================================================================ rates by type: which component responds
TYPES5 = [(0, 'IN_free', 'car', 'Inward, free'), (0, 'IN_overtake', 'car', 'Inward, leader-driven'), (0, 'OUT_return', 'car', 'Outward, return'),
          (0, 'OUT_keepright', 'car', 'Outward, keep-right'), (1, 'IN_free', 'truck', 'Inward, free'), (1, 'IN_overtake', 'truck', 'Inward, leader-driven')]
rows, trows = [], []
for tr, T, cls, nm in TYPES5:
    r, n, k = rate_dict(tr, T)
    base = np.mean([r[x] for x in ORDER8[:3]])
    t8 = contrast_tests(r, days8, ns8)
    for x in ORDER8:
        rows.append(dict(cls=cls, type=T, name=nm, day=x, cfg=CFG[x], delta=DEL[x], n=int(n[x]), km=k[x], rate=r[x], rel=r[x] / base))
    trows.append(dict(cls=cls, type=T, name=nm, same_mean=base, diff_mean=np.mean([r[x] for x in ORDER8[3:]]), n_same=int(sum(n[x] for x in ORDER8[:3])),
                      n_diff=int(sum(n[x] for x in ORDER8[3:])), km_same=sum(k[x] for x in ORDER8[:3]), km_diff=sum(k[x] for x in ORDER8[3:]),
                      **t8))
# within-day contrast (cars): exp(log(free/leader-driven) - mean over same-limit days of the log)
gbase = np.mean([logc[x] for x in ORDER8[:3]])
for x in ORDER8:
    rows.append(dict(cls='car', type='CONTRAST', name='Free / leader-driven (within day)', day=x, cfg=CFG[x], delta=DEL[x],
                     n=int(fn[x]), km=np.nan, rate=np.exp(logc[x]), rel=np.exp(logc[x] - gbase)))
trows.append(dict(cls='car', type='CONTRAST', name='Free / leader-driven (within day)', same_mean=np.exp(gbase),
                  diff_mean=np.exp(np.mean([logc[x] for x in ORDER8[3:]])), n_same=np.nan, n_diff=np.nan, km_same=np.nan, km_diff=np.nan,
                  **F2))
F5 = pd.DataFrame(rows); T5 = pd.DataFrame(trows)
out(F5, 'rates_by_type_data.csv'); out(T5, 'rates_by_type_tests.csv')
print(T5[['cls', 'type', 'n_same', 'n_diff', 'same_mean', 'diff_mean', 'ratio', 'ci_lo', 'ci_hi', 'p2', 'p2_rank']].round(3).to_string(index=False))

# ================================================================================================ fig3: hazard vs distance to the leader
# Same state machine as fig4 (S1 primary): positions x(t-3) <= 1800 m so that the 300 m look-ahead is inside the window.
# 'none within 300 m' = state F exactly (the free regime of fig4); leader bins = states C or M, binned by the gap to the lane-2 leader.
BINS = [0, 50, 100, 150, 300]; LAB = ['<50', '50-100', '100-150', '150-300', 'none within 300']
sel = ((S.truck == 0) & (S.Lane == 2) & (S.PositionX <= 1800)).values
s = S[sel].assign(st=ST[sel]); s = s[s.st.isin(['C', 'M', 'F'])].copy()
s['b'] = np.where(s.st == 'F', 4, pd.cut(s.lead_gap, BINS, right=False, labels=False))
assert s.b.notna().all()
e = E[(E.truck == 0) & (E.direction == 'in') & (E.s3_PositionX <= 1800) & E.typ.isin(['IN_free', 'IN_overtake', 'IN_mid'])].copy()
e['b'] = np.where(e.typ == 'IN_free', 4, pd.cut(e.s3_lead_gap, BINS, right=False, labels=False))
assert e.b.notna().all()
km = s.groupby(['day', 'b']).km.sum(); ev = e.groupby(['day', 'b']).size()
drow = []
for x in ORDER8:
    for b in range(5):
        k_ = float(km.get((x, b), 0.0)); n_ = int(ev.get((x, b), 0))
        drow.append(dict(day=x, cfg=CFG[x], delta=DEL[x], bin=LAB[b], n=n_, km=k_, rate=n_ / k_ if k_ > 0 else np.nan))
    k_ = float(sum(km.get((x, b), 0.0) for b in range(4))); n_ = int(sum(ev.get((x, b), 0) for b in range(4)))
    drow.append(dict(day=x, cfg=CFG[x], delta=DEL[x], bin='leader within 300 (pooled)', n=n_, km=k_, rate=n_ / k_))
DD = pd.DataFrame(drow); out(DD, 'fig3_rates_by_distance_days.csv')
rows = []
for b in LAB + ['leader within 300 (pooled)']:
    z = DD[DD.bin == b].set_index('day')
    t8 = contrast_tests(z.rate.to_dict(), days8, ns8)
    for g, dd in (('same', ORDER8[:3]), ('diff', ORDER8[3:])):
        rows.append(dict(group=g, bin=b, n_days=len(dd), n=int(z.loc[dd, 'n'].sum()), km=z.loc[dd, 'km'].sum(),
                         mean_day_rate=z.loc[dd, 'rate'].mean(), pooled_rate=z.loc[dd, 'n'].sum() / z.loc[dd, 'km'].sum(),
                         min_day_km=z.loc[dd, 'km'].min(), ratio_daymeans=t8['ratio'], ci_lo=t8['ci_lo'], ci_hi=t8['ci_hi'], p2_8=t8['p2'],
                         p2_rank8=t8['p2_rank'], p1_8=t8['p1']))
G3 = pd.DataFrame(rows); out(G3, 'fig3_rates_by_distance_data.csv'); print(G3.round(3).to_string(index=False))
g3f = G3[G3.bin == 'none within 300'].set_index('group')
chk += [dict(item='fig3 free class n same/diff', value=f"{g3f.n['same']}/{g3f.n['diff']}", reference='24/78'),
        dict(item='fig3 free class km same/diff', value=f"{g3f.km['same']:.1f}/{g3f.km['diff']:.1f}", reference='231.0/269.8'),
        dict(item='fig3 free class ratio of day means', value=round(g3f.ratio_daymeans['same'], 4), reference=2.51),
        dict(item='fig3 free class p two-sided', value=round(g3f.p2_8['same'], 4), reference=0.071)]
CHK = pd.DataFrame(chk); out(CHK, "rates_reproduction_check.csv")

# ================================================================================================ robustness of the free inward ratio
L = pd.read_csv(os.path.join(RES, 'lateral_check.csv'))
n0_ = len(E); E = E.merge(L[["day", "GlobalID", "t", "lat_ok"]].drop_duplicates(["day", "GlobalID", "t"]), on=["day", "GlobalID", "t"], how="left"); assert len(E) == n0_
print('lat_ok missing among events:', int(E.lat_ok.isna().sum()))


def qp_rates(vals, diff):
    """day-level quasi-Poisson GLM on day rates (log link, group dummy, no offset, Pearson dispersion, t with n-2 df).
    exp(b1) equals mean(diff days) / mean(same days) exactly; each day carries equal weight."""
    import statsmodels.api as sm
    y = np.asarray(vals, float); X = sm.add_constant(np.asarray(diff, float))
    m = sm.GLM(y, X, family=sm.families.Poisson()).fit(scale='X2')
    b, se = m.params[1], m.bse[1]; tq = stats.t.ppf(0.975, len(y) - 2)
    return float(np.exp(b)), float(np.exp(b - tq * se)), float(np.exp(b + tq * se)), float(m.scale)


def spec_table(k=3, xmin=None, ev_mask=None, **kw):
    typ = classify_events(E, k=k, **kw); st = exposure(S, **kw)
    return day_table(E, S, typ, st, k=k, xmin=xmin, ev_mask=ev_mask)


def hour_adjusted(T='IN_free', state_='F', lane=2, min_km=5.0):
    """pooled O/E with hour-bin expected rates (v_s1_tod2 logic on the S1 definitions): day x hour-bin cells with >= min_km car-km,
    expected = km x pooled hour-bin rate (8 days); statistic = log ratio of pooled O/E (differentiated vs same); exact 56 relabellings
    (expected rates recomputed for every relabelling is unnecessary: they do not depend on the labels). Interval: day-level quasi-Poisson
    with offset log(expected), whose point estimate equals the pooled O/E ratio."""
    import statsmodels.api as sm
    hb_edges = [7.5, 10, 12, 14, 16.5]
    sel = ((S.truck == 0) & (S.Lane == lane)).values
    Sx = S[sel].assign(st=ST[sel]); Sx = Sx[Sx.st == state_]
    Sx['hb'] = pd.cut(Sx.hour, hb_edges, labels=False)
    Ex = E[(E.truck == 0) & (E.typ == T)].copy(); Ex['hb'] = pd.cut(C.hours(Ex.t.values - 3000), hb_edges, labels=False)
    kmc = Sx.groupby(['day', 'hb']).km.sum(); nc = Ex.groupby(['day', 'hb']).size()
    cells = pd.concat([kmc.rename('km'), nc.rename('n')], axis=1).fillna(0).reset_index()
    cells = cells[(cells.km >= min_km) & cells.day.isin(days8)]
    rate_hb = cells.groupby('hb').n.sum() / cells.groupby('hb').km.sum()
    cells = cells.assign(E=cells.km * cells.hb.map(rate_hb))
    g = cells.groupby('day')[['n', 'E']].sum().reindex(days8)
    nd, Ed = g.n.values, g.E.values

    def stat(diff_mask, E_=Ed):
        return np.log((nd[diff_mask].sum() / E_[diff_mask].sum()) / (nd[~diff_mask].sum() / E_[~diff_mask].sum()))
    m0 = np.r_[np.zeros(ns8, bool), np.ones(len(days8) - ns8, bool)]
    masks = []
    for S_ in combinations(range(len(days8)), ns8):
        m = np.ones(len(days8), bool); m[list(S_)] = False; masks.append(m)
    obs = stat(m0); null = np.array([stat(m) for m in masks])
    # equal-tailed test-inverted 95% interval: multiply the expected counts of the differentiated days by r and keep r when both
    # exact one-sided p-values of the adjusted data exceed 0.025 (same construction as ratio_ci)
    grid = np.exp(np.linspace(np.log(0.02), np.log(50), 1200)); ok = []
    for r_ in grid:
        Ea = Ed.copy(); Ea[m0] = Ea[m0] * r_
        o_ = stat(m0, Ea); nl = np.array([stat(m, Ea) for m in masks])
        ok.append(np.mean(nl >= o_ - 1e-12) > 0.025 and np.mean(nl <= o_ + 1e-12) > 0.025)
    ok = np.array(ok); i0 = int(np.argmin(np.abs(np.log(grid) - obs)))
    lo_, hi_ = np.nan, np.nan
    if ok[i0]:
        lo = i0
        while lo > 0 and ok[lo - 1]:
            lo -= 1
        hi = i0
        while hi < len(grid) - 1 and ok[hi + 1]:
            hi += 1
        lo_ = grid[lo] if lo > 0 else 0.0; hi_ = grid[hi] if hi < len(grid) - 1 else np.inf
    X = sm.add_constant(m0.astype(float))
    fit = sm.GLM(nd, X, family=sm.families.Poisson(), offset=np.log(Ed)).fit(scale='X2')
    tq = stats.t.ppf(0.975, len(nd) - 2); b_, se_ = fit.params[1], fit.bse[1]
    oe = dict(zip(days8, nd / Ed))
    return dict(ratio=float(np.exp(obs)), p2=float(np.mean(np.abs(null) >= abs(obs) - 1e-12)), p1=float(np.mean(null >= obs - 1e-12)), nperm=len(null),
                p2_rank=perm_generic(nd / Ed, ns8, 'rank')[0], oe=oe, qp=(float(np.exp(b_)), float(np.exp(b_ - tq * se_)), float(np.exp(b_ + tq * se_)), float(fit.scale)),
                ci_lo=lo_, ci_hi=hi_, n_cells=len(cells), km_used=float(cells.km.sum()))


SPECS = [('primary', 'Primary: state at t − 3 s, no leader within 300 m', dict()),
         ('L150', 'Free = no leader within 150 m', dict(L=150)),
         ('L200', 'Free = no leader within 200 m', dict(L=200)),
         ('L500', 'Free = no leader within 500 m', dict(L=500)),
         ('t5', 'State at t − 5 s', dict(k=5)),
         ('lat', 'Lateral-confirmed changes only', dict(ev_mask='lat')),
         ('x1050', 'x(t − 3 s) ≥ 1050 m', dict(xmin=1050)),
         ('hour', 'Hour-of-day adjusted (pooled O/E)', dict(hour=True))]
SPECS += [(f'drop_{CFG[x]}', f'Without {CFG[x]}', dict(drop=x)) for x in ORDER8]          # full leave-one-day-out
rows, dayrows = [], []
_dx_cache = {}
for key, nice, opt in SPECS:
    kw = {k_: v_ for k_, v_ in opt.items() if k_ in ('k', 'L')}
    mask = (E.lat_ok == 1).values if opt.get('ev_mask') == 'lat' else None
    ck = (tuple(sorted(kw.items())), opt.get('xmin'), opt.get('ev_mask'))
    if ck not in _dx_cache:
        _dx_cache[ck] = spec_table(xmin=opt.get('xmin'), ev_mask=mask, **kw)
    Dx = _dx_cache[ck]
    dd, nsm = group_days(ORDER8, drop=opt.get('drop'))
    for T, reg in (('IN_free', 'free'), ('IN_overtake', 'leader-driven')):
        r, n, k = rate_dict(0, T, Dx)
        if opt.get('hour'):
            h = hour_adjusted(T, 'F' if T == 'IN_free' else 'C', min_km=5.0 if T == 'IN_free' else 0.5)
            res = {k_: h[k_] for k_ in ('ratio', 'ci_lo', 'ci_hi', 'p2', 'p1', 'nperm', 'p2_rank')}
            print('hour-adjusted', T, 'cells', h['n_cells'], 'km used', round(h['km_used'], 1), 'O/E', {CFG[x]: round(v, 2) for x, v in h['oe'].items()})
            res['complete_sep'] = int(min(h['oe'][x] for x in dd[nsm:]) > max(h['oe'][x] for x in dd[:nsm]))
            for x in dd:
                dayrows.append(dict(spec=key, regime=reg, day=x, value=h['oe'][x], n=int(n[x]), km=k[x], measure='O/E'))
        else:
            res = contrast_tests(r, dd, nsm)
            for x in dd:
                dayrows.append(dict(spec=key, regime=reg, day=x, value=r[x], n=int(n[x]), km=k[x], measure='rate'))
        if opt.get('hour'):
            qp = h['qp']
        else:
          try:
            qp = quasi_poisson_ratio([n[x] for x in dd], [k[x] for x in dd], [int(DEL[x] > 0) for x in dd])
          except Exception:
            qp = (np.nan,) * 4
        # day-level quasi-Poisson on the day values themselves (equal weight per day): its point estimate is exactly the ratio of
        # group means, so it is the interval that belongs to the plotted ratio (the count/offset version above estimates the pooled ratio)
        vals = [h['oe'][x] for x in dd] if opt.get('hour') else [r[x] for x in dd]
        qr = qp_rates(vals, [int(DEL[x] > 0) for x in dd])
        rows.append(dict(spec=key, label=nice, regime=reg, n_days=len(dd), n_same_days=nsm, n_same=int(sum(n[x] for x in dd[:nsm])),
                         n_diff=int(sum(n[x] for x in dd[nsm:])), km_same=sum(k[x] for x in dd[:nsm]), km_diff=sum(k[x] for x in dd[nsm:]),
                         **res, qp_ratio=qp[0], qp_lo=qp[1], qp_hi=qp[2], qp_disp=qp[3],
                         qpr_ratio=qr[0], qpr_lo=qr[1], qpr_hi=qr[2], qpr_disp=qr[3]))
R = pd.DataFrame(rows); out(R, 'rates_robustness_data.csv'); out(pd.DataFrame(dayrows), 'rates_robustness_days.csv')
print(R[['spec', 'regime', 'n_days', 'n_same', 'n_diff', 'ratio', 'ci_lo', 'ci_hi', 'p2', 'p2_rank', 'p1', 'nperm', 'complete_sep', 'qp_ratio', 'qp_lo', 'qp_hi']]
      .round(3).to_string(index=False))

# ================================================================================================ context of car inward changes
ci = E[(E.truck == 0) & (E.direction == 'in')].copy()
ci['hw3'] = ci.s3_lead_gap / np.maximum(ci.s3_v / 3.6, 0.1)
ci['state3'] = ci.st3
ci['ctx'] = np.select([ci.st3.isin(['C', 'M']), ci.st3 == 'F', ci.st3 == 'N', ci.st3 == 'U'],
                      ['leader within 300 m', 'no leader within 300 m', 'view truncated', 'not tracked at t-3 s'], 'other')
ci['group'] = np.where(ci.delta > 0, 'diff', 'same')
cols = ['day', 'cfg', 'delta', 'group', 'GlobalID', 't', 'x', 's3_PositionX', 's3_Lane', 's3_v', 's3_lead_gap', 's3_lead_dv', 's3_lead_truck', 'hw3',
        'state3', 'ctx', 'typ', 'lat_ok']
out(ci[cols], 'rates_inward_context_data.csv')
print(pd.crosstab(ci.ctx, ci.group))
print(pd.crosstab(ci.state3, ci.group))
# day-level test of the heavy-vehicle share among leaders of leader-present car inward changes
zl = ci[ci.ctx == 'leader within 300 m']
hv = zl.groupby('day').s3_lead_truck.agg(lambda v: float(np.mean(v == 1))); hn = zl.groupby('day').size()
hvt = contrast_tests(hv.to_dict(), days8, ns8)
H = pd.DataFrame(dict(day=ORDER8, cfg=[CFG[x] for x in ORDER8], n_changes=[int(hn.get(x, 0)) for x in ORDER8],
                      heavy_share=[hv.get(x, np.nan) for x in ORDER8]))
H['same_mean'] = np.mean([hv[x] for x in ORDER8[:3]]); H['diff_mean'] = np.mean([hv[x] for x in ORDER8[3:]])
H['p2_8'] = hvt['p2']; H['p2_rank8'] = hvt['p2_rank']
out(H, 'rates_heavy_leader_days.csv'); print(H.round(3).to_string(index=False))
