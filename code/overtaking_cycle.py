# -*- coding: utf-8 -*-
"""Stay in the inner lane after moving in (Section 5.3).
Stint = a vehicle's stay in lane 1 from an analysis-grade inward change, or from entry in lane 1 at x < 1,050 m, to its next
analysis-grade outward change (return) or the end of its track (censored). An inward change opens a stint only if it is lateral-confirmed
(lat_ok); a return counts only if it is lateral-confirmed as well, otherwise the stint is censored there. Situation of the opening change
from core_events mtype0 (leader-driven 'in_overtake_*', unconstrained 'in_free_*'). Distance = recorded position of the return (or track
end) minus the recorded position of the inward change (or entry).
Computes the Kaplan-Meier share still in the inner lane by distance (log-log 95 % intervals; curves drawn while >= 10 stints are at risk)
for heavy vehicles and cars after a leader-driven change, cars after an unconstrained change and cars already in the inner lane at entry;
numbers at risk; rate ratios (Poisson with day effects, vehicle-clustered errors); and per day the probability of being back in the
outer lane within 600 m, with exact day-level tests. Heavy stints opened by an unconstrained change (36) are not drawn: 11 of them
coincide with an opposite change of another vehicle (see stays_checks.py); they enter the pooled rate ratios.
Option --fig7-only stops after the first part.
Inputs: _cache/work/stints.pkl, lateral_check.pkl, core_events.pkl; results/check_stint_postpass_units.csv.
Outputs: results/final/overtaking_cycle_data.csv, overtaking_cycle_atrisk.csv, overtaking_cycle_stats.csv,
overtaking_cycle_markers.csv, stays_cycle_by_config_data.csv, stays_cycle_by_config_tests.csv; working figures
_cache/working_figures/overtaking_cycle and stays_cycle_by_config (.png/.pdf)."""
import os, sys, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from plot_style import *
warnings.filterwarnings('ignore')

plt = setup()
plt.rcParams['savefig.bbox'] = 'standard'           # export at the nominal width (no tight bbox)
from matplotlib.transforms import blended_transform_factory
from matplotlib.lines import Line2D
FS = 7.0                                              # annotation / legend / table text (pt)

# ------------------------------------------------------------------------------------------------------------ data
S = pd.read_pickle(os.path.join(CACHE, 'stints.pkl'))
L = pd.read_pickle(os.path.join(CACHE, 'lateral_check.pkl')).set_index(['day', 'GlobalID', 't'])
E = pd.read_pickle(os.path.join(CACHE, 'core_events.pkl')).set_index(['day', 'GlobalID', 't'])
S = S[S.truck >= 0].copy()
S['in_ok'] = [L.lat_ok.get((r.day, r.GlobalID, r.t_start), np.nan) if r.start == 'lc' else 1 for r in S.itertuples()]
S['out_ok'] = [L.lat_ok.get((r.day, r.GlobalID, r.t_end), np.nan) if r.ret == 1 else np.nan for r in S.itertuples()]
S['ret_c'] = ((S.ret == 1) & (S.out_ok == 1)).astype(int)
S['ret_any'] = ((S.ret == 1) | (S.nonag_out == 1)).astype(int)          # sensitivity: any recorded move out of lane 1 (AG or not)
S['d'] = S.dist.clip(lower=1.0)
S['km'] = S.d / 1000
S['grp'] = np.where(S.delta > 0, 'diff', 'same')
S['mtype0'] = [E.mtype0.get((r.day, r.GlobalID, r.t_start), 'NA') if r.start == 'lc' else 'entry' for r in S.itertuples()]
S['regime'] = np.where(S.mtype0.str.startswith('in_overtake'), 'leader', np.where(S.mtype0.str.startswith('in_free'), 'free', S.start))
SC = S[S.in_ok == 1].copy()
LC = SC[SC.start == 'lc']
assert (LC.regime.isin(['leader', 'free'])).all()

LC8, SC8 = LC, SC                                         # overtaking cycle: the eight days
SETS = {
    'heavy_ld': LC8[(LC8.truck == 1) & (LC8.regime == 'leader')],
    'car_ld': LC8[(LC8.truck == 0) & (LC8.regime == 'leader')],
    'car_un': LC8[(LC8.truck == 0) & (LC8.regime == 'free')],
    'car_entry': SC8[(SC8.truck == 0) & (SC8.start == 'entry')],
}
EXTRA8 = {   # not drawn, reported in the stats table
    'heavy_un': LC8[(LC8.truck == 1) & (LC8.regime == 'free')],
    'heavy_lc_all': LC8[LC8.truck == 1],
    'car_lc_all': LC8[LC8.truck == 0],
}
EXTRA = {    # used only by the per-configuration part (per-day values)
    'heavy_lc_all': LC[LC.truck == 1],
    'car_lc_all': LC[LC.truck == 0],
}
MIN_RISK = 10                                              # draw each curve only while >= MIN_RISK stints remain at risk
print({k: len(v) for k, v in {**SETS, **EXTRA8}.items()})


def km_curve(d, e):
    """Kaplan-Meier with Greenwood variance and log-log 95% CI. Returns step table (t, S, lo, hi, n_risk)."""
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


def n_risk(d, x):
    return int((np.asarray(d) >= x).sum())


def rr_fit(q, y='ret_c'):
    q = q.assign(lkm=np.log(q.km), gid=q.day + '_' + q.GlobalID.astype(str))
    m = smf.glm(f'{y} ~ truck + day', data=q, family=sm.families.Poisson(), offset=q.lkm).fit(
        cov_type='cluster', cov_kwds={'groups': pd.factorize(q.gid)[0]})
    return (np.exp(m.params['truck']), *np.exp(m.conf_int().loc['truck'].values), m.pvalues['truck'])


# ------------------------------------------------------------------------------------------------------------ overtaking cycle numbers
curves, stats, atrisk = {}, [], []
GRID = [0, 200, 400, 600, 800]
for nm, D in {**SETS, **EXTRA8}.items():
    T = km_curve(D.d, D.ret_c); curves[nm] = T
    r = dict(set=nm, days='8 days', n_stints=len(D), n_veh=D.groupby('day').GlobalID.nunique().sum(), n_days=D.day.nunique(), returns=int(D.ret_c.sum()),
             km=D.km.sum(), returns_per_km=D.ret_c.sum() / D.km.sum(), share_censored=1 - D.ret_c.mean(), median_start_x=D.x_start.median(),
             median_dist_followed=D.dist.median(), returns_any_incl_nonAG=int(D.ret_any.sum()))
    for h in (200, 400, 600):
        s, lo, hi = km_at(T, h)
        r[f'P_back_{h}'] = 1 - s; r[f'P_back_{h}_lo'] = 1 - hi; r[f'P_back_{h}_hi'] = 1 - lo
    below = T[T.S <= 0.5]
    r['median_dist_m'] = below.dist.iloc[0] if len(below) else np.nan
    r['last_dist_with_10_at_risk'] = float(np.sort(D.d.values)[-MIN_RISK]) if len(D) >= MIN_RISK else np.nan
    stats.append(r)
    if nm in SETS:
        for x in GRID:
            atrisk.append(dict(set=nm, dist=x, n_risk=n_risk(D.d, x)))

# heavy/car rate ratios (Poisson, day FE, offset log km in lane 1, clusters = vehicle-day); name the clock and the subset every time
RRS = {
    'RR_heavy_vs_car_leader_driven': (rr_fit(pd.concat([SETS['heavy_ld'], SETS['car_ld']])),
                                      'leader-driven inward changes only; clock at the inward change; 8 days'),
    'RR_heavy_vs_car_all_inward': (rr_fit(LC), 'all lateral-confirmed inward changes; clock at the inward change; 8 days'),
}
RRS['RR_car_leader_vs_car_unconstrained'] = (rr_fit(LC8[LC8.truck == 0].assign(truck=lambda x: (x.regime == 'leader').astype(int))),
                                             'cars only, 8 days: leader-driven vs unconstrained inward change')
RRS['RR_heavy_unconstrained_vs_heavy_leader'] = (rr_fit(LC8[LC8.truck == 1].assign(truck=lambda x: (x.regime == 'free').astype(int))),
                                                 'heavy only, 8 days: unconstrained vs leader-driven inward change')
# car unconstrained vs cars already in the inner lane at entry (persistence of free movers)
qq = pd.concat([SETS['car_un'].assign(truck=1), SETS['car_entry'].assign(truck=0)])
RRS['RR_car_unconstrained_vs_car_entry'] = (rr_fit(qq), 'cars: unconstrained inward change vs already in the inner lane at entry '
                                            '(left-truncated reference, clock at window entry); 8 days')

# post-pass return (clock at the last pass of a lane-2 vehicle made from lane 1), LC-opened runs, from the unit table of check_post_pass.py
U = pd.read_csv(os.path.join(RES, 'check_stint_postpass_units.csv'))
U = U[~((U.start_type == 'lc') & (U.start_lat_ok != 1)) & (U.truck >= 0)].copy()
U['km'] = U.dist / 1000
for sub, q in (('lc_started', U[U.start_type == 'lc']), ('all_runs', U)):
    r_ = rr_fit(q, y='ev_strict')
    RRS[f'RR_heavy_vs_car_after_last_pass_{sub}'] = (r_, f'check_stint_postpass_units, clock at the last pass made from lane 1; cars n={int((q.truck == 0).sum())} '
                                                         f'returns={int(q[q.truck == 0].ev_strict.sum())}, heavy n={int((q.truck == 1).sum())} '
                                                         f'returns={int(q[q.truck == 1].ev_strict.sum())}')

ST = pd.DataFrame(stats)
extra = pd.DataFrame([dict(set=k, estimate=v[0][0], lo=v[0][1], hi=v[0][2], p=v[0][3], note=v[1]) for k, v in RRS.items()])
ST = pd.concat([ST, extra], ignore_index=True)
ST.to_csv(os.path.join(DATF, 'overtaking_cycle_stats.csv'), index=False)
pd.DataFrame(atrisk).pivot(index='set', columns='dist', values='n_risk').to_csv(os.path.join(DATF, 'overtaking_cycle_atrisk.csv'))
KMALL = pd.concat([T.assign(set=nm, drawn=int(nm in SETS)) for nm, T in curves.items()], ignore_index=True)
KMALL.to_csv(os.path.join(DATF, 'overtaking_cycle_data.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(ST.round(3).to_string())
print(pd.DataFrame(atrisk).pivot(index='set', columns='dist', values='n_risk'))

# ------------------------------------------------------------------------------------------------------------ overtaking cycle plot
# class = marker (circle car, triangle heavy); regime = line style (solid leader-driven, dashed unconstrained); background = grey.
# 95% CIs: vertical bars at the marked distances (staggered by curve); narrow band for the background curve.
DASH = (0, (4.0, 1.8))
STY = {
    'heavy_ld': dict(color='#1A1A1A', ls='-', lw=1.2, marker='^', ms=4.4, mfc='white', xm=(100, 300, 500, 700)),
    'car_ld': dict(color='#1A1A1A', ls='-', lw=1.2, marker='o', ms=4.0, mfc='white', xm=(150, 350, 550, 750)),
    'car_un': dict(color='#1A1A1A', ls=DASH, lw=1.2, marker='o', ms=4.0, mfc='#1A1A1A', xm=(200, 400, 600, 800)),
    'car_entry': dict(color='#8C8C8C', ls='-', lw=1.0, marker=None, ms=0, mfc=None, xm=()),
}
LAB = {'heavy_ld': 'Heavy vehicles, leader-driven', 'car_ld': 'Cars, leader-driven',
       'car_un': 'Cars, not leader-driven', 'car_entry': 'Cars, at entry'}
XMAX = 900
FIG_H = 96.0
TOP = 13.5                                                # mm reserved for the legend (3 rows)
fig = plt.figure(figsize=(W1, FIG_H * MM))
# number-at-risk rows are labelled with the curve symbols of the legend (full names in the legend), so the axes can start near the left edge
AXL = 0.175
AX = [AXL, 28.5 / FIG_H, 0.975 - AXL, (FIG_H - TOP - 28.5) / FIG_H]
ax = fig.add_axes(AX)
mk_rows = []
for nm in ('car_entry', 'car_un', 'car_ld', 'heavy_ld'):
    T = curves[nm]; st = STY[nm]
    dmax = SETS[nm].d.max()                                   # extend the step to the last observed distance (censoring)
    dcut = float(np.sort(SETS[nm].d.values)[-MIN_RISK])       # last distance with >= MIN_RISK stints at risk
    dend = min(dmax, XMAX, dcut)
    T = T[T.dist <= dend]
    t = np.r_[T.dist.values, dend]; s = np.r_[T.S.values, T.S.values[-1]]
    lo = np.r_[T.lo.values, T.lo.values[-1]]; hi = np.r_[T.hi.values, T.hi.values[-1]]
    if nm == 'car_entry':
        ax.fill_between(t, lo, hi, step='post', color=st['color'], alpha=0.25, lw=0, zorder=1)
    ax.step(t, s, where='post', color=st['color'], ls=st['ls'], lw=st['lw'], zorder=3)
    for x in st['xm']:
        if x > dend:
            continue
        sv, l_, h_ = km_at(T, x)
        ax.plot([x, x], [l_, h_], color=st['color'], lw=0.7, zorder=3.5, solid_capstyle='butt')
        ax.plot(x, sv, ls='none', marker=st['marker'], ms=st['ms'], mfc=st['mfc'], mec=st['color'], mew=0.8, zorder=4)
        mk_rows.append(dict(set=nm, dist=x, S=sv, lo=l_, hi=h_, n_risk=n_risk(SETS[nm].d, x)))
pd.DataFrame(mk_rows).to_csv(os.path.join(DATF, 'overtaking_cycle_markers.csv'), index=False)
ax.set_xlim(0, XMAX); ax.set_ylim(0, 1.02)
ax.set_xticks(GRID); ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_ylabel('Share still in the inner lane')
ax.set_xlabel('Distance travelled in the inner lane (m)', labelpad=2)
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
h = [Line2D([], [], color=STY[k]['color'], ls=STY[k]['ls'], lw=STY[k]['lw'], marker=STY[k]['marker'], ms=STY[k]['ms'], mfc=STY[k]['mfc'],
            mew=0.8) for k in ('heavy_ld', 'car_ld', 'car_un', 'car_entry')]
lab = [LAB[k] for k in ('heavy_ld', 'car_ld', 'car_un', 'car_entry')]
h.append((Patch(facecolor=STY['car_entry']['color'], alpha=0.25, lw=0), Line2D([], [], color='#1A1A1A', lw=0, marker='|', ms=7, mew=0.7)))
lab.append('95 % CI')
fig.legend(handles=h, labels=lab, loc='upper left', bbox_to_anchor=(0.012, 0.998), ncol=2, fontsize=FS,
           handlelength=2.7, labelspacing=0.3, columnspacing=1.2, borderaxespad=0.0, handletextpad=0.5,
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})
# number-at-risk table under the axis
tr = blended_transform_factory(ax.transData, fig.transFigure)
dy = 3.3 / FIG_H
y0 = 12.9 / FIG_H
fig.text(0.012, y0 + dy, 'Number at risk', fontsize=FS, fontweight='bold', ha='left', va='center')
rows_lab = [(k, LAB[k]) for k in ('heavy_ld', 'car_ld', 'car_un', 'car_entry')]
GX0, GX1 = 0.014, 0.105                                   # row symbol, same drawing as the legend handle
for i, (nm, lab) in enumerate(rows_lab):
    y = y0 - i * dy
    col = '#1A1A1A' if nm != 'car_entry' else '#6E6E6E'
    st = STY[nm]
    fig.add_artist(Line2D([GX0, GX1], [y, y], transform=fig.transFigure, color=st['color'], ls=st['ls'], lw=st['lw']))
    if st['marker']:
        fig.add_artist(Line2D([(GX0 + GX1) / 2], [y], transform=fig.transFigure, ls='none', marker=st['marker'], ms=st['ms'],
                              mfc=st['mfc'], mec=st['color'], mew=0.8))
    for x in GRID:
        fig.text(x, y, f"{n_risk(SETS[nm].d, x):,}", transform=tr, fontsize=FS, ha='center', va='center', color=col)
save(fig, 'overtaking_cycle', FIGW); plt.close(fig)
if '--fig7-only' in sys.argv:
    print('overtaking cycle done (per-configuration part skipped)'); sys.exit(0)

# ------------------------------------------------------------------------------------------------------------ by configuration (day level)
H = 600.0
rows = []
for cls, nm in ((0, 'car_lc_all'), (1, 'heavy_lc_all')):
    D = EXTRA[nm]
    for dname in ORDER8:
        b = D[D.day == dname]
        if len(b) == 0:
            rows.append(dict(cls='car' if cls == 0 else 'heavy', day=dname, n=0)); continue
        T = km_curve(b.d, b.ret_c)
        s, lo, hi = km_at(T, H)
        rows.append(dict(cls='car' if cls == 0 else 'heavy', day=dname, cfg=DAY_META.set_index('day').label[dname], delta=DAY_META.set_index('day').delta[dname],
                         n=len(b), returns=int(b.ret_c.sum()), returns_within_600=int(((b.ret_c == 1) & (b.d <= H)).sum()), km=b.km.sum(),
                         P_back_600=1 - s, lo=1 - hi, hi=1 - lo, n_risk_600=n_risk(b.d, H), returns_per_km=b.ret_c.sum() / b.km.sum()))
DY = pd.DataFrame(rows)
# degenerate KM estimates (0 or 1): exact binomial (Clopper-Pearson) bound on the stints resolved by 600 m (returned before 600 m or followed to 600 m)
DY['n_resolved_600'] = DY.n_risk_600 + DY.returns_within_600
DY['ci_method'] = np.where(DY.n > 0, 'KM log-log', '')
for i, r in DY.iterrows():
    if r.n == 0:
        continue
    ne = int(r.n_resolved_600)
    if r.P_back_600 <= 1e-9:
        DY.loc[i, ['lo', 'hi', 'ci_method']] = [0.0, 1 - 0.025 ** (1 / ne) if ne > 0 else 1.0, 'exact binomial, 0 of n_resolved']
    elif r.P_back_600 >= 1 - 1e-9:
        DY.loc[i, ['lo', 'hi', 'ci_method']] = [0.025 ** (1 / ne) if ne > 0 else 0.0, 1.0, 'exact binomial, n_resolved of n_resolved']
DY['weak_600'] = (DY.n_risk_600 < 2).astype(int)        # fewer than 2 vehicles followed to 600 m: estimate barely identified
tests = []
for cls in ('car', 'heavy'):
    for meas in ('P_back_600', 'returns_per_km'):
        v8 = DY[DY.cls == cls].set_index('day').reindex(ORDER8)[meas].values
        obs, p2, p1 = perm8(v8)
        sel = lambda days: DY[(DY.cls == cls) & DY.day.isin(days)]
        tests.append(dict(cls=cls, measure=meas, same_mean=np.mean(v8[:3]), diff_mean=np.mean(v8[3:8]), diff_minus_same=obs,
                          p2_8days=p2, p_rank_8days=perm_rank8(v8),
                          n_same=int(sel(ORDER8[:3]).n.sum()), n_diff=int(sel(ORDER8[3:]).n.sum()),
                          returns_same=int(sel(ORDER8[:3]).returns.sum()), returns_diff=int(sel(ORDER8[3:]).returns.sum())))
TS = pd.DataFrame(tests)
DY.to_csv(os.path.join(DATF, 'stays_cycle_by_config_data.csv'), index=False)
TS.to_csv(os.path.join(DATF, 'stays_cycle_by_config_tests.csv'), index=False)
print(DY.round(3).to_string()); print(TS.round(3).to_string())

XORD = ORDER8
xpos = {d: i + (0.6 if i >= 3 else 0) for i, d in enumerate(XORD)}
META = DAY_META.set_index('day')
FH = 94.0
fig = plt.figure(figsize=(W15, FH * MM))
ax = fig.add_axes([0.078, 36.0 / FH, 0.912, (FH - 11.5 - 36.0) / FH])
OFF = {'car': -0.17, 'heavy': 0.17}
FADE = 0.35
for cls in ('car', 'heavy'):
    q = DY[(DY.cls == cls)].set_index('day')
    for dname in XORD:
        if q.n.get(dname, 0) == 0:
            continue
        r = q.loc[dname]; x = xpos[dname] + OFF[cls]
        col = day_color(dname)
        a = FADE if r.weak_600 == 1 else 1.0
        ax.plot([x, x], [r.lo, r.hi], color=col, lw=0.8, alpha=0.8 * a, zorder=2, solid_capstyle='butt')
        if cls == 'car':
            ax.plot(x, r.P_back_600, marker='o', ms=4.6, mfc=col, mec='#333333', mew=0.4, ls='none', zorder=4, clip_on=False, alpha=a)
        else:
            ax.plot(x, r.P_back_600, marker='^', ms=5.0, mfc='white', mec=col, mew=1.1, ls='none', zorder=4, clip_on=False, alpha=a)
    t = TS[(TS.cls == cls) & (TS.measure == 'P_back_600')].iloc[0]
    for grp, days, colg in (('same', ORDER8[:3], C_SAME), ('diff', ORDER8[3:], C_DIFF)):
        xs = [xpos[d] for d in days]; mval = t.same_mean if grp == 'same' else t.diff_mean
        ax.plot([min(xs) - 0.4, max(xs) + 0.4], [mval, mval], color=colg, lw=1.5, zorder=3, solid_capstyle='butt',
                ls='-' if cls == 'car' else (0, (3, 1.5)))
ax.set_xlim(-0.6, max(xpos.values()) + 0.5); ax.set_ylim(0, 1.0)
ax.set_xticks([xpos[d] for d in XORD])
ax.set_xticklabels([META.label[d] for d in XORD], fontsize=7)
ax.set_ylabel('Share back in the outer lane within 600 m')
# group labels under the tick labels, n rows below, test results at the bottom
trf = blended_transform_factory(ax.transData, fig.transFigure)
ax.text(np.mean([xpos[d] for d in XORD[:3]]), 25.5 / FH, 'Same limit', transform=trf, ha='center', va='center', fontsize=FS, color='#6E6E6E')
ax.text(np.mean([xpos[d] for d in XORD[3:]]), 25.5 / FH, 'Differentiated (inner/outer, km/h)', transform=trf, ha='center', va='center', fontsize=FS, color=C_DIFF)
YN = (19.0 / FH, 15.0 / FH)
fig.text(0.008, YN[0], 'n cars', fontsize=FS, ha='left', va='center'); fig.text(0.008, YN[1], 'n heavy', fontsize=FS, ha='left', va='center')
for d in XORD:
    for cls, y in (('car', YN[0]), ('heavy', YN[1])):
        nn = DY[(DY.cls == cls) & (DY.day == d)].n
        fig.text(xpos[d], y, str(int(nn.iloc[0])) if len(nn) else '0', transform=trf, fontsize=FS, ha='center', va='center')
tc = TS[(TS.cls == 'car') & (TS.measure == 'P_back_600')].iloc[0]; th = TS[(TS.cls == 'heavy') & (TS.measure == 'P_back_600')].iloc[0]
fig.text(0.008, 8.6 / FH, f"Cars: day means {tc.same_mean:.2f} (same limit) vs {tc.diff_mean:.2f} (differentiated); 8 days {fmt_p(tc.p2_8days)}; "
         f"inconclusive.", fontsize=FS, ha='left', va='center')
fig.text(0.008, 4.4 / FH, f"Heavy vehicles: day means {th.same_mean:.2f} vs {th.diff_mean:.2f}; 8 days {fmt_p(th.p2_8days)}; "
         f"no detectable difference.", fontsize=FS, ha='left', va='center')
h = [Line2D([], [], color='#555555', marker='o', ls='none', ms=4.6, mec='#333333', mew=0.4, label='Cars (day value, 95% CI)'),
     Line2D([], [], color='#555555', marker='^', ls='none', ms=5.0, mfc='white', mew=1.1, label='Heavy vehicles (day value, 95% CI)'),
     Line2D([], [], color='#555555', lw=1.5, label='Mean of day values, cars'),
     Line2D([], [], color='#555555', lw=1.5, ls=(0, (3, 1.5)), label='Mean of day values, heavy'),
     Line2D([], [], color='#555555', marker='^', ls='none', ms=5.0, mfc='white', mew=1.1, alpha=FADE, label='Faded: < 2 vehicles followed to 600 m')]
fig.legend(handles=h, loc='upper left', bbox_to_anchor=(0.03, 0.995), ncol=3, fontsize=FS, handlelength=1.8, columnspacing=1.2,
          labelspacing=0.35, borderaxespad=0.0)
save(fig, 'stays_cycle_by_config', FIGW); plt.close(fig)
print('done')
