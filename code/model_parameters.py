# -*- coding: utf-8 -*-
"""Simulation-ready parameters of the lane-change model and the empirical rates (results/final/sim_parameters.csv).
Recommended model: car inward changes M2F (M0 incentive terms with each driver's own desired speed, plus the bias of unconstrained cars on
differentiated days), car outward changes M0, heavy vehicles M0; alternative M1p for cars (SUMO convention).
Blocks: A stochastic MOBIL: logit coefficients (1-s interval-censored hazard, causal desired-speed proxy, eight days, x = 1,000-1,800 m)
with vehicle-clustered 95 % intervals, MOBIL-form parameters (delta method) and day-level tests; B unconstrained inward rate of cars
(empirical); C keep-right and return by class (empirical); D desired-speed inputs; E SUMO LC2013 settings and their fit; F closed-loop
stochastic MOBIL in SUMO.
Inputs: _cache/work/decision_points.pkl and the result tables read below. Outputs: results/final/model_fits.csv, sim_parameters.csv."""
import gc
from model_lib import *

rows = []


def add(block, cls, dirn, param, est, lo=np.nan, hi=np.nan, unit='', model='', note='', n=''):
    rows.append(dict(block=block, vehicle_class=cls, direction=dirn, model=model, parameter=param, estimate=est, ci95_lo=lo, ci95_hi=hi, unit=unit, n=n, note=note))


# ---------------- A: stochastic MOBIL
D = steps('causal', 0); F = features(D, specs=('M0', 'M1p')); gc.collect()
SPEC = [('car', 'in', 'M2F (recommended)', 'M0', 'free_step'), ('car', 'out', 'M0 (recommended)', 'M0', None),
        ('truck', 'in', 'M0 (recommended)', 'M0', None), ('truck', 'out', 'M0 (recommended)', 'M0', None),
        ('car', 'in', 'M1p (alternative, SUMO convention)', 'M1p', None), ('car', 'out', 'M1p (alternative, SUMO convention)', 'M1p', None)]
LAB = {'const': ('beta_0 (intercept)', 'logit'), 'dA_ego': ('beta_ego: own acceleration gain', 'per m/s2'), 'dA_n': ('beta_n: new follower acceleration change', 'per m/s2'),
       'dA_o': ('beta_o: old follower acceleration change', 'per m/s2'), 'unsafe': ('beta_unsafe: new follower would brake harder than 4 m/s2', 'logit'),
       'vd10': ('beta_v: desired speed, per 10 km/h above 80 km/h', 'logit'), 'state': ('beta_s: vehicle entered the current lane by a lane change', 'logit'),
       'free': ('beta_free: no lane-2 leader within 300 m', 'logit'), 'free_x_diff': ('beta_F: free x differentiated limits (inner-lane bias of unconstrained cars)', 'logit')}
fits = []
for cls, dirn, name, spec, bias in SPEC:
    S, Fs = subset(D, F, cls, dirn, MAIN)
    ev = event_ids(S)
    X, cols = design(S, Fs[spec])
    if bias == 'free_step':
        fr = S.free.values.astype(float)
        X = np.column_stack([X, fr, fr * S.diff_day.values]); cols = cols + ['free', 'free_x_diff']
    b, ll = ich_fit(X, ev)
    V = ich_vcov(b, X, ev, S.GID.values); se = np.sqrt(np.clip(np.diag(V), 0, None))
    lodo = 0.0
    for d in MAIN:
        a = (S.day != d).values
        bb, _ = ich_fit(X[a], ev[a], b0=b); lodo += ich_ll(bb, X[~a], ev[~a])
    nev = int(ev.max() + 1); nveh = S.GID.nunique()
    ntxt = f'{nev} changes, {nveh} vehicles, {len(S)} decision points'
    tag = f'{name} (LL {ll:.1f}, AIC {2 * len(b) - 2 * ll:.1f}, LODO {lodo:.1f})'
    blk = 'A stochastic MOBIL: logit coefficients'
    for j, c in enumerate(cols):
        lab, unit = LAB[c]
        if abs(b[j]) > 10:        # quasi-separation: no change was observed in this state
            add(blk, cls, dirn, lab, b[j], np.nan, np.nan, unit, tag, n=ntxt, note='no change observed in this state (quasi-separation); implement as a veto (probability 0)')
        else:
            note = ''
            if c == 'dA_n' and b[j] - 1.96 * se[j] < 0:
                note = 'CI includes 0 or is negative: set the politeness weight on the new follower to 0 in simulation'
            add(blk, cls, dirn, lab, b[j], b[j] - 1.96 * se[j], b[j] + 1.96 * se[j], unit, tag, n=ntxt, note=note)
    ie = cols.index('dA_ego')

    def ratio(k, sign=1.0):
        g = np.zeros(len(b)); g[k] = sign / b[ie]; g[ie] = -sign * b[k] / b[ie] ** 2
        est = sign * b[k] / b[ie]; s = np.sqrt(g @ V @ g)
        return est, est - 1.96 * s, est + 1.96 * s
    blk = 'A stochastic MOBIL: MOBIL form'
    g = np.zeros(len(b)); g[ie] = -1 / b[ie] ** 2; s = np.sqrt(g @ V @ g)
    add(blk, cls, dirn, 'sigma = 1/beta_ego: logistic scale of the incentive', 1 / b[ie], 1 / b[ie] - 1.96 * s, 1 / b[ie] + 1.96 * s, 'm/s2', tag, n=ntxt)
    for k_, lab in (('dA_n', 'p_n = beta_n/beta_ego: weight on the new follower'), ('dA_o', 'p_o = beta_o/beta_ego: weight on the old follower')):
        e_, l_, h_ = ratio(cols.index(k_)); add(blk, cls, dirn, lab, e_, l_, h_, '-', tag, n=ntxt)
    e_, l_, h_ = ratio(cols.index('const'), -1.0)
    add(blk, cls, dirn, 'a_th = -beta_0/beta_ego: incentive at which the 1-s change probability is 0.5 (v_des = 80 km/h, not entered by a lane change)',
        e_, l_, h_, 'm/s2', tag, n=ntxt, note=f'1-s probability at zero incentive = {expit(b[0]):.4f}')
    for k_, lab in (('free', 'beta_free/beta_ego: offset for cars with no leader (all days)'),
                    ('free_x_diff', 'a_bias = beta_F/beta_ego: inner-lane bias of unconstrained cars when the inner-lane limit is higher')):
        if k_ in cols:
            e_, l_, h_ = ratio(cols.index(k_)); add(blk, cls, dirn, lab, e_, l_, h_, 'm/s2', tag, n=ntxt)
    fits.append(dict(cls=cls, dirn=dirn, model=name, LL=ll, AIC=2 * len(b) - 2 * ll, LODO=lodo, **{f'b_{c}': x for c, x in zip(cols, b)}, **{f'se_{c}': x for c, x in zip(cols, se)}))
    del S, Fs; gc.collect()
pd.DataFrame(fits).to_csv(os.path.join(FIN, 'model_fits.csv'), index=False)

blk = 'A stochastic MOBIL: set-up and tests'
add(blk, 'all', 'both', 'IDM inside MOBIL: a / b / s0 / T / delta', np.nan, unit='1.2 m/s2 / 2.1 m/s2 / 2.5 m / 1.5 s / 4',
    note='a and b as in Ren et al. (2026, AAP 236:108703), s0, T and delta fixed; leaders beyond 400 m ignored; gaps extrapolated 5 s ahead; accelerations floored at -4 m/s2; hazard h = logistic(X b) per 1-s step')
add(blk, 'car', 'in', 'recommended: M2F; unconstrained = no lane-2 vehicle within 300 m ahead', np.nan,
    note='M0 incentive terms (own desired speed, lane limits not used) + free + free x 1[Delta > 0]; LODO free-state ratio differentiated/same x2.79 (observed x2.42), '
         'leader-driven x1.39 (observed x1.03; M0 alone x1.37)')
pt = pd.read_csv(os.path.join(FIN, 'model_perm_free_tests.csv'))
for _, r in pt[pt.stat.isin(['b_free_diff', 'b_lead_diff'])].iterrows():
    what = 'beta_F (free x differentiated)' if r.stat == 'b_free_diff' else 'same bias on leader-driven steps (contrast)'
    add(blk, r.cls, r.dirn, f'exact 56-labelling test of {what}: two-sided p', r.p_two_abs,
        note=f'one-sided p {r.p_one_upper:.3f}; observed coefficient {r.observed:.3f}, rank {int(r.rank_of_obs)} of 56' +
             ('; the same labelling test on the same free-state data as the empirical step (F1), a re-expression rather than independent evidence'
              if (r.cls == 'car' and r.dirn == 'in' and r.stat == 'b_free_diff') else ''))
PT = pd.read_csv(os.path.join(FIN, 'model_perm_tests.csv')); PT = PT[PT.proxy == 'causal (G5)'].set_index('stat')
for st, lab in (('car_in_dLL_M1p', 'M1p vs M0 (lane limits through the lane desired speed)'), ('car_in_dLL_M1', 'M1 (hard cap) vs M0'),
                ('car_in_dLL_M1p_M1', 'M1p vs M1 (channel comparison)'), ('car_out_dLL_M1', 'car outward: M1 vs M0'), ('car_out_dLL_M1p', 'car outward: M1p vs M0')):
    r = PT.loc[st]
    add(blk, 'car', 'in' if 'car_in' in st else 'out', f'pair permutation of (V_in, V_out) over days, in-sample log-lik gain {lab}: one-sided p', r.p_one_upper,
        note=f'observed gain {r.observed:.2f}; {int(r.n_draws)} draws (Monte Carlo SE {r.mc_se_p:.3f})')
RS = pd.read_csv(os.path.join(FIN, 'model_rec_summary.csv')).iloc[0]
add(blk, 'car', 'in', 'not recommended: M1p + free + free x diff (M1pF): beta_F', RS.b_free_x_diff, RS.b_free_x_diff - 1.96 * RS.se_free_x_diff, RS.b_free_x_diff + 1.96 * RS.se_free_x_diff,
    'logit', note=f'a_bias = {RS.b_free_x_diff / RS.b_dA_ego:.2f} m/s2; exact 56-labelling p two-sided {RS.b_free_diff_p_two:.2f} (one-sided {RS.b_free_diff_p_one:.3f}); '
                  f'LODO {RS.LODO:.1f} vs M1p -666.3; keeps a leader-driven response x2.10 that is not observed (x1.03)')
add(blk, 'car', 'out', 'car outward model: limited discrimination', np.nan,
    note='LODO calibration: observed shares flat over the lowest 90 % of predictions (Spearman over deciles 1-9 = -0.25); discrimination comes from the top decile. '
         '7 of 100 changes (all lateral-confirmed) occur at predicted 5-s probability below the 10th percentile: 4 move behind a slower lane-2 vehicle 35-53 m ahead, '
         '2 have a slower lane-1 leader within 150 m, 1 yields to a closing lane-1 follower (model_outward_lowpred_events.csv); treat block C as the check on keep-right behaviour')
add(blk, 'truck', 'out', 'heavy outward model: no discrimination (LODO calibration slope 0.1)', np.nan, note='use the block-C return probabilities for heavy vehicles')

# ---------------- B: free-regime hazard (step in Delta)
T = pd.read_csv(os.path.join(RES, 'sit_tests_primary.csv'))
DR = pd.read_csv(os.path.join(RES, 'sit_day_rates.csv'))
for tr, cls in ((0, 'car'), (1, 'truck')):
    t = T[(T.truck == tr) & (T.type == 'IN_free') & (T.spec == 'primary')].iloc[0]
    dr = DR[(DR.truck == tr) & (DR.type == 'IN_free')]
    same = dr[dr.delta == 0].iloc[:3]
    diff = dr[dr.delta > 0]
    blk = 'B free-regime inward hazard (empirical)'
    add(blk, cls, 'in', 'rate on same-limit days (mean of 3 days)', t.same, unit='per km driven free in lane 2',
        n=f'{int(t.n_same)} changes / {t.km_same:.0f} km', note=f'day range {same.rate.min():.3f}-{same.rate.max():.3f}')
    add(blk, cls, 'in', 'rate on differentiated days (mean of 5 days), Delta = 10-30 km/h', t['diff'], unit='per km driven free in lane 2',
        n=f'{int(t.n_diff)} changes / {t.km_diff:.0f} km', note=f'day range {diff.rate.min():.3f}-{diff.rate.max():.3f}; a step, no gradient within Delta = 10-30 (single Delta = 30 day)')
    add(blk, cls, 'in', 'step multiplier differentiated / same (quasi-Poisson over days)', t.qp_ratio, t.qp_lo, t.qp_hi, 'x',
        note=f'ratio of day means {t.ratio:.2f}; exact day-level p two-sided {t.p2_8:.3f}, one-sided {t.p1_8:.3f}')
    if tr == 0:
        for lab, v in (('same-limit', t.same), ('differentiated', t['diff'])):
            add(blk, cls, 'in', f'equivalent 1-s probability at 80 km/h, {lab}', v * 80 / 3600, unit='per s', note='rate per km x 80/3600 km per s')
P1 = pd.read_csv(os.path.join(RES, 'sit_car_probability.csv'))
ps, pdf = P1[P1.delta == 0].iloc[:3], P1[P1.delta > 0]
add('B free-regime inward hazard (empirical)', 'car', 'in', 'probability that a car free in lane 2 at 1,050 m moves in freely before 1,500 m, same-limit', ps.prob.mean(),
    n=f'{int(ps.moved_free_before_1500.sum())}/{int(ps.cars_free_lane2_at_1050.sum())} cars', note=f'days {", ".join(f"{v:.3f}" for v in ps.prob)}')
add('B free-regime inward hazard (empirical)', 'car', 'in', 'same probability, differentiated', pdf.prob.mean(),
    n=f'{int(pdf.moved_free_before_1500.sum())}/{int(pdf.cars_free_lane2_at_1050.sum())} cars', note=f'days {", ".join(f"{v:.3f}" for v in pdf.prob)}; exact p = 0.036')
t = T[(T.truck == 0) & (T.type == 'IN_overtake') & (T.spec == 'primary')].iloc[0]
add('B free-regime inward hazard (empirical)', 'car', 'in', 'contrast: leader-driven inward rate, differentiated / same', t.qp_ratio, t.qp_lo, t.qp_hi, 'x',
    note=f'same {t.same:.2f}, differentiated {t["diff"]:.2f} per km in that state; exact p two-sided {t.p2_8:.2f}; keep one leader-driven hazard for all configurations')

# ---------------- C: keep-right / return by class
blk = 'C keep-right / return (empirical)'
MI = pd.read_csv(os.path.join(RES, 'check_misc.csv'))
r = MI[MI.check == 'entry_run_leave_per_km'].iloc[0]
n_stints = int(MI[MI.check == 'entry_run_leave_KM600'].detail.iloc[0].split(',')[0].split('=')[1])
add(blk, 'car', 'out', 'cars entering the window in lane 1: leaving rate, same-limit', r.same_mean, unit='per km in lane 1',
    note=f'differentiated {r.diff_mean:.3f}; exact p two-sided {r.p2_8:.2f}; {n_stints:,} stints; one value for all configurations')
r = MI[MI.check == 'entry_run_leave_KM600'].iloc[0]
add(blk, 'car', 'out', 'cars entering in lane 1: probability of leaving within 600 m (Kaplan-Meier), same-limit', r.same_mean, note=f'differentiated {r.diff_mean:.3f}; p = {r.p2_8:.2f}')
RR = pd.read_csv(os.path.join(RES, 'check_stint_postpass_rr_subsets.csv')).set_index('subset')
x = RR.loc['lc_started']
add(blk, 'car', 'out', 'return to lane 2 within 500 m after passing, stints opened by an inward change (lateral-confirmed)', x.car_ret / x.car_n,
    n=f'{int(x.car_ret)}/{int(x.car_n)} stints', note='raw share (censoring at the window exit ignored)')
add(blk, 'truck', 'out', 'same, heavy vehicles', x.truck_ret / x.truck_n, n=f'{int(x.truck_ret)}/{int(x.truck_n)} stints')
add(blk, 'truck vs car', 'out', 'rate ratio of return after passing, heavy / car (Poisson, stints opened by an inward change)', x.RR_truck_car, x.lo, x.hi, 'x')
RT = pd.read_csv(os.path.join(RES, 'check_stint_return_tests.csv'))
r = RT[(RT.variant == 'V1') & (RT.H == 600)].iloc[0]
add(blk, 'car', 'out', 'car return within 600 m after an inward change (KM), same-limit', r.daymean_same,
    note=f'differentiated {r.daymean_diff:.3f}; p two-sided {r.p2_8:.2f}; definition dependent: inconclusive')
for tr, cls in ((0, 'car'), (1, 'truck')):
    t = T[(T.truck == tr) & (T.type == 'OUT_keepright') & (T.spec == 'primary')].iloc[0]
    add(blk, cls, 'out', 'settled keep-right rate (no leader, no follower pressure), same-limit', t.same, unit='per km in lane 1',
        note=f'differentiated {t["diff"]:.3f}; ratio {t.qp_ratio:.2f} ({t.qp_lo:.2f}-{t.qp_hi:.2f}); p {t.p2_8:.2f}')

# ---------------- D: simulation inputs (desired speeds; fitted on the three same-limit days; inputs only)
blk = 'D simulation inputs: desired speed'
FITS = pd.read_csv(os.path.join(FIN, 'model_desired_speed_fits.csv'))
fp = lambda c, v, p_: float(FITS[(FITS.cls == c) & (FITS.variant == v) & (FITS.param == p_)].value.iloc[0])
for c in ('car', 'heavy'):
    add(blk, c, 'both', 'absolute convention (ABS): desired speed ~ N(mean, sd), used by LC2013 ABS runs and by closed-loop M2F', fp(c, 'ABS', 'vmax mean (km/h)'), unit='km/h',
        note=f"sd {fp(c, 'ABS', 'vmax sd (km/h)'):.1f} km/h; SUMO speedFactor = {fp(c, 'ABS', 'tolerance k (cap = k x lane limit)'):.2f} (the lane limit times this factor caps the desired speed); "
             f"reproduces free-flow speed deciles of the 3 same-limit days with rmse {fp(c, 'ABS', 'decile rmse (km/h)'):.1f} km/h; a simulation input, not a behavioural finding")
    add(blk, c, 'both', 'speed-factor convention (REL): desired speed = speedFactor x lane limit, speedFactor ~ N(mean, sd)', fp(c, 'REL', 'speedFactor mean'), unit='-',
        note=f"sd {fp(c, 'REL', 'speedFactor sd'):.2f}; rmse {fp(c, 'REL', 'decile rmse (km/h)'):.1f} km/h" +
             ('; heavy vehicles additionally capped by the ABS distribution' if c == 'heavy' else '') + '; used by LC2013 REL runs and by closed-loop M1p')

# ---------------- E: SUMO LC2013
blk = 'E SUMO LC2013 (5 seeds, replayed demand)'
R = pd.read_csv(os.path.join(FIN, 'sim_ratio_summary.csv')).set_index('series')
GL = pd.read_csv(os.path.join(FIN, 'sim_grid_loss.csv')); b0 = GL.iloc[0]
GLC = pd.read_csv(os.path.join(FIN, 'sim_classlc_loss.csv')); c0 = GLC.iloc[0]
MC = pd.read_csv(os.path.join(FIN, 'sim_mechanism_check.csv')).set_index('series')


def simnote(s):
    g = R.loc[s]; m = MC.loc[s] if s in MC.index else None
    t = (f'sim/obs geometric mean, same-limit / differentiated days: car inward per lane-2 km x{g.car_rate_in_gm_same:.2g} / x{g.car_rate_in_gm_diff:.2g} '
         f'(differentiated days x{g.car_rate_in_diff_day_min:.2g}-{g.car_rate_in_diff_day_max:.2g}); P(move in by 1,800 m) x{g.car_p_in_by1800_gm_same:.2g} / x{g.car_p_in_by1800_gm_diff:.2g}; '
         f'car outward x{g.car_rate_out_gm_same:.2g} / x{g.car_rate_out_gm_diff:.2g}; heavy lane-1 km share x{g.heavy_share_km_L1_gm_same:.2g} / x{g.heavy_share_km_L1_gm_diff:.2g}; '
         f'entry-to-exit car inner-lane share {g.car_shift_sim_same:+.1f} / {g.car_shift_sim_diff:+.1f} pp (observed +0.7 / +6.0), mean abs. error {g.car_shift_mae_pp:.1f} pp')
    if m is not None:
        t += f'; free share of car inward changes {m.free_share_same:.2f} / {m.free_share_diff:.2f} (observed 0.46 / 0.67)'
    return t


add(blk, 'car', 'both', 'default LC2013, speed-factor convention', R.loc['LC2013_REL_sg1_kr1'].car_rate_in_gm_diff, unit='x (car inward, differentiated)', note=simnote('LC2013_REL_sg1_kr1'))
add(blk, 'car', 'both', 'default LC2013, absolute convention', R.loc['LC2013_ABS_sg1_kr1'].car_rate_in_gm_diff, unit='x (car inward, differentiated)', note=simnote('LC2013_ABS_sg1_kr1'))
add(blk, 'car', 'both', f'best single set of 40 (5 lcSpeedGain x 4 lcKeepRight x 2 conventions): {b0.variant}, lcSpeedGain {b0.lcSpeedGain:g}, lcKeepRight {b0.lcKeepRight:g}',
    R.loc[b0.series].car_rate_in_gm_diff, unit='x (car inward, differentiated)', note=simnote(b0.series) + '; loss = mean squared log ratio of car inward + car outward rates over 8 days')
add(blk, 'heavy', 'both', f'class-specific heavy-vehicle set on top of the best car set: lcSpeedGain {c0.hvSpeedGain:g}, lcKeepRight {c0.hvKeepRight:g} (10 heavy sets tried)',
    R.loc[c0.series].heavy_share_km_L1_gm_diff, unit='x (heavy lane-1 km share, differentiated)', note=simnote(c0.series))
add(blk, 'all', 'both', 'caveat', np.nan, note='LC2013 has no lane-preference term for unconstrained vehicles; its response to Delta arises from desired speeds tied to the limits (REL) or '
                                               'from caps that bind (ABS at 80/60); the recommended free-regime bias can be added through TraCI (block F)')

# ---------------- F: closed-loop stochastic MOBIL (SUMO + TraCI)
blk = 'F closed-loop stochastic MOBIL (SUMO 1.18 + TraCI, 5 seeds)'
add(blk, 'all', 'both', 'implementation', np.nan,
    note='SUMO autonomous lane changing off (laneChangeMode 512; TraCI requests respect the safe gaps of others); every 1 s each vehicle in x = 1,000-2,000 m gets the 1-s hazard of '
         'block A from its simulated neighbours (same IDM/MOBIL code as the estimation) and changes lane with that probability; leave-one-day-out coefficients')
add(blk, 'car', 'both', 'recommended M2F (cars in), M0 (cars out, heavy vehicles), absolute desired speeds', R.loc['MOBIL_CL_M2F'].car_rate_in_gm_diff, unit='x (car inward, differentiated)',
    note=simnote('MOBIL_CL_M2F'))
add(blk, 'car', 'both', 'alternative M1p (cars in and out), M0 (heavy vehicles), speed-factor desired speeds', R.loc['MOBIL_CL_M1p'].car_rate_in_gm_diff, unit='x (car inward, differentiated)',
    note=simnote('MOBIL_CL_M1p') + '; reproduces the lane-share gain through fewer outward changes (outward rate differentiated/same x0.35 vs observed x0.84), not through free inward changes')

Rt = pd.DataFrame(rows)
Rt.to_csv(os.path.join(FIN, 'sim_parameters.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
print(Rt[['block', 'vehicle_class', 'direction', 'model', 'parameter', 'estimate', 'ci95_lo', 'ci95_hi']].round(4).to_string(index=False))
