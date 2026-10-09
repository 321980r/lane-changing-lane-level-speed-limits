# -*- coding: utf-8 -*-
"""Checks of the inner-lane stints after an inward change (Section 5.3):
 (1) lateral-confirmed stints by the situation of the opening change (leader-driven: mtype0 'in_overtake_*'; unconstrained: 'in_free_*'):
     returns per km, Kaplan-Meier share back by 600 m, heavy/car rate ratios (Poisson with day effects);
 (2) heavy stints opened by an unconstrained inward change: passes during the stint, a lane-2 vehicle alongside at the change, an opposite
     change of another vehicle within +-3 s and +-50 m (possible lane-label swap), lateral excursion;
 (3) cars already in the inner lane at entry, compared with the entry runs of check_misc.py (1,172 runs, 95 leaves);
 (4) heavy and car leaders of car inward changes behind a leader (< 150 m): all changes and x >= 1,050 m;
 (5) example unconstrained inward changes of cars on the days with Delta = 10 and 20 km/h.
Inputs: _cache/work/stints.pkl, lateral_check.pkl, core_events.pkl, core_vehicles.pkl; results/check_stint_seq_runs_lat.csv; session frames and
events. Outputs: results/final/stays_checks.csv, stays_checks_heavy_free.csv, stays_checks_s8_candidates.csv."""
import os, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from plot_style import *
warnings.filterwarnings('ignore')
out = []


def rec(k, v, note=''):
    out.append(dict(check=k, value=v, note=note)); print(f'{k:70s} {v}  {note}')


def km_p(d, e, h):
    d = np.asarray(d, float); e = np.asarray(e, int); s = 1.0
    for u in np.unique(d[(e == 1) & (d <= h)]):
        n = (d >= u).sum(); k = ((d == u) & (e == 1)).sum(); s *= 1 - k / n
    return 1 - s


def rr(q):
    q = q.assign(lkm=np.log(q.km), g=pd.factorize(q.day + '_' + q.GlobalID.astype(str))[0])
    m = smf.glm('ret_c ~ truck + day', data=q, family=sm.families.Poisson(), offset=q.lkm).fit(cov_type='cluster', cov_kwds={'groups': q.g})
    ci = np.exp(m.conf_int().loc['truck'].values)
    return f"{np.exp(m.params['truck']):.2f} ({ci[0]:.2f}-{ci[1]:.2f})"


# ---------------------------------------------------------------------------------------------------- (1) stints by regime
S = pd.read_pickle(os.path.join(CACHE, 'stints.pkl'))
L = pd.read_pickle(os.path.join(CACHE, 'lateral_check.pkl'))
LK = L.set_index(['day', 'GlobalID', 't'])
S = S[S.truck >= 0].copy()
S['in_ok'] = [LK.lat_ok.get((r.day, r.GlobalID, r.t_start), np.nan) if r.start == 'lc' else 1 for r in S.itertuples()]
S['out_ok'] = [LK.lat_ok.get((r.day, r.GlobalID, r.t_end), np.nan) if r.ret == 1 else np.nan for r in S.itertuples()]
S['ret_c'] = ((S.ret == 1) & (S.out_ok == 1)).astype(int)
S['km'] = S.dist.clip(lower=1.0) / 1000
E = pd.read_pickle(os.path.join(CACHE, 'core_events.pkl'))
EK = E.set_index(['day', 'GlobalID', 't'])
B = S[(S.in_ok == 1) & (S.start == 'lc')].copy()
B['mtype0'] = [EK.mtype0.get((r.day, r.GlobalID, r.t_start), 'NA') for r in B.itertuples()]
B['regime'] = np.where(B.mtype0.str.startswith('in_overtake'), 'leader', np.where(B.mtype0.str.startswith('in_free'), 'free', 'other'))
rec('stints with regime not found', int((B.regime == 'other').sum()))
for cls in (0, 1):
    q = B[B.truck == cls]
    rec(f'{"heavy" if cls else "car"} mtype0 breakdown', q.mtype0.value_counts().to_dict())
    for rg in ('leader', 'free'):
        z = q[q.regime == rg]
        rec(f'{"heavy" if cls else "car"} {rg}: n / veh / returns / km / per km / KM600 / censored / median x_start',
            f'{len(z)} / {z.groupby("day").GlobalID.nunique().sum()} / {z.ret_c.sum()} / {z.km.sum():.1f} / {z.ret_c.sum() / z.km.sum():.3f} / '
            f'{km_p(z.dist.clip(lower=1), z.ret_c, 600):.3f} / {1 - z.ret_c.mean():.2f} / {z.x_start.median():.0f}')
rec('RR heavy/car all confirmed stints', rr(B))
rec('RR heavy/car leader-driven stints only', rr(B[B.regime == 'leader']))
rec('RR heavy leader-driven / car all', rr(B[(B.truck == 0) | (B.regime == 'leader')]))
rec('RR heavy/car unconstrained stints only', rr(B[B.regime == 'free']))
q = B[B.truck == 0].assign(truck=lambda x: (x.regime == 'leader').astype(int))
rec('RR car leader-driven / car unconstrained', rr(q))
q = B[B.truck == 1].assign(truck=lambda x: (x.regime == 'free').astype(int))
rec('RR heavy unconstrained / heavy leader-driven', rr(q))
cen = B[B.truck == 0]
rec('car LC stints censored share / median x_start', f'{1 - cen.ret_c.mean():.3f} / {cen.x_start.median():.0f}')
rec('car LC stints: IQR x_start', f'{cen.x_start.quantile(.25):.0f}-{cen.x_start.quantile(.75):.0f}')
rec('car LC stints: median distance followed', f'{cen.dist.median():.0f}')

# ---------------------------------------------------------------------------------------------------- (2) heavy unconstrained stints
H = B[B.truck == 1].copy()
rows = []
cache_f, cache_e = {}, {}
for r in H.itertuples():
    if r.day not in cache_f:
        f = frames(r.day)[['GlobalID', 'Timestamp', 'PositionX', 'Lane', 'truck', 'frame_source']]
        cache_f[r.day] = f
        cache_e[r.day] = lc_events(r.day, grade=False)
    f, ea = cache_f[r.day], cache_e[r.day]
    ego = f[(f.GlobalID == r.GlobalID) & (f.Timestamp == r.t_start)]
    x0 = float(ego.PositionX.iloc[0]) if len(ego) else r.x_start
    snap = f[(f.Timestamp == r.t_start) & (f.GlobalID != r.GlobalID) & (f.Lane == 2)]
    dx = snap.PositionX - x0
    along = snap[dx.abs() <= 30]
    ahead300 = snap[(dx > 0) & (dx <= 300)]
    # mirror change: another vehicle changes 1->2 within +-3 s and +-50 m of the recorded inward change
    tcol, xcol = 'timestamp_ms', 'position_x_m'
    opp = ea[(ea.from_lane == 1) & (ea.to_lane == 2)]
    mir = opp[(opp.GlobalID != r.GlobalID) & ((opp[tcol] - r.t_start).abs() <= 3000) & ((opp[xcol] - r.x_start).abs() <= 50)]
    lat = LK.loc[(r.day, r.GlobalID, r.t_start)]
    rows.append(dict(day=r.day, GlobalID=r.GlobalID, t_start=r.t_start, x_start=r.x_start, regime=r.regime, mtype0=r.mtype0,
                     ret_c=r.ret_c, dist=r.dist, n_pass=r.n_pass, t_firstpass_rel=(r.t_firstpass - r.t_start) / 1000 if pd.notna(r.t_firstpass) else np.nan,
                     last_passed_truck=r.last_passed_truck, n_lane2_alongside30=len(along), n_lane2_ahead300_at_t=len(ahead300),
                     alongside_truck=int((along.truck == 1).any()) if len(along) else 0, n_mirror_change=len(mir),
                     lat_new_ext=lat.lat_new_ext, meas_new=lat.meas_new, lat_ok_both=lat.lat_ok_both))
HF = pd.DataFrame(rows)
HF.to_csv(os.path.join(DATF, 'stays_checks_heavy_free.csv'), index=False)
for rg in ('leader', 'free'):
    z = HF[HF.regime == rg]
    rec(f'heavy {rg}: n / with a pass during stint / lane-2 vehicle alongside (+-30 m) at change / mirror change (+-3 s, +-50 m) / lat_ok_both',
        f'{len(z)} / {(z.n_pass > 0).sum()} / {(z.n_lane2_alongside30 > 0).sum()} / {(z.n_mirror_change > 0).sum()} / {int(z.lat_ok_both.sum())}')
    rec(f'heavy {rg}: median lateral excursion into new lane (m) / median measured share', f'{z.lat_new_ext.median():.2f} / {z.meas_new.median():.2f}')
    rec(f'heavy {rg}: x_start <1050 / 1350-1550', f'{(z.x_start < 1050).sum()} / {((z.x_start >= 1350) & (z.x_start < 1550)).sum()}')
fl = HF[HF.regime == 'free']
susp = fl[(fl.n_mirror_change > 0) | ((fl.n_lane2_alongside30 > 0) & (fl.n_pass == 0))]
rec('heavy unconstrained stints flagged as possible label swaps (mirror change, or alongside without a pass)', f'{len(susp)} of {len(fl)}')
B2 = B.merge(HF[['day', 'GlobalID', 't_start', 'n_mirror_change', 'n_lane2_alongside30', 'n_pass']], on=['day', 'GlobalID', 't_start'], how='left',
             suffixes=('', '_h'))
flag = (B2.truck == 1) & (B2.regime == 'free') & ((B2.n_mirror_change > 0) | ((B2.n_lane2_alongside30 > 0) & (B2.n_pass == 0)))
rec('RR heavy/car without flagged heavy stints', rr(B2[~flag]), f'{int(flag.sum())} heavy stints dropped')
z = B2[(B2.truck == 1) & ~flag]
rec('heavy without flagged: n / returns / km / per km / KM600', f'{len(z)} / {z.ret_c.sum()} / {z.km.sum():.1f} / {z.ret_c.sum() / z.km.sum():.3f} / {km_p(z.dist.clip(lower=1), z.ret_c, 600):.3f}')
zf = B2[(B2.truck == 1) & (B2.regime == 'free') & ~flag]
rec('heavy unconstrained without flagged: n / returns / km / per km', f'{len(zf)} / {zf.ret_c.sum()} / {zf.km.sum():.1f} / {zf.ret_c.sum() / zf.km.sum():.3f}')
zfp = HF[(HF.regime == 'free')]
rec('heavy unconstrained: share that passed a vehicle during the stint / median time to first pass (s)',
    f'{(zfp.n_pass > 0).mean():.2f} / {zfp.t_firstpass_rel.median():.1f}')

# ---------------------------------------------------------------------------------------------------- (3) entry baseline
ent = S[(S.in_ok == 1) & (S.truck == 0) & (S.start == 'entry')]
rec('car entry stints (this figure): n / returns (AG + lat_ok) / km / per km / KM600',
    f'{len(ent)} / {ent.ret_c.sum()} / {ent.km.sum():.1f} / {ent.ret_c.sum() / ent.km.sum():.3f} / {km_p(ent.dist.clip(lower=1), ent.ret_c, 600):.3f}')
rec('car entry: returns AG but not lat_ok (censored here)', int(((ent.ret == 1) & (ent.out_ok != 1)).sum()))
rec('car entry: x_start max', f'{ent.x_start.max():.0f}')
R = pd.read_csv(os.path.join(RES, 'check_stint_seq_runs_lat.csv'))
V = pd.read_pickle(os.path.join(CACHE, 'core_vehicles.pkl'))
R = R.sort_values(['day', 'GlobalID', 't_start']).reset_index(drop=True)
R['next_pseudo'] = R.groupby(['day', 'GlobalID']).pseudo.shift(-1)
R = R.merge(V[['day', 'GlobalID', 'truck']], on=['day', 'GlobalID'])
er = R[(R.lane == 1) & (R.start_type == 'entry') & (R.truck == 0)].copy()
er['dist'] = (er.x_end - er.x_start).clip(lower=0.1)
er['ev'] = ((er.end_type == 'lc') & (er.next_pseudo != 1)).astype(int)
rec('verifier entry runs: n / leaves / km / per km / KM600 / max x_start',
    f'{len(er)} / {er.ev.sum()} / {er.dist.sum() / 1000:.1f} / {er.ev.sum() / er.dist.sum() * 1000:.3f} / {km_p(er.dist, er.ev, 600):.3f} / {er.x_start.max():.0f}')
mm = ent[['day', 'GlobalID']].assign(a=1).merge(er[['day', 'GlobalID']].assign(b=1), how='outer', on=['day', 'GlobalID'])
rec('entry vehicles only in figure set / only in verifier set', f'{int(mm.b.isna().sum())} / {int(mm.a.isna().sum())}')

# ---------------------------------------------------------------------------------------------------- (4) heavy and car leaders, with and without the first-capture zone
kT = kC = 0.0; nT = nC = 0; kT0 = kC0 = 0.0; nT0 = nC0 = 0
for n in ORDER8:
    f = frames(n); cls = f.groupby('GlobalID').truck.first()
    g2 = f[(f.Lane == 2) & (f.truck == 0) & (f.gap_m < 150)]
    lt = g2.leader_id.map(cls); km = g2.Smoothed_VelocityX.astype(float) * 0.1 / 1000
    kT0 += km[lt == 1].sum(); kC0 += km[lt == 0].sum()
    sel = g2.PositionX >= 1050
    kT += km[(lt == 1) & sel].sum(); kC += km[(lt == 0) & sel].sum()
    e = E[(E.day == n) & (E.truck == 0) & (E.direction == 'in') & (E.own_lead_gap3 < 150)]
    nT0 += (e.own_lead_truck3 == 1).sum(); nC0 += (e.own_lead_truck3 == 0).sum()
    e = e[e.x >= 1050]
    nT += (e.own_lead_truck3 == 1).sum(); nC += (e.own_lead_truck3 == 0).sum()
rec('heavy/car leader, all: events heavy/car, km heavy/car, ratio', f'{nT0}/{nC0}, {kT0:.1f}/{kC0:.1f}, {(nT0 / kT0) / (nC0 / kC0):.3f}')
rec('heavy/car leader, x>=1050 m: events heavy/car, km heavy/car, heavy share events / exposure, ratio',
    f'{nT}/{nC}, {kT:.1f}/{kC:.1f}, {nT / (nT + nC):.3f} / {kT / (kT + kC):.3f}, {(nT / kT) / (nC / kC):.3f}')

# ---------------------------------------------------------------------------------------------------- (5) example candidates on Delta 10/20 days
cand = []
for day in ORDER8[3:7]:
    f = frames(day)[['GlobalID', 'Timestamp', 'PositionX', 'Lane', 'frame_source', 'truck']]
    Ed = E[(E.day == day) & (E.truck == 0) & (E.direction == 'in') & (E.mtype0 == 'in_free_noleader')]
    Ed = Ed.merge(L[['day', 'GlobalID', 't', 'lat_ok']], on=['day', 'GlobalID', 't'], how='left')
    Ed = Ed[Ed.lat_ok == 1]
    for r in Ed.itertuples():
        w = f[(f.Timestamp >= r.t - 20000) & (f.Timestamp <= r.t + 20000)]
        ego = w[w.GlobalID == r.GlobalID][['Timestamp', 'PositionX', 'frame_source', 'Lane']]
        m = w[w.GlobalID != r.GlobalID].merge(ego[['Timestamp', 'PositionX']].rename(columns={'PositionX': 'xe'}), on='Timestamp')
        m['dx'] = m.PositionX - m.xe
        c10 = m[(m.Timestamp >= r.t - 10000) & (m.Timestamp <= r.t + 10000)]
        cand.append(dict(day=day, GlobalID=r.GlobalID, t=r.t, x=r.x, transition_s=r.transition_s, next_dir=r.next_dir,
                         n_outer_ahead300_10=c10[(c10.Lane == 2) & (c10.dx > 0) & (c10.dx <= 300)].GlobalID.nunique(),
                         n_outer_ahead_any_10=c10[(c10.Lane == 2) & (c10.dx > 0)].GlobalID.nunique(),
                         n_near400_20=m[m.dx.abs() <= 400].GlobalID.nunique(),
                         meas_share=float((ego.frame_source <= 1).mean()) if len(ego) else np.nan,
                         t_obs_before=(r.t - ego.Timestamp.min()) / 1000 if len(ego) else np.nan,
                         t_obs_after=(ego.Timestamp.max() - r.t) / 1000 if len(ego) else np.nan))
CA = pd.DataFrame(cand)
CA['clean'] = ((CA.n_outer_ahead300_10 == 0) & CA.transition_s.notna()).astype(int)
CA = CA.sort_values(['clean', 'n_outer_ahead_any_10', 'meas_share'], ascending=[False, True, False])
CA.to_csv(os.path.join(DATF, 'stays_checks_s8_candidates.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(CA.head(20).to_string())
pd.DataFrame(out).to_csv(os.path.join(DATF, 'stays_checks.csv'), index=False)
