# -*- coding: utf-8 -*-
"""Inner-lane stints and a 1 Hz panel of them. A stint is a stay of a vehicle in lane 1 that starts at an analysis-grade inward change
(kind from core_events.mtype0) or at entry in lane 1 (first frame in lane 1 at x < 1,050 m) and ends at the vehicle's next analysis-grade
outward change (return) or at the end of its track (censored). Per stint: passes of lane-2 vehicles made from lane 1, the last pass, and
the clearance to the lane-2 vehicle behind at the return. Panel: one row per stint-second with the traffic around the vehicle and the
passes made so far.
Inputs: session frames and events; _cache/work/core_events.pkl, core_vehicles.pkl, core_passes.pkl.
Outputs: _cache/work/stints.pkl, stint_panel.pkl; results/stints.csv."""
import os, sys, numpy as np, pandas as pd
from lc_common import *

E = pd.read_pickle(os.path.join(CACHE, 'core_events.pkl'))
V = pd.read_pickle(os.path.join(CACHE, 'core_vehicles.pkl'))
P = pd.read_pickle(os.path.join(CACHE, 'core_passes.pkl'))


def kind_of(m):
    if m.startswith('in_overtake'):
        return 'overtake'
    if m in ('in_free_noleader', 'in_free_distant_leader', 'in_free_unknown_view'):
        return 'free'
    return 'other'


def nearest(M, lane, ahead):
    """nearest other vehicle per point in `lane`, ahead (dx>0) or behind (dx<=0)."""
    s = M[(M.Lane_o == lane) & ((M.dx > 0) if ahead else (M.dx <= 0))].copy()
    s['adx'] = s.dx.abs()
    s = s.sort_values(['pid', 'adx']).drop_duplicates('pid')
    return s.set_index('pid')


stints_all, panel_all = [], []
for n, vi, vo in DAYS:
    f = frames(n)
    f12 = f[f.Lane.isin([1, 2])][['GlobalID', 'Timestamp', 'PositionX', 'Lane', 'Smoothed_VelocityX', 'truck', 'frame_source']].copy()
    f12['v'] = f12.Smoothed_VelocityX.astype(float) * 3.6
    ae = lc_events(n, grade=False)                      # all lane changes (for the non-analysis-grade sensitivity)
    ae = ae[ae.from_lane.isin([1, 2]) & ae.to_lane.isin([1, 2])]
    e = E[E.day == n].sort_values(['GlobalID', 't'])
    vd = V[V.day == n].set_index('GlobalID')
    pp = P[(P.day == n) & (P.lane_passer == 1) & (P.lane_passed == 2)]
    byV = {g: d for g, d in f12.groupby('GlobalID')}
    rows = []
    # ---------------- stints
    for gid, d in byV.items():
        ev = e[e.GlobalID == gid]
        starts = []
        if d.Lane.iloc[0] == 1 and d.PositionX.iloc[0] < 1050:
            starts.append(('entry', 'entry', int(d.Timestamp.iloc[0]), float(d.PositionX.iloc[0]), np.nan))
        for r in ev[ev.direction == 'in'].itertuples():
            starts.append(('lc', kind_of(r.mtype0), int(r.t), float(r.x), r.Index))
        for st, kind, t0, x0, eidx in starts:
            nxt = ev[ev.t > t0]
            track_t1, track_x1 = int(d.Timestamp.iloc[-1]), float(d.PositionX.iloc[-1])
            if len(nxt) and nxt.direction.iloc[0] == 'out':
                t1, x1, ret = int(nxt.t.iloc[0]), float(nxt.x.iloc[0]), 1
                ret_eidx = nxt.index[0]
            else:
                t1, x1, ret, ret_eidx = track_t1, track_x1, 0, np.nan
                if len(nxt):          # next AG event is inward (should not happen): censor at it
                    t1, x1 = int(nxt.t.iloc[0]), float(nxt.x.iloc[0])
            # non-analysis-grade outward change after start and before t1 (sensitivity: treat as return)
            na = ae[(ae.GlobalID == gid) & (ae.to_lane == 2) & (ae.timestamp_ms > t0) & (ae.timestamp_ms < t1 + (1 if ret == 0 else 0))]
            na = na[na.analysis_grade == 0]
            q = pp[(pp.passer == gid) & (pp.t >= t0 - 3000) & (pp.t <= t1)]
            row = dict(day=n, cfg=f'{vi}/{vo}', delta=vi - vo, GlobalID=gid, truck=int(d.truck.max()), start=st, kind=kind,
                       t_start=t0, x_start=x0, t_end=t1, x_end=x1, ret=ret, dist=x1 - x0, dur=(t1 - t0) / 1000.0,
                       track_x1=track_x1, n_pass=len(q), ev_in=eidx, ev_out=ret_eidx,
                       nonag_out=int(len(na) > 0), nonag_out_t=int(na.timestamp_ms.iloc[0]) if len(na) else np.nan,
                       nonag_out_x=float(na.position_x_m.iloc[0]) if len(na) else np.nan,
                       v_des=vd.v_des.get(gid, np.nan))
            if len(q):
                ql = q.sort_values('t').iloc[-1]
                row.update(t_lastpass=int(ql.t), x_lastpass=float(ql.x), last_passed=ql.passed, last_passed_truck=int(ql.truck_passed),
                           t_firstpass=int(q.t.min()))
            rows.append(row)
    S = pd.DataFrame(rows)
    # ---------------- return clearance: at the recorded return time t_end, gap to the lane-2 follower and to the last-passed vehicle
    fi = f12.set_index(['GlobalID', 'Timestamp'])
    byT = {t: s for t, s in f12.groupby('Timestamp')}
    clr = []
    for r in S.itertuples():
        out = {}
        if r.ret == 1:
            for lag in (0, 1, 2):
                tau = r.t_end - lag * 1000
                snap = byT.get(tau)
                if snap is None or (r.GlobalID, tau) not in fi.index:
                    continue
                me = fi.loc[(r.GlobalID, tau)]
                o = snap[(snap.GlobalID != r.GlobalID) & (snap.Lane == 2) & (snap.PositionX <= me.PositionX)]
                if len(o):
                    b = o.loc[o.PositionX.idxmax()]
                    out[f'lag2_gap{lag}'] = float(me.PositionX - b.PositionX); out[f'lag2_v{lag}'] = float(b.v); out[f'lag2_id{lag}'] = b.GlobalID
                    out[f'lag2_truck{lag}'] = int(b.truck)
                out[f'ego_v{lag}'] = float(me.v); out[f'ego_meas{lag}'] = int(me.frame_source <= 1)
                if pd.notna(getattr(r, 'last_passed', np.nan)) and (r.last_passed, tau) in fi.index:
                    lp = fi.loc[(r.last_passed, tau)]
                    out[f'lp_gap{lag}'] = float(me.PositionX - lp.PositionX); out[f'lp_v{lag}'] = float(lp.v); out[f'lp_lane{lag}'] = int(lp.Lane)
        clr.append(out)
    S = pd.concat([S.reset_index(drop=True), pd.DataFrame(clr)], axis=1)
    S['sid'] = [f'{n[10:15]}_{i}' for i in range(len(S))]
    # ---------------- 1 Hz panel (lane-1 frames of the stint, tau on the stint's own 1 s grid)
    pts = []
    for r in S.itertuples():
        d = byV[r.GlobalID]
        end = r.t_end if r.ret == 1 else r.t_end + 1
        g = d[(d.Timestamp >= r.t_start) & (d.Timestamp < end) & ((d.Timestamp - r.t_start) % 1000 == 0)]
        g = g[g.Lane == 1]
        if len(g) == 0:
            continue
        g = g[['GlobalID', 'Timestamp', 'PositionX', 'v', 'frame_source']].copy()
        g['sid'] = r.sid
        g['y'] = 0
        if r.ret == 1:
            g.loc[g.index[-1], 'y'] = int(r.t_end - g.Timestamp.iloc[-1] <= 1000)    # return recorded within 1 s of the last grid point
        pts.append(g)
    Pn = pd.concat(pts, ignore_index=True)
    Pn['pid'] = np.arange(len(Pn))
    M = Pn[['pid', 'GlobalID', 'Timestamp', 'PositionX', 'v']].merge(
        f12[['GlobalID', 'Timestamp', 'PositionX', 'Lane', 'v', 'truck']].rename(columns={'GlobalID': 'gid_o', 'PositionX': 'x_o', 'Lane': 'Lane_o', 'v': 'v_o', 'truck': 'truck_o'}),
        on='Timestamp')
    M = M[M.gid_o != M.GlobalID]
    M['dx'] = M.x_o - M.PositionX
    M = M[M.dx.abs() <= 400]
    for tag, lane, ahead in (('a2', 2, True), ('b2', 2, False), ('a1', 1, True), ('b1', 1, False)):
        s = nearest(M, lane, ahead)
        Pn[f'gap_{tag}'] = Pn.pid.map(s.adx)
        # dv > 0 means "closing": for ahead = ego faster; for behind = other faster
        dv = (s.v - s.v_o) if ahead else (s.v_o - s.v)
        Pn[f'dv_{tag}'] = Pn.pid.map(dv); Pn[f'truck_{tag}'] = Pn.pid.map(s.truck_o)
    a2 = M[(M.Lane_o == 2) & (M.dx > 0)]
    for L in (150, 300):
        Pn[f'n_a2_{L}'] = Pn.pid.map(a2[a2.dx <= L].groupby('pid').size()).fillna(0).astype(int)
        slow = a2[(a2.dx <= L) & (a2.v_o < a2.v - 3)]
        Pn[f'n_a2slow_{L}'] = Pn.pid.map(slow.groupby('pid').size()).fillna(0).astype(int)
    # passes made so far in the stint, time since last pass
    Pn = Pn.merge(S[['sid', 'day', 'cfg', 'delta', 'truck', 'start', 'kind', 't_start', 'x_start', 'ret', 'v_des']], on='sid')
    Pn['cum_pass'] = 0; Pn['since_pass'] = np.nan
    for sid, idx in Pn.groupby('sid').groups.items():
        gid = Pn.at[idx[0], 'GlobalID']; t0 = Pn.at[idx[0], 't_start']
        q = pp[(pp.passer == gid) & (pp.t >= t0 - 3000)].t.values
        if len(q) == 0:
            continue
        tt = Pn.loc[idx, 'Timestamp'].values
        c = np.searchsorted(np.sort(q), tt, side='right')
        Pn.loc[idx, 'cum_pass'] = c
        last = np.where(c > 0, np.sort(q)[np.maximum(c - 1, 0)], np.nan)
        Pn.loc[idx, 'since_pass'] = np.where(c > 0, (tt - last) / 1000.0, np.nan)
    Pn['k'] = (Pn.Timestamp - Pn.t_start) // 1000
    Pn['dist'] = Pn.PositionX - Pn.x_start
    Pn = Pn.sort_values(['sid', 'Timestamp']).reset_index(drop=True)
    Pn['dx_next'] = Pn.groupby('sid').PositionX.shift(-1) - Pn.PositionX
    Pn['dx_next'] = Pn.dx_next.fillna(Pn.v / 3.6)
    stints_all.append(S); panel_all.append(Pn.drop(columns=['pid']))
    print(n[10:], 'stints', len(S), 'returns', int(S.ret.sum()), 'panel', len(Pn), 'y', int(Pn.y.sum()), flush=True)

S = pd.concat(stints_all, ignore_index=True); Pn = pd.concat(panel_all, ignore_index=True)
for T in (S, Pn):
    T['diff_day'] = (T.delta > 0).astype(int)
S.to_pickle(os.path.join(CACHE, 'stints.pkl')); Pn.to_pickle(os.path.join(CACHE, 'stint_panel.pkl'))
S.to_csv(os.path.join(RES, 'stints.csv'), index=False)
print('stints', S.shape, 'panel', Pn.shape)
print(S.groupby(['truck', 'kind']).agg(n=('ret', 'size'), ret=('ret', 'sum')))
print('returns without y=1 in panel:', int(S.ret.sum() - Pn.y.sum()))
