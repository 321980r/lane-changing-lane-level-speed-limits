# -*- coding: utf-8 -*-
"""Core tables of the eight days, built from the session frames and the lane-change events of the data set.

Outputs (pickles in _cache/work, CSV copies in results/core):
  core_vehicles   one row per vehicle-day: class, coverage, entry/exit lane, lane-run sequence, v_des proxy
  core_runs       one row per lane run (stint) of a vehicle in lane 1 or 2: start/end type, duration, passes made/received
  core_passes     one row per passing event (x-order flip between two co-present vehicles, 3 m hysteresis)
  core_events     one row per analysis-grade lane change 1<->2 with context at t-5/-3/-1 s, the fate of the original leader,
                  the vehicle's next/previous change, and a first manoeuvre label

Conventions: direction 'in' = 2->1 (towards inner/median lane), 'out' = 1->2. dx = x_other - x_ego (positive = other ahead).
dv = v_ego - v_other in km/h (positive = ego faster / closing on a vehicle ahead). Times in s, positions in corridor m."""
import os, sys, numpy as np, pandas as pd
from lc_common import *

OUT = os.path.join(RES, 'core'); os.makedirs(OUT, exist_ok=True)
HYST = 3.0          # m, hysteresis for a passing (order flip)
LOOK = 300.0        # m, fixed look-ahead used for "no leader within LOOK m"


# ----------------------------------------------------------------------------------------------------------- vehicles & runs
def vehicle_table(n, f):
    g = f.groupby('GlobalID')
    V = pd.DataFrame({'truck': g.truck.max(), 't0': g.Timestamp.min(), 't1': g.Timestamp.max(), 'x0': g.PositionX.min(), 'x1': g.PositionX.max(),
                      'n_frames': g.size(), 'meas_share': g.frame_source.apply(lambda s: float((s <= 1).mean()))})
    # entry / exit lane: modal lane in [1000,1100) and [1900,2000) (edge bins poorly measured)
    for nm, lo, hi in (('lane_entry', 1000, 1100), ('lane_exit', 1900, 2000)):
        s = f[(f.PositionX >= lo) & (f.PositionX < hi) & f.Lane.isin([1, 2])]
        V[nm] = s.groupby('GlobalID').Lane.agg(lambda x: int(x.mode().iloc[0])).reindex(V.index)
    # lane sequence from filtered lane runs (lanes 1/2 only)
    ff = f[f.Lane.isin([1, 2])]
    seq = ff.groupby('GlobalID').Lane.apply(lambda s: ''.join(str(int(v)) for v in s[s != s.shift()].values))
    V['seq_runs'] = seq.reindex(V.index)
    # v_des proxy: mean speed on free-flow frames (no leader or headway > 5 s), >= 20 frames; km/h
    fr = f[(f.leader_id.isna()) | (f.Headway > 5)]
    vv = fr.groupby('GlobalID').Smoothed_VelocityX.agg(['mean', 'size'])
    V['v_des'] = (vv['mean'] * 3.6).where(vv['size'] >= 20).reindex(V.index)
    # share of distance driven in lane 1
    d = f.Smoothed_VelocityX.astype(float) * 0.1
    V['dist_m'] = d.groupby(f.GlobalID).sum()
    V['dist_lane1_m'] = d[f.Lane == 1].groupby(f.GlobalID[f.Lane == 1]).sum().reindex(V.index).fillna(0)
    V['full_traverse'] = ((V.x0 < 1000) & (V.x1 > 2000)).astype(int)
    V = V.reset_index().rename(columns={'index': 'GlobalID'}); V.insert(0, 'day', n)
    return V


def run_table(n, f):
    """lane runs of lanes 1/2 (lane-3 frames dropped and ignored)."""
    ff = f[f.Lane.isin([1, 2])][['GlobalID', 'Timestamp', 'PositionX', 'Lane', 'Smoothed_VelocityX', 'segment_break_before']].copy()
    ff['new'] = (ff.GlobalID != ff.GlobalID.shift()) | (ff.Lane != ff.Lane.shift())
    ff['rid'] = ff.new.cumsum()
    g = ff.groupby('rid')
    R = pd.DataFrame({'GlobalID': g.GlobalID.first(), 'lane': g.Lane.first(), 't_start': g.Timestamp.first(), 't_end': g.Timestamp.last(),
                      'x_start': g.PositionX.first(), 'x_end': g.PositionX.last(), 'n_frames': g.size(), 'breaks': g.segment_break_before.sum() - g.segment_break_before.first()})
    R['k'] = R.groupby('GlobalID').cumcount(); R['nk'] = R.groupby('GlobalID').k.transform('max')
    R['start_type'] = np.where(R.k == 0, 'entry', 'lc'); R['end_type'] = np.where(R.k == R.nk, 'exit', 'lc')
    R['dur_s'] = (R.t_end - R.t_start) / 1000.0 + 0.1; R['len_m'] = R.x_end - R.x_start
    R = R.drop(columns=['k', 'nk']).reset_index(drop=True); R.insert(0, 'day', n)
    return R


# ----------------------------------------------------------------------------------------------------------- passing events
def pass_table(n, f):
    """Order flips between co-present vehicles. A pass of j by i: dx = x_j - x_i goes from > +HYST to < -HYST while both are tracked
    on consecutive common timestamps (gap in co-presence <= 1 s allowed)."""
    a = f[['GlobalID', 'Timestamp', 'PositionX', 'Lane', 'Smoothed_VelocityX', 'truck']]
    m = a.merge(a, on='Timestamp', suffixes=('', '_o'))
    m = m[m.GlobalID != m.GlobalID_o]
    m = m[(m.PositionX_o - m.PositionX).abs() <= 200]
    m['dx'] = m.PositionX_o - m.PositionX
    m = m.sort_values(['GlobalID', 'GlobalID_o', 'Timestamp'])
    rows = []
    for (i, j), d in m.groupby(['GlobalID', 'GlobalID_o'], sort=False):
        dx = d.dx.values
        if dx.max() <= HYST or dx.min() >= -HYST:
            continue
        ts = d.Timestamp.values
        state = 0   # +1 other ahead, -1 other behind
        k_last_ahead = None
        for k in range(len(dx)):
            if k > 0 and ts[k] - ts[k - 1] > 1000:
                state = 0
            if dx[k] > HYST:
                state = 1; k_last_ahead = k
            elif dx[k] < -HYST:
                if state == 1:
                    # crossing point: first frame after last-ahead where dx < 0
                    kk = k_last_ahead + int(np.argmax(dx[k_last_ahead:k + 1] < 0))
                    r = d.iloc[kk]
                    rows.append(dict(passer=i, passed=j, t=int(r.Timestamp), x=float(r.PositionX), lane_passer=int(r.Lane), lane_passed=int(r.Lane_o),
                                     truck_passer=int(r.truck), truck_passed=int(r.truck_o), v_passer=float(r.Smoothed_VelocityX) * 3.6, v_passed=float(r.Smoothed_VelocityX_o) * 3.6))
                state = -1
    P = pd.DataFrame(rows)
    if len(P):
        P.insert(0, 'day', n)
    return P


# ----------------------------------------------------------------------------------------------------------- event context
def snapshot_ctx(snap, gid, lane_from, x, v):
    """context of ego (gid) at one timestamp: nearest vehicles ahead/behind in the from-lane and in the other lane."""
    o = snap[snap.GlobalID != gid]
    out = {}
    for tag, ln in (('own', lane_from), ('tgt', 3 - lane_from)):
        s = o[o.Lane == ln]
        ah = s[s.PositionX > x]; bh = s[s.PositionX <= x]
        if len(ah):
            r = ah.loc[ah.PositionX.idxmin()]
            out[f'{tag}_lead_id'] = r.GlobalID; out[f'{tag}_lead_gap'] = r.PositionX - x; out[f'{tag}_lead_dv'] = v - r.Smoothed_VelocityX * 3.6; out[f'{tag}_lead_truck'] = int(r.truck)
        else:
            out[f'{tag}_lead_id'] = np.nan; out[f'{tag}_lead_gap'] = np.nan; out[f'{tag}_lead_dv'] = np.nan; out[f'{tag}_lead_truck'] = np.nan
        if len(bh):
            r = bh.loc[bh.PositionX.idxmax()]
            out[f'{tag}_lag_id'] = r.GlobalID; out[f'{tag}_lag_gap'] = x - r.PositionX; out[f'{tag}_lag_dv'] = r.Smoothed_VelocityX * 3.6 - v; out[f'{tag}_lag_truck'] = int(r.truck)
        else:
            out[f'{tag}_lag_id'] = np.nan; out[f'{tag}_lag_gap'] = np.nan; out[f'{tag}_lag_dv'] = np.nan; out[f'{tag}_lag_truck'] = np.nan
    return out


def event_table(n, f, e, P, R):
    vi, vo = [(a, b) for d, a, b in DAYS if d == n][0]
    e = e[e.from_lane.isin([1, 2]) & e.to_lane.isin([1, 2])].sort_values(['GlobalID', 'timestamp_ms']).copy()
    byT = {t: s for t, s in f[['GlobalID', 'Timestamp', 'PositionX', 'Lane', 'Smoothed_VelocityX', 'truck']].groupby('Timestamp')}
    byV = {g: d for g, d in f[f.GlobalID.isin(e.GlobalID.unique()) | True][['GlobalID', 'Timestamp', 'PositionX', 'Lane', 'Smoothed_VelocityX', 'frame_source', 'Headway', 'gap_m', 'leader_id']].groupby('GlobalID')}
    rows = []
    for r in e.itertuples():
        gid = r.GlobalID; t = int(r.timestamp_ms); ego = byV[gid]
        d = dict(day=n, cfg=f'{vi}/{vo}', delta=vi - vo, vin=vi, vout=vo, GlobalID=gid, t=t, x=r.position_x_m, v=r.speed_ms * 3.6,
                 direction='in' if r.to_lane == 1 else 'out', from_lane=int(r.from_lane), to_lane=int(r.to_lane),
                 cls=r.vehicle_class, truck=int(ego.truck.iloc[0]) if 'truck' in ego else np.nan, transition_s=r.transition_s,
                 resolved=int(pd.notna(r.transition_s)), unmeas3=r.unmeasured_frames_within_3s)
        d['truck'] = 1 if r.vehicle_class == 'heavy' else (0 if r.vehicle_class == 'car' else -1)
        d['ego_t0'] = int(ego.Timestamp.iloc[0]); d['ego_t1'] = int(ego.Timestamp.iloc[-1]); d['ego_x0'] = float(ego.PositionX.iloc[0]); d['ego_x1'] = float(ego.PositionX.iloc[-1])
        for lag in (5, 3, 1):
            tau = t - lag * 1000
            er = ego[ego.Timestamp == tau]
            if len(er) == 0 or tau not in byT:
                d[f'ok{lag}'] = 0; continue
            er = er.iloc[0]
            d[f'ok{lag}'] = 1; d[f'lane{lag}'] = int(er.Lane); d[f'x{lag}'] = float(er.PositionX); d[f'v{lag}'] = float(er.Smoothed_VelocityX) * 3.6
            d[f'meas{lag}'] = int(er.frame_source <= 1)
            c = snapshot_ctx(byT[tau], gid, int(r.from_lane), float(er.PositionX), float(er.Smoothed_VelocityX) * 3.6)
            for k, v in c.items():
                d[f'{k}{lag}'] = v
            d[f'own_lead_hw{lag}'] = d[f'own_lead_gap{lag}'] / max(float(er.Smoothed_VelocityX), 0.1) if pd.notna(d[f'own_lead_gap{lag}']) else np.nan
            d[f'look_ok{lag}'] = int(float(er.PositionX) + LOOK <= X1)    # a leader within LOOK m would be visible
        rows.append(d)
    E = pd.DataFrame(rows)
    # ---- previous / next analysis-grade change of the same vehicle
    E = E.sort_values(['GlobalID', 't']).reset_index(drop=True)
    g = E.groupby('GlobalID')
    E['k_lc'] = g.cumcount(); E['n_lc_vehicle'] = g.t.transform('size')
    E['prev_dir'] = g.direction.shift(); E['prev_dt'] = (E.t - g.t.shift()) / 1000.0
    E['next_dir'] = g.direction.shift(-1); E['next_dt'] = (g.t.shift(-1) - E.t) / 1000.0; E['next_x'] = g.x.shift(-1)
    # time/distance in the new lane: until the next change or the end of the track (censored)
    E['stay_s'] = np.where(E.next_dt.notna(), E.next_dt, (E.ego_t1 - E.t) / 1000.0)
    E['stay_m'] = np.where(E.next_x.notna(), E.next_x - E.x, E.ego_x1 - E.x)
    E['stay_censored'] = E.next_dt.isna().astype(int)
    # ---- fate of the original own-lane leader at t-3 (inward: did ego pass it?), passes made after the change
    fate, tpass, npass_after, npass_before = [], [], [], []
    Pi = P.groupby('passer') if len(P) else None
    for r in E.itertuples():
        L = getattr(r, 'own_lead_id3', np.nan)
        pp = Pi.get_group(r.GlobalID) if (Pi is not None and r.GlobalID in Pi.groups) else pd.DataFrame(columns=['passed', 't', 'lane_passer', 'lane_passed'])
        t_end = r.t + r.stay_s * 1000
        aft = pp[(pp.t >= r.t - 3000) & (pp.t <= t_end)]
        npass_after.append(int(((aft.lane_passer != aft.lane_passed)).sum()))
        bef = pp[(pp.t >= r.t - 30000) & (pp.t < r.t)]
        npass_before.append(int(((bef.lane_passer != bef.lane_passed)).sum()))
        if pd.isna(L):
            fate.append('no_leader'); tpass.append(np.nan); continue
        hit = pp[(pp.passed == L) & (pp.t >= r.t - 3000)]
        if len(hit):
            fate.append('passed'); tpass.append((hit.t.min() - r.t) / 1000.0); continue
        Ld = byV.get(L)
        if Ld is None:
            fate.append('unknown'); tpass.append(np.nan); continue
        # who left first?
        fate.append('leader_left_first' if Ld.Timestamp.iloc[-1] < r.ego_t1 else 'ego_left_first_or_not_passed'); tpass.append(np.nan)
    E['lead3_fate'] = fate; E['t_pass_lead3'] = tpass; E['n_pass_after'] = npass_after; E['n_pass_before30'] = npass_before
    return E


def classify(E):
    """First-pass manoeuvre labels (to be refined in S1)."""
    lab = []
    for r in E.itertuples():
        hw = getattr(r, 'own_lead_hw3', np.nan); gap = getattr(r, 'own_lead_gap3', np.nan); dv = getattr(r, 'own_lead_dv3', np.nan)
        look = getattr(r, 'look_ok3', 0)
        if r.direction == 'in':
            constrained = pd.notna(hw) and (hw < 4 or (gap < 150 and dv > 3))
            if constrained and r.lead3_fate == 'passed':
                lab.append('in_overtake_completed')
            elif constrained:
                lab.append('in_overtake_unobserved')
            elif pd.isna(gap) and look == 1:
                lab.append('in_free_noleader')
            elif pd.notna(gap) and gap >= LOOK:
                lab.append('in_free_noleader')
            elif pd.isna(gap):
                lab.append('in_free_unknown_view')
            else:
                lab.append('in_free_distant_leader')
        else:
            behind_fast = pd.notna(getattr(r, 'own_lag_gap3', np.nan)) and r.own_lag_gap3 < 100 and r.own_lag_dv3 > 5
            blocked = pd.notna(hw) and hw < 3 and dv > 0
            if r.n_pass_before30 > 0 or (r.prev_dir == 'in' and r.prev_dt <= 60):
                lab.append('out_return_after_pass')
            elif behind_fast:
                lab.append('out_yield_to_follower')
            elif blocked:
                lab.append('out_blocked_in_inner')
            else:
                lab.append('out_free_keepright')
    E['mtype0'] = lab
    return E


if __name__ == '__main__':
    keys = sys.argv[1:] or [n for n, _, _ in DAYS]
    for n, vi, vo in DAYS:
        if n not in keys:
            continue
        f = frames(n); f = f[f.Lane.isin([1, 2, 3])].copy()
        V = vehicle_table(n, f); R = run_table(n, f); P = pass_table(n, f)
        E = classify(event_table(n, f, lc_events(n), P, R))
        # runs: passes made/received during the run
        if len(P):
            PP = P[P.lane_passer != P.lane_passed]
            R['n_pass_made'] = [int(((PP.passer == r.GlobalID) & (PP.t >= r.t_start) & (PP.t <= r.t_end)).sum()) for r in R.itertuples()]
            R['n_passed_by'] = [int(((PP.passed == r.GlobalID) & (PP.t >= r.t_start) & (PP.t <= r.t_end)).sum()) for r in R.itertuples()]
        for nm, T in (('core_vehicles', V), ('core_runs', R), ('core_passes', P), ('core_events', E)):
            T.to_pickle(os.path.join(CACHE, f'{nm}_{n}.pkl'))
        print(n, 'veh', len(V), 'runs', len(R), 'passes', len(P), 'diff-lane passes', int((P.lane_passer != P.lane_passed).sum()) if len(P) else 0,
              'events', len(E), E.mtype0.value_counts().to_dict(), flush=True)
    for nm in ('core_vehicles', 'core_runs', 'core_passes', 'core_events'):
        parts = [pd.read_pickle(os.path.join(CACHE, f'{nm}_{n}.pkl')) for n, _, _ in DAYS if os.path.exists(os.path.join(CACHE, f'{nm}_{n}.pkl'))]
        T = pd.concat(parts, ignore_index=True)
        T['delta'] = T.day.map(delta); T['cfg'] = T.day.map(label); T['diff_day'] = (T.delta > 0).astype(int)
        T.to_pickle(os.path.join(CACHE, f'{nm}.pkl')); T.to_csv(os.path.join(OUT, f'{nm}.csv.gz'), index=False)
        print('saved', nm, T.shape)
