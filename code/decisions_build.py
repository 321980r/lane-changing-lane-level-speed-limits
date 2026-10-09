# -*- coding: utf-8 -*-
"""1 Hz decision points of all vehicles in lane 1 or 2 with the MOBIL neighbourhood: own-lane leader and follower and target-lane leader
and follower (position, speed, class); running desired-speed proxies (mean free-driving speed up to t, and the same divided by the lane
limit); whether the current lane was entered by a lane change. Outcome: y5 / y3 = an analysis-grade change to the other lane within 5 s /
3 s; drop5 = a change that is not analysis-grade within 5 s (such points are left out of the estimation).
Inputs: session frames and events; _cache/work/core_vehicles.pkl. Output: _cache/work/decision_points.pkl."""
import os, sys, numpy as np, pandas as pd
from lc_common import *

LEN = {1: 12.0, 0: 5.0, -1: 5.0}      # vehicle length (m) by class: heavy / car / unknown


def all_events(n):
    return lc_events(n, grade=False)


def build(n, vi, vo):
    f = frames(n)
    f = f[f.Lane.isin([1, 2])][['GlobalID', 'Timestamp', 'PositionX', 'Lane', 'Smoothed_VelocityX', 'truck', 'leader_id', 'Headway', 'frame_source']].copy()
    f = f.sort_values(['GlobalID', 'Timestamp']).reset_index(drop=True)
    v = f.Smoothed_VelocityX.astype(float) * 3.6
    Vlane = np.where(f.Lane == 1, vi, vo).astype(float)
    free = (f.leader_id.isna() | (f.Headway > 5)).astype(float).values
    g = f.GlobalID.values
    # running (expanding, inclusive) free-flow means per vehicle
    df = pd.DataFrame({'g': g, 'fv': free * v.values, 'fn': free, 'fs': free * v.values / Vlane})
    cs = df.groupby('g')[['fv', 'fn', 'fs']].cumsum()
    f['vdes_pre'] = np.where(cs.fn >= 30, cs.fv / cs.fn.clip(lower=1), np.nan)
    f['sf_pre'] = np.where(cs.fn >= 30, cs.fs / cs.fn.clip(lower=1), np.nan)
    f['nfree_pre'] = cs.fn.values
    # lane-2-only running free speed (for inward decisions this is the speed driven under the outer-lane limit)
    fr2 = free * (f.Lane.values == 2)
    df2 = pd.DataFrame({'g': g, 'fv': fr2 * v.values, 'fn': fr2}); cs2 = df2.groupby('g')[['fv', 'fn']].cumsum()
    f['vdes_pre_L2'] = np.where(cs2.fn >= 30, cs2.fv / cs2.fn.clip(lower=1), np.nan)
    f['sv2'] = cs2.fv.values; f['n2'] = cs2.fn.values
    f['sv1'] = cs.fv.values - cs2.fv.values; f['n1'] = cs.fn.values - cs2.fn.values      # lane-1 running free-flow sums (for permuted limits)
    f['v'] = v.values
    # whole-track free speed (endogenous, for sensitivity) from core_vehicles
    V = pd.read_pickle(os.path.join(CACHE, 'core_vehicles.pkl')); V = V[V.day == n].set_index('GlobalID')
    f['vdes_all'] = f.GlobalID.map(V.v_des)
    # since last lane change (any), from lane transitions
    chg = (f.GlobalID == f.GlobalID.shift()) & (f.Lane != f.Lane.shift())
    f['t_lastchg'] = np.where(chg, f.Timestamp, np.nan); f['t_lastchg'] = f.groupby('GlobalID').t_lastchg.ffill()
    f['entered_by_lc'] = f.t_lastchg.notna().astype(int)

    # -------- neighbours at 1 Hz (all vehicles present at that timestamp, lanes 1/2)
    s = f[f.Timestamp % 1000 == 0][['GlobalID', 'Timestamp', 'PositionX', 'Lane', 'v', 'truck']].copy()
    s['PositionX'] = s.PositionX.astype(float)
    s = s.sort_values(['Timestamp', 'Lane', 'PositionX']).reset_index(drop=True)
    grp = (s.Timestamp != s.Timestamp.shift()) | (s.Lane != s.Lane.shift())
    gid = grp.cumsum()
    for tag, sh in (('ol', -1), ('of', 1)):          # own leader (next ahead), own follower (next behind)
        same = gid.shift(sh) == gid
        for c in ('GlobalID', 'PositionX', 'v', 'truck'):
            s[f'{tag}_{c}'] = s[c].shift(sh).where(same)
    # target lane leader / follower via merge_asof on x within (timestamp, lane)
    nb = s[['Timestamp', 'Lane', 'PositionX', 'GlobalID', 'v', 'truck']].rename(columns={'PositionX': 'xn', 'GlobalID': 'gn', 'v': 'vn', 'truck': 'tn'})
    nb['key'] = nb.Timestamp * 10 + nb.Lane
    q = s[['Timestamp', 'Lane', 'PositionX', 'GlobalID']].copy(); q['key'] = q.Timestamp * 10 + (3 - q.Lane); q['row'] = np.arange(len(q))
    q = q.sort_values('PositionX'); nb = nb.sort_values('xn')
    for tag, dirn in (('tl', 'forward'), ('tf', 'backward')):
        m = pd.merge_asof(q, nb[['key', 'xn', 'gn', 'vn', 'tn']], left_on='PositionX', right_on='xn', by='key', direction=dirn, allow_exact_matches=(dirn == 'forward'))
        m = m.sort_values('row')
        s[f'{tag}_GlobalID'] = m.gn.values; s[f'{tag}_PositionX'] = m.xn.values; s[f'{tag}_v'] = m.vn.values; s[f'{tag}_truck'] = m.tn.values
    # attach ego attributes
    keep = ['GlobalID', 'Timestamp', 'vdes_pre', 'sf_pre', 'nfree_pre', 'vdes_pre_L2', 'sv1', 'n1', 'sv2', 'n2', 'vdes_all', 't_lastchg', 'entered_by_lc', 'frame_source']
    s = s.merge(f[keep], on=['GlobalID', 'Timestamp'], how='left')
    # follower desired speeds are not needed (free-road term cancels in the follower's own lane)
    # -------- outcomes
    ea = all_events(n); ea = ea[ea.from_lane.isin([1, 2]) & ea.to_lane.isin([1, 2])]
    ea = ea[['GlobalID', 'timestamp_ms', 'from_lane', 'to_lane', 'analysis_grade', 'transition_s']].sort_values('timestamp_ms')
    s = s.sort_values('Timestamp')
    m = pd.merge_asof(s[['GlobalID', 'Timestamp']].reset_index(), ea.rename(columns={'timestamp_ms': 't_next'}), left_on='Timestamp', right_on='t_next',
                      by='GlobalID', direction='forward', allow_exact_matches=False).set_index('index')
    s = s.join(m[['t_next', 'from_lane', 'to_lane', 'analysis_grade', 'transition_s']])
    dt = (s.t_next - s.Timestamp) / 1000.0
    ok_dir = (s.from_lane == s.Lane)
    for H in (3, 5):
        s[f'y{H}'] = ((dt <= H) & (s.analysis_grade == 1) & ok_dir).astype(int)
    s['drop5'] = ((dt <= 5) & ~((s.analysis_grade == 1) & ok_dir)).astype(int)
    s['dt_next'] = dt; s['resolved_next'] = s.transition_s.notna().astype(int)
    s['since_chg_s'] = (s.Timestamp - s.t_lastchg) / 1000.0
    s['day'] = n; s['vin'] = vi; s['vout'] = vo; s['delta'] = vi - vo; s['diff_day'] = int(vi > vo)
    s['hour'] = C.hours(s.Timestamp.values)
    s = s.sort_values(['GlobalID', 'Timestamp']).reset_index(drop=True)
    s = s.drop(columns=['from_lane', 'to_lane', 'analysis_grade', 'transition_s'])
    print(n, 'points', len(s), 'y5', int(s.y5.sum()), 'drop5', int(s.drop5.sum()), 'vdes_pre known', round(s.vdes_pre.notna().mean(), 3), flush=True)
    return s


if __name__ == '__main__':
    parts = [build(n, vi, vo) for n, vi, vo in DAYS]
    D = pd.concat(parts, ignore_index=True)
    D.to_pickle(os.path.join(CACHE, 'decision_points.pkl')); print('saved', D.shape)
