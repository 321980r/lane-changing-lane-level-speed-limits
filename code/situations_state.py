# -*- coding: utf-8 -*-
"""Per-frame driving state of every car and heavy vehicle in lane 1 or 2 (10 Hz session frames): own-lane leader (gap, closing speed,
class), own-lane follower (gap, approach speed), nearest vehicle ahead in the other lane, time since the vehicle's last pass of a vehicle in
the other lane, time since the current lane run began with a lane change, local hour, distance driven in the frame (v x 0.1 s).
Lane changes are attributed to the state 3 s (or 5 s) before them, so that counts and exposure use one definition.
Inputs: session frames; _cache/work/core_passes.pkl, core_runs.pkl. Output: _cache/work/sit_state_<day>.pkl."""
import os, sys, time
from lc_common import *

P_ALL = pd.read_pickle(os.path.join(CACHE, 'core_passes.pkl'))
R_ALL = pd.read_pickle(os.path.join(CACHE, 'core_runs.pkl'))


def build(n):
    f = frames(n)
    f = f[f.Lane.isin([1, 2, 3])][['GlobalID', 'Timestamp', 'PositionX', 'Lane', 'Smoothed_VelocityX', 'truck', 'frame_source']].copy()
    f['v'] = f.Smoothed_VelocityX.astype('float32') * 3.6
    # ---- own-lane leader / follower by sorting within (Timestamp, Lane)
    f = f.sort_values(['Timestamp', 'Lane', 'PositionX']).reset_index(drop=True)
    same_next = (f.Timestamp.shift(-1) == f.Timestamp) & (f.Lane.shift(-1) == f.Lane)
    same_prev = (f.Timestamp.shift(1) == f.Timestamp) & (f.Lane.shift(1) == f.Lane)
    f['lead_gap'] = np.where(same_next, f.PositionX.shift(-1) - f.PositionX, np.nan).astype('float32')
    f['lead_dv'] = np.where(same_next, f.v - f.v.shift(-1), np.nan).astype('float32')          # >0 closing
    f['lead_truck'] = np.where(same_next, f.truck.shift(-1), np.nan).astype('float32')
    f['lag_gap'] = np.where(same_prev, f.PositionX - f.PositionX.shift(1), np.nan).astype('float32')
    f['lag_dv'] = np.where(same_prev, f.v.shift(1) - f.v, np.nan).astype('float32')             # >0 follower approaching
    # ---- nearest vehicle ahead in the other lane (1<->2)
    parts = []
    for ln in (1, 2):
        L = f[f.Lane == ln][['Timestamp', 'PositionX']].reset_index().sort_values('PositionX')
        O = f[f.Lane == 3 - ln][['Timestamp', 'PositionX', 'v', 'truck']].sort_values('PositionX').rename(columns={'PositionX': 'xo', 'v': 'vo', 'truck': 'to'})
        O['PositionX'] = O.xo
        m = pd.merge_asof(L, O, on='PositionX', by='Timestamp', direction='forward', allow_exact_matches=False)
        m['oth_lead_gap'] = m.xo - m.PositionX; m['oth_lead_dv'] = np.nan
        vv = f.loc[m['index'], 'v'].values
        m['oth_lead_dv'] = vv - m.vo; m['oth_lead_truck'] = m['to']
        parts.append(m.set_index('index')[['oth_lead_gap', 'oth_lead_dv', 'oth_lead_truck']])
    oth = pd.concat(parts)
    f = f.join(oth)
    f = f[f.Lane.isin([1, 2]) & (f.truck >= 0)].copy()
    f = f.sort_values(['GlobalID', 'Timestamp']).reset_index(drop=True)
    # ---- time since last cross-lane pass made by this vehicle
    P = P_ALL[(P_ALL.day == n) & (P_ALL.lane_passer != P_ALL.lane_passed)][['passer', 't']].rename(columns={'passer': 'GlobalID', 't': 't_pass'})
    P = P.sort_values('t_pass')
    left = f[['GlobalID', 'Timestamp']].reset_index().sort_values('Timestamp')
    m = pd.merge_asof(left, P, left_on='Timestamp', right_on='t_pass', by='GlobalID', direction='backward')
    f.loc[m['index'].values, 't_since_pass'] = ((m.Timestamp - m.t_pass) / 1000.0).values
    # ---- time since current lane run started with a lane change (NaN if the run started at track entry)
    R = R_ALL[R_ALL.day == n][['GlobalID', 't_start', 'start_type', 'lane']].sort_values('t_start')
    m = pd.merge_asof(left, R, left_on='Timestamp', right_on='t_start', by='GlobalID', direction='backward')
    ts = (m.Timestamp - m.t_start) / 1000.0
    f.loc[m['index'].values, 't_since_lc'] = np.where(m.start_type.values == 'lc', ts.values, np.nan)
    f['hour'] = C.hours(f.Timestamp).astype('float32')
    f['km'] = (f.Smoothed_VelocityX.astype('float64') * 0.1 / 1000.0).astype('float32')
    f['meas'] = (f.frame_source <= 1).astype('int8')
    keep = ['GlobalID', 'Timestamp', 'PositionX', 'Lane', 'truck', 'v', 'km', 'meas', 'hour', 'lead_gap', 'lead_dv', 'lead_truck', 'lag_gap', 'lag_dv',
            'oth_lead_gap', 'oth_lead_dv', 'oth_lead_truck', 't_since_pass', 't_since_lc']
    f = f[keep]
    for c in ('PositionX', 'v', 'lead_gap', 'lead_dv', 'lag_gap', 'lag_dv', 'oth_lead_gap', 'oth_lead_dv', 't_since_pass', 't_since_lc'):
        f[c] = f[c].astype('float32')
    f.insert(0, 'day', n)
    return f


if __name__ == '__main__':
    for n, vi, vo in DAYS:
        p = os.path.join(CACHE, f'sit_state_{n}.pkl')
        t0 = time.time(); S = build(n); S.to_pickle(p)
        print(n, len(S), f'{time.time() - t0:.1f}s', 'km', round(S.km.sum(), 1), flush=True)
