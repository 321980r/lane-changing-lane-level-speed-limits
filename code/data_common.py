# -*- coding: utf-8 -*-
"""Data paths and loaders: the processed data set (data/trajectories), the eight days (DAYS: release name,
inner-lane limit, outer-lane limit; the three same-limit days first), frames() (session frames of a day, cached in _cache/),
lc_events() and hb_events() (lane-change and hard-braking events of the session) and hours() (local time of day)."""
import os, numpy as np, pandas as pd
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # repository root (this file is in code/)
DATA_ROOT = os.path.join(PROJ, 'data')
CACHE = os.path.join(PROJ, '_cache')                                     # intermediate files, created when the scripts run
REL = os.path.join(DATA_ROOT, 'trajectories')    # trajectory files of the eight days (not part of the repository)
os.makedirs(CACHE, exist_ok=True)
# (release name, inner, outer)  -- order: same-limit first, then differentiated
DAYS = [('scen_2022-08-08_60-60', 60, 60), ('scen_2022-07-29_80-80', 80, 80), ('scen_2022-08-11_100-100', 100, 100), ('scen_2022-08-02_80-70', 80, 70),
        ('scen_2022-08-03_90-70', 90, 70), ('scen_2022-07-28_100-80', 100, 80), ('scen_2022-08-01_80-60', 80, 60), ('scen_2022-08-05_100-70', 100, 70)]
SHORT = {n: n.replace('scen_2022-', '').replace('-', '', 1) for n, _, _ in DAYS}     # e.g. 'scen_2022-08-08_60-60' -> '0808_60-60'
FRAME_COLS = ['GlobalID', 'Timestamp', 'PositionX', 'Lane', 'lateral_position_m', 'ObjectClass_raw', 'Smoothed_VelocityX', 'caV', 'leader_id', 'gap_m', 'Headway', 'TTC',
              'in_monitored_session', 'in_disturbed_period', 'frame_source', 'segment_break_before']
def frames(n, session_only=True):
    """frames of one day (cached); session_only keeps the posted, undisturbed session"""
    p = os.path.join(CACHE, n + ('_s' if session_only else '') + '.pkl')
    if os.path.exists(p):
        return pd.read_pickle(p)
    f = pd.read_csv(os.path.join(REL, 'data', n + '.csv.gz'), usecols=FRAME_COLS)
    if session_only:
        f = f[(f.in_monitored_session == 1) & (f.in_disturbed_period == 0)].copy()
    for c in ('PositionX', 'lateral_position_m', 'Smoothed_VelocityX', 'caV', 'gap_m', 'Headway', 'TTC'):
        f[c] = f[c].astype('float32')
    for c in ('Lane', 'ObjectClass_raw', 'in_monitored_session', 'in_disturbed_period', 'frame_source', 'segment_break_before'):
        f[c] = f[c].astype('int8')
    f['truck'] = np.where(f.ObjectClass_raw < 0, -1, (f.ObjectClass_raw == 0).astype(int)).astype('int8')
    f = f.sort_values(['GlobalID', 'Timestamp']).reset_index(drop=True); f.to_pickle(p); return f
def lc_events(n, grade=True):
    e = pd.read_csv(os.path.join(REL, 'events', 'lc_events_' + n + '.csv'))
    e = e[(e.in_session == 1) & (e.in_disturbed_period == 0)]
    return e[e.analysis_grade == 1].copy() if grade else e.copy()
def hb_events(n):
    e = pd.read_csv(os.path.join(REL, 'events', 'hb_events_' + n + '.csv'))
    return e[(e.in_session == 1) & (e.in_disturbed_period == 0)].copy() if len(e) else e
def hours(ts):
    return ((np.asarray(ts) // 1000 + 8 * 3600) % 86400) / 3600.0
