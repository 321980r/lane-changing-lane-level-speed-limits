# -*- coding: utf-8 -*-
"""Lane sequences of full-traverse cars with a lateral check of every lane change (not only the analysis-grade ones): a change that never
goes 0.8 m past the lane line, and the run it opens, are merged into the previous lane. Among cars entering in lane 2 (first run, or modal
lane at 1,000-1,100 m): shares with the sequences '21' (moved in and stayed), '212', '2' and any move in, per day, with exact tests, a
slope on Delta and a version without 29 Jul.
Inputs: session frames; _cache/work/core_runs.pkl, core_vehicles.pkl.
Outputs: results/check_stint_seq_runs_lat.csv, check_stint_seq_day.csv, check_stint_seq_tests.csv; _cache/work/check_stint_runs_lat.pkl."""
import os, itertools, numpy as np, pandas as pd
from lc_common import *
from check_returns import slope_perm, DAYORDER

LINE = 3.75
R = pd.read_pickle(os.path.join(CACHE, 'core_runs.pkl'))
V = pd.read_pickle(os.path.join(CACHE, 'core_vehicles.pkl'))
cp = os.path.join(CACHE, 'check_stint_runs_lat.pkl')
R = R.sort_values(['day', 'GlobalID', 't_start']).reset_index(drop=True)
R['prev_lane'] = R.groupby(['day', 'GlobalID']).lane.shift()
if not os.path.exists(cp):
    out = []
    for n, _, _ in DAYS:
        f = frames(n); Rn = R[(R.day == n) & (R.start_type == 'lc')]
        g = {k: d for k, d in f[f.GlobalID.isin(Rn.GlobalID.unique())].groupby('GlobalID')}
        for r in Rn.itertuples():
            d = g[r.GlobalID]; m = (d.Timestamp >= r.t_start - 5000) & (d.Timestamp <= r.t_start + 10000)
            lat = d.lateral_position_m[m].values
            depth = (LINE - lat) if r.lane == 1 else (lat - LINE)
            out.append(dict(day=n, GlobalID=r.GlobalID, t_start=r.t_start, maxdepth=float(np.nanmax(depth)) if len(depth) else np.nan))
        print(n, len(Rn), flush=True)
    pd.DataFrame(out).to_pickle(cp)
LR = pd.read_pickle(cp)
R = R.merge(LR, on=['day', 'GlobalID', 't_start'], how='left')
R['pseudo'] = ((R.start_type == 'lc') & R.lane.isin([1, 2]) & R.prev_lane.isin([1, 2]) & (R.maxdepth < 0.8)).astype(int)
R.to_csv(os.path.join(RES, 'check_stint_seq_runs_lat.csv'), index=False)


def seq_filtered(d):
    """collapse pseudo runs into the previous lane; returns (sequence, x of first 2->1 change)"""
    lanes, xs = [], []
    for r in d.itertuples():
        ln = r.lane
        if r.pseudo == 1 and lanes:
            ln = lanes[-1]
        if not lanes or ln != lanes[-1]:
            lanes.append(ln); xs.append(r.x_start)
    s = ''.join(str(int(v)) for v in lanes)
    xin = next((xs[i] for i in range(1, len(lanes)) if lanes[i - 1] == 2 and lanes[i] == 1), np.nan)
    return s, xin


rows = []
for (n, gid), d in R[R.lane.isin([1, 2])].groupby(['day', 'GlobalID']):
    s, xin = seq_filtered(d)
    raw = ''.join(str(int(v)) for v in d.lane.values[np.r_[True, d.lane.values[1:] != d.lane.values[:-1]]])
    rows.append(dict(day=n, GlobalID=gid, seq_lat=s, seq_raw=raw, x_first_in=xin))
S = pd.DataFrame(rows).merge(V[['day', 'GlobalID', 'truck', 'full_traverse', 'lane_entry', 't0']], on=['day', 'GlobalID'])
S['hour'] = C.hours(S.t0) if hasattr(C, 'hours') else np.nan
cars = S[(S.truck == 0) & (S.full_traverse == 1)].copy()

defs = {
    'E1_lat': lambda q: (q[q.seq_lat.str[0] == '2'], 'seq_lat'),
    'E1_raw': lambda q: (q[q.seq_raw.str[0] == '2'], 'seq_raw'),
    'E2_lat': lambda q: (q[q.lane_entry == 2], 'seq_lat'),
}
res_d, res_t = [], []
for dn, fn in defs.items():
    for outc in ('21', '212', '2', 'anyin', '21_x1050'):
        vals = []
        for n in DAYORDER:
            q, col = fn(cars[cars.day == n])
            if outc == 'anyin':
                y = q[col].str.contains('21')
            elif outc == '21_x1050':
                y = (q[col] == '21') & (q.x_first_in >= 1050)
            else:
                y = q[col] == outc
            if dn == 'E2_lat' and outc in ('21', '212'):
                # entry defined at 1000-1100 m: sequence must contain the move after entry; use the part from lane 2
                y = q[col].str.lstrip('1') == outc
            vals.append(y.mean()); res_d.append(dict(defn=dn, outcome=outc, day=n, cfg=label(n), delta=delta(n), n=len(q), k=int(y.sum()), share=y.mean()))
        v = np.array(vals)
        o, p2, p1, _ = perm_same_vs_diff(v[:8]); b, pb = slope_perm(v[:8], [delta(n) for n in DAYORDER[:8]])
        # leave 0729 out: 2 same vs 5 diff (21 relabellings)
        keep = [i for i in range(8) if '07-29' not in DAYORDER[i]]
        vv = v[keep]; obs = vv[2:].mean() - vv[:2].mean()
        null = [vv[[i for i in range(7) if i not in c]].mean() - vv[list(c)].mean() for c in itertools.combinations(range(7), 2)]
        p_no0729 = np.mean(np.abs(null) >= abs(obs) - 1e-12)
        res_t.append(dict(defn=dn, outcome=outc, same_mean=v[:3].mean(), diff_mean=v[3:8].mean(), p2_8=p2, p1_8=p1,
                          slope_per10=b * 10, p_slope=pb, p2_no0729=p_no0729, min_diff=v[3:8].min(), max_same=v[:3].max()))
D = pd.DataFrame(res_d); T = pd.DataFrame(res_t)
D.to_csv(os.path.join(RES, 'check_stint_seq_day.csv'), index=False); T.to_csv(os.path.join(RES, 'check_stint_seq_tests.csv'), index=False)
pd.set_option('display.width', 250)
print(T.round(3).to_string(index=False))
print(D[(D.defn == 'E1_lat') & (D.outcome == '21')][['cfg', 'n', 'k', 'share']].round(3).to_string(index=False))
print('pseudo among lc-started runs:', int(R.pseudo.sum()), 'of', int(((R.start_type == 'lc') & R.lane.isin([1, 2]) & R.prev_lane.isin([1, 2])).sum()))
