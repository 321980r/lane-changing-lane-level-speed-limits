# -*- coding: utf-8 -*-
"""Change in the inner-lane share of full-traverse cars per day (modal lane at 1,000-1,100 m and at 1,900-2,000 m; also 1,100-1,200 m to
1,750-1,850 m), with exact day-level tests.
Input: session frames. Output: results/check_sit_lane_shift.csv."""
import os, numpy as np, pandas as pd
from lc_common import *
import check_rates as VR
rows = []
for n, vi, vo in DAYS:
    f = frames(n); f = f[(f.in_monitored_session == 1) & (f.in_disturbed_period == 0) & (f.truck == 0) & f.Lane.isin([1, 2])]
    g = f.groupby('GlobalID'); x0 = g.PositionX.min(); x1 = g.PositionX.max()
    full = x0.index[(x0 < 1000) & (x1 > 2000)]
    ff = f[f.GlobalID.isin(full)]
    ent = ff[(ff.PositionX >= 1000) & (ff.PositionX < 1100)].groupby('GlobalID').Lane.agg(lambda s: s.mode().iloc[0])
    ext = ff[(ff.PositionX >= 1900) & (ff.PositionX < 2000)].groupby('GlobalID').Lane.agg(lambda s: s.mode().iloc[0])
    j = pd.concat([ent.rename('a'), ext.rename('b')], axis=1).dropna()
    # alternative entry bin 1050-1150 and exit 1850-1950 (edge robustness)
    ent2 = ff[(ff.PositionX >= 1100) & (ff.PositionX < 1200)].groupby('GlobalID').Lane.agg(lambda s: s.mode().iloc[0])
    ext2 = ff[(ff.PositionX >= 1750) & (ff.PositionX < 1850)].groupby('GlobalID').Lane.agg(lambda s: s.mode().iloc[0])
    j2 = pd.concat([ent2.rename('a'), ext2.rename('b')], axis=1).dropna()
    rows.append(dict(day=n, cfg=label(n), n_cars=len(j), in_entry=(j.a == 1).mean(), in_exit=(j.b == 1).mean(), gain_pp=100 * ((j.b == 1).mean() - (j.a == 1).mean()),
                     n_21=int(((j.a == 2) & (j.b == 1)).sum()), n_12=int(((j.a == 1) & (j.b == 2)).sum()),
                     gain_pp_alt=100 * ((j2.b == 1).mean() - (j2.a == 1).mean()), n_alt=len(j2)))
L = pd.DataFrame(rows); L.to_csv(os.path.join(RES, 'check_sit_lane_shift.csv'), index=False)
print(L.round(3).to_string(index=False))
names = VR.names; D8 = VR.D8
for col in ('gain_pp', 'gain_pp_alt', 'in_entry'):
    v = L.set_index('day')[col]
    t8 = VR.exact_same_diff(v.loc[D8].values)
    print(col, 'same', round(v.loc[D8[:3]].mean(), 3), 'diff', round(v.loc[D8[3:]].mean(), 3), 'p2', round(t8['p2'], 3), 'p1', round(t8['p1'], 3), 'p2rank', round(t8['p2_rank'], 3))
