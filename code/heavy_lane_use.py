# -*- coding: utf-8 -*-
"""Share of the kilometres driven in each lane by heavy vehicles and cars (Sections 3.3 and 5.3), per day and pooled over the eight days,
for all session frames and for the analysis section 900 <= x < 2,100 m, also with measured frames only; and the share of vehicles that
stayed in one lane in the section. Distance = Smoothed_VelocityX x 0.1 s summed over the 10 Hz frames (as the car-km of Table 1).
Input: session frames. Outputs: results/final/heavy_lane_use_days.csv, heavy_lane_use_summary.csv."""
import os
import numpy as np, pandas as pd
from plot_style import *
from situations_lib import CFG, DEL

rows = []
for n in ORDER8:
    f = frames(n)
    f = f[(f.in_monitored_session == 1) & (f.in_disturbed_period == 0)]
    d = f.Smoothed_VelocityX.astype(float) * 0.1 / 1000.0
    for win, m in (('all session frames', np.ones(len(f), bool)), ('900-2100 m', ((f.PositionX >= 900) & (f.PositionX < 2100)).values)):
        for src, ms in (('all frames', np.ones(len(f), bool)), ('measured frames (frame_source <= 1)', (f.frame_source <= 1).values)):
            for cls, cm in (('heavy', (f.truck == 1).values), ('car', (f.truck == 0).values)):
                sel = m & ms & cm
                km = d[sel].groupby(f.Lane.values[sel]).sum()
                tot = km.sum()
                rows.append(dict(day=n, cfg=CFG[n], group='same-limit' if DEL[n] == 0 else 'differentiated',
                                 window=win, frames=src, cls=cls, km_total=tot, km_lane1=km.get(1, 0.0), km_lane2=km.get(2, 0.0),
                                 km_lane3=km.get(3, 0.0), share_km_lane1=km.get(1, 0.0) / tot, share_km_lane2=km.get(2, 0.0) / tot,
                                 share_km_lane3=km.get(3, 0.0) / tot))
    # vehicles that stayed in one lane over the analysis window (lanes 1/2 only; shoulder frames ignored)
    w = f[(f.PositionX >= 900) & (f.PositionX < 2100) & f.Lane.isin([1, 2])]
    g = w.groupby('GlobalID').agg(truck=('truck', 'max'), has1=('Lane', lambda v: (v == 1).any()), has2=('Lane', lambda v: (v == 2).any()))
    for cls, t in (('heavy', 1), ('car', 0)):
        z = g[g.truck == t]
        rows.append(dict(day=n, cfg=CFG[n], group='same-limit' if DEL[n] == 0 else 'differentiated',
                         window='900-2100 m', frames='vehicles', cls=cls, n_vehicles=len(z),
                         share_only_inner=float((z.has1 & ~z.has2).mean()), share_only_outer=float((z.has2 & ~z.has1).mean()),
                         share_both=float((z.has1 & z.has2).mean())))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(DATF, 'heavy_lane_use_days.csv'), index=False)

srows = []
for win in ('all session frames', '900-2100 m'):
    for src in ('all frames', 'measured frames (frame_source <= 1)'):
        for cls in ('heavy', 'car'):
            z = R[(R.window == win) & (R.frames == src) & (R.cls == cls)].set_index('day')
            z8 = z.loc[ORDER8]
            srows.append(dict(window=win, frames=src, cls=cls,
                              pooled8_share_lane1=z8.km_lane1.sum() / z8.km_total.sum(), pooled8_share_lane2=z8.km_lane2.sum() / z8.km_total.sum(),
                              pooled8_share_lane3=z8.km_lane3.sum() / z8.km_total.sum(), day_min_lane1=z8.share_km_lane1.min(),
                              day_max_lane1=z8.share_km_lane1.max(), day_min_lane2=z8.share_km_lane2.min(), day_max_lane2=z8.share_km_lane2.max(),
                              same_mean_lane1=z8.share_km_lane1.iloc[:3].mean(), diff_mean_lane1=z8.share_km_lane1.iloc[3:].mean(),
                              p2_8days_lane1=perm8(z8.share_km_lane1.values)[1],
                              km_total_8=z8.km_total.sum(), day_values_lane1=';'.join(f'{CFG[x]}={z.share_km_lane1[x]:.4f}' for x in ORDER8)))
for cls in ('heavy', 'car'):
    z = R[(R.window == '900-2100 m') & (R.frames == 'vehicles') & (R.cls == cls)].set_index('day').loc[ORDER8]
    w_ = z.n_vehicles / z.n_vehicles.sum()
    srows.append(dict(window='900-2100 m', frames='vehicles', cls=cls, n_vehicles_8=int(z.n_vehicles.sum()),
                      pooled8_share_only_inner=float((z.share_only_inner * w_).sum()), pooled8_share_only_outer=float((z.share_only_outer * w_).sum()),
                      day_min_only_outer=z.share_only_outer.min(), day_max_only_outer=z.share_only_outer.max(),
                      day_min_only_inner=z.share_only_inner.min(), day_max_only_inner=z.share_only_inner.max()))
S_ = pd.DataFrame(srows)
S_.to_csv(os.path.join(DATF, 'heavy_lane_use_summary.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(R[(R.cls == 'heavy') & (R.frames != 'vehicles')][['cfg', 'group', 'window', 'frames', 'km_total', 'share_km_lane1', 'share_km_lane2', 'share_km_lane3']].round(4).to_string(index=False))
print(S_.round(4).to_string(index=False))
