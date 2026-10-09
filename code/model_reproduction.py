# -*- coding: utf-8 -*-
"""Per-day counts from the frames and the lane-change events: vehicles, heavy-vehicle share, vehicle-km, analysis-grade lane changes
by class, lane changes per vehicle-km and inward shares.
Inputs: session frames and events. Output: results/model_reproduction.csv (read by table1_data.py)."""
import numpy as np, pandas as pd
from lc_common import *

rows = []
for n, vi, vo in DAYS:
    f = frames(n); e = lc_events(n)
    e = e[e.from_lane.isin([1, 2]) & e.to_lane.isin([1, 2])]
    veh = f.GlobalID.nunique()
    cls = f.groupby('GlobalID').truck.max()
    trk_share = (cls == 1).mean()
    # vehicle-km inside the window: integrate speed over 0.1 s frames
    vkm = (f.Smoothed_VelocityX.astype(float) * 0.1).sum() / 1000.0
    car = e[e.vehicle_class == 'car']; hv = e[e.vehicle_class == 'heavy']
    inw = lambda x: (x.to_lane == 1).mean() if len(x) else np.nan
    rows.append(dict(day=n, cfg=f'{vi}/{vo}', delta=vi - vo, vehicles=veh, truck_share=round(trk_share, 3), vkm=round(vkm, 1),
                     lc=len(e), lc_car=len(car), lc_heavy=len(hv), lc_unknown=len(e) - len(car) - len(hv),
                     rate_per_vkm=round(len(e) / vkm, 3),
                     inward_all=round(inw(e), 2), inward_car=round(inw(car), 2), inward_heavy=round(inw(hv), 2)))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(RES, 'model_reproduction.csv'), index=False)
pd.set_option('display.width', 250); print(T.to_string(index=False))
print('8 days: vehicles', T.vehicles.sum(), 'lc', T.lc.sum(), 'car', T.lc_car.sum(), 'heavy', T.lc_heavy.sum())
