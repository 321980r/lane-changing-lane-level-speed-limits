# -*- coding: utf-8 -*-
"""Per day, the share of cars unconstrained in the outer lane when they reach x = 1,050 m that make an unconstrained inward change with
their position 3 s before the change in [1,050, 1,500) m; cars observed from <= 1,050 m to >= 1,500 m (or changing before).
Inputs: state tables (situations_lib.state), _cache/work/sit_events_typed.pkl. Output: results/sit_car_probability.csv."""
import os
from situations_lib import *
from situations_rates import exposure
S = state(); E = pd.read_pickle(os.path.join(CACHE, 'sit_events_typed.pkl'))
st = exposure(S); S = S.assign(st=st)
rows = []
for d in ORDER8:
    s = S[(S.day == d) & (S.truck == 0)]
    # state at first frame with x >= 1050
    first = s[s.PositionX >= 1050].sort_values('Timestamp').groupby('GlobalID').first()
    cand = first[(first.Lane == 2) & (first.st == 'F') & (first.PositionX < 1080)]
    xmax = s.groupby('GlobalID').PositionX.max()
    ev = E[(E.day == d) & (E.truck == 0) & (E.typ == 'IN_free') & (E.s3_PositionX >= 1050) & (E.s3_PositionX < 1500)]
    moved = cand.index.isin(ev.GlobalID)
    reach = (xmax.reindex(cand.index) >= 1500).values | moved
    rows.append(dict(day=d, cfg=CFG[d], delta=DEL[d], cars_free_lane2_at_1050=int(reach.sum()), moved_free_before_1500=int(moved[reach].sum()),
                     prob=moved[reach].mean() if reach.sum() else np.nan))
R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, 'sit_car_probability.csv'), index=False); print(R.round(3).to_string(index=False))
t = perm_tests(R.set_index('day').prob.to_dict()); print({k: round(v, 3) for k, v in t.items() if isinstance(v, float)})
