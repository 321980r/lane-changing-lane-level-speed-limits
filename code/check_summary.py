# -*- coding: utf-8 -*-
"""Heavy/car rate ratio of the return after the last pass (Poisson with day effects, vehicle-clustered errors) for all runs, runs
opened by an inward change and entry runs.
Input: results/check_stint_postpass_units.csv. Output: results/check_stint_postpass_rr_subsets.csv."""
import os, numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from lc_common import *
U = pd.read_csv(os.path.join(RES, 'check_stint_postpass_units.csv'))
U = U[~((U.start_type == 'lc') & (U.start_lat_ok != 1))].copy()
rows = []
for sub, q in (('all_runs', U), ('lc_started', U[U.start_type == 'lc']), ('entry_runs', U[U.start_type == 'entry'])):
    q = q[q.truck >= 0].copy(); q['lkm'] = np.log(q.dist / 1000); q['gid'] = q.day + '_' + q.GlobalID.astype(str); q['day'] = q.day.astype('category')
    try:
        m = smf.glm('ev_strict ~ truck + day', data=q, family=sm.families.Poisson(), offset=q.lkm).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(q.gid)[0]})
        ci = np.exp(m.conf_int().loc['truck']); rr = np.exp(m.params['truck'])
    except Exception as ex:
        rr, ci = np.nan, [np.nan, np.nan]
    rows.append(dict(subset=sub, car_n=int((q.truck == 0).sum()), car_ret=int(q[q.truck == 0].ev_strict.sum()), truck_n=int((q.truck == 1).sum()),
                     truck_ret=int(q[q.truck == 1].ev_strict.sum()), RR_truck_car=rr, lo=ci[0], hi=ci[1]))
T = pd.DataFrame(rows); T.to_csv(os.path.join(RES, 'check_stint_postpass_rr_subsets.csv'), index=False)
print(T.round(2).to_string(index=False))
