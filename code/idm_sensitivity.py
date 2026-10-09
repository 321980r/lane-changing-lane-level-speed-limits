# -*- coding: utf-8 -*-
"""Sensitivity of the car-inward M2F estimates (beta_F, a_bias) to the IDM parameters inside MOBIL (a, b, T and s0 +-30 %) and to the
anticipation time (eight days, causal desired-speed proxy; Section 5.4).
Input: _cache/work/decision_points.pkl. Output: results/final/idm_sensitivity.csv."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import model_lib as G
import mobil_model as M
D = G.steps('causal')
Gp = M.gaps(D)
m = ((D.cls == 'car') & (D.dirn == 'in') & D.day.isin(G.MAIN)).values
S = D[m].reset_index(drop=True)
ev = G.event_ids(S)
fr = S.free.values.astype(float); dd = S.diff_day.values.astype(float)
rows = []
for name, P in [('base', {}), ('a+30', dict(a=1.56)), ('a-30', dict(a=0.84)), ('b+30', dict(b=2.73)), ('b-30', dict(b=1.47)),
                ('T+30', dict(T=1.95)), ('T-30', dict(T=1.05)), ('s0+30', dict(s0=3.25)), ('no anticipation', dict(tau=0.0))]:
    PP = dict(M.BASE); PP.update(P)
    Fm = M.mobil_terms(D, Gp, 'M0', P=PP)
    Fm = Fm[m].reset_index(drop=True)
    X, cols = G.design(S, Fm)
    Xa = np.column_stack([X, fr, fr*dd]); b, ll = G.ich_fit(Xa, ev)
    rows.append(dict(variant=name, beta_I=b[1], beta_F=b[-1], beta_f=b[-2], a_bias=b[-1]/b[1], a_free=b[-2]/b[1], sigma=1/b[1], LL=ll))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(G.FIN, 'idm_sensitivity.csv'), index=False)
