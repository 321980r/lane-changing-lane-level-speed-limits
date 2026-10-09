# -*- coding: utf-8 -*-
"""Exact day-level test of the bias of unconstrained cars: M2F = M0 + F + F x D is refitted for all 56 labellings of three of the
eight days as same-limit; the same term on decision points with a leader, M0 + F + (1 - F) x D, is tested as a contrast. Car inward
changes, and for comparison car outward and heavy inward changes (causal desired-speed proxy).
usage: python model_permutation_bias.py car in | car out | truck in | tests   ('tests' pools the three groups)
Input: _cache/work/decision_points.pkl. Outputs: results/final/model_perm_free_exact_<class>_<direction>.csv, model_perm_free_tests.csv."""
import gc
from itertools import combinations
from model_lib import *

arg = sys.argv[1:3]
if arg and arg[0] != 'tests':
    cls0, dirn0 = arg
    D = steps('causal', 0)
    D = D[D.day.isin(MAIN) & (D.cls == cls0) & (D.dirn == dirn0)].reset_index(drop=True); gc.collect()
    F0 = M.mobil_terms(D, M.gaps(D), 'M0')
    groups = [(cls0, dirn0)]
else:
    groups = []
rows = []
for cls, dirn in groups:
    m = ((D.cls == cls) & (D.dirn == dirn)).values
    S = D[m].reset_index(drop=True); Fm = F0[m].reset_index(drop=True); ev = event_ids(S)
    X0, _ = design(S, Fm); fr = S.free.values.astype(float)
    Xb = np.column_stack([X0, fr]); bb, llb = ich_fit(Xb, ev)
    for same in combinations(range(8), 3):
        lab = {MAIN[i]: (0 if i in same else 1) for i in range(8)}
        dd = S.day.map(lab).values.astype(float)
        bF, llF = ich_fit(np.column_stack([Xb, fr * dd]), ev, b0=np.r_[bb, 0.0])
        bL, llL = ich_fit(np.column_stack([Xb, (1 - fr) * dd]), ev, b0=np.r_[bb, 0.0])
        rows.append(dict(cls=cls, dirn=dirn, same=''.join(map(str, same)), observed=int(same == (0, 1, 2)),
                         b_free_diff=bF[-1], dLL_free_diff=llF - llb, b_lead_diff=bL[-1], dLL_lead_diff=llL - llb))
    print(cls, dirn, 'done', flush=True)
    del S, Fm; gc.collect()
if groups:
    pd.DataFrame(rows).to_csv(os.path.join(FIN, f'model_perm_free_exact_{groups[0][0]}_{groups[0][1]}.csv'), index=False)
    sys.exit(0)
import glob
R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(FIN, 'model_perm_free_exact_*.csv')))], ignore_index=True)
T = []
for (cls, dirn), g in R.groupby(['cls', 'dirn'], sort=False):
    o = g[g.observed == 1].iloc[0]
    for st in ('b_free_diff', 'dLL_free_diff', 'b_lead_diff', 'dLL_lead_diff'):
        v = g[st].values
        T.append(dict(cls=cls, dirn=dirn, stat=st, observed=o[st], p_one_upper=float(np.mean(v >= o[st] - 1e-9)),
                      p_two_abs=float(np.mean(np.abs(v) >= abs(o[st]) - 1e-9)), rank_of_obs=int((v > o[st] + 1e-9).sum() + 1), n=len(v)))
T = pd.DataFrame(T); T.to_csv(os.path.join(FIN, 'model_perm_free_tests.csv'), index=False)
print(T.round(4).to_string(index=False))
