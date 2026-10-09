# -*- coding: utf-8 -*-
"""Permutation draws for the lane-limit terms with the causal desired-speed proxy (cars, inward and outward). The eight (V_in, V_out)
pairs are reassigned to the days at random; the M1 / M1p incentives, the speed factor and Delta are recomputed and the models refitted.
Statistics: dLL_M1 = LL(M1) - LL(M0), dLL_M1p, dLL_M1p_M1, b_M2L (bias linear in Delta), dLL_M2L. Draw 0 (chunk 0) is the observed
assignment. Slow; run_all.py runs it only with --all.
usage: python model_permutation.py NDRAWS CHUNK causal -> results/permutation/model_perm_causal_c<CHUNK>.csv"""
import time, gc
from model_lib import *

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 0
vsrc = sys.argv[3] if len(sys.argv) > 3 else 'causal'
D = steps(vsrc, 0)
D = D[(D.cls == 'car') & D.day.isin(MAIN)].reset_index(drop=True); gc.collect()
G = M.gaps(D)
pairs = [(vi, vo) for _, vi, vo in DAYS]
F0 = M.mobil_terms(D, G, 'M0')
base = {}
for dirn in ('in', 'out'):
    m = (D.dirn == dirn).values; S = D[m].reset_index(drop=True); ev = event_ids(S)
    X0, _ = design(S, F0[m].reset_index(drop=True)); b0, ll0 = ich_fit(X0, ev)
    base[dirn] = (m, S, ev, X0, b0, ll0)
rng = np.random.default_rng(777 + chunk)
out = os.path.join(RES, 'permutation', f'model_perm_{vsrc}_c{chunk}.csv'); os.makedirs(os.path.dirname(out), exist_ok=True)
rows = []; t0 = time.time()
for k in range(0 if chunk == 0 else 1, N + 1):
    perm = np.arange(8) if k == 0 else rng.permutation(8)
    vin_map = {MAIN[i]: pairs[perm[i]][0] for i in range(8)}; vout_map = {MAIN[i]: pairs[perm[i]][1] for i in range(8)}
    Dp = D[['day', 'Lane', 'sv1', 'sv2', 'n1', 'n2', 'vdes_use', 'PositionX', 'v', 'truck']].copy()
    Dp['vin'] = Dp.day.map(vin_map).astype(float); Dp['vout'] = Dp.day.map(vout_map).astype(float)
    Dp['sf_use'] = M.sf_from(Dp, Dp.vin.values, Dp.vout.values)
    F1 = M.mobil_terms(Dp, G, 'M1'); Fp = M.mobil_terms(Dp, G, 'M1p')
    d10 = ((Dp.vin - Dp.vout) / 10).values
    r = dict(draw=k, chunk=chunk, perm=''.join(map(str, perm)))
    for dirn in ('in', 'out'):
        m, S, ev, X0, b0, ll0 = base[dirn]
        X1, _ = design(S, F1[m].reset_index(drop=True)); _, ll1 = ich_fit(X1, ev, b0=b0)
        Xp, _ = design(S, Fp[m].reset_index(drop=True)); _, llp = ich_fit(Xp, ev, b0=b0)
        XL = np.column_stack([X0, d10[m]]); bL, llL = ich_fit(XL, ev, b0=np.r_[b0, 0.0])
        r.update({f'car_{dirn}_dLL_M1': ll1 - ll0, f'car_{dirn}_dLL_M1p': llp - ll0, f'car_{dirn}_dLL_M1p_M1': llp - ll1,
                  f'car_{dirn}_dLL_M2L': llL - ll0, f'car_{dirn}_b_M2L': bL[-1]})
    rows.append(r)
    if k % 25 == 0:
        pd.DataFrame(rows).to_csv(out, index=False)
        print(k, round(time.time() - t0, 1), 's', {kk: round(v, 2) for kk, v in r.items() if 'car_in' in kk}, flush=True)
pd.DataFrame(rows).to_csv(out, index=False); print('saved', len(rows))
