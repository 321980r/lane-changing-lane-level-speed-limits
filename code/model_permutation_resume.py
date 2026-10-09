# -*- coding: utf-8 -*-
"""Permutation draws as model_permutation.py, resumable: draw k of chunk c uses its own seed (777 + 1000 c + k); draws are appended every 10
draws and draws already in the output are skipped. Slow; run_all.py runs it only with --all.
usage: python model_permutation_resume.py NDRAWS CHUNK -> results/permutation/model_perm_causal_r<CHUNK>.csv"""
import time, gc
from model_lib import *

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 2
out = os.path.join(RES, 'permutation', f'model_perm_causal_r{chunk}.csv'); os.makedirs(os.path.dirname(out), exist_ok=True)
done = set(pd.read_csv(out).draw) if os.path.exists(out) else set()
D = steps('causal', 0)
D = D[(D.cls == 'car') & D.day.isin(MAIN)].reset_index(drop=True); gc.collect()
G = M.gaps(D)
pairs = [(vi, vo) for _, vi, vo in DAYS]
F0 = M.mobil_terms(D, G, 'M0')
base = {}
for dirn in ('in', 'out'):
    m = (D.dirn == dirn).values; S = D[m].reset_index(drop=True); ev = event_ids(S)
    X0, _ = design(S, F0[m].reset_index(drop=True)); b0, ll0 = ich_fit(X0, ev)
    base[dirn] = (m, S, ev, X0, b0, ll0)
Dp0 = D[['day', 'Lane', 'sv1', 'sv2', 'n1', 'n2', 'vdes_use', 'PositionX', 'v', 'truck']].copy()
rows = []; t0 = time.time()


def flush():
    global rows
    if rows:
        pd.DataFrame(rows).to_csv(out, mode='a', header=not os.path.exists(out), index=False)
        rows = []


for k in range(1, N + 1):
    if k in done:
        continue
    perm = np.random.default_rng(777 + 1000 * chunk + k).permutation(8)
    vin_map = {MAIN[i]: pairs[perm[i]][0] for i in range(8)}; vout_map = {MAIN[i]: pairs[perm[i]][1] for i in range(8)}
    Dp = Dp0.copy()
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
    del Dp, F1, Fp
    if k % 10 == 0:
        flush(); gc.collect()
        print(k, round(time.time() - t0, 1), 's', flush=True)
flush(); print('finished chunk', chunk)
