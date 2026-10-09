# -*- coding: utf-8 -*-
"""Permutation draws for the lane-limit terms (cars, inward and outward; desired-speed proxy with whole-track fallback). The eight
(V_in, V_out) pairs are reassigned to the days at random; every limit-dependent quantity (M1 / M1p incentives, speed factor, Delta,
differentiated-day label, permitted gain) is recomputed and the models refitted. Statistics per draw: log-likelihood gains of M1 and M1p
over M0, bias_lin (M2L), bias_step (M2s) and a mechanism model. Draw 0 (chunk 0) is the observed assignment. Slow; run_all.py runs it only
with --all.
usage: python decisions_permutation.py NDRAWS CHUNK -> results/permutation/dec_perm_draws_c<CHUNK>.csv (read by model_permutation_tests.py)"""
import time
from lc_common import *
import mobil_model as M
from interval_censored import *
from decisions_models import prep, design, BASECOLS
M.AMIN = -4.0
MAIN = [n for n, _, _ in DAYS]


def fit_all(D, G, vin_map, vout_map, groups):
    D = D.copy()
    D['vin'] = D.day.map(vin_map).values; D['vout'] = D.day.map(vout_map).values
    D['delta'] = D.vin - D.vout; D['d10'] = D.delta / 10; D['diff_day'] = (D.delta > 0).astype(int)
    D['sf_use'] = M.sf_from(D, D.vin.values.astype(float), D.vout.values.astype(float))
    D['gain10'] = (np.minimum(D.vdes_use, D.vin) - np.minimum(D.vdes_use, D.vout)).clip(lower=0) / 10
    F = {s: M.mobil_terms(D, G, s) for s in ('M0', 'M1', 'M1p')}
    out = {}
    for cls, dirn in groups:
        m = ((D.cls == cls) & (D.dirn == dirn)).values
        S = D[m].reset_index(drop=True); ev = event_index(S)
        Fm = {k: v[m].reset_index(drop=True) for k, v in F.items()}
        X0, _ = design(S, Fm['M0']); b0, ll0 = ich_fit(X0, ev)
        X1, _ = design(S, Fm['M1']); _, ll1 = ich_fit(X1, ev, b0=b0)
        Xp, _ = design(S, Fm['M1p']); _, llp = ich_fit(Xp, ev, b0=b0)
        XL, _ = design(S, Fm['M0'], 'lin'); bL, llL = ich_fit(XL, ev, b0=np.r_[b0, 0])
        XS, _ = design(S, Fm['M0'], 'step'); bS, llS = ich_fit(XS, ev, b0=np.r_[b0, 0])
        XM = np.column_stack([X0, S.diff_day.values, S.diff_day.values * S.gain10.values]); bM, llM = ich_fit(XM, ev, b0=np.r_[b0, 0, 0])
        XV = np.column_stack([X0, S.diff_day.values, S.diff_day.values * S.vd10.values]); bV, llV = ich_fit(XV, ev, b0=np.r_[b0, 0, 0])
        g = f'{cls}_{dirn}'
        out.update({f'{g}_LL0': ll0, f'{g}_dLL_M1': ll1 - ll0, f'{g}_dLL_M1p': llp - ll0, f'{g}_bias_lin': bL[-1], f'{g}_bias_lin_ms2': bL[-1] / bL[1],
                    f'{g}_dLL_M2L': llL - ll0, f'{g}_bias_step': bS[-1], f'{g}_dLL_M2s': llS - ll0,
                    f'{g}_mech_diff': bM[-2], f'{g}_mech_diffxgain': bM[-1], f'{g}_mechv_diff': bV[-2], f'{g}_mechv_diffxvd': bV[-1]})
    return out


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 0          # parallel chunks: chunk 0 includes the observed assignment (draw 0)
    D, G, _ = prep(thetas=())
    keep = (D.cls == 'car')
    D = D[keep].reset_index(drop=True); G = M.gaps(D)
    pairs = [(vi, vo) for _, vi, vo in DAYS]
    rng = np.random.default_rng(20260926 + chunk)
    rows = []; t0 = time.time()
    outp = os.path.join(RES, 'permutation', f'dec_perm_draws_c{chunk}.csv'); os.makedirs(os.path.dirname(outp), exist_ok=True)
    for k in range(0 if chunk == 0 else 1, N + 1):
        perm = np.arange(8) if k == 0 else rng.permutation(8)
        vin_map = {MAIN[i]: pairs[perm[i]][0] for i in range(8)}; vout_map = {MAIN[i]: pairs[perm[i]][1] for i in range(8)}
        r = fit_all(D, G, vin_map, vout_map, (('car', 'in'), ('car', 'out')))
        r['draw'] = k; r['chunk'] = chunk; r['perm'] = ''.join(map(str, perm)); rows.append(r)
        if k % 50 == 0:
            pd.DataFrame(rows).to_csv(outp, index=False)
            print(k, round(time.time() - t0, 1), 's', {kk: round(v, 3) for kk, v in r.items() if 'car_in' in kk and ('dLL' in kk or 'bias_lin' == kk[-8:])}, flush=True)
    pd.DataFrame(rows).to_csv(outp, index=False); print('saved', len(rows))
