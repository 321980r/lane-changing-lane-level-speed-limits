# -*- coding: utf-8 -*-
"""Model helpers used by decisions_permutation.py: prep() (decision points of decision_points.pkl with the MOBIL terms of M0, M1, M1p and M1s) and design()
(design matrix with optional bias terms). Run as a script, it fits the model versions on the eight days (interval-censored 1-s hazard,
interval_censored.py) and writes results/dec_model_compare.csv, dec_params.csv, dec_day_bias.csv, dec_day_repro.csv and _cache/work/dec_feat.pkl; these
outputs are not used by the other scripts.
Terms: const, dA_ego, dA_n, dA_o, unsafe (new follower would brake harder than 4 m/s^2), vd10 (desired speed per 10 km/h above 80),
state (current lane entered by a lane change). Versions: M0 v0 = vdes; M1 v0 = min(vdes, V_lane); M1p v0 = sf x V_lane;
M1s v0 = min(vdes, V_lane + theta); M2s / M2L = M0 + bias (step / per 10 km/h); M3s / M3L = M1 + bias; M3pL = M1p + bias."""
import time
from lc_common import *
import mobil_model as M
from interval_censored import *
M.AMIN = -4.0
pd.set_option('display.width', 250)
GROUPS = (('car', 'in'), ('car', 'out'), ('truck', 'in'), ('truck', 'out'))
BASECOLS = ['dA_ego', 'dA_n', 'dA_o', 'unsafe', 'vd10', 'state']      # state = the current lane stint began with a lane change inside the window
SAME = ['scen_2022-08-08_60-60', 'scen_2022-07-29_80-80', 'scen_2022-08-11_100-100']
MAIN = [n for n, _, _ in DAYS]


def prep(x_lo=1000.0, x_hi=1800.0, H=5, vdes_src='pre', P=M.BASE, bounded=True, thetas=(0, 10, 20, 30)):
    D = M.load_sample(x_lo, x_hi, H, vdes_src)
    G = M.gaps(D)
    D['vd10'] = (D.vdes_use - 80) / 10; D['d10'] = D.delta / 10
    D['olgap'] = D.ol_PositionX - D.PositionX
    D['free'] = (D.olgap.isna() | (D.olgap > 300)).astype(int)
    D['gain10'] = (np.minimum(D.vdes_use, D.vin) - np.minimum(D.vdes_use, D.vout)).clip(lower=0) / 10
    feats = {}
    for spec in ('M0', 'M1', 'M1p'):
        feats[spec] = M.mobil_terms(D, G, spec, P, bounded)
    for th in thetas:
        feats[f'M1s{th}'] = M.mobil_terms(D, G, 'M1s', P, bounded, theta=th)
    return D, G, feats


def design(S, F, bias=None):
    cols = BASECOLS[:]
    X = [np.ones(len(S)), F.dA_ego.values, F.dA_n.values, F.dA_o.values, F.unsafe.values, S.vd10.values, S.entered_by_lc.values.astype(float)]
    if bias == 'step':
        X.append(S.diff_day.values.astype(float)); cols.append('bias_step')
    elif bias == 'lin':
        X.append(S.d10.values); cols.append('bias_per10')
    elif bias == 'day':
        for d in MAIN[1:]:
            X.append((S.day == d).values.astype(float)); cols.append('day_' + d[10:])
    return np.column_stack(X), ['const'] + cols


MODELS = [('M0', 'M0', None), ('M1', 'M1', None), ('M1p', 'M1p', None), ('M1s', 'M1s*', None),
          ('M2s', 'M0', 'step'), ('M2L', 'M0', 'lin'), ('M3s', 'M1', 'step'), ('M3L', 'M1', 'lin'), ('M3pL', 'M1p', 'lin')]


def run_group(D, feats, cls, dirn, verbose=True):
    m = ((D.cls == cls) & (D.dirn == dirn)).values
    S = D[m].reset_index(drop=True)
    tr = np.ones(len(S), bool)
    Ftr = {k: v[m].reset_index(drop=True) for k, v in feats.items()}
    ev_all = event_index(S)
    St = S[tr].reset_index(drop=True); ev = event_index(St)
    # profile theta for M1s on the training days
    best = None
    for k in feats:
        if k.startswith('M1s'):
            X, _ = design(St, Ftr[k][tr].reset_index(drop=True)); b, ll = ich_fit(X, ev)
            if best is None or ll > best[1]:
                best = (k, ll)
    rows, fits = [], {}
    for name, spec, bias in MODELS:
        key = best[0] if spec == 'M1s*' else spec
        F = Ftr[key]; Ft = F[tr].reset_index(drop=True)
        X, cols = design(St, Ft, bias); b, ll = ich_fit(X, ev)
        se, V, _ = ich_se(b, X, ev, St.GID.values)
        k = len(b) + (1 if spec == 'M1s*' else 0)
        # leave-one-day-out
        lodo = 0.0
        for d in MAIN:
            a = (St.day != d).values
            bb, _ = ich_fit(X[a], event_index(St[a].reset_index(drop=True)), b0=b)
            lodo += ich_ll(bb, X[~a], ev[~a])
        rows.append(dict(group=f'{cls}_{dirn}', model=name, v0_spec=key, bias=bias or '', k=k, LL=ll, AIC=2 * k - 2 * ll, LODO_LL=lodo,
                         n_steps=len(St), n_events=int(ev.max() + 1), n_veh=St.GID.nunique(), n_days=St.day.nunique()))
        fits[name] = dict(b=b, se=se, V=V, cols=cols, key=key, bias=bias)
        if verbose:
            print(f'  {cls}-{dirn} {name:5s} ({key:6s}) LL={ll:8.2f} AIC={2 * k - 2 * ll:8.2f} LODO={lodo:8.2f}  ' +
                  ' '.join(f'{c}={x:+.3f}({s:.3f})' for c, x, s in zip(cols, b, se) if c not in ('const',)), flush=True)
    return S, Ftr, rows, fits


def to_accel(fit):
    """MOBIL-scale parameters: sigma = 1/b_ego, threshold = -const/b_ego, p_n, p_o, bias in m/s^2 (delta method)"""
    b, V, cols = fit['b'], fit['V'], fit['cols']
    ie = cols.index('dA_ego'); out = []
    def ratio(i, sign=1.0):
        r = sign * b[i] / b[ie]
        g = np.zeros(len(b)); g[i] = sign / b[ie]; g[ie] = -sign * b[i] / b[ie] ** 2
        return r, float(np.sqrt(g @ V @ g))
    for i, c in enumerate(cols):
        if c == 'dA_ego':
            out.append(('beta_ego (1/(m/s2))', b[i], np.sqrt(V[i, i])))
            out.append(('sigma = 1/beta_ego (m/s2)', 1 / b[i], np.sqrt(V[i, i]) / b[i] ** 2))
        elif c == 'const':
            r, s = ratio(i, -1.0); out.append(('threshold dA_th (m/s2)', r, s))
        elif c == 'dA_n':
            r, s = ratio(i); out.append(('p_new_follower', r, s))
        elif c == 'dA_o':
            r, s = ratio(i); out.append(('p_old_follower', r, s))
        elif c == 'unsafe':
            out.append(('unsafe (logit)', b[i], np.sqrt(V[i, i])))
        elif c == 'state':
            r, s = ratio(i); out.append(('return motivation after an own lane change (m/s2)', r, s))
        elif c == 'vd10':
            r, s = ratio(i); out.append(('threshold shift per +10 km/h desired speed (m/s2)', -r, s))
        else:
            r, s = ratio(i); out.append((c + ' (m/s2)', r, s))
    return out


if __name__ == '__main__':
    t0 = time.time()
    D, G, feats = prep()
    pd.to_pickle((D, feats), os.path.join(CACHE, 'dec_feat.pkl'))
    print('features ready', round(time.time() - t0, 1), 's; steps', len(D))
    allrows, prm, dayb, repro = [], [], [], []
    for cls, dirn in GROUPS:
        S, Ftr, rows, fits = run_group(D, feats, cls, dirn)
        allrows += rows
        for name in ('M0', 'M1', 'M2L', 'M3L', 'M3s', 'M1p', 'M3pL'):
            for lab, v, s in to_accel(fits[name]):
                prm.append(dict(group=f'{cls}_{dirn}', model=name, param=lab, est=v, se=s, lo=v - 1.96 * s, hi=v + 1.96 * s))
        # ---- per-day lane bias (day intercepts) on top of M1 and M0
        tr = np.ones(len(S), bool); St = S[tr].reset_index(drop=True); ev = event_index(St)
        for base in ('M0', 'M1'):
            F = Ftr[base][tr].reset_index(drop=True)
            X, cols = design(St, F, 'day'); b, ll = ich_fit(X, ev); se, V, _ = ich_se(b, X, ev, St.GID.values)
            ie = cols.index('dA_ego')
            # day intercepts relative to the mean of the three same-limit days: alpha_d = const + day_d (0808 is the reference)
            A = np.zeros((len(MAIN), len(b)))
            for j, d in enumerate(MAIN):
                A[j, 0] = 1.0
                if j > 0:
                    A[j, cols.index('day_' + d[10:])] = 1.0
            alpha = A @ b; ref = A[:3].mean(0)
            for j, (d, vi, vo) in enumerate(DAYS):
                cvec = A[j] - ref; val = cvec @ b
                est = val / b[ie]
                g = cvec / b[ie]; g = g.copy(); g[ie] -= val / b[ie] ** 2
                s = float(np.sqrt(g @ V @ g))
                dayb.append(dict(group=f'{cls}_{dirn}', base=base, day=d, cfg=f'{vi}/{vo}', delta=vi - vo, logit_shift=val, logit_se=float(np.sqrt(cvec @ V @ cvec)),
                                 a_bias=est, a_bias_se=s, n_events=int(len(np.unique(ev[(St.day == d).values & (ev >= 0)]))), n_steps=int((St.day == d).sum()), n_veh=St[St.day == d].GID.nunique()))
        # ---- day-level reproduction (leave-one-day-out predictions of counts, by subset)
        subsets = {'all': np.ones(len(S), bool)}
        if dirn == 'in':
            subsets['free'] = (S.free == 1).values; subsets['constrained'] = (S.free == 0).values
        else:
            subsets['after_lc'] = (S.entered_by_lc == 1).values; subsets['no_prior_lc'] = (S.entered_by_lc == 0).values
        evS = event_index(S)
        first = pd.Series(np.arange(len(S)))[evS >= 0].groupby(evS[evS >= 0]).first().values     # first step of each event window
        isfirst = np.zeros(len(S), bool); isfirst[first] = True
        for name in ('M0', 'M1', 'M1p', 'M2L', 'M2s', 'M3L', 'M3s', 'M3pL'):
            fit = fits[name]; F = Ftr[fit['key']]
            X, _ = design(S, F, fit['bias'])
            for d in [n for n, _, _ in DAYS]:
                a = (S.day != d).values
                bb, _ = ich_fit(X[a], event_index(S[a].reset_index(drop=True)), b0=fit['b'])
                sel = (S.day == d).values
                h = expected_events(bb, X[sel], S[sel])
                for sub, msk in subsets.items():
                    ms = msk[sel]
                    repro.append(dict(group=f'{cls}_{dirn}', model=name, day=d, cfg=label(d), delta=delta(d), subset=sub, obs=int((isfirst[sel] & ms).sum()),
                                      pred=float(h[ms].sum()), hours=float(ms.sum()) / 3600.0))
        print(f'{cls}-{dirn} done', round(time.time() - t0, 1), 's', flush=True)
    T = pd.DataFrame(allrows); T.to_csv(os.path.join(RES, 'dec_model_compare.csv'), index=False)
    pd.DataFrame(prm).to_csv(os.path.join(RES, 'dec_params.csv'), index=False)
    pd.DataFrame(dayb).to_csv(os.path.join(RES, 'dec_day_bias.csv'), index=False)
    pd.DataFrame(repro).to_csv(os.path.join(RES, 'dec_day_repro.csv'), index=False)
    print(T[['group', 'model', 'k', 'LL', 'AIC', 'LODO_LL']].round(2).to_string(index=False))
