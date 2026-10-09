# -*- coding: utf-8 -*-
"""Closed-loop simulation of the stochastic MOBIL rule in SUMO 1.18 via TraCI (Section 5.5).
Same network, replayed demand, IDM and counting as simulation.py. SUMO's own lane changing is switched off (laneChangeMode 512;
requested changes respect the safe gaps of others). Every 1 s, each vehicle at corridor x in [1,000, 2,000) m gets the model's 1-s hazard
    h = logistic(b . [1, dA_ego, dA_n, dA_o, unsafe, (v_des - 80)/10, entered_by_lc (, free, free x 1[Delta > 0])])
with the MOBIL terms computed by mobil_model.gaps / mobil_terms from its simulated neighbours (the code of the estimation), and changes lane
with probability h. Coefficients: leave-one-day-out fits (results/final/model_lodo_coefs.csv).
Specifications: M2F (cars in: M0 + free + free x diff; cars out and heavy vehicles: M0; absolute desired speeds) and M1p (cars in and out:
lane desired speed = speed factor x lane limit; heavy vehicles: M0; speed-factor desired speeds). Five seeds per day.
Requires SUMO 1.18 with SUMO_HOME set. Slow; run_all.py runs it only with --all.
usage: python simulation_closed_loop.py test | run [nworkers] [specs]   (resumable: runs already in the output are skipped)
Output: results/final/sim_closedloop_runs.csv"""
import os, sys, time, gc
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
sys_path_tools = os.path.join(os.environ.get('SUMO_HOME', ''), 'tools')     # SUMO 1.18 TraCI (set SUMO_HOME to the SUMO installation)
sys.path.insert(0, sys_path_tools)
import simulation as SG
from simulation import *          # noqa  (write_net, write_routes, summarise, R, Mt, WD, EDGES, COUNT, ENTRY, EXIT, SUMO ...)
import mobil_model as M
from scipy.special import expit

M.AMIN = -4.0
COEF = pd.read_csv(os.path.join(DATF, 'model_lodo_coefs.csv'))
BASE = ['const', 'dA_ego', 'dA_n', 'dA_o', 'unsafe', 'vd10', 'state']
SPECS = {'M2F': {'types': 'ABS', ('car', 'in'): ('M2F', 'M0'), ('car', 'out'): ('M0', 'M0'), ('truck', 'in'): ('M0', 'M0'), ('truck', 'out'): ('M0', 'M0')},
         'M1p': {'types': 'REL', ('car', 'in'): ('M1p', 'M1p'), ('car', 'out'): ('M1p', 'M1p'), ('truck', 'in'): ('M0', 'M0'), ('truck', 'out'): ('M0', 'M0')}}
MAIN = [n for n, _, _ in DAYS]


def coefs(day, cls, dirn, model):
    lo = day if day in MAIN else 'none(8 days)'
    r = COEF[(COEF.cls == cls) & (COEF.dirn == dirn) & (COEF.model == model) & (COEF.left_out == lo)].iloc[0]
    cols = BASE + (['free', 'free_x_diff'] if model == 'M2F' else [])
    return np.array([r[c] for c in cols], float)


def run_cl(day, spec, seed):
    import traci
    vi, vo = float(Mt.vin[day]), float(Mt.vout[day]); diff = float(vi != vo)
    S = SPECS[spec]; variant = S['types']
    tag = f'CL{spec}_{day[5:15]}_s{seed}'
    net = write_net(int(vi), int(vo), 'cfg')
    rou, types, tend = write_routes(tag, day, variant, {}, seed)
    V0, sf, vmax = SG.draw_types(day, variant, seed)
    V0 = V0.copy(); V0['cls'] = np.where(V0.truck == 1, 'heavy', np.where(V0.truck == 0, 'car', 'unk'))
    V0['vid'] = [f'{c[0]}{k}' for k, c in enumerate(V0.cls)]
    info = {r.vid: (int(r.truck == 1), sf[k], vmax[k]) for k, r in V0.iterrows()}
    B = {(c, d): coefs(day, c, d, S[(c, d)][0]) for c in ('car', 'truck') for d in ('in', 'out')}
    SP = {(c, d): S[(c, d)][1] for c in ('car', 'truck') for d in ('in', 'out')}
    add = os.path.join(WD, f'{tag}.add.xml'); lcx = os.path.join(WD, f'{tag}.lc.xml')
    ld = {c: os.path.join(WD, f'{tag}.ld_{c}.xml') for c in ('car', 'heavy')}
    end = int(tend + 900)
    with open(add, 'w') as f:
        f.write('<additional>')
        for c in ('car', 'heavy'):
            f.write(f'<laneData id="ld_{c}" file="{ld[c]}" begin="0" end="{end}" excludeEmpty="true" vTypes="{" ".join(types[c])}"/>')
        f.write('</additional>')
    cmd = [os.path.join(SUMO, 'sumo'), '-n', net, '-r', rou, '-a', add, '--lanechange-output', lcx, '--seed', str(seed), '--no-step-log',
           '--no-warnings', '--begin', '0', '--end', str(end), '--step-length', '0.5', '--collision.action', 'warn', '--max-depart-delay', '60']
    traci.start(cmd, label=tag)
    con = traci.getConnection(tag)
    rng = np.random.default_rng(10_000 * seed + ORDER8.index(day) + (0 if spec == 'M2F' else 500))
    changed = set(); n_trig = 0; n_steps = 0; n_ins = 0
    t = 0.0
    while t < end:
        con.simulationStep(); t = con.simulation.getTime()
        for vid in con.simulation.getDepartedIDList():
            con.vehicle.setLaneChangeMode(vid, 512); n_ins += 1
        if abs(t - round(t)) > 1e-6:
            continue
        ids = con.vehicle.getIDList()
        if not ids:
            continue
        n = len(ids)
        x = np.empty(n); v = np.empty(n); ln = np.empty(n, int)
        for i, vid in enumerate(ids):
            x[i] = con.vehicle.getPosition(vid)[0] + 1000.0
            v[i] = con.vehicle.getSpeed(vid) * 3.6
            ln[i] = 1 if con.vehicle.getLaneIndex(vid) == 1 else 2
        tr = np.array([info[vid][0] for vid in ids]); sfa = np.array([info[vid][1] for vid in ids]); vmx = np.array([info[vid][2] for vid in ids])
        act = np.where((x >= 1000) & (x < 2000))[0]
        if not len(act):
            continue
        rows = []
        for i in act:
            own = (ln == ln[i]); oth = ~own
            r = dict(PositionX=x[i], v=v[i], Lane=ln[i], truck=tr[i])
            for pre, mask, ahead in (('ol', own, True), ('of', own, False), ('tl', oth, True), ('tf', oth, False)):
                cand = np.where(mask & ((x > x[i]) if ahead else (x < x[i])))[0]
                cand = cand[cand != i]
                if len(cand):
                    j = cand[np.argmin(x[cand])] if ahead else cand[np.argmax(x[cand])]
                    r[f'{pre}_PositionX'], r[f'{pre}_v'], r[f'{pre}_truck'] = x[j], v[j], tr[j]
                else:
                    r[f'{pre}_PositionX'], r[f'{pre}_v'], r[f'{pre}_truck'] = np.nan, np.nan, np.nan
            Vl = vi if ln[i] == 1 else vo
            vd = min(vmx[i], sfa[i] * Vl)
            r['vdes_use'] = vd; r['sf_use'] = (sfa[i] if variant == 'REL' else vd / Vl)
            r['entered_by_lc'] = int(ids[i] in changed); r['vid'] = ids[i]
            rows.append(r)
        D = pd.DataFrame(rows); D['vin'] = vi; D['vout'] = vo
        G = M.gaps(D)
        hz = np.zeros(len(D))
        for c, tv in (('car', 0), ('truck', 1)):
            for dirn, lane in (('in', 2), ('out', 1)):
                m = ((D.truck.values == tv) & (D.Lane.values == lane))
                if not m.any():
                    continue
                Dm = D[m].reset_index(drop=True); Gm = {k: vv[m] for k, vv in G.items()}
                Fm = M.mobil_terms(Dm, Gm, SP[(c, dirn)])
                X = [np.ones(len(Dm)), Fm.dA_ego.values, Fm.dA_n.values, Fm.dA_o.values, Fm.unsafe.values, (Dm.vdes_use.values - 80) / 10,
                     Dm.entered_by_lc.values.astype(float)]
                b = B[(c, dirn)]
                if len(b) == 9:
                    olg = Dm.ol_PositionX.values - Dm.PositionX.values
                    fr = (np.isnan(olg) | (olg > 300)).astype(float)
                    X += [fr, fr * diff]
                hz[m] = expit(np.column_stack(X) @ b)
        n_steps += len(D)
        u = rng.random(len(D))
        for k in np.where(u < hz)[0]:
            vid = D.vid.iloc[k]; tgt = 1 if D.Lane.iloc[k] == 2 else 0
            con.vehicle.changeLane(vid, tgt, 1.0)
            changed.add(vid); n_trig += 1
        if t > tend + 60 and not con.vehicle.getIDCount():
            break
    con.close()
    # ----- same measurement as simulation.run
    L = pd.DataFrame([e.attrib for e in ET.parse(lcx).getroot()])
    out = dict(day=day, cfg=Mt.cfg[day], delta=int(Mt.delta[day]), variant=f'CL_{spec}', pre='cfg', seed=seed, n_trig=n_trig, n_decisions=n_steps, n_inserted=n_ins, n_planned=len(V0))
    km = {}
    for c in ('car', 'heavy'):
        for lane in ET.parse(ld[c]).getroot().iter('lane'):
            e, li = lane.attrib['id'].rsplit('_', 1)
            km[(c, e, int(li))] = km.get((c, e, int(li)), 0.0) + float(lane.attrib.get('sampledSeconds', 0)) * float(lane.attrib.get('speed', 0)) / 1000.0
    if len(L):
        for c in ('time', 'dir'):
            L[c] = pd.to_numeric(L[c], errors='coerce')
        L['edge'] = L['from'].str.rsplit('_', n=1).str[0]
        L['olg'] = pd.to_numeric(L.get('origLeaderGap', pd.Series(np.nan, index=L.index)), errors='coerce')
    for c, pfx in (('car', 'c'), ('heavy', 'h')):
        s = L[L.id.str.startswith(pfx) & L.edge.isin(COUNT)] if len(L) else pd.DataFrame(columns=['dir', 'olg', 'reason', 'id'])
        inn = s[s.dir > 0]; ou = s[s.dir < 0]
        out[f'{c}_n_in'] = len(inn); out[f'{c}_n_out'] = len(ou)
        out[f'{c}_n_in_free'] = int(((inn.olg < 0) | (inn.olg > 300) | inn.olg.isna()).sum()) if len(inn) else 0
        dep0 = set(V0.loc[(V0.cls == c) & (V0.lane_ins == 2), 'vid'])
        out[f'{c}_n_dep_L2'] = len(dep0)
        out[f'{c}_n_dep_L2_in'] = len(set(inn.id) & dep0) if len(inn) else 0
        out[f'{c}_km_L2'] = sum(km.get((c, e, 0), 0.0) for e in COUNT); out[f'{c}_km_L1'] = sum(km.get((c, e, 1), 0.0) for e in COUNT)
        for nm, e in (('entry', ENTRY), ('exit', EXIT)):
            a, b_ = km.get((c, e, 1), 0.0), km.get((c, e, 0), 0.0)
            out[f'{c}_share_L1_{nm}'] = a / (a + b_) if a + b_ > 0 else np.nan
    for q in (rou, lcx, add, *ld.values()):
        try:
            os.remove(q)
        except OSError:
            pass
    return out


def _job(a):
    return run_cl(*a)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'test'
    outcsv = os.path.join(DATF, 'sim_closedloop_runs.csv')
    if what == 'test':
        t0 = time.time()
        r = run_cl(ORDER8[7], 'M2F', 1)
        print(summarise(pd.DataFrame([r])).T.to_string()); print(round(time.time() - t0, 1), 's')
    else:
        nw = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        specs = sys.argv[3].split(',') if len(sys.argv) > 3 else ['M2F', 'M1p']
        done = pd.read_csv(outcsv) if os.path.exists(outcsv) else pd.DataFrame(columns=['day', 'variant', 'seed'])
        key = set(zip(done.day, done.variant, done.seed))
        jobs = [(d, sp, s) for sp in specs for d in ORDER8 for s in (1, 2, 3, 4, 5) if (d, f'CL_{sp}', s) not in key]
        print(len(jobs), 'jobs', flush=True)
        rows = [done] if len(done) else []
        t0 = time.time()
        with ProcessPoolExecutor(nw) as ex:
            for k, r in enumerate(ex.map(_job, jobs)):
                rows.append(pd.DataFrame([r]))
                print(k + 1, '/', len(jobs), r['day'][5:15], r['variant'], r['seed'], round(time.time() - t0, 1), 's', flush=True)
                summarise(pd.concat(rows, ignore_index=True)).to_csv(outcsv, index=False)
