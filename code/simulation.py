# -*- coding: utf-8 -*-
"""SUMO LC2013 runs of each day with the replayed demand and the fitted desired speeds (model_inputs.py).
Demand: each day's observed vehicles are inserted at corridor x = 1,000 m in their observed entry lane at their observed time, with the
departure speed min(observed first speed, 0.99 x desired speed), so that SUMO's insertion check keeps them (n_inserted and n_planned are
recorded per run). Desired speeds: REL (speed factor x lane limit; heavy vehicles also capped by an absolute speed) or ABS (absolute
desired speed capped at k x lane limit). The configuration applies on all edges; counting zone [1,000, 1,800) m; lane shares of cars at
[1,000, 1,100) and [1,900, 2,000) m. IDM as in the model (accel 1.2, decel 2.1, tau 1.5, minGap 2.5, delta 4), step 0.5 s, 5 seeds.
Heavy-vehicle LC2013 attributes can be passed as 'hv*' keys (e.g. hvSpeedGain -> lcSpeedGain).
Modes: test (one run, printed), default (lcSpeedGain = lcKeepRight = 1), grid (5 lcSpeedGain x 4 lcKeepRight x REL/ABS), classlc and sgr
(further settings, not used by the other scripts). Requires SUMO 1.18 with SUMO_HOME set. Slow; run_all.py runs it only with --all.
usage: python simulation.py test|default|grid|classlc|sgr -> results/final/sim_runs_<mode>.csv (sgr: sim_lodo_runs_sgr.csv)"""
import subprocess, xml.etree.ElementTree as ET, itertools, tempfile, shutil
from concurrent.futures import ThreadPoolExecutor
from plot_style import *

SUMO = os.path.join(os.environ.get('SUMO_HOME', ''), 'bin')              # SUMO 1.18 (set SUMO_HOME to the SUMO installation)
WD = os.path.join(tempfile.gettempdir(), 'lc_sumo'); os.makedirs(WD, exist_ok=True)       # ASCII path for SUMO
EDGES = [('e1a', 0, 100), ('e1b', 100, 250), ('e2', 250, 800), ('e3', 800, 900), ('e4', 900, 1000), ('e5', 1000, 1300)]   # SUMO x = corridor x - 1000
COUNT = ('e1a', 'e1b', 'e2'); ENTRY = 'e1a'; EXIT = 'e4'
FIT = pd.read_csv(os.path.join(DATF, 'model_desired_speed_fits.csv'))
fp = lambda c, v, p: float(FIT[(FIT.cls == c) & (FIT.variant == v) & (FIT.param == p)].value.iloc[0])
R = pd.read_pickle(os.path.join(CACHE, 'model_replay.pkl'))
Mt = DAY_META.set_index('day')
DEP_FAC = 0.99                     # departure speed <= 0.99 x desired speed (IDM insertion check)
IDM = 'carFollowModel="IDM" accel="1.2" decel="2.1" tau="1.5" minGap="2.5" delta="4" emergencyDecel="9"'


def write_net(vi, vo, pre):
    tag = f'net_{vi}_{vo}_{pre}'
    net = os.path.join(WD, f'{tag}.net.xml')
    if os.path.exists(net):
        return net
    nod = os.path.join(WD, f'{tag}.nod.xml'); edg = os.path.join(WD, f'{tag}.edg.xml')
    xs = [EDGES[0][1]] + [b for _, _, b in EDGES]
    with open(nod, 'w') as f:
        f.write('<nodes>' + ''.join(f'<node id="n{i}" x="{x}" y="0" type="priority"/>' for i, x in enumerate(xs)) + '</nodes>')
    with open(edg, 'w') as f:
        f.write('<edges>')
        for i, (e, a, b) in enumerate(EDGES):
            up = b <= 250                                   # upstream of gantry 1 (corridor < 1250 m)
            lo, li = (vi, vi) if (pre == 'in' and up) else (vo, vi)
            f.write(f'<edge id="{e}" from="n{i}" to="n{i + 1}" numLanes="2" speed="{li / 3.6:.4f}"><lane index="0" speed="{lo / 3.6:.4f}"/><lane index="1" speed="{li / 3.6:.4f}"/></edge>')
        f.write('</edges>')
    tmp = net + '.tmp.xml'
    subprocess.run([os.path.join(SUMO, 'netconvert'), '-n', nod, '-e', edg, '-o', tmp, '--no-turnarounds', '--no-warnings'], check=True, capture_output=True)
    os.replace(tmp, net)
    return net


def draw_types(day, variant, seed):
    """per-vehicle (speedFactor, maxSpeed km/h) for the replayed vehicles of a day"""
    rng = np.random.default_rng(seed * 1000 + ORDER8.index(day))
    V = R[R.day == day].reset_index(drop=True)
    n = len(V); heavy = (V.truck == 1).values
    if variant == 'REL':
        sf = np.where(heavy, rng.normal(fp('heavy', 'REL', 'speedFactor mean'), fp('heavy', 'REL', 'speedFactor sd'), n),
                      rng.normal(fp('car', 'REL', 'speedFactor mean'), fp('car', 'REL', 'speedFactor sd'), n))
        sf = np.clip(sf, 0.5, 1.8)
        vmax = np.where(heavy, np.clip(rng.normal(fp('heavy', 'ABS', 'vmax mean (km/h)'), fp('heavy', 'ABS', 'vmax sd (km/h)'), n), 45, 100), 200.0)
    elif variant == 'ABS':
        sf = np.where(heavy, fp('heavy', 'ABS', 'tolerance k (cap = k x lane limit)'), fp('car', 'ABS', 'tolerance k (cap = k x lane limit)'))
        vmax = np.where(heavy, np.clip(rng.normal(fp('heavy', 'ABS', 'vmax mean (km/h)'), fp('heavy', 'ABS', 'vmax sd (km/h)'), n), 45, 100),
                        np.clip(rng.normal(fp('car', 'ABS', 'vmax mean (km/h)'), fp('car', 'ABS', 'vmax sd (km/h)'), n), 50, 160))
    elif variant == 'OWN':            # replay of each vehicle's own free-flow behaviour (speed factor; heavies limited at own free speed)
        sf = V.sf_own.values.clip(0.5, 1.8)
        vmax = np.where(heavy, V.vdes_own.values * 1.03, 200.0)
    return V, sf, vmax


def write_routes(tag, day, variant, lc, seed):
    V, sf, vmax = draw_types(day, variant, seed)
    vi, vo = Mt.vin[day], Mt.vout[day]
    lcc = ' '.join(f'{k}="{v}"' for k, v in lc.items() if k.startswith('lc'))
    lch = ' '.join(f'lc{k[2:]}="{v}"' for k, v in lc.items() if k.startswith('hv'))
    lch = lch if lch else lcc
    head = ['<routes>', '<route id="r" edges="' + ' '.join(e for e, _, _ in EDGES) + '"/>']
    body = []; types = {'car': [], 'heavy': [], 'unk': []}
    for k, r in V.iterrows():
        c = 'heavy' if r.truck == 1 else ('car' if r.truck == 0 else 'unk')
        tid = f't{k}'; types[c].append(tid)
        L = 12 if c == 'heavy' else 5
        lane = 1 if r.lane_ins == 1 else 0
        vlim = vi if lane == 1 else vo
        vdes = min(vmax[k], sf[k] * vlim)
        dsp = max(1.0, min(r.v_first, DEP_FAC * vdes)) / 3.6
        t_dep = r.t_ins + 100.0 / max(r.v_first / 3.6, 5.0)
        head.append(f'<vType id="{tid}" vClass="{"truck" if c == "heavy" else "passenger"}" length="{L}" {IDM} speedFactor="{sf[k]:.4f}" speedDev="0" '
                    f'maxSpeed="{vmax[k] / 3.6:.3f}" laneChangeModel="LC2013" {lch if c == "heavy" else lcc}/>')
        body.append(f'<vehicle id="{c[0]}{k}" type="{tid}" route="r" depart="{t_dep:.1f}" departLane="{lane}" departSpeed="{dsp:.3f}"/>')
    rou = os.path.join(WD, f'{tag}.rou.xml')
    with open(rou, 'w') as f:
        f.write('\n'.join(head + body + ['</routes>']))
    return rou, types, float(V.t_ins.max())


def run(day, variant, lc, seed, pre='cfg'):
    vi, vo = Mt.vin[day], Mt.vout[day]
    tag = f'{variant}_{pre}_{day[5:15]}_' + '_'.join(f'{k[2:]}{v}' for k, v in lc.items()) + f'_s{seed}'
    net = write_net(vi, vo, pre)
    rou, types, tend = write_routes(tag, day, variant, lc, seed)
    V0 = R[R.day == day].reset_index(drop=True)
    V0['cls'] = np.where(V0.truck == 1, 'heavy', np.where(V0.truck == 0, 'car', 'unk'))
    V0['vid'] = [f'{c[0]}{k}' for k, c in enumerate(V0.cls)]
    add = os.path.join(WD, f'{tag}.add.xml'); lcx = os.path.join(WD, f'{tag}.lc.xml')
    ld = {c: os.path.join(WD, f'{tag}.ld_{c}.xml') for c in ('car', 'heavy')}
    end = int(tend + 900)
    with open(add, 'w') as f:
        f.write('<additional>')
        for c in ('car', 'heavy'):
            f.write(f'<laneData id="ld_{c}" file="{ld[c]}" begin="0" end="{end}" excludeEmpty="true" vTypes="{" ".join(types[c])}"/>')
        f.write('</additional>')
    p = subprocess.run([os.path.join(SUMO, 'sumo'), '-n', net, '-r', rou, '-a', add, '--lanechange-output', lcx, '--seed', str(seed), '--no-step-log',
                        '--no-warnings', '--begin', '0', '--end', str(end), '--step-length', '0.5', '--collision.action', 'warn', '--max-depart-delay', '60', '--duration-log.statistics'],
                       capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    L = pd.DataFrame([e.attrib for e in ET.parse(lcx).getroot()])
    out = dict(day=day, cfg=Mt.cfg[day], delta=int(Mt.delta[day]), variant=variant, pre=pre, seed=seed, **lc)
    import re as _re
    _m = _re.search(r'Inserted: (\d+)', p.stdout)
    out['n_inserted'] = int(_m.group(1)) if _m else np.nan; out['n_planned'] = len(V0)
    out['n_insert_errors'] = len(_re.findall('will not be able to depart', p.stderr))
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
        s = L[L.id.str.startswith(pfx) & L.edge.isin(COUNT)] if len(L) else pd.DataFrame(columns=['dir', 'olg', 'reason'])
        inn = s[s.dir > 0]; ou = s[s.dir < 0]
        out[f'{c}_n_in'] = len(inn); out[f'{c}_n_out'] = len(ou)
        out[f'{c}_n_in_free'] = int(((inn.olg < 0) | (inn.olg > 300) | inn.olg.isna()).sum()) if len(inn) else 0
        out[f'{c}_in_speedgain'] = int(inn.reason.str.contains('speedGain').sum()) if len(inn) else 0
        out[f'{c}_out_keepright'] = int(ou.reason.str.contains('keepRight').sum()) if len(ou) else 0
        dep0 = set(V0.loc[(V0.cls == c) & (V0.lane_ins == 2), 'vid'])
        out[f'{c}_n_dep_L2'] = len(dep0)
        out[f'{c}_n_dep_L2_in'] = len(set(inn.id) & dep0) if len(inn) else 0
        out[f'{c}_km_L2'] = sum(km.get((c, e, 0), 0.0) for e in COUNT); out[f'{c}_km_L1'] = sum(km.get((c, e, 1), 0.0) for e in COUNT)
        for nm, e in (('entry', ENTRY), ('exit', EXIT)):
            a, b = km.get((c, e, 1), 0.0), km.get((c, e, 0), 0.0)
            out[f'{c}_share_L1_{nm}'] = a / (a + b) if a + b > 0 else np.nan
    for q in (rou, lcx, add, *ld.values()):
        try:
            os.remove(q)
        except OSError:
            pass
    return out


def summarise(df):
    for c in ('car', 'heavy'):
        df[f'{c}_share_km_L1'] = df[f'{c}_km_L1'] / (df[f'{c}_km_L1'] + df[f'{c}_km_L2'])
        df[f'{c}_rate_in'] = df[f'{c}_n_in'] / df[f'{c}_km_L2'].replace(0, np.nan)
        df[f'{c}_rate_out'] = df[f'{c}_n_out'] / df[f'{c}_km_L1'].replace(0, np.nan)
        df[f'{c}_shift_pp'] = 100 * (df[f'{c}_share_L1_exit'] - df[f'{c}_share_L1_entry'])
        df[f'{c}_p_in_by1800'] = df[f'{c}_n_dep_L2_in'] / df[f'{c}_n_dep_L2'].replace(0, np.nan)
    return df


def batch(jobs, outcsv, nw=8):
    rows = []
    with ThreadPoolExecutor(nw) as ex:
        for k, r in enumerate(ex.map(lambda a: run(*a), jobs)):
            rows.append(r)
            if (k + 1) % 50 == 0:
                print(k + 1, '/', len(jobs), flush=True)
                summarise(pd.DataFrame(rows)).to_csv(outcsv, index=False)
    D = summarise(pd.DataFrame(rows)); D.to_csv(outcsv, index=False)
    return D


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'default'
    SEEDS = (1, 2, 3, 4, 5)
    if what == 'test':
        import time; t = time.time()
        print(summarise(pd.DataFrame([run(ORDER8[7], 'REL', {}, 1)])).T); print(time.time() - t, 's')
    elif what == 'default':
        jobs = [(d, v, {}, s, 'cfg') for d in ORDER8 for v in ('REL', 'ABS') for s in SEEDS]
        batch(jobs, os.path.join(DATF, 'sim_runs_default.csv'))
    elif what == 'grid':
        grid = list(itertools.product((1.0, 0.3, 0.1, 0.03, 0.01), (1.0, 0.3, 0.1, 0.03)))
        jobs = [(d, v, {'lcSpeedGain': sg, 'lcKeepRight': kr}, s, 'cfg') for sg, kr in grid if not (sg == 1.0 and kr == 1.0)
                for d in ORDER8 for v in ('REL', 'ABS') for s in SEEDS]
        batch(jobs, os.path.join(DATF, 'sim_runs_grid.csv'))
    elif what == 'classlc':
        sets = []
        for car in ({'lcSpeedGain': 1.0, 'lcKeepRight': 1.0}, {'lcSpeedGain': 1.0, 'lcKeepRight': 0.3}, {'lcSpeedGain': 1.0, 'lcKeepRight': 0.1},
                    {'lcSpeedGain': 0.3, 'lcKeepRight': 0.3}):
            for hsg, hkr in itertools.product((0.1, 0.03, 0.01), (0.3, 1.0)):
                sets.append({**car, 'hvSpeedGain': hsg, 'hvKeepRight': hkr})
        jobs = [(d, 'ABS', lc, s, 'cfg') for lc in sets for d in ORDER8 for s in SEEDS]
        batch(jobs, os.path.join(DATF, 'sim_runs_classlc.csv'))
    elif what == 'sgr':
        jobs = [(d, v, {'lcSpeedGain': sg, 'lcKeepRight': kr, 'lcSpeedGainRight': 1.0}, s, 'cfg') for sg, kr in ((1.0, 1.0), (0.1, 0.1), (0.03, 0.1), (0.1, 0.03))
                for d in ORDER8 for v in ('REL', 'ABS') for s in SEEDS]
        batch(jobs, os.path.join(DATF, 'sim_lodo_runs_sgr.csv'))
