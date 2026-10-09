# -*- coding: utf-8 -*-
"""SUMO LC2013 runs with class-specific settings for heavy vehicles (lcSpeedGain in {1, 0.3, 0.1, 0.03, 0.01} x lcKeepRight in {1, 0.3})
on top of the best single car setting (first row of results/final/sim_grid_loss.csv); 8 days x 5 seeds; resumable (runs already in
the output are skipped). Requires SUMO 1.18 with SUMO_HOME set. Slow; run_all.py runs it only with --all.
usage: python simulation_class_lc.py [nthreads] -> results/final/sim_runs_classlc2.csv"""
import itertools, time
from concurrent.futures import ThreadPoolExecutor
import simulation as S
from plot_style import *

nw = int(sys.argv[1]) if len(sys.argv) > 1 else 6
GL = pd.read_csv(os.path.join(DATF, 'sim_grid_loss.csv'))
b = GL.iloc[0]
car = {'lcSpeedGain': float(b.lcSpeedGain), 'lcKeepRight': float(b.lcKeepRight)}
sets = [{**car, 'hvSpeedGain': hs, 'hvKeepRight': hk} for hs, hk in itertools.product((1.0, 0.3, 0.1, 0.03, 0.01), (1.0, 0.3))]
jobs = [(d, b.variant, lc, s, 'cfg') for lc in sets for d in ORDER8 for s in (1, 2, 3, 4, 5)]
out = os.path.join(DATF, 'sim_runs_classlc2.csv')
K = ['lcSpeedGain', 'lcKeepRight', 'hvSpeedGain', 'hvKeepRight']
done = set()
if os.path.exists(out):
    O = pd.read_csv(out)
    done = set(zip(O.day, O.variant, O.seed, *[O[k] for k in K]))
todo = [j for j in jobs if (j[0], j[1], j[3], *[j[2][k] for k in K]) not in done]
print('car set', b.variant, car, len(jobs), 'jobs,', len(todo), 'to run', flush=True)
rows = []; t0 = time.time()


def flush():
    global rows
    if rows:
        new = S.summarise(pd.DataFrame(rows))
        if os.path.exists(out):
            new = pd.concat([pd.read_csv(out), new], ignore_index=True)
        new.to_csv(out, index=False); rows = []


with ThreadPoolExecutor(nw) as ex:
    for k, r in enumerate(ex.map(lambda a: S.run(*a), todo)):
        rows.append(r)
        if (k + 1) % 20 == 0:
            flush(); print(k + 1, '/', len(todo), round(time.time() - t0), 's', flush=True)
flush(); print('finished classlc2')
