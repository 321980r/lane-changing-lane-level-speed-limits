# -*- coding: utf-8 -*-
"""Closed-loop run without lane-limit terms (M0 in every lane-change sub-model), otherwise as simulation_closed_loop.py (imported):
absolute desired speeds; for a given day and seed the desired-speed draws and SUMO's seed equal those of the M2F runs, and the lane-change
draws use the stream of the non-M2F specifications. Seeds 1-5 x 8 days = 40 runs; resumable. Requires SUMO 1.18 with SUMO_HOME set.
Slow; run_all.py runs it only with --all.
usage: python simulation_closed_loop_m0.py test | run [nworkers] -> results/final/sim_closedloop_runs_M0.csv (same columns as
sim_closedloop_runs.csv)"""
import os, sys, time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import simulation_closed_loop as CL

CL.SPECS['M0'] = {'types': 'ABS', ('car', 'in'): ('M0', 'M0'), ('car', 'out'): ('M0', 'M0'),
                  ('truck', 'in'): ('M0', 'M0'), ('truck', 'out'): ('M0', 'M0')}
OUT = os.path.join(CL.DATF, 'sim_closedloop_runs_M0.csv')


def _job(a):
    return CL.run_cl(*a)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'test'
    if what == 'test':
        # determinism check: re-run one existing M2F run and compare with sim_closedloop_runs.csv, then one M0 run
        day = CL.ORDER8[4]
        t0 = time.time()
        r = CL.summarise(pd.DataFrame([CL.run_cl(day, 'M2F', 1)]))
        old = pd.read_csv(os.path.join(CL.DATF, 'sim_closedloop_runs.csv'))
        o = old[(old.day == day) & (old.variant == 'CL_M2F') & (old.seed == 1)].iloc[0]
        for c in ('n_trig', 'n_inserted', 'car_n_in', 'car_n_out', 'car_n_in_free', 'car_shift_pp', 'heavy_n_in'):
            print('M2F rerun', c, r[c].iloc[0], 'stored', o[c])
        print(round(time.time() - t0, 1), 's'); t0 = time.time()
        r = CL.summarise(pd.DataFrame([CL.run_cl(day, 'M0', 1)]))
        print(r[['day', 'variant', 'seed', 'n_trig', 'n_inserted', 'n_planned', 'car_n_in', 'car_n_out', 'car_n_in_free', 'car_shift_pp']].T.to_string())
        print(round(time.time() - t0, 1), 's')
    else:
        nw = int(sys.argv[2]) if len(sys.argv) > 2 else 6
        done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=['day', 'variant', 'seed'])
        key = set(zip(done.day, done.variant, done.seed))
        jobs = [(d, 'M0', s) for d in CL.ORDER8 for s in (1, 2, 3, 4, 5) if (d, 'CL_M0', s) not in key]
        print(len(jobs), 'jobs', flush=True)
        rows = [done] if len(done) else []
        t0 = time.time()
        with ProcessPoolExecutor(nw) as ex:
            for k, r in enumerate(ex.map(_job, jobs)):
                rows.append(pd.DataFrame([r]))
                print(k + 1, '/', len(jobs), r['day'][5:15], r['variant'], r['seed'], round(time.time() - t0, 1), 's', flush=True)
                CL.summarise(pd.concat(rows, ignore_index=True)).to_csv(OUT, index=False)
