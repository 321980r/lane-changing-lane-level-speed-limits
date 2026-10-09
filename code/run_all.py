# -*- coding: utf-8 -*-
"""Run the whole analysis from the data set: every script of code/ in dependency order, with the arguments used for the files in results/.
Each step runs in its own Python process with code/ as working directory; its printed output goes to _cache/logs/<step>.log and one line
per step is printed here. The run stops at the first error and shows the end of that step's log.

The slow steps are skipped by default; their outputs are part of results/ and are used instead:
  permutation draws  decisions_permutation.py, model_permutation.py, model_permutation_resume.py          -> results/permutation/*.csv
  SUMO simulations   simulation.py, simulation_closed_loop.py, simulation_closed_loop_m0.py,
                     simulation_class_lc.py                                    -> results/final/sim_runs_*.csv, sim_closedloop_runs.csv,
                                                                                      sim_closedloop_runs_M0.csv
With --all they run as well (several hours; the SUMO steps need SUMO 1.18 with SUMO_HOME set). The resumable steps (model_permutation_resume.py,
simulation_closed_loop.py, simulation_closed_loop_m0.py, simulation_class_lc.py) only add the draws or runs that are missing from their output
files; delete those files first to recompute them from scratch.
usage: python run_all.py [--all]
"""
import os, sys, time, subprocess

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
LOGS = os.path.join(ROOT, '_cache', 'logs')

# data -> core tables -> situations and rates -> checks -> tables -> decision points -> model fits
PART_A = ['prepare_core.py', 'situations_state.py', 'situations_rates.py', 'situations_car_probability.py', 'lateral_check.py', 'stints_build.py', 'check_situations.py', 'check_rates.py',
          'check_lane_shift.py', 'check_lateral.py', 'check_returns.py', 'check_sequences.py', 'check_misc.py', 'check_post_pass.py', 'check_summary.py',
          'model_reproduction.py', 'table1_data.py', 'table1.py', 'rates_data.py', 'lane_sequences.py',
          'lane_share_data.py', 'free_move_probability.py', 'stays_checks.py', 'overtaking_cycle.py', 'decisions_build.py', 'model_inputs.py',
          'model_verify.py orig', 'model_verify.py causal', 'model_verify.py lag3', 'model_lodo.py', 'model_recent_change.py',
          'model_bias_checks.py', 'model_permutation_bias.py car in', 'model_permutation_bias.py car out', 'model_permutation_bias.py truck in',
          'model_permutation_bias.py tests']
# slow: permutation draws (read by model_permutation_tests.py)
PERM = ['decisions_permutation.py 500 0', 'decisions_permutation.py 500 1', 'decisions_permutation.py 500 2', 'decisions_permutation.py 500 3', 'model_permutation.py 0 0 causal',
        'model_permutation.py 225 1 causal', 'model_permutation_resume.py 300 2', 'model_permutation_resume.py 500 3']
PART_B = ['model_permutation_tests.py', 'model_calibration_data.py', 'simulation_lodo_summary.py']
# slow: SUMO runs; the class-specific runs need the grid loss of a first simulation_summary.py pass
SUMO = ['simulation.py default', 'simulation.py grid', 'simulation_closed_loop.py run', 'simulation_closed_loop_m0.py run',
        'simulation_summary.py', 'simulation_class_lc.py']
# simulation summaries -> parameter tables -> further checks -> Table 3 -> figures
PART_C = ['simulation_summary.py', 'simulation_mechanism.py', 'model_parameters.py', 'model_figure_helpers.py', 'checks_day_level.py',
          'checks_order.py', 'checks_no_desired_speed.py', 'checks_contrast.py', 'heavy_lane_use.py', 'stays.py',
          'table4.py', 'versions_bias_no_desired_speed.py', 'model_versions.py', 'idm_sensitivity.py', 'calibration_check.py',
          'fig3_fig4_day_rates.py', 'fig5_lane_share.py',
          'fig6_model_contrast.py', 'fig7_simulation_check.py']
STEPS = [(s, False) for s in PART_A] + [(s, True) for s in PERM] + [(s, False) for s in PART_B] + [(s, True) for s in SUMO] +\
        [(s, False) for s in PART_C]

# outputs of the slow steps that the default run reads from results/
SLOW_OUTPUTS = ['permutation/dec_perm_draws_c0.csv', 'permutation/dec_perm_draws_c1.csv', 'permutation/dec_perm_draws_c2.csv',
                'permutation/dec_perm_draws_c3.csv', 'permutation/model_perm_causal_c0.csv', 'permutation/model_perm_causal_c1.csv',
                'permutation/model_perm_causal_r2.csv', 'permutation/model_perm_causal_r3.csv', 'final/sim_runs_default.csv',
                'final/sim_runs_grid.csv', 'final/sim_runs_classlc2.csv', 'final/sim_closedloop_runs.csv',
                'final/sim_closedloop_runs_M0.csv']


def main():
    run_slow = '--all' in sys.argv[1:]
    if not run_slow:
        missing = [p for p in SLOW_OUTPUTS if not os.path.exists(os.path.join(ROOT, 'results', p))]
        if missing:
            sys.exit('outputs of the slow steps missing in results/: ' + ', '.join(missing) + ' (run with --all)')
    os.makedirs(LOGS, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING='utf-8', MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
    t_all = time.time()
    n = len(STEPS)
    for i, (step, slow) in enumerate(STEPS, 1):
        if slow and not run_slow:
            print(f'{i:2d}/{n}  {step:40s} skipped (slow step; output in results/, run with --all)', flush=True)
            continue
        args = step.split()
        log = os.path.join(LOGS, '_'.join([args[0][:-3]] + args[1:]) + '.log')
        t0 = time.time()
        with open(log, 'w', encoding='utf-8') as f:
            rc = subprocess.run([sys.executable] + args, cwd=CODE, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        if rc != 0:
            print(f'{i:2d}/{n}  {step:40s} FAILED (exit code {rc}); end of {log}:', flush=True)
            tail = open(log, encoding='utf-8', errors='replace').read().splitlines()[-15:]
            print('\n'.join('      ' + s for s in tail))
            sys.exit(1)
        print(f'{i:2d}/{n}  {step:40s} ok {time.time() - t0:7.1f} s', flush=True)
    print(f'done in {(time.time() - t_all) / 60:.1f} min; logs in {os.path.relpath(LOGS, ROOT)}')


if __name__ == '__main__':
    main()
