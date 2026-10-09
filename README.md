# Lane changing under lane-differentiated speed limits: code and results

Code and result tables for the paper *How drivers change lanes under lane-differentiated speed limits: A field experiment and a
lane-changing model*.

In the field experiment, one lane-level speed-limit configuration was posted per day on a two-lane freeway for eight days:
three days with the same limit in both lanes and five days with a higher limit in the inner lane. The scripts produce the
figures, tables and numbers of the paper from the vehicle trajectories.

## Data

The trajectory data are not part of this repository; see the data availability statement of the paper. The scripts read
them from `data/trajectories/`. Tables with one row per vehicle are written when the scripts run and are not included.

## Contents

| Folder | Content |
|---|---|
| `code/` | Python scripts; `run_all.py` runs them in order |
| `results/` | Result tables (CSV) in `results/` and `results/final/`, permutation draws in `results/permutation/`, Figures 3–7 in `results/figures/` |

## Requirements

- Python 3.10 with the packages in `requirements.txt`.
- The simulation runs also need SUMO 1.18 with TraCI (`SUMO_HOME` set).

## Running the code

```
cd code
python run_all.py          # about 10 minutes; intermediate files are written to _cache/
python run_all.py --all    # also reruns the permutation draws and the SUMO runs (several hours)
```

Without `--all`, the permutation tests and the simulation check use the draws and runs stored in `results/`.

The scripts form five stages:

1. Data preparation: `prepare_core.py`, `situations_*.py`, `stints_build.py` and `lateral_check.py` give the situation of every
   vehicle, the lane changes by situation and the stays in the inner lane; the `check_*.py` scripts recompute these quantities
   independently.
2. Day-level analyses: `table1_data.py` and `table1.py` (Table 1), `rates_data.py` (lane-change rates and day-level tests),
   `lane_share_data.py`, `lane_sequences.py` and `free_move_probability.py` (inner-lane share), `stays_checks.py`,
   `overtaking_cycle.py`, `stays.py` and `heavy_lane_use.py` (stays in the inner lane), and `checks_*.py` (further checks).
3. Lane-changing model: `decisions_build.py` (decision points every second), `mobil_model.py`, `interval_censored.py` and
   `model_lib.py` (stochastic MOBIL), the `model_*.py` scripts (fits, leave-one-day-out predictions, exact tests, calibration),
   `model_versions.py` and `versions_*.py` (model versions), `idm_sensitivity.py` and `table4.py` (Table 4).
4. Simulation check: `model_inputs.py` and the `simulation*.py` scripts.
5. Figures: `fig3_fig4_day_rates.py`, `fig5_lane_share.py`, `fig6_model_contrast.py` and `fig7_simulation_check.py`.

## Where the results of the paper are

Result files are in `results/final/` unless another folder is given.

| Paper | Script (`code/`) | Result files |
|---|---|---|
| Figure 3 | `fig3_fig4_day_rates.py` (data: `rates_data.py`) | `fig3_rates_by_distance_data.csv`, `fig3_rates_by_distance_days.csv` |
| Figure 4 | `fig3_fig4_day_rates.py` (data: `rates_data.py`) | `fig4_day_rates_data.csv`, `fig4_day_rates_tests.csv` |
| Figure 5 | `fig5_lane_share.py` (data: `lane_share_data.py`) | `fig5_lane_share_change_data.csv`, `fig5_group_means.csv`, `fig5_tests.csv` |
| Figure 6 | `fig6_model_contrast.py` (model fits: `model_*.py`) | `fig6_data.csv`, `fig6_days.csv`, `fig6_interval.csv` |
| Figure 7 | `fig7_simulation_check.py` (runs: `simulation_closed_loop.py`, `simulation_closed_loop_m0.py`) | `fig7_summary.csv`, `fig7_data.csv` |
| Table 1 | `table1.py` | `Table1_days.csv` |
| Table 2 | `situations_rates.py` | `results/sit_typology_counts.csv` |
| Table 4 | `table4.py` | `Table4.csv`, `table4_daylevel.csv` |
| Table 5 | `model_lib.py`, `simulation_closed_loop.py` | desired speeds: `model_desired_speed_fits.csv` |
| Section 5.1 | `rates_data.py`, `stays_checks.py` | `fig3_rates_by_distance_*.csv`, `rates_heavy_leader_days.csv`, `stays_checks.csv` |
| Section 5.2 | `rates_data.py`, `free_move_probability.py`, `checks_day_level.py`, `checks_order.py`, `checks_contrast.py` | `fig4_day_rates_tests.csv`, `rates_by_type_tests.csv`, `rates_robustness_data.csv`, `checks_order_check.csv`, `checks_order_adjusted.csv`, `contrast_tests.csv`, `lane_free_probability_data.csv` |
| Section 5.3 | `lane_share_data.py`, `lane_sequences.py`, `overtaking_cycle.py`, `stays.py`, `heavy_lane_use.py` | `fig5_tests.csv`, `fig5_gain_decomposition.csv`, `lane_sequences_tests.csv`, `overtaking_cycle_stats.csv`, `stays_rate_ratios.csv`, `heavy_lane_use_summary.csv`, `results/check_misc.csv` |
| Section 5.4 | `model_*.py`, `table4.py`, `checks_no_desired_speed.py`, `model_versions.py`, `versions_*.py`, `idm_sensitivity.py`, `calibration_check.py` | `Table4.csv`, `fig6_*.csv`, `model_verify_models_causal.csv`, `model_bias_checks.csv`, `model_perm_free_tests.csv`, `model_rec_summary.csv`, `checks_novdes_summary.csv`, `versions_bias_no_desired_speed.csv`, `model_versions.csv`, `idm_sensitivity.csv`, `calibration_check_OE.csv` |
| Section 5.5 | `fig7_simulation_check.py`, `simulation_mechanism.py` | `fig7_summary.csv`, `sim_mechanism_check.csv` |

## Licence

The code is released under the MIT licence (`LICENSE`).
