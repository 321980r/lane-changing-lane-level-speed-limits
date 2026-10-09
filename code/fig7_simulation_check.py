# -*- coding: utf-8 -*-
"""Figure 7 (closed-loop simulation check): change in the inner-lane share of cars between 1,000-1,100 m and 1,900-2,000 m on each of
the eight days, observed (95 % interval) and simulated with M0, M1p and M2F (mean and range over 5 seeds), with a legend table of the
group means, D - S, MAE and the outward-rate ratio. Days: same-limit at x = 0-2, separator at 2.7, differentiated at 3.4-7.4 (x limits
-0.55 to 7.95). Plot helpers from model_figure_helpers.py.
Inputs: results/final/sim_observed_by_day_ci.csv, sim_closedloop_runs.csv, sim_closedloop_runs_M0.csv, model_rec_lodo_day.csv.
Outputs: results/figures/Figure_7.png/.pdf; results/final/fig7_summary.csv (legend-table numbers), fig7_data.csv (plotted
values)."""
import os, sys
_argv = sys.argv; sys.argv = [_argv[0], 'none']
import model_figure_helpers as G6                   # helpers only (what='none' skips both figure branches)
sys.argv = _argv
from model_figure_helpers import (plt, np, pd, DATF, FIGF, RES, MM, W15, Mt, MAIN, SAME3, ORDER8, C_OBS, C_DIFF, GREY_TXT, FS,
                                 fig_mm, axes_mm, legend_table, seedline, obspoint, perm8, save)

NAME = 'Figure_7'

C_M0, C_M1P, C_M2F = '#999999', '#0072B2', '#7B3294'
ORD = list(ORDER8)                                  # eight days, same-limit (3) | differentiated (5, by Delta)
XP = dict(zip(ORD, [0, 1, 2, 3.4, 4.4, 5.4, 6.4, 7.4]))
XSEP, XMAX = 2.7, 7.95
XLAB = -0.55 - 0.2 * (XMAX + 0.55) / (9.1 + 0.55)  # x of the row headers left of the axis

O = pd.read_csv(os.path.join(DATF, 'sim_observed_by_day_ci.csv')).set_index('day').reindex(ORD)
C0 = pd.read_csv(os.path.join(DATF, 'sim_closedloop_runs.csv'))
CM = pd.read_csv(os.path.join(DATF, 'sim_closedloop_runs_M0.csv'))
assert set(CM.variant) == {'CL_M0'} and (CM.groupby('day').size() == 5).all() and len(CM) == 40
C = pd.concat([C0, CM], ignore_index=True)
C = C[C.day.isin(ORD)]
assert (C.n_inserted / C.n_planned).min() > 0.98
SIM = {}
for v, g in C.groupby('variant'):
    a = g.groupby('day')[['car_shift_pp', 'car_rate_in', 'car_rate_out']].agg(['mean', 'min', 'max'])
    a.columns = [f'{x}_{y}' for x, y in a.columns]
    SIM[v] = a.reindex(ORD)

SER = [('CL_M0', 'M0: no lane-limit term', C_M0, 'o', 3.3, -0.40),
       ('CL_M1p', 'M1p: desired speed = speed factor × lane limit', C_M1P, 's', 3.4, -0.21),
       ('CL_M2F', 'M2F (recommended): bias of unconstrained cars', C_M2F, 'v', 4.0, 0.22)]


def grp(v):
    v = np.asarray(v, float)
    return v[SAME3].mean(), v[~SAME3].mean()


def day_axis(ax, rows, y_hdr, row0=-0.155, drow=0.072, xmax=XMAX):
    """model_figure_helpers.day_axis with the x positions of this figure"""
    ax.axvline(XSEP, color='#BBBBBB', lw=0.6, zorder=0)
    ax.set_xlim(-0.55, xmax)
    ax.set_xticks([XP[d] for d in ORD])
    ax.set_xticklabels([Mt.cfg[d] for d in ORD])
    for lab_, d in zip(ax.get_xticklabels(), ORD):
        lab_.set_color(C_DIFF if Mt.delta[d] > 0 else '#444444')
    ax.tick_params(axis='x', length=0, pad=3)
    for x_, s_, c_ in (((XP[ORD[0]] + XP[ORD[2]]) / 2, 'Same limit', '#444444'), (XP[ORD[5]], 'Differentiated', C_DIFF)):
        ax.text(x_, y_hdr, s_, ha='center', va='center', fontsize=7.2, color=c_)
    tr = ax.get_xaxis_transform()
    ax.text(XLAB, -0.022, 'Inner/outer\nlimit (km/h)', transform=tr, ha='right', va='top', fontsize=FS, color=GREY_TXT, linespacing=1.1)
    for k, (nm, vals) in enumerate(rows):
        yy = row0 - drow * k
        for d in ORD:
            ax.text(XP[d], yy, vals[d], transform=tr, ha='center', va='top', fontsize=FS, color='#222222')
        ax.text(XLAB, yy, nm, transform=tr, ha='right', va='top', fontsize=FS, color=GREY_TXT)


# ---------------------------------------------------------------------------------------------------------------- summaries (8 days)
obs_pp = O.reindex(MAIN).car_shift_pp.values
om = grp(obs_pp)
_, p8, _ = perm8(obs_pp)
Rl = pd.read_csv(os.path.join(DATF, 'model_rec_lodo_day.csv'))
pv = Rl[Rl.day.isin(MAIN)].pivot(index='day', columns='state', values='n_obs').reindex(MAIN)
summ = [dict(series='observed', same_pp=om[0], diff_pp=om[1], diff_minus_same_pp=om[1] - om[0], p_two_8days=p8,
             free_share_same=pv.free.values[SAME3].sum() / pv['all'].values[SAME3].sum(),
             free_share_diff=pv.free.values[~SAME3].sum() / pv['all'].values[~SAME3].sum(),
             out_rate_same=O.reindex(MAIN).car_rate_out.values[SAME3].mean(), out_rate_diff=O.reindex(MAIN).car_rate_out.values[~SAME3].mean(),
             in_rate_same=O.reindex(MAIN).car_rate_in.values[SAME3].mean(), in_rate_diff=O.reindex(MAIN).car_rate_in.values[~SAME3].mean())]
for key, *_ in SER:
    v = SIM[key].reindex(MAIN).car_shift_pp_mean.values
    g_ = grp(v)
    mae = float(np.mean(np.abs(v - obs_pp)))
    ms_ = []
    for sd, gs in C[C.variant == key].groupby('seed'):
        vv = gs.set_index('day').reindex(MAIN)
        ms_.append(np.mean(np.abs(vv.car_shift_pp.values - obs_pp)))
    g = C[(C.variant == key) & C.day.isin(MAIN)].groupby('day')[['car_n_in', 'car_n_in_free', 'car_n_out', 'car_km_L1', 'car_km_L2']].mean().reindex(MAIN)
    summ.append(dict(series=key, same_pp=g_[0], diff_pp=g_[1], diff_minus_same_pp=g_[1] - g_[0], mae_pp_8days=mae,
                     mae_pp_seed_min=min(ms_), mae_pp_seed_max=max(ms_),
                     free_share_same=g.car_n_in_free.values[SAME3].sum() / g.car_n_in.values[SAME3].sum(),
                     free_share_diff=g.car_n_in_free.values[~SAME3].sum() / g.car_n_in.values[~SAME3].sum(),
                     out_rate_same=(g.car_n_out / g.car_km_L1).values[SAME3].mean(), out_rate_diff=(g.car_n_out / g.car_km_L1).values[~SAME3].mean(),
                     in_rate_same=(g.car_n_in / g.car_km_L2).values[SAME3].mean(), in_rate_diff=(g.car_n_in / g.car_km_L2).values[~SAME3].mean()))
S = pd.DataFrame(summ)
S['out_rate_ratio'] = S.out_rate_diff / S.out_rate_same; S['in_rate_ratio'] = S.in_rate_diff / S.in_rate_same
S.to_csv(os.path.join(DATF, 'fig7_summary.csv'), index=False)
pd.set_option('display.width', 250)
print(S.round(3).T.to_string())

# ---------------------------------------------------------------------------------------------------------------- figure
fig = fig_mm(99)
ax = axes_mm(fig, 23, 16.5, 2, 28.5)
ax.axhline(0, color='#BDBDBD', lw=0.6, zorder=0)
for key, lab, col, mk, ms, dx in SER:
    s = SIM[key]
    for d in ORD:
        seedline(ax, XP[d] + dx, s.car_shift_pp_mean[d], s.car_shift_pp_min[d], s.car_shift_pp_max[d], col, mk, ms=ms)
for d in ORD:
    obspoint(ax, XP[d], O.car_shift_pp[d], O.car_shift_pp_lo[d], O.car_shift_pp_hi[d], d)
for days, v in ((ORDER8[:3], om[0]), (ORDER8[3:], om[1])):
    ax.plot([XP[days[0]] - 0.48, XP[days[-1]] + 0.42], [v, v], color=C_OBS, lw=0.7, ls=(0, (4, 2)), zorder=1)
lo_all = min(SIM[k].car_shift_pp_min.min() for k, *_ in SER)
hi_all = max(SIM[k].car_shift_pp_max.max() for k, *_ in SER)
print('seed range over all series/days:', round(lo_all, 2), round(hi_all, 2))
ylo = -10 if lo_all > -9.5 else 5 * np.floor((lo_all - 0.5) / 5)
ax.set_ylim(ylo, 22.5)
ax.set_yticks(np.arange(ylo, 22.6, 5))
ax.set_ylabel('Change in inner-lane share of cars,\n1,000–1,100 m to 1,900–2,000 m (pp)')
day_axis(ax, [('Δ (km/h)', {d: str(int(Mt.delta[d])) for d in ORD}),
              ('n cars', {d: str(int(O.car_shift_n_cars[d])) for d in ORD})], y_hdr=20.8, row0=-0.16, drow=0.078)
fmt = lambda x: f'{x:+.1f}'.replace('-', '−')
ob = S[S.series == 'observed'].iloc[0]
rows = [['Observed (95 % CI); dashed lines = group means', fmt(ob.same_pp), fmt(ob.diff_pp), fmt(ob.diff_minus_same_pp), '–', f'{ob.out_rate_ratio:.2f}']]
mks = [('o', C_OBS, True, True)]
Si = S.set_index('series')
for key, lab, col, mk, ms, dx in SER:
    r = Si.loc[key]
    rows.append([lab, fmt(r.same_pp), fmt(r.diff_pp), fmt(r.diff_minus_same_pp), f'{r.mae_pp_8days:.1f}', f'{r.out_rate_ratio:.2f}'])
    mks.append((mk, col, True, True))
legend_table(fig, 0.012, 0.992, ['Closed-loop replay: mean of 5 seeds (bar = seed range)', 'Same\nlimit', 'Differ-\nentiated', 'D − S', 'MAE',
                                 'Outward\nrate D/S'], rows, [0.648, 0.738, 0.812, 0.878, 0.985], mks, lh_mm=3.6, hdr_lines=2,
             span=('Inner-lane share change (pp), eight days', 0.585, 0.884))
save(fig, NAME)
plt.close(fig)
print('saved', os.path.join(FIGF, NAME + '.png'))

D = pd.DataFrame({'day': ORD, 'cfg': [Mt.cfg[d] for d in ORD], 'delta': [int(Mt.delta[d]) for d in ORD],
                  'observed_pp': O.car_shift_pp.values, 'observed_ci_lo': O.car_shift_pp_lo.values, 'observed_ci_hi': O.car_shift_pp_hi.values,
                  'observed_n_cars': O.car_shift_n_cars.values})
for key, *_ in SER:
    for q in ('mean', 'min', 'max'):
        D[f'{key}_pp_{q}' if q == 'mean' else f'{key}_pp_seed_{q}'] = SIM[key][f'car_shift_pp_{q}'].values
    D[f'{key}_rate_in_mean'] = SIM[key]['car_rate_in_mean'].values; D[f'{key}_rate_out_mean'] = SIM[key]['car_rate_out_mean'].values
D.to_csv(os.path.join(DATF, 'fig7_data.csv'), index=False)
print('Figure 7 ok')
