# -*- coding: utf-8 -*-
"""Share of cars unconstrained in the outer lane at x = 1,050 m that made an unconstrained inward change before 1,500 m, per day, with
exact binomial intervals and day-level tests (Section 5.2).
Input: results/sit_car_probability.csv (situations_car_probability.py).
Outputs: results/final/lane_free_probability_data.csv; working figure _cache/working_figures/lane_free_probability.png/.pdf."""
import os
from plot_style import *
from scipy.stats import beta

R = pd.read_csv(os.path.join(RES, 'sit_car_probability.csv')).set_index('day')
k = R.moved_free_before_1500.astype(int); n = R.cars_free_lane2_at_1050.astype(int)
R['lo'] = [0.0 if kk == 0 else beta.ppf(0.025, kk, nn - kk + 1) for kk, nn in zip(k, n)]
R['hi'] = [1.0 if kk == nn else beta.ppf(0.975, kk + 1, nn - kk) for kk, nn in zip(k, n)]
v8 = [R.prob[d] for d in ORDER8]
obs, p2, p1 = perm8(v8); p2r = perm_rank8(v8)
same, diff = np.mean(v8[:3]), np.mean(v8[3:])
print(f'same {same:.3f} diff {diff:.3f} ratio {diff / same:.2f} p2 {p2:.3f} p1 {p1:.3f} rank {p2r:.3f}')
order = ORDER8
R['group'] = np.where(R.delta > 0, 'differentiated', 'same-limit')
out = R.loc[order].reset_index()
out['same_mean8'] = same; out['diff_mean8'] = diff; out['p2_8'] = p2; out['p1_8'] = p1; out['p2_rank8'] = p2r
out.to_csv(os.path.join(DATF, 'lane_free_probability_data.csv'), index=False)

plt = setup()       # working figure (not in the paper)
fig, ax = plt.subplots(figsize=(W1 - 7 * MM, 74 * MM))
ypos = {d: i for i, d in enumerate(order[:3])}
ypos.update({d: 4.0 + i for i, d in enumerate(order[3:])})          # row gap for the group header
for d in order:
    y = ypos[d]; r = R.loc[d]; c = day_color(d)
    ax.plot([r.lo, r.hi], [y, y], color=c, lw=0.9, zorder=2, solid_capstyle='butt')
    ax.plot(r.prob, y, 'o', ms=4.6, color=c, mec='white', mew=0.5, zorder=3)
# group means (unweighted over days): vertical segments across the group rows
ys = [ypos[d] for d in ORDER8[:3]]; yd = [ypos[d] for d in ORDER8[3:]]
ax.plot([same, same], [min(ys) - 0.4, max(ys) + 0.4], color=C_SAME, lw=1.6, zorder=1, solid_capstyle='butt')
ax.plot([diff, diff], [min(yd) - 0.4, max(yd) + 0.4], color=C_DIFF, lw=1.6, zorder=1, solid_capstyle='butt')
ax.text(same, min(ys) - 0.5, f'mean {same:.3f}', ha='center', va='bottom', fontsize=6.5, color=C_SAME)
ax.text(diff, min(yd) - 0.5, f'mean {diff:.3f}', ha='center', va='bottom', fontsize=6.5, color=C_DIFF)
ax.set_yticks([ypos[d] for d in order])
lab = []
for d in order:
    r = R.loc[d]; cfg = DAY_META.set_index('day').cfg[d]
    lab.append(f'{cfg}  ({int(r.moved_free_before_1500)}/{int(r.cars_free_lane2_at_1050)})')
ax.set_yticklabels(lab, fontsize=6.5)
ax.tick_params(axis='y', length=0, pad=3)
ax.set_ylim(max(ypos.values()) + 0.7, -1.9)
ax.set_xlim(0, 0.40); ax.set_xticks(np.arange(0, 0.41, 0.1))
ax.set_xlabel('Probability of an unconstrained inward change\nat 1,050–1,500 m')
ax.spines['left'].set_visible(False)
ax.text(-0.02, -1.35, 'Same limit', transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=6.8, fontweight='bold', color='#444444')
ax.text(-0.02, 2.95, 'Differentiated', transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=6.8, fontweight='bold', color='#444444')
ax.text(0.99, 0.985, f'Exact day-level test\n8 days: {fmt_p(p2)}\n\n'
        f'Horizontal lines: exact binomial 95 % CI\nVertical lines: mean of days\nLabels: moved / at risk at 1,050 m', transform=ax.transAxes,
        ha='right', va='top', fontsize=6.5, linespacing=1.3)
from lane_share_util import save_exact
save_exact(fig, 'lane_free_probability', 90, folder=FIGW)
