# -*- coding: utf-8 -*-
"""Figure 5: change in the inner-lane share of cars along the section relative to each day's share at 1,000-1,100 m (single days
coloured by Delta, means of the three same-limit and five differentiated days, exact day-level test of the gain).
Input: _cache/work/lane_share_data.pkl (lane_share_data.py); the test is recomputed as there (numbers in results/final/fig5_*.csv).
Output: results/figures/Figure_5.png (300 dpi, 90 mm wide) and .pdf."""
import os, sys
from plot_style import *   # noqa: F401,F403
from matplotlib.lines import Line2D

NAME = 'Figure_5'
CP = os.path.join(CACHE, 'lane_share_data.pkl')

if __name__ == '__main__':
    if not os.path.exists(CP):
        raise SystemExit(f'cache {CP} missing: run lane_share_data.py first')
    P = pd.read_pickle(CP)
    base = P[P.x_lo == 1000].set_index('day').share_inner
    P['change_pp'] = 100 * (P.share_inner - P.day.map(base))

    # ---- day-level test on the entry-to-exit gain (1900-2000 m minus 1000-1100 m), eight days, as in lane_share_data.py
    gain = P[P.x_lo == 1900].set_index('day').change_pp
    g8 = [gain[d] for d in ORDER8]
    obs, p2, p1 = perm8(g8)
    t = pd.Series(dict(same_mean=np.mean(g8[:3]), diff_mean=np.mean(g8[3:]), p2_8=p2))
    # per-bin group means (unweighted over days)
    BINS = np.arange(1000, 2001, 100)
    G = []
    for b in BINS[:-1]:
        z = P[P.x_lo == b].set_index('day')
        c8 = [z.change_pp[d] for d in ORDER8]
        G.append(dict(x_lo=b, x_mid=b + 50, same_change_pp=np.mean(c8[:3]), diff_change_pp=np.mean(c8[3:])))
    G = pd.DataFrame(G)
    print(t.round(4).to_string())

    # ------------------------------------------------------------------------------------------------ figure
    from matplotlib.legend_handler import HandlerTuple
    from lane_share_util import save_exact
    plt = setup()
    DL = 'Δ'

    fig, ax = plt.subplots(figsize=(W1, 71 * MM))
    for d in ORDER8:
        z = P[P.day == d].sort_values('x_mid')
        ax.plot(z.x_mid, z['change_pp'].values, color=day_color(d), lw=0.75, zorder=2)
    ax.plot(G.x_mid, G.same_change_pp, color=C_SAME, lw=2.0, marker='o', ms=3.2, zorder=4, solid_capstyle='round')
    ax.plot(G.x_mid, G.diff_change_pp, color=C_DIFF, lw=2.0, marker='o', ms=3.2, zorder=4, solid_capstyle='round')
    gantry_lines(ax, y_text=1.0)
    ax.set_xlim(1000, 2000); ax.set_ylim(-2.7, 11.0); ax.set_yticks(np.arange(-2, 10.1, 2))
    ax.set_xticks(np.arange(1000, 2001, 200)); ax.set_xticklabels([f'{v:,}' for v in range(1000, 2001, 200)])
    ax.set_xlabel('Position along the section, x (m)')
    ax.set_ylabel('Change in inner-lane share of cars relative\nto 1,000–1,100 m (percentage points)')
    # annotation
    ax.axhline(0, color='#BDBDBD', lw=0.5, zorder=1)
    for v, c in ((t.diff_mean, C_DIFF), (t.same_mean, C_SAME)):
        ax.text(1968, v, f'{v:+.1f} pp', ha='left', va='center', fontsize=6.5, color=c, fontweight='bold', clip_on=False)
    ax.text(1925, -0.7, f'Exact day-level test of the gain:\n'
            f'{fmt_p(t.p2_8)} (8 days)', ha='right', va='top', fontsize=6.5, linespacing=1.3, color='#222222')
    # legend: three entries in one column
    dl = tuple(Line2D([], [], color=DELTA_COL[k], lw=1.4) for k in (0, 10, 20, 30))
    H = [Line2D([], [], color=C_DIFF, lw=2.0, marker='o', ms=3.2), Line2D([], [], color=C_SAME, lw=2.0, marker='o', ms=3.2), dl]
    L = ['Differentiated, mean of 5 days', 'Same limit, mean of 3 days', f'Single days, {DL} = 0 / 10 / 20 / 30 km/h']
    ax.legend(H, L, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)}, loc='upper left', bbox_to_anchor=(-0.2, -0.17), ncol=1,
              fontsize=6.5, handlelength=2.4, columnspacing=0.6, handletextpad=0.5, borderaxespad=0.0, labelspacing=0.35)
    save_exact(fig, NAME, 90); plt.close(fig)
