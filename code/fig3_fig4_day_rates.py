# -*- coding: utf-8 -*-
"""Figures 3 and 4 (numbers from rates_data.py; results/final is read only).
  Figure_3   inward lane-change rate of cars in the outer lane by the distance to the vehicle ahead: day rates, means of the three
             same-limit and five differentiated days, per-bin D/S ratio and exact day-level p (fig3_rates_by_distance_*.csv).
  Figure_4   unconstrained and leader-driven inward rates of cars on each day with 95 % intervals, group means, D/S ratio and exact
             day-level p (fig4_day_rates_*.csv).
Outputs: results/figures/Figure_3.png, Figure_4.png (300 dpi) and .pdf; results/final/fig3_plotted.csv (plotted points).
usage: python fig3_fig4_day_rates.py [03 04]"""
import os, sys
from plot_style import *          # noqa
plt = setup()
plt.rcParams['savefig.bbox'] = 'standard'          # fixed page size: every G2 figure is exactly W15 = 140 mm wide
import matplotlib.ticker as mt
import matplotlib.transforms as mtrans
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerTuple

SUFFIX = ''
rd = lambda f: pd.read_csv(os.path.join(DATF, f))
META = DAY_META.set_index('day')
CFG = META.cfg.to_dict(); DEL = META.delta.to_dict()
GREY_TXT = '#555555'
FS = 6.8                                             # annotation font size (pt)


def logfmt(ax, axis='y'):
    f = mt.FuncFormatter(lambda v, p: f'{v:g}')
    (ax.yaxis if axis == 'y' else ax.xaxis).set_major_formatter(f)
    (ax.yaxis if axis == 'y' else ax.xaxis).set_minor_formatter(mt.NullFormatter())


def pstr(p):
    return f'{p:.3f}' if p < 0.1 else f'{p:.2f}'


def pstr_hu(p):
    """as pstr, but halves are rounded up (an exact p = k/56 can end in 5, e.g. 0.625 -> 0.63)"""
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal('0.001') if p < 0.1 else Decimal('0.01')
    return str(Decimal(repr(float(p))).quantize(q, rounding=ROUND_HALF_UP))


def fig_mm(h_mm, w=W15):
    return plt.figure(figsize=(w, h_mm * MM))


def axes_mm(fig, l, b, r, t):
    """axes placed by margins in mm (left, bottom, right, top)"""
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([l / W, b / H, (W - l - r) / W, (H - b - t) / H])


def swarm_x(ax, cx, y, s_area, pad=1.12):
    """deterministic beeswarm: x positions (data units) around cx so that discs of area s_area (pt^2) at y do not overlap"""
    y = np.asarray(y, float)
    if len(y) == 0:
        return np.array([])
    T = ax.transData
    P = T.transform(np.c_[np.full(len(y), cx), y])
    d = np.sqrt(s_area) * ax.figure.dpi / 72 * pad
    placed, xd = [], np.zeros(len(y))
    for i in np.argsort(P[:, 1]):
        for k in range(60):
            cands = [0.0] if k == 0 else [k * d * 0.5, -k * d * 0.5]
            ok = None
            for c in cands:
                xc = P[i, 0] + c
                if all((xc - px) ** 2 + (P[i, 1] - py) ** 2 >= d ** 2 * 0.999 for px, py in placed):
                    ok = xc; break
            if ok is not None:
                break
        placed.append((ok, P[i, 1])); xd[i] = ok
    return T.inverted().transform(np.c_[xd, P[:, 1]])[:, 0]


def close(fig, name):
    out = name + SUFFIX
    path = save(fig, out); plt.close(fig); print('saved', path)


# ============================================================================================================ fig3
def fig03():
    G = rd('fig3_rates_by_distance_data.csv'); DD = rd('fig3_rates_by_distance_days.csv')
    LAB = ['<50', '50-100', '100-150', '150-300', 'none within 300']
    TICK = ['< 50', '50–100', '100–150', '150–300', 'Unconstrained:\nnone within 300 m']
    xs = np.array([0, 1, 2, 3, 4.6]); XB = 3.8                            # XB = x-axis break before the categorical unconstrained class
    Y0 = 0.02                                                               # row for day rates of exactly zero
    fig = fig_mm(84)                                                        # 4 mm taller for the per-bin D/S row
    ax = axes_mm(fig, 21, 21, 2, 17)
    ax.set_yscale('log'); ax.set_ylim(0.0155, 9); ax.set_xlim(-0.55, 5.15)
    ax.axvspan(4.08, 5.15, color='#F2F2F2', lw=0, zorder=0)
    OFF = 0.2; S_DAY = 11
    specs = (('same', C_SAME, -OFF, ORDER8[:3]), ('diff', C_DIFF, OFF, ORDER8[3:]))
    plotted = []
    for g, col, off, days in specs:
        z = G[G.group == g].set_index('bin')
        for i, b in enumerate(LAB):
            cx = xs[i] + off
            m = z.loc[b, 'mean_day_rate']
            ax.plot([cx - 0.14, cx + 0.14], [m, m], color=col, lw=1.8, solid_capstyle='butt', zorder=3)
            dd = DD[DD.bin == b].set_index('day')
            dl = list(days)
            yv = np.array([dd.loc[d, 'rate'] if dd.loc[d, 'rate'] > 0 else Y0 for d in dl])
            xv = swarm_x(ax, cx, yv, S_DAY)
            for d, x_, y_ in zip(dl, xv, yv):
                ax.scatter(x_, y_, s=S_DAY, color=col, edgecolors='white', linewidths=0.4, zorder=4)
                plotted.append(dict(bin=b, group=g, day=d, cfg=CFG[d], rate=dd.loc[d, 'rate'], n=int(dd.loc[d, 'n']), km=dd.loc[d, 'km'],
                                    y_plotted=y_, x_plotted=x_))
            plotted.append(dict(bin=b, group=g, day='mean of day rates', rate=m, n=int(z.loc[b, 'n']), km=z.loc[b, 'km']))
    pd.DataFrame(plotted).to_csv(os.path.join(DATF, 'fig3_plotted.csv'), index=False)
    ax.set_yticks([Y0, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]); ax.set_yticklabels(['0', '0.05', '0.1', '0.2', '0.5', '1', '2', '5'])
    ax.yaxis.set_minor_locator(mt.NullLocator())
    # y-axis break between the zero row and the log scale
    yb = 0.0255
    ax.add_patch(Rectangle((-0.02, yb / 1.07), 0.04, yb * 1.07 - yb / 1.07, transform=ax.get_yaxis_transform(), color='white', clip_on=False, zorder=6))
    for dy in (1 / 1.07, 1.07):
        ax.plot([-0.012, 0.012], [yb * dy / 1.03, yb * dy * 1.03], transform=ax.get_yaxis_transform(), color='black', lw=0.6, clip_on=False, zorder=7)
    # x-axis break before the categorical unconstrained class
    ax.add_patch(Rectangle((XB - 0.06, -0.02), 0.12, 0.04, transform=ax.get_xaxis_transform(), color='white', clip_on=False, zorder=6))
    for dx in (-0.06, 0.06):
        ax.plot([XB + dx - 0.03, XB + dx + 0.03], [-0.018, 0.018], transform=ax.get_xaxis_transform(), color='black', lw=0.6, clip_on=False, zorder=7)
    ax.set_xticks(xs); ax.set_xticklabels(TICK)
    ax.tick_params(axis='x', length=0, pad=3)
    ax.set_xlabel('Distance to the outer-lane leader at t − 3 s (m)', labelpad=2)
    ax.xaxis.set_label_coords(np.mean(xs[:4]), -0.115, transform=mtrans.blended_transform_factory(ax.transData, ax.transAxes))
    ax.set_ylabel('Inward changes per car-km\n(cars in the outer lane)')
    # rows above the axes: changes per group; per-bin D/S ratio of the means of day rates with the exact day-level p (8 days);
    # then the day-level test for the pooled leader bins
    tr = ax.get_xaxis_transform()
    zs = G[G.group == 'same'].set_index('bin'); zd = G[G.group == 'diff'].set_index('bin')
    for i, b in enumerate(LAB):
        ax.text(xs[i] - OFF, 1.02, f'{int(zs.n[b])}', transform=tr, ha='center', va='bottom', fontsize=FS, color=GREY_TXT)
        ax.text(xs[i] + OFF, 1.02, f'{int(zd.n[b])}', transform=tr, ha='center', va='bottom', fontsize=FS, color=C_DIFF)
        ax.text(xs[i], 1.10, f'×{zd.ratio_daymeans[b]:.2f}\np = {pstr_hu(zd.p2_8[b])}', transform=tr, ha='center', va='bottom', fontsize=FS,
                color='#222222', linespacing=1.2, fontweight='bold' if b == 'none within 300' else 'normal')
    ax.text(-0.62, 1.02, 'Changes, n', transform=tr, ha='right', va='bottom', fontsize=FS, color=GREY_TXT)
    ax.text(-0.62, 1.10, 'D/S ratio\np (8 days)', transform=tr, ha='right', va='bottom', fontsize=FS, color=GREY_TXT, linespacing=1.2)
    pl = zd.loc['leader within 300 (pooled)']
    ax.annotate('', xy=(-0.35, 1.245), xytext=(3.35, 1.245), xycoords=tr, textcoords=tr,
                arrowprops=dict(arrowstyle='-', lw=0.6, color='#777777', shrinkA=0, shrinkB=0))
    ax.text(1.5, 1.26, f'Leader within 300 m, all bins pooled: D/S ×{pl.ratio_daymeans:.2f}, p = {pstr(pl.p2_8)}', transform=tr,
            ha='center', va='bottom', fontsize=FS, color='#222222')
    # legend
    h = [Line2D([], [], color=C_SAME, lw=1.8), Line2D([], [], color=C_DIFF, lw=1.8),
         (Line2D([], [], marker='o', ls='', color=C_SAME, mec='white', mew=0.4, ms=3.3),
          Line2D([], [], marker='o', ls='', color=C_DIFF, mec='white', mew=0.4, ms=3.3))]
    lab = ['Mean of day rates, same-limit days (3)', 'Mean of day rates, differentiated days (5)', 'Single day (same-limit / differentiated)']
    fig.legend(h, lab, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)}, loc='lower center', bbox_to_anchor=(0.53, 0.0), ncol=2,
               fontsize=FS, handletextpad=0.5, columnspacing=1.4, borderaxespad=0.3, handlelength=1.8)
    close(fig, 'Figure_3')


# ============================================================================================================ fig4
PX_MM = 25.4 / 300          # one pixel of the 300 dpi PNG, in mm
# height of one legend row (2.886 mm), rounded to a whole number of pixels (34 px = 2.879 mm) so that the parts above and below keep
# their sub-pixel position on the 300 dpi grid; it is subtracted from the figure height and the top margin (legend of two rows)
LEG_ROW_MM = 34 * PX_MM


def fig04():
    F = rd('fig4_day_rates_data.csv'); T = rd('fig4_day_rates_tests.csv')
    order = list(ORDER8)
    xpos = dict(zip(order, [0, 1, 2, 3.4, 4.4, 5.4, 6.4, 7.4]))
    XSEP_, XR_ = 2.7, 8.9                                                  # group separator; right x limit (room for the D/S text as before)
    XLAB = -0.55 - 0.2 * (XR_ + 0.55) / (10.05 + 0.55)                     # row headers at the same distance (mm) left of the axis as before
    fig = fig_mm(92 - LEG_ROW_MM)
    ax = axes_mm(fig, 23, 22, 2, 12 - LEG_ROW_MM)                           # same axes size as before
    OFF = 0.15
    for d in order:
        col = C_DIFF if DEL[d] > 0 else C_SAME
        for reg, off, filled in (('free', -OFF, True), ('leader-driven', OFF, False)):
            r = F[(F.day == d) & (F.regime == reg)].iloc[0]
            ax.errorbar(xpos[d] + off, r.rate, yerr=[[r.rate - r.ci_lo], [r.ci_hi - r.rate]], fmt='o', ms=4.4,
                        color=col, mfc=col if filled else 'white', mec=col, mew=0.9, elinewidth=0.8, capsize=0, zorder=3)
    t = T.set_index(['test', 'days'])
    f8 = t.loc[('free inward rate', 8)]; l8 = t.loc[('leader-driven inward rate', 8)]
    means = {}
    for reg, ls, lw in (('free', '-', 1.3), ('leader-driven', (0, (3, 1.6)), 1.0)):
        for g, days in (('same', ORDER8[:3]), ('diff', ORDER8[3:])):
            m = F[(F.regime == reg) & F.day.isin(days)].rate.mean(); means[(reg, g)] = m
            x0 = xpos[days[0]] - 0.38; x1 = xpos[days[-1]] + 0.38
            ax.plot([x0, x1], [m, m], color='#222222', lw=lw, ls=ls, zorder=2, alpha=0.9)
    # primary day-level test, written at the right end of the differentiated group-mean lines
    for reg, r8 in (('free', f8), ('leader-driven', l8)):
        ax.text(xpos[ORDER8[-1]] + 0.43, means[(reg, 'diff')], f'D/S ×{r8.ratio:.2f}\np = {pstr(r8.p2)}', ha='left', va='center', fontsize=FS,
                color='#222222', linespacing=1.15, fontweight='bold' if reg == 'free' else 'normal')
    ax.axvline(XSEP_, color='#BBBBBB', lw=0.6, zorder=0)
    ax.set_yscale('log')
    logfmt(ax); ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5]); ax.set_ylim(0.025, 8)
    ax.set_xlim(-0.55, XR_)
    ax.set_xticks([xpos[d] for d in order])
    ax.set_xticklabels([CFG[d] for d in order])
    for lab_, d in zip(ax.get_xticklabels(), order):
        lab_.set_color(C_DIFF if DEL[d] > 0 else '#444444')
    ax.tick_params(axis='x', length=0, pad=3)
    ax.set_ylabel('Inward changes per car-km\ndriven in that state')
    for x_, s_, c_ in (((xpos[ORDER8[0]] + xpos[ORDER8[2]]) / 2, 'Same limit', '#444444'), (xpos[ORDER8[5]], 'Differentiated', C_DIFF)):
        ax.text(x_, 5.6, s_, ha='center', va='center', fontsize=7.2, color=c_)
    # rows under the axis: lane differential and counts
    tr = ax.get_xaxis_transform()
    rows = [('Δ (km/h)', lambda d: f'{DEL[d]}'),
            ('n unconstrained', lambda d: f"{int(F[(F.day == d) & (F.regime == 'free')].n.iloc[0])}"),
            ('n leader-driven', lambda d: f"{int(F[(F.day == d) & (F.regime == 'leader-driven')].n.iloc[0])}")]
    for k, (nm, fn) in enumerate(rows):
        yy = -0.155 - 0.075 * k
        for d in order:
            ax.text(xpos[d], yy, fn(d), transform=tr, ha='center', va='top', fontsize=FS, color='#222222')
        ax.text(XLAB, yy, nm, transform=tr, ha='right', va='top', fontsize=FS, color=GREY_TXT)
    ax.text(XLAB, -0.022, 'Inner/outer\nlimit (km/h)', transform=tr, ha='right', va='top', fontsize=FS, color=GREY_TXT, linespacing=1.1)
    # legend (4 entries, 2 x 2)
    h = [Line2D([], [], marker='o', color='#444444', mfc='#444444', mec='#444444', ms=4.2, lw=0.8),
         Line2D([], [], marker='o', color='#444444', mfc='white', mec='#444444', mew=0.9, ms=4.2, lw=0.8),
         Line2D([], [], color='#222222', lw=1.3), Line2D([], [], color='#222222', lw=1.0, ls=(0, (3, 1.6)))]
    lab = ['Unconstrained (no outer-lane vehicle within 300 m)', 'Leader-driven (headway < 4 s, or gap < 150 m and closing)',
           'Group mean, unconstrained', 'Group mean, leader-driven']
    fig.legend(h, lab, loc='upper left', bbox_to_anchor=(14 * MM / W15, 1.0), ncol=2, fontsize=FS, handletextpad=0.5, borderaxespad=0.3,
               labelspacing=0.25, columnspacing=2.0, handlelength=1.8)
    close(fig, 'Figure_4')


if __name__ == '__main__':
    which = [a for a in sys.argv[1:] if not a.startswith('--')] or ['03', '04']
    for w in which:
        globals()['fig' + w]()
