# -*- coding: utf-8 -*-
"""Plot helpers for fig7_simulation_check.py (figure and axes in mm, day axis, legend table, seed bars, observed points), and the day
rates of the car-inward model versions: observed and leave-one-day-out predictions of M0, M1, M1p, M2F and M1pF per day, with the
differentiated/same ratios in the unconstrained and leader situations (model_contrast_tests.csv, read by
fig6_model_contrast.py).
Inputs: results/final/model_rec_lodo_day.csv, model_lodo_day.csv, model_rec_ratios.csv.
Outputs: results/final/model_contrast_data.csv, model_contrast_tests.csv; working figure
_cache/working_figures/model_contrast.png/.pdf.
usage: python model_figure_helpers.py [contrast|all|none]   (none: helpers only)"""
from plot_style import *
import matplotlib.ticker as mt
from matplotlib.lines import Line2D

plt = setup()
plt.rcParams['savefig.bbox'] = 'standard'          # exact 140 mm page width; layouts set by hand
Mt = DAY_META.set_index('day')
ORD = list(ORDER8)                                 # same-limit (3) | differentiated (5, by Delta)
XP = dict(zip(ORD, [0, 1, 2, 3.4, 4.4, 5.4, 6.4, 7.4]))
XSEP = 2.7
MAIN = ORDER8
SAME3 = np.array([Mt.delta[d] == 0 for d in MAIN])
C_OBS, C_M0, C_M1P, C_M2F = '#1A1A1A', '#9A9A9A', C_MODEL['M1p'], '#7B3294'
GREY_TXT = '#555555'
FS = 6.8
what = sys.argv[1] if len(sys.argv) > 1 else 'all'


def fig_mm(h_mm, w=W15):
    return plt.figure(figsize=(w, h_mm * MM))


def axes_mm(fig, l, b, r, t):
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([l / W, b / H, (W - l - r) / W, (H - b - t) / H])


def day_axis(ax, rows, y_hdr, row0=-0.155, drow=0.072, xmax=7.95):
    """x axis as in fig4: configuration tick labels (differentiated days in blue), group headers inside the axes at y_hdr (data units),
    and rows of per-day numbers under the axis. rows = [(label, {day: text})]"""
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
    ax.text(-0.75, -0.022, 'Inner/outer\nlimit (km/h)', transform=tr, ha='right', va='top', fontsize=FS, color=GREY_TXT, linespacing=1.1)
    for k, (nm, vals) in enumerate(rows):
        yy = row0 - drow * k
        for d in ORD:
            ax.text(XP[d], yy, vals[d], transform=tr, ha='center', va='top', fontsize=FS, color='#222222')
        ax.text(-0.75, yy, nm, transform=tr, ha='right', va='top', fontsize=FS, color=GREY_TXT)


def legend_table(fig, x0, y_top, header, rows, colx, markers, lh_mm=3.7, hdr_lines=1, span=None):
    """compact legend in figure coordinates: one row per series, marker glyph + label, right-aligned numbers.
    header: column headers (first = label column); colx: right edges (figure fraction) of the number columns;
    markers: (marker, colour, filled, with_range) or ('dash', colour) or None; span: (text, x_left, x_right) header over the number columns"""
    W, H = fig.get_size_inches() / MM
    lh = lh_mm / H
    if span is not None:
        t, a, b = span
        fig.text((a + b) / 2, y_top, t, ha='center', va='top', fontsize=6.5, color=GREY_TXT)
        fig.add_artist(Line2D([a, b], [y_top - lh * 0.92] * 2, color='#C8C8C8', lw=0.5, transform=fig.transFigure))
        y_top = y_top - lh * 1.1
    for j, h in enumerate(header):
        if not h:
            continue
        x = x0 + 0.024 if j == 0 else colx[j - 1]
        fig.text(x, y_top, h, ha='left' if j == 0 else 'right', va='top', fontsize=6.5, color=GREY_TXT, linespacing=1.1)
    y_rule = y_top - lh * (0.95 + 0.9 * (hdr_lines - 1))
    fig.add_artist(Line2D([x0, colx[-1] + 0.004], [y_rule] * 2, color='#B0B0B0', lw=0.5, transform=fig.transFigure))
    for i, r in enumerate(rows):
        yy = y_rule - lh * (i + 0.2)
        mk = markers[i]
        if mk is not None:
            yc = yy - 0.45 * lh * 0.78
            if mk[0] == 'dash':
                fig.add_artist(Line2D([x0 + 0.0, x0 + 0.014], [yc] * 2, color=mk[1], lw=0.8, ls=(0, (3, 1.6)), transform=fig.transFigure))
            else:
                m_, col, filled, rng_ = mk
                if rng_:
                    fig.add_artist(Line2D([x0 + 0.007, x0 + 0.007], [yc - 0.36 * lh, yc + 0.36 * lh], color=col, lw=0.8, transform=fig.transFigure))
                fig.add_artist(Line2D([x0 + 0.007], [yc], marker=m_, ms=3.9, color=col, mfc=(col if filled else 'white'), mec=('white' if (filled and not rng_ and m_ != 'o') else col),
                                      mew=(0.3 if (filled and not rng_ and m_ != 'o') else 0.8), ls='none', transform=fig.transFigure))
        for j, v in enumerate(r):
            x = x0 + 0.024 if j == 0 else colx[j - 1]
            fig.text(x, yy, v, ha='left' if j == 0 else 'right', va='top', fontsize=FS, color='#222222')


def seedline(ax, x, y, lo, hi, col, mk, ms=3.8, fill=True, zorder=3):
    ax.plot([x, x], [lo, hi], color=col, lw=0.8, zorder=zorder - 1, solid_capstyle='butt')
    ax.plot([x], [y], marker=mk, ms=ms, mfc=(col if fill else 'white'), mec=('white' if fill else col), mew=(0.35 if fill else 0.8), ls='none', zorder=zorder)


def obspoint(ax, x, y, lo, hi, d):
    ax.plot([x, x], [lo, hi], color=C_OBS, lw=0.9, zorder=4, solid_capstyle='butt')
    ax.plot([x], [y], marker='o', ms=4.4, mfc=C_OBS, mec=C_OBS, mew=0.9, ls='none', zorder=5)


# ============================================================================================================== model contrast
if what in ('contrast', 'all'):
    R = pd.read_csv(os.path.join(DATF, 'model_rec_lodo_day.csv'))
    L = pd.read_csv(os.path.join(DATF, 'model_lodo_day.csv'))
    RT = pd.read_csv(os.path.join(DATF, 'model_rec_ratios.csv'))
    f = R[R.state == 'free'].set_index('day').reindex(ORD)
    ci = L[(L.cls == 'car') & (L.dirn == 'in') & (L.state == 'free')].set_index('day').reindex(ORD)
    assert np.allclose(f.rate_obs.values, ci.rate_obs.values)
    rt = lambda st, s: RT[(RT.state == st) & (RT.series == s)].iloc[0]
    # recompute the group means / ratios from the day table as a check on model_rec_ratios.csv
    for s in ('rate_obs', 'rate_pred_M0', 'rate_pred_M1', 'rate_pred_M1p', 'rate_pred_M2F'):
        v = f.reindex(MAIN)[s].values
        assert abs(v[~SAME3].mean() / v[SAME3].mean() - rt('free', s)['ratio']) < 1e-9, s
    SER = [('rate_pred_M0', C_M0, 'o', 3.3, -0.25), ('rate_pred_M1p', C_M1P, 's', 3.3, 0.22), ('rate_pred_M2F', C_M2F, 'v', 3.8, 0.38)]
    fig = fig_mm(92.5)
    ax = axes_mm(fig, 23, 18.5, 2, 28)
    for c, col, mk, ms, dx in SER:
        for d in ORD:
            ax.plot(XP[d] + dx, f[c][d], marker=mk, ms=ms, mfc=col, mec='white', mew=0.3, ls='none', zorder=3)
    for d in ORD:
        obspoint(ax, XP[d], f.rate_obs[d], ci.rate_obs_lo[d], ci.rate_obs_hi[d], d)
    o = rt('free', 'rate_obs')
    for days, v in ((ORDER8[:3], o['same']), (ORDER8[3:], o['diff'])):
        ax.plot([XP[days[0]] - 0.42, XP[days[-1]] + 0.48], [v, v], color=C_OBS, lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.set_ylim(0, 0.70)
    ax.yaxis.set_major_locator(mt.MultipleLocator(0.1))
    ax.set_ylabel('Inward changes per car-km, cars\nwith no leader within 300 m')
    day_axis(ax, [('Δ (km/h)', {d: str(int(Mt.delta[d])) for d in ORD}),
                  ('n changes', {d: str(int(f.n_obs[d])) for d in ORD}),
                  ('car-km', {d: f'{f.km[d]:.0f}' for d in ORD})], y_hdr=0.665)
    rows = [['Observed (95% CI)', f'×{o["ratio"]:.2f}  (p = {o["p_two"]:.3f})', f'×{rt("leader", "rate_obs")["ratio"]:.2f}  (p = {rt("leader", "rate_obs")["p_two"]:.2f})'],
            ['M0: lane limits not used', f'×{rt("free", "rate_pred_M0")["ratio"]:.2f}', f'×{rt("leader", "rate_pred_M0")["ratio"]:.2f}'],
            ['M1p: lane desired speed = speed factor × lane limit', f'×{rt("free", "rate_pred_M1p")["ratio"]:.2f}', f'×{rt("leader", "rate_pred_M1p")["ratio"]:.2f}'],
            ['M2F (recommended): M0 + inner-lane bias of unconstrained cars, differentiated days', f'×{rt("free", "rate_pred_M2F")["ratio"]:.2f}',
             f'×{rt("leader", "rate_pred_M2F")["ratio"]:.2f}'],
            [f'Observed group means: {o["same"]:.2f} (same limit), {o["diff"]:.2f} (differentiated) per car-km', '', '']]
    mks = [('o', C_OBS, True, True), ('o', C_M0, True, False), ('s', C_M1P, True, False), ('v', C_M2F, True, False), ('dash', C_OBS)]
    legend_table(fig, 0.012, 0.992, ['Observed / leave-one-day-out prediction', 'No leader within 300 m', 'Leader within 300 m'], rows,
                 [0.80, 0.985], mks, lh_mm=3.55, span=('Differentiated / same limit, ratio of day means', 0.60, 0.989))
    save(fig, 'model_contrast', FIGW)
    plt.close(fig)
    out = pd.DataFrame({'day': ORD, 'cfg': [Mt.cfg[d] for d in ORD], 'delta': [int(Mt.delta[d]) for d in ORD], 'n_changes_free': f.n_obs.values,
                        'car_km_free': f.km.values, 'n_veh_free': ci.n_veh.values, 'obs': f.rate_obs.values, 'obs_ci_lo': ci.rate_obs_lo.values,
                        'obs_ci_hi': ci.rate_obs_hi.values})
    for m in ('M0', 'M1', 'M1p', 'M2F', 'M1pF'):
        out[f'pred_{m}'] = f[f'rate_pred_{m}'].values
    lead = R[R.state == 'leader'].set_index('day').reindex(ORD)
    out['n_changes_leader'] = lead.n_obs.values; out['car_km_leader'] = lead.km.values; out['obs_leader'] = lead.rate_obs.values
    for m in ('M0', 'M1', 'M1p', 'M2F', 'M1pF'):
        out[f'pred_leader_{m}'] = lead[f'rate_pred_{m}'].values
    out['plotted'] = 'obs, pred_M0, pred_M1p, pred_M2F (free state); ratios in the legend from model_contrast_tests.csv'
    out.to_csv(os.path.join(DATF, 'model_contrast_data.csv'), index=False)
    RT.to_csv(os.path.join(DATF, 'model_contrast_tests.csv'), index=False)
    print('model contrast ok; M1 ratios free / leader:', round(rt('free', 'rate_pred_M1')['ratio'], 3), round(rt('leader', 'rate_pred_M1')['ratio'], 3))

print('ok', what)
