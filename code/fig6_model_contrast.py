# -*- coding: utf-8 -*-
"""Figure 6: within-day contrast implied by each version of the car-inward model, and observed (single panel, 90 mm).
contrast = (D/S factor at unconstrained decision points) / (D/S factor at decision points with a leader within 300 m); factor = mean of the
five differentiated-day rates / mean of the three same-limit-day rates, each day predicted by the fit to the other seven days.
Unconstrained = no outer-lane vehicle within 300 m ahead at the 1-s decision point (x = 1,000-1,800 m). Versions: M0 (no limit term),
M1 (desired speed capped at the lane limit), M1p (speed factor x lane limit), M2F (bias of unconstrained cars on differentiated days),
M2F-Delta (bias linear in Delta), M1pF (M1p + bias). Observed contrast at the same decision points with a day-level 95 % interval from
inverting the exact 56-labelling test on the day log contrasts.
Inputs: results/final/model_rec_lodo_day.csv, model_lodo_day.csv, model_verify_models_causal.csv and model_bias_checks.csv (leave-one-day-out
log-likelihoods), model_contrast_tests.csv (check).
Outputs: results/figures/Figure_6.png/.pdf; results/final/fig6_data.csv (one row per version), fig6_days.csv (day values),
fig6_interval.csv (observed contrast, tests and intervals)."""
from plot_style import *
import matplotlib.ticker as mt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
import contrast_lib as L

plt = setup()
SCR_OUT = DATF                                       # data behind the figure
Mt = DAY_META.set_index('day')
MAIN = ORDER8                                   # 3 same-limit days first
R = pd.read_csv(os.path.join(DATF, 'model_rec_lodo_day.csv'))
G = pd.read_csv(os.path.join(DATF, 'model_lodo_day.csv'))
G = G[(G.cls == 'car') & (G.dirn == 'in')]
T9 = pd.read_csv(os.path.join(DATF, 'model_contrast_tests.csv'))
VM = pd.read_csv(os.path.join(DATF, 'model_verify_models_causal.csv'))
FB = pd.read_csv(os.path.join(DATF, 'model_bias_checks.csv')).set_index('model')

st = {s: R[R.state == s].set_index('day').reindex(MAIN) for s in ('free', 'leader')}
gl = {s: G[G.state == s].set_index('day').reindex(MAIN) for s in ('free', 'leader')}
for s in ('free', 'leader'):                    # same decision points in both files
    assert np.allclose(st[s].km.values, gl[s].km.values) and (st[s].n_obs.values == gl[s].n_obs.values).all()
    assert np.allclose(st[s].rate_pred_M2F.values, gl[s].rate_pred_M2F.values)
    st[s]['rate_pred_M2Fl'] = gl[s].rate_pred_M2Fl.values

LODO = {'M0': VM[(VM.group == 'car_in') & (VM.model == 'M0')].LODO.iloc[0], 'M1': VM[(VM.group == 'car_in') & (VM.model == 'M1')].LODO.iloc[0],
        'M1p': VM[(VM.group == 'car_in') & (VM.model == 'M1p')].LODO.iloc[0], 'M2F': FB.LODO['M0+free+free:diff'],
        'M2Fl': FB.LODO['M0+free+free:d10'], 'M1pF': FB.LODO['M1p+free+free:diff']}

# series: key, label, colour, marker, group
SER = [('obs', 'Observed', '#1A1A1A', 'o', 'obs'),
       ('M0', 'M0', '#999999', 'o', 'none'),
       ('M1', 'M1', '#CC79A7', 'D', 'H1'), ('M1p', 'M1p', '#0072B2', 's', 'H1'),
       ('M2F', 'M2F', '#7B3294', 'v', 'H2'), ('M2Fl', 'M2F-Δ', '#B48FD0', 'v', 'H2'),
       ('M1pF', 'M1pF', '#56B4E9', 's', 'both')]
same = np.array([Mt.delta[d] == 0 for d in MAIN])
rows, drows = [], []
for key, lab, col, mk, grp in SER:
    c = 'rate_obs' if key == 'obs' else f'rate_pred_{key}'
    f, l = st['free'][c].values, st['leader'][c].values
    ff, fl = f[~same].mean() / f[same].mean(), l[~same].mean() / l[same].mean()
    lc = np.log(f / l)
    geo = float(np.exp(lc[~same].mean() - lc[same].mean()))
    rows.append(dict(series=key, label=lab, group=grp, factor_unconstrained=ff, factor_leader300=fl, contrast=ff / fl, contrast_geo=geo,
                     factor_unc_rel_M0=np.nan, factor_lead_rel_M0=np.nan, lodo_ll=LODO.get(key, np.nan),
                     rate_unc_same=f[same].mean(), rate_unc_diff=f[~same].mean(), rate_lead_same=l[same].mean(), rate_lead_diff=l[~same].mean()))
    for d, a, b in zip(MAIN, f, l):
        drows.append(dict(series=key, day=d, cfg=Mt.cfg[d], delta=int(Mt.delta[d]), rate_unconstrained=a, rate_leader300=b, log_contrast=np.log(a / b)))
D = pd.DataFrame(rows).set_index('series')
for k in D.index:
    if k not in ('obs', 'M0'):
        D.loc[k, 'factor_unc_rel_M0'] = D.factor_unconstrained[k] / D.factor_unconstrained['M0']
        D.loc[k, 'factor_lead_rel_M0'] = D.factor_leader300[k] / D.factor_leader300['M0']
# check against the model contrast table
for k in ('obs', 'M0', 'M1', 'M1p', 'M2F', 'M1pF'):
    s = 'rate_obs' if k == 'obs' else f'rate_pred_{k}'
    a = T9[(T9.state == 'free') & (T9.series == s)].ratio.iloc[0]; b = T9[(T9.state == 'leader') & (T9.series == s)].ratio.iloc[0]
    assert abs(a - D.factor_unconstrained[k]) < 1e-9 and abs(b - D.factor_leader300[k]) < 1e-9, k

# observed interval: exact test inversion on the day log contrasts (8 days)
lc8 = np.log(st['free'].rate_obs.values / st['leader'].rate_obs.values)
p2, p1, _ = L.perm_generic(lc8, 3); lo, hi = L.ratio_ci(lc8, 3, log=True)
e2, lo2, hi2, p1b, p2b = L.ratio_of_ratios_ci(st['free'].rate_obs.values, st['leader'].rate_obs.values, 3)
OBS = pd.DataFrame([
    dict(item='observed contrast, ratio of factors (2.420/1.033), 8 days', value=D.contrast['obs'], ci_lo=np.nan, ci_hi=np.nan, p_two=np.nan, p_one=np.nan),
    dict(item='observed contrast, geometric mean of day log contrasts, 8 days (test-inverted 95 % interval; exact 56-labelling p)',
         value=float(np.exp(lc8[3:].mean() - lc8[:3].mean())), ci_lo=lo, ci_hi=hi, p_two=p2, p_one=p1),
    dict(item='sensitivity: inversion of the ratio-of-factors statistic itself, 8 days', value=e2, ci_lo=lo2, ci_hi=hi2, p_two=p2b, p_one=p1b),
    dict(item='same-limit day contrasts (min, max)', value=float(np.exp(lc8[:3].min())), ci_lo=float(np.exp(lc8[:3].max())), ci_hi=np.nan, p_two=np.nan, p_one=np.nan),
    dict(item='differentiated day contrasts (min, max)', value=float(np.exp(lc8[3:].min())), ci_lo=float(np.exp(lc8[3:].max())), ci_hi=np.nan, p_two=np.nan, p_one=np.nan)])
D['inside_observed_interval'] = [int(lo <= v <= hi) if k != 'obs' else np.nan for k, v in D.contrast.items()]
D.reset_index().to_csv(os.path.join(SCR_OUT, 'fig6_data.csv'), index=False)
pd.DataFrame(drows).to_csv(os.path.join(SCR_OUT, 'fig6_days.csv'), index=False)
OBS.to_csv(os.path.join(SCR_OUT, 'fig6_interval.csv'), index=False)
pd.set_option('display.width', 250)
print(D.round(3).to_string()); print(OBS.round(4).to_string())

# ------------------------------------------------------------------------------------------------------------------ figure
FS = 6.8; GREY_TXT = '#555555'
XP = {'obs': 0.0, 'M0': 1.35, 'M1': 2.45, 'M1p': 3.35, 'M2F': 4.45, 'M2Fl': 5.35, 'M1pF': 6.45}
plt.rcParams['savefig.bbox'] = 'standard'          # exact 90 mm width; margins set by hand
Wm, Hm = 90.0, 70.0
fig = plt.figure(figsize=(W1, Hm * MM))
ax = fig.add_axes([16.5 / Wm, 11.0 / Hm, (Wm - 16.5 - 1.0) / Wm, (Hm - 11.0 - 1.5) / Hm])
ax.set_yscale('log')
ax.set_ylim(0.4, 9.0)
ax.set_xlim(-0.6, 7.2)
ax.axhspan(lo, hi, color='#E4E4E4', lw=0, zorder=0)
ax.axhline(D.contrast['obs'], color='#1A1A1A', lw=0.8, zorder=1)
ax.axhline(1.0, color='#9E9E9E', lw=0.6, ls=(0, (1.5, 1.5)), zorder=1)
ax.axvline(0.68, color='#BBBBBB', lw=0.6, zorder=0)
ax.text(7.15, hi * 1.04, f'Observed contrast {D.contrast["obs"]:.2f}; band = day-level 95 % interval {lo:.2f}–{hi:.2f}', ha='right', va='bottom',
        fontsize=6.5, color='#333333', bbox=dict(facecolor='white', edgecolor='none', pad=0.4), zorder=5)   # the box hides the separator
DX = 0.24
for key, lab, col, mk, grp in SER:
    x = XP[key]; r = D.loc[key]
    # factor pair: open = unconstrained, filled = leader within 300 m; the vertical distance on the log axis is the contrast
    ax.plot([x + DX, x + DX], [r.factor_unconstrained, r.factor_leader300], color=col, lw=0.6, zorder=2)
    ax.plot([x + DX], [r.factor_unconstrained], marker='o', ms=3.0, mfc='white', mec=col, mew=0.8, ls='none', zorder=3)
    ax.plot([x + DX], [r.factor_leader300], marker='o', ms=3.0, mfc=col, mec=col, mew=0.8, ls='none', zorder=3)
    ax.plot([x], [r.contrast], marker=mk, ms=(5.6 if mk != 'D' else 4.6), mfc=col, mec='white', mew=0.4, ls='none', zorder=4)
    if key != 'obs':
        ax.text(x - 0.13, r.contrast, f'{r.contrast:.2f}', ha='right', va='center', fontsize=6.5, color='#222222')
ax.set_xticks([XP[k] for k, *_ in SER])
ax.set_xticklabels([lab for _, lab, *_ in SER])
for t, (_, _, col, _, _) in zip(ax.get_xticklabels(), SER):
    t.set_color('#222222')
ax.tick_params(axis='x', length=0, pad=2.5)
ax.yaxis.set_major_locator(mt.FixedLocator([0.5, 1, 2, 4, 8]))
ax.yaxis.set_major_formatter(mt.FixedFormatter(['0.5', '1', '2', '4', '8']))
ax.yaxis.set_minor_locator(mt.FixedLocator([0.4, 0.6, 0.7, 0.8, 0.9, 3, 5, 6, 7, 9]))
ax.yaxis.set_minor_formatter(mt.NullFormatter())
ax.set_ylabel('Differentiated / same-limit ratio (log scale)')
# group headers inside the axes
tr = ax.get_xaxis_transform()
for (a, b), txt in (((XP['M0'], XP['M0']), 'No limit\nterm'), ((XP['M1'], XP['M1p']), 'Limit in the\ndesired speed'),
                    ((XP['M2F'], XP['M2Fl']), 'Bias of\nunconstrained cars'), ((XP['M1pF'], XP['M1pF']), 'Both')):
    ax.text((a + b) / 2 + DX / 2, 0.985, txt, transform=tr, ha='center', va='top', fontsize=6.5, color=GREY_TXT, linespacing=1.05)
    ax.plot([a - 0.3, b + DX + 0.2], [0.89, 0.89], transform=tr, color='#BDBDBD', lw=0.5, clip_on=False)
# row under the axis: leave-one-day-out log-likelihood
yy = -0.085
ax.text(-0.62, yy, 'LODO LL', transform=tr, ha='right', va='top', fontsize=FS, color=GREY_TXT)
for key, *_ in SER:
    v = D.lodo_ll[key]
    ax.text(XP[key] + DX / 2 - 0.05, yy, '–' if np.isnan(v) else f'{v:.1f}'.replace('-', '−'), transform=tr, ha='center', va='top', fontsize=FS, color='#222222')
# marker key (lower right, below 1)
kx = 3.98; tx = kx + 0.2
ax.plot([kx], [0.70], marker='o', ms=5.0, mfc='#666666', mec='white', mew=0.4, ls='none')
ax.text(tx, 0.70, 'Contrast = open ÷ filled', ha='left', va='center', fontsize=6.5, color='#222222')
ax.plot([kx], [0.58], marker='o', ms=3.0, mfc='white', mec='#666666', mew=0.8, ls='none')
ax.text(tx, 0.58, 'Factor, unconstrained', ha='left', va='center', fontsize=6.5, color='#222222')
ax.plot([kx], [0.48], marker='o', ms=3.0, mfc='#666666', mec='#666666', mew=0.8, ls='none')
ax.text(tx, 0.48, 'Factor, with a leader\nwithin 300 m', ha='left', va='top', fontsize=6.5, color='#222222', linespacing=1.05,
        transform=ax.transData + mtransforms.ScaledTranslation(0, 3.2 / 72, fig.dpi_scale_trans))
save(fig, 'Figure_6')
plt.close(fig)
print('Fig13 ok')
