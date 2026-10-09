# -*- coding: utf-8 -*-
"""Figure style and day metadata: output folders (results/figures for the figures of the paper, results/final for tables and figure
data, _cache/working_figures for working figures), DAY_META (limits, dates, sessions), ORDER8 (the three same-limit days, then the five
differentiated days by Delta), colours, Matplotlib settings (Arial; PNG at 300 dpi and vector PDF), save(), gantry lines and the exact
day-level tests over the 56 labellings of the eight days (perm8, perm_rank8)."""
import os
from itertools import combinations
import numpy as np, pandas as pd
from lc_common import *   # noqa: F401,F403  (paths, DAYS, frames, lc_events, perm_same_vs_diff ...)

FIGF = os.path.join(RES, 'figures'); DATF = os.path.join(RES, 'final')
FIGW = os.path.join(os.path.dirname(RES), '_cache', 'working_figures')   # working figures of the data scripts (not in the paper)
for _p in (FIGF, DATF, FIGW):
    os.makedirs(_p, exist_ok=True)

MM = 1 / 25.4
W1, W15, W2 = 90 * MM, 140 * MM, 190 * MM          # single, 1.5 and double column widths (inches)

# --- colours -----------------------------------------------------------------------------------------------------------------
C_IN, C_OUT = '#1B9E77', '#D95F02'                   # direction: towards inner lane / towards outer lane
C_SAME, C_DIFF = '#8C8C8C', '#1F4E9A'                # group colours: same-limit days / differentiated days
DELTA_COL = {0: '#9E9E9E', 10: '#9ECAE1', 20: '#4292C6', 30: '#08306B'}   # lane differential grey -> dark blue
C_MODEL = {'M0': '#999999', 'M1': '#CC79A7', 'M1p': '#0072B2', 'bias': '#E69F00', 'SUMO': '#D55E00'}
C_GANTRY = '#555555'

# --- day metadata ------------------------------------------------------------------------------------------------------------
DAY_META = pd.DataFrame([
    # name, V_in, V_out, date, session, am_pm
    ('scen_2022-08-08_60-60', 60, 60, '8 Aug (Mon)', '09:53-15:53', 'day'),
    ('scen_2022-07-29_80-80', 80, 80, '29 Jul (Fri)', '08:00-12:00', 'am'),
    ('scen_2022-08-11_100-100', 100, 100, '11 Aug (Thu)', '09:58-15:58', 'day'),
    ('scen_2022-08-02_80-70', 80, 70, '2 Aug (Tue)', '07:56-12:00', 'am'),
    ('scen_2022-08-03_90-70', 90, 70, '3 Aug (Wed)', '07:53-10:30', 'am'),
    ('scen_2022-07-28_100-80', 100, 80, '28 Jul (Thu)', '07:50-12:00', 'am'),
    ('scen_2022-08-01_80-60', 80, 60, '1 Aug (Mon)', '07:56-11:59', 'am'),
    ('scen_2022-08-05_100-70', 100, 70, '5 Aug (Fri)', '10:00-16:00', 'day'),
], columns=['day', 'vin', 'vout', 'date', 'session', 'ampm'])
DAY_META['delta'] = DAY_META.vin - DAY_META.vout
DAY_META['cfg'] = DAY_META.vin.astype(str) + '/' + DAY_META.vout.astype(str)
DAY_META['label'] = DAY_META.cfg
ORDER8 = list(DAY_META.day)   # same-limit (3), differentiated (5, by Delta)


def day_color(n):
    return DELTA_COL[int(DAY_META.set_index('day').delta[n])]


# --- style -------------------------------------------------------------------------------------------------------------------
def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 7.5, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'legend.frameon': False,
        'legend.handlelength': 1.6, 'legend.borderaxespad': 0.3,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
        'lines.linewidth': 1.0, 'lines.markersize': 4, 'errorbar.capsize': 0,
        'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.default': 'regular', 'axes.unicode_minus': True})
    return plt


def save(fig, name, folder=None):
    """save as results/figures/<name>.png (300 dpi) and .pdf (folder=FIGW: working figure in _cache/working_figures)"""
    folder = folder or FIGF
    fig.savefig(os.path.join(folder, name + '.png'), dpi=300)
    fig.savefig(os.path.join(folder, name + '.pdf'))
    return os.path.join(folder, name + '.png')


def gantry_lines(ax, ymin=0, ymax=1, label=True, y_text=1.0):
    for k, g in enumerate(GANTRY):
        ax.axvline(g, color=C_GANTRY, lw=0.6, ls=(0, (4, 2)), zorder=0)
        if label:
            ax.text(g, y_text, f'G{k + 1}', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=6.5, color=C_GANTRY)


# --- day-level exact tests ---------------------------------------------------------------------------------------------------
def perm8(vals8):
    """exact 56-relabelling test, first 3 = same-limit days; returns (diff-same, p two-sided, p one-sided (diff>same))"""
    obs, p2, p1, _ = perm_same_vs_diff(vals8)
    return obs, p2, p1


def perm_rank8(vals8):
    """exact 56-relabelling test on the rank-sum (Mann-Whitney type), two-sided"""
    r = pd.Series(vals8).rank().values
    return perm_same_vs_diff(r)[1]


def fmt_p(p):
    return 'p < 0.001' if p < 0.001 else (f'p = {p:.3f}' if p < 0.1 else f'p = {p:.2f}')
