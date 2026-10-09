# -*- coding: utf-8 -*-
"""Shared settings and helpers: result and cache folders (results/, _cache/work), the eight days (DAYS) with their labels and limit
differences, the analysis section and gantry positions, the frame and event loaders (data_common.py), plot defaults and the exact
day-level relabelling test of the same-limit against the differentiated days (perm_same_vs_diff)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'code'))
import numpy as np, pandas as pd
import data_common as C

PROJ = C.PROJ
RES = os.path.join(PROJ, 'results'); FIG = os.path.join(C.CACHE, 'working_figures'); CACHE = os.path.join(C.CACHE, 'work')
for p in (RES, FIG, CACHE):
    os.makedirs(p, exist_ok=True)

DAYS = C.DAYS                      # the eight days (name, V_in, V_out)
X0, X1 = 900.0, 2100.0             # analysis window
GANTRY = (1250.0, 1948.0)
COL_IN, COL_OUT = '#1B9E77', '#D95F02'


def label(n):
    for d, vi, vo in DAYS:
        if d == n:
            return f'{vi}/{vo}'
    return n


def delta(n):
    for d, vi, vo in DAYS:
        if d == n:
            return vi - vo
    return np.nan


def frames(n):
    return C.frames(n)


def lc_events(n, grade=True):
    return C.lc_events(n, grade)


def diff_colors(deltas):
    """grey -> dark blue by lane differential (0, 10, 20, 30)"""
    cmap = {0: '#9E9E9E', 10: '#9ECAE1', 20: '#4292C6', 30: '#08306B'}
    return [cmap.get(int(d), '#333333') for d in deltas]


def setup_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 8.5, 'axes.labelsize': 9, 'axes.titlesize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5, 'legend.frameon': False,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.7,
        'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'xtick.major.size': 3, 'ytick.major.size': 3,
        'lines.linewidth': 1.2, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
        'pdf.fonttype': 42, 'mathtext.default': 'regular'})
    return plt


def perm_same_vs_diff(values_by_day, same_days=3, stat='mean_diff', two_sided=True):
    """Exact day-level permutation for the same-limit vs differentiated label.
    values_by_day: array of 8 day-level values, first `same_days` are the same-limit days.
    Returns observed statistic (mean(diff) - mean(same)) and exact p-values over C(8,3)=56 relabellings."""
    from itertools import combinations
    v = np.asarray(values_by_day, float); k = len(v)
    obs = v[same_days:].mean() - v[:same_days].mean()
    null = []
    for S in combinations(range(k), same_days):
        m = np.ones(k, bool); m[list(S)] = False
        null.append(v[m].mean() - v[~m].mean())
    null = np.array(null)
    p_two = np.mean(np.abs(null) >= abs(obs) - 1e-12)
    p_one = np.mean(null >= obs - 1e-12)
    return obs, p_two, p_one, len(null)
