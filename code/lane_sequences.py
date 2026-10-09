# -*- coding: utf-8 -*-
"""Lane sequences of vehicles that traverse the whole section and enter it in the outer lane (Section 5.3).
Cohort as in lane_share_data.py (modal lane at 1,000-1,100 m = lane 2). Sequence: the entry lane followed by the lateral-confirmed
analysis-grade lane changes recorded at 1,000 <= x < 2,000 m, in time order ('lat'); other versions: all analysis-grade changes ('ag') and
all lane runs ('runs_trim'). Classes: '2' stayed in the outer lane, '21' moved in and stayed, '212' moved in and returned, 'other'.
Day shares with exact day-level tests; a sensitivity check for 3 Aug (vehicles reaching 1,000 m after 10:20 left out); a working figure
of the cars.
Inputs: session frames; _cache/work/core_events.pkl, core_vehicles.pkl; results/lateral_check.csv, results/check_stint_seq_day.csv.
Outputs: results/final/lane_sequences_vehicles.csv, lane_sequences_data.csv, lane_sequences_tests.csv, lane_sequences_sensitivity_0803.csv,
lane_sequences_plotted.csv; _cache/work/lane_seq_cohort.pkl; working figure _cache/working_figures/lane_sequences.png/.pdf."""
import os
from plot_style import *
from matplotlib.patches import Patch

CP = os.path.join(CACHE, 'lane_seq_cohort.pkl')


def cohort():
    rows = []
    for n, vi, vo in DAYS:
        f = frames(n)
        f = f[(f.in_monitored_session == 1) & (f.in_disturbed_period == 0) & (f.truck >= 0) & f.Lane.isin([1, 2])]
        g = f.groupby('GlobalID'); x0 = g.PositionX.min(); x1 = g.PositionX.max(); tr = g.truck.max()
        full = x0.index[(x0 < 1000) & (x1 > 2000)]
        ff = f[f.GlobalID.isin(full)]
        ent = ff[(ff.PositionX >= 1000) & (ff.PositionX < 1100)].groupby('GlobalID').Lane.agg(lambda s: int(s.mode().iloc[0]))
        ext = ff[(ff.PositionX >= 1900) & (ff.PositionX < 2000)].groupby('GlobalID').Lane.agg(lambda s: int(s.mode().iloc[0]))
        j = pd.concat([ent.rename('lane_entry'), ext.rename('lane_exit')], axis=1).dropna().astype(int)
        j['truck'] = tr.reindex(j.index).astype(int); j['day'] = n
        rows.append(j.reset_index().rename(columns={'index': 'GlobalID'}))
        print(n, len(j), flush=True)
    return pd.concat(rows, ignore_index=True)


def classify(s):
    if s == '2':
        return '2'
    if s == '21':
        return '21'
    if s == '212':
        return '212'
    return 'other'


def build_sequences(V, E, lat):
    ev = E[(E.x >= 1000) & (E.x < 2000)]
    if lat:
        ev = ev[ev.lat_ok == 1]
    ev = ev.sort_values('t')
    byv = {k: d for k, d in ev.groupby(['day', 'GlobalID'])}
    out = []
    for r in V.itertuples():
        cur = r.lane_entry; s = str(cur)
        d = byv.get((r.day, r.GlobalID))
        if d is not None:
            for to in d.to_lane.values:
                if to != cur:
                    s += str(int(to)); cur = to
        out.append(s)
    return out


if __name__ == '__main__':
    V = cohort(); V.to_pickle(CP)
    E = pd.read_pickle(os.path.join(CACHE, 'core_events.pkl'))
    L = pd.read_csv(os.path.join(RES, 'lateral_check.csv'))
    E = E.merge(L[['day', 'GlobalID', 't', 'lat_ok']], on=['day', 'GlobalID', 't'], how='left')
    CV = pd.read_pickle(os.path.join(CACHE, 'core_vehicles.pkl'))[['day', 'GlobalID', 'seq_runs']]
    V = V.merge(CV, on=['day', 'GlobalID'], how='left')
    V['seq_lat'] = build_sequences(V, E, True)
    V['seq_ag'] = build_sequences(V, E, False)
    # run-based sequence trimmed to start at the entry lane (drops runs that end before the entry bin is reached is not possible here;
    # use the full run sequence and strip leading lanes that differ from the entry lane)
    V['seq_runs_trim'] = [s[s.index(str(e)):] if isinstance(s, str) and str(e) in s else np.nan for s, e in zip(V.seq_runs, V.lane_entry)]
    for c in ('seq_lat', 'seq_ag', 'seq_runs_trim'):
        V[c.replace('seq', 'cls')] = [classify(s) if isinstance(s, str) else np.nan for s in V[c]]
    V['end_lat_matches_exit'] = (V.seq_lat.str[-1].astype(int) == V.lane_exit).astype(int)
    V['end_ag_matches_exit'] = (V.seq_ag.str[-1].astype(int) == V.lane_exit).astype(int)
    V.to_csv(os.path.join(DATF, 'lane_sequences_vehicles.csv'), index=False)

    rows, tests = [], []
    for tr in (0, 1):
        W = V[(V.truck == tr) & (V.lane_entry == 2)]
        for ver in ('lat', 'ag', 'runs_trim'):
            for d in ORDER8:
                w = W[W.day == d]; c = w[f'cls_{ver}']
                r = dict(truck=tr, version=ver, day=d, cfg=DAY_META.set_index('day').cfg[d], delta=int(DAY_META.set_index('day').delta[d]), n=len(w))
                for k in ('2', '21', '212', 'other'):
                    r[f'n_{k}'] = int((c == k).sum()); r[f'share_{k}'] = float((c == k).mean())
                mi = c.isin(['21', '212', 'other'])
                r['n_moved_in'] = int(mi.sum()); r['return_given_in'] = float((c[mi] != '21').mean()) if mi.sum() else np.nan
                r['end_matches_exit'] = float(w[f'end_{ver}_matches_exit'].mean()) if ver in ('lat', 'ag') else np.nan
                rows.append(r)
    D = pd.DataFrame(rows)
    for (tr, ver), z in D.groupby(['truck', 'version']):
        z = z.set_index('day')
        for col in ('share_2', 'share_21', 'share_212', 'share_other', 'return_given_in'):
            v8 = [z[col][d] for d in ORDER8]
            if any(not np.isfinite(x) for x in v8):
                continue
            obs, p2, p1 = perm8(v8)
            tests.append(dict(truck=tr, version=ver, quantity=col, same_mean=np.mean(v8[:3]), diff_mean=np.mean(v8[3:]), p2_8=p2, p1_8=p1,
                              p2_rank8=perm_rank8(v8), n_same=int(z.n[ORDER8[:3]].sum()), n_diff=int(z.n[ORDER8[3:]].sum())))
    T = pd.DataFrame(tests)
    # reference: run-level lateral version of check_sequences.py (entry = modal lane 1000-1100 m)
    ref = pd.read_csv(os.path.join(RES, 'check_stint_seq_day.csv'))
    ref = ref[(ref.defn == 'E2_lat') & ref.outcome.isin(['2', '21', '212'])].pivot_table(index='day', columns='outcome', values='share').add_prefix('v_s3_E2lat_share_')
    D = D.merge(ref, left_on='day', right_index=True, how='left')
    D.loc[D.truck == 1, [c for c in D.columns if c.startswith('v_s3_')]] = np.nan
    D.to_csv(os.path.join(DATF, 'lane_sequences_data.csv'), index=False)
    T.to_csv(os.path.join(DATF, 'lane_sequences_tests.csv'), index=False)
    pd.set_option('display.width', 250)
    print(D[(D.version == 'lat')][['truck', 'cfg', 'n', 'share_2', 'share_21', 'share_212', 'share_other', 'n_moved_in', 'return_given_in', 'end_matches_exit']].round(3).to_string(index=False))
    print(T.round(3).to_string(index=False))
    print('agreement lat vs ag class (cars entering lane 2):', (V[(V.truck == 0) & (V.lane_entry == 2)].cls_lat == V[(V.truck == 0) & (V.lane_entry == 2)].cls_ag).mean().round(3))
    print('pooled (vehicle-weighted) 21 share, cars, lat:',
          {g: round(float((V[(V.truck == 0) & (V.lane_entry == 2) & V.day.isin(dd)].cls_lat == '21').mean()), 3) for g, dd in (('same', ORDER8[:3]), ('diff', ORDER8[3:]))})

    # ---- sensitivity: 0803 (90/70) 10:20-10:30, a burst of heavy-vehicle in-and-out moves in the last minutes of the session.
    #      Exclude vehicles of 0803 that reach 1000 m at or after 10:20 and repeat the day-level tests (cars and heavy vehicles).
    d0803 = 'scen_2022-08-03_90-70'
    f = frames(d0803)
    f = f[(f.in_monitored_session == 1) & (f.in_disturbed_period == 0) & (f.PositionX >= 1000)]
    t_in = f.groupby('GlobalID').Timestamp.min()
    h_in = pd.Series(C.hours(t_in.values), index=t_in.index)
    late = set(h_in.index[h_in >= 10 + 20 / 60])
    sens = []
    for tr in (0, 1):
        W = V[(V.truck == tr) & (V.lane_entry == 2)]
        for excl in (False, True):
            WW = W[~((W.day == d0803) & W.GlobalID.isin(late))] if excl else W
            vals = {q: [float((WW[WW.day == d].cls_lat == q).mean()) for d in ORDER8] for q in ('21', '212')}
            for q in ('21', '212'):
                v8 = vals[q][:8]; o, p2, p1 = perm8(v8)
                sens.append(dict(truck=tr, exclude_0803_after_1020=excl, cls=q, n_0803=int((WW.day == d0803).sum()), share_0803=vals[q][4],
                                 same_mean=np.mean(v8[:3]), diff_mean=np.mean(v8[3:]), p2_8=p2))
    SENS = pd.DataFrame(sens); SENS.to_csv(os.path.join(DATF, 'lane_sequences_sensitivity_0803.csv'), index=False)
    print(SENS.round(3).to_string(index=False))

    # ------------------------------------------------------------------------------------------------ figure (cars)
    # Truncated at 40 %: only the classes with an inward move are drawn; the remainder of each row (to 100 %) stayed in the outer
    # lane (class 2, share given in lane_sequences_plotted.csv). Heavy vehicles are not drawn; their numbers are in
    # lane_sequences_data.csv / lane_sequences_tests.csv (truck == 1).
    from lane_share_util import save_exact
    plt = setup()       # working figure (not in the paper)
    COLS = {'21': '#332288', '212': '#CC6677', 'other': '#777777'}       # neutral class colours (C_IN / C_OUT stay reserved for direction)
    LABS = {'21': 'Moved in, still in inner lane at 2,000 m (21)', '212': 'Moved in and returned (212)', 'other': 'Other (e.g. 2121)'}
    KEYS = ('21', '212', 'other')
    XMAX = 40
    z = D[(D.truck == 0) & (D.version == 'lat')].set_index('day')
    t = T[(T.truck == 0) & (T.version == 'lat')].set_index('quantity')
    ALLK = ('21', '212', 'other', '2')
    rows = []          # (y, label, shares dict, n, kind)
    y = 0.0
    for d in ORDER8[:3]:
        rows.append((y, DAY_META.set_index('day').cfg[d], {k: z[f'share_{k}'][d] for k in ALLK}, int(z.n[d]), 'day')); y += 1
    rows.append((y, 'Mean of 3 days', {k: t.loc[f'share_{k}'].same_mean for k in ALLK}, int(t.loc['share_21'].n_same), 'mean')); y += 1.9
    y_diff_head = y - 0.8
    for d in ORDER8[3:]:
        rows.append((y, DAY_META.set_index('day').cfg[d], {k: z[f'share_{k}'][d] for k in ALLK}, int(z.n[d]), 'day')); y += 1
    rows.append((y, 'Mean of 5 days', {k: t.loc[f'share_{k}'].diff_mean for k in ALLK}, int(t.loc['share_21'].n_diff), 'mean'))
    fig, ax = plt.subplots(figsize=(W15, 80 * MM))
    out = []
    for yy, lab, sh, n, kind in rows:
        left = 0.0
        for k in KEYS:
            v = 100 * sh[k]
            if v <= 0:
                continue
            ax.barh(yy, v, left=left, height=0.72, color=COLS[k], edgecolor='white' if kind != 'mean' else '#222222', lw=0.4 if kind != 'mean' else 0.7)
            if k == '21' and v >= 4:
                ax.text(left + v / 2, yy, f'{v:.1f} %' if kind == 'mean' else f'{v:.0f} %', ha='center', va='center', fontsize=6.5, color='white',
                        fontweight='bold' if kind == 'mean' else 'normal')
            left += v
        ax.text(XMAX + 0.4, yy, f'n = {n}', ha='left', va='center', fontsize=6.5, color='#333333', clip_on=False)
        out.append(dict(row=lab, kind=kind, y=yy, n=n, **{f'share_{k}': sh[k] for k in ALLK}))
    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[1] for r in rows], fontsize=7)
    for tl, r in zip(ax.get_yticklabels(), rows):
        if r[4] == 'mean':
            tl.set_fontweight('bold')
    ax.tick_params(axis='y', length=0, pad=3)
    ax.set_ylim(rows[-1][0] + 0.6, -1.25)
    ax.set_xlim(0, XMAX); ax.set_xticks(np.arange(0, XMAX + 1, 10))
    ax.set_xlabel('Share of cars that entered in the outer lane (%)')
    ax.spines['left'].set_visible(False)
    ax.text(-0.012, -0.85, 'Same limit', transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=7, fontweight='bold', color='#444444')
    ax.text(-0.012, y_diff_head, 'Differentiated', transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=7, fontweight='bold', color='#444444')
    H = [Patch(facecolor=COLS[k], edgecolor='none', label=LABS[k]) for k in KEYS]
    H.append(Patch(facecolor='white', edgecolor='#9A9A9A', lw=0.5, label='Not drawn, rest of row to 100 %: stayed in outer lane (2)'))
    ax.legend(handles=H, loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=2, fontsize=6.5, handlelength=1.3, columnspacing=1.6, borderaxespad=0.2,
              labelspacing=0.35)
    a, b = t.loc['share_21'], t.loc['share_212']
    ax.text(0.0, -0.165, f'Class 21: exact day-level {fmt_p(a.p2_8)} (8 days)\n'
            f'Class 212: {fmt_p(b.p2_8)} (8 days)',
            transform=ax.transAxes, ha='left', va='top', fontsize=6.5, linespacing=1.35, color='#222222')
    pd.DataFrame(out).to_csv(os.path.join(DATF, 'lane_sequences_plotted.csv'), index=False)
    save_exact(fig, 'lane_sequences', 140, folder=FIGW); plt.close(fig)
