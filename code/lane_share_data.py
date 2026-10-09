# -*- coding: utf-8 -*-
"""Inner-lane share of cars along the section (Figure 5, Section 5.3).
Cohort: cars with session frames in lanes 1/2 that traverse the whole section (first frame < 1,000 m, last frame > 2,000 m) with a modal
lane in 1,000-1,100 m and in 1,900-2,000 m. For every 100 m bin in [1,000, 2,000) m the car's modal lane is taken; the inner-lane share of
a bin is the share of cohort cars whose modal lane is lane 1. Day-level tests of the change from 1,000-1,100 m to 1,900-2,000 m and of
the entry share, and the decomposition of the change into net lane changes by type. The measured-frame share per bin is kept to flag bins
where lane changes are placed late.
Inputs: session frames; results/final/lane_sequences_vehicles.csv; _cache/work/sit_events_typed.pkl.
Outputs: results/final/fig5_lane_share_change_data.csv, fig5_group_means.csv, fig5_tests.csv, fig5_gain_decomposition.csv,
fig5_gain_decomposition_tests.csv; _cache/work/lane_share_data.pkl (read by fig5_lane_share.py)."""
import os
from plot_style import *

BINS = np.arange(1000, 2001, 100)
CP = os.path.join(CACHE, 'lane_share_data.pkl')


def build():
    rows = []
    for n, vi, vo in DAYS:
        f = frames(n)
        f = f[(f.in_monitored_session == 1) & (f.in_disturbed_period == 0) & (f.truck == 0) & f.Lane.isin([1, 2])]
        g = f.groupby('GlobalID'); x0 = g.PositionX.min(); x1 = g.PositionX.max()
        full = x0.index[(x0 < 1000) & (x1 > 2000)]
        ff = f[f.GlobalID.isin(full) & (f.PositionX >= 1000) & (f.PositionX < 2000)].copy()
        ff['bin'] = (np.floor(ff.PositionX / 100) * 100).astype(int)
        M = ff.groupby(['GlobalID', 'bin']).Lane.agg(lambda s: int(s.mode().iloc[0])).unstack('bin')
        M = M[M[1000].notna() & M[1900].notna()]            # modal lane in entry and exit bins (cohort of check_lane_shift.py)
        fm = ff[ff.GlobalID.isin(M.index)]
        meas = fm.groupby('bin').frame_source.apply(lambda s: float((s <= 1).mean()))
        nfr = fm.groupby('bin').size()
        for b in BINS[:-1]:
            col = M[b].dropna()
            rows.append(dict(day=n, cfg=f'{vi}/{vo}', delta=vi - vo, x_lo=int(b), x_mid=int(b) + 50, n_cars=len(col), n_inner=int((col == 1).sum()),
                             share_inner=float((col == 1).mean()), meas_share=float(meas.get(b, np.nan)), n_frames=int(nfr.get(b, 0))))
        print(n, 'cohort', len(M), flush=True)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    P = build(); P.to_pickle(CP)
    base = P[P.x_lo == 1000].set_index('day').share_inner
    P['change_pp'] = 100 * (P.share_inner - P.day.map(base))
    P['group'] = np.where(P.delta > 0, 'differentiated', 'same-limit')

    # ---- day-level tests on the entry-to-exit gain (1900-2000 m minus 1000-1100 m)
    gain = P[P.x_lo == 1900].set_index('day').change_pp
    ent = base
    g8 = [gain[d] for d in ORDER8]
    obs, p2, p1 = perm8(g8); p2r = perm_rank8(g8)
    e8 = [ent[d] for d in ORDER8]; oe, pe2, pe1 = perm8(e8)
    T = pd.DataFrame([dict(quantity='gain_pp_exit_minus_entry', same_mean=np.mean(g8[:3]), diff_mean=np.mean(g8[3:]), diff_minus_same=obs,
                           p2_8=p2, p1_8=p1, p2_rank8=p2r),
                      dict(quantity='entry_share_1000_1100', same_mean=np.mean(e8[:3]), diff_mean=np.mean(e8[3:]), diff_minus_same=oe,
                           p2_8=pe2, p1_8=pe1, p2_rank8=perm_rank8(e8))])
    # per-bin group means (unweighted over days) and exact test of the change in each bin
    G = []
    for b in BINS[:-1]:
        z = P[P.x_lo == b].set_index('day')
        c8 = [z.change_pp[d] for d in ORDER8]; s8 = [z.share_inner[d] for d in ORDER8]
        w = z.loc[ORDER8]
        G.append(dict(x_lo=b, x_mid=b + 50, same_share=np.mean(s8[:3]), diff_share=np.mean(s8[3:]),
                      same_change_pp=np.mean(c8[:3]), diff_change_pp=np.mean(c8[3:]),
                      p2_change_8=perm8(c8)[1] if b > 1000 else np.nan, n_cars_same=int(z.n_cars[ORDER8[:3]].sum()), n_cars_diff=int(z.n_cars[ORDER8[3:]].sum()),
                      meas_share_all=float((w.meas_share * w.n_frames).sum() / w.n_frames.sum())))
    G = pd.DataFrame(G)
    pd.set_option('display.width', 220)
    print(T.round(4).to_string(index=False)); print(G.round(3).to_string(index=False))

    # ---- data files
    P[['day', 'cfg', 'delta', 'group', 'x_lo', 'x_mid', 'n_cars', 'n_inner', 'share_inner', 'change_pp', 'meas_share', 'n_frames']].to_csv(
        os.path.join(DATF, 'fig5_lane_share_change_data.csv'), index=False)
    G.to_csv(os.path.join(DATF, 'fig5_group_means.csv'), index=False)
    T.to_csv(os.path.join(DATF, 'fig5_tests.csv'), index=False)


    # ---- where the gain comes from: net lane-change balance of the cohort by change type (s1 typing at t-3 s),
    #      analysis-grade changes recorded at 1000 <= x < 2000 m by cohort cars; pp of the day's cohort
    V = pd.read_csv(os.path.join(DATF, 'lane_sequences_vehicles.csv')); V = V[V.truck == 0]
    EV = pd.read_pickle(os.path.join(CACHE, 'sit_events_typed.pkl'))
    EV = EV[(EV.truck == 0) & (EV.x >= 1000) & (EV.x < 2000)].merge(V[['day', 'GlobalID']], on=['day', 'GlobalID'])
    EV['grp'] = np.where(EV.to_lane == 1, np.where(EV.typ == 'IN_free', 'in_free', np.where(EV.typ == 'IN_overtake', 'in_leader', 'in_other')), 'out')
    ncoh = V.groupby('day').size()
    TT = EV.groupby(['day', 'grp']).size().unstack(fill_value=0).reindex(columns=['in_free', 'in_leader', 'in_other', 'out'], fill_value=0)
    DEC = pd.DataFrame({c: 100 * TT[c] / ncoh for c in TT.columns}).loc[ORDER8]
    DEC['net_events_pp'] = DEC.in_free + DEC.in_leader + DEC.in_other - DEC.out
    DEC['gain_pp_modal_lane'] = gain.reindex(ORDER8).values; DEC['n_cohort'] = ncoh.reindex(ORDER8).values
    DEC.index.name = 'day'
    dt = []
    for c in ('in_free', 'in_leader', 'in_other', 'out', 'net_events_pp', 'gain_pp_modal_lane'):
        v8 = DEC[c].values[:8].tolist(); o, p2_, p1_ = perm8(v8)
        dt.append(dict(component=c, same_mean=np.mean(v8[:3]), diff_mean=np.mean(v8[3:]), diff_minus_same=o, p2_8=p2_))
    dt = pd.DataFrame(dt); print(dt.round(3).to_string(index=False))
    DEC.to_csv(os.path.join(DATF, 'fig5_gain_decomposition.csv')); dt.to_csv(os.path.join(DATF, 'fig5_gain_decomposition_tests.csv'), index=False)
