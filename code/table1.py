# -*- coding: utf-8 -*-
"""Table 1 (observation days): the three same-limit and five differentiated days with group subtotals and the eight-day total.
Inputs: results/final/Table1_days_values.csv (table1_data.py); results/sit_day_rates.csv (unconstrained inward changes of cars and car-km
in that situation); results/lateral_check.csv (lateral confirmation of each lane change).
Outputs: results/final/Table1_days.csv (layout, footnote markers a-f), Table1_days_numbers.csv (the same rows as numbers)."""
import os
from plot_style import *

T = pd.read_csv(os.path.join(DATF, 'Table1_days_values.csv'))
T['day'] = ['scen_' + d + '_' + c.replace('/', '-') for d, c in zip(T.date, T.configuration_inner_outer_kmh)]
R = pd.read_csv(os.path.join(RES, 'sit_day_rates.csv'))
fr = R[(R.truck == 0) & (R.type == 'IN_free')].set_index('day')
T['free_in_car'] = T.day.map(fr.n).astype(int)
T['free_car_km'] = T.day.map(fr.km)
T['free_rate'] = T.free_in_car / T.free_car_km
L = pd.read_csv(os.path.join(RES, 'lateral_check.csv'))
lat = L.groupby(['day', 'truck']).lat_ok.agg(['size', 'sum']).unstack()
T['lat_car'] = T.day.map(lat[('sum', 0)]).astype(int)
T['lat_heavy'] = T.day.map(lat[('sum', 1)]).astype(int)
assert (T.day.map(lat[('size', 0)]) == T.lc_car).all() and (T.day.map(lat[('size', 1)]) == T.lc_heavy).all()

T = T.set_index('day').loc[ORDER8].reset_index()
FOOT = {'scen_2022-07-29_80-80': 'a', 'scen_2022-08-03_90-70': 'b', 'scen_2022-08-08_60-60': 'c',
        'scen_2022-08-11_100-100': 'd'}


def fdate(d):
    t = pd.Timestamp(d); return f"{t.day} {t.strftime('%b')} ({t.strftime('%a')})"


rows, nums = [], []


def add_group(title, sub):
    rows.append({'Configuration, inner/outer (km/h)': title})
    for r in sub.itertuples():
        mark = FOOT.get(r.day, '')
        rows.append({
            'Configuration, inner/outer (km/h)': r.configuration_inner_outer_kmh + (f' ({mark})' if mark else ''),
            'Delta (km/h)': f'{r.delta_kmh}',
            'Date, 2022': fdate(r.date),
            'Session analysed (local time)': r.session_analysed.replace('-', '–'),
            'Vehicles': f'{r.vehicles:,}',
            'Flow (veh/h)': f'{r.flow_vph:.0f}',
            'Heavy vehicles (%)': f'{r.heavy_share * 100:.0f}',
            'Car-km': f'{r.car_km:,.1f}',
            'Lane changes, car/heavy': f'{r.lc_car}/{r.lc_heavy}',
            'Lateral-confirmed, car/heavy (e)': f'{r.lat_car}/{r.lat_heavy}',
            'Car inward share': f'{r.inward_share_cars:.2f}',
            'Free inward car changes / free car-km': f'{r.free_in_car}/{r.free_car_km:.1f}',
            'Free inward rate (per car-km)': f'{r.free_rate:.3f}'})
        nums.append(dict(day=r.day, group=r.group, cfg=r.configuration_inner_outer_kmh, delta=r.delta_kmh, date=r.date,
                         session=r.session_analysed, hours=r.session_hours, vehicles=r.vehicles, flow_vph=r.flow_vph,
                         heavy_share=r.heavy_share, car_km=r.car_km, heavy_km=r.heavy_km, lc_car=r.lc_car, lc_heavy=r.lc_heavy,
                         lat_car=r.lat_car, lat_heavy=r.lat_heavy, car_inward=r.car_inward, inward_share_cars=r.inward_share_cars,
                         free_in_car=r.free_in_car, free_car_km=r.free_car_km, free_rate=r.free_rate))
    if len(sub) > 1:
        car_in = sub.car_inward.sum(); lcc = sub.lc_car.sum()
        rows.append({
            'Configuration, inner/outer (km/h)': f'Subtotal, {len(sub)} days',
            'Vehicles': f'{sub.vehicles.sum():,}',
            'Flow (veh/h)': f'{sub.vehicles.sum() / sub.session_hours.sum():.0f}',
            'Heavy vehicles (%)': f'{(sub.vehicles * sub.heavy_share).sum() / sub.vehicles.sum() * 100:.0f}',
            'Car-km': f'{sub.car_km.sum():,.1f}',
            'Lane changes, car/heavy': f'{lcc}/{sub.lc_heavy.sum()}',
            'Lateral-confirmed, car/heavy (e)': f'{sub.lat_car.sum()}/{sub.lat_heavy.sum()}',
            'Car inward share': f'{car_in / lcc:.2f}',
            'Free inward car changes / free car-km': f'{sub.free_in_car.sum()}/{sub.free_car_km.sum():.1f}',
            'Free inward rate (per car-km)': f'{sub.free_rate.mean():.3f} (f)'})
        nums.append(dict(day='subtotal', group=sub.group.iloc[0], vehicles=sub.vehicles.sum(),
                         flow_vph=sub.vehicles.sum() / sub.session_hours.sum(), car_km=sub.car_km.sum(), heavy_km=sub.heavy_km.sum(),
                         lc_car=lcc, lc_heavy=sub.lc_heavy.sum(), lat_car=sub.lat_car.sum(), lat_heavy=sub.lat_heavy.sum(),
                         car_inward=car_in, inward_share_cars=car_in / lcc, free_in_car=sub.free_in_car.sum(),
                         free_car_km=sub.free_car_km.sum(), free_rate=sub.free_rate.mean()))


add_group('Same limit on both lanes', T[T.group == 'same-limit'])
add_group('Differentiated (inner lane higher)', T[T.group == 'differentiated'])
main = T
rows.append({'Configuration, inner/outer (km/h)': 'Eight days',
             'Vehicles': f'{main.vehicles.sum():,}', 'Flow (veh/h)': f'{main.vehicles.sum() / main.session_hours.sum():.0f}',
             'Heavy vehicles (%)': f'{(main.vehicles * main.heavy_share).sum() / main.vehicles.sum() * 100:.0f}',
             'Car-km': f'{main.car_km.sum():,.1f}', 'Lane changes, car/heavy': f'{main.lc_car.sum()}/{main.lc_heavy.sum()}',
             'Lateral-confirmed, car/heavy (e)': f'{main.lat_car.sum()}/{main.lat_heavy.sum()}',
             'Car inward share': f'{main.car_inward.sum() / main.lc_car.sum():.2f}',
             'Free inward car changes / free car-km': f'{main.free_in_car.sum()}/{main.free_car_km.sum():.1f}'})
nums.append(dict(day='all', group='all', vehicles=main.vehicles.sum(), car_km=main.car_km.sum(), heavy_km=main.heavy_km.sum(),
                 lc_car=main.lc_car.sum(), lc_heavy=main.lc_heavy.sum(), lat_car=main.lat_car.sum(), lat_heavy=main.lat_heavy.sum(),
                 car_inward=main.car_inward.sum(), free_in_car=main.free_in_car.sum(), free_car_km=main.free_car_km.sum()))

OUT = pd.DataFrame(rows).fillna('')
OUT.to_csv(os.path.join(DATF, 'Table1_days.csv'), index=False, encoding='utf-8-sig')
NUM = pd.DataFrame(nums)
NUM.to_csv(os.path.join(DATF, 'Table1_days_numbers.csv'), index=False, float_format='%.4f')
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 30)
print(OUT.to_string(index=False))

# numbers quoted in the caption
free_share = (T.free_car_km / T.car_km).round(3)
print('free share of car-km by day:', dict(zip(T.configuration_inner_outer_kmh, free_share)))
print('lateral confirmed, 8 days:', main.lat_car.sum() + main.lat_heavy.sum(), 'of', main.lc_car.sum() + main.lc_heavy.sum())
