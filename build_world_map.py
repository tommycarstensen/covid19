#!/usr/bin/env python3
"""Build site/worldmap/worldmap.json, the data and outlines that site/worldmap/worldmap.js draws as an interactive world map of ECDC's weekly COVID-19 cases and deaths per million people.

The map replaces the four world-map GIFs of plot_choropleth.py, which jumped between frames and whose weekly versions showed seven-week totals. The counts and populations are ecdc.csv (ECDC's weekly data, ISO weeks 2020-01 to 2021-01, populations of 2019). The outlines are Natural Earth's 1:110m countries, version 4.1.0, in the Equal Earth projection. The markup for the page is written to tmp/build_world_map_markup.html.
"""

import json
import math
from datetime import date
from itertools import pairwise
from pathlib import Path

import pandas as pd
import shapefile

ROOT = Path(__file__).resolve().parent
ECDC = ROOT / 'ecdc.csv'
COUNTRIES = ROOT / 'map' / 'data' / 'countries_110m' / 'ne_110m_admin_0_countries.zip'
OUT = ROOT / 'site' / 'worldmap' / 'worldmap.json'
MARKUP = ROOT / 'tmp' / 'build_world_map_markup.html'

# Natural Earth code -> ECDC code where the two differ. Somaliland takes Somalia's figures, as it did in plot_choropleth.py.
CODE_FIXES = {'KOS': 'XKX', 'TWN': 'CNG1925', 'SOL': 'SOM'}

WIDTH = 1000  # width of the SVG viewBox
PAD = 4

# Equal Earth projection (Šavrič, Patterson and Jenny 2018).
A1, A2, A3, A4 = 1.340264, -0.081106, 0.000893, 0.003796
M = math.sqrt(3) / 2


def equal_earth(lon: float, lat: float) -> tuple[float, float]:
    theta = math.asin(M * math.sin(math.radians(lat)))
    t2 = theta * theta
    t6 = t2 ** 3
    x = math.radians(lon) * math.cos(theta) / (M * (A1 + 3 * A2 * t2 + t6 * (7 * A3 + 9 * A4 * t2)))
    y = theta * (A1 + A2 * t2 + t6 * (A3 + A4 * t2))
    return x, y


def read_shapes() -> list[dict]:
    """One dict per Natural Earth country: its ECDC code, its name and its projected rings."""
    reader = shapefile.Reader(str(COUNTRIES))
    shapes = []
    for shape_record in reader.iterShapeRecords():
        shape, record = shape_record.shape, shape_record.record
        if shape is None or record is None:
            continue
        fields = record.as_dict()
        if fields['ADM0_A3'] == 'ATA':  # Antarctica
            continue
        code = str(fields['ISO_A3'] if fields['ISO_A3'] != '-99' else fields['ADM0_A3'])
        parts = [*shape.parts, len(shape.points)]
        rings = [[equal_earth(point[0], point[1]) for point in shape.points[start:end]] for start, end in pairwise(parts)]
        shapes.append({'code': CODE_FIXES.get(code, code), 'name': str(fields['NAME']), 'rings': rings})
    return shapes


def number(value: float) -> str:
    return f'{value:.1f}'.rstrip('0').rstrip('.')


def to_paths(shapes: list[dict]) -> tuple[list[dict], int]:
    """SVG path data for each shape, scaled to the viewBox, and the viewBox height."""
    xs = [x for shape in shapes for ring in shape['rings'] for x, _ in ring]
    ys = [y for shape in shapes for ring in shape['rings'] for _, y in ring]
    scale = (WIDTH - 2 * PAD) / (max(xs) - min(xs))
    height = math.ceil((max(ys) - min(ys)) * scale + 2 * PAD)
    paths = []
    for shape in shapes:
        d = []
        for ring in shape['rings']:
            points = []
            for x, y in ring:
                point = (number(PAD + (x - min(xs)) * scale), number(PAD + (max(ys) - y) * scale))
                if not points or point != points[-1]:
                    points.append(point)
            if len(points) >= 3:
                d.append('M' + points[0][0] + ' ' + points[0][1] + 'L' + ' '.join(x + ' ' + y for x, y in points[1:]) + 'Z')
        paths.append({'code': shape['code'], 'name': shape['name'], 'd': ''.join(d)})
    return paths, height


def read_ecdc() -> tuple[list[str], dict]:
    """The ISO weeks, and per ECDC country code its name, population and weekly cases and deaths (None for a week without a row)."""
    df = pd.read_csv(ECDC).dropna(subset=['countryterritoryCode', 'popData2019'])
    weeks = sorted(df['year_week'].unique())
    countries = {}
    for code, group in df.groupby('countryterritoryCode'):
        cases = dict(zip(group['year_week'], group['cases_weekly']))
        deaths = dict(zip(group['year_week'], group['deaths_weekly']))
        countries[str(code)] = {
            'name': str(group['countriesAndTerritories'].iloc[0]).replace('_', ' '),
            'pop': int(group['popData2019'].iloc[0]),
            'cases': [int(cases[week]) if week in cases else None for week in weeks],
            'deaths': [int(deaths[week]) if week in deaths else None for week in weeks],
            }
    return weeks, countries


def main() -> None:
    print(f'Reading {ECDC.name} and {COUNTRIES.name}')
    weeks, countries = read_ecdc()
    paths, height = to_paths(read_shapes())
    week_starts = [date.fromisocalendar(int(week[:4]), int(week[5:]), 1).isoformat() for week in weeks]
    data = {
        'source': 'ECDC weekly COVID-19 cases and deaths (ecdc.csv), populations of 2019; outlines: Natural Earth 1:110m countries 4.1.0, Equal Earth projection',
        'viewBox': [0, 0, WIDTH, height],
        'weeks': weeks,
        'weekStarts': week_starts,
        'shapes': paths,
        'countries': countries,
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(',', ':'), ensure_ascii=False) + '\n', encoding='utf-8')

    MARKUP.parent.mkdir(parents=True, exist_ok=True)
    MARKUP.write_text(
        '<link rel="stylesheet" href="worldmap/worldmap.css">\n'
        '<div class="worldmap" data-src="worldmap/worldmap.json">\n'
        '<noscript>This interactive map of ECDC\'s weekly COVID-19 cases and deaths needs JavaScript.</noscript>\n'
        '</div>\n'
        '<script src="worldmap/worldmap.js" defer></script>\n', encoding='utf-8')

    on_map = {path['code'] for path in paths}
    no_data = sorted(path['name'] for path in paths if path['code'] not in countries)
    print(f'{len(weeks)} weeks ({weeks[0]} to {weeks[-1]}), {len(countries)} countries and territories, {len(paths)} outlines')
    print(f'Outlines without ECDC data: {", ".join(no_data)}')
    print(f'Too small for the map, in its table only: {len(set(countries) - on_map)}')
    print(f'Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB) and {MARKUP.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
