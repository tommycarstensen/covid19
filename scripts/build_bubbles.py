"""Write the data behind the page's interactive bubble charts.

The page showed, for each of eight regions, plot_bubble.py's chart of one moment: how each country's cases and deaths in the latest two weeks compared with the two weeks before, sized by deaths to date and coloured by the change in tests. site/bubble/bubble.js draws the same chart for every week from the data written here, with a play button and a slider.

This writes site/bubble/bubble_data.js: each country's weekly cases and deaths from ecdc.csv (ECDC's weekly file, to the week of 4 to 10 January 2021) and its weekly tests from owid.csv (seven times OWID's new_tests_smoothed on the Sunday that ends the week, which is the week's total). A week a country did not report is null, never zero. The regions are plot_bubble.py's. Nothing is downloaded.

Usage: python3 scripts/build_bubbles.py
"""

import json
import math
import sys
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path

import pandas as pd

from plot_bubble import d_regions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'site' / 'bubble' / 'bubble_data.js'


def week_end(year_week: str) -> date:
    """The Sunday that ends an ISO week written as ECDC writes it, '2020-53'."""
    year, week = (int(part) for part in year_week.split('-'))
    return date.fromisocalendar(year, week, 7)


def count(value: object) -> int | None:
    """A weekly count as an int, or None when it is missing."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(str(value)))


def main() -> None:
    ecdc = pd.read_csv(ROOT / 'data' / 'ecdc.csv')
    owid = pd.read_csv(ROOT / 'data' / 'owid.csv', usecols=['iso_code', 'location', 'date', 'new_tests_smoothed'])
    print(f'ecdc.csv: {len(ecdc):,} rows; owid.csv: {len(owid):,} rows')

    weeks = sorted(str(week) for week in ecdc['year_week'].unique())
    ends = [week_end(week) for week in weeks]
    gaps = [b for a, b in pairwise(ends) if b - a != timedelta(weeks=1)]
    if gaps:
        sys.exit(f'ecdc.csv skips the weeks before {gaps}')
    if ecdc.duplicated(['countryterritoryCode', 'year_week']).any():
        sys.exit('ecdc.csv has more than one row for a country and week')
    print(f'Weeks {weeks[0]} to {weeks[-1]}: {len(weeks)}, ending {ends[0]} to {ends[-1]}')

    sundays = [end.isoformat() for end in ends]
    owid = owid[owid['date'].isin(sundays)]
    names = dict(zip(owid['iso_code'], owid['location']))
    tests = {(str(iso), str(day)): value for iso, day, value in zip(owid['iso_code'], owid['date'], owid['new_tests_smoothed'])}

    wanted = sorted(set().union(*d_regions.values()))
    countries: dict[str, dict[str, object]] = {}
    absent: list[str] = []
    with_tests = 0
    for code in wanted:
        rows = ecdc[ecdc['countryterritoryCode'] == code].set_index('year_week')
        if rows.empty:
            absent.append(code)
            continue
        cases = [count(rows['cases_weekly'].get(week)) for week in weeks]
        deaths = [count(rows['deaths_weekly'].get(week)) for week in weeks]
        weekly_tests = [count(7 * value) if (value := tests.get((code, end.isoformat()))) is not None else None for end in ends]
        name = names.get(code) or str(rows['countriesAndTerritories'].iloc[0]).replace('_', ' ')
        countries[code] = {'name': name, 'cases': cases, 'deaths': deaths, 'tests': weekly_tests}
        with_tests += any(value is not None for value in weekly_tests)

    regions = {region: [code for code in codes if code in countries] for region, codes in d_regions.items()}
    data = {
        'weeks': [end.isoformat() for end in ends],
        'regions': regions,
        'countries': countries,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
    OUT.write_text(f'// Built by build_bubbles.py from ecdc.csv and owid.csv. Do not edit.\nwindow.BUBBLE_DATA = {body};\n', encoding='utf-8')

    for region, codes in regions.items():
        print(f'  {region}: {len(codes)} countries')
    print(f'Not in ecdc.csv, left out: {", ".join(absent) or "none"}')
    print(f'Wrote {OUT.relative_to(ROOT)}: {len(countries)} countries, {with_tests} with test data, {OUT.stat().st_size / 1024:.0f} KB')


if __name__ == '__main__':
    main()
