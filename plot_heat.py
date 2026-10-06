"""Draw the page's regional heat maps of weekly COVID-19 cases and deaths per million people, in the interactive world map's colours.

The 2020 heat maps (plot_heat_{cases,deaths}_<region>.png, drawn by plot_series.doHeatMaps) used matplotlib's OrRd scale, rescaled for every chart, so the same colour meant different rates in different charts; their country names were too small to read; they averaged seven weekly rows, a leftover from ECDC's daily data; and the EU chart left out Czechia. These show single weeks, with the world map's eight classes and colours (blue for cases, orange for deaths, half-decade steps per million people), so a colour means the same rate in every heat map and on the map. Cases and deaths sit side by side with the country names beside them.

Reads ecdc.csv (ECDC weekly cases and deaths to ISO week 2021-01, with ECDC's populations of 2019, as the world map uses) and the regions in regions.py, and downloads nothing. A week before a country's first report counts as 0, as does a week whose count ECDC revised below 0; a missing week after the first report would be hatched, as on the map, but ecdc.csv has none for these countries. Countries of fewer than 100,000 people are left out, because one case there is ten or more per million; the figure names them.

Writes site/heat_<region>.png (wide screens, 2x pixel density) and site/heat_<region>_narrow.png (phones, the two maps stacked, 3x), and their <picture> tags to tmp/plot_heat_markup.html.

Usage: python3 plot_heat.py
"""

import re
import textwrap
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image

from regions import REGIONS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'site'
WORLDMAP_JS = ROOT / 'site' / 'worldmap' / 'worldmap.js'

# The world map's ramps, class breaks (per million people per week) and colour for zero; checked against worldmap.js on every run.
RAMPS = {
    'cases': ['#cde2fb', '#a5c9f5', '#7bafee', '#5095e7', '#2c7ad8', '#2062b3', '#164b8f', '#0d366b'],
    'deaths': ['#fdd6c8', '#f6b49c', '#ed906e', '#e26a3c', '#cd4903', '#a83a00', '#832c02', '#611e01'],
}
BREAKS = {
    'cases': [3, 10, 30, 100, 300, 1000, 3000],
    'deaths': [0.3, 1, 3, 10, 30, 100, 300],
}
ZERO = '#e2e1dc'
HATCH = '#a9a7a0'

INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'

MIN_POPULATION = 100_000

NAMES = {
    'United_States_of_America': 'United States',
    'United_Republic_of_Tanzania': 'Tanzania',
    'Democratic_Republic_of_the_Congo': 'DR Congo',
    'Saint_Vincent_and_the_Grenadines': 'St Vincent & Grenadines',
    'Cote_dIvoire': "Côte d'Ivoire",
    'Sao_Tome_and_Principe': 'São Tomé and Príncipe',
    'Brunei_Darussalam': 'Brunei',
    'Guinea_Bissau': 'Guinea-Bissau',
    'Central_African_Republic': 'Central African Rep.',
}
# On the phone figure's narrower label column.
NARROW_NAMES = {
    'Saint_Vincent_and_the_Grenadines': 'St Vincent & Gren.',
    'Bosnia_and_Herzegovina': 'Bosnia & Herzegovina',
    'Sao_Tome_and_Principe': 'São Tomé & Príncipe',
}
TITLES = {'EU': 'the EU', 'Americas': 'the Americas', 'Nordic': 'the Nordic countries'}


@dataclass(frozen=True)
class Measure:
    name: str
    column: str


MEASURES = [Measure('cases', 'cases_weekly'), Measure('deaths', 'deaths_weekly')]


@dataclass(frozen=True)
class Layout:
    suffix: str
    width: float  # inches; one inch is 100 CSS pixels at the size the page shows
    label: float
    row: float
    dpi: int
    font: float


LAYOUTS = [
    Layout('', width=13.55, label=1.75, row=0.15, dpi=200, font=8.5),
    Layout('_narrow', width=4.15, label=1.25, row=0.12, dpi=300, font=7),
]


def check_worldmap() -> None:
    """Stop if worldmap.js no longer uses these ramps and breaks, so the two cannot drift apart."""
    js = WORLDMAP_JS.read_text(encoding='utf-8')
    for name in ('cases', 'deaths'):
        ramp = re.search(rf"\b{name}: \[([^\]]*)\]", js)
        breaks = re.search(rf"'{name} weekly': \[([^\]]*)\]", js)
        if ramp is None or breaks is None:
            raise SystemExit(f'Cannot find the {name} ramp or breaks in {WORLDMAP_JS}')
        if re.findall(r"'(#[0-9a-f]{6})'", ramp.group(1)) != RAMPS[name]:
            raise SystemExit(f'The {name} ramp differs from {WORLDMAP_JS}')
        if [float(v) for v in breaks.group(1).split(',')] != [float(v) for v in BREAKS[name]]:
            raise SystemExit(f'The {name} breaks differ from {WORLDMAP_JS}')
    if f"var ZERO = '{ZERO}'" not in js:
        raise SystemExit(f'The colour for zero differs from {WORLDMAP_JS}')


def monday(year_week: str) -> date:
    """The Monday that starts an ISO week written as ECDC writes it, '2020-53'."""
    year, week = (int(part) for part in year_week.split('-'))
    return date.fromisocalendar(year, week, 1)


@dataclass(frozen=True)
class Country:
    name: str
    population: int
    rates: dict[str, list[float | None]]  # per measure, one per week: per million people, or None for no report


def load() -> tuple[list[str], dict[str, Country]]:
    df = pd.read_csv(ROOT / 'ecdc.csv')
    weeks = sorted(str(week) for week in df['year_week'].unique())
    starts = [monday(week) for week in weeks]
    if any(b - a != timedelta(weeks=1) for a, b in pairwise(starts)):
        raise SystemExit('ecdc.csv skips a week')
    if df.duplicated(['countriesAndTerritories', 'year_week']).any():
        raise SystemExit('ecdc.csv has more than one row for a country and week')
    countries = {}
    for name, rows in df.groupby('countriesAndTerritories'):
        if rows['popData2019'].isna().to_numpy().any():
            continue  # only 'Cases_on_an_international_conveyance_Japan', which is in no region
        population = int(rows['popData2019'].iloc[0])
        by_week = rows.set_index('year_week')
        first = weeks.index(str(min(by_week.index)))
        rates: dict[str, list[float | None]] = {}
        for measure in MEASURES:
            values: list[float | None] = []
            for i, week in enumerate(weeks):
                if week in by_week.index:
                    values.append(float(by_week.at[week, measure.column]) / population * 1e6)
                else:
                    values.append(0.0 if i < first else None)
            rates[measure.name] = values
        countries[str(name)] = Country(str(name), population, rates)
    return weeks, countries


def display_name(country: str, narrow: bool = False) -> str:
    if narrow and country in NARROW_NAMES:
        return NARROW_NAMES[country]
    return NAMES.get(country, country.replace('_', ' '))


def colour(rate: float | None, measure: str) -> tuple[float, float, float]:
    if rate is None:
        return (1.0, 1.0, 1.0)
    if rate <= 0:
        return to_rgb(ZERO)
    k = sum(rate >= b for b in BREAKS[measure])
    return to_rgb(RAMPS[measure][k])


def number(value: float) -> str:
    return f'{value:,.0f}' if value >= 1 else f'{value:g}'


def heat(ax: Axes, rows: list[Country], measure: str, weeks: list[str], layout: Layout, labels: str) -> None:
    """One heat map: a row per country, a column per week."""
    grid = [[colour(rate, measure) for rate in c.rates[measure]] for c in rows]
    n, m = len(rows), len(weeks)
    ax.imshow(grid, aspect='auto', interpolation='nearest', extent=(0, m, n, 0))
    for i, c in enumerate(rows):
        for j, rate in enumerate(c.rates[measure]):
            if rate is None:
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor='white', edgecolor=HATCH, hatch='////', linewidth=0))
    ax.hlines([i for i in range(1, n)], 0, m, colors='white', linewidth=0.6 if layout.suffix == '' else 0.4)
    ax.set_xlim(0, m)
    ax.set_ylim(n, 0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_yticks([i + 0.5 for i in range(n)])
    ax.set_yticklabels([display_name(c.name, layout.suffix != '') for c in rows], fontsize=layout.font, color=INK)
    ax.tick_params(axis='y', length=0, pad=4)
    if labels == 'right':
        ax.yaxis.tick_right()
    elif labels == 'none':
        ax.set_yticklabels([])

    # Months: a short tick where each starts, the name under its middle.
    origin = monday(weeks[0])
    end = monday(weeks[-1]) + timedelta(weeks=1)
    starts = [date(2020, month, 1) for month in range(1, 13)] + [date(2021, 1, 1)]
    xs = [(d - origin).days / 7 for d in starts] + [(end - origin).days / 7]
    ax.set_xticks([x for x in xs[:-1] if 0 < x < m], minor=True)
    ax.tick_params(axis='x', which='minor', length=3, width=0.6, color=MUTED)
    every = 1 if layout.suffix == '' else 3
    centres, names = [], []
    for k, d in enumerate(starts):
        if k % every:
            continue
        left, right = max(xs[k], 0), min(xs[k + 1], m)
        if right - left < 1:
            continue
        centres.append((left + right) / 2)
        names.append(f'{d:%b}\n{d.year}' if d.month == 1 else f'{d:%b}')
    ax.set_xticks(centres)
    ax.set_xticklabels(names, fontsize=layout.font, color=MUTED, linespacing=1.15)
    ax.tick_params(axis='x', which='major', length=0, pad=3)


def legend(ax: Axes, measure: str, layout: Layout, any_missing: bool) -> None:
    """The classes as one bar of swatches, each break written under the join it marks, and a swatch for 0."""
    ax.set_xlim(0, 11 if any_missing else 9.4)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.add_patch(Rectangle((0, 0.55), 0.8, 0.4, facecolor=ZERO, linewidth=0))
    ax.text(0.4, 0.4, '0', ha='center', va='top', fontsize=layout.font, color=INK_2)
    for k, hex_colour in enumerate(RAMPS[measure]):
        ax.add_patch(Rectangle((1.2 + k, 0.55), 1, 0.4, facecolor=hex_colour, linewidth=0))
    for k, b in enumerate(BREAKS[measure]):
        ax.text(2.2 + k, 0.4, number(b), ha='center', va='top', fontsize=layout.font, color=INK_2)
    if any_missing:
        ax.add_patch(Rectangle((9.6, 0.55), 0.8, 0.4, facecolor='white', edgecolor=HATCH, hatch='////', linewidth=0))
        ax.text(10, 0.4, 'no report', ha='center', va='top', fontsize=layout.font, color=INK_2)


def place(fig: Figure, left: float, top: float, width: float, height: float, size: tuple[float, float]) -> Axes:
    """An axes at a position given in inches from the figure's top left corner."""
    w, h = size
    return fig.add_axes((left / w, 1 - (top + height) / h, width / w, height / h))


def draw(region: str, rows: list[Country], left_out: list[Country], weeks: list[str], layout: Layout) -> Path:
    narrow = layout.suffix != ''
    n = len(rows)
    any_missing = any(rate is None for c in rows for m in MEASURES for rate in c.rates[m.name])
    panel_h = n * layout.row
    first = weeks[0]
    last_day = monday(weeks[-1]) + timedelta(days=6)
    where = TITLES.get(region, region)
    left_names = ', '.join(display_name(c.name) for c in left_out)
    if narrow:
        title = f'Weekly COVID-19 cases and deaths\nper million people in {where}'
        subtitle = (f'One row per country, one column per week,\n{monday(first):%-d %b %Y} to {last_day:%-d %b %Y}. '
                    'Colours as on\nthe world map.')
        paragraphs = ['Data: ECDC, weekly, populations of 2019. Before its first report a country counts as 0.']
        if left_out:
            paragraphs.append(f'Left out, under 100,000 people: {left_names}.')
        paragraphs.append('Drawn October 2026.')
    else:
        title = f'Weekly COVID-19 cases and deaths per million people in {where}'
        subtitle = (f'One row per country, one column per week, from {monday(first):%-d %B %Y} to {last_day:%-d %B %Y} '
                    f'(ISO weeks {first} to {weeks[-1]}). The colours are those of the world map.')
        paragraphs = [('Data: European Centre for Disease Prevention and Control (ECDC), weekly cases and deaths by '
                       'country, per million people of 2019. A country counts as 0 in the weeks before its first report.'),
                      (f'Left out, with fewer than 100,000 people: {left_names}. ' if left_out else '')
                      + 'Drawn October 2026.']
    source_size = 8 if not narrow else 7.5
    source = '\n'.join(textwrap.fill(p, 62 if narrow else 210) for p in paragraphs)
    footer = 0.25 + (source.count('\n') + 1) * source_size * 1.35 / 72

    title_h, axis_h, legend_h = 0.32, (0.42 if not narrow else 0.36), 0.42
    header = 1.0 if not narrow else 1.55
    block = title_h + panel_h + axis_h + legend_h
    if narrow:
        panel_w = layout.width - layout.label - 0.1
        height = header + 2 * block + 0.2 + footer
    else:
        panel_w = (layout.width - 2 * layout.label - 0.35) / 2
        height = header + block + footer
    size = (layout.width, height)
    fig = plt.figure(figsize=size, dpi=100, facecolor='white')

    for k, measure in enumerate(MEASURES):
        if narrow:
            x, top, labels = layout.label, header + k * (block + 0.2), 'left'
        else:
            x = layout.label if k == 0 else layout.label + panel_w + 0.35
            top, labels = header, ('left' if k == 0 else 'right')
        ax = place(fig, x, top + title_h, panel_w, panel_h, size)
        heat(ax, rows, measure.name, weeks, layout, labels)
        fig.text(x / layout.width, 1 - (top + 0.05) / height, f'{measure.name.capitalize()} per million people, per week',
                 ha='left', va='top', fontsize=10 if not narrow else 8.5, fontweight='bold', color=INK)
        key = place(fig, x, top + title_h + panel_h + axis_h, min(panel_w, 4.6), legend_h, size)
        legend(key, measure.name, layout, any_missing)

    edge = 0.12 / layout.width
    fig.text(edge, 1 - 0.12 / height, title, ha='left', va='top', fontsize=13 if not narrow else 11.5,
             fontweight='bold', color=INK, linespacing=1.2)
    fig.text(edge, 1 - (0.5 if not narrow else 0.68) / height, subtitle, ha='left', va='top',
             fontsize=9.5 if not narrow else 8.5, color=INK_2, linespacing=1.35)
    fig.text(edge, 0.1 / height, source, ha='left', va='bottom', fontsize=source_size, color=MUTED, linespacing=1.35)

    clipped = escaping_text(fig)
    if clipped:
        raise SystemExit(f'heat_{region}{layout.suffix}: text runs off the figure: {clipped}')
    path = OUT / f'heat_{region}{layout.suffix}.png'
    fig.savefig(path, dpi=layout.dpi, facecolor='white')
    plt.close(fig)
    return path


def escaping_text(fig: Figure) -> list[str]:
    """Every visible piece of text, tick labels included, that reaches past an edge of the figure."""
    fig.canvas.draw()
    box = fig.bbox
    texts = list(fig.texts)
    for ax in fig.axes:
        texts += ax.texts
        if ax.axison:
            texts += [*ax.get_xticklabels(), *ax.get_yticklabels()]
    out = []
    for t in texts:
        if not t.get_visible() or not t.get_text():
            continue
        e = t.get_window_extent()
        if e.x0 < box.x0 - 0.5 or e.x1 > box.x1 + 0.5 or e.y0 < box.y0 - 0.5 or e.y1 > box.y1 + 0.5:
            out.append(t.get_text().replace('\n', ' '))
    return out


def css_size(path: Path, dpi: int) -> tuple[int, int]:
    """The image's size in CSS pixels: 100 per inch, as the figures are laid out."""
    with Image.open(path) as im:
        return round(im.size[0] * 100 / dpi), round(im.size[1] * 100 / dpi)


def peak(rows: list[Country], measure: str, weeks: list[str]) -> str:
    rate, j, c = max(((r, j, c) for c in rows for j, r in enumerate(c.rates[measure]) if r is not None),
                     key=lambda t: t[0])
    return f'{display_name(c.name)}, {rate:,.0f} per million in the week of {monday(weeks[j]):%-d %B %Y}'


def picture(region: str, rows: list[Country], weeks: list[str], paths: dict[str, Path]) -> str:
    """A <picture> that serves the stacked figure below 700 CSS pixels."""
    wide, narrow = LAYOUTS
    w, h = css_size(paths[wide.suffix], wide.dpi)
    nw, nh = css_size(paths[narrow.suffix], narrow.dpi)
    alt = (f'Heat maps of weekly COVID-19 cases and deaths per million people in {TITLES.get(region, region)}, '
           f'one row per country and one column per week from {monday(weeks[0]):%-d %B %Y} to January 2021. '
           f'Highest weekly cases: {peak(rows, "cases", weeks)}. Highest weekly deaths: {peak(rows, "deaths", weeks)}. '
           f'Countries: {", ".join(display_name(c.name) for c in rows)}.')
    return (f'<picture>\n'
            f'<source media="(max-width: 700px)" srcset="{paths[narrow.suffix].name}" width="{nw}" height="{nh}">\n'
            f'<img src="{paths[wide.suffix].name}" width="{w}" height="{h}" loading="lazy" alt="{alt}">\n'
            f'</picture>')


def main() -> None:
    matplotlib.use('Agg')
    plt.rcParams['hatch.color'] = HATCH
    plt.rcParams['hatch.linewidth'] = 0.6
    check_worldmap()
    weeks, countries = load()
    print(f'ecdc.csv: {len(countries)} countries, ISO weeks {weeks[0]} to {weeks[-1]}. '
          f'Drawing {len(REGIONS) * len(LAYOUTS)} figures into {OUT.relative_to(ROOT)}/')
    pictures = []
    for region, names in REGIONS.items():
        members = sorted((countries[c] for c in names), key=lambda c: display_name(c.name))
        rows = [c for c in members if c.population >= MIN_POPULATION]
        left_out = [c for c in members if c.population < MIN_POPULATION]
        paths = {layout.suffix: draw(region, rows, left_out, weeks, layout) for layout in LAYOUTS}
        sizes = ', '.join(f'{p.name} {css_size(p, lay.dpi)[0]}x{css_size(p, lay.dpi)[1]} CSS px'
                          for lay, p in zip(LAYOUTS, paths.values()))
        print(f'  {region}: {len(rows)} countries, {len(left_out)} left out; {sizes}')
        pictures.append(f'<!-- {region} -->\n' + picture(region, rows, weeks, paths))
    markup = ROOT / 'tmp' / 'plot_heat_markup.html'
    markup.parent.mkdir(exist_ok=True)
    markup.write_text('\n'.join(pictures) + '\n', encoding='utf-8')
    print(f'Done. <picture> tags for site/index.html are in {markup.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
