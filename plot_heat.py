"""Draw the page's regional heat maps, one fixed log scale per measure: weekly COVID-19 cases and deaths per million people, and weekly tests per thousand people beside the share of tests that were positive.

The 2020 heat maps (plot_heat_{cases,deaths}_<region>.png, drawn by plot_series.doHeatMaps) used matplotlib's OrRd on a linear scale rescaled for every chart, so the same colour meant different rates in different charts and a region's one big wave washed out everything else; their country names were too small to read; they averaged seven weekly rows, a leftover from ECDC's daily data; and the EU chart left out Czechia. These keep OrRd, but on one logarithmic scale per measure shared by every region (cases 1 to 10,000 per million people per week, deaths 0.1 to 1,000; a rate outside it takes the colour of its end), with a light grey for 0, and show single weeks. Rows run from the largest population to the smallest, with each population beside the name, so the rows where one case or death is a large rate per million sit together at the bottom and say so. On wide screens the two maps of a figure sit side by side with the names between them; on phones they are stacked.

The tests heat maps (October 2026) show whether more cases came from more testing. Tests per thousand people per week are in Blues, the colour map of plot_choropleth.py's weekly tests map, on 0.01 to 100; the share of those tests that were positive, a week's cases divided by its tests, is in the cases' OrRd, on 0.1% to 100%. The tests are build_world_map.read_owid_tests', as on the page's world maps. Only places with a test count in some week get a row; the figure names the others.

Reads ecdc.csv (ECDC weekly cases and deaths to ISO week 2021-01, with ECDC's populations of 2019), owid.csv (Our World in Data's tests) and the regions in regions.py, and downloads nothing. A week before a place's first report counts as 0 cases or deaths, as does a week whose count ECDC revised below 0; a missing week after the first report is hatched, as is a week without a test count. Places of fewer than 100,000 people are left out, because one case there is ten or more per million; the figure names them.

Writes site/heat_<region>.png and site/heat_tests_<region>.png (wide screens, 2x pixel density), each with a _narrow.png (phones, 3x), and their <picture> tags to tmp/plot_heat_markup.html and tmp/plot_heat_tests_markup.html. Stops if any text runs off a figure or a name runs into its population.

Usage: python3 plot_heat.py
"""

import math
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from functools import cached_property
from itertools import pairwise
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, LogNorm, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import blended_transform_factory
from PIL import Image

from build_world_map import read_owid_tests
from regions import REGIONS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'site'

# plot_series.doHeatMaps' OrRd, and the weekly tests map's Blues, each without its palest fifth, so the lowest value stays apart from the grey for 0.
RAMP = ListedColormap(matplotlib.colormaps['OrRd'](np.linspace(0.2, 1.0, 256)))
BLUES = ListedColormap(matplotlib.colormaps['Blues'](np.linspace(0.2, 1.0, 256)))
ZERO = '#f1f0ed'
HATCH = '#a9a7a0'

INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#706f6a'  # 5.0:1 on white

MIN_POPULATION = 100_000
EDGE = 0.12  # inches of margin at the figure's left and right

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
    'Timor_Leste': 'Timor-Leste',
    'Micronesia_(Federated_States_of)': 'Micronesia',
    'United_States_Virgin_Islands': 'US Virgin Islands',
    'Turks_and_Caicos_islands': 'Turks and Caicos Islands',
}
# On the phone figure's narrower label column.
NARROW_NAMES = {
    'Saint_Vincent_and_the_Grenadines': 'St Vincent & Gren.',
    'Bosnia_and_Herzegovina': 'Bosnia & Herzegovina',
    'Sao_Tome_and_Principe': 'São Tomé & Príncipe',
}
TITLES = {'EU': 'the EU', 'Americas': 'the Americas', 'Nordic': 'the Nordic countries'}


def number(value: float) -> str:
    return f'{value:,.0f}' if value >= 1 else f'{value:g}'


def percent(value: float) -> str:
    return f'{value * 100:g}%'


@dataclass(frozen=True)
class Measure:
    name: str  # its key in Country.rates
    title: str  # over its map
    ramp: ListedColormap
    lo: float  # the ends of its log scale, the same in every region
    hi: float
    label: Callable[[float], str] = number

    @cached_property
    def norm(self) -> LogNorm:
        return LogNorm(self.lo, self.hi, clip=True)


CASES = Measure('cases', 'Cases per million people, per week', RAMP, 1.0, 10_000.0)
DEATHS = Measure('deaths', 'Deaths per million people, per week', RAMP, 0.1, 1_000.0)
TESTS = Measure('tests', 'Tests per thousand people, per week', BLUES, 0.01, 100.0)
POSITIVE = Measure('positive', 'Share of tests positive', RAMP, 0.001, 1.0, percent)


@dataclass(frozen=True)
class Layout:
    suffix: str
    width: float  # inches; one inch is 100 CSS pixels at the size the page shows
    row: float
    dpi: int
    font: float


LAYOUTS = [
    Layout('', width=13.55, row=0.15, dpi=200, font=8.5),
    Layout('_narrow', width=4.15, row=0.12, dpi=300, font=7),
]


def monday(year_week: str) -> date:
    """The Monday that starts an ISO week written as ECDC writes it, '2020-53'."""
    year, week = (int(part) for part in year_week.split('-'))
    return date.fromisocalendar(year, week, 1)


@dataclass(frozen=True)
class Country:
    name: str
    population: int
    rates: dict[str, list[float | None]]  # per measure, one per week, or None for no report


def share_positive(cases: float | None, tests: float | None) -> float | None:
    """A week's cases per million people over its tests per thousand people, as a share."""
    if cases is None or tests is None or tests <= 0:
        return None
    return cases / (tests * 1000)


def load() -> tuple[list[str], dict[str, Country]]:
    df = pd.read_csv(ROOT / 'ecdc.csv')
    weeks = sorted(str(week) for week in df['year_week'].unique())
    starts = [monday(week) for week in weeks]
    if any(b - a != timedelta(weeks=1) for a, b in pairwise(starts)):
        raise SystemExit('ecdc.csv skips a week')
    if df.duplicated(['countriesAndTerritories', 'year_week']).any():
        raise SystemExit('ecdc.csv has more than one row for a country and week')
    tests = read_owid_tests(weeks)
    countries = {}
    for name, rows in df.groupby('countriesAndTerritories'):
        if rows['popData2019'].isna().to_numpy().any():
            continue  # Wallis and Futuna and the cases on a ship off Japan, which regions.py keeps out of every region
        population = int(rows['popData2019'].iloc[0])
        by_week = rows.set_index('year_week')
        first = weeks.index(str(min(by_week.index)))
        rates: dict[str, list[float | None]] = {}
        for measure, column in ((CASES, 'cases_weekly'), (DEATHS, 'deaths_weekly')):
            values: list[float | None] = []
            for i, week in enumerate(weeks):
                if week in by_week.index:
                    values.append(float(by_week.at[week, column]) / population * 1e6)
                else:
                    values.append(0.0 if i < first else None)
            rates[measure.name] = values
        code = str(rows['countryterritoryCode'].iloc[0])
        rates[TESTS.name] = list(tests.get(code, {}).get('tests', [None] * len(weeks)))
        rates[POSITIVE.name] = [share_positive(c, t) for c, t in zip(rates[CASES.name], rates[TESTS.name])]
        countries[str(name)] = Country(str(name), population, rates)
    return weeks, countries


def display_name(country: str, narrow: bool = False) -> str:
    if narrow and country in NARROW_NAMES:
        return NARROW_NAMES[country]
    return NAMES.get(country, country.replace('_', ' '))


def millions(population: int) -> str:
    m = population / 1e6
    return f'{m:,.0f} M' if m >= 10 else f'{m:.1f} M' if m >= 1 else f'{m:.2f} M'


def colour(rate: float | None, measure: Measure) -> tuple[float, float, float]:
    if rate is None:
        return (1.0, 1.0, 1.0)
    if rate <= 0:
        return to_rgb(ZERO)
    r, g, b, _ = measure.ramp(float(measure.norm(rate)))
    return (r, g, b)


@dataclass(frozen=True)
class Texts:
    title: str
    subtitle: str
    paragraphs: list[str]  # the source, one string per paragraph


@dataclass(frozen=True)
class Chart:
    prefix: str  # of its file names
    measures: tuple[Measure, Measure]
    missing: str  # under the hatched swatch
    missing_x: float  # where the hatched swatch starts on the key, which ends at key_end
    key_end: float
    texts: Callable[[bool, str, list[str], str, str], Texts]  # narrow, where, weeks, places left out, places without a test count


def cases_texts(narrow: bool, where: str, weeks: list[str], left_names: str, _without: str) -> Texts:
    first, last_day = monday(weeks[0]), monday(weeks[-1]) + timedelta(days=6)
    if narrow:
        paragraphs = ['Data: ECDC, weekly, populations of 2019. Before its first report a place counts as 0.']
        if left_names:
            paragraphs.append(f'Left out, under 100,000 people: {left_names}.')
        paragraphs.append('Drawn October 2026.')
        return Texts(f'Weekly COVID-19 cases and deaths\nper million people in {where}', textwrap.fill(
            f'One row per country or territory, largest population first, one column per week, '
            f'{first:%-d %b %Y} to {last_day:%-d %b %Y}. The colour scale is logarithmic and the same in '
            'every region.', 52), paragraphs)
    return Texts(f'Weekly COVID-19 cases and deaths per million people in {where}',
                 f'One row per country or territory, largest population first, one column per week, from '
                 f'{first:%-d %B %Y} to {last_day:%-d %B %Y} (ISO weeks {weeks[0]} to {weeks[-1]}).\n'
                 'The colour scale is logarithmic and the same in every region.',
                 [('Data: European Centre for Disease Prevention and Control (ECDC), weekly cases and deaths by '
                   'country, per million people of 2019. A place counts as 0 in the weeks before its first report.'),
                  (f'Left out, with fewer than 100,000 people: {left_names}. ' if left_names else '')
                  + 'Drawn October 2026.'])


def tests_texts(narrow: bool, where: str, weeks: list[str], left_names: str, without: str) -> Texts:
    first, last_day = monday(weeks[0]), monday(weeks[-1]) + timedelta(days=6)
    if narrow:
        paragraphs = [('Data: Our World in Data, tests; ECDC, weekly cases. Countries count tests differently, so the '
                       'shares compare better along a row than between rows.')]
        if without:
            paragraphs.append(f'No test count: {without}.')
        if left_names:
            paragraphs.append(f'Left out, under 100,000 people: {left_names}.')
        paragraphs.append('Drawn October 2026.')
        return Texts(f'Weekly COVID-19 tests, and the share\nthat were positive, in {where}', textwrap.fill(
            f'One row per country or territory with a test count, largest population first, one column per week, '
            f'{first:%-d %b %Y} to {last_day:%-d %b %Y}. The share positive is a week\'s cases divided by its tests. '
            'Both colour scales are logarithmic and the same in every region.', 52), paragraphs)
    return Texts(f'Weekly COVID-19 tests, and the share that were positive, in {where}',
                 f'One row per country or territory with a test count, largest population first, one column per week, '
                 f'from {first:%-d %B %Y} to {last_day:%-d %B %Y} (ISO weeks {weeks[0]} to {weeks[-1]}).\n'
                 'The share positive is a week\'s cases divided by its tests. Both colour scales are logarithmic and the '
                 'same in every region.',
                 [('Data: Our World in Data, tests, from running totals interpolated between the days a country reported '
                   'them, or from daily counts where it has no running total; European Centre for Disease Prevention and '
                   'Control (ECDC), weekly cases. Countries count tests differently (tests performed, people tested, '
                   'samples), so the shares compare better along a row than between rows.'),
                  (f'No test count: {without}. ' if without else '')
                  + (f'Left out, with fewer than 100,000 people: {left_names}. ' if left_names else '')
                  + 'Drawn October 2026.'])


CHARTS = [
    Chart('heat', (CASES, DEATHS), 'no report', 9.2, 10.6, cases_texts),
    Chart('heat_tests', (TESTS, POSITIVE), 'no test count', 10.4, 12.2, tests_texts),
]


def heat(ax: Axes, rows: list[Country], measure: Measure, weeks: list[str], layout: Layout) -> None:
    """One heat map: a row per place, a column per week."""
    grid = [[colour(rate, measure) for rate in c.rates[measure.name]] for c in rows]
    n, m = len(rows), len(weeks)
    ax.imshow(grid, aspect='auto', interpolation='nearest', extent=(0, m, n, 0))
    for i, c in enumerate(rows):
        for j, rate in enumerate(c.rates[measure.name]):
            if rate is None:
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor='white', edgecolor=HATCH, hatch='////', linewidth=0))
    ax.hlines([i for i in range(1, n)], 0, m, colors='white', linewidth=0.6 if layout.suffix == '' else 0.4)
    ax.set_xlim(0, m)
    ax.set_ylim(n, 0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_yticks([])

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


def row_labels(fig: Figure, ax: Axes, rows: list[Country], name_x: float, pop_x: float,
               layout: Layout) -> list[tuple[Text, Text]]:
    """Each row's name, starting at name_x, and population, ending at pop_x (inches from the left), with a heading over the populations."""
    width = fig.get_figwidth()
    where = blended_transform_factory(fig.transFigure, ax.transData)
    narrow = layout.suffix != ''
    pairs = []
    for i, c in enumerate(rows):
        name = ax.text(name_x / width, i + 0.5, display_name(c.name, narrow), transform=where, ha='left',
                       va='center', fontsize=layout.font, color=INK)
        pop = ax.text(pop_x / width, i + 0.5, millions(c.population), transform=where, ha='right', va='center',
                      fontsize=layout.font, color=INK_2)
        pairs.append((name, pop))
    ax.text(pop_x / width, -0.25, 'population', transform=where, ha='right', va='bottom',
            fontsize=layout.font - 1, color=MUTED)
    return pairs


def legend(ax: Axes, measure: Measure, chart: Chart, layout: Layout, any_missing: bool) -> None:
    """A swatch for 0, then the colour scale as one bar on a log axis, labelled at every power of ten."""
    start, end = 1.0, 9.0
    ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=measure.ramp, aspect='auto', interpolation='bilinear',
              extent=(start, end, 0.55, 0.95))
    ax.set_xlim(0, chart.key_end if any_missing else end + 0.5)  # room for half the last label
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.add_patch(Rectangle((0, 0.55), 0.8, 0.4, facecolor=ZERO, linewidth=0))
    ax.text(0.4, 0.4, '0', ha='center', va='top', fontsize=layout.font, color=INK_2)
    decades = round(math.log10(measure.hi / measure.lo))
    for k in range(decades + 1):
        x = start + (end - start) * k / decades
        ax.plot([x, x], [0.47, 0.55], color=MUTED, linewidth=0.6)
        ax.text(x, 0.4, measure.label(measure.lo * 10 ** k), ha='center', va='top', fontsize=layout.font, color=INK_2)
    if any_missing:
        x = chart.missing_x
        ax.add_patch(Rectangle((x, 0.55), 0.8, 0.4, facecolor='white', edgecolor=HATCH, hatch='////', linewidth=0))
        ax.text(x + 0.4, 0.4, chart.missing, ha='center', va='top', fontsize=layout.font, color=INK_2)


def place(fig: Figure, left: float, top: float, width: float, height: float, size: tuple[float, float]) -> Axes:
    """An axes at a position given in inches from the figure's top left corner."""
    w, h = size
    return fig.add_axes((left / w, 1 - (top + height) / h, width / w, height / h))


def text_height(text: str, size: float, spacing: float) -> float:
    """Inches taken by a block of text of this many lines."""
    return (text.count('\n') + 1) * size * spacing / 72


def widest(fig: Figure, texts: list[str], size: float) -> float:
    """The width in inches of the widest of these texts at this size."""
    widths = []
    for s in texts:
        t = fig.text(0, 0, s, fontsize=size)
        widths.append(t.get_window_extent().width / fig.dpi)
        t.remove()
    return max(widths)


def draw(chart: Chart, region: str, rows: list[Country], left_out: list[Country], without: list[Country],
         weeks: list[str], layout: Layout) -> Path:
    narrow = layout.suffix != ''
    n = len(rows)
    any_missing = any(rate is None for c in rows for m in chart.measures for rate in c.rates[m.name])
    panel_h = n * layout.row
    where = TITLES.get(region, region)
    texts = chart.texts(narrow, where, weeks, ', '.join(display_name(c.name) for c in left_out),
                        ', '.join(display_name(c.name) for c in without))
    title, subtitle = texts.title, texts.subtitle
    title_size, subtitle_size = (13, 9.5) if not narrow else (11.5, 8.5)
    source_size = 8 if not narrow else 7.5
    source = '\n'.join(textwrap.fill(p, 62 if narrow else 210) for p in texts.paragraphs)
    subtitle_top = 0.12 + text_height(title, title_size, 1.2) + 0.12
    header = subtitle_top + text_height(subtitle, subtitle_size, 1.35) + 0.12
    footer = 0.3 + text_height(source, source_size, 1.35)

    title_h, axis_h, legend_h = 0.32, (0.42 if not narrow else 0.36), 0.42
    block = title_h + panel_h + axis_h + legend_h
    height = header + (2 * block + 0.2 if narrow else block) + footer
    size = (layout.width, height)
    fig = plt.figure(figsize=size, dpi=100, facecolor='white')

    # The label columns are as wide as this figure's longest name and population, and the maps take the rest.
    names_w = widest(fig, [display_name(c.name, narrow) for c in rows], layout.font)
    pops_w = max(widest(fig, [millions(c.population) for c in rows], layout.font),
                 widest(fig, ['population'], layout.font - 1))
    labels_w = names_w + 0.15 + pops_w
    pad = 0.1
    if narrow:
        panel_x = EDGE + labels_w + 0.08
        panel_w = layout.width - panel_x - EDGE
        xs = [panel_x, panel_x]
        tops = [header, header + block + 0.2]
    else:
        middle = pad + labels_w + pad
        panel_w = (layout.width - 2 * EDGE - middle) / 2
        xs = [EDGE, EDGE + panel_w + middle]
        tops = [header, header]

    pairs = []
    for k, measure in enumerate(chart.measures):
        x, top = xs[k], tops[k]
        ax = place(fig, x, top + title_h, panel_w, panel_h, size)
        heat(ax, rows, measure, weeks, layout)
        if narrow:
            pairs += row_labels(fig, ax, rows, EDGE, x - 0.08, layout)
        elif k == 0:
            pairs += row_labels(fig, ax, rows, x + panel_w + pad, xs[1] - pad, layout)
        fig.text((EDGE if narrow else x) / layout.width, 1 - (top + 0.05) / height, measure.title,
                 ha='left', va='top', fontsize=10 if not narrow else 8.5, fontweight='bold', color=INK)
        key_x, key_w = (EDGE, layout.width - 2 * EDGE) if narrow else (x, min(panel_w, 4.6))
        key = place(fig, key_x, top + title_h + panel_h + axis_h, key_w, legend_h, size)
        legend(key, measure, chart, layout, any_missing)

    edge = EDGE / layout.width
    fig.text(edge, 1 - 0.12 / height, title, ha='left', va='top', fontsize=title_size, fontweight='bold', color=INK,
             linespacing=1.2)
    fig.text(edge, 1 - subtitle_top / height, subtitle, ha='left', va='top', fontsize=subtitle_size, color=INK_2,
             linespacing=1.35)
    fig.text(edge, 0.1 / height, source, ha='left', va='bottom', fontsize=source_size, color=MUTED, linespacing=1.35)

    name = f'{chart.prefix}_{region}{layout.suffix}'
    clipped = escaping_text(fig)
    if clipped:
        raise SystemExit(f'{name}: text runs off the figure: {clipped}')
    crowded = [label.get_text() for label, pop in pairs
               if label.get_window_extent().x1 + 4 > pop.get_window_extent().x0]
    if crowded:
        raise SystemExit(f'{name}: names run into their populations: {crowded}')
    path = OUT / f'{name}.png'
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


def peak(rows: list[Country], measure: Measure, weeks: list[str]) -> str:
    rate, j, c = max(((r, j, c) for c in rows for j, r in enumerate(c.rates[measure.name]) if r is not None),
                     key=lambda t: t[0])
    return f'{display_name(c.name)}, {rate:,.0f} per million in the week of {monday(weeks[j]):%-d %B %Y}'


def alt(chart: Chart, region: str, rows: list[Country], without: list[Country], weeks: list[str]) -> str:
    where = TITLES.get(region, region)
    names = ', '.join(display_name(c.name) for c in rows)
    if chart.prefix == 'heat':
        return (f'Heat maps of weekly COVID-19 cases and deaths per million people in {where}, '
                f'one row per country or territory, largest population first, and one column per week from '
                f'{monday(weeks[0]):%-d %B %Y} to January 2021. '
                f'Highest weekly cases: {peak(rows, CASES, weeks)}. Highest weekly deaths: {peak(rows, DEATHS, weeks)}. '
                f'Rows: {names}.')
    missing = f' No test count: {", ".join(display_name(c.name) for c in without)}.' if without else ''
    return (f'Heat maps of weekly COVID-19 tests per thousand people, and of the share of tests that were positive, '
            f'in {where}, one row per country or territory with a test count, largest population first, and one '
            f'column per week from {monday(weeks[0]):%-d %B %Y} to January 2021. Rows: {names}.{missing}')


def picture(chart: Chart, region: str, rows: list[Country], without: list[Country], weeks: list[str],
            paths: dict[str, Path]) -> str:
    """A <picture> that serves the stacked figure below 700 CSS pixels."""
    wide, narrow = LAYOUTS
    w, h = css_size(paths[wide.suffix], wide.dpi)
    nw, nh = css_size(paths[narrow.suffix], narrow.dpi)
    return (f'<picture>\n'
            f'<source media="(max-width: 700px)" srcset="{paths[narrow.suffix].name}" width="{nw}" height="{nh}">\n'
            f'<img src="{paths[wide.suffix].name}" width="{w}" height="{h}" loading="lazy" '
            f'alt="{alt(chart, region, rows, without, weeks)}">\n'
            f'</picture>')


def main() -> None:
    matplotlib.use('Agg')
    plt.rcParams['hatch.color'] = HATCH
    plt.rcParams['hatch.linewidth'] = 0.6
    weeks, countries = load()
    print(f'ecdc.csv: {len(countries)} places, ISO weeks {weeks[0]} to {weeks[-1]}. '
          f'Drawing {len(CHARTS) * len(REGIONS) * len(LAYOUTS)} figures into {OUT.relative_to(ROOT)}/')
    for chart in CHARTS:
        pictures = []
        for region, names in REGIONS.items():
            members = sorted((countries[c] for c in names), key=lambda c: -c.population)
            rows = [c for c in members if c.population >= MIN_POPULATION]
            left_out = sorted((c for c in members if c.population < MIN_POPULATION), key=lambda c: display_name(c.name))
            without: list[Country] = []
            if TESTS in chart.measures:
                without = sorted((c for c in rows if all(v is None for v in c.rates[TESTS.name])),
                                 key=lambda c: display_name(c.name))
                rows = [c for c in rows if any(v is not None for v in c.rates[TESTS.name])]
            paths = {layout.suffix: draw(chart, region, rows, left_out, without, weeks, layout) for layout in LAYOUTS}
            sizes = ', '.join(f'{p.name} {css_size(p, lay.dpi)[0]}x{css_size(p, lay.dpi)[1]} CSS px'
                              for lay, p in zip(LAYOUTS, paths.values()))
            print(f'  {region}: {len(rows)} rows, {len(left_out)} left out, {len(without)} without tests; {sizes}')
            pictures.append(f'<!-- {region} -->\n' + picture(chart, region, rows, without, weeks, paths))
        markup = ROOT / 'tmp' / f'plot_{chart.prefix}_markup.html'
        markup.parent.mkdir(exist_ok=True)
        markup.write_text('\n'.join(pictures) + '\n', encoding='utf-8')
        print(f'Done. <picture> tags for site/index.html are in {markup.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
