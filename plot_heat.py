"""Draw the page's regional heat maps of weekly COVID-19 cases and deaths per million people, in Tommy's 2020 OrRd on one fixed log scale.

The 2020 heat maps (plot_heat_{cases,deaths}_<region>.png, drawn by plot_series.doHeatMaps) used matplotlib's OrRd on a linear scale rescaled for every chart, so the same colour meant different rates in different charts and a region's one big wave washed out everything else; their country names were too small to read; they averaged seven weekly rows, a leftover from ECDC's daily data; and the EU chart left out Czechia. These keep OrRd, but on one logarithmic scale per measure shared by every region (cases 1 to 10,000 per million people per week, deaths 0.1 to 1,000; a rate outside it takes the colour of its end), with a light grey for 0, and show single weeks. Rows run from the largest population to the smallest, with each population beside the name, so the rows where one case or death is a large rate per million sit together at the bottom and say so. On wide screens cases and deaths sit side by side with the names between them; on phones the two maps are stacked.

Reads ecdc.csv (ECDC weekly cases and deaths to ISO week 2021-01, with ECDC's populations of 2019) and the regions in regions.py, and downloads nothing. A week before a place's first report counts as 0, as does a week whose count ECDC revised below 0; a missing week after the first report would be hatched, but ecdc.csv has none for these places. Places of fewer than 100,000 people are left out, because one case there is ten or more per million; the figure names them.

Writes site/heat_<region>.png (wide screens, 2x pixel density) and site/heat_<region>_narrow.png (phones, 3x), and their <picture> tags to tmp/plot_heat_markup.html. Stops if any text runs off a figure or a name runs into its population.

Usage: python3 plot_heat.py
"""

import math
import textwrap
from dataclasses import dataclass
from datetime import date, timedelta
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

from regions import REGIONS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'site'

# plot_series.doHeatMaps' OrRd, without its palest fifth, so the lowest rate stays apart from the grey for 0.
RAMP = ListedColormap(matplotlib.colormaps['OrRd'](np.linspace(0.2, 1.0, 256)))
# Per million people per week: the ends of each measure's log scale, the same in every region.
SCALES = {'cases': (1.0, 10_000.0), 'deaths': (0.1, 1_000.0)}
NORMS = {measure: LogNorm(lo, hi, clip=True) for measure, (lo, hi) in SCALES.items()}
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


@dataclass(frozen=True)
class Measure:
    name: str
    column: str


MEASURES = [Measure('cases', 'cases_weekly'), Measure('deaths', 'deaths_weekly')]


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
            continue  # Wallis and Futuna and the cases on a ship off Japan, which regions.py keeps out of every region
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


def millions(population: int) -> str:
    m = population / 1e6
    return f'{m:,.0f} M' if m >= 10 else f'{m:.1f} M' if m >= 1 else f'{m:.2f} M'


def colour(rate: float | None, measure: str) -> tuple[float, float, float]:
    if rate is None:
        return (1.0, 1.0, 1.0)
    if rate <= 0:
        return to_rgb(ZERO)
    r, g, b, _ = RAMP(float(NORMS[measure](rate)))
    return (r, g, b)


def number(value: float) -> str:
    return f'{value:,.0f}' if value >= 1 else f'{value:g}'


def heat(ax: Axes, rows: list[Country], measure: str, weeks: list[str], layout: Layout) -> None:
    """One heat map: a row per place, a column per week."""
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


def legend(ax: Axes, measure: str, layout: Layout, any_missing: bool) -> None:
    """A swatch for 0, then the colour scale as one bar on a log axis, labelled at every power of ten."""
    lo, hi = SCALES[measure]
    start, end = 1.0, 9.0
    ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=RAMP, aspect='auto', interpolation='bilinear',
              extent=(start, end, 0.55, 0.95))
    ax.set_xlim(0, 10.6 if any_missing else end + 0.5)  # room for half the last label
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.add_patch(Rectangle((0, 0.55), 0.8, 0.4, facecolor=ZERO, linewidth=0))
    ax.text(0.4, 0.4, '0', ha='center', va='top', fontsize=layout.font, color=INK_2)
    decades = round(math.log10(hi / lo))
    for k in range(decades + 1):
        x = start + (end - start) * k / decades
        ax.plot([x, x], [0.47, 0.55], color=MUTED, linewidth=0.6)
        ax.text(x, 0.4, number(lo * 10 ** k), ha='center', va='top', fontsize=layout.font, color=INK_2)
    if any_missing:
        ax.add_patch(Rectangle((9.2, 0.55), 0.8, 0.4, facecolor='white', edgecolor=HATCH, hatch='////', linewidth=0))
        ax.text(9.6, 0.4, 'no report', ha='center', va='top', fontsize=layout.font, color=INK_2)


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
        subtitle = textwrap.fill(
            f'One row per country or territory, largest population first, one column per week, '
            f'{monday(first):%-d %b %Y} to {last_day:%-d %b %Y}. The colour scale is logarithmic and the same in '
            'every region.', 52)
        paragraphs = ['Data: ECDC, weekly, populations of 2019. Before its first report a place counts as 0.']
        if left_out:
            paragraphs.append(f'Left out, under 100,000 people: {left_names}.')
        paragraphs.append('Drawn October 2026.')
    else:
        title = f'Weekly COVID-19 cases and deaths per million people in {where}'
        subtitle = (f'One row per country or territory, largest population first, one column per week, from '
                    f'{monday(first):%-d %B %Y} to {last_day:%-d %B %Y} (ISO weeks {first} to {weeks[-1]}).\n'
                    'The colour scale is logarithmic and the same in every region.')
        paragraphs = [('Data: European Centre for Disease Prevention and Control (ECDC), weekly cases and deaths by '
                       'country, per million people of 2019. A place counts as 0 in the weeks before its first report.'),
                      (f'Left out, with fewer than 100,000 people: {left_names}. ' if left_out else '')
                      + 'Drawn October 2026.']
    title_size, subtitle_size = (13, 9.5) if not narrow else (11.5, 8.5)
    source_size = 8 if not narrow else 7.5
    source = '\n'.join(textwrap.fill(p, 62 if narrow else 210) for p in paragraphs)
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
    for k, measure in enumerate(MEASURES):
        x, top = xs[k], tops[k]
        ax = place(fig, x, top + title_h, panel_w, panel_h, size)
        heat(ax, rows, measure.name, weeks, layout)
        if narrow:
            pairs += row_labels(fig, ax, rows, EDGE, x - 0.08, layout)
        elif k == 0:
            pairs += row_labels(fig, ax, rows, x + panel_w + pad, xs[1] - pad, layout)
        fig.text((EDGE if narrow else x) / layout.width, 1 - (top + 0.05) / height,
                 f'{measure.name.capitalize()} per million people, per week',
                 ha='left', va='top', fontsize=10 if not narrow else 8.5, fontweight='bold', color=INK)
        key_x, key_w = (EDGE, layout.width - 2 * EDGE) if narrow else (x, min(panel_w, 4.6))
        key = place(fig, key_x, top + title_h + panel_h + axis_h, key_w, legend_h, size)
        legend(key, measure.name, layout, any_missing)

    edge = EDGE / layout.width
    fig.text(edge, 1 - 0.12 / height, title, ha='left', va='top', fontsize=title_size, fontweight='bold', color=INK,
             linespacing=1.2)
    fig.text(edge, 1 - subtitle_top / height, subtitle, ha='left', va='top', fontsize=subtitle_size, color=INK_2,
             linespacing=1.35)
    fig.text(edge, 0.1 / height, source, ha='left', va='bottom', fontsize=source_size, color=MUTED, linespacing=1.35)

    clipped = escaping_text(fig)
    if clipped:
        raise SystemExit(f'heat_{region}{layout.suffix}: text runs off the figure: {clipped}')
    crowded = [name.get_text() for name, pop in pairs
               if name.get_window_extent().x1 + 4 > pop.get_window_extent().x0]
    if crowded:
        raise SystemExit(f'heat_{region}{layout.suffix}: names run into their populations: {crowded}')
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
           f'one row per country or territory, largest population first, and one column per week from '
           f'{monday(weeks[0]):%-d %B %Y} to January 2021. '
           f'Highest weekly cases: {peak(rows, "cases", weeks)}. Highest weekly deaths: {peak(rows, "deaths", weeks)}. '
           f'Rows: {", ".join(display_name(c.name) for c in rows)}.')
    return (f'<picture>\n'
            f'<source media="(max-width: 700px)" srcset="{paths[narrow.suffix].name}" width="{nw}" height="{nh}">\n'
            f'<img src="{paths[wide.suffix].name}" width="{w}" height="{h}" loading="lazy" alt="{alt}">\n'
            f'</picture>')


def main() -> None:
    matplotlib.use('Agg')
    plt.rcParams['hatch.color'] = HATCH
    plt.rcParams['hatch.linewidth'] = 0.6
    weeks, countries = load()
    print(f'ecdc.csv: {len(countries)} places, ISO weeks {weeks[0]} to {weeks[-1]}. '
          f'Drawing {len(REGIONS) * len(LAYOUTS)} figures into {OUT.relative_to(ROOT)}/')
    pictures = []
    for region, names in REGIONS.items():
        members = sorted((countries[c] for c in names), key=lambda c: -c.population)
        rows = [c for c in members if c.population >= MIN_POPULATION]
        left_out = sorted((c for c in members if c.population < MIN_POPULATION), key=lambda c: display_name(c.name))
        paths = {layout.suffix: draw(region, rows, left_out, weeks, layout) for layout in LAYOUTS}
        sizes = ', '.join(f'{p.name} {css_size(p, lay.dpi)[0]}x{css_size(p, lay.dpi)[1]} CSS px'
                          for lay, p in zip(LAYOUTS, paths.values()))
        print(f'  {region}: {len(rows)} rows, {len(left_out)} left out; {sizes}')
        pictures.append(f'<!-- {region} -->\n' + picture(region, rows, weeks, paths))
    markup = ROOT / 'tmp' / 'plot_heat_markup.html'
    markup.parent.mkdir(exist_ok=True)
    markup.write_text('\n'.join(pictures) + '\n', encoding='utf-8')
    print(f'Done. <picture> tags for site/index.html are in {markup.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
