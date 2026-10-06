"""Draw the page's aligned time-series figures, for the World1 and World2 country sets and for every region the 2020 page charted, as small multiples.

The 2020 figures (days100_*_perCapitaFalse_<set>.png, drawn by plot_series.doLinePlots) put 4 to 55 countries on one 480x360 plot with a 10-colour cycle, so countries shared colours and the legend, drawn over the lines, could not tell them apart. These give each country its own panel: the country in red over the other countries of its set in grey, the colours of plot_series.py's own comparison charts, all on one log scale and aligned on the week the country's cumulative count first passed 1,000 cases (100 deaths), as before. In the World sets, chosen in 2020 to be compared, a country that never passed the threshold gets a panel that says so, instead of disappearing; a regional figure names such countries under the figure, with their totals, because a region lists every country in it and Oceania's would otherwise be mostly empty panels. A set of more than 20 countries (the EU, Europe, the Americas, Asia except China, Africa) gets a denser grid, eight columns wide and three on phones. A wide figure has at least four columns. The regions are regions.py's, which are plot_series.py's under ECDC's names. The run stops if a panel's name runs into its total or any text runs off a figure.

Reads ecdc.csv (ECDC weekly cases and deaths per country to ISO week 2021-01, the same data as the rest of the page) and downloads nothing. The EU is summed from its 27 member states under their ECDC names; the 2020 list said 'Czech Republic', which ECDC calls Czechia, so the old EU total left Czechia out.

A country with a section of its own on the page (COUNTRIES: the United States) gets one figure of both measures instead, in place of the 2020 page's pair of days100_*_<country>.png: its cases and its deaths in two panels, each in red over every other country and territory in grey, as those charts showed it.

Writes site/aligned_{cases,deaths}_<set>.png (three rows of panels, or eight columns for a large set, for wide screens) and ..._narrow.png (two columns, or three, for phones), and site/aligned_<country>.png and ..._narrow.png (two panels side by side, or stacked), at 2x and 3x pixel density, as 8-bit palette PNGs, and their <picture> tags to tmp/plot_days100_world_markup.html.

Usage: python3 scripts/plot_days100_world.py [set or country ...]   (default: every set and country)
"""

import io
import math
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import accumulate
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from PIL import Image

import regions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'site'

EU = [
    'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia',
    'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece',
    'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg',
    'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia',
    'Slovenia', 'Spain', 'Sweden',
]

# The country sets of plot_series.py, under ECDC's names.
SETS = {
    'World1': [
        'EU', 'United_States_of_America', 'Brazil', 'India', 'Mexico',
        'United_Kingdom', 'Iran', 'Russia', 'Argentina', 'Colombia', 'Peru',
        'South_Africa', 'Indonesia', 'Sweden',
    ],
    'World2': [
        'Taiwan', 'Vietnam', 'Thailand', 'Singapore', 'New_Zealand',
        'South_Korea', 'Malaysia', 'Japan', 'Australia', 'Uruguay', 'Denmark',
        'Sweden', 'Iceland', 'Norway',
    ],
    # The regional figures of the 2020 page, one per pair of days100_* charts, from regions.py. Western Asia leaves out Iran, which World1 already shows.
    'AsiaWesternExIran': [c for c in regions.PARTS['AsiaWestern'] if c != 'Iran'],
    'EU': regions.REGIONS['EU'],
    'Europe': regions.REGIONS['Europe'],
    'Americas': regions.REGIONS['Americas'],
    'AmericaNorth': regions.PARTS['AmericaNorth'],
    'AmericaSouthExVenezuela': [c for c in regions.PARTS['AmericaSouth'] if c != 'Venezuela'],
    'AsiaExChina': [c for c in regions.REGIONS['Asia'] if c != 'China'],
    'AsiaSouthEast': regions.PARTS['AsiaSouthEast'],
    'AsiaEastExChina': [c for c in regions.PARTS['AsiaEast'] if c != 'China'],
    'Africa': regions.REGIONS['Africa'],
    'Oceania': regions.REGIONS['Oceania'],
    'Nordic': regions.REGIONS['Nordic'],
}

# Countries with a section of their own, drawn by draw_country, and how a sentence names each.
COUNTRIES = {'United_States_of_America': 'the United States'}

# What a regional figure's title calls its set; the World sets need no name.
REGIONS = {
    'AsiaWesternExIran': 'Western Asia except Iran',
    'EU': 'the EU',
    'Europe': 'Europe',
    'Americas': 'the Americas',
    'AmericaNorth': 'North America',
    'AmericaSouthExVenezuela': 'South America except Venezuela',
    'AsiaExChina': 'Asia except China',
    'AsiaSouthEast': 'South-East Asia',
    'AsiaEastExChina': 'East Asia except China',
    'Africa': 'Africa',
    'Oceania': 'Oceania',
    'Nordic': 'the Nordic countries',
}

# A set of more than this many countries gets the dense grid, and its countries that never passed the threshold are named under the figure instead of each taking a panel.
DENSE = 20

NAMES = {
    'United_States_of_America': 'United States',
    'Cote_dIvoire': "Côte d'Ivoire",
    'Guinea_Bissau': 'Guinea-Bissau',
    'Sao_Tome_and_Principe': 'São Tomé and Príncipe',
    'Brunei_Darussalam': 'Brunei',
    'United_Republic_of_Tanzania': 'Tanzania',
    'Democratic_Republic_of_the_Congo': 'DR Congo',
    'Timor_Leste': 'Timor-Leste',
    'Micronesia_(Federated_States_of)': 'Micronesia',
    'Turks_and_Caicos_islands': 'Turks and Caicos Islands',
}
# On the narrower panels these run into their totals.
NARROW_NAMES = {
    'United_Kingdom': 'UK', 'United_Arab_Emirates': 'UAE',
    'Bosnia_and_Herzegovina': 'Bosnia & Herz.', 'Trinidad_and_Tobago': 'Trinidad & T.',
    'Dominican_Republic': 'Dominican Rep.', 'Central_African_Republic': 'C. African Rep.',
    'Equatorial_Guinea': 'Eq. Guinea', 'North_Macedonia': 'N. Macedonia',
    'Sao_Tome_and_Principe': 'São Tomé & P.', 'Saint_Vincent_and_the_Grenadines': 'St Vincent & G.',
    'United_States_Virgin_Islands': 'US Virgin Isl.', 'Turks_and_Caicos_islands': 'Turks & Caicos',
    'Saint_Kitts_and_Nevis': 'St Kitts & Nevis', 'Antigua_and_Barbuda': 'Antigua & Barb.',
    'Northern_Mariana_Islands': 'N. Mariana Isl.', 'Falkland_Islands_(Malvinas)': 'Falklands',
}


@dataclass(frozen=True)
class Measure:
    name: str
    column: str
    threshold: int


MEASURES = [
    Measure('cases', 'cases_weekly', 1000),
    Measure('deaths', 'deaths_weekly', 100),
]


@dataclass(frozen=True)
class Layout:
    suffix: str
    width: float
    ncols: int | None
    panel_height: float
    dpi: int
    dense_ncols: int
    dense_panel_height: float

    @property
    def narrow(self) -> bool:
        return self.suffix == '_narrow'

    def columns(self, panels: int, dense: bool) -> int:
        """The dense grid's column count, the fixed one, or else as many as three rows need with a slot left for the key."""
        if dense:
            return self.dense_ncols
        return self.ncols or max(4, math.ceil((panels + 1) / 3))


# Inches at 100 dpi, so one inch is 100 CSS pixels at the size the page shows.
LAYOUTS = [
    Layout('', width=13.55, ncols=None, panel_height=1.9, dpi=200, dense_ncols=8, dense_panel_height=1.45),
    Layout('_narrow', width=4.15, ncols=2, panel_height=1.55, dpi=300, dense_ncols=3, dense_panel_height=1.3),
]

# Ink and marks. The country's red is plot_series.doLinePlots' (#e41a1c, over grey), as Tommy drew it in 2020.
INK = '#0b0b0b'
INK_2 = '#52514e'
# Tick labels, the key's note and the source line: 4.9:1 on white, above WCAG's 4.5:1 for text (the earlier #898781 was 3.6:1).
MUTED = '#73716b'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
CONTEXT = '#d3d1c9'
FOCUS = '#e41a1c'

# The gid of the week-0 date under each panel, which collisions() checks against every other text in the figure.
WEEK0 = 'week0'


def rgb(colour: str) -> tuple[int, int, int]:
    return int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)


def palette() -> np.ndarray:
    """The figures' colours, each blended into the white ground in steps, and the red over each grey: what antialiasing draws, in 218 of a PNG palette's 256 entries."""
    white = (255, 255, 255)
    colours: list[tuple[int, int, int]] = []

    def blend(a: tuple[int, int, int], b: tuple[int, int, int], steps: int) -> None:
        for k in range(steps + 1):
            c = (round(a[0] + (b[0] - a[0]) * k / steps), round(a[1] + (b[1] - a[1]) * k / steps),
                 round(a[2] + (b[2] - a[2]) * k / steps))
            if c not in colours:
                colours.append(c)

    for colour, steps in ((INK, 32), (INK_2, 24), (MUTED, 24), (FOCUS, 32), (CONTEXT, 10), (GRID, 6), (AXIS, 10)):
        blend(white, rgb(colour), steps)
    for grey in (CONTEXT, GRID, AXIS):
        blend(rgb(grey), rgb(FOCUS), 20)
    for a, b, steps in ((GRID, CONTEXT, 4), (AXIS, CONTEXT, 4), (GRID, AXIS, 4), (GRID, MUTED, 8), (CONTEXT, INK_2, 8)):
        blend(rgb(a), rgb(b), steps)
    return np.array(colours, dtype=np.int32)


PALETTE = palette()


def load() -> pd.DataFrame:
    """ECDC weekly rows, with the EU added as the sum of its members."""
    df = pd.read_csv(ROOT / 'data' / 'ecdc.csv')
    df['dateRep'] = pd.to_datetime(df['dateRep'], format='%d/%m/%Y')
    missing = sorted(set(EU) - set(df['countriesAndTerritories']))
    if missing:
        raise SystemExit(f'ecdc.csv has no rows for EU members {missing}')
    members = df[df['countriesAndTerritories'].isin(EU)]
    eu = pd.DataFrame(members.groupby('dateRep')[['cases_weekly', 'deaths_weekly']].sum()).reset_index()
    eu['countriesAndTerritories'] = 'EU'
    return pd.concat([df, eu], ignore_index=True)


@dataclass(frozen=True)
class Series:
    dates: list[datetime]
    cumulative: list[float]


def series(df: pd.DataFrame, country: str, column: str) -> Series:
    """A country's report dates in order, and its running total of the weekly column."""
    rows = df.loc[df['countriesAndTerritories'] == country, ['dateRep', column]]
    if rows.empty:
        raise SystemExit(f'ecdc.csv has no rows for {country}')
    pairs = sorted(zip(pd.to_datetime(rows['dateRep']).dt.to_pydatetime(), rows[column].astype(float)))
    return Series([d for d, _ in pairs], list(accumulate(v for _, v in pairs)))


def first_week(s: Series, threshold: int) -> int | None:
    """The index of the first week whose cumulative count is above the threshold."""
    return next((i for i, v in enumerate(s.cumulative) if v > threshold), None)


def aligned(s: Series, threshold: int) -> tuple[list[float], list[float]] | None:
    """Weeks since the first week above the threshold, and the cumulative counts from then on."""
    first = first_week(s, threshold)
    if first is None:
        return None
    start = s.dates[first]
    return [(d - start).days / 7 for d in s.dates[first:]], s.cumulative[first:]


def week0_end(s: Series, threshold: int) -> datetime | None:
    """The last day of week 0: ECDC dates each ISO week by the Monday after it, so the day before its report date."""
    first = first_week(s, threshold)
    return None if first is None else s.dates[first] - timedelta(days=1)


def short(n: float) -> str:
    """9,666 / 56.8k / 828k / 8.13M / 22.4M: three significant figures above 10,000."""
    if n >= 1e6:
        return f'{n / 1e6:.3g}M'
    if n >= 1e4:
        return f'{n / 1e3:.3g}k'
    return f'{n:,.0f}'


def tick(value: float, _pos: float) -> str:
    for size, unit in ((1e6, 'M'), (1e3, 'k')):
        if value >= size:
            return f'{value / size:g}{unit}'
    return f'{value:g}'


def display_name(country: str) -> str:
    return NAMES.get(country, country.replace('_', ' '))


def ceiling(value: float) -> float:
    """The first of 1, 2, 5, 10, 20, 50, ... at or above the value, so the axis ends just above the highest line."""
    exponent = math.floor(math.log10(value))
    return next(m * 10.0 ** exponent for m in (1, 2, 5, 10) if m * 10.0 ** exponent >= value)


def style(ax: Axes, decades: list[float], ymax: float, xmax: float, labelsize: float = 8) -> None:
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(FixedLocator(decades))
    ax.yaxis.set_major_formatter(FuncFormatter(tick))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0, xmax)
    ax.set_ylim(decades[0], ymax)
    ax.xaxis.set_major_locator(FixedLocator(list(range(0, int(xmax) + 1, 10))))
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=MUTED, labelsize=labelsize, length=2.5, width=0.6)


def draw(df: pd.DataFrame, set_name: str, measure: Measure, layout: Layout) -> Path:
    countries = SETS[set_name]
    dense = len(countries) > DENSE
    data = {c: series(df, c, measure.column) for c in countries}
    totals = {c: s.cumulative[-1] for c, s in data.items()}
    lines = {c: aligned(s, measure.threshold) for c, s in data.items()}
    drawn = {c: xy for c, xy in lines.items() if xy is not None}
    starts = {c: week0_end(s, measure.threshold) for c, s in data.items()}
    # A regional figure gives panels only to the countries that passed the threshold, and names the others.
    listed = set_name in REGIONS
    order = sorted(drawn if listed else countries, key=lambda c: -totals[c])
    never = sorted((c for c in countries if c not in drawn), key=lambda c: -totals[c]) if listed else []

    # Room above the highest line for its end marker.
    ymax = ceiling(1.15 * max(max(y) for _, y in drawn.values()))
    decades = [10.0 ** e for e in range(int(math.log10(measure.threshold)), math.floor(math.log10(ymax)) + 1)]
    xmax = math.ceil(max(max(x) for x, _ in drawn.values()) / 5) * 5 + 1

    ncols = layout.columns(len(order), dense)
    nrows = math.ceil(len(order) / ncols)
    narrow = layout.narrow
    panel_height = layout.dense_panel_height if dense else layout.panel_height
    region = REGIONS.get(set_name)
    # A regional title takes one more line on the narrow figure.
    extra = 0.2 if narrow and region else 0
    header = (1.45 if narrow else 0.95) + extra

    noun = measure.name
    threshold = f'{measure.threshold:,}'
    others = len(order) - 1 if listed else len(countries) - 1
    eu = 'EU' in countries
    source_size = 8 if not narrow else 7.5
    if narrow:
        title = f'Cumulative COVID-19 {noun}\nsince passing {threshold}'
        if region:
            title = f'{region[0].upper()}{region[1:]}\n{title}'
        subtitle = (f'Weeks since each country passed {threshold} {noun}.\nLog scale. Red: the country; grey: the\n'
                    f'other {others}. Number: total on 10 Jan 2021.')
        source = ('Data: ECDC, weekly, to 10 January 2021.\nEU: the 27 member states. Drawn 2026.' if eu else
                  'Data: ECDC, weekly, to 10 January 2021.\nDrawn October 2026.')
    else:
        title = f'Cumulative COVID-19 {noun}, counted from the week each country passed {threshold}'
        if region:
            title = f'{region[0].upper()}{region[1:]}: c{title[1:]}'
        subtitle = (f'Weeks since that week on the horizontal axis; cumulative {noun} on a log scale. '
                    f'Each panel shows one country in red over the other {others} in grey.')
        source = ('Data: European Centre for Disease Prevention and Control (ECDC), weekly cases and deaths by '
                  'country, to 10 January 2021 (ISO week 2021-01).' + (' EU: the 27 member states.' if eu else '')
                  + ' Drawn October 2026.')
    # Room below the bottom panels for their tick labels, the date of week 0 under them and the source line, and for the list of countries that never passed.
    labelsize = 7 if dense else 8
    footer = (0.8 if narrow else 0.62) + (labelsize * 1.3 + 2) / 72
    if never:
        names = ', '.join(f'{display_name(c)} ({totals[c]:,.0f})' for c in never)
        named = textwrap.fill(f'Never passed {threshold} {noun} (their totals on 10 January 2021): {names}.',
                              62 if narrow else 210)
        source = f'{named}\n{source}'
        footer += (named.count('\n') + 1) * source_size * 1.35 / 72

    width = layout.width
    height = header + nrows * panel_height + footer
    fig = plt.figure(figsize=(width, height), dpi=100, facecolor='white')
    grid = fig.add_gridspec(
        nrows, ncols,
        left=0.5 / width, right=1 - 0.12 / width,
        top=1 - header / height, bottom=footer / height,
        hspace=0.42 if not (dense and narrow) else 0.62, wspace=0.12 if not narrow else 0.16)

    # Three columns on a phone leave no room for a name and its total on one line, so the total goes under the name.
    stacked = dense and narrow
    name_size = (8 if narrow else 8.5) if dense else (9 if narrow else 9.5)
    total_size = (7.5 if narrow else 8) if dense else (8.5 if narrow else 9)
    for i, country in enumerate(order):
        ax = fig.add_subplot(grid[i // ncols, i % ncols])
        style(ax, decades, ymax, xmax, labelsize)
        for other, (x, y) in drawn.items():
            if other != country:
                ax.plot(x, y, color=CONTEXT, linewidth=0.8 if not dense else 0.6, zorder=1, solid_capstyle='round')
        # Panels narrower than five to a row take the short names.
        name = NARROW_NAMES.get(country, display_name(country)) if narrow or ncols > 5 else display_name(country)
        total = f'{short(totals[country])}'
        if stacked:
            ax.set_title(name, loc='left', fontsize=name_size, color=INK, fontweight='bold', pad=12)
            ax.text(0, 1.03, total, transform=ax.transAxes, ha='left', va='bottom', fontsize=total_size, color=INK_2)
        else:
            ax.set_title(name, loc='left', fontsize=name_size, color=INK, fontweight='bold', pad=4)
            ax.set_title(total, loc='right', fontsize=total_size, color=INK_2, pad=4)
        if country in drawn:
            x, y = drawn[country]
            ax.plot(x, y, color=FOCUS, linewidth=2 if not dense else 1.7, zorder=3, solid_capstyle='round')
            ax.plot(x[-1], y[-1], 'o', color=FOCUS, markersize=4 if not dense else 3.5, zorder=4)
        else:
            ax.text(0.5, 0.5, f'Never passed\n{measure.threshold:,} {measure.name}',
                    transform=ax.transAxes, ha='center', va='center', zorder=5,
                    fontsize=8.5, color=INK_2, linespacing=1.3,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 3})
        if i % ncols:
            ax.tick_params(labelleft=False)
        ticked = i >= len(order) - ncols
        if not ticked:
            ax.tick_params(labelbottom=False)
        # Week 0's calendar date under the panel's origin, below the tick labels where there are any: the alignment otherwise hides whether a country passed the threshold in the spring 2020 wave or the autumn one.
        start = starts[country]
        if start is not None:
            below = 3 + (2.5 + 3.5 + labelsize * 1.2 + 1.5 if ticked else 0)
            ax.annotate(f'from {start.day} {start:%b %Y}', (0, 0), xycoords='axes fraction', xytext=(0, -below),
                        textcoords='offset points', ha='left', va='top', fontsize=labelsize, color=MUTED, gid=WEEK0)

    # A key in the first empty slot, if there is one.
    if len(order) % ncols:
        i = len(order)
        key = fig.add_subplot(grid[i // ncols, i % ncols])
        key.axis('off')
        key_size = 9 if not dense else 8 if not narrow else 7.5
        top, step, note_y = (0.62, 0.2, 0.22) if not dense else (0.8, 0.2, 0.3)
        for row, (color, lw, text) in enumerate((
                (FOCUS, 2, 'The country'),
                (CONTEXT, 0.8, f'The other {others}'))):
            yk = top - row * step
            key.plot([0.08, 0.26], [yk, yk], color=color, linewidth=lw, transform=key.transAxes)
            key.text(0.31, yk, text, transform=key.transAxes, va='center', fontsize=key_size, color=INK_2)
        note = ('Number at top right:\ntotal on 10 January 2021' if not dense else
                'Number at top right:\ntotal, 10 Jan 2021' if not stacked else 'Number under the\nname: total on\n10 Jan 2021')
        key.text(0.08, note_y, note, transform=key.transAxes,
                 va='center', fontsize=key_size - 0.5, color=MUTED, linespacing=1.3)

    left = 0.12 / width
    fig.text(left, 1 - 0.12 / height, title, ha='left', va='top', fontsize=13 if not narrow else 11.5,
             fontweight='bold', color=INK, linespacing=1.2)
    fig.text(left, 1 - ((0.52 if not narrow else 0.62) + extra) / height, subtitle, ha='left', va='top',
             fontsize=9.5 if not narrow else 8.5, color=INK_2, linespacing=1.35)
    fig.text(left, 0.1 / height, source, ha='left', va='bottom', fontsize=source_size, color=MUTED,
             linespacing=1.35)

    path = OUT / f'aligned_{measure.name}_{set_name}{layout.suffix}.png'
    problems = collisions(fig)
    if problems:
        raise SystemExit(f'{path.name}: {problems}')
    save(fig, path, layout.dpi)
    plt.close(fig)
    return path


def draw_country(df: pd.DataFrame, country: str, layout: Layout) -> Path:
    """One country's cumulative cases and deaths, each aligned on the week it passed the threshold, in red over every other country and territory that passed it, in grey: side by side on wide screens, stacked on phones."""
    narrow = layout.narrow
    name = display_name(country)
    prose = COUNTRIES[country]
    places = sorted(set(df['countriesAndTerritories']) - {'EU', country})
    panels = []
    for measure in MEASURES:
        data = {c: series(df, c, measure.column) for c in [country, *places]}
        drawn = {c: xy for c, s in data.items() if (xy := aligned(s, measure.threshold)) is not None}
        if country not in drawn:
            raise SystemExit(f'{name} never passed {measure.threshold:,} {measure.name}')
        panels.append((measure, data, drawn))
    others = [len(drawn) - 1 for _, _, drawn in panels]

    if narrow:
        title = f'{name}: cumulative COVID-19\ncases and deaths since passing\n1,000 cases and 100 deaths'
        subtitle = (f'Weeks since it passed 1,000 cases (100 deaths).\nLog scales. Red: {prose}; grey: the\n'
                    f'other {others[0]} places that passed 1,000 cases\nand {others[1]} that passed 100 deaths.')
        source = 'Data: ECDC, weekly, to 10 January 2021.\nDrawn October 2026.'
        header, panel_height, footer = 1.75, 2.7, 0.8 + (8 * 1.3 + 2) / 72
    else:
        title = f'{name}: cumulative COVID-19 cases and deaths, counted from the week it passed 1,000 cases and 100 deaths'
        subtitle = (f'Weeks since that week on the horizontal axis; cumulative counts on log scales. {prose[0].upper()}{prose[1:]} in red, '
                    f'and in grey the other {others[0]} countries and territories that passed 1,000 cases\n'
                    f'and the {others[1]} that passed 100 deaths.')
        source = ('Data: European Centre for Disease Prevention and Control (ECDC), weekly cases and deaths by '
                  'country, to 10 January 2021 (ISO week 2021-01). Drawn October 2026.')
        header, panel_height, footer = 1.15, 3.6, 0.62 + (8 * 1.3 + 2) / 72
    nrows, ncols = (2, 1) if narrow else (1, 2)
    width = layout.width
    height = header + nrows * panel_height + (0.45 if narrow else 0) + footer
    fig = plt.figure(figsize=(width, height), dpi=100, facecolor='white')
    grid = fig.add_gridspec(
        nrows, ncols,
        left=0.55 / width, right=1 - 0.12 / width,
        top=1 - header / height, bottom=footer / height,
        hspace=0.42, wspace=0.14)
    for i, (measure, data, drawn) in enumerate(panels):
        ymax = ceiling(1.15 * max(max(y) for _, y in drawn.values()))
        decades = [10.0 ** e for e in range(int(math.log10(measure.threshold)), math.floor(math.log10(ymax)) + 1)]
        xmax = math.ceil(max(max(x) for x, _ in drawn.values()) / 5) * 5 + 1
        ax = fig.add_subplot(grid[i, 0] if narrow else grid[0, i])
        style(ax, decades, ymax, xmax)
        for other, (x, y) in drawn.items():
            if other != country:
                ax.plot(x, y, color=CONTEXT, linewidth=0.8, zorder=1, solid_capstyle='round')
        x, y = drawn[country]
        ax.plot(x, y, color=FOCUS, linewidth=2, zorder=3, solid_capstyle='round')
        ax.plot(x[-1], y[-1], 'o', color=FOCUS, markersize=4, zorder=4)
        ax.set_title(f'{measure.name.capitalize()}, from the week it passed {measure.threshold:,}', loc='left',
                     fontsize=9 if narrow else 9.5, color=INK, fontweight='bold', pad=4)
        ax.set_title(short(data[country].cumulative[-1]), loc='right', fontsize=8.5 if narrow else 9, color=INK_2, pad=4)
        start = week0_end(data[country], measure.threshold)
        if start is not None:
            ax.annotate(f'from {start.day} {start:%b %Y}', (0, 0), xycoords='axes fraction',
                        xytext=(0, -(3 + 2.5 + 3.5 + 8 * 1.2 + 1.5)), textcoords='offset points', ha='left', va='top',
                        fontsize=8, color=MUTED, gid=WEEK0)

    left = 0.12 / width
    fig.text(left, 1 - 0.12 / height, title, ha='left', va='top', fontsize=13 if not narrow else 11.5,
             fontweight='bold', color=INK, linespacing=1.2)
    fig.text(left, 1 - (0.52 if not narrow else 0.86) / height, subtitle, ha='left', va='top',
             fontsize=9.5 if not narrow else 8.5, color=INK_2, linespacing=1.35)
    fig.text(left, 0.1 / height, source, ha='left', va='bottom', fontsize=8 if not narrow else 7.5, color=MUTED,
             linespacing=1.35)

    path = OUT / f'aligned_{country}{layout.suffix}.png'
    problems = collisions(fig)
    if problems:
        raise SystemExit(f'{path.name}: {problems}')
    save(fig, path, layout.dpi)
    plt.close(fig)
    return path


def country_picture(df: pd.DataFrame, country: str, paths: dict[str, Path]) -> str:
    """A <picture> that serves the stacked figure below 700 CSS pixels."""
    wide, narrow = LAYOUTS
    w, h = css_size(paths[wide.suffix], wide.dpi)
    nw, nh = css_size(paths[narrow.suffix], narrow.dpi)
    facts = []
    for measure in MEASURES:
        s = series(df, country, measure.column)
        start = week0_end(s, measure.threshold)
        facts.append(f'{s.cumulative[-1]:,.0f} {measure.name}, counted from the week to {start:%-d %B %Y}' if start else '')
    alt = (f'Cumulative COVID-19 cases and deaths in {COUNTRIES[country]}, from the week it passed 1,000 cases and 100 deaths, '
           f'in red over every other country in grey; log scales. Totals on 10 January 2021: {facts[0]}; {facts[1]}.')
    return (f'<picture>\n'
            f'<source media="(max-width: 700px)" srcset="{paths[narrow.suffix].name}" width="{nw}" height="{nh}">\n'
            f'<img src="{paths[wide.suffix].name}" width="{w}" height="{h}" alt="{alt}" loading="lazy">\n'
            f'</picture>')


def save(fig: Figure, path: Path, dpi: int) -> None:
    """Write the figure as an 8-bit palette PNG, about a third of the bytes of matplotlib's RGBA one, with each pixel mapped to its nearest colour in PALETTE, so the white, the greys and the red stay exact.

    Pillow's own quantize(palette=...) looks colours up at reduced precision and turns the white ground light grey, hence the exact nearest-colour search here. A colour the palette lacks (a new mark drawn in another colour) stops the run, rather than being silently replaced."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=dpi, facecolor='white')
    with Image.open(buffer) as im:
        pixels = np.asarray(im.convert('RGB'), dtype=np.int32)
    keys = (pixels[..., 0] << 16) | (pixels[..., 1] << 8) | pixels[..., 2]
    unique, inverse = np.unique(keys.ravel(), return_inverse=True)
    colours = np.stack([(unique >> 16) & 255, (unique >> 8) & 255, unique & 255], axis=1)
    distance = ((colours[:, None, :] - PALETTE[None, :, :]) ** 2).sum(axis=2)
    nearest = distance.argmin(axis=1)
    error = np.abs(colours - PALETTE[nearest]).max(axis=1)
    if error.max() > 24:
        worst = colours[error.argmax()]
        raise SystemExit(f'{path.name}: #{worst[0]:02x}{worst[1]:02x}{worst[2]:02x} is not near any colour in PALETTE')
    out = Image.fromarray(nearest[inverse].reshape(keys.shape).astype(np.uint8), 'P')
    out.putpalette(PALETTE.astype(np.uint8).ravel().tolist())
    out.save(path, optimize=True, dpi=(dpi, dpi))


def collisions(fig: Figure) -> list[str]:
    """Panel titles that run into each other, a week-0 date that runs into any other text, and any visible text that reaches past an edge of the figure."""
    fig.canvas.draw()
    box = fig.bbox
    found = []
    texts = list(fig.texts)
    for ax in fig.axes:
        own = [t for t in ax.get_children() if isinstance(t, Text) and t.get_visible() and t.get_text()]
        texts += own
        if ax.axison:
            texts += [*ax.get_xticklabels(), *ax.get_yticklabels()]
        extents = [(t.get_text(), t.get_window_extent()) for t in own]
        for k, (a, ea) in enumerate(extents):
            for b, eb in extents[k + 1:]:
                if ea.overlaps(eb):
                    found.append(f'{a!r} runs into {b!r}')
    shown = [t for t in texts if t.get_visible() and t.get_text()]
    for date in (t for t in shown if t.get_gid() == WEEK0):
        extent = date.get_window_extent()
        found += [f'{date.get_text()!r} runs into {t.get_text()!r}' for t in shown
                  if t is not date and t.get_window_extent().overlaps(extent)]
    for t in texts:
        if not t.get_visible() or not t.get_text():
            continue
        e = t.get_window_extent()
        if e.x0 < box.x0 - 0.5 or e.x1 > box.x1 + 0.5 or e.y0 < box.y0 - 0.5 or e.y1 > box.y1 + 0.5:
            found.append(f'{t.get_text()!r} runs off the figure')
    return found


def css_size(path: Path, dpi: int) -> tuple[int, int]:
    """The image's size in CSS pixels: 100 per inch, as the figures are laid out."""
    with Image.open(path) as im:
        return round(im.size[0] * 100 / dpi), round(im.size[1] * 100 / dpi)


def picture(df: pd.DataFrame, set_name: str, measure: Measure, paths: dict[str, Path]) -> str:
    """A <picture> that serves the two-column figure below 700 CSS pixels, with every total in its alt text."""
    wide, narrow = LAYOUTS
    w, h = css_size(paths[wide.suffix], wide.dpi)
    nw, nh = css_size(paths[narrow.suffix], narrow.dpi)
    totals = sorted(((series(df, c, measure.column).cumulative[-1], c) for c in SETS[set_name]), reverse=True)
    listed = '; '.join(f'{display_name(c)} {v:,.0f}' for v, c in totals)
    where = f' in {REGIONS[set_name]}' if set_name in REGIONS else ''
    alt = (f'Cumulative COVID-19 {measure.name}{where} from the week each country passed {measure.threshold:,}, '
           f'one panel per country. Totals on 10 January 2021: {listed}.')
    return (f'<picture>\n'
            f'<source media="(max-width: 700px)" srcset="{paths[narrow.suffix].name}" width="{nw}" height="{nh}">\n'
            f'<img src="{paths[wide.suffix].name}" width="{w}" height="{h}" alt="{alt}">\n'
            f'</picture>')


def main() -> None:
    sets = sys.argv[1:] or [*SETS, *COUNTRIES]
    unknown = [s for s in sets if s not in SETS and s not in COUNTRIES]
    if unknown:
        raise SystemExit(f'Unknown set {unknown}; the sets are {list(SETS)} and the countries {list(COUNTRIES)}')
    matplotlib.use('Agg')
    df = load()
    figures = sum(len(LAYOUTS) if s in COUNTRIES else len(MEASURES) * len(LAYOUTS) for s in sets)
    print(f'Drawing {figures} figures from ecdc.csv into '
          f'{OUT.relative_to(ROOT)}/ (last report {df["dateRep"].max():%Y-%m-%d})')
    pictures = []
    for set_name in sets:
        if set_name in COUNTRIES:
            paths = {}
            for layout in LAYOUTS:
                path = paths[layout.suffix] = draw_country(df, set_name, layout)
                with Image.open(path) as im:
                    print(f'  {path.name}: {im.size[0]}x{im.size[1]} px, {path.stat().st_size / 1e3:.0f} kB')
            pictures.append(country_picture(df, set_name, paths))
            continue
        for measure in MEASURES:
            paths = {}
            for layout in LAYOUTS:
                path = paths[layout.suffix] = draw(df, set_name, measure, layout)
                with Image.open(path) as im:
                    print(f'  {path.name}: {im.size[0]}x{im.size[1]} px, {path.stat().st_size / 1e3:.0f} kB')
            pictures.append(picture(df, set_name, measure, paths))
    markup = ROOT / 'tmp' / 'plot_days100_world_markup.html'
    markup.parent.mkdir(exist_ok=True)
    markup.write_text('\n'.join(pictures) + '\n')
    print(f'Done. <picture> tags for site/index.html are in {markup.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
