"""Draw the page's aligned time-series figures for the World1 and World2 country sets as small multiples.

The 2020 figures (days100_*_perCapitaFalse_World{1,2}.png, drawn by plot_series.doLinePlots) put 14 countries on one 480x360 plot with a 10-colour cycle, so four pairs of countries shared a colour and the legend could not tell them apart. These give each country its own panel: the country in blue over the other countries of its set in grey, all on one log scale and aligned on the week the country's cumulative count first passed 1,000 cases (100 deaths), as before. A country that never passed the threshold gets a panel that says so, instead of disappearing.

Reads ecdc.csv (ECDC weekly cases and deaths per country to ISO week 2021-01, the same data as the rest of the page) and downloads nothing. The EU is summed from its 27 member states under their ECDC names; the 2020 list said 'Czech Republic', which ECDC calls Czechia, so the old EU total left Czechia out.

Writes site/aligned_{cases,deaths}_{World1,World2}.png (five columns, for wide screens) and ..._narrow.png (two columns, for phones), at 2x and 3x pixel density.

Usage: python3 plot_days100_world.py
"""

import math
from dataclasses import dataclass
from datetime import datetime
from itertools import accumulate
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from PIL import Image

ROOT = Path(__file__).resolve().parent
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
}

NAMES = {'United_States_of_America': 'United States'}
# On the two-column figure 'United Kingdom' runs into its total.
NARROW_NAMES = {'United_Kingdom': 'UK'}


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
    ncols: int
    panel_width: float
    panel_height: float
    dpi: int


# Inches at 100 dpi, so one inch is 100 CSS pixels at the size the page shows.
LAYOUTS = [
    Layout('', ncols=5, panel_width=2.6, panel_height=1.9, dpi=200),
    Layout('_narrow', ncols=2, panel_width=1.8, panel_height=1.55, dpi=300),
]

# Ink and marks, from the dataviz reference palette.
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
CONTEXT = '#d3d1c9'
FOCUS = '#2a78d6'


def load() -> pd.DataFrame:
    """ECDC weekly rows, with the EU added as the sum of its members."""
    df = pd.read_csv(ROOT / 'ecdc.csv')
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


def aligned(s: Series, threshold: int) -> tuple[list[float], list[float]] | None:
    """Weeks since the first week above the threshold, and the cumulative counts from then on."""
    first = next((i for i, v in enumerate(s.cumulative) if v > threshold), None)
    if first is None:
        return None
    start = s.dates[first]
    return [(d - start).days / 7 for d in s.dates[first:]], s.cumulative[first:]


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


def style(ax: Axes, decades: list[float], xmax: float) -> None:
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(FixedLocator(decades))
    ax.yaxis.set_major_formatter(FuncFormatter(tick))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0, xmax)
    ax.set_ylim(decades[0], decades[-1])
    ax.xaxis.set_major_locator(FixedLocator(list(range(0, int(xmax) + 1, 10))))
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=MUTED, labelsize=8, length=2.5, width=0.6)


def draw(df: pd.DataFrame, set_name: str, measure: Measure, layout: Layout) -> Path:
    countries = SETS[set_name]
    data = {c: series(df, c, measure.column) for c in countries}
    totals = {c: s.cumulative[-1] for c, s in data.items()}
    lines = {c: aligned(s, measure.threshold) for c, s in data.items()}
    order = sorted(countries, key=lambda c: -totals[c])
    drawn = {c: xy for c, xy in lines.items() if xy is not None}

    top = max(max(y) for _, y in drawn.values())
    decades = [10.0 ** e for e in range(int(math.log10(measure.threshold)), math.ceil(math.log10(top)) + 1)]
    xmax = math.ceil(max(max(x) for x, _ in drawn.values()) / 5) * 5 + 1

    ncols = layout.ncols
    nrows = math.ceil(len(order) / ncols)
    narrow = ncols < 5
    header = 1.45 if narrow else 0.95
    # Room below the bottom panels for their tick labels and the source line.
    footer = 0.8 if narrow else 0.62
    width = ncols * layout.panel_width + 0.55
    height = header + nrows * layout.panel_height + footer
    fig = plt.figure(figsize=(width, height), dpi=100, facecolor='white')
    grid = fig.add_gridspec(
        nrows, ncols,
        left=0.5 / width, right=1 - 0.12 / width,
        top=1 - header / height, bottom=footer / height,
        hspace=0.42, wspace=0.12 if not narrow else 0.16)

    for i, country in enumerate(order):
        ax = fig.add_subplot(grid[i // ncols, i % ncols])
        style(ax, decades, xmax)
        for other, (x, y) in drawn.items():
            if other != country:
                ax.plot(x, y, color=CONTEXT, linewidth=0.8, zorder=1, solid_capstyle='round')
        name = NARROW_NAMES.get(country, display_name(country)) if narrow else display_name(country)
        total = f'{short(totals[country])}'
        ax.set_title(name, loc='left', fontsize=9 if narrow else 9.5, color=INK, fontweight='bold', pad=4)
        ax.set_title(total, loc='right', fontsize=8.5 if narrow else 9, color=INK_2, pad=4)
        if country in drawn:
            x, y = drawn[country]
            ax.plot(x, y, color=FOCUS, linewidth=2, zorder=3, solid_capstyle='round')
            ax.plot(x[-1], y[-1], 'o', color=FOCUS, markersize=4, zorder=4)
        else:
            ax.text(0.5, 0.5, f'Never passed\n{measure.threshold:,} {measure.name}',
                    transform=ax.transAxes, ha='center', va='center', zorder=5,
                    fontsize=8.5, color=INK_2, linespacing=1.3,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 3})
        if i % ncols:
            ax.tick_params(labelleft=False)
        if i < len(order) - ncols:
            ax.tick_params(labelbottom=False)

    # A key in the first empty slot, if there is one.
    others = len(countries) - 1
    if len(order) % ncols:
        i = len(order)
        key = fig.add_subplot(grid[i // ncols, i % ncols])
        key.axis('off')
        for row, (color, lw, text) in enumerate((
                (FOCUS, 2, 'The country'),
                (CONTEXT, 0.8, f'The other {others}'))):
            yk = 0.62 - row * 0.2
            key.plot([0.08, 0.26], [yk, yk], color=color, linewidth=lw, transform=key.transAxes)
            key.text(0.31, yk, text, transform=key.transAxes, va='center', fontsize=9, color=INK_2)
        key.text(0.08, 0.22, 'Number at top right:\ntotal on 10 January 2021', transform=key.transAxes,
                 va='center', fontsize=8.5, color=MUTED, linespacing=1.3)

    noun = measure.name
    threshold = f'{measure.threshold:,}'
    if narrow:
        title = f'Cumulative COVID-19 {noun}\nsince passing {threshold}'
        subtitle = (f'Weeks since each country passed {threshold} {noun}.\nLog scale. Blue: the country; grey: the\n'
                    f'other {others}. Number: total on 10 Jan 2021.')
        source = 'Data: ECDC, weekly, to 10 January 2021.\nEU: the 27 member states. Drawn 2026.'
    else:
        title = f'Cumulative COVID-19 {noun}, counted from the week each country passed {threshold}'
        subtitle = (f'Weeks since that week on the horizontal axis; cumulative {noun} on a log scale. '
                    f'Each panel shows one country in blue over the other {others} in grey.')
        source = ('Data: European Centre for Disease Prevention and Control (ECDC), weekly cases and deaths by '
                  'country, to 10 January 2021 (ISO week 2021-01). EU: the 27 member states. Drawn October 2026.')
    left = 0.12 / width
    fig.text(left, 1 - 0.12 / height, title, ha='left', va='top', fontsize=13 if not narrow else 11.5,
             fontweight='bold', color=INK, linespacing=1.2)
    fig.text(left, 1 - (0.52 if not narrow else 0.62) / height, subtitle, ha='left', va='top',
             fontsize=9.5 if not narrow else 8.5, color=INK_2, linespacing=1.35)
    fig.text(left, 0.1 / height, source, ha='left', va='bottom', fontsize=8 if not narrow else 7.5, color=MUTED,
             linespacing=1.35)

    path = OUT / f'aligned_{measure.name}_{set_name}{layout.suffix}.png'
    fig.savefig(path, dpi=layout.dpi, facecolor='white')
    plt.close(fig)
    return path


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
    alt = (f'Cumulative COVID-19 {measure.name} from the week each country passed {measure.threshold:,}, '
           f'one panel per country. Totals on 10 January 2021: {listed}.')
    return (f'<picture>\n'
            f'<source media="(max-width: 700px)" srcset="{paths[narrow.suffix].name}" width="{nw}" height="{nh}">\n'
            f'<img src="{paths[wide.suffix].name}" width="{w}" height="{h}" alt="{alt}">\n'
            f'</picture>')


def main() -> None:
    matplotlib.use('Agg')
    df = load()
    print(f'Drawing {len(SETS) * len(MEASURES) * len(LAYOUTS)} figures from ecdc.csv into '
          f'{OUT.relative_to(ROOT)}/ (last report {df["dateRep"].max():%Y-%m-%d})')
    pictures = []
    for set_name in SETS:
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
