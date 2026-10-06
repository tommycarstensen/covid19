"""Draw the page's aligned time-series figures, for the World1 and World2 country sets and for every region the 2020 page charted, as small multiples.

The 2020 figures (days100_*_perCapitaFalse_<set>.png, drawn by plot_series.doLinePlots) put 4 to 55 countries on one 480x360 plot with a 10-colour cycle, so countries shared colours and the legend, drawn over the lines, could not tell them apart. These give each country its own panel: the country in red over the other countries of its set in grey, the colours of plot_series.py's own comparison charts, all on one log scale and aligned on the week the country's cumulative count first passed 1,000 cases (100 deaths), as before. In a set of up to 20 countries, a country that never passed the threshold gets a panel that says so, instead of disappearing. A larger set (the EU, Europe, the Americas, Asia except China, Africa) gets a denser grid, eight columns wide and three on phones, and names its countries that never passed the threshold under the figure, with their totals. The regions are regions.py's, which are plot_series.py's under ECDC's names. The run stops if a panel's name runs into its total or any text runs off a figure.

Reads ecdc.csv (ECDC weekly cases and deaths per country to ISO week 2021-01, the same data as the rest of the page) and downloads nothing. The EU is summed from its 27 member states under their ECDC names; the 2020 list said 'Czech Republic', which ECDC calls Czechia, so the old EU total left Czechia out.

Writes site/aligned_{cases,deaths}_<set>.png (three rows of panels, or eight columns for a large set, for wide screens) and ..._narrow.png (two columns, or three, for phones), at 2x and 3x pixel density, and their <picture> tags to tmp/plot_days100_world_markup.html.

Usage: python3 plot_days100_world.py [set ...]   (default: every set)
"""

import math
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from itertools import accumulate
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from PIL import Image

import regions

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
    # plot_series.py's AsiaWestern without Iran, which World1 already shows.
    'AsiaWesternExIran': [
        'Armenia', 'Azerbaijan', 'Bahrain', 'Egypt', 'Qatar', 'Kuwait', 'Oman',
        'United_Arab_Emirates', 'Saudi_Arabia', 'Israel', 'Iraq', 'Georgia',
        'Turkey', 'Lebanon', 'Jordan', 'Palestine',
    ],
    # The other regional figures of the 2020 page, one per pair of days100_* charts, from regions.py.
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
}
# On the narrower panels these run into their totals.
NARROW_NAMES = {
    'United_Kingdom': 'UK', 'United_Arab_Emirates': 'UAE',
    'Bosnia_and_Herzegovina': 'Bosnia & Herz.', 'Trinidad_and_Tobago': 'Trinidad & T.',
    'Dominican_Republic': 'Dominican Rep.', 'Central_African_Republic': 'C. African Rep.',
    'Equatorial_Guinea': 'Eq. Guinea', 'North_Macedonia': 'N. Macedonia',
    'Sao_Tome_and_Principe': 'São Tomé & P.', 'Saint_Vincent_and_the_Grenadines': 'St Vincent & G.',
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
        return self.ncols or math.ceil((panels + 1) / 3)


# Inches at 100 dpi, so one inch is 100 CSS pixels at the size the page shows.
LAYOUTS = [
    Layout('', width=13.55, ncols=None, panel_height=1.9, dpi=200, dense_ncols=8, dense_panel_height=1.45),
    Layout('_narrow', width=4.15, ncols=2, panel_height=1.55, dpi=300, dense_ncols=3, dense_panel_height=1.3),
]

# Ink and marks. The country's red is plot_series.doLinePlots' (#e41a1c, over grey), as Tommy drew it in 2020.
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
CONTEXT = '#d3d1c9'
FOCUS = '#e41a1c'


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
    # A dense set gives panels only to the countries that passed the threshold, and names the others.
    order = sorted(drawn if dense else countries, key=lambda c: -totals[c])
    never = sorted((c for c in countries if c not in drawn), key=lambda c: -totals[c]) if dense else []

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
    others = len(order) - 1 if dense else len(countries) - 1
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
    # Room below the bottom panels for their tick labels and the source line, and for the list of countries that never passed.
    footer = 0.8 if narrow else 0.62
    if never:
        listed = ', '.join(f'{display_name(c)} ({totals[c]:,.0f})' for c in never)
        named = textwrap.fill(f'Never passed {threshold} {noun} (their totals on 10 January 2021): {listed}.',
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
        style(ax, decades, ymax, xmax, 7 if dense else 8)
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
        if i < len(order) - ncols:
            ax.tick_params(labelbottom=False)

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
    fig.savefig(path, dpi=layout.dpi, facecolor='white')
    plt.close(fig)
    return path


def collisions(fig: Figure) -> list[str]:
    """Panel titles that run into each other, and any visible text that reaches past an edge of the figure."""
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
    sets = sys.argv[1:] or list(SETS)
    unknown = [s for s in sets if s not in SETS]
    if unknown:
        raise SystemExit(f'Unknown set {unknown}; the sets are {list(SETS)}')
    matplotlib.use('Agg')
    df = load()
    print(f'Drawing {len(sets) * len(MEASURES) * len(LAYOUTS)} figures from ecdc.csv into '
          f'{OUT.relative_to(ROOT)}/ (last report {df["dateRep"].max():%Y-%m-%d})')
    pictures = []
    for set_name in sets:
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
