"""Draw the page's EU scatter plots: each member state's highest weekly COVID-19 cases and deaths against its population.

The 2020 scatter plots (scatter_EU_{cases,deaths}.png, drawn by plot_series.doScatterPlots) left out Czechia, which plot_series.py spelt 'Czech Republic', took populations of about 2014 from countryinfo, labelled their axes only 'Cases' and 'Deaths', and printed the names too small to read. These show the same thing in the style of the page's other October 2026 figures: one point per country, its highest single week against its population, both on log scales, in the 2020 charts' blue, every country named. Dashed lines mark equal rates per million people, so a point above a line had a higher weekly rate than it.

Reads ecdc.csv (ECDC weekly cases and deaths to ISO week 2021-01, with ECDC's populations of 2019) and the EU in regions.py, and downloads nothing.

Writes site/scatter_EU.png (wide screens, 2x pixel density, cases and deaths side by side) and site/scatter_EU_narrow.png (phones, 3x, stacked), and their <picture> tag to tmp/plot_scatter_markup.html. Stops if a name cannot be placed clear of the other names and points, or if any text runs off the figure.

Usage: python3 plot_scatter.py
"""

import math
import random
import textwrap
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg, RendererAgg
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.ticker import NullLocator
from matplotlib.transforms import Bbox

from plot_heat import (
    EDGE,
    INK,
    INK_2,
    MUTED,
    css_size,
    escaping_text,
    monday,
    place,
    text_height,
)
from regions import REGIONS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'site'

BLUE = '#1f77b4'  # matplotlib's first colour, which the 2020 scatter plots' points had
GUIDE = '#b9b8b2'
GRID = '#ebeae6'
MARKER = 42  # square points: about 6.5 points across, 9 CSS pixels
XLIM = (3e5, 3e8)  # room right of Germany, the largest, for names
SHUFFLES = 30  # orders tried when placing the names, besides the most crowded first
POPULATION_TICKS = [5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8]


@dataclass(frozen=True)
class Measure:
    name: str
    column: str
    ylim: tuple[float, float]
    rates: tuple[int, ...]  # per million people per week, one dashed line each


MEASURES = [
    Measure('cases', 'cases_weekly', (1e3, 1e6), (1_000, 10_000)),
    Measure('deaths', 'deaths_weekly', (10.0, 2e4), (10, 100)),
]


@dataclass(frozen=True)
class Country:
    name: str
    population: int
    peaks: dict[str, int]  # per measure, the highest weekly count
    weeks: dict[str, str]  # per measure, the ISO week of that count


@dataclass(frozen=True)
class Layout:
    suffix: str
    width: float  # inches; one inch is 100 CSS pixels at the size the page shows
    panel_h: float
    dpi: int
    font: float
    label: float


LAYOUTS = [
    Layout('', width=13.55, panel_h=5.4, dpi=200, font=8.5, label=8),
    Layout('_narrow', width=4.15, panel_h=4.6, dpi=300, font=7, label=6.5),
]


def directions() -> list[tuple[float, float, str, str]]:
    """Sixteen directions around a point, nearest the horizontal first, each with the alignment that keeps the text clear of the point."""
    out = []
    for degrees in sorted((i * 22.5 for i in range(16)), key=lambda d: (min(d % 180, 180 - d % 180), d)):
        dx, dy = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
        ha = 'left' if dx > 0.3 else 'right' if dx < -0.3 else 'center'
        va = 'bottom' if dy > 0.3 else 'top' if dy < -0.3 else 'center'
        out.append((dx, dy, ha, va))
    return out


DIRECTIONS = directions()


def load() -> tuple[list[str], list[Country]]:
    df = pd.read_csv(ROOT / 'ecdc.csv')
    weeks = sorted(str(week) for week in df['year_week'].unique())
    eu = df.loc[df['countriesAndTerritories'].isin(REGIONS['EU'])]
    missing = sorted(set(REGIONS['EU']) - set(eu['countriesAndTerritories']))
    if missing:
        raise SystemExit(f'ecdc.csv has no rows for {missing}')
    countries = []
    for name, rows in eu.groupby('countriesAndTerritories'):
        peaks: dict[str, int] = {}
        at: dict[str, str] = {}
        for measure in MEASURES:
            row = rows.loc[rows[measure.column].idxmax()]
            peaks[measure.name] = int(row[measure.column])
            at[measure.name] = str(row['year_week'])
        countries.append(Country(str(name), int(rows['popData2019'].iloc[0]), peaks, at))
    return weeks, sorted(countries, key=lambda c: -c.population)


def renderer_of(fig: Figure) -> RendererAgg:
    """The renderer that measures text, as the figure will be saved."""
    canvas = fig.canvas
    if not isinstance(canvas, FigureCanvasAgg):
        raise SystemExit('plot_scatter.py measures its text with the Agg backend')
    return canvas.get_renderer()


def number(value: float) -> str:
    return f'{value:,.0f}' if value >= 1 else f'{value:g}'


def style(ax: Axes, measure: Measure, layout: Layout, show_xlabel: bool) -> None:
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*XLIM)
    ax.set_ylim(*measure.ylim)
    ax.set_xticks(POPULATION_TICKS, [f'{tick / 1e6:g}' for tick in POPULATION_TICKS])
    ax.xaxis.set_minor_locator(NullLocator())
    lo, hi = math.ceil(math.log10(measure.ylim[0])), math.floor(math.log10(measure.ylim[1]))
    yticks = [10.0 ** k for k in range(lo, hi + 1)]
    ax.set_yticks(yticks, [number(tick) for tick in yticks])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(True, which='major', color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelcolor=MUTED, labelsize=layout.font, width=0.6, length=3)
    if show_xlabel:
        ax.set_xlabel('Population in 2019, millions', fontsize=layout.font, color=INK_2, labelpad=4)
    ax.set_ylabel(f'{measure.name.capitalize()} in the highest week', fontsize=layout.font, color=INK_2, labelpad=4)


def slant_boxes(anchor: tuple[float, float], angle: float, w: float, h: float, above: bool) -> list[Bbox]:
    """The space a text of this width and height takes when it ends at the anchor and rises at this angle, as small boxes along its slant (one box round all of it would cover a wide diagonal band)."""
    ax_, ay_ = anchor
    normal = (h / 2) if above else (-h / 2)
    steps = 12
    boxes = []
    for i in range(steps):
        along = -(i + 0.5) * w / steps
        cx = ax_ + along * math.cos(angle) - normal * math.sin(angle)
        cy = ay_ + along * math.sin(angle) + normal * math.cos(angle)
        hw = w / steps / 2 * abs(math.cos(angle)) + h / 2 * abs(math.sin(angle))
        hh = w / steps / 2 * abs(math.sin(angle)) + h / 2 * abs(math.cos(angle))
        boxes.append(Bbox.from_extents(cx - hw, cy - hh, cx + hw, cy + hh))
    return boxes


def guides(fig: Figure, ax: Axes, measure: Measure, layout: Layout, marks: list[Bbox]) -> list[Bbox]:
    """A dashed line for each rate per million people per week, named above the highest line and below the others, on the side where no country lies, at the first place from its upper end where the name covers no point. Returns the space the names take."""
    renderer = renderer_of(fig)
    frame = ax.get_window_extent(renderer)
    taken: list[Bbox] = []
    xs = np.geomspace(*XLIM, 60)
    for k, rate in enumerate(measure.rates):
        ax.plot(xs, rate * xs / 1e6, color=GUIDE, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
        label = f'{rate:,} per million'
        flat = ax.text(0, 0, label, fontsize=layout.font - 1, transform=None)
        extent = flat.get_window_extent(renderer)
        flat.remove()
        # Where the line is inside the axes, from the population where it enters to the one where it leaves.
        x_in = max(XLIM[0], measure.ylim[0] / rate * 1e6)
        x_out = min(XLIM[1], measure.ylim[1] / rate * 1e6)
        upper = k == len(measure.rates) - 1
        tries = [(share, side) for side in (upper, not upper) for share in (0.92, 0.8, 0.68, 0.56, 0.44, 0.32, 0.2)]
        for share, above in tries:
            end = x_in * (x_out / x_in) ** share
            (x0, y0), (x1, y1) = ax.transData.transform([(end / 2, rate * end / 2e6), (end, rate * end / 1e6)])
            angle = math.atan2(y1 - y0, x1 - x0)
            y = rate * end / 1e6 * (1.1 if above else 1 / 1.1)
            boxes = slant_boxes(tuple(ax.transData.transform((end, y))), angle, extent.width, extent.height, above)
            inside = all(frame.x0 <= b.x0 and b.x1 <= frame.x1 and frame.y0 <= b.y0 and b.y1 <= frame.y1 for b in boxes)
            if inside and not any(b.overlaps(m) for b in boxes for m in marks):
                ax.text(end, y, label, rotation=math.degrees(angle), rotation_mode='anchor', ha='right',
                        va='bottom' if above else 'top', fontsize=layout.font - 1, color=MUTED)
                taken += boxes
                break
        else:
            raise SystemExit(f'scatter: no room along the line for {label!r} ({measure.name}, {layout.suffix or "wide"})')
    return taken


Segment = tuple[tuple[float, float], tuple[float, float]]


def crosses(a: Segment, b: Segment) -> bool:
    """Whether two line segments cross."""
    def side(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    (p1, p2), (q1, q2) = a, b
    return side(p1, p2, q1) * side(p1, p2, q2) < 0 and side(q1, q2, p1) * side(q1, q2, p2) < 0


def gap(point: np.ndarray, box: Bbox) -> float:
    """The distance from a point to the nearest edge of a box, 0 inside it."""
    x, y = point
    return math.hypot(max(box.x0 - x, 0, x - box.x1), max(box.y0 - y, 0, y - box.y1))


def marker_boxes(fig: Figure, ax: Axes, countries: list[Country], measure: Measure) -> dict[str, Bbox]:
    """The space each country's point takes, in display pixels."""
    radius = math.sqrt(MARKER) / 2 * fig.dpi / 72
    boxes = {}
    for c in countries:
        x, y = ax.transData.transform((c.population, c.peaks[measure.name]))
        boxes[c.name] = Bbox.from_bounds(x - radius, y - radius, 2 * radius, 2 * radius)
    return boxes


def name_points(fig: Figure, ax: Axes, countries: list[Country], measure: Measure, layout: Layout,
                taken: list[Bbox]) -> list[str]:
    """Put each country's name beside its point, where it overlaps no other name, point or guide label; failing that, further out with a thin leader line. Names are placed one at a time, so the result depends on their order: the most crowded points first, and SHUFFLES fixed shuffles, of which the layout that places every name with the fewest leader lines, then the shortest longest one, then the shortest in all, is kept. Returns the names that found no room."""
    fig.canvas.draw()
    renderer = renderer_of(fig)
    points = {c.name: ax.transData.transform((c.population, c.peaks[measure.name])) for c in countries}
    marks = marker_boxes(fig, ax, countries, measure)
    frame = ax.get_window_extent(renderer)
    r_pt = math.sqrt(MARKER) / 2
    distances = (r_pt + 2, r_pt + 9, r_pt + 17, r_pt + 27, r_pt + 38, r_pt + 50, r_pt + 65, r_pt + 80, r_pt + 100, r_pt + 125)
    clearance = 2 * fig.dpi / 72  # so a leader cannot run just under or over a name
    near = 16 * fig.dpi / 72  # how close a leader's name may come to another point
    margin = 5 * fig.dpi / 72  # how much nearer its own point than any other a name without a leader must be

    def fits(c: Country, box: Bbox, distance: float, placed: list[Bbox], leaders: list[tuple[float, float]],
             segments: list[Segment], strict: bool) -> tuple[bool, list[tuple[float, float]]]:
        if not (frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0 and box.y1 <= frame.y1):
            return False, []
        if any(box.overlaps(b) for b in placed) or any(box.contains(x, y) for x, y in leaders):
            return False, []
        if any(box.overlaps(m) for n, m in marks.items() if n != c.name):
            return False, []
        if distance == distances[0]:
            # Without a leader line, a name must sit clearly nearer its own point than any other, or it reads as another's.
            own = gap(points[c.name], box) + margin
            return all(own < gap(p, box) for n, p in points.items() if n != c.name), []
        # A leader line may cross no other name, point or leader, nor run close along a name, and its name may not sit beside another point.
        (px, py), (qx, qy) = points[c.name], ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
        if strict and any(crosses(((px, py), (qx, qy)), seg) for seg in segments):
            return False, []
        path = [(px + (qx - px) * f / 24, py + (qy - py) * f / 24) for f in range(3, 22)]
        others = [b.padded(clearance) for b in placed] + [m for n, m in marks.items() if n != c.name]
        if any(o.contains(x, y) for x, y in path for o in others):
            return False, []
        return all(gap(p, box) > near for n, p in points.items() if n != c.name), path

    def attempt(order: list[Country], strict: bool) -> tuple[list[Text], list[str], list[float]]:
        placed = list(taken)
        leaders: list[tuple[float, float]] = []
        segments: list[Segment] = []
        names: list[Text] = []
        unplaced: list[str] = []
        used: list[float] = []
        for c in order:
            done = False
            for distance in distances:
                for dx, dy, ha, va in DIRECTIONS:
                    leader = {'arrowstyle': '-', 'color': MUTED, 'linewidth': 0.5, 'shrinkA': 0, 'shrinkB': r_pt + 1}
                    t = ax.annotate(c.name, (c.population, c.peaks[measure.name]),
                                    xytext=(dx * distance, dy * distance), textcoords='offset points', ha=ha, va=va,
                                    fontsize=layout.label, color=INK_2,
                                    arrowprops=leader if distance > distances[0] else None, zorder=4)
                    # The text's own box: an annotation's extent would include its leader line.
                    t.update_positions(renderer)
                    box = Text.get_window_extent(t, renderer).padded(1)
                    ok, path = fits(c, box, distance, placed, leaders, segments, strict)
                    if ok:
                        placed.append(box)
                        leaders += path
                        if path:
                            segments.append((path[0], path[-1]))
                        names.append(t)
                        used.append(distance)
                        done = True
                        break
                    t.remove()
                if done:
                    break
            if not done:
                unplaced.append(c.name)
        return names, unplaced, used

    def crowding(c: Country) -> int:
        x, y = points[c.name]
        return -sum(1 for px, py in points.values() if abs(px - x) < 60 * fig.dpi / 100 and abs(py - y) < 25 * fig.dpi / 100)

    shuffles = [random.Random(seed).sample(countries, len(countries)) for seed in range(SHUFFLES)]
    orders = [sorted(countries, key=crowding), *shuffles]
    # Leader lines may cross only if no order places every name without.
    best: tuple[tuple[int, int, int, float, float], list[Country], bool] | None = None
    for strict in (True, False):
        for order in orders:
            names, unplaced, used = attempt(order, strict)
            for t in names:
                t.remove()
            leaders = [d for d in used if d > distances[0]]
            score = (len(unplaced), 0 if strict else 1, len(leaders), max(leaders, default=0.0), sum(leaders))
            if best is None or score < best[0]:
                best = (score, order, strict)
        if best is not None and best[0][0] == 0:
            break
    assert best is not None
    return attempt(best[1], best[2])[1]


def draw(countries: list[Country], weeks: list[str], layout: Layout) -> Path:
    narrow = layout.suffix != ''
    last_day = monday(weeks[-1]) + timedelta(days=6)
    first_day = monday(weeks[0])
    if narrow:
        title = "Each EU country's highest weekly\nCOVID-19 cases and deaths,\nagainst its population"
        subtitle = textwrap.fill(
            f'One point per member state: its highest single week from {first_day:%-d %b %Y} to {last_day:%-d %b %Y}. '
            'Both scales are logarithmic. The dashed lines mark equal weekly rates per million people.', 52)
        source = textwrap.fill('Data: ECDC, weekly, populations of 2019. Drawn October 2026.', 62)
    else:
        title = "Each EU country's highest weekly COVID-19 cases and deaths, against its population"
        subtitle = (f'One point per member state: its highest single week from {first_day:%-d %B %Y} to {last_day:%-d %B %Y} '
                    f'(ISO weeks {weeks[0]} to {weeks[-1]}).\n'
                    'Both scales are logarithmic. The dashed lines mark equal weekly rates per million people.')
        source = ('Data: European Centre for Disease Prevention and Control (ECDC), weekly cases and deaths by country, '
                  'populations of 2019. Drawn October 2026.')
    title_size, subtitle_size = (13, 9.5) if not narrow else (11.5, 8.5)
    source_size = 8 if not narrow else 7.5
    subtitle_top = 0.12 + text_height(title, title_size, 1.2) + 0.12
    header = subtitle_top + text_height(subtitle, subtitle_size, 1.35) + 0.2
    footer = 0.3 + text_height(source, source_size, 1.35)

    head_h = 0.32  # each panel's own title
    axis_h = 0.5 if not narrow else 0.42  # tick labels and the x label under a panel
    left = 0.95 if not narrow else 0.78  # tick labels and the y label left of a panel
    block = head_h + layout.panel_h + axis_h
    gap = 0.25
    height = header + (2 * block + gap if narrow else block) + footer
    size = (layout.width, height)
    fig = plt.figure(figsize=size, dpi=100, facecolor='white')
    if narrow:
        panel_w = layout.width - EDGE - left - 0.15
        spots = [(EDGE, header), (EDGE, header + block + gap)]
    else:
        middle = 0.45
        panel_w = (layout.width - 2 * EDGE - 2 * left - middle - 0.15) / 2
        spots = [(EDGE, header), (EDGE + left + panel_w + middle, header)]

    unplaced = []
    for measure, (x, top) in zip(MEASURES, spots, strict=True):
        ax = place(fig, x + left, top + head_h, panel_w, layout.panel_h, size)
        style(ax, measure, layout, show_xlabel=True)
        ax.scatter([c.population for c in countries], [c.peaks[measure.name] for c in countries], s=MARKER,
                   marker='s', color=BLUE, edgecolors='white', linewidths=0.6, zorder=3)
        fig.canvas.draw()
        taken = guides(fig, ax, measure, layout, list(marker_boxes(fig, ax, countries, measure).values()))
        unplaced += [f'{name} ({measure.name})' for name in name_points(fig, ax, countries, measure, layout, taken)]
        fig.text(x / layout.width, 1 - (top + 0.05) / height, f'Highest weekly {measure.name}',
                 ha='left', va='top', fontsize=10 if not narrow else 8.5, fontweight='bold', color=INK)

    edge = EDGE / layout.width
    fig.text(edge, 1 - 0.12 / height, title, ha='left', va='top', fontsize=title_size, fontweight='bold', color=INK,
             linespacing=1.2)
    fig.text(edge, 1 - subtitle_top / height, subtitle, ha='left', va='top', fontsize=subtitle_size, color=INK_2,
             linespacing=1.35)
    fig.text(edge, 0.1 / height, source, ha='left', va='bottom', fontsize=source_size, color=MUTED, linespacing=1.35)

    if unplaced:
        draft = ROOT / 'tmp' / f'scatter_EU{layout.suffix}_draft.png'
        fig.savefig(draft, dpi=100, facecolor='white')
        raise SystemExit(f'scatter_EU{layout.suffix}: no room for {unplaced}; the draft is {draft.relative_to(ROOT)}')
    clipped = escaping_text(fig)
    if clipped:
        raise SystemExit(f'scatter_EU{layout.suffix}: text runs off the figure: {clipped}')
    path = OUT / f'scatter_EU{layout.suffix}.png'
    fig.savefig(path, dpi=layout.dpi, facecolor='white')
    plt.close(fig)
    return path


def extreme(countries: list[Country], measure: str, weeks: list[str], highest: bool) -> str:
    c = (max if highest else min)(countries, key=lambda c: c.peaks[measure] / c.population)
    rate = c.peaks[measure] / c.population * 1e6
    return f'{c.name}, {rate:,.0f} per million in the week of {monday(c.weeks[measure]):%-d %B %Y}'


def picture(countries: list[Country], weeks: list[str], paths: dict[str, Path]) -> str:
    """A <picture> that serves the stacked figure below 700 CSS pixels."""
    wide, narrow = LAYOUTS
    w, h = css_size(paths[wide.suffix], wide.dpi)
    nw, nh = css_size(paths[narrow.suffix], narrow.dpi)
    last_day = monday(weeks[-1]) + timedelta(days=6)
    alt = (f"Scatter plots on log scales of each EU country's highest weekly COVID-19 cases and deaths, from "
           f'{monday(weeks[0]):%-d %B %Y} to {last_day:%-d %B %Y}, against its population of 2019, with dashed lines '
           'at 1,000 and 10,000 cases and at 10 and 100 deaths per million people per week. '
           f'Highest weekly cases per million: {extreme(countries, "cases", weeks, True)}; lowest peak: '
           f'{extreme(countries, "cases", weeks, False)}. Highest weekly deaths per million: '
           f'{extreme(countries, "deaths", weeks, True)}; lowest peak: {extreme(countries, "deaths", weeks, False)}. '
           f'Countries, largest population first: {", ".join(c.name for c in countries)}.')
    return (f'<picture>\n'
            f'<source media="(max-width: 700px)" srcset="{paths[narrow.suffix].name}" width="{nw}" height="{nh}">\n'
            f'<img src="{paths[wide.suffix].name}" width="{w}" height="{h}" loading="lazy" alt="{alt}">\n'
            f'</picture>')


def main() -> None:
    matplotlib.use('Agg')
    weeks, countries = load()
    print(f'ecdc.csv: {len(countries)} EU countries, ISO weeks {weeks[0]} to {weeks[-1]}. '
          f'Drawing {len(LAYOUTS)} figures into {OUT.relative_to(ROOT)}/')
    paths = {}
    for layout in LAYOUTS:
        path = draw(countries, weeks, layout)
        paths[layout.suffix] = path
        print(f'  {path.name}: {"x".join(str(v) for v in css_size(path, layout.dpi))} CSS px')
    markup = ROOT / 'tmp' / 'plot_scatter_markup.html'
    markup.parent.mkdir(exist_ok=True)
    markup.write_text(picture(countries, weeks, paths) + '\n', encoding='utf-8')
    print(f'Done. The <picture> tag for site/index.html is in {markup.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
