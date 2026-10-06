"""Draw weekly COVID-19 tests, cases and deaths per million people for 16 countries, one panel each, on one log scale, to show how much of the rise in cases came from more testing.

On a log scale the gap between the tests and cases lines is the share of tests that were positive (one step of the scale is 10%, two steps 1%), and the gap between cases and deaths is the number of cases confirmed for each death. Where cases rise in step with tests, the share positive stays the same; where cases close in on tests, a larger share was positive, which more testing does not explain. Each panel names the share positive in the country's week of most cases with a test count.

The colours are ColorBrewer's Dark2, which plot_series.define_colors lists: the hues of Set2, whose teal and orange plot_series.py's 2020 bar charts used for cases and deaths, at a strength that reads as a thin line (Set2's pastels fail as lines, on contrast and on telling teal from lavender).

Reads ecdc.csv and owid.csv through plot_heat.load (ECDC's weekly cases and deaths per million people of 2019; Our World in Data's tests per thousand, as on the page's world maps), and downloads nothing. A week without a figure, or with 0, has no line.

Writes site/tests_cases_deaths.png (wide screens, 2x) and site/tests_cases_deaths_narrow.png (phones, 3x), and their <picture> tag to tmp/plot_tests_markup.html. Stops if any text runs into other text or off the figure.

Usage: python3 plot_tests.py
"""

import math
import textwrap
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
from PIL import Image

from plot_heat import CASES, DEATHS, POSITIVE, TESTS, Country, display_name, load

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'site'

COUNTRIES = [
    'Denmark', 'Sweden', 'United_Kingdom', 'Germany', 'France', 'Italy', 'Spain', 'Netherlands', 'Belgium', 'Poland',
    'Czechia', 'United_States_of_America', 'Canada', 'South_Korea', 'Japan', 'South_Africa',
]

INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#73716b'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
LINES = [  # name, its key in Country.rates, colour, per million people from that rate
    ('Tests', TESTS.name, '#7570b3', 1000.0),
    ('Cases', CASES.name, '#1b9e77', 1.0),
    ('Deaths', DEATHS.name, '#d95f02', 1.0),
]
YMIN, YMAX = 0.1, 200_000.0


@dataclass(frozen=True)
class Layout:
    suffix: str
    width: float  # inches; one inch is 100 CSS pixels at the size the page shows
    ncols: int
    panel_height: float
    dpi: int
    font: float


LAYOUTS = [
    Layout('', width=13.55, ncols=4, panel_height=1.95, dpi=200, font=8),
    Layout('_narrow', width=4.15, ncols=2, panel_height=1.5, dpi=300, font=6.5),
]


def thursday(year_week: str) -> date:
    """The middle of an ISO week written as ECDC writes it, '2020-53'."""
    year, week = (int(part) for part in year_week.split('-'))
    return date.fromisocalendar(year, week, 4)


def day_number(d: date) -> float:
    """Matplotlib's number for a day: days since its epoch, so the date locator and formatter read it."""
    return float((d - date.fromisoformat(mdates.get_epoch()[:10])).days)


def tick(value: float, _pos: float) -> str:
    return f'{value / 1000:g}k' if value >= 1000 else f'{value:g}'


def peak_share(c: Country) -> float | None:
    """The share of tests positive in the country's week of most cases among the weeks with a test count."""
    weeks = [(cases, share) for cases, share in zip(c.rates[CASES.name], c.rates[POSITIVE.name])
             if cases is not None and share is not None]
    return max(weeks)[1] if weeks else None


def share_text(share: float) -> str:
    return f'{share:.0%}' if share >= 0.01 else f'{share:.1%}'


def panel(ax: Axes, c: Country, days: list[float], layout: Layout, first_column: bool, labels: bool) -> None:
    for name, key, colour, scale in LINES:
        ys = [v * scale if v is not None and v > 0 else np.nan for v in c.rates[key]]
        ax.plot(days, ys, color=colour, linewidth=1.6 if not layout.suffix else 1.2, solid_capstyle='round')
        if labels:
            last = max(i for i, v in enumerate(ys) if not np.isnan(v))
            ax.annotate(name.lower(), (days[last], ys[last]), xytext=(4, 0), textcoords='offset points', va='center',
                        fontsize=layout.font, color=INK_2)
    ax.set_yscale('log')
    ax.set_ylim(YMIN, YMAX)
    ax.yaxis.set_major_locator(FixedLocator([10.0 ** e for e in range(-1, 6)]))
    ax.yaxis.set_major_formatter(FuncFormatter(tick) if first_column else NullFormatter())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(days[0], days[-1] + (26 if labels else 3))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=MUTED, labelsize=layout.font, length=2.5, width=0.6)


def draw(countries: dict[str, Country], weeks: list[str], layout: Layout) -> Path:
    narrow = layout.suffix != ''
    width, ncols = layout.width, layout.ncols
    nrows = math.ceil(len(COUNTRIES) / ncols)
    first, last = date.fromisocalendar(2020, 1, 1), thursday(weeks[-1]) + timedelta(days=3)
    if narrow:
        title = 'Weekly COVID-19 tests, cases\nand deaths per million people'
        subtitle = textwrap.fill(
            f'One panel per country, {first:%-d %b %Y} to {last:%-d %b %Y}, on one log scale. The gap between tests '
            'and cases is the share of tests positive: one step is 10%, two steps 1%. Where cases close in on tests, '
            'more tests were positive, which more testing does not explain. The gap between cases and deaths is the '
            'cases confirmed per death. Under each name: the share positive in the week of most cases.', 54)
        source = textwrap.fill(
            'Data: ECDC, weekly cases and deaths, per million people of 2019; Our World in Data, tests. Countries count '
            'tests differently, so the gap compares better within a country than between countries. '
            'Drawn October 2026.', 62)
        title_size, subtitle_size, source_size = 11.5, 8.5, 7.5
        left, right, col_gap, row_gap, panel_top = 0.42, 0.1, 0.16, 0.62, 0.36
    else:
        title = 'Weekly COVID-19 tests, cases and deaths per million people'
        subtitle = textwrap.fill(
            f'One panel per country, week by week from {first:%-d %B %Y} to {last:%-d %B %Y}, on one logarithmic '
            'scale. The gap between tests and cases is the share of tests that were positive: one step of the scale is '
            '10%, two steps 1%. Where cases rise in step with tests, the share stays the same and the extra cases may '
            'be the extra testing; where cases close in on tests, more of the tests were positive, which more testing '
            'does not explain. The gap between cases and deaths is the number of cases confirmed for each death. Top '
            'right: the share of tests positive in the week of most cases.', 178)
        source = textwrap.fill(
            'Data: European Centre for Disease Prevention and Control (ECDC), weekly cases and deaths, per million '
            'people of 2019; Our World in Data, tests, from running totals, or from daily counts where it has no '
            'running total. Countries count tests differently (tests performed, people tested, samples), so the gap '
            'compares better within a country than between countries. Drawn October 2026.', 205)
        title_size, subtitle_size, source_size = 13, 9.5, 8
        left, right, col_gap, row_gap, panel_top = 0.62, 0.12, 0.3, 0.45, 0.3
    subtitle_top = 0.12 + (title.count('\n') + 1) * title_size * 1.2 / 72 + 0.12
    legend_y = subtitle_top + (subtitle.count('\n') + 1) * subtitle_size * 1.35 / 72 + 0.2
    header = legend_y + 0.2 + panel_top
    footer = 0.3 + (source.count('\n') + 1) * source_size * 1.35 / 72 + 0.12
    height = header + nrows * layout.panel_height + (nrows - 1) * row_gap + footer
    fig = plt.figure(figsize=(width, height), dpi=100, facecolor='white')
    pw = (width - left - right - (ncols - 1) * col_gap) / ncols
    days = [day_number(thursday(w)) for w in weeks]

    for k, country in enumerate(COUNTRIES):
        r, col = divmod(k, ncols)
        x0, y0 = left + col * (pw + col_gap), header + r * (layout.panel_height + row_gap)
        ax = fig.add_axes((x0 / width, 1 - (y0 + layout.panel_height) / height, pw / width, layout.panel_height / height))
        c = countries[country]
        panel(ax, c, days, layout, col == 0, k == 0 and not narrow)
        share = peak_share(c)
        note = f'{share_text(share)} positive at peak' if share is not None else 'no test count'
        if narrow:
            ax.annotate(display_name(c.name), (0, 1), xycoords='axes fraction', xytext=(0, 13), textcoords='offset points',
                        ha='left', va='bottom', fontsize=8, fontweight='bold', color=INK)
            ax.annotate(note, (0, 1), xycoords='axes fraction', xytext=(0, 4), textcoords='offset points', ha='left',
                        va='bottom', fontsize=layout.font, color=INK_2)
        else:
            ax.set_title(display_name(c.name), loc='left', fontsize=10, fontweight='bold', color=INK, pad=4)
            ax.set_title(note, loc='right', fontsize=7.5, color=INK_2, pad=5)

    edge = 0.12 / width
    fig.text(edge, 1 - 0.12 / height, title, ha='left', va='top', fontsize=title_size, fontweight='bold', color=INK,
             linespacing=1.2)
    fig.text(edge, 1 - subtitle_top / height, subtitle, ha='left', va='top', fontsize=subtitle_size, color=INK_2,
             linespacing=1.35)
    x = 0.12
    for name, _, colour, _ in LINES:
        fig.add_artist(Line2D([x / width, (x + 0.3) / width], [1 - legend_y / height] * 2, color=colour, linewidth=2))
        label = fig.text((x + 0.38) / width, 1 - legend_y / height, name, ha='left', va='center',
                         fontsize=layout.font + 1, color=INK)
        fig.canvas.draw()
        x += 0.38 + label.get_window_extent().width / fig.dpi + 0.35
    fig.text(edge, 0.1 / height, source, ha='left', va='bottom', fontsize=source_size, color=MUTED, linespacing=1.35)

    path = OUT / f'tests_cases_deaths{layout.suffix}.png'
    problems = collisions(fig)
    if problems:
        raise SystemExit(f'{path.name}: {problems}')
    fig.savefig(path, dpi=layout.dpi, facecolor='white')
    plt.close(fig)
    return path


def collisions(fig: Figure) -> list[str]:
    """Any two visible texts that run into each other, and any that reach past an edge of the figure."""
    fig.canvas.draw()
    box = fig.bbox
    texts: list[Text] = list(fig.texts)
    for ax in fig.axes:
        texts += [t for t in ax.get_children() if isinstance(t, Text)]
        texts += [*ax.get_xticklabels(), *ax.get_yticklabels()]
    shown = [(t.get_text().replace('\n', ' '), t.get_window_extent()) for t in texts if t.get_visible() and t.get_text()]
    found = []
    for k, (a, ea) in enumerate(shown):
        if ea.x0 < box.x0 - 0.5 or ea.x1 > box.x1 + 0.5 or ea.y0 < box.y0 - 0.5 or ea.y1 > box.y1 + 0.5:
            found.append(f'{a!r} runs off the figure')
        found += [f'{a!r} runs into {b!r}' for b, eb in shown[k + 1:] if ea.overlaps(eb)]
    return found


def css_size(path: Path, dpi: int) -> tuple[int, int]:
    """The image's size in CSS pixels: 100 per inch, as the figures are laid out."""
    with Image.open(path) as im:
        return round(im.size[0] * 100 / dpi), round(im.size[1] * 100 / dpi)


def picture(countries: dict[str, Country], paths: dict[str, Path]) -> str:
    """A <picture> that serves the two-column figure below 700 CSS pixels."""
    wide, narrow = LAYOUTS
    w, h = css_size(paths[wide.suffix], wide.dpi)
    nw, nh = css_size(paths[narrow.suffix], narrow.dpi)
    shares = []
    for country in COUNTRIES:
        share = peak_share(countries[country])
        shares.append(f'{display_name(country)} {share_text(share) if share is not None else "no test count"}')
    alt = ('Weekly COVID-19 tests, cases and deaths per million people in 16 countries, one panel each, on one log '
           'scale, from 30 December 2019 to 10 January 2021. The gap between tests and cases is the share of tests '
           'positive. Share positive in each country\'s week of most cases: ' + ', '.join(shares) + '.')
    return (f'<picture>\n'
            f'<source media="(max-width: 700px)" srcset="{paths[narrow.suffix].name}" width="{nw}" height="{nh}">\n'
            f'<img src="{paths[wide.suffix].name}" width="{w}" height="{h}" loading="lazy" alt="{alt}">\n'
            f'</picture>')


def main() -> None:
    matplotlib.use('Agg')
    weeks, countries = load()
    print(f'ecdc.csv and owid.csv: {len(COUNTRIES)} countries, ISO weeks {weeks[0]} to {weeks[-1]}. '
          f'Drawing {len(LAYOUTS)} figures into {OUT.relative_to(ROOT)}/')
    paths = {layout.suffix: draw(countries, weeks, layout) for layout in LAYOUTS}
    for layout, path in zip(LAYOUTS, paths.values()):
        print(f'  {path.name} {css_size(path, layout.dpi)[0]}x{css_size(path, layout.dpi)[1]} CSS px')
    markup = ROOT / 'tmp' / 'plot_tests_markup.html'
    markup.parent.mkdir(exist_ok=True)
    markup.write_text(picture(countries, paths) + '\n', encoding='utf-8')
    print(f'Done. The <picture> tag for site/index.html is in {markup.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
