"""Redraw the table's missing line charts for tommycarstensen.com/covid19/.

Every row of the table on the page links two charts, of cumulated weekly
cases and deaths of one country against the rest of the world (until
27797c4 it showed their thumbnails, which this still draws but the page no
longer uses). The run of plot_series.py on 27 January 2021 drew these
charts only for the countries in its 'website' list, so for 63 of the 87
rows the page asked for thumbnails that were never made, and the full-size
charts behind them were left from 18 April 2020.

This draws both charts and both thumbnails for each such country with
plot_series.doLinePlots, the function that drew the others, into site/. A
country needs them when upload.py never moved its thumbnail into 2020/archive/
(upload.py moved every file it uploaded there). The data is ecdc.csv, the
ECDC weekly file of 14 January 2021, which runs to the week of 4 to 10
January 2021: the file of 27 January that the other charts came from was
overwritten in November 2021, so these charts end a week earlier than the
table's counts.

    python3 scripts/redraw_charts.py               # the missing countries, into site/
    python3 scripts/redraw_charts.py Denmark       # a trial, into tmp/redraw/ only

The log is tmp/redraw_charts.log.
"""

import argparse
import contextlib
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import TextIO

import matplotlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
ARCHIVE = ROOT / "2020" / "archive"
DATA = ROOT / "data" / "ecdc.csv"
WORK = ROOT / "tmp" / "redraw"
LOG = ROOT / "tmp" / "redraw_charts.log"
# A table row: its country and the file affix of the cases chart it links to. Since 27797c4 the row links the full-size charts instead of showing thumbnails, and since the rows got ids some open with <tr id="...">.
ROW = re.compile(
    r'<tr[^>]*><td>([^<]+)</td>.*?days100_cases_perCapitaFalse_([^"]+?)\.png"'
)
REGIONS = (
    "Europe",
    "AmericaNorth",
    "AmericaSouth",
    "Oceania",
    "Asia",
    "Africa",
)


class Tee:
    """Write to the terminal and to the log."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def table_rows() -> list[tuple[str, str]]:
    """(country as the data names it, file affix) for every table row."""
    html = (SITE / "index.html").read_text(encoding="utf-8")
    return ROW.findall(html)


def read_data() -> pd.DataFrame:
    """ecdc.csv prepared as plot_series.main() prepared its data."""
    df0 = pd.read_csv(DATA)
    df0["dateRep"] = pd.to_datetime(df0["dateRep"], format="%d/%m/%Y")
    names = df0["countriesAndTerritories"].str.replace("_", " ")
    df0["countriesAndTerritories"] = names
    return df0


def setup(df0: pd.DataFrame, log: TextIO) -> argparse.Namespace:
    """The arguments plot_series.main() built, with its country lists and
    each country's continent. parseArgs() needs a region or a list of
    countries; which one does not matter to doLinePlots."""
    import plot_series

    argv = sys.argv
    sys.argv = [argv[0], "--region", "EU"]
    try:
        with contextlib.redirect_stdout(log):
            args = plot_series.parseArgs()
    finally:
        sys.argv = argv
    plot_series.doCountry2Continent(args, df0)
    for region in REGIONS:
        for country in args.d_region2countries[region]:
            args.d_country2continent.setdefault(country, region)
    return args


def draw(
    args: argparse.Namespace, df0: pd.DataFrame, country: str, log: TextIO
) -> None:
    """Both charts and thumbnails, per capita and not, into WORK."""
    import matplotlib.pyplot as plt

    import plot_series

    here = Path.cwd()
    os.chdir(WORK)
    try:
        with contextlib.redirect_stdout(log):
            plot_series.doLinePlots(args, df0, country, comparison=True)
    finally:
        plt.clf()
        os.chdir(here)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redraw the covid19 page's missing line charts."
    )
    parser.add_argument(
        "countries",
        nargs="*",
        help="draw only these, into tmp/redraw/, as a trial",
    )
    trial = parser.parse_args().countries
    # plot_series imports pyplot, so the backend is set before it is loaded.
    matplotlib.use("Agg")
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as log:
        out = Tee(sys.stdout, log)
        print(f"log: {LOG}", file=out)
        print(f"start {time.strftime('%Y-%m-%d %H:%M:%S')}", file=out)
        if trial:
            wanted = [(name, name.replace(" ", "_")) for name in trial]
        else:
            wanted = [
                (name, affix)
                for name, affix in table_rows()
                if not (
                    ARCHIVE / f"days100_cases_perCapitaFalse_{affix}_thumb.png"
                ).exists()
            ]
        print(
            f"{len(wanted)} countries to draw from {DATA.name} into "
            f"{'tmp/redraw/' if trial else 'site/'}",
            file=out,
        )
        df0 = read_data()
        args = setup(df0, log)
        if WORK.exists():
            shutil.rmtree(WORK)
        WORK.mkdir(parents=True)
        failed: list[str] = []
        started = time.time()
        for done, (name, affix) in enumerate(wanted, 1):
            try:
                draw(args, df0, name, log)
            except KeyError as error:
                print(f"  {name}: no continent for {error}", file=out)
                failed.append(name)
                continue
            files = [
                f"days100_{k}_perCapitaFalse_{affix}{end}.png"
                for k in ("cases", "deaths")
                for end in ("", "_thumb")
            ]
            absent = [f for f in files if not (WORK / f).exists()]
            if absent:
                print(f"  {name}: not drawn: {', '.join(absent)}", file=out)
                failed.append(name)
                continue
            if not trial:
                for f in files:
                    shutil.copy2(WORK / f, SITE / f)
            if done % 10 == 0 or done == len(wanted):
                rate = (time.time() - started) / done
                left = rate * (len(wanted) - done)
                print(
                    f"  {done}/{len(wanted)} ({100 * done / len(wanted):.0f}%)"
                    f", {rate:.1f} s each, about {left:.0f} s left",
                    file=out,
                )
        status = 1 if failed else 0
        drawn = len(wanted) - len(failed)
        print(f"{drawn} drawn, {len(failed)} failed: {failed}", file=out)
        print(f"exit status: {status}", file=out)
    sys.exit(status)


if __name__ == "__main__":
    main()
