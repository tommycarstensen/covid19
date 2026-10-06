# Decisions about the covid19 page

Choices made while repairing and editing https://tommycarstensen.com/covid19/ (`site/index.html`) in October 2026, with the reason for each and what was considered instead. Newest last. `todo.md` holds work in progress; this file holds what was decided and why.

## How the October 2026 changes were chosen

On 6 October 2026 Tommy asked for the page's bugs to be fixed and for advisers to propose content to add, remove or restructure. Three advisers (Sonnet sub-agents) read the page from three angles: a 2026 visitor arriving from a search or a 2020 citation, information architecture and editing, and an epidemiologist and archivist checking provenance. Their suggestions were merged into one list (C1 to C13, below) alongside the bugs found by an automated audit (B1 to B6). Tommy chose to start with the ones recommended as the best value and work down the list. Covid19-4f does them one commit each, in this order: B1, B6, C1, C2, C4, C8, C9, C11, then C3 and C10, then the restructure (C5, C6, C7).

What the audit found sound, on 6 October 2026: no broken images, no horizontal overflow at 1,440 or 390 pixels wide, every file the page uses either in `site/` or on the server, every table row with all 14 cells, and all five map videos playable.

## 2026-10-06: the USA's continent in the table is Americas

`plot_series.py` set the USA's continent to "North America" by hand (`doCountry2Continent`), while `countryinfo` gives "Americas" for every other country of the Americas. Sorting the table by continent therefore put the USA on its own. The page now says Americas. The cause in `plot_series.py` is left for its own commit, because that script carries 80 ruff and 35 pycodestyle findings that have to be cleaned when it is touched (B5).

## 2026-10-06: the EU table row is labelled "EU without Czechia" and divides by its own population

The EU row summed the 27 countries in `plot_series.py`'s list, but the list spells Czechia "Czech Republic", which matches nothing in ECDC's data, so Czechia's cases (835,454 to 10 January 2021) and deaths were left out. The row then divided by a hard-coded 512.6 million, the population of the EU of 28 including the UK. Cases per million showed 33,062.9, about 15% too low for the 26 countries actually summed.

Chosen: keep the counts as published, label the row "EU without Czechia", and divide by the population of those 26 countries from `countryinfo` 0.1.2 (434.854 million), the source every other row of the table uses. Cases per million become 38,974.0 and deaths per million 944.2. The case fatality rate and tests per thousand do not depend on the denominator and are unchanged; the tests figure is OWID's population-weighted average over the same list, which also misses Czechia (OWID calls it "Czechia" too).

Considered and rejected: adding Czechia's cases, because the January 2021 download behind the table (one week later than `ecdc.csv`) is lost and Czechia's figure for that week cannot be recovered exactly; dividing by ECDC's 2019 populations (436.2 million), because the rest of the table uses `countryinfo`'s older figures and the row would no longer be comparable with them (that question is C12).

The EU charts under the European Union heading plot each member country's line from the same list, so they also lack Czechia. The aligned small multiples drawn in October 2026 (`plot_days100_world.py`) include it.
