# data

What the scripts in `scripts/` read. The scripts find this folder from their own location, so they can be run from anywhere, but run them from the repo root as `python3 scripts/<name>.py`. The Europe, Denmark and Peru maps keep their own inputs in `2020/regional_maps/`.

Most of these files cannot be downloaded again: their sources have been withdrawn or now serve later or revised data. Only `ecdc.csv` is in git (force-added past the `*.csv` rule); the others exist only on this disk and in the disk copy of 6 October 2026 (`/Volumes/black2tb/archive/mbp2019/covid19_backup_2026-10-06/`). `SHA256SUMS` records them as they are; check them with `cd data && shasum -a 256 -c SHA256SUMS`.

| File | What | Source | As of | In git | Read by |
|---|---|---|---|---|---|
| `ecdc.csv` | ECDC weekly cases and deaths per country, with 2019 populations, ISO weeks 2020-01 to 2021-01 | https://opendata.ecdc.europa.eu/covid19/casedistribution/csv, withdrawn | 14 January 2021 | yes, the only copy | `build_bubbles`, `build_world_map`, `plot_days100_world`, `plot_heat` (and through it `plot_tests`), `plot_scatter`, `plot_series`, `plot_choropleth`, `redraw_charts`, `regions` |
| `owid.csv` | Our World in Data, 1 January 2020 to 5 November 2021 | https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv, which now serves later data | 6 November 2021 | no | `build_bubbles`, `build_world_map` (and through it `plot_heat` and `plot_tests`), `plot_bubble`, `plot_choropleth`, `plot_series` |
| `bsg.csv` | The Oxford COVID-19 Government Response Tracker (Blavatnik School of Government) | https://github.com/OxCGRT/covid-policy-tracker/raw/master/data/OxCGRT_latest.csv | 26 January 2021 | no | `plot_series` (the policy heat maps) |
| `csv` | ECDC's frozen daily file, to 14 December 2020 | https://opendata.ecdc.europa.eu/covid19/casedistribution/csv | 6 November 2021 | no | `plot_bubble` |
| `COVID-19-geographic-disbtribution-worldwide.xlsx` | ECDC's daily file | ECDC | 15 April 2020 | no | nothing |
| `natural_earth/` | Natural Earth's 1:110m countries, version 4.1.0 | https://www.naturalearthdata.com | | the zip, its README and VERSION; not the unzipped shapefile | `build_world_map` |
| `wikimedia_images.txt`, `image_credits.json` | The flags and continent maps the page shows, and their authors and licences | Wikimedia Commons | October 2026 | yes | `fetch_images` |

## ecdc.csv

ECDC weekly cases and deaths per country, up to ISO week 2021-01 (4 to 10 January 2021). ECDC has since withdrawn the URL, so this local file is the only copy. The table and the January 2021 charts on the page came from a later download (27 January 2021, one more week) that is lost: `csv`, its file name, was overwritten in November 2021 by ECDC's frozen daily file, which ends on 14 December 2020.

ECDC's current weekly series (`nationalcasedeath`) cannot stand in for it: it is revised, it now covers only the 30 EU/EEA countries (2020 included), and it ends with ISO week 2023-47. Its worldwide version survives in the Wayback Machine: the snapshot of https://opendata.ecdc.europa.eu/covid19/nationalcasedeath/csv taken on 13 April 2021 (`https://web.archive.org/web/20210413075459id_/` before that URL) covers 214 countries up to ISO week 2021-13 (29 March to 4 April 2021), and agrees with `ecdc.csv` in 9,606 of their 9,672 shared country-weeks of cases (ECDC's revisions account for the rest). It is long format (one row per country, week and indicator) with 2020 populations. There are snapshots of that URL until December 2021, and of `casedistribution/csv` in early February 2021. The worldwide data were dropped between 19 June 2022 (228 countries, to week 2022-23) and 23 June 2022 (25 countries). On 6 October 2026 Tommy decided not to extend the page with these files: it stays at January 2021.

## Keeping them intact

Do not run the plotting scripts casually. `ecdc.csv` is committed (464ee79), so an overwrite shows in `git status` and can be reverted; the other files have no such protection, which is what `SHA256SUMS` is for. The `download_and_read` functions in `plot_choropleth.py` (since 4e8f47e) and `plot_series.py` (since 8009ced) download a file only when it is missing. Before, they fetched any copy more than two hours old again and wrote the response unchecked, and ECDC's URL now serves the daily file, which would have replaced the weekly `ecdc.csv`.
