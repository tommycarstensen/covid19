# covid19

Source for https://tommycarstensen.com/covid19/, a COVID-19 dashboard run from March 2020 to January 2021 and kept as an archive since. `CLAUDE.md` describes every script and data file in detail; this is the map of the folders.

## Folders

- `site/`: the live page and every file it uses, laid out as on the server (`/www/covid19/`). `deploy.py` uploads from here. Its paths are the page's URLs, so do not move anything inside it.
- `archive/`: every image the 2020-2021 pipeline uploaded (about 37,000 PNGs plus the GIFs), moved here by `upload.py` after each upload.
- `regional_maps/`: the Europe, Denmark and Peru map animations, each folder with its script, the files it reads and its MP4. Run a script from inside its folder.
- `map/`: an earlier world-map animation and other map experiments from 2020. `map/data/countries_110m/` holds the Natural Earth countries that `build_world_map.py` reads.
- `tables/`: the table rows `plot_series.py` wrote for `upload.py` to fill into `index.html`.
- `tmp/`: logs, markup snippets and deploy backups; not committed.
- `notes/`: links collected in 2020 for the press pages.
- `trash/`: discarded scripts and images.
- `other/`: files that are not about COVID-19.
- `economy/`, `C19DK/`, `fork/`, `owid/`: separate git repositories, not part of the page.
- `deepti/`: a separate MODIS aerosol-to-PM2.5 task.

## At the root

- The scripts: `plot_*.py` and `build_*.py` draw the charts and maps into `site/`, `deploy.py` uploads the page, `regions.py` lists the regions, `redraw_charts.py` and `fetch_images.py` fill gaps in `site/`.
- The data they read: `ecdc.csv` (ECDC weekly cases and deaths to January 2021), `owid.csv` (Our World in Data), `bsg.csv` (the Oxford policy tracker) and `csv` (ECDC's frozen daily file).
- The 2020-2021 daily pipeline: `wrapper.sh`, `countries.txt`, `upload.py`, the page template `index.html` and the press pages. It is no longer run.
- `CLAUDE.md` (how everything works), `DECISIONS.md` (why), `todo.md` (what is open).
