# covid19

Source for https://tommycarstensen.com/covid19/, a COVID-19 dashboard run from March 2020 to January 2021 and kept as an archive since. `CLAUDE.md` describes every script and data file in detail; this is the map of the folders.

## Folders

- `site/`: the live page and every file it uses, laid out as on the server (`/www/covid19/`). `scripts/deploy.py` uploads from here. Its paths are the page's URLs, so do not move anything inside it.
- `scripts/`: every script that draws, builds or deploys `site/`: `plot_*.py` and `build_*.py` draw the charts, maps and animations, `regions.py` lists the regions, `redraw_charts.py` and `fetch_images.py` fill gaps in `site/`, `backup_external_images.py` copies the images the pages take from other sites, and `deploy.py` uploads the page. Run them from the repo root: `python3 scripts/<name>.py`.
- `data/`: what the scripts read: `ecdc.csv` (ECDC weekly cases and deaths to January 2021), `owid.csv` (Our World in Data), `bsg.csv` (the Oxford policy tracker), `csv` (ECDC's frozen daily file), `natural_earth/` (the world map's country outlines), and `wikimedia_images.txt` with `image_credits.json` (the page's flags and continent maps, and their licences).
- `external_images/`: copies of the images the pages take from other sites, saved by `backup_external_images.py`.
- `2020/`: the 2020-2021 dashboard's own code and output, kept as it last ran:
  - `pipeline/`: the daily pipeline (`wrapper.sh`, `countries.txt`, `upload.py`, the page template `index.html`, the table rows in `tables/` and the press pages). It is no longer run, and must not be: it uploads by FTP.
  - `regional_maps/`: the Europe, Denmark and Peru map animations, each folder with its script, the files it reads and its MP4. Run a script from inside its folder.
  - `map/`: an earlier world-map animation.
  - `archive/`: every image the pipeline uploaded (about 37,000 PNGs plus the GIFs), moved here by `upload.py` after each upload.
  - `notes/`: links collected in 2020 for the press pages.
- `other/`: files that are not about COVID-19, among them the 2020 obesity and alcohol map experiments (`other/maps/`) and a separate MODIS aerosol-to-PM2.5 task (`other/deepti/`).
- `repos/`: separate git repositories (`economy`, `C19DK`, and forks of Our World in Data's data), not part of the page.
- `trash/`: discarded scripts and images.
- `tmp/`: logs, markup snippets and deploy backups; not committed.

## At the root

- `CLAUDE.md` (how everything works), `DECISIONS.md` (why), `todo.md` (what is open).
- `requirements.txt`, `setup.cfg` (pycodestyle) and `ruff.toml`.
