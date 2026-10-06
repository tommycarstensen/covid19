# covid19

Source for https://tommycarstensen.com/covid19/, a COVID-19 dashboard run from March 2020 to January 2021 and kept as an archive since. This is the map of the folders; `data/`, `scripts/` and `2020/` each have a README with the detail, and `CLAUDE.md` holds the rules for working here.

## History

The page is an archive and nothing updates it. Its data stop in January 2021: the table's counts run to mid-January 2021, the world GIFs date from 14 January and `europe.gif` from 24 January 2021. The last upload, on 6 November 2021, re-sent the page with those January data; a note at the top of the page says so. A server listing on 6 October 2026 confirms it: before the October 2026 deploys, the newest files in `/www/covid19/` were the January 2021 images and `press_denmark.html` (6 November 2021). The page's two Our World in Data testing charts run to 23 June 2022, but they are live iframes served by OWID, which stopped updating its testing data that day, not files of this site.

In October 2026 the charts the page had lost or never had were redrawn, the world maps and bubble charts made interactive, and the folders reorganised. The repository was copied from the 2019 MacBook Pro onto an external archive disk, and on 6 October 2026 moved from there to `~/covid19` after the disk disconnected in the middle of the work; the disk keeps the copy as it was at the move.

## Folders

- `site/`: the live page and every file it uses, laid out as on the server (`/www/covid19/`). `scripts/deploy.py` uploads from here. Its paths are the page's URLs, so do not move anything inside it.
- `scripts/`: every script that draws, builds or deploys `site/`: `plot_*.py` and `build_*.py` draw the charts, maps and animations, `regions.py` lists the regions, `redraw_charts.py` and `fetch_images.py` fill gaps in `site/`, `backup_external_images.py` copies the images the pages take from other sites, and `deploy.py` uploads the page. Run them from the repo root: `python3 scripts/<name>.py`. `scripts/README.md` describes each, and what rebuilds `site/` on a fresh checkout.
- `data/`: what the scripts read: `ecdc.csv` (ECDC weekly cases and deaths to January 2021), `owid.csv` (Our World in Data), `bsg.csv` (the Oxford policy tracker), `csv` (ECDC's frozen daily file), `natural_earth/` (the world map's country outlines), and `wikimedia_images.txt` with `image_credits.json` (the page's flags and continent maps, and their licences). `data/README.md` says where each came from; most cannot be downloaded again, and `data/SHA256SUMS` records them.
- `external_images/`: copies of the images the pages take from other sites, saved by `backup_external_images.py`.
- `2020/`: the 2020-2021 dashboard's own code and output, kept as it last ran (`2020/README.md`):
  - `pipeline/`: the daily pipeline (`wrapper.sh`, `countries.txt`, `upload.py`, the page template `index.html`, the table rows in `tables/` and the press pages). It is no longer run, and must not be: it uploaded by FTP, and `upload.py` now exits at once.
  - `regional_maps/`: the Europe, Denmark and Peru map animations, each folder with its script, the files it reads and its MP4. Run a script from inside its folder.
  - `map/`: an earlier world-map animation.
  - `archive/`: every image the pipeline uploaded (about 37,000 PNGs plus the GIFs), moved here by `upload.py` after each upload.
  - `notes/`: links collected in 2020 for the press pages.
- `other/`: files that are not about COVID-19, among them the 2020 obesity and alcohol map experiments (`other/maps/`) and a separate MODIS aerosol-to-PM2.5 task (`other/deepti/`).
- `repos/`: separate git repositories (`economy`, `C19DK`, and forks of Our World in Data's data), not part of the page.
- `trash/`: discarded scripts and images.
- `tmp/`: logs, markup snippets and deploy backups; not committed.

## At the root

- `CLAUDE.md` (the rules for working here), `DECISIONS.md` (why), `todo.md` (what is open).
- `requirements.txt` (the scripts, as run in October 2026) and `requirements-2020.txt` (the 2020 code that needs older libraries), `setup.cfg` (pycodestyle) and `ruff.toml`.
