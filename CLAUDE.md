# covid19

Source for https://tommycarstensen.com/covid19/, a COVID-19 dashboard Tommy ran from March 2020 to January 2021. The page is now an archive that nothing updates, and its data stop in January 2021. `README.md` maps the folders and tells the page's history. `data/`, `scripts/` and `2020/` each have a `README.md` with the detail this file used to carry: read the one for the folder you are about to work in before you start.

The repository has been `~/covid19` since 6 October 2026, when the external disk it lived on disconnected in the middle of the work. `/Volumes/black2tb/archive/mbp2019/covid19` is a symlink to it, and the disk keeps the copy as it was at the move (HEAD 83fc64d) in `covid19_backup_2026-10-06`; do not edit that copy.

## Folders

Laid out like this since 6 October 2026; commits and `DECISIONS.md` entries from before then use the old paths (the scripts and `deploy.py` at the root, `pipeline_2020/`, `regional_maps/`, `map/`, `notes/` and `archive/` at the root).

- `site/`: the live page and every file it uses, laid out as on the server (`/www/covid19/`). Its paths are the page's URLs, so nothing in it moves.
- `scripts/`: every script that draws, builds or deploys `site/`, with the 2020 plotting scripts the October 2026 ones import. Run them from the repo root: `python3 scripts/<name>.py`. See `scripts/README.md`.
- `data/`: what the scripts read. See `data/README.md`.
- `external_images/`: copies of the images the pages take from other sites.
- `2020/`: the 2020-2021 dashboard's own code and output, kept as it last ran: `pipeline/`, `regional_maps/`, `map/`, `notes/` and `archive/`. See `2020/README.md`.
- `other/` (not about COVID-19), `repos/` (separate git repositories), `trash/` and `tmp/` (logs, markup snippets and deploy backups, not committed).

## What is live, and what is not

- `site/index.html` is the source of truth for the live page. It started as a byte-for-byte copy of the server's page on 5 October 2026, and the server's page is newer than anything else in this folder. `2020/pipeline/index.html` is the December 2020 template: do not edit it to change the live page.
- Since 6 October 2026 every h2 and h3 in `site/index.html` has an id, which outside links may point to (`#denmark`, `#table`), so do not rename one. The Contents list after the archive note is written by hand: a new h2 or h3 needs an id, a `<a class="anchor">` link and a line in the Contents.
- Do not load the live page many times in a row from a headless browser: each load fetches about 250 images. On 6 October 2026 the host stopped answering at about 04:45, after two sessions had sent some 2,500 requests in a few minutes; whether that caused it is not known. To check a local `site/index.html`, serve it at the live URL through Playwright's `page.route` and block the images unless the check needs them.
- `2020/archive/` holds every image the 2020 pipeline uploaded (about 37,000 PNGs plus the GIFs). The newest copy of a file there is what the server serves, unless `site/` has a newer one.

## Deploying

Never upload by FTP, or by any route other than the repo's deploy script. Do not run `2020/pipeline/upload.py` or `wrapper.sh`: the first uploaded every `*.png` and `*.gif` in its working directory by FTP and then moved them into `archive/`, and the second ends by running it. `upload.py` has exited at once since aa0c31f, because `.password` sits beside it.

`python3 scripts/deploy.py` uploads over SFTP (password from `~/lego/.password`, or `TC_PASSWORD_FILE`) to `/www/covid19/`, which holds about 4,000 files from 2020 and 2021. It considers only the files `site/index.html` uses and those used by the pages it links to (the two press pages): each one in `site/` is uploaded when it differs from the server's copy, through a temporary name; each one not in `site/` must already be on the server. It never deletes. It refuses unless `site/index.html` is committed and the server's `index.html` is one this repository has committed. Every file git tracks (pages, scripts, styles, data) goes up as committed at HEAD, whatever the working tree holds, so commit a change before deploying it; the git-ignored images go up from the working tree, and a file in `site/` that git neither tracks nor ignores is refused. It saves every server copy it replaces under `tmp/deploy_backup/<time>/`, checks that the plain live URL serves exactly `site/index.html` and that every uploaded file answers 200, and logs to `tmp/deploy.log`. `--dry` lists the uploads without making them. Run it as a command of its own: the owner's publish guard stops a chained one. The images in `site/` are ignored by git, so on a fresh checkout rebuild them before deploying: `scripts/README.md` lists the eight scripts that draw them.

## Data

`data/README.md` says where each data file came from and which scripts read it. Most cannot be downloaded again: `ecdc.csv` is the only copy of ECDC's weekly file (committed, 464ee79), and `owid.csv`, `bsg.csv` and `csv` are git-ignored and exist only on this disk and in the disk copy. Check them with `cd data && shasum -a 256 -c SHA256SUMS`, and the regional maps' inputs with `2020/regional_maps/SHA256SUMS`. Do not run the plotting scripts casually; their download functions fetch a file only when it is missing.

## Scripts

`scripts/README.md` describes each script, the environment (`requirements.txt`; `requirements-2020.txt` for `plot_choropleth.py` and the regional maps, which need `geopandas<1.0` and `pandas<1.5`) and what rebuilds `site/`. Rules that hold for every change:

- Redrawn charts and maps keep the colour maps and layout of Tommy's 2020 scripts (the world maps: one colour map per GIF of `plot_choropleth.py`; the bubbles: viridis). Do not swap them for a generic palette: Tommy rejected the eight-class blue and orange ramps the world maps first had.
- Do not regenerate `plot_choropleth.py`'s GIFs: the January 2021 GIFs stay on the server as published, and the page no longer uses them.
- Every script in `scripts/` and `2020/regional_maps/` is clean under ruff, pycodestyle (`setup.cfg`: docstrings and comments are one line per paragraph, so E501 is ignored) and pyright (`ruff.toml` tells ruff's import sorting that `scripts/` holds the project's own modules). The other 2020 code is kept as it last ran.

## Not part of the page

- `repos/` holds `economy/`, `C19DK/`, `fork/covid-19-data/` and `owid/covid-19-data/`, separate git repositories, ignored here. Their `.git/HEAD` was lost on the archive disk, as this repository's was, so git commands run inside them acted on this repository; it was restored on 6 October 2026 (`economy` at its last commit, f0078c4, which had no branch left; the others at `master`). The OWID clone is a one-commit shallow clone whose object store is damaged (148 objects missing); its files are intact and OWID's repository is public. `other/deepti/pyhdf_github/` had the same fault and the same fix.
- `other/` holds files that were in this folder but are not about COVID-19: a yield-curve plot (`3d.py`), a stock-market chart, a zipped spreadsheet of financial statements, the 2020 obesity, alcohol and Texas map experiments (`other/maps/`), and `other/deepti/`, a separate MODIS aerosol-to-PM2.5 task whose HDF4/pyhdf/Miniconda downloads and `pyhdf_github/` repository are ignored.
- `trash/` holds discarded scripts and images.

## Git

- History goes back to April 2020; the remote is github.com/tommycarstensen/covid19. Do not push without asking.
- The archive copy had lost `.git/HEAD`. It was restored on 5 October 2026, and `core.fileMode` is set to false because the disk was mounted `noowners`, which made every file look executable; the move to `~/covid19` kept those modes, so it stays false.
- `.gitignore` excludes all binaries (images, video, archives, shapefiles, HDF), the large CSV/GeoJSON downloads and `2020/pipeline/.password`, which holds the FTP password `upload.py` read. Every image is rebuilt by a script, so commit the script and never the image. `~/.gitignore_global` ignores `.gitignore` itself, so it must be committed with `git add -f`.
- Several Claude sessions have worked in this folder at the same time. Commit by explicit pathspec, and ask `ListAgents` who owns a file before editing it.
