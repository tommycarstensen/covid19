# covid19

Source for https://tommycarstensen.com/covid19/, a COVID-19 dashboard Tommy ran from March 2020 to January 2021, copied from the 2019 MacBook Pro onto the archive disk. The page is now an archive and nothing updates it. Its data stop in January 2021: the table's counts run to mid-January 2021, the world GIFs date from 14 January and `europe.gif` from 24 January 2021. The last upload, on 6 November 2021, re-sent the page with those January data; a note at the top of the page says so. A server listing on 6 October 2026 confirms it: before the October 2026 deploys, the newest files in `/www/covid19/` were the January 2021 images and `press_denmark.html` (6 November 2021). The page's two Our World in Data testing charts run to 23 June 2022, but they are live iframes served by OWID, which stopped updating its testing data that day, not files of this site.

## What is live, and what is not

- `site/index.html` is the source of truth for the live page. It started as a byte-for-byte copy of the server's page on 5 October 2026, and the server's page is newer than anything else in this folder.
- `index.html` at the root is the December 2020 template. `upload.py` filled in its `xxxDATExxx` and `xxxTABLEROWSxxx` placeholders at upload time. Do not edit it to change the live page.
- `archive/` holds every image `upload.py` ever uploaded (about 37,000 PNGs plus the GIFs): it uploaded each file from the root, then moved it here. The newest copy of a file in `archive/` is what the server serves, unless `site/` has a newer one.
- The `?dummy=YYYY-MM-DD` on image URLs is only a cache-buster set to the upload date. It does not say when the image was drawn: the live `days100_*_World1.png` dates from 24 December 2020.

## Deploying

Never upload by FTP, or by any route other than the repo's deploy script. Do not run `upload.py`: it uploads every `*.png` and `*.gif` in the root and then moves them into `archive/`.

`python3 deploy.py` uploads over SFTP (password from `~/lego/.password`, or `TC_PASSWORD_FILE`) to `/www/covid19/`, which holds about 4,000 files from 2020 and 2021. It considers only the files `site/index.html` uses: each one in `site/` is uploaded when it differs from the server's copy, through a temporary name; each one not in `site/` must already be on the server. It never deletes. It refuses unless `site/index.html` is committed and the server's `index.html` is one this repository has committed. It saves every server copy it replaces under `tmp/deploy_backup/<time>/`, checks that the plain live URL serves exactly `site/index.html` and that every uploaded file answers 200, and logs to `tmp/deploy.log`. `--dry` lists the uploads without making them. Run it as a command of its own: the owner's publish guard stops a chained one. The images in `site/` are ignored by git, so rebuild them (`redraw_charts.py`, `fetch_images.py`) on a fresh checkout before deploying.

## Data

- `ecdc.csv`: ECDC weekly cases and deaths per country, up to ISO week 2021-01 (4 to 10 January 2021). ECDC has since withdrawn the URL, so this local file is the only copy. The table and the January 2021 charts on the page came from a later download (27 January 2021, one more week) that is lost: `csv`, its file name, was overwritten in November 2021 by ECDC's frozen daily file, which ends on 14 December 2020. ECDC's current weekly series (`nationalcasedeath`) cannot stand in for it: it is revised, and after 2020 it covers EU/EEA countries only.
- `owid.csv`: Our World in Data, 1 January 2020 to 5 November 2021.
- Do not run the plotting scripts casually. `download_and_read` in `plot_choropleth.py` re-downloads `owid.csv` and `ecdc.csv` when the local copy is more than 2 hours old, and the 2020 version writes the response without checking its HTTP status. A dead URL would therefore replace the only `ecdc.csv` with an error page. Back up the CSVs, or make sure the mtime check skips the download, before running anything.

## Scripts

- `wrapper.sh`: the 2020-2021 daily pipeline. It downloads OWID, runs `plot_series.py` per region and per country in `countries.txt`, then `plot_choropleth_europe.py`, `plot_choropleth.py` and `plot_bubble.py`, and ends with `upload.py`.
- `plot_series.py`: the time-series charts (`days100_*`, sigmoid fits) and the HTML table rows in `tables/`. Needs `countryinfo==0.1.2`, because 1.0 removed `CountryInfo().all()`.
- `plot_choropleth.py`: the world choropleth GIFs (`covid19_*_logTrue.gif`). Needs `geopandas<1.0`.
- `plot_choropleth_europe.py`, `plot_choropleth_denmark.py`, `plot_choropleth_peru.py`: the NUTS-region map animations (`europe.gif`, `denmark.gif`, `peru.gif`, plus an MP4 of each).
- `plot_bubble.py`: the bubble charts. `map/mapcovid19.py`: an earlier world-map animation, used only by `map/index.html`.
- `redraw_charts.py` (October 2026): draws, into `site/`, the two line charts and thumbnails of each table country whose thumbnail `upload.py` never moved into `archive/`: 63 countries, whose thumbnails had never existed and whose full-size charts were from April 2020. It calls `plot_series.doLinePlots` on `ecdc.csv` and downloads nothing. `python3 redraw_charts.py Denmark` is a trial into `tmp/redraw/`.
- `fetch_images.py` (October 2026): saves the flags and continent maps in `wikimedia_images.txt` at 120 pixels into `site/img/`, with their authors and licences in `image_credits.json`. Wikimedia answers HTTP 400 to thumbnails at non-standard widths, which blanked the page's 45 and 440 pixel ones.
- `build_animations.py` (October 2026): re-encodes the page's map GIFs as MP4s plus a poster JPEG each in `site/anim/`, cropped to the frames' content (9.7 MB of GIF becomes 1.6 MB), and writes their `<video>` tags to `tmp/build_animations_markup.html`. Europe comes from the 43 PNG frames in `archive/`, the world maps from the GIFs; a GIF that `plot_choropleth.py` has written to the root takes precedence over `archive/`. `site/anim/scrubber.js` turns each `<video data-fps>` into a player with play/pause and a slider that steps one frame at a time. Testing it locally needs a server that honours HTTP Range requests, as the live Apache does: Chrome resets every seek to 0 on Python's `http.server`.
- The weekly cases and deaths world GIFs (`covid19_{cases,deaths}perweek_*`) show seven-week totals: the 2020 code kept a 7-row rolling sum after ECDC switched to weekly rows. The fix is in `plot_choropleth.py`, but the GIFs have not been regenerated, so the page still shows the server's GIFs for those two and no players. Regenerate them, then rerun `build_animations.py` and swap in their `<video>` tags.
- `plot_days100_world.py` (October 2026): the World1/World2 aligned figures as small multiples, one panel per country, from `ecdc.csv`, into `site/aligned_{cases,deaths}_{World1,World2}{,_narrow}.png`, with `<picture>` tags in `tmp/plot_days100_world_markup.html`. It replaces `plot_series.py`'s 480x360 `days100_*_World{1,2}.png`, whose 14 lines shared 10 colours. The EU there is summed under ECDC's names (`Czechia`, which the 2020 list missed as `Czech Republic`).
- `setup.cfg`: pycodestyle settings. Docstrings and comments are one line per paragraph, so E501 is ignored.
- `requirements.txt`: what `plot_series.py`, `redraw_charts.py` and `deploy.py` import.

## Not part of the page

- `economy/`, `C19DK/`, `fork/covid-19-data/`, `owid/covid-19-data/` and `deepti/pyhdf_github/` are separate git repositories, and are ignored here.
- `deepti/` is a separate MODIS aerosol-to-PM2.5 task. Its HDF4/pyhdf/Miniconda downloads are ignored.
- `trash/` holds discarded scripts and images.

## Git

- History goes back to April 2020; the remote is github.com/tommycarstensen/covid19. Do not push without asking.
- The archive copy had lost `.git/HEAD`. It was restored on 5 October 2026, and `core.fileMode` is set to false because the disk is mounted `noowners`, which makes every file look executable.
- `.gitignore` excludes all binaries (images, video, archives, shapefiles, HDF), the large CSV/GeoJSON downloads and `.password`, which holds the FTP password. Every image is rebuilt by a script, so commit the script and never the image. `~/.gitignore_global` ignores `.gitignore` itself, so it must be committed with `git add -f`.
- Several Claude sessions have worked in this folder at the same time. Commit by explicit pathspec, and ask `ListAgents` who owns a file before editing it.
