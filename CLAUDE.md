# covid19

Source for https://tommycarstensen.com/covid19/, a COVID-19 dashboard Tommy ran from April 2020 to November 2021, copied from the 2019 MacBook Pro onto the archive disk. The page is now an archive: its data stop on 6 November 2021 and nothing updates it.

## What is live, and what is not

- `site/index.html` is the source of truth for the live page. It started as a byte-for-byte copy of the server's page on 5 October 2026, and the server's page is newer than anything else in this folder.
- `index.html` at the root is the December 2020 template. `upload.py` filled in its `xxxDATExxx` and `xxxTABLEROWSxxx` placeholders at upload time. Do not edit it to change the live page.
- `archive/` holds every image `upload.py` ever uploaded (about 37,000 PNGs plus the GIFs): it uploaded each file from the root, then moved it here. The newest copy of a file in `archive/` is what the server serves, unless `site/` has a newer one.
- The `?dummy=YYYY-MM-DD` on image URLs is only a cache-buster set to the upload date. It does not say when the image was drawn: the live `days100_*_World1.png` dates from 24 December 2020.

## Deploying

Never upload by FTP, or by any route other than the repo's deploy script. Do not run `upload.py`: it uploads every `*.png` and `*.gif` in the root and then moves them into `archive/`.

## Data

- `ecdc.csv`: ECDC weekly cases and deaths per country, up to ISO week 2021-01. ECDC has since withdrawn the URL, so this local file is the only copy.
- `owid.csv`: Our World in Data, 1 January 2020 to 5 November 2021.
- Do not run the plotting scripts casually. `download_and_read` in `plot_choropleth.py` re-downloads `owid.csv` and `ecdc.csv` when the local copy is more than 2 hours old, and the 2020 version writes the response without checking its HTTP status. A dead URL would therefore replace the only `ecdc.csv` with an error page. Back up the CSVs, or make sure the mtime check skips the download, before running anything.

## Scripts

- `wrapper.sh`: the 2020-2021 daily pipeline. It downloads OWID, runs `plot_series.py` per region and per country in `countries.txt`, then `plot_choropleth_europe.py`, `plot_choropleth.py` and `plot_bubble.py`, and ends with `upload.py`.
- `plot_series.py`: the time-series charts (`days100_*`, sigmoid fits) and the HTML table rows in `tables/`. Needs `countryinfo==0.1.2`, because 1.0 removed `CountryInfo().all()`.
- `plot_choropleth.py`: the world choropleth GIFs (`covid19_*_logTrue.gif`). Needs `geopandas<1.0`.
- `plot_choropleth_europe.py`, `plot_choropleth_denmark.py`, `plot_choropleth_peru.py`: the NUTS-region map animations (`europe.gif`, `denmark.gif`, `peru.gif`, plus an MP4 of each).
- `plot_bubble.py`: the bubble charts. `map/mapcovid19.py`: an earlier world-map animation, used only by `map/index.html`.

## Not part of the page

- `economy/`, `C19DK/`, `fork/covid-19-data/`, `owid/covid-19-data/` and `deepti/pyhdf_github/` are separate git repositories, and are ignored here.
- `deepti/` is a separate MODIS aerosol-to-PM2.5 task. Its HDF4/pyhdf/Miniconda downloads are ignored.
- `trash/` holds discarded scripts and images.

## Git

- History goes back to April 2020; the remote is github.com/tommycarstensen/covid19. Do not push without asking.
- The archive copy had lost `.git/HEAD`. It was restored on 5 October 2026, and `core.fileMode` is set to false because the disk is mounted `noowners`, which makes every file look executable.
- `.gitignore` excludes all binaries (images, video, archives, shapefiles, HDF), the large CSV/GeoJSON downloads and `.password`, which holds the FTP password. Every image is rebuilt by a script, so commit the script and never the image. `~/.gitignore_global` ignores `.gitignore` itself, so it must be committed with `git add -f`.
- Several Claude sessions have worked in this folder at the same time. Commit by explicit pathspec, and ask `ListAgents` who owns a file before editing it.
