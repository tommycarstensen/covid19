# 2020

The 2020-2021 dashboard's own code and output, kept as it last ran. Nothing here is run any more: the page is drawn and deployed by `scripts/`, whose README describes the 2020 plotting scripts the October 2026 ones import (`plot_series.py`, `plot_choropleth.py`, `plot_bubble.py`).

## pipeline/

The daily pipeline of 2020-2021, which ran from the repo root, so its relative paths no longer resolve here. `wrapper.sh` downloaded OWID, ran `plot_series.py` per region and per country in `countries.txt`, moved the table rows into `tables/`, then ran `plot_choropleth_europe.py` (inside `regional_maps/europe/`, moving its frames and GIF back to the root), `plot_choropleth.py` and `plot_bubble.py`, and ended with `upload.py`.

`upload.py` filled the template `index.html` with the date and the rows in `tables/` (its `xxxDATExxx` and `xxxTABLEROWSxxx` placeholders), uploaded it by FTP with the two press pages beside it (2020 working copies; the live ones are in `site/`) and the day's images, and moved each image into `archive/`. It has been retired since 6 October 2026 (aa0c31f): it exits at once, because `.password`, the FTP password it read, sits beside it (git-ignored), and a run would send a 2020 template over the live page. Do not run it or `wrapper.sh`, and do not edit `index.html` to change the live page: `site/index.html` is the live page.

## regional_maps/

The Europe, Denmark and Peru map animations (`europe.gif`, `denmark.gif`, `peru.gif`, plus an MP4 of each), one folder each with its script, the files it reads and its MP4: Europe, Eurostat's NUTS boundaries, Scotland's and Wales's health boards and ECDC's subnational data of 21 January 2021; Denmark, SSI's cases per municipality, Statistics Denmark's FOLK1A and the municipality GeoJSON; Peru, MINSA's positive tests and the HDX boundaries. The scripts open their inputs by bare file name, so run each from inside its folder.

They have been committed and clean under ruff, pycodestyle and pyright since 6 October 2026 (6ed5421), but have not been run since 2021 and need geopandas. Peru's `DataFrame.append`, which pandas 2 removed, is now `pd.concat`; Europe's downloads only a missing file, so ECDC's subnational data cannot be overwritten; Peru's still reads its populations from Wikipedia at every run. The inputs are git-ignored and most cannot be downloaded again: `SHA256SUMS` records all 213 of them, checked with `cd 2020/regional_maps && shasum -a 256 -c SHA256SUMS`.

## map/

The first world-map animation, of March 2020, before `plot_choropleth.py`: `mapcovid19.py` drew the GIFs that `index.html` shows. Both are committed as they last ran (not cleaned for the linters); the GIFs, the Natural Earth zips and the 42 MB `covid19map.html` are git-ignored.

## notes/

The links collected in 2020 for the press pages (`links.txt`, `add.txt`, `regeringen.htm`).

## archive/

Every image `upload.py` uploaded, about 37,000 PNGs plus the GIFs: it uploaded each file from the repo root, then moved it here. The file names carry the chart, the region or country and often the date (`<chart>_<region>_<YYYY-MM-DD>.png`, 21 February 2020 to 27 January 2021). A name without a date was replaced at every upload, so it holds the last version uploaded. The newest copy of a file here is what the server serves, unless `site/` has a newer one; `scripts/build_animations.py` and `scripts/redraw_charts.py` read from here, so keep it flat and keep the names.

The `?dummy=YYYY-MM-DD` on the page's image URLs is only a cache-buster set to the upload date. It does not say when the image was drawn: the live `days100_*_World1.png` dates from 24 December 2020.

The images are git-ignored. On 6 October 2026 the folder held 37,393 files and 1,026,102,921 bytes, and the SHA-256 of their sorted per-file checksums was `309f1aa8ecbc2f060061b81cf1628ca53762cdab39ccd6d19c17bcbfb7a089f0`. Recompute it with `cd 2020/archive && find . -type f | LC_ALL=C sort | tr '\n' '\0' | xargs -0 shasum -a 256 | shasum -a 256`. If it differs, `rsync -rcni --delete /Volumes/black2tb/archive/mbp2019/covid19_backup_2026-10-06/archive/ 2020/archive/` lists the files that differ from the disk copy.
