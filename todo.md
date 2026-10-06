# covid19 todo

Several Claude sessions work in this folder at once. Before starting a task, read this file and `git log -5`; add a line for your own work under "In progress" with your session name and the time, edit only your own lines, commit this file the moment you change it, and move your line to "Done" (or delete it) when the work has landed.

## In progress

- C13, the press pages: the international page is done and live (Tommy's choices, 832d99c). The Danish page's tab title "Press clippings, March to October 2020: Danish press" and the same "← COVID-19 tracker" link, decided after ten advisers (08be1d8, 86510df), are being added by covid19-ac with its dead-link repairs; no archive note or heading on either page.
- Restructuring the folders, which Tommy asked for. Done: the roughly 200 untracked inputs at the root went into `regional_maps/{europe,denmark,peru}/` with their scripts, plus `notes/` and `other/` and a folder map in `README.md` (272e6cd, e3467d2, 77c45aa). Still to do, held back because other sessions are editing the scripts it touches: move `ecdc.csv`, `owid.csv`, `bsg.csv`, `csv` and the unread `COVID-19-geographic-disbtribution-worldwide.xlsx` into `data/` (11 root scripts open them by name: `build_bubbles`, `build_world_map`, `plot_bubble`, `plot_choropleth`, `plot_days100_world`, `plot_heat`, `plot_scatter`, `plot_series`, `plot_tests`, `redraw_charts`, `regions`), and the 2020 pipeline (`wrapper.sh`, `upload.py`, `countries.txt`, the template `index.html`, `tables/`, the root press pages) into a folder of its own (`plot_series.py` writes `tables/`). Do it only when no session is editing a root script: covid19-32 (6 Oct 2026, 06:36).

## Known problems

None open.
