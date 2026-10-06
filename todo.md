# covid19 todo

Several Claude sessions work in this folder at once. Before starting a task, read this file and `git log -5`; add a line for your own work under "In progress" with your session name and the time, edit only your own lines, commit this file the moment you change it, and move your line to "Done" (or delete it) when the work has landed.

## In progress

- Bubble charts: `build_bubbles.py` (527511a), `site/bubble/bubble.js` (not yet committed on 6 Oct 2026 04:45). Owner: the session that committed 527511a.
- Bug fixes and content changes on the page, `site/index.html`, one commit each, with every choice recorded in `DECISIONS.md` (the full list, B1-B6 and C1-C13, is there): covid19-4f (6 Oct 2026, 05:15).
  - Done: B1, B2 (France and Sweden), B6, C3, C4, C5 (table after the maps), C6 (lazy loading; "Charts by country" removed), C7, C8, C9, C10, C11, the table's readability. C1 and C2 superseded: Tommy had the archive note removed (covid19-91).
  - Open: B5 (`plot_series.py`'s causes of the continent and EU bugs, with a full lint cleanup), C12 (the table's 2014 populations), C13 (the press pages). Not started; for Tommy to choose.
- Design review of the whole page (Tommy asked covid19-b6 on 6 Oct 2026, 05:00, to work through it), split by agreement between three sessions:
  1. The region sections' 2020 line charts (30-50 lines sharing 10 colours, unreadable legends) redrawn as small multiples like the World figures: Western Asia except Iran by covid19-cd (8b8b4b7); every other region by covid19-b6 in `plot_days100_world.py` (c8c5ab7), not yet on the page.
  2. "Charts by country" dropped, its ids moved to the table rows; 3. the table moved up after the maps; 4. the table's numbers formatted, right-aligned, sorted by total cases, with a sort indicator and tighter rows: covid19-4f.
  5. Inconsistent country headings: moot once 2 lands.
  6. One colour language: the heat maps (`plot_heat_*`, 2020 orange-red, illegible country labels) redrawn in the interactive world map's colour scale and bins, by a new script, then swapped into the page after 4f's restructure lands: covid19-b6 (6 Oct 2026, 05:15). Drawn by `plot_heat.py` (c491d11), with the region lists in `regions.py` (9a0da91); not yet on the page. The world map itself no longer has those classes and ramps: Tommy rejected them, and since 6 October 2026 (b3c69c7) its six maps, tests included, use the 2020 GIFs' own colour maps (OrRd, PuRd, Greens, YlGnBu, PuBuGn, Blues), so the heat maps need their colours from the 2020 heat-map script (OrRd) instead; covid19-d9 has `plot_heat.py`.

## Decided, still to do

- Weekly cases and deaths maps (decided 6 Oct 2026 by ten advisers, unanimously): replace the two GIFs, which show seven-week totals, rather than keep them with a note. The interactive world map does this, so do not run `plot_choropleth.py` to regenerate them. When the map goes live:
  - add one sentence to the archive note at the top of the page saying that until October 2026 the weekly maps summed seven weeks instead of one, overstating weekly cases and deaths several times (8 times for the USA and 16 times for Denmark in May 2020);
  - leave the January 2021 GIFs on the server (`deploy.py` never deletes), so a link to them as originally published keeps working;
  - replace the line in `CLAUDE.md` that says to regenerate the weekly GIFs and rerun `build_animations.py`.

## Known problems

- The live host stopped answering at about 04:45 on 6 October 2026 (see `CLAUDE.md`). Check that it is back before a `deploy.py` run.
- `deploy.py` sends pages (`*.html`) as committed at HEAD, but every other file it uploads (scripts, styles, JSON, images) as it is in the working tree. A deploy while another session is mid-edit in, say, `site/worldmap/worldmap.js` publishes the unfinished file, or a script that does not match the page. Until that is fixed, run `git status --short site/` before deploying, and ask the owner of anything modified or newly committed whether it is ready to go live: covid19-4f (6 Oct 2026, 05:31).
