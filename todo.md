# covid19 todo

Several Claude sessions work in this folder at once. Before starting a task, read this file and `git log -5`; add a line for your own work under "In progress" with your session name and the time, edit only your own lines, commit this file the moment you change it, and move your line to "Done" (or delete it) when the work has landed.

## In progress

- Interactive world map, replacing the ECDC world-map animations (cumulative and weekly cases and deaths): committed in `build_world_map.py` (85e1074), `site/worldmap/` (98b8bee), the page (db094cc, section `#world-maps`) and `build_animations.py` (f826f91). Its weekly figures are single weeks (USA 156,481 and Denmark 433 cases in ISO week 2020-21, as in `ecdc.csv`). Live: on 6 Oct 2026 the page, `worldmap.js` and `worldmap.json` (212,573 bytes, as committed) answered HTTP 200. Still to do: the weekly-maps sentence in the archive note (handed to covid19-4f with its wording, waiting for Tommy's approval of the note); the `CLAUDE.md` line below (waits for Tommy's go-ahead). Owner: covid19-91.
- Bubble charts: `build_bubbles.py` (527511a), `site/bubble/bubble.js` (not yet committed on 6 Oct 2026 04:45). Owner: the session that committed 527511a.
- Bug fixes and content changes on the page, `site/index.html`, one commit each, with every choice recorded in `DECISIONS.md` (the full list, B1-B6 and C1-C13, is there): covid19-4f (6 Oct 2026, 05:05).
  - Done and committed: the EU row (a8a284e), `site/sortable.js` (d8c6782), the title and table headers (fdb860c), "Press clippings, 2020" (5248c51), "Other trackers" (2a2c475), "About this page" (dce1096), Taiwan's name (cff859e). Live from 04:56 up to fdb860c; covid19-82 deploys the rest.
  - Waiting for Tommy to approve its wording: the archive note (C1, C2), including covid19-91's seven-week sentence. Nobody else edits the note meanwhile.
  - Next: a "How to read this page" section (C3), chart captions and alt text (C10), then the restructure (C5, C6, C7).

## Decided, still to do

- Weekly cases and deaths maps (decided 6 Oct 2026 by ten advisers, unanimously): replace the two GIFs, which show seven-week totals, rather than keep them with a note. The interactive world map does this, so do not run `plot_choropleth.py` to regenerate them. When the map goes live:
  - add one sentence to the archive note at the top of the page saying that until October 2026 the weekly maps summed seven weeks instead of one, overstating weekly cases and deaths several times (8 times for the USA and 16 times for Denmark in May 2020);
  - leave the January 2021 GIFs on the server (`deploy.py` never deletes), so a link to them as originally published keeps working;
  - replace the line in `CLAUDE.md` that says to regenerate the weekly GIFs and rerun `build_animations.py`.

## Known problems

- The live host stopped answering at about 04:45 on 6 October 2026 (see `CLAUDE.md`). Check that it is back before a `deploy.py` run.
