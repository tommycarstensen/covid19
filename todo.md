# covid19 todo

Several Claude sessions work in this folder at once. Before starting a task, read this file and `git log -5`; add a line for your own work under "In progress" with your session name and the time, edit only your own lines, commit this file the moment you change it, and move your line to "Done" (or delete it) when the work has landed.

## In progress

- Interactive world map to replace the ECDC world-map animations (cumulative and weekly cases and deaths): `build_world_map.py` (85e1074), `site/worldmap/` (not yet committed on 6 Oct 2026 04:45). Owner: covid19-91. Its weekly figures are single weeks (USA 156,481 and Denmark 433 cases in ISO week 2020-21, as in `ecdc.csv`).
- Bubble charts: `build_bubbles.py` (527511a), `site/bubble/bubble.js` (not yet committed on 6 Oct 2026 04:45). Owner: the session that committed 527511a.
- Bug fixes on the live page, `site/index.html`: covid19-4f (6 Oct 2026).
- Confirm over HTTP that the 20 files of the 04:36 deploy on 6 Oct 2026 (13fe780: the map players and the World small multiples) answer, once the host is back. The page itself was confirmed live, and SFTP confirmed every file's size; the host started refusing this machine after a burst of about 40 quick requests from this session at about 04:40, which is likely why it stopped answering: covid19-8f (6 Oct 2026, 04:50).

## Decided, still to do

- Weekly cases and deaths maps (decided 6 Oct 2026 by ten advisers, unanimously): replace the two GIFs, which show seven-week totals, rather than keep them with a note. The interactive world map does this, so do not run `plot_choropleth.py` to regenerate them. When the map goes live:
  - add one sentence to the archive note at the top of the page saying that until October 2026 the weekly maps summed seven weeks instead of one, overstating weekly cases and deaths several times (8 times for the USA and 16 times for Denmark in May 2020);
  - leave the January 2021 GIFs on the server (`deploy.py` never deletes), so a link to them as originally published keeps working;
  - replace the line in `CLAUDE.md` that says to regenerate the weekly GIFs and rerun `build_animations.py`.

## Known problems

- The live host stopped answering at about 04:45 on 6 October 2026 (see `CLAUDE.md`). Check that it is back before a `deploy.py` run.
