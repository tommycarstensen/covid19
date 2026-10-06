# covid19 todo

Several Claude sessions work in this folder at once. Before starting a task, read this file and `git log -5`; add a line for your own work under "In progress" with your session name and the time, edit only your own lines, commit this file the moment you change it, and move your line to "Done" (or delete it) when the work has landed.

## In progress

- Interactive world map, replacing the ECDC world-map animations (cumulative and weekly cases and deaths): committed in `build_world_map.py` (85e1074), `site/worldmap/` (98b8bee), the page (db094cc, section `#world-maps`) and `build_animations.py` (f826f91). Its weekly figures are single weeks (USA 156,481 and Denmark 433 cases in ISO week 2020-21, as in `ecdc.csv`). Still to do: the deploy (covid19-82, once the host answers); the weekly-maps sentence in the archive note (handed to covid19-4f with its wording); the `CLAUDE.md` line below (waits for Tommy's go-ahead). Owner: covid19-91 (6 Oct 2026, 04:55).
- Bubble charts: `build_bubbles.py` (527511a), `site/bubble/bubble.js` (not yet committed on 6 Oct 2026 04:45). Owner: the session that committed 527511a.
- Bug fixes and content changes on the page, `site/index.html`, in this order, one commit each, with every choice recorded in `DECISIONS.md`: covid19-4f (6 Oct 2026, 04:55).
  - The EU table row: it leaves out Czechia but divides by the EU-28 population.
  - Check that `sortable.js` sorts numbers as numbers, once the host is back.
  - The archive note, `<title>` and meta description: data end dates, what happened after January 2021, links to current data. covid19-91 adds its weekly-maps sentence to the end of the note.
  - The table's column headers: "most recent week" becomes the actual week.
  - Trim "Infographics and other news clips" and "References" to what still works and belongs on an archive.
  - A footer with data and basemap credits, licences, and how to cite the page.
  - Then: a "How to read this page" section, chart captions and alt text, and a restructure (aligned charts first, per-country charts on a subpage, the table's thumbnail columns).

## Decided, still to do

- Weekly cases and deaths maps (decided 6 Oct 2026 by ten advisers, unanimously): replace the two GIFs, which show seven-week totals, rather than keep them with a note. The interactive world map does this, so do not run `plot_choropleth.py` to regenerate them. When the map goes live:
  - add one sentence to the archive note at the top of the page saying that until October 2026 the weekly maps summed seven weeks instead of one, overstating weekly cases and deaths several times (8 times for the USA and 16 times for Denmark in May 2020);
  - leave the January 2021 GIFs on the server (`deploy.py` never deletes), so a link to them as originally published keeps working;
  - replace the line in `CLAUDE.md` that says to regenerate the weekly GIFs and rerun `build_animations.py`.

## Known problems

- The live host stopped answering at about 04:45 on 6 October 2026 (see `CLAUDE.md`). Check that it is back before a `deploy.py` run.
