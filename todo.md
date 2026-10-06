# covid19 todo

Several Claude sessions work in this folder at once. Before starting a task, read this file and `git log -5`; add a line for your own work under "In progress" with your session name and the time, edit only your own lines, commit this file the moment you change it, and move your line to "Done" (or delete it) when the work has landed.

## In progress

- The rest of covid19-4f's list (`DECISIONS.md`, B1-B6 and C1-C13; covid19-4f has ended): B5 (`plot_series.py`'s EU and continent bugs, with a full lint cleanup) and C12 (the table's 2014 populations), plus `deploy.py`'s working-tree uploads and the USA's and EU's 2020 charts below: covid19-67 (6 Oct 2026, 05:47). C13 (the press pages) is with covid19-ac (`press_denmark.html`) and covid19-f8 (`press_international.html`), at Tommy's request.

## Known problems

- Left from covid19-b6's design review of 6 Oct 2026, whose six items have landed (the region small multiples 95611d4 and 7dd3ccf, the heat maps 7dd3ccf, the table and 'Charts by country' by covid19-4f): the USA's own pair of 2020 charts and the EU's two 2020 scatter plots are still in the old style. All of it is live (deploys up to 05:35, covid19-52's 'live and identical').
- `deploy.py` sends pages (`*.html`) as committed at HEAD, but every other file it uploads (scripts, styles, JSON, images) as it is in the working tree. A deploy while another session is mid-edit in, say, `site/worldmap/worldmap.js` publishes the unfinished file, or a script that does not match the page. Until that is fixed, run `git status --short site/` before deploying, and ask the owner of anything modified or newly committed whether it is ready to go live: covid19-4f (6 Oct 2026, 05:31).
