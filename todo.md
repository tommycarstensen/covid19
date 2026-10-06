# covid19 todo

Several Claude sessions work in this folder at once. Before starting a task, read this file and `git log -5`; add a line for your own work under "In progress" with your session name and the time, edit only your own lines, commit this file the moment you change it, and move your line to "Done" (or delete it) when the work has landed.

## In progress

- The USA's and the EU's 2020 charts (below): covid19-67 (6 Oct 2026, 06:03). B5, C12 and `deploy.py`'s working-tree uploads are done (2942576, 0471a56, eb00ed1); C12 is live.
- Testing against cases: the tests-cases-deaths small multiples and a tests and share-positive heat map per region, which Tommy liked as drafts; and `build_world_map.read_owid_tests`, which leaves out the 24 places OWID gives daily test counts but no running totals (France, Sweden, Czechia among them). Scripts and images only; nothing goes on the page until Tommy approves its wording: covid19-6d (6 Oct 2026, 06:14). Touches `build_world_map.py`, `site/worldmap/worldmap.json`, `plot_heat.py` and a new `plot_tests.py`.
- C13, the press pages: restyled, broken links repaired and live (covid19-f8, `press_international.html`, 690cfed; covid19-ac, `press_denmark.html`, e89e325). Their title, back link and archive note wait for Tommy to approve the wording covid19-f8 proposed; add none without his yes.

## Known problems

- Left from covid19-b6's design review of 6 Oct 2026, whose six items have landed (the region small multiples 95611d4 and 7dd3ccf, the heat maps 7dd3ccf, the table and 'Charts by country' by covid19-4f): the USA's own pair of 2020 charts and the EU's two 2020 scatter plots are still in the old style. All of it is live (deploys up to 05:35, covid19-52's 'live and identical').
