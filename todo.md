# covid19 todo

Several Claude sessions work in this folder at once. Before starting a task, read this file and `git log -5`; add a line for your own work under "In progress" with your session name and the time, edit only your own lines, commit this file the moment you change it, and move your line to "Done" (or delete it) when the work has landed.

## In progress

- Testing against cases: the tests-cases-deaths small multiples and a tests and share-positive heat map per region, which Tommy liked as drafts; and `build_world_map.read_owid_tests`, which leaves out the 8 places OWID gives daily test counts but no usable running totals (France, Sweden, Czechia among them). Built and committed (31a5d2f, live since covid19-f8's 06:19 deploy; 465a81e; 6aaebc2): the figures are in `site/heat_tests_<region>*.png` and `site/tests_cases_deaths*.png`, their tags in `tmp/plot_heat_tests_markup.html` and `tmp/plot_tests_markup.html`. Nothing goes on the page until Tommy approves where and with what wording: covid19-6d (6 Oct 2026, 06:30).
- C13, the press pages: restyled, broken links repaired and live (covid19-f8, `press_international.html`, 690cfed; covid19-ac, `press_denmark.html`, e89e325). Their title, back link and archive note wait for Tommy to approve the wording covid19-f8 proposed; add none without his yes.

## Known problems

None open.
