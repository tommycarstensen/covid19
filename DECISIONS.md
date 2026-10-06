# Decisions about the covid19 page

Choices made while repairing and editing https://tommycarstensen.com/covid19/ (`site/index.html`) in October 2026, with the reason for each and what was considered instead. Newest last. `todo.md` holds work in progress; this file holds what was decided and why.

## How the October 2026 changes were chosen

On 6 October 2026 Tommy asked for the page's bugs to be fixed and for advisers to propose content to add, remove or restructure. Three advisers (Sonnet sub-agents) read the page from three angles: a 2026 visitor arriving from a search or a 2020 citation, information architecture and editing, and an epidemiologist and archivist checking provenance. Their suggestions were merged into one list (C1 to C13, below) alongside the bugs found by an automated audit (B1 to B6). Tommy chose to start with the ones recommended as the best value and work down the list. Covid19-4f does them one commit each, in this order: B1, B6, C1, C2, C4, C8, C9, C11, then C3 and C10, then the restructure (C5, C6, C7).

What the audit found sound, on 6 October 2026: no broken images, no horizontal overflow at 1,440 or 390 pixels wide, every file the page uses either in `site/` or on the server, every table row with all 14 cells, and all five map videos playable.

### The list

Bugs:

- B1. The EU table row left out Czechia but divided by the EU-28 population. Done, see below.
- B2. France, Sweden and Czechia have no table row: the January 2021 run wrote no table file for them, and its data are lost. Open: add rows from `ecdc.csv` marked one week earlier, or say on the page that they are missing.
- B3. Reference links that land somewhere else or nowhere. Handled under C9.
- B4. The weekly world maps showed seven-week totals. Covid19-91's interactive world map replaces them (see `todo.md`).
- B5. `plot_series.py` still holds the causes of the continent bug and B1. Open: fixing it means cleaning the whole script under ruff, pycodestyle and pyright.
- B6. Whether `sortable.js` sorts numbers as numbers. Open: the file exists only on the server, which stopped answering at about 04:45 on 6 October 2026.

Content, from the three advisers (who suggested it, effort):

- C1. Reword the archive note: when the data end and what that means; move the October 2026 repair note to a footer (all three, small). Drafted; waits for Tommy to approve the wording, because he did not ask for the original note.
- C2. A short "what happened after January 2021" with links to current data (reader, small). Part of the C1 draft.
- C3. A "How to read this page" section: sources, what "aligned" means, log scales, case fatality rate, fits are not forecasts, what the EU row covers (all three, medium).
- C4. The `<title>` and the table's column headers (all three, small).
- C5. Lead with the aligned small multiples and group the maps by measure (reader, structure; medium).
- C6. Move the per-country chart pairs to a subpage (structure, large).
- C7. Replace the table's three thumbnail columns with one link (structure, medium).
- C8. Trim "Infographics and other news clips" of live embeds and pre-2020 items (all three, small).
- C9. Cut the References to what still works (all three, small).
- C10. Captions and alt text for every chart; one heading for the loose bubble, heat and scatter charts (structure, accuracy; medium).
- C11. Credits for basemaps and data, licences, and how to cite the page (accuracy, reader; small).
- C12. The population column uses `countryinfo`'s figures of about 2014, which puts per-capita columns 3 to 8% high (accuracy, medium).
- C13. Give the press pages a title, a back link and an archive note, and check their links (accuracy, medium).

One adviser claim was checked and rejected: the table's totals are higher than `ecdc.csv` (the USA 23,938,288 against 22,423,006) not because they are wrong, but because the table came from a download one week later, to 17 January 2021. The table's "most recent week" for the USA, 1,515,282, is exactly the difference.

## 2026-10-06: the USA's continent in the table is Americas

`plot_series.py` set the USA's continent to "North America" by hand (`doCountry2Continent`), while `countryinfo` gives "Americas" for every other country of the Americas. Sorting the table by continent therefore put the USA on its own. The page now says Americas. The cause in `plot_series.py` is left for its own commit, because that script carries 80 ruff and 35 pycodestyle findings that have to be cleaned when it is touched (B5).

## 2026-10-06: the EU table row is labelled "EU without Czechia" and divides by its own population

The EU row summed the 27 countries in `plot_series.py`'s list, but the list spells Czechia "Czech Republic", which matches nothing in ECDC's data, so Czechia's cases (835,454 to 10 January 2021) and deaths were left out. The row then divided by a hard-coded 512.6 million, the population of the EU of 28 including the UK. Cases per million showed 33,062.9, about 15% too low for the 26 countries actually summed.

Chosen: keep the counts as published, label the row "EU without Czechia", and divide by the population of those 26 countries from `countryinfo` 0.1.2 (434.854 million), the source every other row of the table uses. Cases per million become 38,974.0 and deaths per million 944.2. The case fatality rate and tests per thousand do not depend on the denominator and are unchanged; the tests figure is OWID's population-weighted average over the same list, which also misses Czechia (OWID calls it "Czechia" too).

Considered and rejected: adding Czechia's cases, because the January 2021 download behind the table (one week later than `ecdc.csv`) is lost and Czechia's figure for that week cannot be recovered exactly; dividing by ECDC's 2019 populations (436.2 million), because the rest of the table uses `countryinfo`'s older figures and the row would no longer be comparable with them (that question is C12).

The EU charts under the European Union heading plot each member country's line from the same list, so they also lack Czechia. The aligned small multiples drawn in October 2026 (`plot_days100_world.py`) include it.

## 2026-10-06: the page title and the table's headers say what they hold (C4)

The browser title was "COVID-19 / 2019-nCoV", a name the WHO retired in February 2020; it now matches the page heading and the Twitter title: "COVID-19 tracker, 2020–2021 (archive)". The meta description already said "archive" and is unchanged.

The table's headers said "most recent week", which in 2026 reads as now. The week is 11 to 17 January 2021 (ISO week 2021-02): the USA's total in the table minus its ECDC total to 10 January is exactly its "most recent week", 1,515,282. "Cumulated ... (n)" became "Total", "Fatalities" became "Deaths" (the word the rest of the page uses), "plot" became "chart", and "Weekly bar plot (n)" became "Weekly cases and deaths chart", which is what those charts show (weekly bars of both, with a deaths-to-cases line). The sentence above the table now says where the numbers come from and when they end, and is a paragraph instead of loose text and a `<br>`.

## 2026-10-06: the table sorts by country and continent again (B6)

`sortable.js` existed only on the server, adapted in 2020 from the w3schools example credited in the page's `<head>`. It compared every cell as `Number(innerHTML)`, so numeric columns sorted, but the Country and Continent columns became `NaN` on both sides, every comparison was false, and clicking those headers did nothing. The server's copy of 6 October 2026 is kept, unversioned, in `tmp/bugfix/sortable_server_2026-10-06.js`.

The new `site/sortable.js` keeps the page's interface (`sortTable(n)` on each header) and its behaviour: a click sorts ascending, and a click on a column already in ascending order sorts it descending. It compares numbers as numbers and text alphabetically, sorts once instead of the example's repeated swapping, and keeps ties in their previous order. Tested in headless Chrome on all four kinds of column (country, continent, totals, per million), both directions. The three chart columns still have headers that can be clicked and do nothing useful; that is left to C7, which would replace them.

## 2026-10-06: "Infographics and other news clips" becomes "Press clippings, 2020" (C8)

All three advisers proposed trimming this section, for different reasons: it was the only part of the page that kept changing after January 2021, it loaded third-party scripts, and half of it predated the pandemic. Removed:

- The two Our World in Data iframes (tests per 1,000 people, world and South America). They are live charts that now run to June 2022, so they contradicted the page's January 2021 end date. The world chart's URL also redirected, so both iframes showed the same chart. The charts are now plain links in the trackers list (C9).
- The Datawrapper map "COVID-19 confirmed and recovered cases". It is a live map fed by Johns Hopkins, not a 2020 snapshot; Datawrapper's collection of such charts stays linked in the trackers list.
- The embedded tweet by Kan Nishida (14 April 2020) on Apple mobility data, and with it `platform.twitter.com/widgets.js`, the page's last third-party script. Apple withdrew the data in April 2022. (The page has no analytics of its own; the Google Analytics requests seen in the audit came from inside the OWID frames, which are gone too.)
- The 2018 Washington Post article on the White House pandemic office, the 2015 Vox and TED videos of Bill Gates, and Visual Capitalist's history of pandemics. They were background reading from the first weeks of 2020, not about the data on this page, and the two YouTube embeds were the page's heaviest third-party content.

Kept: the links to the two pages of press clippings Tommy collected in 2020 (`press_international.html`, `press_denmark.html`), with one sentence saying what they are. The section's anchor is now `#press` (`#infographics` was hours old and had never been linked to). One of the two `<hr>` before the section went, as did the two CSS rules for iframes, which nothing used any more. Everything removed is in the git history of `site/index.html` before this commit.

## 2026-10-06: "References" becomes "Other trackers", checked link by link (C9)

Each of the 18 links was opened in headless Chrome on 6 October 2026 (`tmp/bugfix/linkcheck.json`, unversioned):

- Replaced, because they now open an ArcGIS sign-in page: the WHO's 2020 dashboard, by the WHO COVID-19 dashboard at data.who.int; the Johns Hopkins ArcGIS dashboard, by the Johns Hopkins Coronavirus Resource Center map (Cloudflare blocks automated requests to it, so that link was not checked automatically).
- Removed: uscovid-19map.org, which now redirects to a notice that the site was decommissioned on 28 February 2023; Apple's Mobility Trends Reports, which say Apple stopped providing them on 14 April 2022.
- Updated to where they had moved: Our World in Data (the coronavirus page rather than the old data page, which redirected), Datawrapper's blog post, Reuters' US tracker.
- Kept as they were, with what they are now: Worldometer ("not updated since 13 April 2024", from its own notice), the CDC excess deaths page ("archived by the CDC", its own banner), Google's mobility reports, covidexitstrategy.org.
- Kept without checking, because their sites refuse automated browsers (a robot check or an HTTP 403): Bloomberg, the South China Morning Post, The Guardian, the New York Times (two links), the Washington Post and The Economist. They are major publishers that keep their 2020 pages.

The links are now a list grouped by kind (data sources, news trackers, excess deaths, mobility, reopening) instead of lines broken by `<br>`. The heading says what the section is ("Other trackers"; the anchor stays `#references`). The two OWID testing charts that C8 took out of the page are linked here. The invitation to e-mail suggestions to covid19@tommycarstensen.com is gone, because it implied someone still maintains the page.
