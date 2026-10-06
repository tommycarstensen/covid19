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
- B5. `plot_series.py` still holds the causes of the continent bug and B1. Done, see below.
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
- C12. The population column uses `countryinfo`'s figures of about 2014, which puts per-capita columns 3 to 8% high (accuracy, medium). Done, see below: for a few countries the error was far larger.
- C13. Give the press pages a title, a back link and an archive note, and check their links (accuracy, medium). Restyled and their broken links repaired by covid19-f8 (international, 690cfed) and covid19-ac (Danish, e89e325); the title, back link and note wait for Tommy to approve their wording.

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

## 2026-10-06: an "About this page" section with authorship, sources, licences and how to cite (C11)

The page credited only its flags and continent icons. It now ends with a section, also listed in the contents, that says who made the page and when, where the code is (github.com/tommycarstensen/covid19, public), and how to cite it, then credits the data and the map outlines:

- ECDC for cases, deaths and the European regional rates. ECDC's copyright notice allows reproduction provided the source is acknowledged.
- Our World in Data for tests, under CC BY 4.0, which requires the credit.
- Natural Earth (public domain) for the world maps: `plot_choropleth.py` used geopandas' `naturalearth_lowres`, and `build_world_map.py` uses Natural Earth's 1:110m countries.
- For the European map (`plot_choropleth_europe.py`): Eurostat GISCO's NUTS 2021 boundaries, whose terms require "© EuroGeographics for the administrative boundaries", and the NHS health boards of Scotland (Scottish Government, 2019) and Wales (ONS, December 2016), both under the Open Government Licence v3.0 with Ordnance Survey data.

The existing flag credits moved into the same section, unchanged. The sentence about the October 2026 repairs is to move here from the archive note, in the same commit as the new note (C1), so that it is never on the page twice.

## 2026-10-06: Taiwan's name in traditional characters

The heading said "Taiwan / 台湾", in the simplified characters of the People's Republic of China. Taiwan writes traditional characters, so it is now "台灣", the common form (the official form is 臺灣). The other headings' local names were not changed.

## 2026-10-06: a "How to read this page" section (C3)

All three advisers asked for it: the page explained none of its terms. It is a short list right after the contents, also listed in them, and covers only what a reader needs to read the charts correctly: what a confirmed case is and why counts compare poorly between countries; what "aligned" means (the week a country passed 1,000 cases or 100 deaths, which the charts' own subtitles confirm); how to read a log scale; that the case fatality rate is not the infection fatality rate; that the table's populations are from about 2014 (`countryinfo` 0.1.2: Germany 80,783,000, Denmark 5,655,750); and which "EU" each chart means.

Left out on purpose: sources and licences, which "About this page" has (C11); end dates and what happened later, which the archive note is to say (C1, C2); and a warning that fitted curves are not forecasts, which one adviser asked for, because no chart on the page shows a fitted curve. Two claims were narrowed after checking: not every chart uses a log scale (the Europe map and the heat maps are linear), and not every country grew after 2014, so the populations are described as a few per cent off rather than too low.

## 2026-10-06: alt text for every chart, and a sentence on what each group of charts shows (C10)

351 of the page's 394 images had no alt text: everything except the flags (`alt=""`, decorative, correct) and the charts added in October 2026. Each now has one, written by rule from its file name, after looking at an example of each kind:

- `days100_{cases,deaths}_perCapitaFalse_<country>.png`: cumulative cases (deaths) in the country, in red, by week since it first passed 1,000 cases (100 deaths), with the other countries of its continent dark grey and the rest of the world light grey, log scale. The USA's two charts say "every other country in light grey", because `plot_series.py` gave the USA a continent of its own ("North America"), so no other line is dark.
- The same file names for a region (EU, Europe, the Americas and its parts, Asia and its parts, Africa, Oceania, the Nordic countries): one line per country, aligned the same way.
- The table's thumbnails, which are links: "Chart of cumulative cases (deaths, weekly cases and deaths) in <country>", describing where the link goes.
- `plot_heat_*`: weekly cases (deaths) per million, one row per country, week by week from the start of 2020.
- `scatter_EU_*`: each EU country's highest number in a single week against its population. `doScatterPlots` plots `max()` of the weekly column, so the charts' axis label "Cases" is wrong; they show each country's peak week (France about 370,000 cases, in November 2020), not its total (2.9 million). A sentence under the two charts says so, since the PNGs cannot be redrawn with the January 2021 data.

Three short paragraphs explain the groups: one before the regions (the four kinds of chart), the scatter sentence, and one under "Charts by country". The structure adviser counted 87 countries in that section; it has 23 (46 charts): 15 countries' charts from 27 January 2021 and 8 redrawn by `redraw_charts.py` in October 2026, which the paragraph says. The alt texts were added by a one-off script, not kept, because the page is now edited by hand and its rules are the list above.

Correction to C3, the same day: the "How to read" bullet on the EU said that all the charts under the EU heading leave out Czechia. The interactive bubble chart there, built in October 2026 by `build_bubbles.py`, includes it. The bullet now names what leaves Czechia out: the table's row and the January 2021 line charts, heat maps and scatter plots.

## 2026-10-06: the table's three thumbnail columns become one "Charts" column of links (C7)

The table had three columns of 45-pixel-high thumbnails (total cases, total deaths, weekly bars) for each of its 87 rows: 261 images, about 3 MB, each too small to read, and each only a link to the full-size chart. They also made the table four columns wider than its numbers needed, which on a phone meant more sideways scrolling, and their headers could be clicked to "sort", which did nothing. They are replaced by one last column, "Charts", with three text links per row (cases, deaths, weekly) to the same full-size charts. The numeric columns now sit together, are renumbered for `sortTable`, and were re-tested in both directions; the "Charts" header is not clickable. The cell keeps its three links on one line, so each row is one or two lines high instead of the thumbnails' height: 13 rows fit on a 900-pixel screen instead of 8.

The thumbnail files stay on the server (`deploy.py` never deletes), and `redraw_charts.py` still draws them; nothing on the page uses them any more. Their alt texts from C10 went with them.

Considered and rejected: keeping one thumbnail column as a sparkline. The thumbnails are scaled-down full charts, with the country's red line among 80 grey ones, not sparklines; at 45 pixels the red line is barely visible.

## 2026-10-06: the charts load as the reader reaches them, instead of a subpage for the per-country charts (C6, revised)

C6 proposed moving the per-country charts to a subpage, to take about 175 full-size images off the page. A count found 46 (23 countries), and the table's 261 thumbnails, the real weight, went in C7. What remained was that every image loaded at once, and that 88 charts had no width or height, so the page jumped as they arrived. Every chart outside a `<noscript>` (94) now has `loading="lazy"` and its real width and height, read from the copy the server serves (`site/`, else the newest in `archive/`), with a CSS rule (`img[loading="lazy"] { height: auto; }`) so that a phone scales them in proportion. Tested in headless Chrome with every image served locally: none broken or distorted at 1,440 or 390 pixels, about 40 of the 125 images load before the reader scrolls, and a link to `#countries` lands on its heading.

Whether "Charts by country" should stay at all is a separate question, taken up below.

## 2026-10-06: the table comes right after the maps (C5, revised)

C5 proposed leading with the aligned small multiples and grouping the maps by measure. By the time it came up, covid19-91's interactive world map had already grouped the world maps by measure, and a design review by covid19-b6 pointed out that the table, probably what most visitors come for, started about 21,500 pixels down, after every region chart. The sections now run: archive note, contents, how to read, maps, table, aligned time series (world and regions), charts by country, press, other trackers, about. The maps stay first because the interactive world map is the page's best overview and is short; the table follows because it is the page's reference; the long run of aligned charts comes after both. The contents list follows the same order.

## 2026-10-06: France and Sweden get their table rows back (B2)

France and Sweden had charts on the page but no row in the table: the January 2021 run drew their charts but wrote no `tables/table<Country>.txt`, perhaps because Our World in Data had no testing totals for either. Their rows are rebuilt from what survives of that run:

- Totals to 17 January 2021 from the legends of the EU charts drawn on 27 January 2021 (`days100_{cases,deaths}_perCapitaFalse_EU.png`): France 2,910,989 cases and 70,283 deaths, Sweden 531,145 and 10,764. Check: with them, the table's 24 other EU countries sum exactly to the EU row (16,948,020 cases, 410,569 deaths).
- The week of 11 to 17 January as the total minus `ecdc.csv`'s total to 10 January: France 127,733 cases and 2,533 deaths, Sweden 28,918 and 1,098. The same subtraction gives the table's own weekly figures exactly for 84 of its 86 rows; the other two differ because ECDC revised earlier weeks, so these two rows may be off by such a revision.
- Populations from `countryinfo` like every other row (France 66.1 million, Sweden 9.7 million), and the case fatality rate and figures per million computed the same way. Tests per thousand: "–", because Our World in Data has none for either.
- Charts: the 27 January 2021 charts already on the server.

Czechia, the third country missing, is left out: no chart of it from 2021 survives, so its total to 17 January cannot be recovered, and a row to 10 January would not compare with the others.

## 2026-10-06: "Charts by country" is removed; its anchors point at the table rows

The section showed two full-size charts each for 23 of the table's 89 countries, in alphabetical order, about 10,000 pixels or a quarter of the page. Every one of those charts is also linked from its country's row in the table, which covers all 89, so the section added length but no charts. Covid19-b6's design review proposed dropping it, and C6's subpage would only have moved the same repetition elsewhere.

The section's 23 anchors (`#denmark`, `#united-kingdom` and so on, added by covid19-82 earlier the same day) now sit on the matching table rows, so any link to them still works: it scrolls the row a third of the way down the screen and highlights it. Tested in headless Chrome at 1,440 and 390 pixels. Lost with the section: the headings' links to the Wikipedia articles on the pandemic in eight of those countries, and the countries' names in their own languages (Danmark, Deutschland, 日本 and so on). The contents list no longer has the section or its 23 entries.

## 2026-10-06: the table is easier to read: separators, right-aligned numbers, largest first, a sort arrow

From covid19-b6's design review of the table:

- Thousands separators (23,938,288; 74,980.8), with `sortable.js` ignoring commas when it compares.
- Numbers right-aligned in tabular figures (population and columns 4 to 11), so digits line up.
- The rows start in order of total cases, largest first (the USA, then the EU without Czechia, India, Brazil). They had been in the order the 2021 run happened to write them (EU, USA, Germany, Spain, Bahrain...).
- The sorted column's header shows ▲ or ▼, from an `aria-sort` attribute that `sortable.js` sets, which screen readers also announce. "Total cases" starts with ▼.
- Cell padding 8 by 12 pixels instead of 16 (covid19-82's phone padding, 6 by 8, is unchanged): 19 rows fit on a 900-pixel screen instead of 13.
- Cells with no data ("–": tests for France and Sweden) stay at the bottom whichever way a column is sorted; before this fix they came first when sorting high to low.

Kept as it was: a first click sorts low to high, as the 2020 script did.

## 2026-10-06: the archive note is removed, on Tommy's instruction (C1 and C2 superseded)

C1 (reword the archive note at the top of the page) and C2 (a short "what happened after January 2021" with links to current data) were drafted but held back, because Tommy had not asked for the original note and was to approve any new wording first. Tommy then asked covid19-91 to remove the note altogether. No replacement goes in unless Tommy asks for one. Gone with the note: its sentence about the October 2026 repairs (the 63 redrawn countries) and the planned correction about the weekly world maps' seven-week totals. The page still says when its data end in the table's introduction ("up to 17 January 2021"), in the aligned charts' captions and in the browser title ("archive").

## 2026-10-06: Western Asia except Iran's line charts become small multiples (covid19-cd)

Tommy asked whether the `days100_cases_perCapitaFalse_AsiaWesternExIran.png` figure was any good, and to redo it if not. It was not: 16 countries shared a 10-colour cycle (Turkey and Armenia both blue, Iraq and Kuwait both orange, Saudi Arabia and Qatar both red), the legend sat over the lines, the subtitle said "Weekly cases" over cumulative counts, and the y axis said "Cumulated". The cases and deaths pair is replaced by `plot_days100_world.py`'s small multiples, as the World1/World2 pair was: one panel per country, the country in blue over the other 15 in grey, three rows of six on wide screens and two columns on phones, with every total in the alt text.

- The set is kept as it was (`plot_series.py`'s AsiaWestern without Iran, which World1 shows), so the heading and the figures beside it are unchanged; it matches `regions.py`'s PARTS.
- "UAE" in the panel titles, where "United Arab Emirates" ran into its total; the alt text keeps the full name.
- Every aligned figure's y axis now ends at the next 1-2-5 step above its highest line, instead of the next whole decade, which had left the top quarter of each Western Asia panel (10M) and World1 panel (100M) empty. The World1/World2 figures were redrawn with it and nothing else changed.

Considered instead: one figure per continent in place of the 2020 subregion pairs (covid19-b6's suggestion). Not taken here, because it restructures the page beyond what was asked; covid19-b6 is redrawing the other regional pairs one figure per pair.

## 2026-10-06: the country table is folded away by default (covid19-b1)

Tommy asked whether the table should show by default or open with a click, since its 89 rows of 12 columns were a long block of text on an otherwise visual page. It is now folded: the heading and the sentence introducing it stay where C5 put them, right after the maps, so the table is still found there, and the rows sit in a `<details>` labelled "Table of 88 countries and the EU", one click away. That takes the table off the path from the maps to the aligned charts, and it is a reference that people look things up in, not something they read from top to bottom.

- It looks like the world map's own fold ("Table of all 213 countries and territories"): the browser's triangle and a plain label, with the same `cursor` and margin. A boxed "Show the table" / "Hide the table" button was tried first and dropped, so that the page has one style for one kind of control.
- Outside links point to `#table` and to rows such as `#denmark`. A short inline script opens the fold when the URL's fragment is either, and on `hashchange`; Chrome opens it for a row by itself. Chrome and Edge also open it when Ctrl+F finds a match inside; not every browser does.
- Without JavaScript the fold still opens and closes; only the opening on a link needs the script.

Considered instead: showing the ten largest rows with a "show all" control, which would need `sortable.js` to sort the hidden rows too, for little gain over a fold.

## 2026-10-06: one line under the heading says the page is an archive and when its data end

After the archive note was removed, nothing near the top said that the page is frozen or when its data end. Tommy asked ten advisers (Sonnet sub-agents) with different roles what to put there, with one constraint: no year of the repairs. The roles: a member of the public arriving from a search engine, a journalist who linked to the page in 2020, an epidemiologist, a web archivist, a UX writer, a newspaper graphics editor, an accessibility specialist, a search specialist, a hiring manager looking at Tommy's work, and a fact-checker worried about screenshots shared as if current.

All ten wanted a line, and all ten wanted it as a plain paragraph straight under the heading, with no coloured box, banner or alert, and no account of repairs. Nine wanted both end dates (table 17 January, charts and maps 10 January), eight wanted ECDC named, and several wanted the first words in bold so they survive a cropped screenshot. Tommy asked for the good advice to be acted on, and the line is that consensus:

> **This page is an archive and is no longer updated.** Its data, from the European Centre for Disease Prevention and Control (ECDC), end in January 2021: the table's counts run to 17 January and the charts and maps to 10 January.

Left out, because the advisers split or the page already does it: the worldwide totals (89.8 million cases, 1.94 million deaths; three for, three against as a number easy to quote out of context); a link to current data (two for, one against because such links date; "Other trackers" already links the WHO and Our World in Data); first-person wording ("a dashboard I ran", one adviser), because the page speaks in the third person elsewhere. The line uses a new class, `standfirst`, styled only with the page's text width; the yellow-box CSS of the removed `archive-note` went with it.

## 2026-10-06: every region's line charts become small multiples, in the 2020 charts' red (covid19-b6)

Tommy asked covid19-b6 for a design review of the whole page and then to work through it. Its first finding was that the region sections still showed `plot_series.doLinePlots`' 2020 charts: 4 to 55 countries on one 480x360 plot with a 10-colour cycle, so countries shared colours, and a legend too small to read that in Europe ran over the x-axis label. The World figures had already been redrawn as small multiples (`plot_days100_world.py`), and covid19-cd had done Western Asia except Iran (8b8b4b7).

Chosen: the other eleven 2020 pairs (the EU, Europe, the Americas, North America, South America except Venezuela, Asia except China, South-East Asia, East Asia except China, Africa, Oceania, the Nordic countries) redrawn one for one by the same script, so each section keeps its own figure and the page's headings and anchors stay as they were. The country lists are `plot_series.py`'s, moved to `regions.py` under ECDC's names, which also brings Czechia back into the EU and Europe.

- A set of more than 20 countries gets eight columns, three on phones, instead of three rows, which at 55 countries would have meant 19 columns. On the phone's three columns a name and its total do not fit on one line, so the total goes under the name.
- In those large sets, the countries that never passed 1,000 cases (100 deaths) are named under the figure with their totals, instead of each taking an empty panel: Africa has 21 such countries for deaths.
- The country's line is red, `#e41a1c`, over grey, as in `plot_series.doLinePlots`' own comparison charts, following the rule that a redrawn chart takes Tommy's 2020 colours (the project memory "palette-from-original-scripts"). This replaced the dataviz blue the World figures had used, so they changed colour too; nothing else in them changed.
- The script stops if a panel's name runs into its total or any text runs off a figure, which caught Bosnia and Herzegovina, North Macedonia and Liechtenstein on the first phone draw.

Considered instead: one figure per continent, dropping the five subregion pairs, which would have made the page shorter. Not done, because covid19-cd had just been asked for the Western Asia figure, so the subregions were evidently wanted. Also considered: dropping the region line charts altogether in favour of the bubble charts and heat maps; not done, because the aligned charts are what the section is named for.

Left as they were: the USA's own pair of 2020 charts under its heading, and the EU's two scatter plots.

## 2026-10-06: search results and link previews say "archive, no longer updated" (from the search specialist's advice)

Search engines and social sites show a page's meta description and Open Graph tags, often before anyone opens it. The description now begins "An archive, no longer updated" and ends "from ECDC data to 17 January 2021", and the Twitter description, which had said only "to January 2021", is the same text. Facebook and LinkedIn read Open Graph tags, which the page lacked; it now has `og:type`, `og:url`, `og:title`, `og:description` and `og:image`, with the image's size and a text alternative. The preview image is the World1 small multiples of cumulative cases (`aligned_cases_World1.png`, 2,710 by 1,454 pixels) instead of the 2020 EU line chart with 26 lines in 10 colours. The Twitter title uses an en dash, like the page's title.

## 2026-10-06: "About this page" says the source file is withdrawn and its figures revised (from the epidemiologist's advice)

The epidemiologist among the ten advisers wanted the top line to warn that the ECDC data were later withdrawn and may differ from figures published now. That is for someone citing the numbers, not for every visitor, so it went into the data paragraph of "About this page", next to the source and licence: ECDC has withdrawn the weekly file by country that the page used (`ecdc.csv` in this repository is the only copy kept), and its figures have since been revised, so they may differ from what ECDC, the WHO or national agencies publish now.

## 2026-10-06: no repair history on the page

The intro to "Other trackers" (C9) said that in October 2026 two links had been replaced and two removed. Tommy wants no year of the repairs on the page, and all ten advisers on the top line argued against an account of repairs anywhere a reader meets it. The sentence is gone; this file keeps the record. The page now contains no "2026".

## 2026-10-06: the table's populations are ECDC's of 2019 (C12, covid19-67)

The table divided by `countryinfo` 0.1.2's populations, of about 2014, while the world maps and the heat maps divide by ECDC's of 2019 in `ecdc.csv`, so the same country had two different rates per million on one page. The gap was not a few per cent everywhere. Over the 89 rows the per-million columns were a median 4.4% too high (a quarter of the rows by more than 8%), but `countryinfo` counted the whole island of Ireland (6.4 million against 4.9), and had Jordan at 6.7 million (10.1), the Maldives at 0.3 (0.53), Kuwait at 3.3 (4.2) and Bosnia and Herzegovina at 3.8 (3.3). Ireland's cases per million were 23% too low, Jordan's 52% and the Maldives' 56% too high.

Each row now takes its population from `popData2019` in `ecdc.csv`, and the EU row the sum of its 26 countries (436.2 million), and cases and deaths per million are recomputed from the row's own totals. Nothing else in a row changes: the totals, the case fatality rate, the last week, and tests per thousand, which is Our World in Data's own figure on its own populations. The USA becomes 329.1 million and 72,746.4 cases per million (was 319.3 and 74,980.8). B1's reason for dividing the EU row by `countryinfo`'s populations, that every other row used them, no longer holds. The "How to read" line that said the table used populations from about 2014 now says 2019.

## 2026-10-06: `plot_series.py` fixed where the EU, continent and population bugs were made (B5, covid19-67)

The page's repairs (B1, the USA's continent, C12) corrected the published table, but the 2020 script still produced the old errors, and `redraw_charts.py` still calls it. It now takes its region lists from `regions.py` (Czechia spelt as ECDC spells it, the 24 places the 2020 lists missed, each place once), gives the USA the continent Americas, and divides by ECDC's populations of 2019 instead of `countryinfo`'s of about 2014 and a hand-written EU of 28. Run on `ecdc.csv`, its EU row has 27 countries and 446.8 million people.

Fixed with it, because they stopped the script from running at all on the installed pandas 3 and matplotlib 3.11 or were plain mistakes: `DataFrame.append` and `Figure.set_tight_layout`, both since removed; a debug `exit()` at the top of the bar charts; an undefined name in `parseArgs`; columns the weekly data do not have. `main()` reads `ecdc.csv`, because ECDC's URL now serves another file. The whole file is clean under ruff, pycodestyle and pyright, as Tommy's rules require of a touched file.

Not done: redrawing anything. The page's images are as before, and the January 2021 table stays as published apart from its repairs. A rerun of `redraw_charts.py` would add the 24 places to its charts' grey lines.

## 2026-10-06: the international press page restyled, and ten advisers on what is left (C13, covid19-f8)

Tommy asked for `press_international.html` to look nice. It was raw 2020 HTML: no doctype, no CSS, no title, entries separated by `<br><br>`. It is now one `<article>` per entry, with a grey line of date and source above the headline, which is the link, in the main page's fonts and link blue on a 720-pixel column (690cfed). It adds no words. The restyle dropped the " - " separators, the bold on some headlines (it followed no pattern) and CNBC's standfirst, which repeated its headline word for word. It fixed the date "202-04-22", and put the date first in "Vox 2020-03-13" and "Business Insider 2020-03-09". The Reckitt link, which now lands on Reckitt's home page, and a Business Insider article that answers 404 point to Wayback copies of 1 May and 14 March 2020. MedPage Today's link was empty in 2020 and its address could not be found, so the entry is plain text. All 14 videos, 24 tweets and 4 images still load. NYT, AP, Reuters, Bloomberg, Time and the Washington Post refuse scripts, so their links could not be checked.

Ten advisers (Sonnet sub-agents) then judged twelve open decisions, from the angles of a 2026 visitor from a search, information architecture, an archivist, a typographer, accessibility, performance and privacy, a news editor, Tommy's own rules, rights and attribution, and a devil's advocate for doing less. Their tally, and what was done:

- Keep every headline alike rather than restore the 2020 bold: 10 of 10. Dropping CNBC's repeated standfirst was right: 10 of 10. Keep MedPage Today as plain text, never remove it: 10 of 10.
- Keep the Wayback links: 8 of 10 (2 wanted both links shown, which adds words). Done, with the original address kept in `data-original-href`.
- Switch YouTube to youtube-nocookie.com: 8 of 10. Done, with `twitter:dnt` for the tweets (7ecd0f7).
- Fill in "AP" and the uncaptioned video's title as visible text: 4 for, 6 against. Not done, but the video's iframe takes its YouTube title as its accessible name, which shows nowhere.
- Month headings: 3 for, 7 against. An archive note: 0 for, 9 against, 1 for asking Tommy. Neither done.
- A `<title>`: 9 of 10 for, several calling it required (WCAG 2.4.2); a visible h1: 3 for. A back link to the tracker: 3 for, the others only if Tommy asks. These are words on the page, so they wait for him. The wording most advisers converged on, from the main page's own "Press clippings, 2020" and "international and US press": "Press clippings, January to May 2020: international and US press".
- The four hotlinked images: 4 for copying them here, 3 for keeping the hotlinks, the rights adviser against copying (it republishes others' work). Left as they are, for Tommy: they now reserve their height and send no referrer. The SCMP image is a looping 674 KB GIF, which fails WCAG 2.2.2 (no pause) while it stays.
- One `press.css` for both press pages: 6 for, 3 for inline. Not done: `deploy.py` uploads only the files `index.html` names, so a stylesheet linked from the press pages would never be sent.

Fixed from the advisers' other findings: date and source had no space between them, so a screen reader read "2020-05-14ProPublica"; MEDPAGETODAY is written MedPage Today; links show a focus ring; headlines break in balanced lines, with COVID-19 kept whole. Not taken: a `noindex` (one adviser), hidden headings for the tweets and videos, and making the source names consistent ("AP" and "AP News", "The New York Times" and "New York Times"), which would rewrite Tommy's labels.

Tommy then chose (832d99c): the tab title "Press clippings, January to May 2020: international and US press", with no visible heading; a "← COVID-19 tracker" link at the top of the page; and the four images left hotlinked. He asked for backups of every image the pages load from other sites, which covid19-ac is writing.

## 2026-10-06: the EU's scatter plots and the USA's charts redrawn in the October 2026 style (covid19-67)

The last two items of covid19-b6's design review: the USA section's pair of 2020 line charts and the EU's two 2020 scatter plots, all 480x360 in the old style, between figures redrawn in October 2026.

The scatter plots also had a data bug: `plot_series.doScatterPlots` read the EU list that spelt Czechia "Czech Republic", so Czechia was missing, and took `countryinfo`'s populations of about 2014. `plot_scatter.py` redraws them from `ecdc.csv` with all 27 member states: one point per country, its highest single week against its population, both on log scales as before, in the 2020 charts' matplotlib blue (#1f77b4), every country named. Added: dashed lines at equal weekly rates per million people (1,000 and 10,000 cases, 10 and 100 deaths), because on log-log axes of a count against a population the point's height alone mostly says how big the country is, and the lines let a reader see which countries had the highest peak rates (Ireland 9,324 cases and Slovenia 458 deaths per million in a week). Considered and rejected: plotting rates per million instead of counts, which would be a different chart from Tommy's. The names are placed by a search, because the countries of about 10 million (Sweden, Portugal, Greece, Czechia, Austria, Hungary) nearly coincide: each name must overlap no other name, point or guide label, a name beside its point must be clearly nearer it than any other point, and a leader line may cross no name, point or other leader. The page's caption kept its first sentence; "(left)" and "(right)" went, because on phones the panels are stacked, and so did the note that the axes say only "Cases" and "Deaths". "How to read" now counts the scatter plots among the charts that cover all 27 member states.

The USA's figure comes from `plot_days100_world.py`, whose small multiples replaced every other aligned chart: two panels, cases and deaths, the USA in red over every other country and territory that passed the threshold in grey, as the 2020 pair showed it. A one-country set of small multiples would have left most of a four-column row empty, hence the two-panel figure.

The January 2021 files (`scatter_EU_{cases,deaths}.png`, `days100_*_United_States_of_America.png`) stay on the server: `deploy.py` never deletes, and the table's chart links still open the 2020 per-country charts.

## 2026-10-06: the Danish press page's title and back link, and the testing figures, after ten advisers (covid19-67)

Tommy asked for the two decisions left open, the press pages' words and covid19-6d's testing figures, to go to ten advisers (Sonnet sub-agents) and then be decided. By then he had decided the international press page himself (832d99c), and covid19-6d had already put the testing figures to a panel of its own at his request, so these ten took the Danish page in full and gave the testing plan a second look. Their roles: a 2026 visitor from a search, a web archivist, a Danish journalist, an information architect, accessibility, the guardian of Tommy's rules, an epidemiologist, a newspaper graphics editor, a copy editor, and a devil's advocate for doing less.

- The Danish page's tab title becomes "Press clippings, March to October 2020: Danish press", in the international page's pattern: 7 of 10 (1 wanted it in Danish; 2 wanted the 2020 "Danish Press on COVID-19" kept, one of them misreading the pattern's dates). The advisers proposed "January", from a DR Detektor entry of 23 January 2020, but covid19-ac found that entry inside an HTML comment since 2021, so no visitor sees it: the visible clippings run from 1 March to 3 October 2020. The title carries lang="en", because the page is lang="da".
- The same "← COVID-19 tracker" link at the top as on the international page: 10 of 10, with lang="en".
- No archive note and no visible heading: 10 of 10.
- The main page's "Their links have not been checked since." is deleted (342d5e3): 4 to delete; 6 to reword it to the Danish page alone, which covid19-ac's repair of that page's dead links (Wayback copies) makes untrue as well.
- covid19-6d's five page edits for the testing figures (a "Tests, cases and deaths" section between the maps and the table, each region's tests heat map under its cases-and-deaths heat map, the chart-types sentence, a link from "how much a country tested"): no objection from 9 of 10; the devil's advocate objected to the volume of seven more heat maps, and they stay. Asked of the figures first: the United Kingdom panel's title crowding its share label; the Dutch caveat (OWID counts people tested there) in the 16-country figure's source line and the shares' approximate nature in its alt text; a note in the heat maps' source line that a share from very few tests, as in early 2020, is unreliable.

covid19-ac adds the title and link to the Danish page in its link-repair pass; covid19-6d makes the testing edits.

## 2026-10-06: a "Tests, cases and deaths" section and a tests heat map per region, after ten advisers (covid19-6d)

Tommy asked whether the page had explored how far more testing explains more cases. It had not: "How to read" says case counts depend on testing, the bubble charts colour each country by its change in tests, and the world maps and the table show tests per thousand, but nothing put tests and cases on one scale or showed the share of tests positive. Tommy liked two drafts and asked for the open questions to go to ten advisers (Sonnet sub-agents): a 2026 visitor from a search, a surveillance epidemiologist, a biostatistician, a newspaper graphics editor, an information architect, accessibility, a web archivist, the guardian of Tommy's rules, a fact-checker who spent 2020 debunking "cases only rose because of testing", and a devil's advocate for doing less. Their tally, and what was done:

- The 16-country figure of weekly tests, cases and deaths on one log scale (`plot_tests.py`): 10 of 10. In a new h2 between the world maps and the table: 10 of 10; headed "Tests, cases and deaths" (`#testing`): 8 (one wanted "Testing", one "Tests and cases").
- A tests heat map under each region's cases and deaths heat map, with no heading of its own (`plot_heat.py`): 8 (one wanted it only where nearly every country has tests, the devil's advocate none).
- The chart-types sentence before the regions names them, and the words "how much a country tested" in "How to read" link to `#testing`: 9 of 10 each. No caption under the figure: 9 of 10.
- A caveat that the cases (ECDC) and the tests (Our World in Data) may not count the same things, rapid antigen tests among them, in the figures' own source lines and not on the page: 10 of 10, because a screenshot loses the page.
- The colours stay: OrRd for the share positive (the 2020 heat maps' colour map), Blues for tests (the 2020 weekly tests map's), and Dark2 for the lines (the hues of the 2020 bar charts' Set2, whose pastels fail contrast as thin lines): 10 of 10. The tests scale, 0.01 to 100 per thousand people per week in every region: 5 to keep it, 5 to narrow it to 0.1 to 100, so it stays.
- From their findings: the 16-country figure's subtitle lost its sentence that a rise "more testing does not explain" (an overclaim, since the share also depends on who was tested) and now says only how to read the gap, "if both count the same thing", with half a step about 30%; "positive at peak" became "positive when cases peaked"; the source line says a week with none reported leaves a gap; the heat maps' key gives its 0 swatch room; the alt texts give the region's most tests in a week and say a darker colour means more.

covid19-67's panel, above, asked for the Dutch caveat, now in the 16-country figure's source line and alt text (Our World in Data's Dutch figures count people tested; the epidemiologist found RIVM reporting 16.6% positive at its test sites in late October 2020, while the figure's Dutch share peaks at 34%), and for a note that a share from very few tests is unreliable, now in the heat maps' source lines. The early-2020 European shares it pointed to are 9% to 30%, not near 100%. But 20 weeks of 2020 in Peru, Brazil and Ecuador have more confirmed cases than counted tests (up to 228% in Brazil), which the share heat map had drawn in its darkest red; they are now hatched, as "no usable share".

Found and fixed on the way (31a5d2f, live since 06:19): `build_world_map.read_owid_tests` read only Our World in Data's running totals, so the world maps showed no tests at all for 8 places that have only daily counts, France, Sweden and Czechia among them.

Not changed: the weekly tests world map keeps its 2020 GIF's range, 0.001 to 10 per thousand, so from autumn 2020 most of Europe and North America is its darkest blue.

## 2026-10-06: the Danish press page restyled, its links repaired, and Tommy's corrections after ten advisers (covid19-ac)

Tommy asked for `press_denmark.html` to look nice. It was raw 2020 HTML, like the international page, and was restyled the same way (e89e325), keeping every word: one `<article class="clip">` per entry, the international page's 720-pixel column, fonts and colours, the sections as h2 with ids. The nine ft.dk committee answers it framed at 827 by 1,169 pixels were blank, because ft.dk refuses to be framed, so each question's label links to its answer instead. The 2020 Nordic charts gave way to the main page's redrawn ones.

Repairs, not decisions (6a414fa): the eight sst.dk pages and DR's Deadline episode, all 404, link to their 2020 Wayback copies, the original kept in `data-original-href`; two entries whose 2020 links were broken link to their articles again, found through the Wayback Machine's index. TV Avisen of 12 May 2020 is 404 with no copy and keeps its dead link (the advisers split 5 to 5 on unlinking it). The 14 ft.dk links answer 403 to scripts (Cloudflare), but all work in a browser: checked in Chrome later the same day, each answer's title matches its label. "Spørgsmål nr. 852" links to a document filed under question 857 that answers 852, because the minister answered 852, 856 and 857 together. covid19-67 decided the tab title and the back link (above).

Ten advisers (Sonnet sub-agents: an archivist, a Danish media historian, an accessibility specialist, a front-end engineer, an information designer, a fact-checker, an epidemiologist, a news copy editor, a first-time visitor and a privacy and media-law adviser) answered eight questions, and Tommy chose (0759d54, 965b908):

- The two Berlingske entries whose 2020 link text was empty show their date and source, from the articles' metadata: 2020-05-07 and 2020-05-04 (10 of 10).
- Two labels that contradicted their links are corrected: "2020-05-14 DR" on a Berlingske article, and the second "Spørgsmål nr. 796" on the answer to 794 (10 of 10).
- The second, identical copy of "2020-05-12 Berlingske – Internt dokument…" is gone (8 of 10).
- "2020-04-15 Dansk Journalistforbund", which embedded Journalisten's tweet of 16 April a second time, is gone (5 of 10; the others would have kept the page as published).
- The Nordic aligned-cases figure at the top is gone; it stays in Visualisering (6 of 10).
- The three Folketing clips of 29 April 2020 are MP4s (10 of 10): the server sent the .mov files as video/quicktime, which Firefox does not play. `build_press_videos.py` copies their streams unchanged into MP4s with the index first, with a poster each; the .mov stays as a second source. `deploy.py` now also uploads the files the linked pages use (83fc64d), which it did not before, so the international page's `press.css` idea above is no longer blocked by it.
- Twitter's widgets.js is gone from both press pages (1 of 10, the privacy adviser: it sent every visitor's IP address to X and set X's cookies without consent). The tweets show as the styled quotes they fell back to, with no word changed (0759d54, edb77dd). One of them, DR2 Deadline's of 22 April 2020, had already been deleted.

Tommy also asked for a backup of every image the pages take from other sites: `backup_external_images.py` saved 63 files to `external_images/` (343a91b), with a manifest; the deleted DR2 tweet's image came from the Wayback Machine's copy of the tweet.

## 2026-10-06: world totals and the record last week under the archive line (covid19-59)

Tommy asked whether the page should have headline numbers at the top. The earlier panel had split three to three on putting the world totals in the archive line, so Tommy asked ten new advisers (Sonnet sub-agents): a member of the public from a search engine, a data journalist who linked to the page in 2020, a surveillance epidemiologist, a web archivist, a UX writer, a newspaper graphics editor, an accessibility specialist, a fact-checker of out-of-context screenshots, a hiring manager, and a devil's advocate for adding nothing. They saw a draft of two sentences: the world totals to 10 January 2021 and the week of 4 to 10 January, which in `ecdc.csv` has the highest weekly cases and deaths of any week (5,270,998 and 93,681; the previous highest, 14 to 20 December, had 4,579,414 and 81,186).

Six voted for the paragraph and four against (the archivist, accessibility, the fact-checker and the devil's advocate). The devil's advocate conceded that a plain sentence with its date inside "mostly neutralises" the screenshot risk. All ten rejected big-number tiles, which read as a live dashboard and get cropped away from their dates. The paragraph sits straight under the archive line, as a second `standfirst`:

> By 10 January 2021, 214 countries and territories had reported 89.8 million cases and 1.94 million deaths to ECDC, figures that undercount the real toll. The week of 4 to 10 January 2021, the last in the data, had the highest weekly counts reported up to then: 5.27 million cases and 93,700 deaths.

The draft's wording changed where most of the advisers agreed:

- "Yet" became "up to then" (9 of 10): on a frozen page "yet" reads as now.
- The undercount sits next to the totals (8 of 10), so a quote or a cropped screenshot carries it. "How to read this page" says the same further down.
- "The last week" became "the week of 4 to 10 January 2021, the last in the data" (the UX writer: "last week" reads as last week from today), with the year in both sentences, so each one carries its own date.
- "Weekly counts reported" (the graphics editor and the epidemiologist): the week's high may partly be reports held over the Christmas and New Year holidays, so the sentence claims the most reported, not the most infections.
- 93,681 became 93,700, three significant figures like the other numbers (three advisers flagged mixed precision).

Left out: WHO's later total of about 7 million deaths and the excess-death estimates (only the hiring manager wanted a clause; the others said a second source turns the paragraph into commentary); a link to "How to read" (two advisers); a note that the table runs a week later (the archive line already says so). The 214 does not count the Diamond Princess, which `ecdc.csv` lists as a 215th place. The world totals were never on the page in 2020, which the archivist raised; they are sums of the same ECDC file the charts and maps were drawn from.

## 2026-10-06: the folders reorganised (covid19-0b)

Tommy asked for a complete reorganisation of the folders and files. The root held sixteen scripts beside the docs, and folders of three kinds side by side: what the page is built from, what the 2020 dashboard left behind, and things not about COVID-19. The layout now follows those kinds:

- `scripts/`: all sixteen scripts, `deploy.py` included, because `build_press_videos.py` imports from it and the October 2026 scripts import one another (`regions`, `plot_heat`, `build_world_map`, `plot_bubble`, `plot_series`). Keeping `deploy.py`, the one command run by hand, at the root was considered, but would have split those imports across two folders. Each script's `ROOT` is now `parents[1]`, still the repo root, and `ruff.toml` tells ruff's import sorting that `scripts/` holds the project's own modules.
- `data/`: gains Natural Earth's countries (from `map/data/countries_110m/`; `build_world_map.py` writes `worldmap.json` from them byte for byte as before) and the Wikimedia list and credits that `fetch_images.py` reads and writes.
- `2020/`: what the 2020-2021 dashboard left: its pipeline, regional maps, world-map animation, notes, and `archive/`, every image it uploaded. `archive/` keeps the name `upload.py` and `wrapper.sh` use. The scripts in `2020/` are kept as they last ran, so their paths still name the old root folders.
- `other/` gains `map/`'s obesity, alcohol, Texas and Bokeh experiments (`other/maps/`) and `deepti/`; `repos/` holds the four separate repositories.
- Unchanged: `site/`, whose paths are the page's URLs, `external_images/`, `trash/`, `tmp/` and the docs at the root.

Left alone: what git tracks. The untracked 2020 files (`upload.py`, `countries.txt`, the press pages' 2020 working copies, `tables/`, `notes/`, `map/`) moved without being added, and the nested repositories' lost `.git/HEAD` was not restored.
