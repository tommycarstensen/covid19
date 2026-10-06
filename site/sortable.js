// Sort the country table (id myTable2) by column n: numbers as numbers (thousands separators ignored), text alphabetically.
// The sorted column's header gets aria-sort, which the page's CSS shows as an arrow.
// A click sorts ascending; a click on a column that is already in ascending order sorts it descending. Empty cells and "\u2013" (no data) stay last either way.
// Until October 2026 every cell went through Number(), which made the Country and Continent columns NaN, so clicking them did nothing.
function sortTable(n) {
  var table = document.getElementById("myTable2");
  var rows = Array.prototype.slice.call(table.rows, 1);
  if (rows.length === 0) return;

  function key(row) {
    var text = row.cells[n].textContent.trim();
    if (text === "" || text === "\u2013") return null;
    var number = Number(text.replace(/,/g, ""));
    return text !== "" && !isNaN(number) ? number : text.toLowerCase();
  }

  var keyed = rows.map(function (row, i) { return {row: row, key: key(row), i: i}; });
  var missing = keyed.filter(function (k) { return k.key === null; });
  keyed = keyed.filter(function (k) { return k.key !== null; });
  keyed.sort(function (a, b) {
    if (typeof a.key === "number" && typeof b.key === "number") return a.key - b.key || a.i - b.i;
    if (typeof a.key === "number") return -1;
    if (typeof b.key === "number") return 1;
    return a.key.localeCompare(b.key) || a.i - b.i;
  });

  var alreadyAscending = keyed.concat(missing).every(function (k, i) { return k.i === i; });
  if (alreadyAscending) keyed.reverse();
  keyed = keyed.concat(missing);

  var headers = table.rows[0].cells;
  for (var h = 0; h < headers.length; h++) headers[h].removeAttribute("aria-sort");
  headers[n].setAttribute("aria-sort", alreadyAscending ? "descending" : "ascending");

  var parent = rows[0].parentNode;
  keyed.forEach(function (k) { parent.appendChild(k.row); });
}
