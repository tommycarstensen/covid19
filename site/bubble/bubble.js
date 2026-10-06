// Interactive bubble charts. Each <figure class="bubble" data-region> gets a chart of how every country's COVID-19 cases and deaths in two weeks compared with the two weeks before, for every week from early 2020 to 10 January 2021, with a play button and a slider like the map players' (anim/scrubber.js). Clicking a bubble traces that country's path. Data: bubble_data.js, written by build_bubbles.py. Without JavaScript the figure shows its <noscript> image, plot_bubble.py's chart of one week.
(function () {
  'use strict';

  // A change is shown only when the earlier two weeks had at least this many cases (or deaths): below it the change is mostly chance, as when Iceland's deaths went from 1 to 9 to 13.
  var MIN_EVENTS = 20;
  // Both axes, as log2 of the ratio to the earlier two weeks: from a quarter (-75%) to eight times (+700%). Values beyond are drawn at the edge.
  var LO = -2;
  var HI = 3;
  // The test colour is fully blue at half the earlier tests and fully red at double.
  var TEST_SPAN = 1;
  // The countries with most deaths to date get a label, up to this many, where it fits.
  var LABELS = 10;
  var STEP_MS = 700;

  var INK = '#0b0b0b';
  var BLUE = [42, 120, 214];
  var GRAY = [168, 166, 160];
  var RED = [227, 73, 72];

  var CSS = [
    '.bubble{margin:16px 0 24px;max-width:860px;font:13px/1.35 system-ui,-apple-system,"Segoe UI",sans-serif;color:#333}',
    '.bubble .plot{position:relative}',
    '.bubble .plot svg{display:block;width:100%;overflow:visible;font:11px system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-user-select:none;user-select:none}',
    '.bubble .grid{stroke:#e9e8e3;stroke-width:1}',
    '.bubble .zero{stroke:#a9a7a0;stroke-width:1}',
    '.bubble .band{fill:#f6f5f2}',
    '.bubble .tick,.bubble .quad,.bubble .bandlabel{fill:#76746e}',
    '.bubble .quad{font-size:11px;fill:#a3a19a}',
    '.bubble .title{fill:#52514e;font-size:12px}',
    '.bubble .head{fill:#0b0b0b;font-size:13px;font-weight:600}',
    '.bubble g.b{cursor:pointer;transition:transform .45s ease}',
    '.bubble g.b circle.dot{stroke:#fff;stroke-width:1.5;transition:r .45s ease,fill .45s ease}',
    '.bubble g.b.hollow circle.dot{fill:#fff;stroke:#76746e;stroke-width:1.5}',
    '.bubble g.b.on circle.dot{stroke:' + INK + ';stroke-width:2}',
    '.bubble g.lab{pointer-events:none;transition:transform .45s ease}',
    '.bubble g.lab text{fill:' + INK + ';stroke:#fff;stroke-width:3px;stroke-linejoin:round;paint-order:stroke}',
    '.bubble .trail{stroke:#52514e;stroke-width:1.5;stroke-linecap:round}',
    '.bubble .trail-dot{fill:#52514e}',
    '.bubble.still g.b,.bubble.still g.lab,.bubble.still g.b circle.dot{transition:none}',
    '.bubble .tip{position:absolute;z-index:2;pointer-events:none;background:#fff;border:1px solid #d6d5cf;border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,.12);padding:6px 9px;font-size:12px;line-height:1.4;white-space:nowrap}',
    '.bubble .tip b{font-size:13px}',
    '.bubble .tip span{color:#52514e}',
    '.bubble .legend{display:flex;flex-wrap:wrap;align-items:center;gap:6px 20px;margin:6px 0 0;font-size:12px;color:#52514e}',
    '.bubble .legend > span{display:inline-flex;align-items:center;gap:6px}',
    '.bubble .legend .key{display:inline-flex;align-items:center;gap:3px;margin-left:4px}',
    '.bubble .ramp{display:inline-block;width:72px;height:10px;border-radius:5px}',
    '.bubble .legend svg{width:auto;display:inline-block;overflow:visible}',
    '.bubble .note{margin:6px 0 0;font-size:12px;color:#52514e}',
    '.bubble figcaption{margin:6px 0 0;font-size:12px;color:#52514e;max-width:60em}',
    '.bubble details{margin:8px 0 0;font-size:12px}',
    '.bubble summary{cursor:pointer;color:#333}',
    '.bubble .tablewrap{overflow-x:auto}',
    '.bubble table{width:auto;border-spacing:0;margin:6px 0 0;font-variant-numeric:tabular-nums}',
    '.bubble th,.bubble td{padding:3px 10px;text-align:right;white-space:nowrap}',
    '.bubble th:first-child,.bubble td:first-child{text-align:left}',
    '.bubble tr:nth-child(even){background:#f6f5f2}',
    // The same player as anim/scrubber.js, so the two look alike whichever loads first.
    '.scrub{display:flex;align-items:center;gap:8px;max-width:100%;margin:4px 0 0;font:13px/1.2 system-ui,-apple-system,"Segoe UI",sans-serif;color:#333}',
    '.scrub button{flex:none;display:grid;place-items:center;width:32px;height:32px;padding:0;border:1px solid #bbb;border-radius:50%;background:#fff;color:#333;cursor:pointer}',
    '.scrub button:hover{border-color:#666}',
    '.scrub button:focus-visible,.scrub input:focus-visible{outline:2px solid #1a73e8;outline-offset:2px}',
    '.scrub svg{width:12px;height:12px;fill:currentColor}',
    '.scrub input{flex:1;min-width:60px;margin:0;accent-color:#555;cursor:pointer}',
    '.scrub output{flex:none;min-width:6.2em;text-align:right;font-variant-numeric:tabular-nums}'
  ].join('\n');

  var PLAY = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1v10l9-5z"/></svg>';
  var PAUSE = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1h3v10H2zM7 1h3v10H7z"/></svg>';
  var SVGNS = 'http://www.w3.org/2000/svg';
  var MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var measureCtx = document.createElement('canvas').getContext('2d');

  function el(name, attrs, parent) {
    var node = document.createElementNS(SVGNS, name);
    for (var key in attrs) node.setAttribute(key, attrs[key]);
    if (parent) parent.appendChild(node);
    return node;
  }

  function html(name, className, parent) {
    var node = document.createElement(name);
    if (className) node.className = className;
    if (parent) parent.appendChild(node);
    return node;
  }

  function textWidth(text) {
    measureCtx.font = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
    return measureCtx.measureText(text).width;
  }

  function longDate(iso) {
    var parts = iso.split('-');
    return parseInt(parts[2], 10) + ' ' + MONTHS[parseInt(parts[1], 10) - 1] + ' ' + parts[0];
  }

  function shortDate(iso) {
    var parts = iso.split('-');
    return parseInt(parts[2], 10) + ' ' + MONTHS[parseInt(parts[1], 10) - 1].slice(0, 3);
  }

  function number(n) {
    return n.toLocaleString('en-GB');
  }

  function percent(log2) {
    var p = Math.round((Math.pow(2, log2) - 1) * 100);
    return (p > 0 ? '+' : p < 0 ? '−' : '') + number(Math.abs(p)) + '%';
  }

  // Sum of weeks t-1 and t, or null when either is missing.
  function fortnight(series, t) {
    if (t < 1) return null;
    var a = series[t];
    var b = series[t - 1];
    return a === null || b === null ? null : a + b;
  }

  function ratio(now, before) {
    if (now === null || before === null || now < 0 || before < MIN_EVENTS) return null;
    return Math.log2(Math.max(now, 0.5) / before);
  }

  function mix(a, b, k) {
    return 'rgb(' + [0, 1, 2].map(function (i) { return Math.round(a[i] + (b[i] - a[i]) * k); }).join(',') + ')';
  }

  function testColour(z) {
    var u = Math.max(-1, Math.min(1, z / TEST_SPAN));
    return mix(GRAY, u < 0 ? BLUE : RED, Math.abs(u));
  }

  // Every country's numbers for every week, and the weeks with anything to plot.
  function prepare(data, region) {
    var codes = data.regions[region] || [];
    var weeks = data.weeks;
    var countries = codes.map(function (code) {
      var c = data.countries[code];
      var total = 0;
      var cumulative = c.deaths.map(function (d) { total += d || 0; return Math.max(0, total); });
      var frames = weeks.map(function (week, t) {
        var cn = fortnight(c.cases, t);
        var cp = fortnight(c.cases, t - 2);
        var dn = fortnight(c.deaths, t);
        var dp = fortnight(c.deaths, t - 2);
        var tn = fortnight(c.tests, t);
        var tp = fortnight(c.tests, t - 2);
        return {
          cn: cn, cp: cp, dn: dn, dp: dp,
          x: ratio(cn, cp),
          y: ratio(dn, dp),
          z: tn > 0 && tp > 0 ? Math.log2(tn / tp) : null,
          deaths: cumulative[t]
        };
      });
      return { code: code, name: c.name, frames: frames };
    });
    var first = weeks.length - 1;
    countries.forEach(function (c) {
      for (var t = 0; t < first; t++) {
        if (c.frames[t].x !== null) { first = t; break; }
      }
    });
    var last = weeks.length - 1;
    var maxDeaths = Math.max.apply(null, countries.map(function (c) { return c.frames[last].deaths; }).concat([1]));
    // Biggest drawn first, so small bubbles stay on top; fixed for all weeks so the bubbles never re-stack mid-animation.
    countries.sort(function (a, b) { return b.frames[last].deaths - a.frames[last].deaths; });
    return { weeks: weeks, countries: countries, first: first, last: last, maxDeaths: maxDeaths };
  }

  function setup(figure, data) {
    var region = figure.getAttribute('data-region');
    var model = prepare(data, region);
    if (!model.countries.length) return;
    var noscript = figure.querySelector('noscript');
    if (noscript) figure.removeChild(noscript);

    var frame = model.last;
    var playing = false;
    var visible = true;
    var timer = null;
    var selected = null;
    var layout = null;

    var plot = html('div', 'plot', figure);
    var svg = el('svg', { role: 'img' }, plot);
    var tip = html('div', 'tip', plot);
    tip.hidden = true;

    var bar = html('div', 'scrub', figure);
    var button = html('button', '', bar);
    button.type = 'button';
    var range = html('input', '', bar);
    range.type = 'range';
    range.min = String(model.first);
    range.max = String(model.last);
    range.step = '1';
    range.setAttribute('aria-label', 'Week');
    var output = html('output', '', bar);

    var legend = html('div', 'legend', figure);
    var note = html('p', 'note', figure);
    var caption = html('figcaption', '', figure);
    caption.textContent = 'Each bubble is a country: its cases (across) and deaths (up) in two weeks against the two weeks before, on log scales, so halving and doubling are the same distance from 0. Area: deaths to date. Colour: the change in tests over the same weeks; hollow where none were reported. A change needs at least ' + MIN_EVENTS + ' cases or deaths in the earlier two weeks; countries with fewer deaths sit in the strip below. Click a bubble to trace its path. ECDC weekly data to 10 January 2021; tests from Our World in Data.';
    var details = html('details', '', figure);
    var summary = html('summary', '', details);
    var tableWrap = html('div', 'tablewrap', details);

    function radius(deaths) {
      return Math.max(3.5, layout.rMax * Math.sqrt(deaths / model.maxDeaths));
    }

    function sx(v) { return layout.left + (Math.max(LO, Math.min(HI, v)) - LO) / (HI - LO) * layout.w; }
    function sy(v) { return layout.top + (HI - Math.max(LO, Math.min(HI, v))) / (HI - LO) * layout.h; }

    // Where a country sits in week t, or null when it is not plotted.
    function position(c, t) {
      var f = c.frames[t];
      if (f.x === null) return null;
      return { x: sx(f.x), y: f.y === null ? layout.bandY : sy(f.y), f: f };
    }

    function build() {
      var width = Math.max(280, plot.clientWidth);
      var narrow = width < 560;
      var left = narrow ? 58 : 62;
      var right = 14;
      var top = 30;
      var h = Math.round(Math.max(220, Math.min(440, width * 0.52)));
      var band = 30;
      var w = width - left - right;
      layout = {
        left: left, top: top, w: w, h: h, narrow: narrow,
        bandY: top + h + 8 + band / 2,
        rMax: Math.max(14, Math.min(34, width / 24))
      };
      var height = top + h + 8 + band + 40;
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
      svg.setAttribute('width', width);
      svg.setAttribute('height', height);

      layout.head = el('text', { 'class': 'head', x: 0, y: 14 }, svg);

      var grid = el('g', {}, svg);
      el('rect', { 'class': 'band', x: left, y: top + h + 8, width: w, height: band, rx: 4 }, grid);
      var bandLabel = el('text', { 'class': 'bandlabel', x: left - 8, y: layout.bandY - 3, 'text-anchor': 'end' }, grid);
      bandLabel.textContent = 'Too few';
      var bandLabel2 = el('text', { 'class': 'bandlabel', x: left - 8, y: layout.bandY + 9, 'text-anchor': 'end' }, grid);
      bandLabel2.textContent = 'deaths';
      for (var v = LO; v <= HI; v++) {
        var x = sx(v);
        var y = sy(v);
        el('line', { 'class': v === 0 ? 'zero' : 'grid', x1: x, x2: x, y1: top, y2: top + h + 8 + band }, grid);
        el('line', { 'class': v === 0 ? 'zero' : 'grid', x1: left, x2: left + w, y1: y, y2: y }, grid);
        var tx = el('text', { 'class': 'tick', x: x, y: top + h + 8 + band + 15, 'text-anchor': 'middle' }, grid);
        tx.textContent = percent(v);
        var ty = el('text', { 'class': 'tick', x: left - 8, y: y + 4, 'text-anchor': 'end' }, grid);
        ty.textContent = percent(v);
      }
      var xTitle = el('text', { 'class': 'title', x: left + w / 2, y: height - 4, 'text-anchor': 'middle' }, grid);
      xTitle.textContent = 'Change in cases';
      var yTitle = el('text', { 'class': 'title', transform: 'translate(10,' + (top + h / 2) + ') rotate(-90)', 'text-anchor': 'middle' }, grid);
      yTitle.textContent = 'Change in deaths';

      var pad = 6;
      var quads = narrow
        ? [['Only deaths rising', left + pad, top + 13, 'start'], ['Both rising', left + w - pad, top + 13, 'end'],
          ['Both falling', left + pad, top + h - pad, 'start'], ['Only cases rising', left + w - pad, top + h - pad, 'end']]
        : [['Deaths rising, cases falling', left + pad, top + 13, 'start'], ['Cases and deaths rising', left + w - pad, top + 13, 'end'],
          ['Cases and deaths falling', left + pad, top + h - pad, 'start'], ['Cases rising, deaths falling', left + w - pad, top + h - pad, 'end']];
      quads.forEach(function (q) {
        var t = el('text', { 'class': 'quad', x: q[1], y: q[2], 'text-anchor': q[3] }, grid);
        t.textContent = q[0];
      });

      layout.trail = el('g', {}, svg);
      var dots = el('g', {}, svg);
      // Labels sit in a layer of their own, above every bubble, and move with them.
      var labels = el('g', {}, svg);
      layout.groups = {};
      layout.labels = {};
      model.countries.forEach(function (c) {
        var g = el('g', { 'class': 'b' }, dots);
        el('circle', { 'class': 'hit', r: 12, fill: 'transparent' }, g);
        el('circle', { 'class': 'dot', r: 4 }, g);
        g.style.display = 'none';
        var lab = el('g', { 'class': 'lab' }, labels);
        el('text', {}, lab).textContent = c.name;
        lab.style.display = 'none';
        layout.labels[c.code] = lab;
        g.addEventListener('pointerenter', function () { showTip(c); });
        g.addEventListener('pointermove', function () { showTip(c); });
        g.addEventListener('pointerleave', hideTip);
        g.addEventListener('click', function (e) {
          e.stopPropagation();
          select(selected === c ? null : c);
          showTip(c);
        });
        layout.groups[c.code] = g;
      });
      buildLegend();
    }

    function buildLegend() {
      while (legend.firstChild) legend.removeChild(legend.firstChild);
      var tests = html('span', '', legend);
      tests.appendChild(document.createTextNode('Tests halved'));
      var ramp = html('span', 'ramp', tests);
      ramp.style.background = 'linear-gradient(to right,' + testColour(-1) + ',' + testColour(0) + ',' + testColour(1) + ')';
      tests.appendChild(document.createTextNode('doubled'));

      var none = html('span', '', legend);
      var noneSvg = el('svg', { width: 12, height: 12, viewBox: '-6 -6 12 12' }, none);
      el('circle', { r: 4.5, fill: '#fff', stroke: '#76746e', 'stroke-width': 1.5 }, noneSvg);
      none.appendChild(document.createTextNode('no test data'));

      // Round numbers no bigger than the biggest bubble, and big enough not to be drawn at the smallest size.
      var top = Math.pow(10, Math.floor(Math.log10(model.maxDeaths)));
      var keys = [top / 10, top, 5 * top].filter(function (k) { return k >= 1 && k <= model.maxDeaths && radius(k) >= 4.5; });
      var sizes = html('span', '', legend);
      sizes.appendChild(document.createTextNode('Deaths to date'));
      keys.forEach(function (k) {
        var r = radius(k);
        var key = html('span', 'key', sizes);
        var s = el('svg', { width: 2 * r + 2, height: 2 * r + 2, viewBox: (-r - 1) + ' ' + (-r - 1) + ' ' + (2 * r + 2) + ' ' + (2 * r + 2) }, key);
        el('circle', { r: r, fill: 'none', stroke: '#76746e', 'stroke-width': 1 }, s);
        key.appendChild(document.createTextNode(number(k)));
      });
    }

    function overlaps(a, b) {
      return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
    }

    // Greedy label placement, the selected country first and then by deaths to date: the first spot around the bubble that overlaps no placed label and no other bubble, or failing that no placed label. A label that fits nowhere is left to the tooltip and the table.
    function placeLabels(shown) {
      var placed = [];
      var discs = shown.map(function (item) {
        var r = radius(item.p.f.deaths);
        return { c: item.c, x: item.p.x - r, y: item.p.y - r, w: 2 * r, h: 2 * r };
      });
      var order = shown.slice().sort(function (a, b) {
        if (a.c === selected) return -1;
        if (b.c === selected) return 1;
        return b.p.f.deaths - a.p.f.deaths;
      });
      var budget = LABELS;
      var bottom = layout.bandY + 15;
      shown.forEach(function (item) { layout.labels[item.c.code].style.display = 'none'; });
      order.forEach(function (item) {
        if (budget <= 0 && item.c !== selected) return;
        var r = radius(item.p.f.deaths);
        var d = r * 0.71;
        var tw = textWidth(item.c.name);
        // [dx, dy] of the text anchor from the centre, and the anchor.
        var spots = [
          [r + 3, 4, 'start'], [-r - 3, 4, 'end'],
          [0, -r - 4, 'middle'], [0, r + 12, 'middle'],
          [d + 2, -d - 2, 'start'], [d + 2, d + 10, 'start'],
          [-d - 2, -d - 2, 'end'], [-d - 2, d + 10, 'end']
        ];
        var boxes = spots.map(function (s) {
          var x = item.p.x + s[0] - (s[2] === 'end' ? tw : s[2] === 'middle' ? tw / 2 : 0);
          return { x: x, y: item.p.y + s[1] - 10, w: tw, h: 13, s: s };
        }).filter(function (box) {
          return box.x >= layout.left - 2 && box.x + box.w <= layout.left + layout.w + 12 && box.y >= layout.top - 2 && box.y + box.h <= bottom;
        });
        var free = function (box) { return !placed.some(function (b) { return overlaps(box, b); }); };
        var clear = function (box) { return !discs.some(function (b) { return b.c !== item.c && overlaps(box, b); }); };
        var best = boxes.filter(function (box) { return free(box) && clear(box); })[0] || boxes.filter(free)[0];
        if (!best) return;
        placed.push(best);
        var lab = layout.labels[item.c.code];
        var text = lab.firstChild;
        text.setAttribute('x', best.s[0]);
        text.setAttribute('y', best.s[1]);
        text.setAttribute('text-anchor', best.s[2]);
        lab.style.display = '';
        budget--;
      });
    }

    // The selected country's path up to the week shown, one segment per week, fading with age. Weeks it sat in the strip or was not plotted break the path, since they have no place on the axes.
    function drawTrail() {
      var layer = layout.trail;
      while (layer.firstChild) layer.removeChild(layer.firstChild);
      if (!selected) return;
      var prev = null;
      for (var t = model.first; t <= frame; t++) {
        var p = position(selected, t);
        if (!p || p.f.y === null) { prev = null; continue; }
        var opacity = Math.max(0.12, Math.pow(0.85, frame - t)).toFixed(2);
        if (prev) el('line', { 'class': 'trail', x1: prev.x, y1: prev.y, x2: p.x, y2: p.y, 'stroke-opacity': opacity }, layer);
        if (t < frame) el('circle', { 'class': 'trail-dot', cx: p.x, cy: p.y, r: 1.8, 'fill-opacity': opacity }, layer);
        prev = p;
      }
    }

    function select(c) {
      if (selected) {
        // Back to its place in the stack, behind every smaller bubble.
        var g0 = layout.groups[selected.code];
        g0.classList.remove('on');
        var next = model.countries[model.countries.indexOf(selected) + 1];
        g0.parentNode.insertBefore(g0, next ? layout.groups[next.code] : null);
      }
      selected = c;
      if (c) {
        var g = layout.groups[c.code];
        g.classList.add('on');
        // Bring it to the front.
        g.parentNode.appendChild(g);
      }
      show(frame, false);
    }

    function show(t, animate) {
      frame = t;
      figure.classList.toggle('still', !animate || reduceMotion);
      var week = model.weeks[t];
      var from = model.weeks[t - 1] ? shortDate(addDays(model.weeks[t], -13)) : '';
      layout.head.textContent = layout.narrow ? 'Two weeks to ' + shortDate(week) + ' ' + week.slice(0, 4) + ' vs the two before' : 'Two weeks to ' + longDate(week) + ', against the two weeks before';
      svg.setAttribute('aria-label', 'Bubble chart, ' + region + ': change in COVID-19 cases and deaths in the two weeks ' + from + ' to ' + longDate(week) + ' against the two weeks before');
      range.value = String(t);
      output.textContent = week;
      range.setAttribute('aria-valuetext', 'Two weeks to ' + longDate(week));
      summary.textContent = 'Table: the two weeks to ' + longDate(week);

      var shown = [];
      var hidden = [];
      model.countries.forEach(function (c) {
        var g = layout.groups[c.code];
        var lab = layout.labels[c.code];
        var p = position(c, t);
        if (!p) {
          g.style.display = 'none';
          lab.style.display = 'none';
          hidden.push(c.name);
          return;
        }
        var wasHidden = g.style.display === 'none';
        if (wasHidden) {
          // Appear in place rather than fly in from where the country last was.
          g.style.transition = lab.style.transition = 'none';
          g.style.display = '';
        }
        g.style.transform = lab.style.transform = 'translate(' + p.x.toFixed(1) + 'px,' + p.y.toFixed(1) + 'px)';
        if (wasHidden) {
          void g.getBoundingClientRect();
          g.style.transition = lab.style.transition = '';
        }
        var dot = g.querySelector('circle.dot');
        dot.setAttribute('r', radius(p.f.deaths).toFixed(1));
        g.querySelector('circle.hit').setAttribute('r', Math.max(12, radius(p.f.deaths)).toFixed(1));
        g.classList.toggle('hollow', p.f.z === null);
        if (p.f.z !== null) dot.style.fill = testColour(p.f.z);
        else dot.style.fill = '';
        shown.push({ c: c, p: p });
      });
      placeLabels(shown);
      drawTrail();
      hidden.sort();
      note.textContent = hidden.length ? 'Not shown, with fewer than ' + MIN_EVENTS + ' cases in the earlier two weeks or a week unreported: ' + hidden.join(', ') + '.' : '';
      if (details.open) fillTable();
      button.innerHTML = playing ? PAUSE : PLAY;
      button.setAttribute('aria-label', playing ? 'Pause' : 'Play');
      if (!tip.hidden && tip.country) showTip(tip.country);
    }

    function addDays(iso, days) {
      var d = new Date(iso + 'T00:00:00Z');
      d.setUTCDate(d.getUTCDate() + days);
      return d.toISOString().slice(0, 10);
    }

    function tipLines(c) {
      var f = c.frames[frame];
      var lines = [];
      function line(label, now, before, log2) {
        if (now === null || before === null) return [label, 'not reported'];
        var text = number(now) + ' against ' + number(before);
        if (log2 !== null) return [label, percent(log2) + '  (' + text + ')'];
        return [label, text + (before < MIN_EVENTS ? ', too few to compare' : '')];
      }
      lines.push(line('Cases', f.cn, f.cp, f.x));
      lines.push(line('Deaths', f.dn, f.dp, f.y));
      lines.push(['Tests', f.z === null ? 'no data' : percent(f.z)]);
      lines.push(['Deaths to date', number(f.deaths)]);
      return lines;
    }

    function showTip(c) {
      tip.country = c;
      while (tip.firstChild) tip.removeChild(tip.firstChild);
      var name = html('b', '', tip);
      name.textContent = c.name;
      tipLines(c).forEach(function (l) {
        html('br', '', tip);
        var label = html('span', '', tip);
        label.textContent = l[0] + ': ';
        tip.appendChild(document.createTextNode(l[1]));
      });
      tip.hidden = false;
      var p = position(c, frame);
      if (!p) { tip.hidden = true; return; }
      var scale = plot.clientWidth / parseFloat(svg.getAttribute('width'));
      var x = p.x * scale;
      var y = p.y * scale;
      var tw = tip.offsetWidth;
      var left = x + 14 + tw > plot.clientWidth ? x - 14 - tw : x + 14;
      tip.style.left = Math.max(0, left) + 'px';
      tip.style.top = Math.max(0, y - tip.offsetHeight / 2) + 'px';
    }

    function hideTip() {
      tip.hidden = true;
      tip.country = null;
    }

    function fillTable() {
      while (tableWrap.firstChild) tableWrap.removeChild(tableWrap.firstChild);
      var table = html('table', '', tableWrap);
      var head = html('tr', '', table);
      ['Country', 'Cases', 'Change', 'Deaths', 'Change', 'Tests, change', 'Deaths to date'].forEach(function (h) {
        html('th', '', head).textContent = h;
      });
      model.countries.slice().sort(function (a, b) { return b.frames[frame].deaths - a.frames[frame].deaths; }).forEach(function (c) {
        var f = c.frames[frame];
        var row = html('tr', '', table);
        [
          c.name,
          f.cn === null ? '–' : number(f.cn),
          f.x === null ? '–' : percent(f.x),
          f.dn === null ? '–' : number(f.dn),
          f.y === null ? '–' : percent(f.y),
          f.z === null ? '–' : percent(f.z),
          number(f.deaths)
        ].forEach(function (v) { html('td', '', row).textContent = v; });
      });
    }

    function schedule() {
      clearTimeout(timer);
      timer = null;
      if (!playing || !visible) return;
      timer = setTimeout(function () {
        if (frame >= model.last) {
          setPlaying(false);
          return;
        }
        show(frame + 1, true);
        schedule();
      }, STEP_MS);
    }

    function setPlaying(play) {
      playing = play;
      if (play && frame >= model.last) show(model.first, false);
      else show(frame, false);
      schedule();
    }

    button.addEventListener('click', function () { setPlaying(!playing); });
    range.addEventListener('input', function () {
      if (playing) {
        playing = false;
        schedule();
      }
      show(parseInt(range.value, 10), true);
    });
    svg.addEventListener('click', function () {
      if (selected) select(null);
      hideTip();
    });
    details.addEventListener('toggle', function () { if (details.open) fillTable(); });

    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (entries) {
        visible = entries[entries.length - 1].isIntersecting;
        schedule();
      }).observe(svg);
    }
    var lastWidth = 0;
    function relayout() {
      if (plot.clientWidth === lastWidth) return;
      lastWidth = plot.clientWidth;
      var keep = selected;
      build();
      selected = null;
      if (keep) select(keep);
      else show(frame, false);
    }
    if ('ResizeObserver' in window) new ResizeObserver(relayout).observe(plot);
    else window.addEventListener('resize', relayout);
    relayout();
  }

  function init() {
    var data = window.BUBBLE_DATA;
    if (!data) return;
    var style = document.createElement('style');
    style.textContent = CSS;
    document.head.appendChild(style);
    var figures = document.querySelectorAll('figure.bubble[data-region]');
    for (var i = 0; i < figures.length; i++) setup(figures[i], data);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
