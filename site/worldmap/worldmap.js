// Six interactive world maps: ECDC's weekly COVID-19 cases and deaths per million people and Our World in Data's tests per thousand, cumulative and per week.
// Fills the <div class="wm-row" data-period="cumulative|weekly"> rows of every <div class="worldmap" data-src="..."> from the JSON that build_world_map.py writes, with one week slider for all six. data-autoplay="false" stops it playing once when first scrolled into view.
(function () {
  'use strict';

  var SVG_NS = 'http://www.w3.org/2000/svg';
  // The nine ColorBrewer colours of each matplotlib colour map, which matplotlib interpolates linearly.
  var CMAPS = {
    OrRd: ['#fff7ec', '#fee8c8', '#fdd49e', '#fdba83', '#fc8c59', '#ef6447', '#d62f1e', '#b20000', '#7f0000'],
    PuRd: ['#f7f4f9', '#e7e1ef', '#d4b9da', '#c993c7', '#df64af', '#e72989', '#cd1256', '#970042', '#67001f'],
    Greens: ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a0d99b', '#73c476', '#40aa5d', '#228a44', '#006c2c', '#00441b'],
    YlGnBu: ['#ffffd9', '#edf8b1', '#c6e9b4', '#7ecdbb', '#40b5c4', '#1d90c0', '#225da8', '#243392', '#081d58'],
    PuBuGn: ['#fff7fb', '#ece2f0', '#d0d1e6', '#a5bddb', '#66a9cf', '#3590bf', '#028189', '#016b58', '#014636'],
    Blues: ['#f7fbff', '#deebf7', '#c6dbef', '#9dcae1', '#6aaed6', '#4191c6', '#2070b4', '#08509b', '#08306b']
  };
  // Each map's colour map and log10 range, from plot_choropleth.py's 2020 GIFs; a value outside the range takes the end colour.
  var SCALES = {
    cumulative: {
      cases: { cmap: 'OrRd', min: -2, max: 4 },
      deaths: { cmap: 'PuRd', min: -3, max: 3 },
      tests: { cmap: 'Greens', min: -3, max: 3 }
    },
    weekly: {
      cases: { cmap: 'YlGnBu', min: -2, max: 4 },
      deaths: { cmap: 'PuBuGn', min: -3, max: 3 },
      tests: { cmap: 'Blues', min: -3, max: 1 }
    }
  };
  var MEASURES = ['cases', 'deaths', 'tests'];
  var PERIODS = ['weekly', 'cumulative'];
  var TITLES = { cases: 'Cases per million people', deaths: 'Deaths per million people', tests: 'Tests per thousand people' };
  var UNITS = { cases: 'per million', deaths: 'per million', tests: 'per thousand' };
  var ZERO = '#ffffff';
  var FRAME_MS = 350;
  var MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  var count = new Intl.NumberFormat('en-GB');
  var threeDigits = new Intl.NumberFormat('en-GB', { maximumSignificantDigits: 3 });
  var millions = new Intl.NumberFormat('en-GB', { minimumSignificantDigits: 3, maximumSignificantDigits: 3 });
  var instances = 0;

  Object.keys(CMAPS).forEach(function (name) {
    CMAPS[name] = CMAPS[name].map(function (hex) {
      return [1, 3, 5].map(function (i) { return parseInt(hex.slice(i, i + 2), 16); });
    });
  });

  function colour(scale, value) {
    var stops = CMAPS[scale.cmap];
    var t = (Math.log10(value) - scale.min) / (scale.max - scale.min);
    t = Math.min(1, Math.max(0, t)) * (stops.length - 1);
    var i = Math.min(stops.length - 2, Math.floor(t));
    var f = t - i;
    return 'rgb(' + [0, 1, 2].map(function (j) { return Math.round(stops[i][j] + (stops[i + 1][j] - stops[i][j]) * f); }).join(',') + ')';
  }

  function gradient(scale) {
    return 'linear-gradient(to right, ' + CMAPS[scale.cmap].map(function (c) { return 'rgb(' + c.join(',') + ')'; }).join(', ') + ')';
  }

  function tickText(power) {
    return power < 0 ? Math.pow(10, power).toFixed(-power) : count.format(Math.pow(10, power));
  }

  function el(tag, className, parent, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }

  function svgEl(tag, attrs, parent) {
    var node = document.createElementNS(SVG_NS, tag);
    Object.keys(attrs).forEach(function (name) { node.setAttribute(name, attrs[name]); });
    parent.appendChild(node);
    return node;
  }

  function day(iso, offset) {
    var parts = iso.split('-').map(Number);
    return new Date(Date.UTC(parts[0], parts[1] - 1, parts[2] + offset));
  }

  function dayText(date, withYear) {
    return date.getUTCDate() + ' ' + MONTHS[date.getUTCMonth()] + (withYear ? ' ' + date.getUTCFullYear() : '');
  }

  function weekText(start) {
    var first = day(start, 0);
    var last = day(start, 6);
    if (first.getUTCFullYear() !== last.getUTCFullYear()) return dayText(first, true) + ' – ' + dayText(last, true);
    if (first.getUTCMonth() !== last.getUTCMonth()) return dayText(first, false) + ' – ' + dayText(last, true);
    return first.getUTCDate() + '–' + dayText(last, true);
  }

  function population(pop, short) {
    return pop >= 1e6 ? millions.format(pop / 1e6) + (short ? 'm' : ' million') : count.format(pop);
  }

  // Running totals; null until the country's first weekly row, and a missing week adds nothing.
  function cumulate(values) {
    var sum = null;
    return values.map(function (value) {
      if (value !== null) sum = (sum || 0) + value;
      return sum;
    });
  }

  function build(root, data) {
    var id = 'worldmap' + (++instances);
    var last = data.weeks.length - 1;
    var state = { week: last, timer: null, sortKey: 'weekly cases', sortUp: false };
    var series = {};
    Object.keys(data.countries).forEach(function (code) {
      var country = data.countries[code];
      series[code] = {
        weekly: { cases: country.cases, deaths: country.deaths, tests: country.tests || null },
        cumulative: { cases: cumulate(country.cases), deaths: cumulate(country.deaths), tests: country.testsTotal || null }
      };
    });
    var onMap = {};
    data.shapes.forEach(function (shape) { onMap[shape.code] = true; });
    var offMap = Object.keys(data.countries).filter(function (code) { return !onMap[code]; }).length;

    // The count for cases and deaths; tests come per thousand people already.
    function value(code, period, measure) {
      var values = series[code] && series[code][period][measure];
      return values ? values[state.week] : null;
    }

    function rate(code, period, measure) {
      var v = value(code, period, measure);
      if (v === null || measure === 'tests') return v;
      return v * 1e6 / data.countries[code].pop;
    }

    function periodText(period) {
      var start = data.weekStarts[state.week];
      return period === 'weekly' ? 'in the week ' + weekText(start) : 'in total up to ' + dayText(day(start, 6), true);
    }

    // The controls go under the first heading, so that they do not run on from the player above the section.
    var controls = el('div', 'wm-controls');
    var firstRow = root.querySelector('.wm-row[data-period]');
    root.insertBefore(controls, firstRow || root.firstChild);
    var play = el('button', 'wm-play', controls, '▶ Play');
    play.type = 'button';
    var slider = el('input', 'wm-slider', controls);
    slider.type = 'range';
    slider.min = '0';
    slider.max = String(last);
    slider.step = '1';
    slider.setAttribute('aria-label', 'Week');
    var weekLabel = el('span', 'wm-week', controls);

    var maps = [];
    if (!root.querySelector('.wm-row[data-period]')) {
      ['cumulative', 'weekly'].forEach(function (period) { el('div', 'wm-row', root).dataset.period = period; });
    }
    root.querySelectorAll('.wm-row[data-period]').forEach(function (row) {
      MEASURES.forEach(function (measure) { maps.push(makeMap(row, row.dataset.period, measure)); });
    });
    var tip = el('div', 'wm-tip');
    tip.hidden = true;

    var key = el('div', 'wm-key', root);
    el('span', 'wm-swatch', key).style.background = ZERO;
    el('span', '', key, '0');
    el('span', 'wm-swatch wm-nodata', key);
    el('span', '', key, 'no data');
    el('p', 'wm-source', root, 'Data: ECDC, weekly cases and deaths reported in ISO weeks ' + data.weeks[0] + ' to ' + data.weeks[last] + ', per million people of 2019; Our World in Data, tests per thousand people. Countries without a report that week are hatched. ' + offMap + ' territories too small for the maps are in the table.');

    var details = el('details', 'wm-table', root);
    var summary = el('summary', '', details, 'Table of all ' + Object.keys(data.countries).length + ' countries and territories');
    summary.id = id + '-summary';
    var scroll = el('div', 'wm-table-scroll', details);
    var table = el('table', '', scroll);
    var thead = el('thead', '', table);
    var groupRow = el('tr', '', thead);
    var headRow = el('tr', '', thead);
    var groupCells = {};
    var sortButtons = [];
    el('th', '', groupRow);
    PERIODS.forEach(function (period) {
      groupCells[period] = el('th', 'wm-group', groupRow);
      groupCells[period].colSpan = MEASURES.length;
    });
    el('th', '', groupRow);
    sortHeader('name', 'Country or territory', '');
    PERIODS.forEach(function (period) {
      MEASURES.forEach(function (measure) { sortHeader(period + ' ' + measure, (measure === 'tests' ? 'Tests ' : measure === 'cases' ? 'Cases ' : 'Deaths ') + UNITS[measure], 'wm-num'); });
    });
    sortHeader('pop', 'Population', 'wm-num');
    var tbody = el('tbody', '', table);
    details.addEventListener('toggle', function () { if (details.open) renderTable(); });

    function sortHeader(sortKey, label, className) {
      var th = el('th', className, headRow);
      var button = el('button', 'wm-sort', th, label);
      button.type = 'button';
      button.addEventListener('click', function () {
        state.sortUp = state.sortKey === sortKey ? !state.sortUp : sortKey === 'name';
        state.sortKey = sortKey;
        renderTable();
      });
      sortButtons.push({ key: sortKey, th: th, button: button, label: label });
    }

    function makeMap(row, period, measure) {
      var n = maps.length;
      var scale = SCALES[period][measure];
      var cell = el('figure', 'wm-cell', row);
      var caption = el('figcaption', 'wm-caption', cell);
      el('span', 'wm-title', caption, TITLES[measure]);
      var when = el('span', 'wm-when', caption);
      var figure = el('div', 'wm-figure', cell);
      var svg = svgEl('svg', { viewBox: data.viewBox.join(' '), class: 'wm-map', role: 'img' }, figure);
      var hatch = id + '-hatch' + n;
      var pattern = svgEl('pattern', { id: hatch, width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, svgEl('defs', {}, svg));
      svgEl('rect', { width: 5, height: 5, fill: '#ffffff' }, pattern);
      svgEl('line', { x1: 0, y1: 0, x2: 0, y2: 5, stroke: '#a9a7a0', 'stroke-width': 1.6 }, pattern);
      var group = svgEl('g', {}, svg);
      var byCode = {};
      var paths = data.shapes.map(function (shape) {
        var path = svgEl('path', { d: shape.d }, group);
        path.dataset.code = shape.code;
        path.dataset.name = shape.name;
        byCode[shape.code] = (byCode[shape.code] || []).concat(path);
        return path;
      });
      var bar = el('div', 'wm-bar', cell);
      el('div', 'wm-ramp', bar).style.background = gradient(scale);
      var ticks = el('div', 'wm-ticks', bar);
      for (var power = scale.min; power <= scale.max; power++) {
        el('span', '', ticks, tickText(power)).style.left = (100 * (power - scale.min) / (scale.max - scale.min)) + '%';
      }
      var map = { period: period, measure: measure, scale: scale, when: when, figure: figure, svg: svg, group: group, paths: paths, byCode: byCode, hatch: hatch };
      svg.addEventListener('pointermove', function (event) {
        var path = event.target.closest ? event.target.closest('path') : null;
        if (path && path.dataset.code) showTip(map, path, event);
        else hideTip();
      });
      svg.addEventListener('pointerleave', hideTip);
      return map;
    }

    function render() {
      maps.forEach(function (map) {
        map.paths.forEach(function (path) {
          var r = rate(path.dataset.code, map.period, map.measure);
          path.setAttribute('fill', r === null ? 'url(#' + map.hatch + ')' : r <= 0 ? ZERO : colour(map.scale, r));
        });
        map.when.textContent = periodText(map.period);
        map.svg.setAttribute('aria-label', 'World map of ' + TITLES[map.measure].toLowerCase() + ' ' + periodText(map.period) + '. The table below the maps lists every value.');
      });
      slider.value = String(state.week);
      weekLabel.textContent = weekText(data.weekStarts[state.week]);
      slider.setAttribute('aria-valuetext', weekLabel.textContent);
      if (details.open) renderTable();
      if (hovered) fillTip(hovered.map, hovered.path);
    }

    function renderTable() {
      PERIODS.forEach(function (period) { groupCells[period].textContent = (period === 'weekly' ? 'In the week ' + weekText(data.weekStarts[state.week]) : 'In total up to ' + dayText(day(data.weekStarts[state.week], 6), true)); });
      sortButtons.forEach(function (b) {
        var active = b.key === state.sortKey;
        b.button.textContent = b.label + (active ? (state.sortUp ? ' ▲' : ' ▼') : '');
        if (active) b.th.setAttribute('aria-sort', state.sortUp ? 'ascending' : 'descending');
        else b.th.removeAttribute('aria-sort');
      });
      var rows = Object.keys(data.countries).map(function (code) {
        var row = { code: code, name: data.countries[code].name, pop: data.countries[code].pop };
        PERIODS.forEach(function (period) {
          MEASURES.forEach(function (measure) { row[period + ' ' + measure] = rate(code, period, measure); });
        });
        return row;
      });
      var sign = state.sortUp ? 1 : -1;
      rows.sort(function (a, b) {
        var x = a[state.sortKey];
        var y = b[state.sortKey];
        if (x === null || y === null) return x === null ? (y === null ? 0 : 1) : -1;
        if (typeof x === 'string') return sign * x.localeCompare(y, 'en');
        return sign * (x - y);
      });
      tbody.textContent = '';
      rows.forEach(function (row) {
        var tr = el('tr', '', tbody);
        el('td', '', tr, row.name + (onMap[row.code] ? '' : ' *'));
        PERIODS.forEach(function (period) {
          MEASURES.forEach(function (measure) {
            var r = row[period + ' ' + measure];
            el('td', 'wm-num', tr, r === null ? '–' : threeDigits.format(r));
          });
        });
        el('td', 'wm-num', tr, population(row.pop, true));
      });
      var note = el('tr', 'wm-note', tbody);
      var cell = el('td', '', note, '* Too small for the maps. – No report.');
      cell.colSpan = 2 + PERIODS.length * MEASURES.length;
    }

    var hovered = null;

    function fillTip(map, path) {
      var code = path.dataset.code;
      var country = data.countries[code];
      var r = rate(code, map.period, map.measure);
      tip.textContent = '';
      el('strong', '', tip, r === null ? 'No data' : threeDigits.format(r) + ' ' + UNITS[map.measure]);
      el('span', 'wm-tip-name', tip, country ? country.name : path.dataset.name);
      if (country && r !== null) {
        if (map.measure === 'tests') el('span', '', tip, 'Tests per thousand people ' + periodText(map.period));
        else el('span', '', tip, count.format(value(code, map.period, map.measure)) + ' ' + map.measure + ' ' + periodText(map.period));
        el('span', '', tip, 'Population ' + population(country.pop));
      } else if (country) {
        el('span', '', tip, 'No report ' + periodText(map.period));
      }
    }

    // The country under the pointer is outlined in all six maps; the tooltip shows in the map it is over.
    function highlight(code, on) {
      maps.forEach(function (m) {
        (m.byCode[code] || []).forEach(function (path) {
          path.classList.toggle('wm-hover', on);
          if (on) m.group.appendChild(path);
        });
      });
    }

    function showTip(map, path, event) {
      if (!hovered || hovered.path !== path) {
        if (hovered) highlight(hovered.path.dataset.code, false);
        hovered = { map: map, path: path };
        highlight(path.dataset.code, true);
        map.figure.appendChild(tip);
      }
      fillTip(map, path);
      tip.hidden = false;
      var box = map.figure.getBoundingClientRect();
      var x = event.clientX - box.left + 14;
      var y = event.clientY - box.top + 14;
      if (x + tip.offsetWidth > box.width) x = Math.max(0, event.clientX - box.left - tip.offsetWidth - 14);
      if (y + tip.offsetHeight > box.height) y = Math.max(0, event.clientY - box.top - tip.offsetHeight - 14);
      tip.style.left = x + 'px';
      tip.style.top = y + 'px';
    }

    function hideTip() {
      if (hovered) highlight(hovered.path.dataset.code, false);
      hovered = null;
      tip.hidden = true;
    }

    function setWeek(week) {
      state.week = week;
      render();
    }

    function setPlaying(on) {
      clearInterval(state.timer);
      state.timer = null;
      if (on) {
        if (state.week >= last) setWeek(0);
        state.timer = setInterval(function () {
          if (state.week >= last) setPlaying(false);
          else setWeek(state.week + 1);
        }, FRAME_MS);
      }
      play.textContent = on ? '❚❚ Pause' : '▶ Play';
    }

    play.addEventListener('click', function () { setPlaying(!state.timer); });
    slider.addEventListener('input', function () {
      setPlaying(false);
      setWeek(Number(slider.value));
    });

    render();

    var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (maps.length && root.dataset.autoplay !== 'false' && !reduceMotion && 'IntersectionObserver' in window) {
      var observer = new IntersectionObserver(function (entries) {
        if (!entries.some(function (entry) { return entry.isIntersecting; })) return;
        observer.disconnect();
        if (!state.timer && state.week === last) {
          setWeek(0);
          setPlaying(true);
        }
      }, { threshold: 0.6 });
      observer.observe(maps[0].svg);
    }
  }

  function init(root) {
    fetch(root.dataset.src)
      .then(function (response) {
        if (!response.ok) throw new Error('HTTP ' + response.status);
        return response.json();
      })
      .then(function (data) { build(root, data); })
      .catch(function (error) {
        root.insertBefore(el('p', 'wm-error', null, 'The maps could not be loaded (' + error.message + ').'), root.firstChild);
      });
  }

  document.querySelectorAll('.worldmap[data-src]').forEach(init);
})();
