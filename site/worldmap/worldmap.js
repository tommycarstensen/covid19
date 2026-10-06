// Interactive world map of ECDC's weekly COVID-19 cases and deaths per million people.
// Draws every <div class="worldmap" data-src="..."> from the JSON that build_world_map.py writes. data-autoplay="false" stops it playing once when first scrolled into view.
(function () {
  'use strict';

  var SVG_NS = 'http://www.w3.org/2000/svg';
  // One-hue sequential ramps of eight classes at evenly spaced OKLCH lightness (0.905 to 0.338): blue for cases, orange for deaths.
  var RAMPS = {
    cases: ['#cde2fb', '#a5c9f5', '#7bafee', '#5095e7', '#2c7ad8', '#2062b3', '#164b8f', '#0d366b'],
    deaths: ['#fdd6c8', '#f6b49c', '#ed906e', '#e26a3c', '#cd4903', '#a83a00', '#832c02', '#611e01']
  };
  // Lower bounds of classes 2 to 8, per million people: half-decade steps, the same in every week so that a colour always means the same rate.
  var BREAKS = {
    'cases weekly': [3, 10, 30, 100, 300, 1000, 3000],
    'cases cumulative': [30, 100, 300, 1000, 3000, 10000, 30000],
    'deaths weekly': [0.3, 1, 3, 10, 30, 100, 300],
    'deaths cumulative': [1, 3, 10, 30, 100, 300, 1000]
  };
  var ZERO = '#e2e1dc';
  var FRAME_MS = 350;
  var MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  var count = new Intl.NumberFormat('en-GB');
  var threeDigits = new Intl.NumberFormat('en-GB', { maximumSignificantDigits: 3 });
  var millions = new Intl.NumberFormat('en-GB', { minimumSignificantDigits: 3, maximumSignificantDigits: 3 });
  var instances = 0;

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

  function classOf(rate, breaks) {
    if (rate === null) return 'nodata';
    if (rate <= 0) return 'zero';
    var i = 0;
    while (i < breaks.length && rate >= breaks[i]) i++;
    return i;
  }

  function classLabels(breaks) {
    var labels = ['under ' + count.format(breaks[0])];
    for (var i = 1; i < breaks.length; i++) labels.push(count.format(breaks[i - 1]) + '–' + count.format(breaks[i]));
    labels.push(count.format(breaks[breaks.length - 1]) + ' or more');
    return labels;
  }

  function segmented(parent, label, options, onChange) {
    var group = el('div', 'wm-seg', parent);
    group.setAttribute('role', 'group');
    group.setAttribute('aria-label', label);
    var buttons = options.map(function (option) {
      var button = el('button', '', group, option[1]);
      button.type = 'button';
      button.addEventListener('click', function () { onChange(option[0]); });
      return { value: option[0], node: button };
    });
    return function update(current) {
      buttons.forEach(function (b) { b.node.setAttribute('aria-pressed', String(b.value === current)); });
    };
  }

  function build(root, data) {
    var id = 'worldmap' + (++instances);
    var last = data.weeks.length - 1;
    var state = { metric: 'cases', period: 'weekly', week: last, timer: null };
    var series = {};
    Object.keys(data.countries).forEach(function (code) {
      var country = data.countries[code];
      series[code] = {
        weekly: { cases: country.cases, deaths: country.deaths },
        cumulative: { cases: cumulate(country.cases), deaths: cumulate(country.deaths) }
      };
    });
    root.textContent = '';

    var controls = el('div', 'wm-controls', root);
    var updateMetric = segmented(controls, 'Measure', [['cases', 'Cases'], ['deaths', 'Deaths']], function (value) {
      state.metric = value;
      render();
    });
    var updatePeriod = segmented(controls, 'Period', [['weekly', 'Per week'], ['cumulative', 'Cumulative']], function (value) {
      state.period = value;
      render();
    });
    var player = el('div', 'wm-player', controls);
    var play = el('button', 'wm-play', player, '▶ Play');
    play.type = 'button';
    var slider = el('input', 'wm-slider', player);
    slider.type = 'range';
    slider.min = '0';
    slider.max = String(last);
    slider.step = '1';
    slider.setAttribute('aria-label', 'Week');

    var caption = el('p', 'wm-caption', root);
    var figure = el('div', 'wm-figure', root);
    var svg = svgEl('svg', { viewBox: data.viewBox.join(' '), class: 'wm-map', role: 'img' }, figure);
    svg.setAttribute('aria-label', 'World map coloured by the measure and week chosen above. The table below the map lists every value.');
    var pattern = svgEl('pattern', { id: id + '-hatch', width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, svgEl('defs', {}, svg));
    svgEl('rect', { width: 5, height: 5, fill: '#ffffff' }, pattern);
    svgEl('line', { x1: 0, y1: 0, x2: 0, y2: 5, stroke: '#a9a7a0', 'stroke-width': 1.6 }, pattern);
    var group = svgEl('g', {}, svg);
    var paths = data.shapes.map(function (shape) {
      var path = svgEl('path', { d: shape.d }, group);
      path.dataset.code = shape.code;
      path.dataset.name = shape.name;
      return path;
    });
    var tip = el('div', 'wm-tip', figure);
    tip.hidden = true;

    var legend = el('div', 'wm-legend', root);
    var onMap = {};
    data.shapes.forEach(function (shape) { onMap[shape.code] = true; });
    var offMap = Object.keys(data.countries).filter(function (code) { return !onMap[code]; }).length;
    el('p', 'wm-source', root, 'Data: ECDC, weekly cases and deaths reported in ISO weeks ' + data.weeks[0] + ' to ' + data.weeks[last] + ', per million people of 2019. Countries without a report that week are hatched. ' + offMap + ' territories too small for the map are in the table.');

    var details = el('details', 'wm-table', root);
    var summary = el('summary', '', details, 'Table of all ' + Object.keys(data.countries).length + ' countries and territories');
    summary.id = id + '-summary';
    var scroll = el('div', 'wm-table-scroll', details);
    var table = el('table', '', scroll);
    var headRow = el('tr', '', el('thead', '', table));
    var countHeader;
    el('th', '', headRow, 'Country or territory');
    el('th', 'wm-num', headRow, 'Per million');
    countHeader = el('th', 'wm-num', headRow);
    el('th', 'wm-num', headRow, 'Population');
    var tbody = el('tbody', '', table);
    details.addEventListener('toggle', function () { if (details.open) renderTable(); });

    function value(code) {
      var s = series[code];
      return s ? s[state.period][state.metric][state.week] : null;
    }

    function rate(code) {
      var v = value(code);
      return v === null ? null : v * 1e6 / data.countries[code].pop;
    }

    function periodText() {
      var start = data.weekStarts[state.week];
      return state.period === 'weekly' ? 'in the week ' + weekText(start) : 'in total up to ' + dayText(day(start, 6), true);
    }

    function render() {
      var breaks = BREAKS[state.metric + ' ' + state.period];
      var ramp = RAMPS[state.metric];
      paths.forEach(function (path) {
        var k = classOf(rate(path.dataset.code), breaks);
        path.setAttribute('fill', k === 'nodata' ? 'url(#' + id + '-hatch)' : k === 'zero' ? ZERO : ramp[k]);
      });
      updateMetric(state.metric);
      updatePeriod(state.period);
      slider.value = String(state.week);
      var label = (state.metric === 'cases' ? 'Cases' : 'Deaths') + ' per million people ' + periodText();
      caption.textContent = label;
      slider.setAttribute('aria-valuetext', weekText(data.weekStarts[state.week]));
      renderLegend(breaks, ramp);
      if (details.open) renderTable();
      if (hovered) fillTip(hovered);
    }

    function key(color, label, extraClass) {
      var item = el('span', 'wm-key', legend);
      var swatch = el('span', 'wm-swatch' + (extraClass ? ' ' + extraClass : ''), item);
      if (color) swatch.style.background = color;
      el('span', '', item, label);
    }

    function renderLegend(breaks, ramp) {
      legend.textContent = '';
      el('span', 'wm-legend-title', legend, (state.metric === 'cases' ? 'Cases' : 'Deaths') + ' per million people' + (state.period === 'weekly' ? ', per week' : ', cumulative'));
      key(ZERO, '0');
      classLabels(breaks).forEach(function (label, i) { key(ramp[i], label); });
      key(null, 'no data', 'wm-nodata');
    }

    function renderTable() {
      countHeader.textContent = (state.metric === 'cases' ? 'Cases' : 'Deaths') + (state.period === 'weekly' ? ' that week' : ' in total');
      var rows = Object.keys(data.countries).map(function (code) {
        return { code: code, rate: rate(code), value: value(code) };
      });
      rows.sort(function (a, b) {
        if (a.rate === null || b.rate === null) return a.rate === null ? (b.rate === null ? 0 : 1) : -1;
        return b.rate - a.rate;
      });
      tbody.textContent = '';
      rows.forEach(function (row) {
        var country = data.countries[row.code];
        var tr = el('tr', '', tbody);
        el('td', '', tr, country.name + (onMap[row.code] ? '' : ' *'));
        el('td', 'wm-num', tr, row.rate === null ? '–' : threeDigits.format(row.rate));
        el('td', 'wm-num', tr, row.value === null ? '–' : count.format(row.value));
        el('td', 'wm-num', tr, population(country.pop, true));
      });
      var note = el('tr', 'wm-note', tbody);
      var cell = el('td', '', note, '* Too small for the map. – No report.');
      cell.colSpan = 4;
    }

    var hovered = null;

    function fillTip(path) {
      var code = path.dataset.code;
      var country = data.countries[code];
      var r = rate(code);
      tip.textContent = '';
      el('strong', '', tip, r === null ? 'No data' : threeDigits.format(r) + ' per million');
      el('span', 'wm-tip-name', tip, country ? country.name : path.dataset.name);
      if (country && r !== null) {
        var v = value(code);
        el('span', '', tip, count.format(v) + ' ' + state.metric + ' ' + periodText());
        el('span', '', tip, 'Population ' + population(country.pop));
      } else if (country) {
        el('span', '', tip, 'No report ' + periodText());
      }
    }

    function showTip(path, event) {
      if (hovered !== path) {
        if (hovered) hovered.classList.remove('wm-hover');
        hovered = path;
        path.classList.add('wm-hover');
        group.appendChild(path);
      }
      fillTip(path);
      tip.hidden = false;
      var box = figure.getBoundingClientRect();
      var x = event.clientX - box.left + 14;
      var y = event.clientY - box.top + 14;
      if (x + tip.offsetWidth > box.width) x = Math.max(0, event.clientX - box.left - tip.offsetWidth - 14);
      if (y + tip.offsetHeight > box.height) y = Math.max(0, event.clientY - box.top - tip.offsetHeight - 14);
      tip.style.left = x + 'px';
      tip.style.top = y + 'px';
    }

    function hideTip() {
      if (hovered) hovered.classList.remove('wm-hover');
      hovered = null;
      tip.hidden = true;
    }

    svg.addEventListener('pointermove', function (event) {
      var path = event.target.closest ? event.target.closest('path') : null;
      if (path && path.dataset.code) showTip(path, event);
      else hideTip();
    });
    svg.addEventListener('pointerleave', hideTip);

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
    if (root.dataset.autoplay !== 'false' && !reduceMotion && 'IntersectionObserver' in window) {
      var observer = new IntersectionObserver(function (entries) {
        if (!entries.some(function (entry) { return entry.isIntersecting; })) return;
        observer.disconnect();
        if (!state.timer && state.week === last) {
          setWeek(0);
          setPlaying(true);
        }
      }, { threshold: 0.6 });
      observer.observe(svg);
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
        root.textContent = 'The map could not be loaded (' + error.message + ').';
      });
  }

  document.querySelectorAll('.worldmap[data-src]').forEach(init);
})();
