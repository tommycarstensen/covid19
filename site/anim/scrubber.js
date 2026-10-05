// Scrubbable map animations. Each <video data-fps> gets a play/pause button and a slider that steps one frame at a time, with the frame's date beside it. The video loops like the GIF it replaced, holding the last frame for data-hold ms, and pauses while scrolled out of view. Without JavaScript the video keeps the browser's own controls. Built by build_animations.py.
(function () {
  'use strict';

  var CSS = [
    '.scrub{display:flex;align-items:center;gap:8px;max-width:100%;margin:4px 0 0;font:13px/1.2 system-ui,-apple-system,"Segoe UI",sans-serif;color:#333}',
    '.scrub button{flex:none;display:grid;place-items:center;width:32px;height:32px;padding:0;border:1px solid #bbb;border-radius:50%;background:#fff;color:#333;cursor:pointer}',
    '.scrub button:hover{border-color:#666}',
    '.scrub button:focus-visible,.scrub input:focus-visible{outline:2px solid #1a73e8;outline-offset:2px}',
    '.scrub svg{width:12px;height:12px;fill:currentColor}',
    '.scrub input{flex:1;min-width:60px;margin:0;accent-color:#555;cursor:pointer}',
    '.scrub output{flex:none;min-width:6.2em;text-align:right;font-variant-numeric:tabular-nums}',
    'video[data-fps]{display:block;max-width:100%;height:auto;cursor:pointer}'
  ].join('\n');

  var PLAY = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1v10l9-5z"/></svg>';
  var PAUSE = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1h3v10H2zM7 1h3v10H7z"/></svg>';

  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function setup(video) {
    var fps = parseFloat(video.getAttribute('data-fps'));
    var hold = parseInt(video.getAttribute('data-hold') || '0', 10);
    var labels = (video.getAttribute('data-labels') || '').split(',').filter(Boolean);
    var count = labels.length;
    var wantPlay = !reduceMotion;
    var visible = true;
    var holdTimer = null;

    video.controls = false;
    video.loop = false;
    video.removeAttribute('autoplay');
    if (!wantPlay) video.pause();

    var bar = document.createElement('div');
    bar.className = 'scrub';
    var button = document.createElement('button');
    button.type = 'button';
    var range = document.createElement('input');
    range.type = 'range';
    range.min = '0';
    range.step = '1';
    range.setAttribute('aria-label', 'Frame');
    var output = document.createElement('output');
    bar.appendChild(button);
    bar.appendChild(range);
    bar.appendChild(output);
    video.parentNode.insertBefore(bar, video.nextSibling);
    bar.style.width = video.getAttribute('width') + 'px';

    function frameAt(time) {
      return Math.max(0, Math.min(count - 1, Math.floor(time * fps + 1e-3)));
    }

    function label(i) {
      return labels[i] || (i + 1) + ' / ' + count;
    }

    function show(i) {
      range.value = String(i);
      output.textContent = label(i);
      range.setAttribute('aria-valuetext', label(i));
    }

    function render(time) {
      if (count) show(frameAt(typeof time === 'number' ? time : video.currentTime));
      button.innerHTML = wantPlay ? PAUSE : PLAY;
      button.setAttribute('aria-label', wantPlay ? 'Pause' : 'Play');
    }

    function seek(i) {
      // Aim at the middle of the frame so rounding never lands on its neighbour.
      video.currentTime = (i + 0.5) / fps;
      show(i);
    }

    function cancelHold() {
      if (holdTimer !== null) {
        clearTimeout(holdTimer);
        holdTimer = null;
      }
    }

    function run() {
      if (!wantPlay || !visible || holdTimer !== null) return;
      var promise = video.play();
      if (promise && promise.catch) {
        // Autoplay can be refused (iOS Low Power Mode): show a play button instead.
        promise.catch(function () {
          wantPlay = false;
          render();
        });
      }
    }

    function setPlaying(play) {
      wantPlay = play;
      cancelHold();
      if (play) {
        if (video.ended || frameAt(video.currentTime) >= count - 1) seek(0);
        run();
      } else {
        video.pause();
      }
      render();
    }

    function ready() {
      if (!count) count = Math.round(video.duration * fps);
      range.max = String(count - 1);
      if (wantPlay) run();
      else seek(count - 1);
      render();
    }

    button.addEventListener('click', function () { setPlaying(!wantPlay); });
    video.addEventListener('click', function () { setPlaying(!wantPlay); });
    range.addEventListener('input', function () {
      if (wantPlay) setPlaying(false);
      seek(parseInt(range.value, 10));
    });
    video.addEventListener('ended', function () {
      if (!wantPlay) return render();
      holdTimer = setTimeout(function () {
        holdTimer = null;
        video.currentTime = 0;
        run();
      }, hold);
      render();
    });
    ['play', 'pause', 'seeked', 'timeupdate'].forEach(function (type) {
      video.addEventListener(type, function () { render(); });
    });
    if ('requestVideoFrameCallback' in video) {
      // mediaTime is the time of the frame on screen; currentTime can trail it by a frame.
      var onFrame = function (now, meta) {
        render(meta.mediaTime);
        video.requestVideoFrameCallback(onFrame);
      };
      video.requestVideoFrameCallback(onFrame);
    }
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (entries) {
        visible = entries[entries.length - 1].isIntersecting;
        if (visible) run();
        else if (!video.paused) video.pause();
      }).observe(video);
    }

    if (video.readyState >= 1) ready();
    else video.addEventListener('loadedmetadata', ready, { once: true });
    render();
  }

  function init() {
    var style = document.createElement('style');
    style.textContent = CSS;
    document.head.appendChild(style);
    var videos = document.querySelectorAll('video[data-fps]');
    for (var i = 0; i < videos.length; i++) setup(videos[i]);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
