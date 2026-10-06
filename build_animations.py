"""Re-encode the page's map animations as MP4s for the scrubbable player.

The page used to show seven animated GIFs (9.6 MB in all). This writes each one to site/anim/ as an H.264 MP4 plus a JPEG poster of its last frame, cropped to the area the map, title and colour bar use in any frame. site/anim/scrubber.js turns each <video data-fps> into a player with a play/pause button and a slider that steps one frame at a time.

Sources are the files the live page served, as upload.py left them in archive/: the 43 PNG frames of europe.gif (sharper than the GIF's 256-colour palette), and the six world GIFs, whose frames no longer exist separately. A world GIF that plot_choropleth.py has written to the repo root since takes precedence over the archive/ copy. The weekly cases and deaths GIFs are not encoded: they show seven-week totals (the code was fixed in plot_choropleth.py in 40f795d, the GIFs never regenerated), and the interactive world map from build_world_map.py replaces them.

Usage: python3 build_animations.py. The <video> tags for site/index.html are written to tmp/build_animations_markup.html.
"""

import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from functools import partial
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'archive'
OUT = ROOT / 'site' / 'anim'

# White margin kept around the union of all frames' content, in pixels.
PAD = 6
# A pixel counts as content when it differs from white by more than this.
WHITE_TOLERANCE = 12
# How long the last frame is held before the loop restarts, as in the GIFs.
HOLD_MS = 4000
# x264 quality: at 22 the titles and colour-bar labels are indistinguishable from the GIF at 2x zoom, and the file is 37% smaller than at 18.
CRF = 22
# Frames between keyframes. Dragging decodes forward from the last keyframe, which for frames this small is instant; a keyframe every 2 frames instead more than doubled the size.
GOP = 30


@dataclass(frozen=True)
class Animation:
    name: str
    source: Path
    frames: list[Image.Image]
    fps: int
    labels: list[str]
    alt: str


def flatten(im: Image.Image) -> Image.Image:
    """Return the image as RGB, compositing any transparency onto white."""
    rgba = im.convert('RGBA')
    white = Image.new('RGBA', rgba.size, 'white')
    return Image.alpha_composite(white, rgba).convert('RGB')


def gif_frames(path: Path) -> tuple[list[Image.Image], list[int]]:
    """Return every frame of a GIF and each frame's duration in ms."""
    frames: list[Image.Image] = []
    durations: list[int] = []
    with Image.open(path) as gif:
        for i in range(getattr(gif, 'n_frames', 1)):
            gif.seek(i)
            frames.append(flatten(gif))
            durations.append(int(gif.info.get('duration', 0)))
    return frames, durations


def fps_from(durations: list[int], path: Path) -> int:
    """Frame rate of a GIF whose frames all last the same, bar the last."""
    steady = set(durations[:-1])
    if len(steady) != 1:
        sys.exit(f'{path.name}: uneven frame durations {sorted(steady)}')
    return round(1000 / steady.pop())


def weekly(first: date, count: int) -> list[str]:
    return [(first + timedelta(weeks=i)).isoformat() for i in range(count)]


def europe() -> Animation:
    paths = sorted(ARCHIVE.glob('europe_20[0-9][0-9]-W[0-9][0-9].png'))
    gif = ARCHIVE / 'europe.gif'
    _, durations = gif_frames(gif)
    if len(paths) != len(durations):
        sys.exit(f'europe: {len(paths)} PNG frames, {len(durations)} in GIF')
    frames = []
    for p in paths:
        with Image.open(p) as im:
            frames.append(flatten(im))
    return Animation(
        name='europe',
        source=gif,
        frames=frames,
        fps=fps_from(durations, gif),
        labels=[p.stem.removeprefix('europe_') for p in paths],
        alt=('ECDC 14-day notification rate of new COVID-19 cases per '
             '100,000 inhabitants, by European region, weekly from '
             '2020-W13 to 2021-W02'),
    )


# File name, the date in the first frame's title, and what it shows. Check the first date against a regenerated GIF's first frame: the labels are counted from it in weeks.
# Not the ECDC world maps covid19_{cases,deaths}{cumulated,perweek}_*: the interactive world map (build_world_map.py, site/worldmap/) replaces them, as the weekly GIFs show seven-week totals and all four jump between frames.
WORLD = [
    ('covid19_total_tests_per_thousand', date(2020, 1, 5),
     'Cumulative COVID-19 tests per thousand, by country'),
    ('covid19_weekly_tests_per_thousand', date(2020, 1, 5),
     'COVID-19 tests per thousand per week, by country'),
]


def world(name: str, first: date, what: str) -> Animation:
    gif = ROOT / f'{name}.gif'
    if not gif.exists():
        gif = ARCHIVE / gif.name
    frames, durations = gif_frames(gif)
    labels = weekly(first, len(frames))
    return Animation(
        name=name,
        source=gif,
        frames=frames,
        fps=fps_from(durations, gif),
        labels=labels,
        alt=f'{what}, weekly from {labels[0]} to {labels[-1]}',
    )


def content_box(frames: list[Image.Image]) -> tuple[int, int, int, int]:
    """Union of every frame's non-white area, padded, with even sides."""
    width, height = frames[0].size
    white = Image.new('RGB', (width, height), 'white')
    threshold = [255 if v > WHITE_TOLERANCE else 0 for v in range(256)]
    left, top, right, bottom = width, height, 0, 0
    for im in frames:
        mask = ImageChops.difference(im, white).convert('L')
        box = mask.point(threshold).getbbox()
        if box is None:
            continue
        left, top = min(left, box[0]), min(top, box[1])
        right, bottom = max(right, box[2]), max(bottom, box[3])
    left, top = max(0, left - PAD), max(0, top - PAD)
    right, bottom = min(width, right + PAD), min(height, bottom + PAD)
    # yuv420p needs even dimensions: grow by a pixel where there is room.
    if (right - left) % 2:
        right = right + 1 if right < width else right - 1
    if (bottom - top) % 2:
        bottom = bottom + 1 if bottom < height else bottom - 1
    return left, top, right, bottom


def encode(anim: Animation) -> tuple[int, int, int, int]:
    """Write <name>.mp4 and <name>.jpg; return width, height and sizes."""
    box = content_box(anim.frames)
    frames = [im.crop(box) for im in anim.frames]
    width, height = frames[0].size
    mp4 = OUT / f'{anim.name}.mp4'
    with tempfile.TemporaryDirectory() as tmp:
        for i, im in enumerate(frames):
            im.save(Path(tmp) / f'{i:04d}.png')
        subprocess.run([
            'ffmpeg', '-v', 'error', '-y',
            '-framerate', str(anim.fps), '-i', str(Path(tmp) / '%04d.png'),
            '-vf', ('scale=out_color_matrix=bt709:out_range=tv'
                    ':flags=bicubic+accurate_rnd+full_chroma_int'
                    ',format=yuv420p'),
            '-c:v', 'libx264', '-preset', 'slow', '-crf', str(CRF),
            '-g', str(GOP),
            '-colorspace', 'bt709', '-color_primaries', 'bt709',
            '-color_trc', 'bt709',
            '-movflags', '+faststart', '-an', str(mp4),
        ], check=True)
    jpg = OUT / f'{anim.name}.jpg'
    frames[-1].save(jpg, quality=90, optimize=True)
    return width, height, mp4.stat().st_size, jpg.stat().st_size


def markup(anim: Animation, width: int, height: int) -> str:
    return (
        f'<video src="anim/{anim.name}.mp4" poster="anim/{anim.name}.jpg" '
        f'width="{width}" height="{height}" autoplay muted loop playsinline '
        f'controls data-fps="{anim.fps}" data-hold="{HOLD_MS}" '
        f'data-labels="{",".join(anim.labels)}" '
        f'aria-label="{anim.alt}"></video>'
    )


def main() -> None:
    if shutil.which('ffmpeg') is None:
        sys.exit('ffmpeg not found: brew install ffmpeg')
    OUT.mkdir(parents=True, exist_ok=True)
    specs: list[Callable[[], Animation]] = [
        europe, *(partial(world, n, d, w) for n, d, w in WORLD)]
    print(f'Encoding {len(specs)} animations into {OUT.relative_to(ROOT)}/')
    before = after = 0
    snippets = []
    for spec in specs:
        anim = spec()
        gif_size = anim.source.stat().st_size
        width, height, mp4_size, jpg_size = encode(anim)
        before += gif_size
        after += mp4_size + jpg_size
        print(f'  {anim.name}: {len(anim.frames)} frames at {anim.fps} fps '
              f'from {anim.source.relative_to(ROOT)}, '
              f'{anim.frames[0].size[0]}x{anim.frames[0].size[1]} cropped '
              f'to {width}x{height}; GIF {gif_size / 1e6:.2f} MB '
              f'-> MP4 {mp4_size / 1e6:.2f} MB + poster '
              f'{jpg_size / 1e3:.0f} kB')
        snippets.append(markup(anim, width, height))
    snippet_path = ROOT / 'tmp' / 'build_animations_markup.html'
    snippet_path.parent.mkdir(exist_ok=True)
    snippet_path.write_text('\n'.join(snippets) + '\n')
    print(f'Done: {before / 1e6:.1f} MB of GIF -> {after / 1e6:.1f} MB of '
          f'MP4 and posters. <video> tags for site/index.html are in '
          f'{snippet_path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
