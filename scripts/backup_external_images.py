"""Save a copy of every image the pages show from other sites, in case those sites drop them.

Reads each page in site/ as committed at HEAD and collects four kinds of image: <img>, <source> and <video poster> files hotlinked from other hosts; the photos and video stills of every embedded tweet (with the tweet's own JSON from Twitter's syndication service, which names the author, the date and the text), and for a tweet Twitter no longer serves, the image of the Wayback Machine's earliest copy of its page; a PNG and an SVG of every Our World in Data chart shown in an iframe, with the iframe's countries and dates; and the thumbnail of every embedded YouTube video. Each is saved once under external_images/<kind>/ and listed in external_images/manifest.json with the page, the source URL, the time it was fetched, its size, type and SHA-256. A file already saved is not fetched again, so a second run only adds what is new; --refresh fetches everything again.

The images are ignored by git, like every other image here; manifest.json and the tweets' JSON are committed. Nothing is sent to tommycarstensen.com, and requests go one at a time, a second apart. The log is tmp/backup_external_images.log.

    python3 scripts/backup_external_images.py
    python3 scripts/backup_external_images.py --refresh
"""

import argparse
import hashlib
import html
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external_images"
MANIFEST = OUT / "manifest.json"
LOG = ROOT / "tmp" / "backup_external_images.log"
OWN_HOST = "tommycarstensen.com"
PAUSE = 1.0
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36"}
EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
    "application/json": ".json",
}


class Tee:
    """Write to the terminal and to the log."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@dataclass
class Item:
    """One file to save: where it comes from, where it goes, and which page shows it."""

    page: str
    kind: str
    url: str
    name: str
    note: str = ""


def pages() -> dict[str, str]:
    """Every page in site/, as committed at HEAD."""
    listed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "site/*.html"],
        capture_output=True, check=True, text=True,
    ).stdout.split()
    found = {}
    for path in listed:
        shown = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"HEAD:{path}"],
            capture_output=True, check=True, text=True,
        ).stdout
        found[Path(path).name] = re.sub(r"<!--.*?-->", "", shown, flags=re.DOTALL)
    return found


def is_external(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return url.startswith(("http://", "https://")) and not (host == OWN_HOST or host.endswith("." + OWN_HOST))


def slug(text: str, keep: int = 60) -> str:
    """A file-name-safe version of text."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[:keep]


def hotlinked(page: str, text: str) -> list[Item]:
    """<img>, <source> and <video poster> files from other hosts."""
    urls: list[str] = []
    for tag in re.findall(r"<(?:img|source|video)\b[^>]*>", text):
        urls += re.findall(r'\s(?:src|poster)="([^"]+)"', tag)
        for srcset in re.findall(r'\ssrcset="([^"]+)"', tag):
            urls += [part.split()[0] for part in srcset.split(",") if part.strip()]
    items = []
    for url in dict.fromkeys(html.unescape(u) for u in urls):
        if is_external(url):
            parsed = urlparse(url)
            items.append(Item(page, "hotlinked", url, slug(f"{parsed.hostname}_{parsed.path}")))
    return items


def tweets(page: str, text: str) -> list[Item]:
    """The JSON of every embedded tweet; its photos are added once the JSON is read."""
    items = []
    for quote in re.findall(r'<blockquote class="twitter-tweet".*?</blockquote>', text, flags=re.DOTALL):
        found = re.findall(r"twitter\.com/([^/\"]+)/status/(\d+)", quote)
        if found:
            user, tweet = found[-1]
            url = f"https://cdn.syndication.twimg.com/tweet-result?id={tweet}&token=x&lang=da"
            items.append(Item(page, "tweets", url, tweet, note=f"https://twitter.com/{user}/status/{tweet}"))
    return items


def deleted_tweet_media(session: requests.Session, item: Item) -> list[Item]:
    """For a tweet Twitter no longer serves, its image as the Wayback Machine's earliest copy of the tweet's page shows it."""
    cdx = fetch(session, f"https://web.archive.org/cdx/search/cdx?url={item.note}&filter=statuscode:200&fl=timestamp&limit=1")
    stamp = cdx.text.strip() if cdx is not None and cdx.status_code == 200 else ""
    if not stamp.isdigit():
        return []
    snapshot = fetch(session, f"https://web.archive.org/web/{stamp}id_/{item.note}")
    if snapshot is None or snapshot.status_code != 200:
        return []
    images = re.findall(r'<meta\s+property="og:image" content="(https://pbs\.twimg\.com/media/[^"]+)"', snapshot.text)
    return [
        Item(item.page, "tweets", f"https://web.archive.org/web/{stamp}im_/{image}", f"{item.name}_{n}", note=image)
        for n, image in enumerate(dict.fromkeys(images), 1)
    ]


def owid(page: str, text: str) -> list[Item]:
    """A PNG and an SVG of every Our World in Data chart, with the iframe's settings."""
    items = []
    for src in re.findall(r'<iframe\b[^>]*\ssrc="(https://ourworldindata\.org/grapher/[^"]+)"', text):
        src = html.unescape(src)
        base, _, query = src.partition("?")
        chart = base.rsplit("/", 1)[1]
        name = slug(f"{chart}_{query}", 120)
        for ext in (".png", ".svg"):
            items.append(Item(page, "owid", f"{base}{ext}" + (f"?{query}" if query else ""), name, note=src))
    return items


def youtube(page: str, text: str) -> list[Item]:
    """The largest thumbnail of every embedded YouTube video."""
    items = []
    for video in dict.fromkeys(re.findall(r"youtube(?:-nocookie)?\.com/embed/([\w-]+)", text)):
        items.append(Item(page, "youtube", f"https://i.ytimg.com/vi/{video}/maxresdefault.jpg", video, note=f"https://www.youtube.com/watch?v={video}"))
    return items


def fetch(session: requests.Session, url: str) -> requests.Response | None:
    if not is_external(url):
        raise SystemExit(f"refusing to fetch from {OWN_HOST}: {url}")
    time.sleep(PAUSE)
    try:
        return session.get(url, timeout=60)
    except requests.RequestException as error:
        print(f"  failed {url}: {type(error).__name__}", file=sys.stderr)
        return None


def save(item: Item, response: requests.Response, out: Tee) -> dict[str, object] | None:
    """Write the file and return its manifest row, or None if it is not an image or JSON."""
    kind = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
    ext = EXTENSIONS.get(kind)
    if response.status_code != 200 or ext is None:
        print(f"  skipped {item.url} (HTTP {response.status_code}, {kind or 'no type'})", file=out)
        return None
    path = OUT / item.kind / f"{item.name}{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return {
        "page": item.page,
        "kind": item.kind,
        "source": item.url,
        "shown_as": item.note or item.url,
        "file": str(path.relative_to(OUT)),
        "type": kind,
        "bytes": len(response.content),
        "sha256": hashlib.sha256(response.content).hexdigest(),
        "fetched": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def tweet_media(item: Item, data: dict[str, object]) -> list[Item]:
    """The photos and video stills of a tweet and of the tweet it quotes, at their original size."""
    found: list[Item] = []
    for tweet in (data, data.get("quoted_tweet")):
        if not isinstance(tweet, dict):
            continue
        media = tweet.get("mediaDetails")
        if not isinstance(media, list):
            continue
        for n, entry in enumerate(media, 1):
            if isinstance(entry, dict) and isinstance(entry.get("media_url_https"), str):
                url = str(entry["media_url_https"])
                found.append(Item(item.page, "tweets", f"{url}?name=orig", f"{tweet.get('id_str', item.name)}_{n}", note=url))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Back up the images the pages show from other sites.")
    parser.add_argument("--refresh", action="store_true", help="fetch every file again, even if saved")
    args = parser.parse_args()
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as log:
        out = Tee(sys.stdout, log)
        print(f"log: {LOG}", file=out)
        print(f"start {time.strftime('%Y-%m-%d %H:%M:%S')}", file=out)
        manifest: dict[str, dict[str, object]] = {}
        if MANIFEST.exists():
            manifest = {row["source"]: row for row in json.loads(MANIFEST.read_text(encoding="utf-8"))}
        queue: list[Item] = []
        for page, text in pages().items():
            found = hotlinked(page, text) + tweets(page, text) + owid(page, text) + youtube(page, text)
            counts = {kind: sum(1 for i in found if i.kind == kind) for kind in ("hotlinked", "tweets", "owid", "youtube")}
            print(f"{page}: {counts['hotlinked']} hotlinked images, {counts['tweets']} tweets, {counts['owid']} OWID chart files, {counts['youtube']} YouTube thumbnails", file=out)
            queue += found
        print(f"{sum(1 for i in queue if i.url in manifest)} already saved; --refresh fetches them again" if manifest else "nothing saved before", file=out)
        session = requests.Session()
        session.headers.update(HEADERS)
        saved = skipped = failed = 0
        started = time.time()
        done = 0
        while queue:
            item = queue.pop(0)
            done += 1
            row = manifest.get(item.url)
            if row and not args.refresh and (OUT / str(row["file"])).exists():
                skipped += 1
                if item.kind == "tweets" and str(row["file"]).endswith(".json"):
                    data = json.loads((OUT / str(row["file"])).read_text(encoding="utf-8"))
                    deleted = data.get("__typename") == "TweetTombstone"
                    queue = (deleted_tweet_media(session, item) if deleted else tweet_media(item, data)) + queue
                continue
            response = fetch(session, item.url)
            if response is not None and response.status_code == 404 and item.kind == "youtube" and "maxresdefault" in item.url:
                queue.insert(0, Item(item.page, item.kind, item.url.replace("maxresdefault", "hqdefault"), item.name, item.note))
                continue
            new = save(item, response, out) if response is not None else None
            if new is None or response is None:
                failed += 1
                continue
            manifest[item.url] = new
            saved += 1
            if item.kind == "tweets" and new["type"] == "application/json":
                data = response.json()
                media = tweet_media(item, data)
                if data.get("__typename") == "TweetTombstone":
                    media = deleted_tweet_media(session, item)
                    print(f"  {item.note} is deleted; {len(media)} image(s) found in the Wayback Machine", file=out)
                queue = media + queue
            if done % 10 == 0:
                rate = (time.time() - started) / done
                print(f"  {done} done, {len(queue)} queued, about {rate * len(queue):.0f} s left", file=out)
        OUT.mkdir(exist_ok=True)
        rows = sorted(manifest.values(), key=lambda r: (str(r["page"]), str(r["kind"]), str(r["file"])))
        MANIFEST.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        total = sum(int(str(r["bytes"])) for r in rows) / 1e6
        print(f"saved {saved}, already there {skipped}, failed or not an image {failed}; the manifest lists {len(rows)} files, {total:.1f} MB, in {OUT}", file=out)
        print("exit status: 0", file=out)


if __name__ == "__main__":
    main()
