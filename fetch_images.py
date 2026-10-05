"""Copy the page's Wikimedia flags and locator maps into site/img/.

The page showed its flags and continent maps straight from
upload.wikimedia.org at widths such as 45 and 440 pixels. Wikimedia now
serves thumbnails only at standard widths and answers HTTP 400 to the rest,
so 66 of the page's 74 Wikimedia images had gone blank by 2026. This saves
each file in wikimedia_images.txt at 120 pixels wide (shown at 45) as
site/img/<file name>.png, and writes its author and licence, read from the
file's description page, to image_credits.json for the page's credit line.

    python3 fetch_images.py
"""

import html
import json
import re
import sys
import time
import urllib.parse
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
LIST = ROOT / "wikimedia_images.txt"
OUT = ROOT / "site" / "img"
CREDITS = ROOT / "image_credits.json"
WIDTH = 120
API = {
    "commons": "https://commons.wikimedia.org/w/api.php",
    "en": "https://en.wikipedia.org/w/api.php",
}
AGENT = "tommycarstensen.com covid19 page (covid19@tommycarstensen.com)"
HEADERS = {"User-Agent": AGENT}


def plain(text: str) -> str:
    """The text of a bit of HTML from the file's metadata."""
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


def fetch(project: str, name: str) -> dict[str, str]:
    """Save one file's thumbnail and return its credit."""
    title = "File:" + urllib.parse.unquote(name).replace("_", " ")
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": str(WIDTH),
    }
    api = API[project]
    reply = requests.get(api, params=params, headers=HEADERS, timeout=30)
    reply.raise_for_status()
    page = next(iter(reply.json()["query"]["pages"].values()))
    info = page["imageinfo"][0]
    meta = info["extmetadata"]
    image = requests.get(info["thumburl"], headers=HEADERS, timeout=30)
    image.raise_for_status()
    local = f"{urllib.parse.unquote(name)}.png"
    (OUT / local).write_bytes(image.content)
    return {
        "file": f"img/{local}",
        "title": page["title"],
        "page": info["descriptionurl"],
        "artist": plain(meta.get("Artist", {}).get("value", "")),
        "licence": plain(meta.get("LicenseShortName", {}).get("value", "")),
        "licence_url": meta.get("LicenseUrl", {}).get("value", ""),
    }


def main() -> None:
    rows = [
        line.split("\t")
        for line in LIST.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} files from {LIST.name} into {OUT.relative_to(ROOT)}/")
    credits: list[dict[str, str]] = []
    failed: list[str] = []
    for done, (project, name) in enumerate(rows, 1):
        try:
            credits.append(fetch(project, name))
        except (requests.RequestException, KeyError, StopIteration) as error:
            print(f"  {name}: {error}")
            failed.append(name)
        if done % 10 == 0 or done == len(rows):
            print(f"  {done}/{len(rows)}")
        time.sleep(1.0)  # Wikimedia asks for a gentle pace
    CREDITS.write_text(
        json.dumps(credits, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    licences = sorted({c["licence"] for c in credits})
    print(f"{len(credits)} saved, {len(failed)} failed: {failed}")
    print(f"licences: {licences}")
    status = 1 if failed else 0
    print(f"exit status: {status}")
    sys.exit(status)


if __name__ == "__main__":
    main()
