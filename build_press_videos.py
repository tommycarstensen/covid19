"""Repackage the Danish press page's three Folketing clips as MP4, so every browser plays them.

press_denmark.html shows three clips from the Folketing's sitting of 29 April 2020 as QuickTime files (2020-04-29_*.mov, H.264 and AAC), which only exist on the server. The server sends them as video/quicktime, which Firefox does not play, and each file keeps its index at the end, so a browser has to fetch the end of the file before it can start. For each clip this script fetches the .mov over SFTP into site/ if it is not there, copies its streams unchanged into site/<clip>.mp4 with the index first (ffmpeg -c copy -movflags +faststart, so nothing is re-encoded), checks that the MP4 has the same streams and duration, and saves a poster frame as site/<clip>.jpg. The page offers the MP4 first and the .mov as a fallback.

    python3 build_press_videos.py            # build what is missing
    python3 build_press_videos.py --force    # build again

The log is tmp/build_press_videos.log. The password is read as deploy.py reads it.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import paramiko

from deploy import HOST, PASSWORD_FILE, REMOTE, USER, Tee

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
LOG = ROOT / "tmp" / "build_press_videos.log"
CLIPS = ["2020-04-29_Jakob", "2020-04-29_Soeren", "2020-04-29_Alex"]
# Seconds into each clip for the poster: past the cut at the start, on the speaker.
POSTER_AT = 2.0


def fetch(names: list[str], out: Tee) -> None:
    """Copy the server's .mov files that site/ lacks, through a temporary name."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=HOST,
        port=22,
        username=USER,
        password=PASSWORD_FILE.read_text().strip(),
        look_for_keys=False,
        allow_agent=False,
    )
    try:
        sftp = client.open_sftp()
        for name in names:
            target = SITE / f"{name}.mov"
            part = target.with_suffix(".mov.part")
            sftp.get(f"{REMOTE}/{name}.mov", str(part))
            part.rename(target)
            print(f"  fetched {target.name} from the server, {target.stat().st_size / 1e6:.1f} MB", file=out)
        sftp.close()
    finally:
        client.close()


def probe(path: Path) -> tuple[list[str], float]:
    """The codecs of a file's streams and its duration in seconds."""
    done = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name:format=duration", "-of", "json", str(path)],
        capture_output=True, check=True, text=True,
    )
    info = json.loads(done.stdout)
    return [s["codec_name"] for s in info["streams"]], float(info["format"]["duration"])


def index_first(path: Path) -> bool:
    """Whether the MP4's index (moov) comes before its media data (mdat)."""
    head = path.read_bytes()[:4096 * 1024]
    moov, mdat = head.find(b"moov"), head.find(b"mdat")
    return moov != -1 and (mdat == -1 or moov < mdat)


def build(name: str, out: Tee) -> None:
    mov, mp4, jpg = (SITE / f"{name}{ext}" for ext in (".mov", ".mp4", ".jpg"))
    part = mp4.with_name(f"{name}.part.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(mov), "-map", "0", "-c", "copy", "-movflags", "+faststart", str(part)],
        check=True,
    )
    before, after = probe(mov), probe(part)
    if before[0] != after[0] or abs(before[1] - after[1]) > 0.1 or not index_first(part):
        part.unlink()
        raise SystemExit(f"{name}: the MP4 differs from the .mov ({before} against {after}) or keeps its index at the end")
    part.rename(mp4)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", str(POSTER_AT), "-i", str(mov), "-frames:v", "1", "-q:v", "3", str(jpg)],
        check=True,
    )
    print(f"  {mp4.name}: {', '.join(after[0])}, {after[1]:.1f} s, {mp4.stat().st_size / 1e6:.1f} MB, index first; poster {jpg.name}", file=out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repackage the Folketing clips as MP4.")
    parser.add_argument("--force", action="store_true", help="build every clip again")
    args = parser.parse_args()
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as log:
        out = Tee(sys.stdout, log)
        print(f"log: {LOG}", file=out)
        print(f"start {time.strftime('%Y-%m-%d %H:%M:%S')}", file=out)
        missing = [name for name in CLIPS if not (SITE / f"{name}.mov").exists()]
        print(f"{len(CLIPS)} clips; {len(missing)} .mov to fetch from the server", file=out)
        if missing:
            fetch(missing, out)
        todo = [name for name in CLIPS if args.force or not (SITE / f"{name}.mp4").exists() or not (SITE / f"{name}.jpg").exists()]
        print(f"{len(todo)} to build, {len(CLIPS) - len(todo)} already built", file=out)
        for name in todo:
            build(name, out)
        print("exit status: 0", file=out)


if __name__ == "__main__":
    main()
