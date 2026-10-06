"""Deploy the covid19 page to https://tommycarstensen.com/covid19/ over SFTP.

site/ maps to /www/covid19/ on the host, which holds about 4,000 files from
2020 and 2021. Only the files site/index.html uses are considered: each one
that exists in site/ is uploaded when it differs from the server's copy, and
each one that does not must already be on the server. Nothing is deleted.

    python3 deploy.py           # upload what differs, check the live page
    python3 deploy.py --dry     # list what would be uploaded

The pages in site/ must be committed, and the server's copy of a page must
be one this repository has committed: a copy it has never seen means
someone changed the page on the server, and is refused. The charts and images in
site/ are ignored by git and rebuilt by redraw_charts.py and
fetch_images.py, so before a server file is overwritten its copy is saved
under tmp/deploy_backup/<time>/. The log is tmp/deploy.log.
"""

import argparse
import hashlib
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO

import paramiko
import requests

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
REMOTE = "/www/covid19"
URL = "https://tommycarstensen.com/covid19/"
HOST = "ssh.tommycarstensen.com"
USER = "tommycarstensen.com"
PASSWORD_FILE = Path(
    os.environ.get("TC_PASSWORD_FILE", Path.home() / "lego" / ".password")
)
LOG = ROOT / "tmp" / "deploy.log"
BACKUP = ROOT / "tmp" / "deploy_backup" / time.strftime("%Y-%m-%d_%H%M%S")


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


def git(*args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True,
        check=True,
        text=True,
    )
    return done.stdout.strip()


def used_files(html: str) -> list[str]:
    """The files on this site that the page loads or links."""
    live = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
    found = re.findall(r'(?:src|href|poster)="([^"]+)"', live)
    # srcset lists candidates as "url [descriptor], url [descriptor]", e.g. a <picture>'s phone image.
    for srcset in re.findall(r'srcset="([^"]+)"', live):
        found += [candidate.split()[0] for candidate in srcset.split(",") if candidate.strip()]
    files = set()
    for value in found:
        if value.startswith(("http:", "https:", "//", "mailto:", "#")):
            continue
        files.add(value.split("#")[0].split("?")[0])
    return sorted(files)


def blob_id(data: bytes) -> str:
    """The id git gives these bytes, to look a server copy up in history."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def committed_blobs(rel: str) -> set[str]:
    """Every version of site/<rel> that this repository has committed."""
    path = f"site/{rel}"
    blobs: set[str] = set()
    for commit in git("log", "--all", "--format=%H", "--", path).split():
        found = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--verify", "--quiet"]
            + [f"{commit}:{path}"],
            capture_output=True,
            check=False,
            text=True,
        )
        if found.returncode == 0:
            blobs.add(found.stdout.strip())
    return blobs


def read_remote(sftp: paramiko.SFTPClient, rel: str) -> bytes | None:
    try:
        with sftp.open(f"{REMOTE}/{rel}", "rb") as fh:
            fh.prefetch()
            return fh.read()
    except FileNotFoundError:
        return None


def remote_size(sftp: paramiko.SFTPClient, rel: str) -> int | None:
    try:
        attrs = sftp.stat(f"{REMOTE}/{rel}")
    except FileNotFoundError:
        return None
    if attrs.st_mode is not None and stat.S_ISDIR(attrs.st_mode):
        return None
    return attrs.st_size


def ensure_dir(sftp: paramiko.SFTPClient, rel: str) -> None:
    """Create the folders above a file, as needed."""
    path = REMOTE
    for part in Path(rel).parent.parts:
        path = f"{path}/{part}"
        try:
            sftp.stat(path)
        except FileNotFoundError:
            sftp.mkdir(path)


def put(sftp: paramiko.SFTPClient, rel: str, data: bytes) -> None:
    """Write to a temporary name and rename it into place, so the live
    site never serves half a file."""
    ensure_dir(sftp, rel)
    remote = f"{REMOTE}/{rel}"
    tmp = remote + ".tmp"
    with sftp.open(tmp, "wb") as fh:
        fh.write(data)
    try:
        sftp.posix_rename(tmp, remote)
    except OSError:
        try:
            sftp.remove(remote)
        except FileNotFoundError:
            pass
        sftp.rename(tmp, remote)
    size = sftp.stat(remote).st_size
    if size != len(data):
        raise SystemExit(f"{rel}: {size} bytes on the server, {len(data)} here")


def plan(
    sftp: paramiko.SFTPClient, out: Tee
) -> tuple[list[tuple[str, bytes]], dict[str, bytes]]:
    """What to upload, and the server copies it would replace."""
    html = (SITE / "index.html").read_text(encoding="utf-8")
    files = ["index.html", *used_files(html)]
    local = [rel for rel in files if (SITE / rel).is_file()]
    pages = [f"site/{rel}" for rel in local if rel.endswith(".html")]
    dirty = git("status", "--porcelain", "--", *pages)
    if dirty:
        raise SystemExit(f"commit the pages before deploying:\n{dirty}")
    elsewhere = [rel for rel in files if not (SITE / rel).is_file()]
    print(
        f"the page uses {len(files)} files: {len(local)} in site/, "
        f"{len(elsewhere)} expected on the server",
        file=out,
    )
    absent = [rel for rel in elsewhere if remote_size(sftp, rel) is None]
    if absent:
        names = ", ".join(absent)
        raise SystemExit(f"neither in site/ nor on the server: {names}")
    uploads: list[tuple[str, bytes]] = []
    replaced: dict[str, bytes] = {}
    for done, rel in enumerate(local, 1):
        data = (SITE / rel).read_bytes()
        size = remote_size(sftp, rel)
        if size == len(data):
            remote = read_remote(sftp, rel)
            if remote == data:
                continue
        else:
            remote = read_remote(sftp, rel) if size is not None else None
        if (
            rel.endswith(".html")
            and remote is not None
            and blob_id(remote) not in committed_blobs(rel)
        ):
            raise SystemExit(
                f"the server's {rel} is not one this repository has "
                "committed, so someone changed it on the server. Save it, "
                f"commit it as site/{rel}, put the change back on top, and "
                "deploy again."
            )
        if remote is not None:
            replaced[rel] = remote
        uploads.append((rel, data))
        if done % 50 == 0:
            print(f"  compared {done}/{len(local)}", file=out)
    return uploads, replaced


def check_live(out: Tee, uploaded: list[str]) -> None:
    """The plain URL must serve exactly site/index.html, and every file
    uploaded must answer. Varnish in front of the host serves its old copy
    for a while after an upload, so the check asks again."""
    expected = (SITE / "index.html").read_bytes()
    tries = 9
    for attempt in range(1, tries + 1):
        live = requests.get(URL, timeout=30)
        if live.content == expected:
            break
        print(
            f"  {URL} still serves {len(live.content)} bytes, not the "
            f"{len(expected)} deployed (try {attempt} of {tries})",
            file=out,
        )
        if attempt == tries:
            raise SystemExit(f"{URL} differs from site/index.html")
        time.sleep(10)
    broken = {}
    for rel in uploaded:
        if rel == "index.html":
            continue
        status = answers(URL + rel)
        if status != 200:
            broken[rel] = status
    if broken:
        raise SystemExit(f"not served (file: HTTP status): {broken}")
    print(f"live and identical, {len(uploaded)} uploaded files answer", file=out)


def answers(url: str, tries: int = 3, wait: float = 5.0) -> int:
    """The status a file answers with. The host refuses a quick burst of
    requests, so they are paced and a failure is asked again."""
    status = 0
    for _ in range(tries):
        time.sleep(0.5)
        status = requests.head(url, timeout=30).status_code
        if status == 200:
            break
        time.sleep(wait)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy tommycarstensen.com/covid19/ over SFTP."
    )
    parser.add_argument(
        "--dry", action="store_true", help="list what would be uploaded"
    )
    args = parser.parse_args()
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as log:
        out = Tee(sys.stdout, log)
        print(f"log: {LOG}", file=out)
        print(f"start {time.strftime('%Y-%m-%d %H:%M:%S')}", file=out)
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # With keys and the agent allowed, paramiko offers keys first and
        # the server hits MaxAuthTries before it asks for the password.
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
            uploads, replaced = plan(sftp, out)
            total = sum(len(data) for _, data in uploads) / 1e6
            print(
                f"{len(uploads)} to upload ({total:.1f} MB), "
                f"{len(replaced)} of them replace a server copy",
                file=out,
            )
            if args.dry:
                for rel, _ in uploads:
                    tag = "replace" if rel in replaced else "new"
                    print(f"  {tag}\t{rel}", file=log)
                print(f"the list is in {LOG}", file=out)
                print("exit status: 0", file=out)
                return
            for rel, data in replaced.items():
                saved = BACKUP / rel
                saved.parent.mkdir(parents=True, exist_ok=True)
                saved.write_bytes(data)
            if replaced:
                print(f"server copies saved in {BACKUP}", file=out)
            started = time.time()
            for done, (rel, data) in enumerate(uploads, 1):
                put(sftp, rel, data)
                print(f"  uploaded {rel}", file=log)
                if done % 25 == 0 or done == len(uploads):
                    rate = (time.time() - started) / done
                    left = rate * (len(uploads) - done)
                    print(
                        f"  {done}/{len(uploads)}, about {left:.0f} s left",
                        file=out,
                    )
            sftp.close()
        finally:
            client.close()
        check_live(out, [rel for rel, _ in uploads])
        print("exit status: 0", file=out)


if __name__ == "__main__":
    main()
