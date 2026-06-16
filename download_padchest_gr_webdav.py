#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlparse

import requests
from requests.auth import HTTPBasicAuth
from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout


DEFAULT_BASE_URL = "https://b2drop.bsc.es/nextcloud/public.php/webdav/"
DEFAULT_SHARE_TOKEN = "PadChest-GR"
CHUNK_SIZE = 1024 * 1024
DAV_NS = {"d": "DAV:"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the public PadChest-GR share through Nextcloud WebDAV."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--share-token", default=DEFAULT_SHARE_TOKEN)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--retries", type=int, default=10)
    return parser.parse_args()


def propfind(session: requests.Session, base_url: str, remote_path: str) -> list[dict[str, object]]:
    url = remote_url(base_url, remote_path)
    response = session.request("PROPFIND", url, headers={"Depth": "1"}, timeout=120)
    response.raise_for_status()

    entries: list[dict[str, object]] = []
    root = ET.fromstring(response.content)
    for node in root.findall("d:response", DAV_NS):
        href = node.findtext("d:href", namespaces=DAV_NS)
        if not href:
            continue
        name = href_to_name(href)
        prop = node.find("d:propstat/d:prop", DAV_NS)
        if prop is None:
            continue
        resource_type = prop.find("d:resourcetype", DAV_NS)
        is_dir = resource_type is not None and resource_type.find("d:collection", DAV_NS) is not None
        size_text = prop.findtext("d:getcontentlength", default="0", namespaces=DAV_NS)
        entries.append({"name": name, "is_dir": is_dir, "size": int(size_text or 0)})
    return entries


def href_to_name(href: str) -> str:
    path = unquote(urlparse(href).path.rstrip("/"))
    return Path(path).name


def remote_url(base_url: str, remote_path: str) -> str:
    quoted = "/".join(quote(part) for part in remote_path.strip("/").split("/") if part)
    return urljoin(base_url.rstrip("/") + "/", quoted)


def walk(session: requests.Session, base_url: str, remote_path: str = "") -> list[dict[str, object]]:
    entries = propfind(session, base_url, remote_path)
    current_name = Path(remote_path).name if remote_path else ""
    files: list[dict[str, object]] = []
    for entry in entries:
        name = str(entry["name"])
        if name == current_name:
            continue
        child_remote = f"{remote_path.rstrip('/')}/{name}".strip("/")
        if entry["is_dir"]:
            files.extend(walk(session, base_url, child_remote))
        else:
            files.append({"remote_path": child_remote, "size": int(entry["size"])})
    return files


def download_file(
    session: requests.Session,
    base_url: str,
    remote_path: str,
    output_path: Path,
    expected_size: int,
    retries: int,
) -> None:
    if output_path.exists() and expected_size and output_path.stat().st_size == expected_size:
        print(f"OK: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    if output_path.exists() and not part_path.exists():
        output_path.rename(part_path)

    for attempt in range(1, retries + 1):
        resume_from = part_path.stat().st_size if part_path.exists() else 0
        headers = {}
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"

        try:
            response = session.get(remote_url(base_url, remote_path), stream=True, headers=headers, timeout=120)
            response.raise_for_status()
            if resume_from and response.status_code != 206:
                print(f"\nNo resume for {remote_path}; restarting this file.")
                part_path.unlink(missing_ok=True)
                resume_from = 0

            mode = "ab" if resume_from else "wb"
            downloaded = resume_from
            with part_path.open(mode) as handle:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    print_progress(remote_path, downloaded, expected_size, attempt, retries)

            if expected_size and part_path.stat().st_size < expected_size:
                raise RuntimeError(f"incomplete file {part_path.stat().st_size}/{expected_size}")

            part_path.rename(output_path)
            print(f"\nSaved: {output_path}")
            return
        except (ChunkedEncodingError, ConnectionError, ReadTimeout, RuntimeError) as exc:
            wait = min(60, attempt * 5)
            print(f"\nInterrupted {remote_path} ({attempt}/{retries}): {exc}")
            print(f"Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"Could not download {remote_path}")


def print_progress(remote_path: str, downloaded: int, total: int, attempt: int, retries: int) -> None:
    if total:
        percent = downloaded * 100 / total
        print(
            f"\r{remote_path}: {percent:6.2f}% ({format_bytes(downloaded)} / {format_bytes(total)}) "
            f"[{attempt}/{retries}]",
            end="",
            flush=True,
        )
    else:
        print(
            f"\r{remote_path}: {format_bytes(downloaded)} [{attempt}/{retries}]",
            end="",
            flush=True,
        )


def format_bytes(size: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def main() -> None:
    args = parse_args()
    session = requests.Session()
    session.auth = HTTPBasicAuth(args.share_token, "")

    files = walk(session, args.base_url)
    print(f"Found {len(files)} files in public share.")
    for item in files:
        remote_path = str(item["remote_path"])
        output_path = args.output_dir / remote_path
        download_file(session, args.base_url, remote_path, output_path, int(item["size"]), args.retries)


if __name__ == "__main__":
    main()
