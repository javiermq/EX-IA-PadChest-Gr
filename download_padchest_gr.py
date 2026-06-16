#!/usr/bin/env python3
"""
Download the public PadChest-GR archive from B2Drop/Nextcloud.

Usage:
    python download_padchest_gr.py
    python download_padchest_gr.py --output PadChest-GR.zip
"""

from __future__ import annotations

import argparse
import shutil
import time
import zipfile
from pathlib import Path

import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout


DEFAULT_URL = "https://b2drop.bsc.es/nextcloud/s/PadChest-GR/download"
DEFAULT_OUTPUT = "PadChest-GR.zip"
CHUNK_SIZE = 1024 * 1024
DEFAULT_RETRIES = 20


def format_bytes(size: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def download(url: str, output_path: Path, retries: int, list_zip: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_suffix(output_path.suffix + ".part")

    session = requests.Session()
    attempt = 1
    while attempt <= retries:
        resume_from = part_path.stat().st_size if part_path.exists() else 0
        headers = {"User-Agent": "Mozilla/5.0"}
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"

        try:
            with session.get(url, stream=True, timeout=120, headers=headers) as response:
                response.raise_for_status()

                if resume_from and response.status_code != 206:
                    print("\nEl servidor no acepto resume; reiniciando descarga parcial.")
                    part_path.unlink(missing_ok=True)
                    resume_from = 0

                total = expected_total(response, resume_from)
                downloaded = resume_from
                mode = "ab" if resume_from else "wb"

                with part_path.open(mode) as file:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        file.write(chunk)
                        downloaded += len(chunk)
                        print_progress(downloaded, total, attempt, retries)

            if total and part_path.stat().st_size < total:
                raise RuntimeError(
                    f"Descarga incompleta: {format_bytes(part_path.stat().st_size)} / {format_bytes(total)}"
                )

            shutil.move(str(part_path), str(output_path))
            print(f"\nArchivo guardado en: {output_path.resolve()}")
            if list_zip:
                list_zip_contents(output_path)
            return
        except (ChunkedEncodingError, ConnectionError, ReadTimeout, RuntimeError) as exc:
            wait = min(60, 5 * attempt)
            print(f"\nDescarga interrumpida ({attempt}/{retries}): {exc}")
            print(f"Reintentando en {wait}s desde {format_bytes(part_path.stat().st_size if part_path.exists() else 0)}...")
            time.sleep(wait)
            attempt += 1

    raise RuntimeError(f"No se pudo completar la descarga tras {retries} intentos.")


def expected_total(response: requests.Response, resume_from: int) -> int:
    content_range = response.headers.get("content-range", "")
    if "/" in content_range:
        return int(content_range.rsplit("/", 1)[1])
    content_length = int(response.headers.get("content-length", 0))
    return resume_from + content_length if resume_from else content_length


def print_progress(downloaded: int, total: int, attempt: int, retries: int) -> None:
    if total:
        percent = downloaded * 100 / total
        print(
            f"\rDescargando: {percent:6.2f}% "
            f"({format_bytes(downloaded)} / {format_bytes(total)}) "
            f"[intento {attempt}/{retries}]",
            end="",
            flush=True,
        )
    else:
        print(
            f"\rDescargando: {format_bytes(downloaded)} [intento {attempt}/{retries}]",
            end="",
            flush=True,
        )


def list_zip_contents(path: Path) -> None:
    print("\nContenido del ZIP:")
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist()[:200]:
            print(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga PadChest-GR desde el enlace publico de B2Drop."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"URL de descarga. Por defecto: {DEFAULT_URL}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Ruta del archivo de salida. Por defecto: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Numero de reintentos. Por defecto: {DEFAULT_RETRIES}",
    )
    parser.add_argument(
        "--list-zip",
        action="store_true",
        help="Lista los primeros ficheros dentro del ZIP al terminar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download(args.url, Path(args.output), args.retries, args.list_zip)


if __name__ == "__main__":
    main()
