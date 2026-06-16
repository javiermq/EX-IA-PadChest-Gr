#!/usr/bin/env python3
"""
Download the public PadChest-GR archive from B2Drop/Nextcloud.

Usage:
    python download_padchest_gr.py
    python download_padchest_gr.py --output PadChest-GR.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests


DEFAULT_URL = "https://b2drop.bsc.es/nextcloud/s/PadChest-GR/download"
DEFAULT_OUTPUT = "PadChest-GR.zip"
CHUNK_SIZE = 1024 * 1024


def format_bytes(size: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def download(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue

                file.write(chunk)
                downloaded += len(chunk)

                if total:
                    percent = downloaded * 100 / total
                    print(
                        f"\rDescargando: {percent:6.2f}% "
                        f"({format_bytes(downloaded)} / {format_bytes(total)})",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\rDescargando: {format_bytes(downloaded)}",
                        end="",
                        flush=True,
                    )

    print(f"\nArchivo guardado en: {output_path.resolve()}")


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download(args.url, Path(args.output))


if __name__ == "__main__":
    main()
