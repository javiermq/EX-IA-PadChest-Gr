#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or extract the PadChest-GR ZIP archive.")
    parser.add_argument("--zip", type=Path, default=Path("data/PadChest-GR.zip"))
    parser.add_argument("--extract-to", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.zip.exists():
        raise FileNotFoundError(args.zip)

    with zipfile.ZipFile(args.zip) as archive:
        names = archive.namelist()
        print(f"Archivos en ZIP: {len(names)}")
        print("\nPosibles anotaciones tabulares:")
        for name in names:
            lower = name.lower()
            if lower.endswith((".csv", ".tsv", ".json", ".jsonl", ".xlsx")):
                print(name)

        print(f"\nPrimeros {min(args.limit, len(names))} archivos:")
        for name in names[: args.limit]:
            print(name)

        if args.extract_to:
            args.extract_to.mkdir(parents=True, exist_ok=True)
            archive.extractall(args.extract_to)
            print(f"\nExtraido en: {args.extract_to.resolve()}")


if __name__ == "__main__":
    main()
