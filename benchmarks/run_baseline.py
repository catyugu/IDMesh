#!/usr/bin/env python3
"""Run IDMesh topology benchmarks in fresh processes and aggregate raw CSV rows."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

CASES = [
    ("2d-mixed-192", ["192", "5", "--dim", "2", "--shape", "mixed"]),
    ("3d-tet-32", ["32", "5", "--dim", "3", "--shape", "simplex"]),
    ("3d-hex-48", ["48", "5", "--dim", "3", "--shape", "tensor"]),
    (
        "3d-mixed-40-scrambled",
        [
            "40",
            "5",
            "--dim",
            "3",
            "--shape",
            "mixed",
            "--scramble",
            "--seed",
            "2026",
            "--jitter",
            "0.10",
        ],
    ),
]
LARGE_CASES = [
    ("3d-tet-48", ["48", "3", "--dim", "3", "--shape", "simplex"]),
    ("3d-hex-64", ["64", "3", "--dim", "3", "--shape", "tensor"]),
    (
        "3d-mixed-56-scrambled",
        [
            "56",
            "3",
            "--dim",
            "3",
            "--shape",
            "mixed",
            "--scramble",
            "--seed",
            "2026",
            "--jitter",
            "0.10",
        ],
    ),
]
CONFIGS = ("minimal", "finite_volume", "full")


def parse_rows(text: str) -> list[dict[str, str]]:
    lines = [line for line in text.splitlines() if line.strip()]
    rows = list(csv.DictReader(lines))
    if not rows:
        raise RuntimeError("expected at least one benchmark CSV row")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--large", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be positive")

    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"benchmark executable not found: {executable}")

    args.output.mkdir(parents=True, exist_ok=True)
    cases = CASES + (LARGE_CASES if args.large else [])
    records: list[dict[str, str]] = []
    commands: list[list[str]] = []

    for case_name, case_args in cases:
        for configuration in CONFIGS:
            for repeat in range(args.repeats):
                command = [
                    str(executable),
                    *case_args,
                    "--config",
                    configuration,
                    "--trials",
                    "1",
                ]
                commands.append(command)
                process = subprocess.run(
                    command, check=True, text=True, capture_output=True
                )
                for row in parse_rows(process.stdout):
                    records.append(
                        {
                            "case": case_name,
                            "process_repeat": str(repeat),
                            **row,
                        }
                    )

    if not records:
        return 0

    raw_path = args.output / "raw.csv"
    with raw_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    metadata = {
        "platform": platform.platform(),
        "python": sys.version,
        "executable": str(executable),
        "repeats": args.repeats,
        "large": args.large,
        "commands": commands,
        "environment": {
            key: os.environ[key]
            for key in ("OMP_NUM_THREADS", "KMP_AFFINITY", "GOMP_CPU_AFFINITY")
            if key in os.environ
        },
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(raw_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
