#!/usr/bin/env python3
"""Run IDMesh topology workloads in fresh processes and summarize their results."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

CASES = [
    ("2d-mixed-192", ["192", "5", "--dim", "2", "--shape", "mixed"]),
    ("3d-tet-32", ["32", "5", "--dim", "3", "--shape", "simplex"]),
    ("3d-hex-48", ["48", "5", "--dim", "3", "--shape", "tensor"]),
    (
        "3d-mixed-40-scrambled",
        [
            "40", "5", "--dim", "3", "--shape", "mixed",
            "--scramble", "--seed", "2026", "--jitter", "0.10",
        ],
    ),
]
LARGE_CASES = [
    ("3d-tet-48", ["48", "3", "--dim", "3", "--shape", "simplex"]),
    ("3d-hex-64", ["64", "3", "--dim", "3", "--shape", "tensor"]),
    (
        "3d-mixed-56-scrambled",
        [
            "56", "3", "--dim", "3", "--shape", "mixed",
            "--scramble", "--seed", "2026", "--jitter", "0.10",
        ],
    ),
]
SCALE_CASES = [
    ("2d-mixed-128", ["128", "32", "--dim", "2", "--shape", "mixed"]),
    ("2d-mixed-256", ["256", "32", "--dim", "2", "--shape", "mixed"]),
    ("3d-tet-16", ["16", "32", "--dim", "3", "--shape", "simplex"]),
    ("3d-tet-24", ["24", "32", "--dim", "3", "--shape", "simplex"]),
    ("3d-tet-32", ["32", "32", "--dim", "3", "--shape", "simplex"]),
    ("3d-hex-16", ["16", "32", "--dim", "3", "--shape", "tensor"]),
    ("3d-hex-28", ["28", "32", "--dim", "3", "--shape", "tensor"]),
    ("3d-hex-40", ["40", "32", "--dim", "3", "--shape", "tensor"]),
    (
        "3d-mixed-16-scrambled",
        ["16", "32", "--dim", "3", "--shape", "mixed",
         "--scramble", "--seed", "2026", "--jitter", "0.10"],
    ),
    (
        "3d-mixed-28-scrambled",
        ["28", "32", "--dim", "3", "--shape", "mixed",
         "--scramble", "--seed", "2026", "--jitter", "0.10"],
    ),
    (
        "3d-mixed-40-scrambled",
        ["40", "32", "--dim", "3", "--shape", "mixed",
         "--scramble", "--seed", "2026", "--jitter", "0.10"],
    ),
]
CONFIGS = ("minimal", "finite_volume", "full")


def parse_rows(text: str) -> list[dict[str, str]]:
    rows = list(csv.DictReader(line for line in text.splitlines() if line.strip()))
    if not rows:
        raise RuntimeError("expected at least one benchmark CSV row")
    return rows


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise RuntimeError("missing benchmark records")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def median(rows: list[dict[str, str]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def summarize(records: list[dict[str, str]], output: Path, repeats: int) -> None:
    # Each invocation produces one line per available relation. Count memory
    # and finalization once per process, not once per reported relation.
    processes: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    walks: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in records:
        group = (row["case"], row["configuration"])
        processes[group].setdefault(row["process_repeat"], row)
        walks[group + (row["relation"],)].append(row)

    layouts: list[dict] = []
    for (case, config), repeat_rows in sorted(processes.items()):
        if len(repeat_rows) != repeats:
            raise RuntimeError(f"incomplete independent runs: {case}/{config}")
        rows = list(repeat_rows.values())
        bytes_per_cell = median(rows, "rss_delta_bytes_per_cell_p50")
        layouts.append({
            "case": case,
            "configuration": config,
            "dimension": rows[0]["dimension"],
            "shape": rows[0]["shape"],
            "cells": rows[0]["cells"],
            "repeats": len(rows),
            "visible_payload_mib": round(median(rows, "visible_payload_bytes") / (1024 ** 2), 3),
            "rss_delta_mib": round(median(rows, "rss_delta_bytes_p50") / (1024 ** 2), 3),
            "peak_delta_mib": round(median(rows, "peak_delta_bytes_p50") / (1024 ** 2), 3),
            "rss_bytes_per_cell": round(bytes_per_cell, 2),
            "input_ms": round(median(rows, "input_ms_p50"), 3),
            "finalize_ms": round(median(rows, "finalize_ms_p50"), 3),
            "validate_ms": round(median(rows, "validate_ms_p50"), 3),
        })

    full = {row["case"]: row for row in layouts if row["configuration"] == "full"}
    for row in layouts:
        corresponding = full.get(row["case"])
        row["payload_vs_full"] = (
            round(row["visible_payload_mib"] / corresponding["visible_payload_mib"], 3)
            if corresponding and corresponding["visible_payload_mib"] else ""
        )
        row["rss_vs_full"] = (
            round(row["rss_delta_mib"] / corresponding["rss_delta_mib"], 3)
            if corresponding and corresponding["rss_delta_mib"] else ""
        )
        row["finalize_vs_full"] = (
            round(row["finalize_ms"] / corresponding["finalize_ms"], 3)
            if corresponding and corresponding["finalize_ms"] else ""
        )

    queries: list[dict] = []
    for (case, config, relation), rows in sorted(walks.items()):
        if len(rows) != repeats:
            raise RuntimeError(f"incomplete traversal runs: {case}/{config}/{relation}")
        throughput = [
            int(row["incidences_per_round"]) * int(row["rounds"])
            / (float(row["traversal_ms_p50"]) * 1000.0)
            for row in rows if float(row["traversal_ms_p50"]) > 0
        ]
        if not throughput:
            raise RuntimeError(f"zero traversal time: {case}/{config}/{relation}")
        queries.append({
            "case": case,
            "configuration": config,
            "relation": relation,
            "cells": rows[0]["cells"],
            "incidences_per_round": rows[0]["incidences_per_round"],
            "rounds": rows[0]["rounds"],
            "repeats": len(rows),
            "traversal_ms": round(median(rows, "traversal_ms_p50"), 3),
            "million_incidences_per_second": round(statistics.median(throughput), 3),
        })
    write_csv(output / "layout_summary.csv", layouts)
    write_csv(output / "traversal_summary.csv", queries)

    with (output / "report.md").open("w") as stream:
        stream.write("# IDMesh exploratory topology measurements\n\n")
        stream.write("Independent-process medians; see raw.csv and metadata.json. "
                     "Shared-runner timings are diagnostic, not a performance guarantee. "
                     "RSS is process-wide, not heap accounting; payload is a logical lower bound.\n\n")
        stream.write("| Case | Layout | Cells | Payload MiB | RSS delta MiB | "
                     "Peak delta MiB | Finalize ms | Finalize/full | RSS/full |\n")
        stream.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in layouts:
            stream.write(
                f"| {row['case']} | {row['configuration']} | {row['cells']} | "
                f"{row['visible_payload_mib']} | {row['rss_delta_mib']} | "
                f"{row['peak_delta_mib']} | {row['finalize_ms']} | "
                f"{row['finalize_vs_full']} | {row['rss_vs_full']} |\n")
        stream.write("\n## Topology traversal\n\n")
        stream.write("| Case | Layout | Relation | Million incidences/s | Traversal ms |\n")
        stream.write("| --- | --- | --- | ---: | ---: |\n")
        for row in queries:
            stream.write(
                f"| {row['case']} | {row['configuration']} | {row['relation']} | "
                f"{row['million_incidences_per_second']} | {row['traversal_ms']} |\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--scale", action="store_true",
                        help="2D/3D size sweep for layout, RSS and traversal scaling")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.scale and args.large:
        parser.error("--scale and --large cannot be combined")
    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"benchmark executable not found: {executable}")

    args.output.mkdir(parents=True, exist_ok=True)
    cases = SCALE_CASES if args.scale else CASES + (LARGE_CASES if args.large else [])
    records: list[dict[str, str]] = []
    commands: list[list[str]] = []

    for case_name, case_args in cases:
        for configuration in CONFIGS:
            for repeat in range(args.repeats):
                command = [
                    str(executable), *case_args, "--config", configuration,
                    "--trials", "1",
                ]
                commands.append(command)
                process = subprocess.run(command, check=True, text=True, capture_output=True)
                rows = parse_rows(process.stdout)
                for row in rows:
                    records.append({
                        "case": case_name,
                        "process_repeat": str(repeat),
                        **row,
                    })
            print(f"{case_name}/{configuration}: {args.repeats} fresh processes", flush=True)

    write_csv(args.output / "raw.csv", records)
    summarize(records, args.output, args.repeats)

    cpu_model = ""
    if sys.platform == "linux" and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    metadata = {
        "platform": platform.platform(),
        "processor": cpu_model or platform.processor(),
        "python": sys.version,
        "executable": str(executable),
        "commit": os.environ.get("GITHUB_SHA"),
        "repeats": args.repeats,
        "profile": "scale" if args.scale else ("large" if args.large else "baseline"),
        "commands": commands,
        "memory": "fresh-process current/peak RSS; values include allocator retained pages",
        "timing": "steady_clock, sequential processes, no machine-load isolation",
        "environment": {
            key: os.environ[key]
            for key in ("OMP_NUM_THREADS", "KMP_AFFINITY", "GOMP_CPU_AFFINITY")
            if key in os.environ
        },
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
