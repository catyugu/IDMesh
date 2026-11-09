#!/usr/bin/env python3
"""Summarize independent Kokkos, MPI halo and shard-startup CSV measurements."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"missing measurements in {path}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mib(value: str) -> float:
    return round(float(value) / (1024 ** 2), 3)


def summarize(folder: Path) -> None:
    kokkos: list[dict] = []
    halos: list[dict] = []
    startup: list[dict] = []

    for path in sorted(folder.glob("kokkos-*.csv")):
        rows = read_csv(path)
        phases = {row["phase"]: row for row in rows}
        if "device_geometry_upload" not in phases:
            raise RuntimeError(f"missing geometry upload: {path}")
        base = phases["device_geometry_upload"]
        rel = phases.get("device_relation_upload")
        kokkos.append({
            "case": path.stem,
            "configuration": base["configuration"],
            "cells": base["cells"],
            "mesh_payload_mib": mib(base["device_mesh_payload_bytes"]),
            "relation_payload_mib": mib(base["device_relation_payload_bytes"]),
            "mirror_rss_delta_mib": mib(base["rss_delta_mirrors_bytes_p50"]),
            "mesh_upload_ms": base["ms_p50"],
            "relation_upload_ms": rel["ms_p50"] if rel else "",
            "centroid_kernel_ms": phases["device_centroid_kernel"]["ms_p50"],
            "geometry_kernel_ms": phases["device_geometry_checksum"]["ms_p50"],
            "relation_kernel_ms": phases["device_incidence_kernel"]["ms_p50"]
            if "device_incidence_kernel" in phases else "",
            "rounds": base["rounds"],
        })

    for path in sorted(folder.glob("mpi-host-*.csv")) + sorted(folder.glob("mpi-device-*.csv")):
        row, = read_csv(path)
        device = path.name.startswith("mpi-device-")
        start_key = "host_staging_start_ms_p50_max" if device else "reused_start_ms_p50_max"
        wait_key = "host_staging_wait_ms_p50_max" if device else "reused_wait_ms_p50_max"
        total_key = "host_staging_total_ms_p50_max" if device else "reused_total_ms_p50_max"
        setup_key = "device_halo_setup_ms_max" if device else "request_setup_ms_max"
        halos.append({
            "case": path.stem,
            "transport": "Kokkos Serial host-staging" if device else "host request",
            "ranks": row["ranks"],
            "cells": row["cells"],
            "components": row["components"],
            "ghost_layers": row["ghost_layers"],
            "send_kib_per_exchange": round(int(row["send_bytes_per_exchange"]) / 1024, 2),
            "receive_kib_per_exchange": round(int(row["receive_bytes_per_exchange"]) / 1024, 2),
            "setup_ms_max": row[setup_key],
            "start_ms_p50_max": row[start_key],
            "wait_ms_p50_max": row[wait_key],
            "total_ms_p50_max": row[total_key],
            "iterations": row["iterations"],
        })

    for path in sorted(folder.glob("startup-*.csv")):
        row, = read_csv(path)
        startup.append({
            "case": path.stem,
            "ranks": row["ranks"],
            "layout": row["layout"],
            "local_cells_max": row["local_cells_max"],
            "local_facets_max": row["local_facets_max"],
            "local_edges_max": row["local_edges_max"],
            "total_ms_max": row["total_ms_max"],
            "parse_ms_max": row["parse_ms_max"],
            "local_mesh_ms_max": row["local_mesh_ms_max"],
            "metadata_ms_max": row["metadata_ms_max"],
            "exchange_plan_ms_max": row["exchange_plan_ms_max"],
            "validate_ms_max": row["validate_ms_max"],
            "peak_rss_delta_mib_max": mib(row["peak_rss_delta_bytes_max"]),
        })

    write_csv(folder / "kokkos_summary.csv", kokkos)
    write_csv(folder / "mpi_halo_summary.csv", halos)
    write_csv(folder / "mpi_startup_summary.csv", startup)
    with (folder / "report.md").open("w") as stream:
        stream.write("# IDMesh exploratory backend measurements\n\n")
        stream.write("Linux/OpenMPI/Kokkos Serial on shared CI CPU. "
                     "Device mirror payloads are logical Kokkos-view bytes, "
                     "not GPU allocation telemetry; RSS is host-process wide. "
                     "Halo timings are median-of-iterations, reduced by the slowest-rank "
                     "median (not a global per-iteration critical path).\n\n")
        stream.write("## Kokkos mirrors\n\n")
        stream.write("| Case | Cells | Device MiB (mesh + relation) | "
                     "Mirror RSS delta MiB | Upload ms (mesh + relation) |\n")
        stream.write("| --- | ---: | ---: | ---: | ---: |\n")
        for row in kokkos:
            rel = float(row["relation_upload_ms"] or 0)
            stream.write(f"| {row['case']} | {row['cells']} | "
                         f"{round(row['mesh_payload_mib'] + row['relation_payload_mib'], 3)} | "
                         f"{row['mirror_rss_delta_mib']} | "
                         f"{round(float(row['mesh_upload_ms']) + rel, 3)} |\n")
        stream.write("\n## MPI halo\n\n")
        stream.write("| Case | Mode | Ranks | Send KiB | Setup ms | Start ms | Wait ms | Total ms |\n")
        stream.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in halos:
            stream.write(f"| {row['case']} | {row['transport']} | {row['ranks']} | "
                         f"{row['send_kib_per_exchange']} | {row['setup_ms_max']} | "
                         f"{row['start_ms_p50_max']} | {row['wait_ms_p50_max']} | "
                         f"{row['total_ms_p50_max']} |\n")
        stream.write("\n## MPI shard startup\n\n")
        stream.write("| Case | Local cells max | Total ms max | Local mesh ms max | "
                     "Exchange plan ms max | Peak RSS delta MiB max |\n")
        stream.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in startup:
            stream.write(f"| {row['case']} | {row['local_cells_max']} | "
                         f"{row['total_ms_max']} | {row['local_mesh_ms_max']} | "
                         f"{row['exchange_plan_ms_max']} | "
                         f"{row['peak_rss_delta_mib_max']} |\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    if not args.folder.is_dir():
        parser.error("backend measurement directory not found")
    summarize(args.folder)
    print(args.folder / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
