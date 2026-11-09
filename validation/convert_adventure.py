#!/usr/bin/env python3
"""Convert ADVENTURE's Pantheon TET4 sample into native Gmsh 2.2 binary.

No third-party Python dependencies. The source ZIP is SHA-256 verified by
fetch_meshes.py; neither the original archive nor the derived .msh is tracked.
ADVENTURE sample format: tetrahedron count, 0-based TET4 connectivity,
node count, then XYZ coordinates (no region table for this dataset).
"""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import zipfile


def scalar_line(source, what):
    line = source.readline()
    if not line:
        raise ValueError(f"missing {what}")
    parts = line.split()
    if len(parts) != 1:
        raise ValueError(f"invalid {what}: expected one integer")
    result = int(parts[0])
    if result <= 0:
        raise ValueError(f"invalid {what}: must be positive")
    return result


def convert(archive, destination, expected_cells, expected_nodes, member,
            member_bytes, report=None):
    start = time.perf_counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(archive) as source_zip:
            files = [item for item in source_zip.infolist() if not item.is_dir()]
            if len(files) != 1 or files[0].filename != member:
                raise ValueError("unexpected ADVENTURE archive member")
            if files[0].file_size != member_bytes:
                raise ValueError("ADVENTURE archive member size changed")
            with source_zip.open(files[0]) as source:
                elements = scalar_line(source, "tetrahedron count")
                if elements != expected_cells or elements > 2**31 - 1:
                    raise ValueError(f"unexpected tetrahedron count: {elements}")

                # Array-backed memory: 4 x 32-bit integers per tetrahedron.
                connectivity = array("I")
                for index in range(elements):
                    row = source.readline().split()
                    if len(row) != 4:
                        raise ValueError(f"tetrahedron {index}: expected four node indices")
                    nodes = tuple(int(value) for value in row)
                    if any(node < 0 or node >= expected_nodes for node in nodes):
                        raise ValueError(f"tetrahedron {index}: invalid node index")
                    connectivity.extend(nodes)

                nodes = scalar_line(source, "vertex count")
                if nodes != expected_nodes or nodes > 2**31 - 1:
                    raise ValueError(f"unexpected vertex count: {nodes}")

                with temporary.open("wb") as target:
                    target.write(b"$MeshFormat\n2.2 1 8\n")
                    target.write(struct.pack("<i", 1))
                    target.write(b"\n$EndMeshFormat\n$Nodes\n")
                    target.write(f"{nodes}\n".encode("ascii"))
                    bounds_min = [math.inf, math.inf, math.inf]
                    bounds_max = [-math.inf, -math.inf, -math.inf]
                    for index in range(nodes):
                        row = source.readline().split()
                        if len(row) != 3:
                            raise ValueError(f"vertex {index}: expected three coordinates")
                        xyz = tuple(float(value) for value in row)
                        if not all(math.isfinite(v) for v in xyz):
                            raise ValueError(f"vertex {index}: non-finite coordinate")
                        for axis, value in enumerate(xyz):
                            bounds_min[axis] = min(bounds_min[axis], value)
                            bounds_max[axis] = max(bounds_max[axis], value)
                        target.write(struct.pack("<i3d", index + 1, *xyz))

                    if source.read().strip():
                        raise ValueError("unexpected trailing ADVENTURE records")

                    target.write(b"\n$EndNodes\n$Elements\n")
                    target.write(f"{elements}\n".encode("ascii"))
                    target.write(struct.pack("<iii", 4, elements, 2))
                    for index in range(elements):
                        a, b, c, d = connectivity[index * 4:index * 4 + 4]
                        # Source uses zero-based nodes; no physical groups are
                        # provided, so tag=0 denotes an unassigned material.
                        target.write(struct.pack(
                            "<7i", index + 1, 0, 1, a + 1, b + 1, c + 1, d + 1))
                    target.write(b"\n$EndElements\n")
                del connectivity

        temporary.replace(destination)
        result = {
            "source": "ADVENTURE Pantheon TET4",
            "tetrahedra": elements,
            "vertices": nodes,
            "coordinate_min": bounds_min,
            "coordinate_max": bounds_max,
            "output_format": "Gmsh 2.2 binary little-endian",
            "output_bytes": destination.stat().st_size,
            "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "conversion_seconds": time.perf_counter() - start,
        }
        if report is not None:
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--member", default="pantheon_b60cmc.msh")
    parser.add_argument("--member-bytes", type=int, default=41795315)
    parser.add_argument("--expected-cells", type=int, default=1329027)
    parser.add_argument("--expected-nodes", type=int, default=273157)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    convert(args.archive, args.output, args.expected_cells,
            args.expected_nodes, args.member, args.member_bytes, args.report)


if __name__ == "__main__":
    main()
