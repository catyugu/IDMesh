#!/usr/bin/env python3
"""Fetch pinned external mesh fixtures without putting binaries in Git."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "meshes.json"
CACHE = ROOT / "data"


def download(mesh):
    destination = CACHE / (mesh["id"] + (".zip" if mesh.get("storage") == "https-zip" else ".msh"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mesh.get("storage") == "https-zip":
        url = mesh["url"]
    elif mesh.get("storage") == "git-lfs":
        url = ("https://media.githubusercontent.com/media/"
               + mesh["source"].removeprefix("https://github.com/")
               + "/" + mesh["revision"] + "/" + mesh["path"])
    else:
        url = ("https://raw.githubusercontent.com/"
               + mesh["source"].removeprefix("https://github.com/")
               + "/" + mesh["revision"] + "/" + mesh["path"])
    def verify(path):
        data = path.read_bytes()
        if len(data) != mesh["bytes"]:
            raise ValueError(f"{mesh['id']}: size mismatch")
        if "sha256" in mesh and hashlib.sha256(data).hexdigest() != mesh["sha256"]:
            raise ValueError(f"{mesh['id']}: sha256 mismatch")
        if "git_blob_sha1" in mesh:
            digest = hashlib.sha1(b"blob " + str(len(data)).encode()
                                  + b"\0" + data).hexdigest()
            if digest != mesh["git_blob_sha1"]:
                raise ValueError(f"{mesh['id']}: Git blob hash mismatch")
        if data.startswith(b"version https://git-lfs"):
            raise ValueError(f"{mesh['id']}: downloaded LFS pointer instead of data")
        return path
    if destination.is_file():
        try:
            return verify(destination)
        except ValueError:
            destination.unlink()
    temporary = destination.with_suffix(".download")
    try:
        with urllib.request.urlopen(url, timeout=120) as stream, temporary.open("wb") as out:
            while chunk := stream.read(1024 * 1024):
                out.write(chunk)
        verify(temporary)
        temporary.replace(destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["ci", "manual", "candidate", "all"], default="ci")
    parser.add_argument("--id", action="append", help="Download only named dataset(s)")
    args = parser.parse_args()
    meshes = json.loads(MANIFEST.read_text(encoding="utf-8"))["meshes"]
    selection = [m for m in meshes if (args.tier == "all" or m["tier"] == args.tier)
                 and (not args.id or m["id"] in args.id)]
    if args.id and set(args.id) != {m["id"] for m in selection}:
        raise SystemExit("Unknown mesh id or tier mismatch")
    for mesh in selection:
        print(download(mesh), flush=True)


if __name__ == "__main__":
    main()
