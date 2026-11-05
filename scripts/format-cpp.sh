#!/usr/bin/env bash
# Normalize all tracked C and C++ source files with the repository style.
set -euo pipefail

formatter="${CLANG_FORMAT:-clang-format}"
if ! command -v "$formatter" >/dev/null 2>&1; then
    echo "clang-format is required (recommended: clang-format 19.1.7)." >&2
    exit 1
fi

while IFS= read -r -d '' file; do
    "$formatter" -i --style=file "$file"
done < <(git ls-files -z -- '*.c' '*.cc' '*.cpp' '*.cxx' '*.h' '*.hh' '*.hpp' '*.hxx')
