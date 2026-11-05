#!/usr/bin/env bash
# Verify repository-wide C++ formatting without modifying the working tree.
set -euo pipefail

formatter="${CLANG_FORMAT:-clang-format}"
if ! command -v "$formatter" >/dev/null 2>&1; then
    echo "clang-format is required (recommended: clang-format 19.1.7)." >&2
    exit 1
fi

while IFS= read -r -d '' file; do
    "$formatter" --dry-run --Werror --style=file "$file"
done < <(git ls-files -z -- '*.c' '*.cc' '*.cpp' '*.cxx' '*.h' '*.hh' '*.hpp' '*.hxx')
