#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.venvs/smartpix/bin:/opt/venv/bin:$PATH"

BUILD_DIR="${1:-$HOME/.cache/josim-build}"
SRC_DIR="${2:-/mnt/c/dev/JoSIM}"

mkdir -p "$BUILD_DIR"
if [ ! -f "$BUILD_DIR/build.ninja" ]; then
    cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSING_OPENMP=ON -DBUILD_TESTING=ON
fi
cmake --build "$BUILD_DIR" -j"$(nproc)"
ctest --test-dir "$BUILD_DIR" --output-on-failure
