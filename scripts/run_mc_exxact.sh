#!/usr/bin/env bash
set -euo pipefail

export PATH="/u/home/smandal/.venvs/josim/bin:/usr/local/cuda/bin:${PATH}"
BUILD_DIR="/u/home/smandal/.cache/josim-gpu-build"
REPO_DIR="/u/home/smandal/JoSIM"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
ninja -j16

mkdir -p /u/home/smandal/.cache/josim-gpu-bench
python3 -u "${REPO_DIR}/scripts/benchmark_mc_exxact.py" 2>&1 | tee /u/home/smandal/.cache/josim-gpu-bench/mc_results.txt
