#!/usr/bin/env bash
# Copyright (c) 2026 JoSIM Developers
# This file is part of JoSIM. Details are in the LICENSE file.
#
# Layer 3A Helper Script: Run Cadence Spectre with JoSIM Co-Simulation
# Works on BNL cluster ASIC hosts (asic1..asic10, pmos) and local Linux workstations.
#
# Usage:
#   ./run_spectre_cosim.sh <spectre_deck.scs> [additional spectre args...]

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <spectre_deck.scs> [spectre args...]" >&2
    exit 1
fi

SCS_DECK="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 1. Ensure libjosim_spectre_dpi.so is available in the working directory
if [ ! -f ./libjosim_spectre_dpi.so ]; then
    if [ -f "${REPO_ROOT}/build/libjosim_spectre_dpi.so" ] && [ -z "${JOSIM_COSIM_SERVER_BIN:-}" ]; then
        cp -f "${REPO_ROOT}/build/libjosim_spectre_dpi.so" ./libjosim_spectre_dpi.so
    elif [ -f "/tmp/josim-build/libjosim_spectre_dpi.so" ] && [ -z "${JOSIM_COSIM_SERVER_BIN:-}" ]; then
        cp -f "/tmp/josim-build/libjosim_spectre_dpi.so" ./libjosim_spectre_dpi.so
    else
        echo "[run_spectre_cosim] Compiling standalone socket-mode libjosim_spectre_dpi.so..."
        gcc -shared -fPIC -O2 -DJOSIM_DPI_SOCKET_ONLY \
            -I"${REPO_ROOT}/include" \
            "${SCRIPT_DIR}/josim_cosim_dpi.c" \
            -o ./libjosim_spectre_dpi.so -lm
    fi
fi

if [ ! -f ./josim_cosim_port.va ]; then
    cp -f "${SCRIPT_DIR}/josim_cosim_port.va" ./josim_cosim_port.va
fi

# 2. Optionally launch josim-cosim-server if JOSIM_COSIM_SERVER_BIN is provided
SERVER_PID=""
cleanup() {
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
    if [ -n "${JOSIM_SOCK_PATH:-}" ] && [ -S "${JOSIM_SOCK_PATH}" ]; then
        rm -f "${JOSIM_SOCK_PATH}"
    fi
}
trap cleanup EXIT

if [ -n "${JOSIM_COSIM_SERVER_BIN:-}" ] && [ -z "${JOSIM_COSIM_ENDPOINT:-}" ]; then
    export JOSIM_SOCK_PATH="/tmp/josim_spectre_$$.sock"
    export JOSIM_COSIM_ENDPOINT="unix:${JOSIM_SOCK_PATH}"
    SERVER_DIR="$(cd "$(dirname "${JOSIM_COSIM_SERVER_BIN}")" && pwd)"
    BUNDLE_LOADER="${HOME}/local/josim-2.6.5/lib/ld-linux-x86-64.so.2"
    if "${JOSIM_COSIM_SERVER_BIN}" --help >/dev/null 2>&1; then
        LD_LIBRARY_PATH="${SERVER_DIR}:${LD_LIBRARY_PATH:-}" \
            "${JOSIM_COSIM_SERVER_BIN}" --unix "${JOSIM_SOCK_PATH}" >/dev/null 2>&1 &
    elif [ -x "${BUNDLE_LOADER}" ]; then
        "${BUNDLE_LOADER}" --library-path "${SERVER_DIR}:${HOME}/local/josim-2.6.5/lib" \
            "${JOSIM_COSIM_SERVER_BIN}" --unix "${JOSIM_SOCK_PATH}" >/dev/null 2>&1 &
    else
        "${JOSIM_COSIM_SERVER_BIN}" --unix "${JOSIM_SOCK_PATH}" >/dev/null 2>&1 &
    fi
    SERVER_PID=$!
    for _ in $(seq 1 100); do
        [ -S "${JOSIM_SOCK_PATH}" ] && break
        sleep 0.05
    done
fi

# 3. Run Spectre (via BNL cluster asic_tools.csh if present, else direct spectre on PATH)
ASIC_TOOLS="${ASIC_TOOLS:-$HOME/Documents/LDRD_2025/dig_flows/resources/asic_tools.csh}"
if [ -x "${ASIC_TOOLS}" ]; then
    "${ASIC_TOOLS}" spectre -64 "${SCS_DECK}" "$@"
else
    spectre -64 "${SCS_DECK}" "$@"
fi
