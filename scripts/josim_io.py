#!/usr/bin/env python3
"""Fast loader for JoSIM output files (.bin, .npy, .rawb, .csv).

Binary format (.bin) layout:
  - 8B magic: b"JOSIMBIN"
  - uint32 version (1)
  - uint32 flags (0 = little-endian IEEE-754 float64, column-major)
  - uint64 num_cols
  - uint64 num_rows
  - For each column in 0..num_cols-1:
      - uint32 name_len
      - char[name_len] UTF-8 trace name
      - char trace_type ('T', 'V', 'P', 'I')
  - Contiguous column-major float64 payload:
      - float64[num_cols, num_rows]
"""

from __future__ import annotations

import mmap
import struct
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np


def load_josim_bin(
    path: Union[str, Path],
    strip_quotes: bool = True,
    use_mmap: bool = False,
) -> Dict[str, np.ndarray]:
    """Load a JoSIM .bin column-major binary file into a dict of 1D NumPy arrays."""
    p = Path(path)
    raw = p.read_bytes() if not use_mmap else None
    fObj = open(p, "rb") if use_mmap else None
    buf = mmap.mmap(fObj.fileno(), 0, access=mmap.ACCESS_READ) if use_mmap else raw
    try:
        if len(buf) < 32 or buf[:8] != b"JOSIMBIN":
            raise ValueError(f"{p}: invalid JOSIMBIN header")
        version, _flags, num_cols, num_rows = struct.unpack_from("<IIQQ", buf, 8)
        if version != 1:
            raise ValueError(f"{p}: unsupported JOSIMBIN version {version}")
        offset = 32
        names = []
        for _ in range(num_cols):
            (name_len,) = struct.unpack_from("<I", buf, offset)
            offset += 4
            name = bytes(buf[offset : offset + name_len]).decode("utf-8")
            if strip_quotes and len(name) >= 2 and name[0] == '"' and name[-1] == '"':
                name = name[1:-1]
            offset += name_len + 1  # +1 for trace type byte
            names.append(name)

        data = np.frombuffer(buf, dtype="<f8", count=num_cols * num_rows, offset=offset)
        if not use_mmap:
            data = data.copy()
        cols = data.reshape((num_cols, num_rows))
        return {name: cols[idx] for idx, name in enumerate(names)}
    finally:
        if fObj is not None:
            fObj.close()


def load_josim_npy(path: Union[str, Path], strip_quotes: bool = True) -> Dict[str, np.ndarray]:
    """Load a JoSIM .npy structured array file into a dict of 1D NumPy arrays."""
    arr = np.load(Path(path), allow_pickle=False, max_header_size=10_000_000)
    out: Dict[str, np.ndarray] = {}
    for name in arr.dtype.names or ():
        key = name[1:-1] if (strip_quotes and len(name) >= 2 and name[0] == '"' and name[-1] == '"') else name
        out[key] = arr[name]
    return out


def load_josim(path: Union[str, Path], strip_quotes: bool = True) -> Dict[str, np.ndarray]:
    """Auto-detect and load any JoSIM output file (.bin, .npy, .csv)."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".bin":
        return load_josim_bin(p, strip_quotes=strip_quotes)
    if ext == ".npy":
        return load_josim_npy(p, strip_quotes=strip_quotes)
    if ext in (".csv", ".dat"):
        delim = "," if ext == ".csv" else None
        with open(p, "r", encoding="utf-8") as f:
            header = f.readline().strip()
        names = [h.strip() for h in (header.split(",") if delim == "," else header.split())]
        if strip_quotes:
            names = [n[1:-1] if (len(n) >= 2 and n[0] == '"' and n[-1] == '"') else n for n in names]
        mat = np.loadtxt(p, delimiter=delim, skiprows=1)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        return {name: mat[:, idx] for idx, name in enumerate(names)}
    raise ValueError(f"Unsupported JoSIM file extension: {ext}")
