#!/usr/bin/env python3
"""
josim_cosim.py - JoSIM / SPICE Co-Simulation Coordinator & IPC Client (Layers 1 & 2)

Provides:
  1. CosimSocketClient & CosimServerProcess:
     Binary-frame IPC client and process manager for `josim-cosim-server`
     over AF_UNIX (`unix:/path`) and AF_INET (`tcp:host:port`, TCP_NODELAY) sockets.
  2. run_ngspice_cosim:
     Executes real `ngspice` with the native `josim_cosim.cm` XSPICE code model
     in either Step 1 (in-process `libjosim_c.so`) or Step 2 (`josim-cosim-server`
     UNIX/TCP socket) mode.
  3. parse_ltspice_raw, run_ltspice_branin_cosim, run_ltspice_pwl_cosim:
     Executes real ADI `LTspice.exe -b` in batch mode with double-precision binary
     `.raw` parsing (`.options numdgt=7`), coupled to JoSIM (in-process or IPC server)
     via Branin wave-relaxation (`PORT_WAVE_BRANIN`) or iterative Norton-PWL relaxation.
"""

from __future__ import annotations

import csv
import os
import shutil
import socket
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyjosim import (
    PORT_CURRENT_IN_VOLTAGE_OUT,
    PORT_VOLTAGE_IN_CURRENT_OUT,
    PORT_WAVE_BRANIN,
    Session,
)

# ---------------------------------------------------------------------------
# Binary IPC Wire Protocol Constants (matching include/JoSIM/josim_cosim_ipc.h)
# ---------------------------------------------------------------------------
JOSIM_IPC_MAGIC = 0x4A43534D  # "JCSM"

JOSIM_IPC_OP_INIT_FILE = 1
JOSIM_IPC_OP_INIT_STRING = 2
JOSIM_IPC_OP_STEP_TO = 3
JOSIM_IPC_OP_COMMIT = 4
JOSIM_IPC_OP_ROLLBACK = 5
JOSIM_IPC_OP_RUN_WIN = 6
JOSIM_IPC_OP_FINALIZE = 7
JOSIM_IPC_OP_GET_TRACE = 8
JOSIM_IPC_OP_GET_EVENTS = 9
JOSIM_IPC_OP_SHUTDOWN = 99

_REQ_HDR_STRUCT = struct.Struct("<IHHQ")  # magic, opcode, flags, payload_len (16 bytes)
_RESP_HDR_STRUCT = struct.Struct("<IiQ")  # magic, status, payload_len (16 bytes)


@dataclass
class PortSpec:
    pos_node: str
    neg_node: str = "0"
    mode: int = PORT_VOLTAGE_IN_CURRENT_OUT
    z0: float = 50.0


def find_cosim_server_binary(explicit_path: Optional[Union[str, Path]] = None) -> str:
    if explicit_path and Path(explicit_path).exists():
        return str(Path(explicit_path).resolve())
    env_bin = os.environ.get("JOSIM_COSIM_SERVER_BIN")
    if env_bin and Path(env_bin).exists():
        return str(Path(env_bin).resolve())
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        Path("/tmp/josim-build/josim-cosim-server"),
        repo_root / "build" / "josim-cosim-server",
        repo_root / "build" / "Release" / "josim-cosim-server.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    raise FileNotFoundError("Could not locate `josim-cosim-server` binary.")


def find_xspice_cm_binary(explicit_path: Optional[Union[str, Path]] = None) -> str:
    if explicit_path and Path(explicit_path).exists():
        return str(Path(explicit_path).resolve())
    env_cm = os.environ.get("JOSIM_XSPICE_CM")
    if env_cm and Path(env_cm).exists():
        return str(Path(env_cm).resolve())
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        Path("/tmp/josim-build/josim_cosim.cm"),
        repo_root / "build" / "josim_cosim.cm",
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    raise FileNotFoundError("Could not locate `josim_cosim.cm` XSPICE code model.")


def find_ngspice_binary() -> str:
    env_ng = os.environ.get("NGSPICE_BIN")
    if env_ng and Path(env_ng).exists():
        return env_ng
    for p in ["/usr/local/bin/ngspice", "/usr/bin/ngspice"]:
        if Path(p).exists():
            return p
    which_ng = shutil.which("ngspice")
    if which_ng:
        return which_ng
    raise FileNotFoundError("Could not locate `ngspice` binary.")


def find_ltspice_binary() -> str:
    env_lt = os.environ.get("LTSPICE_BIN")
    if env_lt and Path(env_lt).exists():
        return env_lt
    candidates = [
        "/mnt/c/Users/super/AppData/Local/Programs/ADI/LTspice/LTspice.exe",
        r"C:\Users\super\AppData\Local\Programs\ADI\LTspice\LTspice.exe",
        "/mnt/c/Program Files/ADI/LTspice/LTspice.exe",
        "/mnt/c/Program Files/LTC/LTspiceXVII/XVIIx64.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise FileNotFoundError("Could not locate `LTspice.exe` binary.")


# ---------------------------------------------------------------------------
# Layer 2: Binary IPC Socket Client & Server Process Manager
# ---------------------------------------------------------------------------
class CosimSocketClient:
    """Client for the `josim-cosim-server` binary IPC socket protocol (`JCSM` v1)."""

    def __init__(self, addr: str, timeout: float = 30.0):
        self.addr = addr
        self.sock: Optional[socket.socket] = None
        self.tstep: float = 0.0
        self.num_ports: int = 0
        self.current_time: float = 0.0
        self.rollback_count: int = 0
        self._connect(addr, timeout)

    def _connect(self, addr: str, timeout: float) -> None:
        if addr.startswith("unix:"):
            path = addr[5:]
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect(path)
            self.sock = s
        else:
            host_port = addr[4:] if addr.startswith("tcp:") else addr
            if ":" in host_port:
                host, port_s = host_port.rsplit(":", 1)
                port = int(port_s)
            else:
                host, port = "127.0.0.1", int(host_port)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(timeout)
            s.connect((host, port))
            self.sock = s

    def _recv_exact(self, n: int) -> bytes:
        assert self.sock is not None
        chunks: List[bytes] = []
        rem = n
        while rem > 0:
            b = self.sock.recv(rem)
            if not b:
                raise ConnectionError("Unexpected EOF from josim-cosim-server")
            chunks.append(b)
            rem -= len(b)
        return b"".join(chunks)

    def _request(self, op: int, payload: bytes = b"") -> bytes:
        assert self.sock is not None
        hdr = _REQ_HDR_STRUCT.pack(JOSIM_IPC_MAGIC, op, 0, len(payload))
        self.sock.sendall(hdr + payload)
        rhdr = self._recv_exact(_RESP_HDR_STRUCT.size)
        magic, status, plen = _RESP_HDR_STRUCT.unpack(rhdr)
        if magic != JOSIM_IPC_MAGIC:
            raise RuntimeError(f"Invalid IPC header magic={magic:#x}")
        rpay = self._recv_exact(plen) if plen > 0 else b""
        if status != 0:
            msg = rpay.decode("utf-8", errors="replace")
            raise RuntimeError(f"josim-cosim-server error (status={status}): {msg}")
        return rpay

    def init_netlist(
        self,
        netlist: str,
        ports: Sequence[PortSpec],
        phase_mode: bool = True,
        is_file: bool = False,
    ) -> Tuple[float, int]:
        nl_bytes = netlist.encode("utf-8")
        parts = [struct.pack("<iiI", 1 if phase_mode else 0, len(ports), len(nl_bytes)), nl_bytes]
        for p in ports:
            pos_b = p.pos_node.encode("utf-8")
            neg_b = p.neg_node.encode("utf-8")
            parts.append(struct.pack("<idHH", int(p.mode), float(p.z0), len(pos_b), len(neg_b)))
            parts.append(pos_b)
            parts.append(neg_b)
        op = JOSIM_IPC_OP_INIT_FILE if is_file else JOSIM_IPC_OP_INIT_STRING
        rpay = self._request(op, b"".join(parts))
        dt, n_p = struct.unpack("<dq", rpay)
        self.tstep = dt
        self.num_ports = int(n_p)
        self.current_time = 0.0
        self.rollback_count = 0
        return dt, int(n_p)

    def cosim_step_to(self, t_target: float, u_in: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(u_in, dtype=np.float64)
        if arr.size != self.num_ports:
            raise ValueError(f"Expected {self.num_ports} port inputs, got {arr.size}")
        payload = struct.pack("<d", float(t_target)) + arr.tobytes()
        rpay = self._request(JOSIM_IPC_OP_STEP_TO, payload)
        n = self.num_ports
        y_out = np.frombuffer(rpay, dtype=np.float64, count=n, offset=0).copy()
        dy_du = np.frombuffer(rpay, dtype=np.float64, count=n, offset=8 * n).copy()
        t_cur, rb_cnt = struct.unpack_from("<dq", rpay, offset=16 * n)
        self.current_time = t_cur
        self.rollback_count = int(rb_cnt)
        return y_out, dy_du

    def cosim_commit(self) -> float:
        rpay = self._request(JOSIM_IPC_OP_COMMIT)
        t_cur, rb_cnt = struct.unpack("<dq", rpay)
        self.current_time = t_cur
        self.rollback_count = int(rb_cnt)
        return self.current_time

    def cosim_rollback(self) -> Tuple[float, int]:
        rpay = self._request(JOSIM_IPC_OP_ROLLBACK)
        t_cur, rb_cnt = struct.unpack("<dq", rpay)
        self.current_time = t_cur
        self.rollback_count = int(rb_cnt)
        return t_cur, int(rb_cnt)

    def cosim_run_window(
        self,
        t_stop: float,
        t_grid: Sequence[float],
        u_grid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        t_arr = np.asarray(t_grid, dtype=np.float64)
        u_arr = np.ascontiguousarray(u_grid, dtype=np.float64)
        n_pts = int(t_arr.size)
        if u_arr.shape != (n_pts, self.num_ports):
            raise ValueError(f"Expected u_grid shape {(n_pts, self.num_ports)}, got {u_arr.shape}")
        payload = struct.pack("<dq", float(t_stop), n_pts) + t_arr.tobytes() + u_arr.tobytes()
        rpay = self._request(JOSIM_IPC_OP_RUN_WIN, payload)
        n_out = struct.unpack_from("<q", rpay, 0)[0]
        offset = 8
        t_out = np.frombuffer(rpay, dtype=np.float64, count=n_out, offset=offset).copy()
        offset += 8 * n_out
        y_out = (
            np.frombuffer(rpay, dtype=np.float64, count=n_out * self.num_ports, offset=offset)
            .reshape((n_out, self.num_ports))
            .copy()
        )
        return t_out, y_out

    def cosim_finalize(self, out_csv: str = "") -> Tuple[int, int, int, int]:
        rpay = self._request(JOSIM_IPC_OP_FINALIZE, out_csv.encode("utf-8"))
        n_tr, n_st, n_ev, rb_cnt = struct.unpack("<qqqq", rpay)
        self.rollback_count = int(rb_cnt)
        return int(n_tr), int(n_st), int(n_ev), int(rb_cnt)

    def shutdown(self) -> None:
        if self.sock is not None:
            try:
                self._request(JOSIM_IPC_OP_SHUTDOWN)
            except Exception:
                pass
            self.close()

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None


class CosimServerProcess:
    """Context manager that launches `josim-cosim-server` over a UNIX or TCP socket."""

    def __init__(
        self,
        transport: str = "unix",
        port: int = 0,
        once: bool = True,
        server_bin: Optional[str] = None,
    ):
        self.transport = transport
        self.port = port
        self.once = once
        self.server_bin = find_cosim_server_binary(server_bin)
        self.proc: Optional[subprocess.Popen] = None
        self.addr: str = ""
        self._sock_path: Optional[str] = None

    def __enter__(self) -> "CosimServerProcess":
        cmd = [self.server_bin]
        if self.once:
            cmd.append("--once")
        if self.transport == "unix":
            fd, spath = tempfile.mkstemp(prefix="josim_cosim_", suffix=".sock")
            os.close(fd)
            os.unlink(spath)
            self._sock_path = spath
            cmd.extend(["--unix", spath])
        else:
            if self.port == 0:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", 0))
                    self.port = s.getsockname()[1]
            cmd.extend(["--port", str(self.port)])

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline().strip()
        if not line.startswith("JOSIM_COSIM_SERVER_READY "):
            stderr_out = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"Failed to start josim-cosim-server: {line} / {stderr_out}")
        self.addr = line.split(" ", 1)[1].strip()
        return self

    def connect(self) -> CosimSocketClient:
        return CosimSocketClient(self.addr)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.proc is not None:
            try:
                self.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
        if self._sock_path and os.path.exists(self._sock_path):
            try:
                os.unlink(self._sock_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# CSV / Event Output Readers
# ---------------------------------------------------------------------------
def read_josim_csv(csv_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    p = Path(csv_path)
    if not p.exists():
        return {}
    with p.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return {}
        rows = [[float(x) for x in row] for row in reader if row]
    if not rows:
        return {h: np.zeros(0, dtype=np.float64) for h in header}
    mat = np.asarray(rows, dtype=np.float64)
    return {h: mat[:, idx].copy() for idx, h in enumerate(header)}


def read_josim_events_csv(csv_path: Union[str, Path]) -> List[Dict[str, Union[str, float, int]]]:
    p = Path(str(csv_path) + ".events.csv")
    if not p.exists():
        return []
    out: List[Dict[str, Union[str, float, int]]] = []
    with p.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
                {
                    "trace": row["trace"],
                    "time": float(row["time"]),
                    "t_peak": float(row["t_peak"]),
                    "v_peak": float(row["v_peak"]),
                    "fwhm": float(row["fwhm"]),
                    "dt_pi": float(row["dt_pi"]),
                    "area": float(row["area"]),
                    "phase": float(row["phase"]),
                    "polarity": int(row["polarity"]),
                    "winding": int(row["winding"]),
                    "trace_id": int(row["trace_id"]),
                    "trial": int(row["trial"]),
                }
            )
    return out


def read_josim_meta(csv_path: Union[str, Path]) -> Dict[str, int]:
    p = Path(str(csv_path) + ".meta.txt")
    if not p.exists():
        return {}
    meta: Dict[str, int] = {}
    for line in p.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            meta[k.strip()] = int(v.strip())
    return meta


# ---------------------------------------------------------------------------
# Real ngspice-45.2 + josim_cosim.cm Runner
# ---------------------------------------------------------------------------
@dataclass
class NgspiceCosimResult:
    ng_time: np.ndarray
    ng_traces: Dict[str, np.ndarray]
    josim_traces: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    josim_events: Dict[str, List[Dict[str, Union[str, float, int]]]] = field(default_factory=dict)
    josim_meta: Dict[str, Dict[str, int]] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""


def parse_ngspice_wrdata(path: Union[str, Path], vec_names: Sequence[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Parse ngspice `wrdata` file where each vector `k` is written as `(time, val)` columns:
    col 0 = time, col 1 = vec_0, col 2 = time, col 3 = vec_1, ...
    """
    mat = np.loadtxt(str(path), dtype=np.float64)
    if mat.ndim == 1:
        mat = mat.reshape((1, -1))
    t = mat[:, 0].copy()
    traces: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(vec_names):
        col = 2 * idx + 1
        if col < mat.shape[1]:
            traces[name] = mat[:, col].copy()
    return t, traces


def run_ngspice_deck(
    spice_deck: str,
    wrdata_vecs: Sequence[str],
    tran_cmd: str = "tran 0.2p 60p",
    josim_csv_map: Optional[Dict[str, str]] = None,
    cm_path: Optional[str] = None,
) -> NgspiceCosimResult:
    """
    Execute real `ngspice -b` with `pre_codemodel <josim_cosim.cm>` on `spice_deck`.
    """
    ng_bin = find_ngspice_binary()
    cm_bin = find_xspice_cm_binary(cm_path)

    with tempfile.TemporaryDirectory(prefix="ngspice_cosim_") as tmpd:
        tmp_dir = Path(tmpd)
        wr_file = tmp_dir / "ng_out.txt"
        vecs_str = " ".join(wrdata_vecs)

        # Inject .control block with pre_codemodel before .end
        lines = [ln for ln in spice_deck.splitlines() if ln.strip().lower() != ".end"]
        control_block = [
            ".control",
            f"pre_codemodel {cm_bin}",
            "set wr_singlescale",
            tran_cmd,
            f"wrdata {wr_file.as_posix()} {vecs_str}",
            "quit",
            ".endc",
            ".end",
        ]
        full_deck = "\n".join(lines + control_block) + "\n"
        sp_file = tmp_dir / "deck.sp"
        sp_file.write_text(full_deck)

        res = subprocess.run(
            [ng_bin, "-b", str(sp_file)],
            cwd=str(tmp_dir),
            capture_output=True,
            text=True,
        )
        if res.returncode != 0 or not wr_file.exists():
            raise RuntimeError(
                f"ngspice failed (rc={res.returncode}):\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )

        mat = np.loadtxt(str(wr_file), dtype=np.float64)
        if mat.ndim == 1:
            mat = mat.reshape((1, -1))
        ng_time = mat[:, 0].copy()
        ng_traces: Dict[str, np.ndarray] = {}
        # With `set wr_singlescale`, column 0 is time and columns 1..N are the vectors
        if mat.shape[1] == 1 + len(wrdata_vecs):
            for idx, nm in enumerate(wrdata_vecs):
                ng_traces[nm] = mat[:, idx + 1].copy()
        else:
            ng_time, ng_traces = parse_ngspice_wrdata(wr_file, wrdata_vecs)

        j_traces: Dict[str, Dict[str, np.ndarray]] = {}
        j_events: Dict[str, List[Dict[str, Union[str, float, int]]]] = {}
        j_meta: Dict[str, Dict[str, int]] = {}
        if josim_csv_map:
            for label, csv_p in josim_csv_map.items():
                j_traces[label] = read_josim_csv(csv_p)
                j_events[label] = read_josim_events_csv(csv_p)
                j_meta[label] = read_josim_meta(csv_p)

        return NgspiceCosimResult(
            ng_time=ng_time,
            ng_traces=ng_traces,
            josim_traces=j_traces,
            josim_events=j_events,
            josim_meta=j_meta,
            stdout=res.stdout,
            stderr=res.stderr,
        )


# ---------------------------------------------------------------------------
# Real ADI LTspice.exe Batch Runner & Binary .raw Parser
# ---------------------------------------------------------------------------
def wsl_to_win_path(p: Union[str, Path]) -> str:
    s = str(Path(p).resolve())
    if s.startswith("/mnt/") and len(s) > 6 and s[6] == "/":
        drive = s[5].upper()
        rest = s[6:].replace("/", "\\")
        return f"{drive}:{rest}"
    res = subprocess.run(["wslpath", "-w", s], capture_output=True, text=True)
    if res.returncode == 0:
        return res.stdout.strip()
    return s


def parse_ltspice_raw(raw_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Parse ADI LTspice `.raw` file generated with `.options numdgt=7` (UTF-16LE header
    followed by contiguous row-major float64 binary matrix).
    """
    raw_bytes = Path(raw_path).read_bytes()
    marker = "Binary:\n".encode("utf-16le")
    idx = raw_bytes.find(marker)
    if idx < 0:
        marker_crlf = "Binary:\r\n".encode("utf-16le")
        idx = raw_bytes.find(marker_crlf)
        if idx < 0:
            raise ValueError(f"Could not find Binary marker in LTspice raw file {raw_path}")
        hdr_bytes = raw_bytes[:idx]
        bin_offset = idx + len(marker_crlf)
    else:
        hdr_bytes = raw_bytes[:idx]
        bin_offset = idx + len(marker)

    hdr_text = hdr_bytes.decode("utf-16le", errors="replace")
    n_vars = 0
    n_pts = 0
    var_names: List[str] = []
    in_vars = False
    for line in hdr_text.splitlines():
        ls = line.strip()
        if ls.startswith("No. Variables:"):
            n_vars = int(ls.split(":", 1)[1].strip())
        elif ls.startswith("No. Points:"):
            n_pts = int(ls.split(":", 1)[1].strip())
        elif ls.startswith("Variables:"):
            in_vars = True
        elif in_vars and ls:
            parts = ls.split()
            if len(parts) >= 2 and parts[0].isdigit():
                var_names.append(parts[1].lower())

    if n_vars <= 0 or n_pts <= 0:
        raise ValueError(f"Invalid LTspice header n_vars={n_vars}, n_pts={n_pts}")

    data = np.frombuffer(raw_bytes, dtype=np.float64, count=n_pts * n_vars, offset=bin_offset).reshape(
        (n_pts, n_vars)
    )
    # LTspice negates time points on internal solver breakpoint flags; abs() recovers exact time
    time_vec = np.abs(data[:, 0]).copy()
    traces: Dict[str, np.ndarray] = {}
    for col in range(1, min(n_vars, len(var_names))):
        traces[var_names[col]] = data[:, col].copy()
    return time_vec, traces


def run_ltspice_batch(netlist_str: str, work_dir: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Run ADI `LTspice.exe -b` on `netlist_str` (ensuring `.options numdgt=7` is present)
    and return `(time_vec, traces_dict)`.
    """
    lt_bin = find_ltspice_binary()
    # Place temp files under /mnt/c/dev/JoSIM/build or user-provided Windows-visible path
    base_tmp = Path(work_dir) if work_dir else Path("/mnt/c/dev/JoSIM")
    with tempfile.TemporaryDirectory(prefix=".ltspice_tmp_", dir=str(base_tmp)) as tmpd:
        tmp_path = Path(tmpd)
        lines = [ln for ln in netlist_str.splitlines() if ln.strip().lower() != ".end"]
        if not any("numdgt" in ln.lower() for ln in lines):
            lines.append(".options numdgt=7 plotwinsize=0")
        lines.append(".end")
        cir_path = tmp_path / "lt_cosim.cir"
        raw_path = tmp_path / "lt_cosim.raw"
        cir_path.write_text("\n".join(lines) + "\n", encoding="ascii")

        win_cir = wsl_to_win_path(cir_path)
        res = subprocess.run([lt_bin, "-b", win_cir], capture_output=True)
        if res.returncode != 0 or not raw_path.exists():
            log_path = tmp_path / "lt_cosim.log"
            log_txt = log_path.read_text(errors="replace") if log_path.exists() else ""
            raise RuntimeError(f"LTspice batch run failed (rc={res.returncode}):\n{log_txt}")

        return parse_ltspice_raw(raw_path)


def format_spice_pwl_file(path: Union[str, Path], t: np.ndarray, y: np.ndarray) -> str:
    """Write `(t, y)` pairs to a file suitable for LTspice `PWL file=...`."""
    p = Path(path)
    with p.open("w", encoding="ascii") as f:
        for ti, yi in zip(t, y):
            f.write(f"{ti:.12e} {yi:.12e}\n")
    return wsl_to_win_path(p)
