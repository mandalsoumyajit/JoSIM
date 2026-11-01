#!/usr/bin/env python3
"""
josim_cosim.py - JoSIM / SPICE Co-Simulation Coordinator & IPC Client

Provides:
  1. CosimSocketClient & CosimServerProcess:
     Binary-frame IPC client and process manager for `josim-cosim-server`
     over AF_UNIX (`unix:/path`) and AF_INET (`tcp:host:port`, TCP_NODELAY) sockets.
  2. run_ngspice_cosim:
     Executes real `ngspice` with the native `josim_cosim.cm` XSPICE code model
     in either in-process (`libjosim_c.so`) or out-of-process (`josim-cosim-server`
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
# Binary IPC Socket Client & Server Process Manager
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

    def get_trace(self, name: str) -> np.ndarray:
        """Fetch a finalized JoSIM trace array by name directly over IPC (`JOSIM_IPC_OP_GET_TRACE`)."""
        rpay = self._request(JOSIM_IPC_OP_GET_TRACE, name.encode("utf-8"))
        return np.frombuffer(rpay, dtype=np.float64).copy()

    def get_events(self) -> List[Dict[str, Union[str, float, int]]]:
        """Fetch all finalized SFQ events directly over IPC (`JOSIM_IPC_OP_GET_EVENTS`)."""
        rpay = self._request(JOSIM_IPC_OP_GET_EVENTS)
        if len(rpay) < 16:
            return []
        n_ev, tbl_len = struct.unpack_from("<QQ", rpay, 0)
        tbl_str = rpay[16 : 16 + tbl_len].decode("utf-8", errors="replace") if tbl_len > 0 else ""
        trace_names = tbl_str.split("\n") if tbl_str else []
        ev_struct = struct.Struct("<dddddddiiii")  # 7 doubles + 4 int32s = 72 bytes (matches josim_sfq_event_t)
        offset = 16 + tbl_len
        out: List[Dict[str, Union[str, float, int]]] = []
        for _ in range(int(n_ev)):
            if offset + ev_struct.size > len(rpay):
                break
            t_ev, t_pk, v_pk, fwhm, dt_pi, area, phase, pol, wind, tid, trial = ev_struct.unpack_from(rpay, offset)
            offset += ev_struct.size
            tname = trace_names[tid] if 0 <= tid < len(trace_names) else f"trace_{tid}"
            out.append(
                {
                    "trace": tname,
                    "time": float(t_ev),
                    "t_peak": float(t_pk),
                    "v_peak": float(v_pk),
                    "fwhm": float(fwhm),
                    "dt_pi": float(dt_pi),
                    "area": float(area),
                    "phase": float(phase),
                    "polarity": int(pol),
                    "winding": int(wind),
                    "trace_id": int(tid),
                    "trial": int(trial),
                }
            )
        return out

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
        expected_clients: int = 1,
        server_bin: Optional[str] = None,
    ):
        self.transport = transport
        self.port = port
        self.once = once
        self.expected_clients = expected_clients
        self.server_bin = find_cosim_server_binary(server_bin)
        self.proc: Optional[subprocess.Popen] = None
        self.addr: str = ""
        self._sock_path: Optional[str] = None

    def __enter__(self) -> "CosimServerProcess":
        cmd = [self.server_bin]
        if self.once:
            cmd.append("--once")
            if self.expected_clients > 1:
                cmd.extend(["--expected-clients", str(self.expected_clients)])
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


def format_inline_pwl(t: np.ndarray, y: np.ndarray) -> str:
    """Format `(t, y)` arrays as an inline SPICE `PWL(...)` expression."""
    pairs = " ".join(f"{ti:.10e} {yi:.10e}" for ti, yi in zip(t, y))
    return f"PWL({pairs})"


# ---------------------------------------------------------------------------
# Unified Co-Simulation Session (In-Process or Socket IPC)
# ---------------------------------------------------------------------------
class UnifiedCosimSession:
    """
    Unified interface wrapping either an in-process `pyjosim.Session`
    or an out-of-process `CosimSocketClient` (UNIX/TCP socket).
    """

    def __init__(
        self,
        netlist: str,
        ports: Sequence[PortSpec],
        phase_mode: bool = True,
        is_file: bool = False,
        endpoint: str = "inproc",
    ):
        self.endpoint = endpoint or "inproc"
        self.ports = list(ports)
        self._sess: Optional[Session] = None
        self._client: Optional[CosimSocketClient] = None
        self._res = None

        if self.endpoint in ("", "inproc"):
            s = Session()
            s.set_cli_args(["-a", "1" if phase_mode else "0"])
            if is_file:
                s.load_file(netlist)
            else:
                s.load_string(netlist)
            for p in self.ports:
                s.cosim_add_port(p.pos_node, p.neg_node, p.mode, p.z0)
            s.cosim_init()
            self._sess = s
            self.tstep = s.cosim_tstep
            self.num_ports = s.cosim_num_ports
        else:
            cli = CosimSocketClient(self.endpoint)
            dt, n_p = cli.init_netlist(netlist, self.ports, phase_mode=phase_mode, is_file=is_file)
            self._client = cli
            self.tstep = dt
            self.num_ports = n_p

    @property
    def current_time(self) -> float:
        if self._sess is not None:
            return self._sess.cosim_current_time
        assert self._client is not None
        return self._client.current_time

    @property
    def rollback_count(self) -> int:
        if self._sess is not None:
            return self._sess.cosim_rollback_count
        assert self._client is not None
        return self._client.rollback_count

    def step_to(self, t_target: float, u_in: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        if self._sess is not None:
            return self._sess.cosim_step_to(t_target, u_in)
        assert self._client is not None
        return self._client.cosim_step_to(t_target, u_in)

    def commit(self) -> float:
        if self._sess is not None:
            self._sess.cosim_commit()
            return self._sess.cosim_current_time
        assert self._client is not None
        return self._client.cosim_commit()

    def rollback(self) -> Tuple[float, int]:
        if self._sess is not None:
            self._sess.cosim_rollback()
            return self._sess.cosim_current_time, self._sess.cosim_rollback_count
        assert self._client is not None
        return self._client.cosim_rollback()

    def run_window(
        self,
        t_stop: float,
        t_grid: Sequence[float],
        u_grid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._sess is not None:
            t_arr = np.asarray(t_grid, dtype=np.float64)
            t_start = float(t_arr[0]) if t_arr.size > 0 else 0.0
            return self._sess.cosim_run_window(t_start, t_stop, t_arr, u_grid)
        assert self._client is not None
        return self._client.cosim_run_window(t_stop, t_grid, u_grid)

    def finalize(self, out_csv: str = "") -> None:
        if self._sess is not None:
            self._res = self._sess.cosim_finalize()
            if out_csv:
                self._sess.write_output(out_csv)
        else:
            assert self._client is not None
            self._client.cosim_finalize(out_csv)

    def get_trace(self, name: str) -> np.ndarray:
        if self._sess is not None:
            if self._res is None:
                self._res = self._sess.cosim_finalize()
            return np.asarray(self._res[name], dtype=np.float64).copy()
        assert self._client is not None
        return self._client.get_trace(name)

    def get_events(self) -> List[Dict[str, Union[str, float, int]]]:
        if self._sess is not None:
            if self._res is None:
                self._res = self._sess.cosim_finalize()
            out: List[Dict[str, Union[str, float, int]]] = []
            ev_arr = self._res.events
            for row in ev_arr:
                tid = int(row["trace_id"])
                tname = self._res.event_traces[tid] if 0 <= tid < len(self._res.event_traces) else f"trace_{tid}"
                out.append(
                    {
                        "trace": tname,
                        "time": float(row["time"]),
                        "t_peak": float(row["t_peak"]),
                        "v_peak": float(row["v_peak"]),
                        "fwhm": float(row["fwhm"]),
                        "dt_pi": float(row["dt_pi"]),
                        "area": float(row["area"]),
                        "phase": float(row["phase"]),
                        "polarity": int(row["polarity"]),
                        "winding": int(row["winding"]),
                        "trace_id": tid,
                        "trial": int(row["trial"]),
                    }
                )
            return out
        assert self._client is not None
        return self._client.get_events()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self._sess = None


# ---------------------------------------------------------------------------
# Multi-Window Branin Wave Relaxation Coordinator (ngspice & LTspice)
# ---------------------------------------------------------------------------
@dataclass
class BraninPortLink:
    """Specification of a transmission-line port coupling SPICE and JoSIM."""
    josim_pos_node: str
    josim_neg_node: str = "0"
    spice_port_trace: str = ""      # e.g. "v(n_sp_port)" in SPICE
    spice_pwl_placeholder: str = "" # e.g. "{PWL_PORT_0}" in the SPICE template
    z0: float = 50.0
    td: float = 20e-12


@dataclass
class BraninCosimResult:
    t_grid: np.ndarray
    spice_traces: Dict[str, np.ndarray]
    a_into_josim: np.ndarray
    b_from_josim: np.ndarray
    num_windows: int
    window_span: float
    josim_events: List[Dict[str, Union[str, float, int]]] = field(default_factory=list)
    josim_traces: Dict[str, np.ndarray] = field(default_factory=dict)


class BraninWaveWindowCoordinator:
    """
    Multi-window causal Branin transmission-line wave relaxation coordinator.
    Advances a hybrid SPICE (`ltspice` or `ngspice`) + JoSIM system across
    windows `[w * W, min((w + 1) * W, t_stop)]` where `W <= min(td_p)`.
    JoSIM steps strictly forward on each window via `cosim_run_window` without
    re-running previous windows.
    """

    def __init__(
        self,
        simulator: str,
        spice_template: str,
        josim_netlist: str,
        links: Sequence[BraninPortLink],
        t_stop: float,
        window_size: Optional[float] = None,
        phase_mode: bool = True,
        josim_is_file: bool = False,
        endpoint: str = "inproc",
        spice_vectors: Optional[Sequence[str]] = None,
        josim_trace_names: Optional[Sequence[str]] = None,
        work_dir: Optional[Union[str, Path]] = None,
    ):
        self.simulator = simulator.lower()
        if self.simulator not in ("ltspice", "ngspice"):
            raise ValueError(f"Unsupported simulator for BraninWaveWindowCoordinator: {simulator}")
        self.spice_template = spice_template
        self.josim_netlist = josim_netlist
        self.links = list(links)
        if not self.links:
            raise ValueError("At least one BraninPortLink is required.")
        self.t_stop = float(t_stop)
        min_td = min(lnk.td for lnk in self.links)
        self.window_size = float(window_size) if window_size is not None else min_td
        if self.window_size > min_td * (1.0 + 1e-6):
            raise ValueError(
                f"Window size ({self.window_size*1e12:.3f} ps) exceeds minimum T-line delay ({min_td*1e12:.3f} ps)"
            )
        self.phase_mode = phase_mode
        self.josim_is_file = josim_is_file
        self.endpoint = endpoint
        self.spice_vectors = list(spice_vectors) if spice_vectors else [lnk.spice_port_trace for lnk in self.links]
        for lnk in self.links:
            if lnk.spice_port_trace and lnk.spice_port_trace not in self.spice_vectors:
                self.spice_vectors.append(lnk.spice_port_trace)
        self.josim_trace_names = list(josim_trace_names) if josim_trace_names else []
        self.work_dir = work_dir

    def run(self) -> BraninCosimResult:
        port_specs = [
            PortSpec(lnk.josim_pos_node, lnk.josim_neg_node, PORT_WAVE_BRANIN, lnk.z0) for lnk in self.links
        ]
        jsess = UnifiedCosimSession(
            self.josim_netlist,
            port_specs,
            phase_mode=self.phase_mode,
            is_file=self.josim_is_file,
            endpoint=self.endpoint,
        )
        try:
            dt = jsess.tstep
            n_total = int(round(self.t_stop / dt)) + 1
            t_full = np.linspace(0.0, self.t_stop, n_total)
            n_p = len(self.links)

            # History arrays over [0, t_stop]
            a_into_josim = np.zeros((n_total, n_p), dtype=np.float64)
            b_from_josim = np.zeros((n_total, n_p), dtype=np.float64)
            v_src_spice = np.zeros((n_total, n_p), dtype=np.float64)

            n_windows = max(1, int(np.ceil(self.t_stop / self.window_size - 1e-9)))
            final_spice_traces: Dict[str, np.ndarray] = {}

            base_tmp = Path(self.work_dir) if self.work_dir else Path("/mnt/c/dev/JoSIM")
            with tempfile.TemporaryDirectory(prefix=".branin_win_", dir=str(base_tmp)) as tmpd:
                tmp_path = Path(tmpd)

                for w in range(n_windows):
                    t_w_start = w * self.window_size
                    t_w_end = min((w + 1) * self.window_size, self.t_stop)
                    i_start = max(0, int(round(t_w_start / dt)))
                    i_end = min(n_total - 1, int(round(t_w_end / dt)))
                    if i_end <= i_start and w > 0:
                        continue

                    t_prefix = t_full[: i_end + 1]

                    # 1. Populate v_src_spice on [t_w_start, t_w_end] from causal delayed b_from_josim(t - td)
                    for p_idx, lnk in enumerate(self.links):
                        v_src_spice[: i_end + 1, p_idx] = np.interp(
                            t_prefix - lnk.td,
                            t_full[: i_start + 1],
                            b_from_josim[: i_start + 1, p_idx],
                            left=0.0,
                            right=0.0,
                        )

                    # 2. Format SPICE deck with PWL wave sources up to t_w_end
                    deck_str = self.spice_template.replace("{T_STOP}", f"{t_w_end:.12e}")
                    for p_idx, lnk in enumerate(self.links):
                        if not lnk.spice_pwl_placeholder:
                            continue
                        if self.simulator == "ltspice":
                            pwl_file = tmp_path / f"wave_p{p_idx}_w{w}.pwl"
                            win_pwl = format_spice_pwl_file(pwl_file, t_prefix, v_src_spice[: i_end + 1, p_idx])
                            deck_str = deck_str.replace(lnk.spice_pwl_placeholder, win_pwl)
                        else:
                            # ngspice inline PWL or file
                            inline_pwl = format_inline_pwl(t_prefix, v_src_spice[: i_end + 1, p_idx])
                            deck_str = deck_str.replace(lnk.spice_pwl_placeholder, inline_pwl)

                    # 3. Execute SPICE up to t_w_end
                    if self.simulator == "ltspice":
                        t_sp, tr_sp = run_ltspice_batch(deck_str, work_dir=tmp_path)
                    else:
                        ng_res = run_ngspice_deck(
                            deck_str,
                            wrdata_vecs=self.spice_vectors,
                            tran_cmd=f"tran {dt*2:.6e} {t_w_end:.6e}",
                        )
                        t_sp = ng_res.ng_time
                        tr_sp = {k.lower(): v for k, v in ng_res.ng_traces.items()}

                    # 4. Extract outgoing SPICE wave a_spice(t) = 2 * V_port(t) - V_src_spice(t)
                    #    and delay by td to form incoming wave at JoSIM a_into_josim(t) = a_spice(t - td)
                    for p_idx, lnk in enumerate(self.links):
                        key = lnk.spice_port_trace.lower()
                        v_p_sp = np.interp(t_prefix, t_sp, tr_sp[key])
                        a_sp = 2.0 * v_p_sp - v_src_spice[: i_end + 1, p_idx]
                        a_into_josim[i_start : i_end + 1, p_idx] = np.interp(
                            t_full[i_start : i_end + 1] - lnk.td,
                            t_prefix,
                            a_sp,
                            left=0.0,
                            right=0.0,
                        )

                    # 5. Advance JoSIM strictly forward across [t_w_start, t_w_end] via run_window
                    t_win_in = t_full[i_start : i_end + 1]
                    u_win_in = a_into_josim[i_start : i_end + 1, :]
                    t_win_out, b_win_out = jsess.run_window(t_w_end, t_win_in, u_win_in)
                    for p_idx in range(n_p):
                        b_from_josim[i_start : i_end + 1, p_idx] = np.interp(
                            t_win_in, t_win_out, b_win_out[:, p_idx]
                        )

                    if w == n_windows - 1:
                        for k, arr in tr_sp.items():
                            final_spice_traces[k] = np.interp(t_full, t_sp, arr)

            jsess.finalize()
            j_events = jsess.get_events()
            j_traces: Dict[str, np.ndarray] = {}
            for tname in self.josim_trace_names:
                try:
                    j_traces[tname] = jsess.get_trace(tname)
                except Exception:
                    pass

            return BraninCosimResult(
                t_grid=t_full,
                spice_traces=final_spice_traces,
                a_into_josim=a_into_josim,
                b_from_josim=b_from_josim,
                num_windows=n_windows,
                window_span=self.window_size,
                josim_events=j_events,
                josim_traces=j_traces,
            )
        finally:
            jsess.close()


def run_ltspice_branin_cosim(
    spice_template: str,
    josim_netlist: str,
    links: Sequence[BraninPortLink],
    t_stop: float,
    window_size: Optional[float] = None,
    phase_mode: bool = True,
    endpoint: str = "inproc",
    spice_vectors: Optional[Sequence[str]] = None,
    josim_trace_names: Optional[Sequence[str]] = None,
) -> BraninCosimResult:
    coord = BraninWaveWindowCoordinator(
        simulator="ltspice",
        spice_template=spice_template,
        josim_netlist=josim_netlist,
        links=links,
        t_stop=t_stop,
        window_size=window_size,
        phase_mode=phase_mode,
        endpoint=endpoint,
        spice_vectors=spice_vectors,
        josim_trace_names=josim_trace_names,
    )
    return coord.run()


def run_ngspice_branin_cosim(
    spice_template: str,
    josim_netlist: str,
    links: Sequence[BraninPortLink],
    t_stop: float,
    window_size: Optional[float] = None,
    phase_mode: bool = True,
    endpoint: str = "inproc",
    spice_vectors: Optional[Sequence[str]] = None,
    josim_trace_names: Optional[Sequence[str]] = None,
) -> BraninCosimResult:
    coord = BraninWaveWindowCoordinator(
        simulator="ngspice",
        spice_template=spice_template,
        josim_netlist=josim_netlist,
        links=links,
        t_stop=t_stop,
        window_size=window_size,
        phase_mode=phase_mode,
        endpoint=endpoint,
        spice_vectors=spice_vectors,
        josim_trace_names=josim_trace_names,
    )
    return coord.run()


# ---------------------------------------------------------------------------
# Damped Robin/Thevenin PWL Waveform Relaxation Coordinator
# ---------------------------------------------------------------------------
@dataclass
class PWLCosimResult:
    t_grid: np.ndarray
    v_port: np.ndarray
    i_port: np.ndarray
    spice_traces: Dict[str, np.ndarray]
    iterations: int
    final_max_delta_v: float


def run_ltspice_pwl_cosim(
    spice_template: str,
    josim_netlist: str,
    josim_pos_node: str,
    josim_neg_node: str = "0",
    spice_port_trace: str = "v(nd)",
    pwl_placeholder: str = "{PWL_TH}",
    req_placeholder: str = "{R_EQ_VAL}",
    r_eq: float = 20.0,
    max_iters: int = 5,
    tol_v: float = 100e-6,
    damping: float = 0.85,
    phase_mode: bool = True,
    endpoint: str = "inproc",
    work_dir: Optional[Union[str, Path]] = None,
) -> PWLCosimResult:
    """
    Damped Robin/Thevenin PWL waveform relaxation coupling JoSIM and ADI `LTspice.exe -b`.
    """
    # 1. Open-circuit pass to obtain initial Thevenin voltage trajectory V_J^(0)(t)
    s0 = UnifiedCosimSession(
        josim_netlist,
        [PortSpec(josim_pos_node, josim_neg_node, PORT_CURRENT_IN_VOLTAGE_OUT)],
        phase_mode=phase_mode,
        endpoint=endpoint,
    )
    try:
        dt = s0.tstep
        # Infer stop time from netlist .tran card
        t_stop = 60e-12
        for ln in josim_netlist.splitlines():
            ls = ln.strip().lower()
            if ls.startswith(".tran"):
                parts = ls.split()
                if len(parts) >= 3:
                    val_s = parts[2].replace("ps", "e-12").replace("p", "e-12").replace("ns", "e-9").replace("n", "e-9")
                    try:
                        t_stop = float(val_s)
                    except ValueError:
                        pass
        n_pts = int(round(t_stop / dt)) + 1
        t_grid = np.linspace(0.0, t_stop, n_pts)
        v_th = np.zeros(n_pts, dtype=np.float64)
        for k in range(1, n_pts):
            y_out, _ = s0.step_to(float(t_grid[k]), [0.0])
            v_th[k] = float(y_out[0])
            s0.commit()
        s0.finalize()
    finally:
        s0.close()

    v_port_prev = v_th.copy()
    i_port = np.zeros(n_pts, dtype=np.float64)
    final_traces: Dict[str, np.ndarray] = {}
    max_delta = 0.0
    iters_done = 0

    base_tmp = Path(work_dir) if work_dir else Path("/mnt/c/dev/JoSIM")
    with tempfile.TemporaryDirectory(prefix=".ltspice_pwl_", dir=str(base_tmp)) as tmpd:
        tmp_path = Path(tmpd)
        for it in range(1, max_iters + 1):
            iters_done = it
            pwl_file = tmp_path / f"vth_iter{it}.pwl"
            win_pwl = format_spice_pwl_file(pwl_file, t_grid, v_th)
            deck = spice_template.replace(pwl_placeholder, win_pwl).replace(req_placeholder, f"{r_eq:.6g}")
            t_lt, tr_lt = run_ltspice_batch(deck, work_dir=tmp_path)
            v_nd_lt = np.interp(t_grid, t_lt, tr_lt[spice_port_trace.lower()])

            # Re-evaluate JoSIM under terminal voltage v_nd_lt(t)
            sj = UnifiedCosimSession(
                josim_netlist,
                [PortSpec(josim_pos_node, josim_neg_node, PORT_VOLTAGE_IN_CURRENT_OUT)],
                phase_mode=phase_mode,
                endpoint=endpoint,
            )
            try:
                for k in range(1, n_pts):
                    y_i, _ = sj.step_to(float(t_grid[k]), [float(v_nd_lt[k])])
                    i_port[k] = float(y_i[0])
                    sj.commit()
                sj.finalize()
            finally:
                sj.close()

            v_th_new = v_nd_lt - r_eq * i_port
            v_th = damping * v_th_new + (1.0 - damping) * v_th
            max_delta = float(np.max(np.abs(v_nd_lt - v_port_prev)))
            v_port_prev = v_nd_lt.copy()
            final_traces = {k: np.interp(t_grid, t_lt, v) for k, v in tr_lt.items()}
            if it >= 2 and max_delta < tol_v:
                break

    return PWLCosimResult(
        t_grid=t_grid,
        v_port=v_port_prev,
        i_port=i_port,
        spice_traces=final_traces,
        iterations=iters_done,
        final_max_delta_v=max_delta,
    )


# ---------------------------------------------------------------------------
# Cadence Spectre Co-Simulation Runner (Local or BNL Cluster SSH)
# ---------------------------------------------------------------------------
@dataclass
class SpectreCosimResult:
    ok: bool
    dc_ok: bool
    tran_cpu_s: float
    tran_elapsed_s: float
    accepted_steps: int
    time: np.ndarray
    traces: Dict[str, np.ndarray]
    log_text: str = ""


class SpectreCosimRunner:
    """
    Executes a Cadence Spectre (`.scs`) co-simulation deck coupled with one or more
    JoSIM `.cir` subcircuit files via `libjosim_spectre_dpi.so`, `josim_cosim_port.va`,
    and a single multiplexed `josim-cosim-server` daemon (locally or on a BNL cluster
    host such as `asic2` over SSH).
    """

    def __init__(
        self,
        ssh_host: str = "asic2",
        tools_wrapper: str = "$HOME/Documents/LDRD_2025/dig_flows/resources/asic_tools.csh",
        build_dir: Optional[Union[str, Path]] = None,
    ):
        self.ssh_host = ssh_host
        self.tools_wrapper = tools_wrapper
        self.repo_root = Path(__file__).resolve().parent.parent
        self.build_dir = Path(build_dir or os.environ.get("JOSIM_BUILD_DIR", "/tmp/josim-build"))

    def run(
        self,
        scs_content: str,
        josim_files: Dict[str, str],
        saved_signals: Sequence[str],
        extra_files: Optional[Sequence[Union[str, Path]]] = None,
    ) -> SpectreCosimResult:
        rdir = f"/tmp/josim_spectre_l3b_{os.getpid()}_{int(time.time()*1000)%100000}"
        local_tmp = Path(tempfile.mkdtemp(prefix="josim_spectre_l3b_"))
        try:
            scs_path = local_tmp / "input.scs"
            scs_path.write_text(scs_content, encoding="utf-8", newline="\n")
            cir_paths: List[Path] = []
            for fname, content in josim_files.items():
                cp = local_tmp / fname
                cp.write_text(content, encoding="utf-8", newline="\n")
                cir_paths.append(cp)

            prep = f"set -e\nmkdir -p {rdir}/JoSIM\n"
            subprocess.run(
                ["ssh", "-o", "BatchMode=yes", self.ssh_host, "bash", "-s"],
                input=prep.encode("utf-8"),
                check=True,
                capture_output=True,
                timeout=20,
            )

            uploads: List[Path] = [
                self.build_dir / "josim-cosim-server",
                self.build_dir / "libjosim_c.so",
                Path("/lib/x86_64-linux-gnu/libgomp.so.1"),
                self.repo_root / "include/JoSIM/josim_spectre_dpi.h",
                self.repo_root / "include/JoSIM/josim_cosim_ipc.h",
                self.repo_root / "cosim/spectre/josim_cosim_dpi.c",
                self.repo_root / "cosim/spectre/josim_cosim_port.va",
                scs_path,
            ] + cir_paths
            if extra_files:
                uploads.extend(Path(f) for f in extra_files)

            subprocess.run(
                ["scp", "-q", "-o", "BatchMode=yes"] + [str(f) for f in uploads] + [f"{self.ssh_host}:{rdir}/"],
                check=True,
                capture_output=True,
                timeout=60,
            )

            sigs_json = repr(list(saved_signals))
            remote_sh = f"""set -e
cd {rdir}
cp -f josim_spectre_dpi.h josim_cosim_ipc.h JoSIM/
gcc -shared -fPIC -O2 -DJOSIM_DPI_SOCKET_ONLY -I. josim_cosim_dpi.c -o libjosim_spectre_dpi.so -lm
LOADER="$HOME/local/josim-2.6.5/lib/ld-linux-x86-64.so.2"
LIBPATH="{rdir}:$HOME/local/josim-2.6.5/lib"
TOOLS="{self.tools_wrapper}"
SOCK="{rdir}/josim_mux.sock"
rm -f "$SOCK"
"$LOADER" --library-path "$LIBPATH" ./josim-cosim-server --unix "$SOCK" > server.log 2>&1 &
SPID=$!
for _ in $(seq 1 60); do [ -S "$SOCK" ] && break; sleep 0.02; done
JOSIM_COSIM_ENDPOINT="unix:$SOCK" "$TOOLS" spectre -64 input.scs -format psfascii -raw psf_out +log spectre.log >/dev/null 2>&1
kill $SPID 2>/dev/null || true
rm -f "$SOCK"

python3 - << 'PYEOF'
import json, re
from pathlib import Path

txt = Path("spectre.log").read_text(errors="replace")
ok = "0 errors" in txt
dc_ok = "Convergence achieved" in txt
m_tran = re.search(r"Intrinsic tran analysis time:\\s*CPU\\s*=\\s*([0-9.]+)\\s*(ms|s|us),\\s*elapsed\\s*=\\s*([0-9.]+)\\s*(ms|s|us)", txt)
m_steps = re.search(r"Number of accepted tran steps\\s*=\\s*(\\d+)", txt)
scale = {{"us": 1e-6, "ms": 1e-3, "s": 1.0}}
cpu_s = float(m_tran.group(1)) * scale[m_tran.group(2)] if m_tran else 0.0
el_s = float(m_tran.group(3)) * scale[m_tran.group(4)] if m_tran else 0.0
steps = int(m_steps.group(1)) if m_steps else 0

sigs = {sigs_json}
out = {{s: [] for s in ["time"] + sigs}}
in_val = False
tran_files = list(Path("psf_out").glob("*.tran.tran"))
if tran_files:
    for line in tran_files[0].read_text(errors="replace").splitlines():
        if line.startswith("VALUE"):
            in_val = True
            continue
        if line.startswith("END"):
            break
        if in_val:
            parts = line.split()
            if len(parts) >= 2:
                k = parts[0].strip('"')
                if k in out:
                    out[k].append(float(parts[1]))

print("SPECTRE_L3B_JSON_START")
print(json.dumps({{
    "ok": ok,
    "dc_ok": dc_ok,
    "tran_cpu_s": cpu_s,
    "tran_elapsed_s": el_s,
    "accepted_steps": steps,
    "waveforms": out,
}}))
print("SPECTRE_L3B_JSON_END")
PYEOF
rm -rf {rdir}
"""
            proc = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", self.ssh_host, "bash", "-s"],
                input=remote_sh.replace("\r", "").encode("utf-8"),
                capture_output=True,
                timeout=180,
            )
            stdout_s = proc.stdout.decode("utf-8", errors="replace")
            stderr_s = proc.stderr.decode("utf-8", errors="replace")
            if proc.returncode != 0:
                raise RuntimeError(f"SpectreCosimRunner failed:\nSTDOUT:\n{stdout_s}\nSTDERR:\n{stderr_s}")

            import json as _json
            import re as _re

            m = _re.search(r"SPECTRE_L3B_JSON_START\s*(\{.*?\})\s*SPECTRE_L3B_JSON_END", stdout_s, _re.S)
            if not m:
                raise RuntimeError(f"Missing SPECTRE_L3B_JSON output:\n{stdout_s}")
            payload = _json.loads(m.group(1))
            wfs = payload["waveforms"]
            t_arr = np.asarray(wfs.get("time", []), dtype=np.float64)
            tr_dict = {s: np.asarray(wfs.get(s, []), dtype=np.float64) for s in saved_signals}
            return SpectreCosimResult(
                ok=bool(payload["ok"]),
                dc_ok=bool(payload["dc_ok"]),
                tran_cpu_s=float(payload["tran_cpu_s"]),
                tran_elapsed_s=float(payload["tran_elapsed_s"]),
                accepted_steps=int(payload["accepted_steps"]),
                time=t_arr,
                traces=tr_dict,
                log_text=stdout_s,
            )
        finally:
            shutil.rmtree(local_tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Unified CLI (`python3 scripts/josim_cosim.py --config spec.json`)
# ---------------------------------------------------------------------------
def run_cosim_from_config(config: Dict) -> Dict:
    """
    Execute a co-simulation job described by a JSON dictionary (`config`).
    Supported `mode` values:
      - `"ngspice_xspice"`: lockstep ngspice + `josim_cosim.cm` (in-process or multiplexed socket server)
      - `"branin_window"` (or `"ltspice_branin"` / `"ngspice_branin"`): multi-window Branin wave relaxation (`ngspice` or `ltspice`)
      - `"ltspice_pwl"`: damped Robin/Thevenin PWL relaxation (`ltspice`)
    """
    mode = config.get("mode", "ngspice_xspice").lower()
    transport = config.get("transport", config.get("endpoint", "inproc")).lower()

    if mode == "ngspice_xspice":
        if "spice_deck" in config:
            spice_deck = config["spice_deck"]
        else:
            spice_deck = Path(config["spice_deck_file"]).read_text(encoding="utf-8")
        wrdata_vecs = config["wrdata_vecs"]
        tran_cmd = config.get("tran_cmd", "tran 0.2p 60p")
        josim_csv_map = config.get("josim_csv_map", {})
        if transport in ("unix", "tcp"):
            n_clients = int(config.get("expected_clients", max(1, len(josim_csv_map))))
            with CosimServerProcess(transport=transport, expected_clients=n_clients) as srv:
                deck_filled = spice_deck.replace("{SERVER_ADDR}", srv.addr)
                res = run_ngspice_deck(deck_filled, wrdata_vecs=wrdata_vecs, tran_cmd=tran_cmd, josim_csv_map=josim_csv_map)
        else:
            deck_filled = spice_deck.replace("{SERVER_ADDR}", "")
            res = run_ngspice_deck(deck_filled, wrdata_vecs=wrdata_vecs, tran_cmd=tran_cmd, josim_csv_map=josim_csv_map)

        summary_traces = {
            k: {
                "min": float(np.min(v)),
                "max": float(np.max(v)),
                "pp": float(np.max(v) - np.min(v)),
                "final": float(v[-1]),
            }
            for k, v in res.ng_traces.items()
        }
        return {
            "mode": mode,
            "transport": transport,
            "num_timepoints": int(res.ng_time.size),
            "traces": summary_traces,
            "josim_events": {k: len(v) for k, v in res.josim_events.items()},
        }

    if mode in ("branin_window", "ltspice_branin", "ngspice_branin"):
        sim = "ngspice" if mode == "ngspice_branin" else config.get("simulator", "ltspice")
        if "spice_template" in config:
            spice_template = config["spice_template"]
        else:
            spice_template = Path(config["spice_template_file"]).read_text(encoding="utf-8")
        if "josim_netlist" in config:
            josim_netlist = config["josim_netlist"]
        else:
            josim_netlist = Path(config["josim_netlist_file"]).read_text(encoding="utf-8")
        links = [BraninPortLink(**lnk) for lnk in config["links"]]
        if transport in ("unix", "tcp"):
            with CosimServerProcess(transport=transport) as srv:
                coord = BraninWaveWindowCoordinator(
                    simulator=sim,
                    spice_template=spice_template,
                    josim_netlist=josim_netlist,
                    links=links,
                    t_stop=float(config["t_stop"]),
                    window_size=float(config.get("window_size", links[0].td)),
                    phase_mode=bool(config.get("phase_mode", True)),
                    endpoint=srv.addr,
                    spice_vectors=config.get("spice_vectors"),
                    josim_trace_names=config.get("josim_trace_names"),
                )
                bres = coord.run()
        else:
            coord = BraninWaveWindowCoordinator(
                simulator=sim,
                spice_template=spice_template,
                josim_netlist=josim_netlist,
                links=links,
                t_stop=float(config["t_stop"]),
                window_size=float(config.get("window_size", links[0].td)),
                phase_mode=bool(config.get("phase_mode", True)),
                endpoint="inproc",
                spice_vectors=config.get("spice_vectors"),
                josim_trace_names=config.get("josim_trace_names"),
            )
            bres = coord.run()

        return {
            "mode": mode,
            "simulator": sim,
            "transport": transport,
            "num_windows": bres.num_windows,
            "window_span": bres.window_span,
            "num_events": len(bres.josim_events),
            "traces": {
                k: {
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                    "pp": float(np.max(v) - np.min(v)),
                }
                for k, v in bres.spice_traces.items()
            },
        }

    if mode == "ltspice_pwl":
        if "spice_template" in config:
            spice_template = config["spice_template"]
        else:
            spice_template = Path(config["spice_template_file"]).read_text(encoding="utf-8")
        if "josim_netlist" in config:
            josim_netlist = config["josim_netlist"]
        else:
            josim_netlist = Path(config["josim_netlist_file"]).read_text(encoding="utf-8")

        def _exec_pwl(ep: str) -> PWLCosimResult:
            return run_ltspice_pwl_cosim(
                spice_template=spice_template,
                josim_netlist=josim_netlist,
                josim_pos_node=config.get("josim_pos_node", config.get("josim_port_pos", "nd")),
                josim_neg_node=config.get("josim_neg_node", config.get("josim_port_neg", "0")),
                spice_port_trace=config.get("spice_port_trace", config.get("spice_v_expr", "v(nd)")),
                pwl_placeholder=config.get("pwl_placeholder", "{PWL_TH}"),
                req_placeholder=config.get("req_placeholder", "{R_EQ_VAL}"),
                r_eq=float(config.get("r_eq", config.get("r_th", 20.0))),
                max_iters=int(config.get("max_iters", 5)),
                tol_v=float(config.get("tol_v", 100e-6)),
                damping=float(config.get("damping", 0.85)),
                phase_mode=bool(config.get("phase_mode", True)),
                endpoint=ep,
            )

        if transport in ("unix", "tcp"):
            with CosimServerProcess(transport=transport) as srv:
                pres = _exec_pwl(srv.addr)
        else:
            pres = _exec_pwl("inproc")

        return {
            "mode": mode,
            "simulator": "ltspice",
            "transport": transport,
            "iterations": pres.iterations,
            "final_max_delta_v": pres.final_max_delta_v,
            "traces": {
                k: {
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                    "pp": float(np.max(v) - np.min(v)),
                }
                for k, v in pres.spice_traces.items()
            },
        }

    raise ValueError(f"Unsupported co-simulation config mode: {mode}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="JoSIM / SPICE Unified Co-Simulation Coordinator")
    parser.add_argument("--config", "-c", required=True, help="Path to co-simulation JSON specification file")
    parser.add_argument("--output-json", "-o", default="", help="Optional path to write JSON summary report")
    args = parser.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    report = run_cosim_from_config(cfg)
    out_str = json.dumps(report, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(out_str + "\n", encoding="utf-8")
    else:
        print(out_str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
