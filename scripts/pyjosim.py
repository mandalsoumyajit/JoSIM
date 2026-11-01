#!/usr/bin/env python3
"""In-memory Python / C binding (pyjosim) with zero-copy NumPy views for JoSIM.

Provides:
  - Session: persistent in-memory simulator supporting zero-reparse parameter overrides
  - SimulationResult: zero-copy NumPy array views over waveforms, 2D/3D matrices,
    structured SfqEvent arrays, .meas results, and exported .param variables
  - simulate_netlist / simulate_file: one-call in-memory simulation helpers
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

PORT_VOLTAGE_IN_CURRENT_OUT = 0
PORT_CURRENT_IN_VOLTAGE_OUT = 1
PORT_WAVE_BRANIN = 2

SFQ_EVENT_DTYPE = np.dtype(
    [
        ("time", "<f8"),
        ("t_peak", "<f8"),
        ("v_peak", "<f8"),
        ("fwhm", "<f8"),
        ("dt_pi", "<f8"),
        ("area", "<f8"),
        ("phase", "<f8"),
        ("polarity", "<i4"),
        ("winding", "<i4"),
        ("trace_id", "<i4"),
        ("trial", "<i4"),
    ],
    align=True,
)


class _CSfqEvent(ctypes.Structure):
    _fields_ = [
        ("time", ctypes.c_double),
        ("t_peak", ctypes.c_double),
        ("v_peak", ctypes.c_double),
        ("fwhm", ctypes.c_double),
        ("dt_pi", ctypes.c_double),
        ("area", ctypes.c_double),
        ("phase", ctypes.c_double),
        ("polarity", ctypes.c_int32),
        ("winding", ctypes.c_int32),
        ("trace_id", ctypes.c_int32),
        ("trial", ctypes.c_int32),
    ]


_LIB_CACHE: Dict[str, ctypes.CDLL] = {}


def _find_libjosim_c(lib_path: Optional[Union[str, Path]] = None) -> Path:
    if lib_path is not None:
        p = Path(lib_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Specified libjosim_c shared library not found: {p}")

    env_path = os.environ.get("JOSIM_C_LIB")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p

    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        Path("/tmp/josim-build/libjosim_c.so"),
        repo_root / "build" / "libjosim_c.so",
        repo_root / "build" / "libjosim_c.dylib",
        repo_root / "build" / "josim_c.dll",
        repo_root / "build" / "Release" / "josim_c.dll",
        Path.cwd() / "libjosim_c.so",
        Path.cwd() / "josim_c.dll",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "Could not locate libjosim_c shared library (.so/.dll/.dylib). "
        "Set JOSIM_C_LIB or pass lib_path=..."
    )


def _load_cdll(lib_path: Optional[Union[str, Path]] = None) -> ctypes.CDLL:
    resolved = str(_find_libjosim_c(lib_path).resolve())
    if resolved in _LIB_CACHE:
        return _LIB_CACHE[resolved]

    lib = ctypes.CDLL(resolved)

    lib.josim_version.argtypes = []
    lib.josim_version.restype = ctypes.c_char_p

    lib.josim_session_create.argtypes = []
    lib.josim_session_create.restype = ctypes.c_void_p

    lib.josim_session_destroy.argtypes = [ctypes.c_void_p]
    lib.josim_session_destroy.restype = None

    lib.josim_session_last_error.argtypes = [ctypes.c_void_p]
    lib.josim_session_last_error.restype = ctypes.c_char_p

    lib.josim_set_cli_args.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.josim_set_cli_args.restype = ctypes.c_int

    lib.josim_load_netlist_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.josim_load_netlist_file.restype = ctypes.c_int

    lib.josim_load_netlist_string.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]
    lib.josim_load_netlist_string.restype = ctypes.c_int

    lib.josim_set_parameter.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_double,
        ctypes.c_char_p,
    ]
    lib.josim_set_parameter.restype = ctypes.c_int

    lib.josim_clear_parameter_overrides.argtypes = [ctypes.c_void_p]
    lib.josim_clear_parameter_overrides.restype = ctypes.c_int

    lib.josim_run.argtypes = [ctypes.c_void_p]
    lib.josim_run.restype = ctypes.c_int

    lib.josim_num_traces.argtypes = [ctypes.c_void_p]
    lib.josim_num_traces.restype = ctypes.c_int64

    lib.josim_num_steps.argtypes = [ctypes.c_void_p]
    lib.josim_num_steps.restype = ctypes.c_int64

    lib.josim_trace_name.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    lib.josim_trace_name.restype = ctypes.c_char_p

    lib.josim_trace_type.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    lib.josim_trace_type.restype = ctypes.c_char

    lib.josim_trace_data.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    lib.josim_trace_data.restype = ctypes.POINTER(ctypes.c_double)

    lib.josim_trace_data_by_name.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int64),
    ]
    lib.josim_trace_data_by_name.restype = ctypes.POINTER(ctypes.c_double)

    lib.josim_traces_matrix_col_major.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ]
    lib.josim_traces_matrix_col_major.restype = ctypes.POINTER(ctypes.c_double)

    lib.josim_mc_trace_matrix.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ]
    lib.josim_mc_trace_matrix.restype = ctypes.POINTER(ctypes.c_double)

    lib.josim_num_events.argtypes = [ctypes.c_void_p]
    lib.josim_num_events.restype = ctypes.c_int64

    lib.josim_events_data.argtypes = [ctypes.c_void_p]
    lib.josim_events_data.restype = ctypes.POINTER(_CSfqEvent)

    lib.josim_num_event_traces.argtypes = [ctypes.c_void_p]
    lib.josim_num_event_traces.restype = ctypes.c_int64

    lib.josim_event_trace_name.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.josim_event_trace_name.restype = ctypes.c_char_p

    lib.josim_num_parameters.argtypes = [ctypes.c_void_p]
    lib.josim_num_parameters.restype = ctypes.c_int64

    lib.josim_parameter_name.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    lib.josim_parameter_name.restype = ctypes.c_char_p

    lib.josim_parameter_value.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    lib.josim_parameter_value.restype = ctypes.c_double

    lib.josim_get_parameter.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.josim_get_parameter.restype = ctypes.c_int

    lib.josim_num_measurements.argtypes = [ctypes.c_void_p]
    lib.josim_num_measurements.restype = ctypes.c_int64

    lib.josim_measurement_name.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    lib.josim_measurement_name.restype = ctypes.c_char_p

    lib.josim_measurement_value.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    lib.josim_measurement_value.restype = ctypes.c_double

    lib.josim_measurement_trial_values.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64),
    ]
    lib.josim_measurement_trial_values.restype = ctypes.POINTER(ctypes.c_double)

    # Co-simulation stepping, Norton Jacobian & rollback API
    lib.josim_cosim_add_port.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_double,
    ]
    lib.josim_cosim_add_port.restype = ctypes.c_int

    lib.josim_cosim_clear_ports.argtypes = [ctypes.c_void_p]
    lib.josim_cosim_clear_ports.restype = ctypes.c_int

    lib.josim_cosim_num_ports.argtypes = [ctypes.c_void_p]
    lib.josim_cosim_num_ports.restype = ctypes.c_int64

    lib.josim_cosim_init.argtypes = [ctypes.c_void_p]
    lib.josim_cosim_init.restype = ctypes.c_int

    lib.josim_cosim_tstep.argtypes = [ctypes.c_void_p]
    lib.josim_cosim_tstep.restype = ctypes.c_double

    lib.josim_cosim_current_time.argtypes = [ctypes.c_void_p]
    lib.josim_cosim_current_time.restype = ctypes.c_double

    lib.josim_cosim_rollback_count.argtypes = [ctypes.c_void_p]
    lib.josim_cosim_rollback_count.restype = ctypes.c_int64

    lib.josim_cosim_step_to.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.josim_cosim_step_to.restype = ctypes.c_int

    lib.josim_cosim_commit.argtypes = [ctypes.c_void_p]
    lib.josim_cosim_commit.restype = ctypes.c_int

    lib.josim_cosim_rollback.argtypes = [ctypes.c_void_p]
    lib.josim_cosim_rollback.restype = ctypes.c_int

    lib.josim_cosim_run_window.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ]
    lib.josim_cosim_run_window.restype = ctypes.c_int

    lib.josim_cosim_finalize.argtypes = [ctypes.c_void_p]
    lib.josim_cosim_finalize.restype = ctypes.c_int

    _LIB_CACHE[resolved] = lib
    return lib


class SimulationResult:
    """Zero-copy NumPy view container over an in-memory JoSIM simulation run."""

    def __init__(self, session: "Session") -> None:
        self._session = session
        self._lib = session._lib
        self._ptr = session._ptr

        n_traces = self._lib.josim_num_traces(self._ptr)
        n_steps = self._lib.josim_num_steps(self._ptr)
        self.num_traces: int = int(n_traces)
        self.num_steps: int = int(n_steps)

        self.traces: Dict[str, np.ndarray] = {}
        self._upper_map: Dict[str, str] = {}
        for idx in range(self.num_traces):
            raw_name = self._lib.josim_trace_name(self._ptr, idx)
            name = raw_name.decode("utf-8") if raw_name else f"trace_{idx}"
            data_ptr = self._lib.josim_trace_data(self._ptr, idx)
            if data_ptr and self.num_steps > 0:
                view = np.ctypeslib.as_array(data_ptr, shape=(self.num_steps,))
            else:
                view = np.empty((0,), dtype=np.float64)
            self.traces[name] = view
            self._upper_map[name.upper()] = name

        # Zero-copy SFQ events structured array
        n_events = int(self._lib.josim_num_events(self._ptr))
        ev_ptr = self._lib.josim_events_data(self._ptr)
        if ev_ptr and n_events > 0:
            raw_ev = np.ctypeslib.as_array(ev_ptr, shape=(n_events,))
            self.events: np.ndarray = raw_ev.view(SFQ_EVENT_DTYPE)
        else:
            self.events = np.empty((0,), dtype=SFQ_EVENT_DTYPE)

        n_ev_traces = int(self._lib.josim_num_event_traces(self._ptr))
        self.event_traces: List[str] = []
        for tid in range(n_ev_traces):
            tn = self._lib.josim_event_trace_name(self._ptr, tid)
            self.event_traces.append(tn.decode("utf-8") if tn else f"P_{tid}")

        # Parameters (including exported IV_*, SHAPIRO_*, SQUID_*, BETA_L_*, JITTER_* metrics)
        n_params = int(self._lib.josim_num_parameters(self._ptr))
        self.parameters: Dict[str, float] = {}
        for idx in range(n_params):
            pn = self._lib.josim_parameter_name(self._ptr, idx)
            if pn:
                self.parameters[pn.decode("utf-8")] = float(
                    self._lib.josim_parameter_value(self._ptr, idx)
                )

        # Evaluated .meas results (scalar + zero-copy per-trial arrays)
        n_meas = int(self._lib.josim_num_measurements(self._ptr))
        self.measurements: Dict[str, float] = {}
        self.measurement_trials: Dict[str, np.ndarray] = {}
        for idx in range(n_meas):
            mn = self._lib.josim_measurement_name(self._ptr, idx)
            if not mn:
                continue
            mname = mn.decode("utf-8")
            self.measurements[mname] = float(
                self._lib.josim_measurement_value(self._ptr, idx)
            )
            n_tr = ctypes.c_int64(0)
            t_ptr = self._lib.josim_measurement_trial_values(
                self._ptr, idx, ctypes.byref(n_tr)
            )
            if t_ptr and n_tr.value > 0:
                self.measurement_trials[mname] = np.ctypeslib.as_array(
                    t_ptr, shape=(int(n_tr.value),)
                )

    @property
    def time(self) -> np.ndarray:
        """Return the zero-copy 1D time vector."""
        return self.traces.get("time", np.empty((0,), dtype=np.float64))

    def __getitem__(self, name: str) -> np.ndarray:
        if name in self.traces:
            return self.traces[name]
        key = name.strip().strip('"').upper()
        if key in self._upper_map:
            return self.traces[self._upper_map[key]]
        raise KeyError(f"Trace '{name}' not found. Available: {list(self.traces.keys())}")

    def __contains__(self, name: str) -> bool:
        if name in self.traces:
            return True
        return name.strip().strip('"').upper() in self._upper_map

    @property
    def matrix(self) -> np.ndarray:
        """Return a zero-copy 2D float64 array of shape (num_traces, num_steps)."""
        cols = ctypes.c_int64(0)
        rows = ctypes.c_int64(0)
        ptr = self._lib.josim_traces_matrix_col_major(
            self._ptr, ctypes.byref(cols), ctypes.byref(rows)
        )
        if not ptr or cols.value <= 0 or rows.value <= 0:
            return np.empty((0, 0), dtype=np.float64)
        return np.ctypeslib.as_array(ptr, shape=(int(cols.value), int(rows.value)))

    @property
    def mc_matrix(self) -> np.ndarray:
        """Return a zero-copy 3D float64 array of shape (trials, rows, steps) in batched mode."""
        trials = ctypes.c_int64(0)
        rows = ctypes.c_int64(0)
        steps = ctypes.c_int64(0)
        ptr = self._lib.josim_mc_trace_matrix(
            self._ptr, ctypes.byref(trials), ctypes.byref(rows), ctypes.byref(steps)
        )
        if not ptr or trials.value <= 0 or rows.value <= 0 or steps.value <= 0:
            return np.empty((0, 0, 0), dtype=np.float64)
        return np.ctypeslib.as_array(
            ptr, shape=(int(trials.value), int(rows.value), int(steps.value))
        )

    def events_for(self, trace: str, trial: Optional[int] = None) -> np.ndarray:
        """Filter the structured SFQ events array for a specific junction or phase trace."""
        t_clean = trace.strip().strip('"').upper()
        if not t_clean.startswith("P("):
            t_cand = f"P({t_clean})"
        else:
            t_cand = t_clean
        matched_ids = [
            idx
            for idx, nm in enumerate(self.event_traces)
            if nm.upper() == t_clean or nm.upper() == t_cand
        ]
        if not matched_ids or len(self.events) == 0:
            return self.events[:0]
        mask = np.isin(self.events["trace_id"], matched_ids)
        if trial is not None:
            mask &= self.events["trial"] == int(trial)
        return self.events[mask]


class Session:
    """Persistent in-memory JoSIM session supporting zero-I/O parameter updates and re-simulation."""

    def __init__(
        self,
        netlist: Optional[str] = None,
        *,
        path: Optional[Union[str, Path]] = None,
        cli_args: Optional[Sequence[str]] = None,
        base_dir: Optional[Union[str, Path]] = None,
        lib_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self._lib = _load_cdll(lib_path)
        self._ptr = self._lib.josim_session_create()
        if not self._ptr:
            raise RuntimeError("Failed to allocate josim_session_t")

        if cli_args:
            self.set_cli_args(cli_args)
        if path is not None:
            self.load_file(path)
        elif netlist is not None:
            if "\n" not in netlist and Path(netlist).is_file():
                self.load_file(netlist)
            else:
                self.load_string(netlist, base_dir=base_dir)

    def __del__(self) -> None:
        if getattr(self, "_ptr", None):
            self._lib.josim_session_destroy(self._ptr)
            self._ptr = None

    def _raise_last_error(self, prefix: str = "JoSIM error") -> None:
        err = self._lib.josim_session_last_error(self._ptr)
        msg = err.decode("utf-8", errors="replace") if err else "Unknown error"
        raise RuntimeError(f"{prefix}: {msg}")

    @property
    def version(self) -> str:
        v = self._lib.josim_version()
        return v.decode("utf-8") if v else "2.7"

    def set_cli_args(self, args: Sequence[str]) -> "Session":
        encoded = [str(a).encode("utf-8") for a in args]
        c_arr = (ctypes.c_char_p * len(encoded))(*encoded)
        rc = self._lib.josim_set_cli_args(self._ptr, len(encoded), c_arr)
        if rc != 0:
            self._raise_last_error("josim_set_cli_args failed")
        return self

    def load_file(self, path: Union[str, Path]) -> "Session":
        p_str = str(Path(path).resolve()).encode("utf-8")
        rc = self._lib.josim_load_netlist_file(self._ptr, p_str)
        if rc != 0:
            self._raise_last_error(f"Failed to load netlist file '{path}'")
        return self

    def load_string(
        self, netlist_text: str, base_dir: Optional[Union[str, Path]] = None
    ) -> "Session":
        txt_bytes = netlist_text.encode("utf-8")
        dir_bytes = str(base_dir).encode("utf-8") if base_dir is not None else None
        rc = self._lib.josim_load_netlist_string(self._ptr, txt_bytes, dir_bytes)
        if rc != 0:
            self._raise_last_error("Failed to parse in-memory netlist string")
        return self

    def set_param(
        self, name: str, value: float, subckt: Optional[str] = None
    ) -> "Session":
        n_bytes = name.encode("utf-8")
        s_bytes = subckt.encode("utf-8") if subckt else None
        rc = self._lib.josim_set_parameter(self._ptr, n_bytes, float(value), s_bytes)
        if rc != 0:
            self._raise_last_error(f"Failed to set parameter '{name}'")
        return self

    def set_params(self, params: Mapping[str, float]) -> "Session":
        for k, v in params.items():
            if "|" in k:
                pname, sub = k.split("|", 1)
                self.set_param(pname, v, subckt=sub)
            else:
                self.set_param(k, v)
        return self

    def clear_params(self) -> "Session":
        self._lib.josim_clear_parameter_overrides(self._ptr)
        return self

    def run(
        self, params: Optional[Mapping[str, float]] = None
    ) -> SimulationResult:
        if params:
            self.set_params(params)
        rc = self._lib.josim_run(self._ptr)
        if rc != 0:
            self._raise_last_error("Simulation failed")
        return SimulationResult(self)

    # ------------------------------------------------------------------
    # Co-simulation stepping, Norton Jacobian & rollback methods
    # ------------------------------------------------------------------

    def cosim_add_port(
        self,
        pos_node: str,
        neg_node: str = "0",
        mode: int = PORT_VOLTAGE_IN_CURRENT_OUT,
        z0: float = 50.0,
    ) -> int:
        pos_b = str(pos_node).encode("utf-8")
        neg_b = str(neg_node).encode("utf-8") if neg_node else b"0"
        idx = int(
            self._lib.josim_cosim_add_port(
                self._ptr, pos_b, neg_b, int(mode), float(z0)
            )
        )
        if idx < 0:
            self._raise_last_error("josim_cosim_add_port failed")
        return idx

    def cosim_clear_ports(self) -> "Session":
        self._lib.josim_cosim_clear_ports(self._ptr)
        return self

    @property
    def cosim_num_ports(self) -> int:
        return int(self._lib.josim_cosim_num_ports(self._ptr))

    def cosim_init(
        self, params: Optional[Mapping[str, float]] = None
    ) -> "Session":
        if params:
            self.set_params(params)
        rc = self._lib.josim_cosim_init(self._ptr)
        if rc != 0:
            self._raise_last_error("josim_cosim_init failed")
        self._n_ports_cached = self.cosim_num_ports
        self._u_buf = np.zeros(self._n_ports_cached, dtype=np.float64)
        self._y_buf = np.zeros(self._n_ports_cached, dtype=np.float64)
        self._j_buf = np.zeros(self._n_ports_cached, dtype=np.float64)
        return self

    @property
    def cosim_tstep(self) -> float:
        return float(self._lib.josim_cosim_tstep(self._ptr))

    @property
    def cosim_current_time(self) -> float:
        return float(self._lib.josim_cosim_current_time(self._ptr))

    @property
    def cosim_rollback_count(self) -> int:
        return int(self._lib.josim_cosim_rollback_count(self._ptr))

    def cosim_step_to(
        self, t_target: float, u_in: Sequence[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_p = getattr(self, "_n_ports_cached", self.cosim_num_ports)
        if len(u_in) != n_p:
            raise ValueError(
                f"Expected {n_p} port inputs, got {len(u_in)}"
            )
        u_arr = self._u_buf
        y_arr = self._y_buf
        j_arr = self._j_buf
        u_arr[:] = u_in
        rc = self._lib.josim_cosim_step_to(
            self._ptr,
            float(t_target),
            u_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            j_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            self._raise_last_error("josim_cosim_step_to failed")
        return y_arr.copy(), j_arr.copy()

    def cosim_commit(self) -> "Session":
        rc = self._lib.josim_cosim_commit(self._ptr)
        if rc != 0:
            self._raise_last_error("josim_cosim_commit failed")
        return self

    def cosim_rollback(self) -> "Session":
        rc = self._lib.josim_cosim_rollback(self._ptr)
        if rc != 0:
            self._raise_last_error("josim_cosim_rollback failed")
        return self

    def cosim_run_window(
        self,
        t_start: float,
        t_end: float,
        t_pts: np.ndarray,
        u_in_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run a Branin wave or PWL window [t_start, t_end].

        Parameters
        ----------
        t_start, t_end : float
            Window start and end times (s).
        t_pts : 1D array of shape (n_pts,)
            Monotonically non-decreasing sample times covering [t_start, t_end].
        u_in_matrix : 2D array of shape (n_pts, n_ports) or 1D if n_ports == 1
            Input values at each sample time for each port.

        Returns
        -------
        out_t : 1D float64 array of shape (out_n,)
        out_y : 2D float64 array of shape (out_n, n_ports)
        """
        n_p = getattr(self, "_n_ports_cached", self.cosim_num_ports)
        t_arr = np.ascontiguousarray(t_pts, dtype=np.float64)
        u_arr = np.ascontiguousarray(u_in_matrix, dtype=np.float64)
        if u_arr.ndim == 1:
            u_arr = u_arr.reshape(-1, n_p)
        if u_arr.shape != (len(t_arr), n_p):
            raise ValueError(
                f"u_in_matrix shape {u_arr.shape} must match (len(t_pts)={len(t_arr)}, n_ports={n_p})"
            )
        out_n = ctypes.c_int64(0)
        out_t_ptr = ctypes.POINTER(ctypes.c_double)()
        out_y_ptr = ctypes.POINTER(ctypes.c_double)()
        rc = self._lib.josim_cosim_run_window(
            self._ptr,
            float(t_end),
            len(t_arr),
            t_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            u_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(out_n),
            ctypes.byref(out_t_ptr),
            ctypes.byref(out_y_ptr),
        )
        if rc != 0:
            self._raise_last_error("josim_cosim_run_window failed")
        n_out = int(out_n.value)
        if n_out <= 0:
            return np.empty((0,), dtype=np.float64), np.empty((0, n_p), dtype=np.float64)
        out_t = np.ctypeslib.as_array(out_t_ptr, shape=(n_out,)).copy()
        out_y = np.ctypeslib.as_array(out_y_ptr, shape=(n_out, n_p)).copy()
        return out_t, out_y

    def cosim_finalize(self) -> SimulationResult:
        rc = self._lib.josim_cosim_finalize(self._ptr)
        if rc != 0:
            self._raise_last_error("josim_cosim_finalize failed")
        return SimulationResult(self)


def simulate_netlist(
    netlist_text: str,
    *,
    params: Optional[Mapping[str, float]] = None,
    cli_args: Optional[Sequence[str]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    lib_path: Optional[Union[str, Path]] = None,
) -> SimulationResult:
    """Run an in-memory simulation from a SPICE netlist string and return zero-copy NumPy views."""
    sess = Session(
        netlist=netlist_text,
        cli_args=cli_args,
        base_dir=base_dir,
        lib_path=lib_path,
    )
    return sess.run(params=params)


def simulate_file(
    path: Union[str, Path],
    *,
    params: Optional[Mapping[str, float]] = None,
    cli_args: Optional[Sequence[str]] = None,
    lib_path: Optional[Union[str, Path]] = None,
) -> SimulationResult:
    """Run an in-memory simulation from a netlist file and return zero-copy NumPy views."""
    sess = Session(path=path, cli_args=cli_args, lib_path=lib_path)
    return sess.run(params=params)
