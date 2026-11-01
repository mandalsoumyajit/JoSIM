// Copyright (c) 2026 JoSIM Developers
// This file is part of JoSIM. Details are in the LICENSE file.
//
// Layer 3A: Cadence Spectre Verilog-A (CDS_VA_DPI) & IEEE 1800 SystemVerilog (DPI-C)
// Co-Simulation Interface Header.
//
// Supports both:
//   1. In-process mode (endpoint = "" or "inproc"): calls libjosim_c directly in-process.
//   2. Socket IPC mode (endpoint = "unix:/path" or "tcp:host:port" or $JOSIM_COSIM_ENDPOINT):
//      connects to josim-cosim-server over AF_UNIX or AF_INET (TCP_NODELAY) sockets.
//      When compiled with -DJOSIM_DPI_SOCKET_ONLY, has zero C++ or libjosim_c dependencies.

#ifndef JOSIM_SPECTRE_DPI_H
#define JOSIM_SPECTRE_DPI_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #define JOSIM_DPI_API __declspec(dllexport)
#elif __GNUC__ >= 4
  #define JOSIM_DPI_API __attribute__((visibility("default")))
#else
  #define JOSIM_DPI_API
#endif

// Open or reuse an idempotent co-simulation session handle (>= 1 on success, < 0 on error).
// - netlist_path:   path to JoSIM .cir netlist (ignored if socket server already loaded a deck
//                   and netlist_path is empty).
// - port_nodes_csv: comma/space-separated JoSIM boundary node names (e.g. "b4" or "tx_p,rx_p"
//                   or differential "p1:n1,p2:n2").
// - port_modes_csv: comma/space-separated integer port modes:
//                   0 = JOSIM_PORT_V_IN_I_OUT (Norton: SPICE applies V, JoSIM returns I & G_eq)
//                   1 = JOSIM_PORT_I_IN_V_OUT (Thevenin: SPICE applies I, JoSIM returns V & R_eq)
//                   2 = JOSIM_PORT_WAVE_BRANIN (Wave: SPICE exchanges V/I with characteristic Z0)
// - endpoint:       "" or "inproc" for in-process libjosim_c, or "unix:/path/to.sock" /
//                   "tcp:127.0.0.1:port" for josim-cosim-server IPC. If empty and the environment
//                   variable JOSIM_COSIM_ENDPOINT is set, JOSIM_COSIM_ENDPOINT is used.
JOSIM_DPI_API int josim_dpi_open(
    const char* netlist_path,
    const char* port_nodes_csv,
    const char* port_modes_csv,
    const char* endpoint);

// Query JoSIM internal timestep (seconds), port count, and rollback count.
JOSIM_DPI_API double josim_dpi_tstep(int handle);
JOSIM_DPI_API int    josim_dpi_num_ports(int handle);
JOSIM_DPI_API int    josim_dpi_rollback_count(int handle);

// Single-port Norton convenience API (port 0, mode 0: V -> I, G_eq):
// Note: josim_dpi_eval_ihist returns I_hist = I_port(V) - G_eq * V directly from C so that
// Verilog-A automatic differentiation treats I_hist and G_eq as constants during a single
// stamp `I(p, n) <+ i_hist + g_eq * V(p, n);`, stamping exact value I_port(V) and exact
// analytical Jacobian dI/dV = G_eq into Spectre's MNA matrix.
JOSIM_DPI_API double josim_dpi_eval_i(int handle, double t_s, double v_pn);
JOSIM_DPI_API double josim_dpi_eval_ihist(int handle, double t_s, double v_pn);
JOSIM_DPI_API double josim_dpi_eval_g(int handle, double t_s, double v_pn);

// Single-port Thevenin convenience API (port 0, mode 1: I -> V, R_eq):
JOSIM_DPI_API double josim_dpi_eval_v(int handle, double t_s, double i_pn);
JOSIM_DPI_API double josim_dpi_eval_vhist(int handle, double t_s, double i_pn);
JOSIM_DPI_API double josim_dpi_eval_r(int handle, double t_s, double i_pn);

// Multi-port staging and evaluation API (ports 0 .. P-1):
JOSIM_DPI_API int    josim_dpi_set_input(int handle, int port_idx, double val);
JOSIM_DPI_API int    josim_dpi_eval(int handle, double t_s);
JOSIM_DPI_API double josim_dpi_get_output(int handle, int port_idx);
JOSIM_DPI_API double josim_dpi_get_jacobian(int handle, int port_idx);
JOSIM_DPI_API double josim_dpi_get_hist(int handle, int port_idx);

// Configure optional JoSIM output CSV (for .print traces) and switching-events CSV (-e format)
// to be written automatically when the session closes / finalizes.
JOSIM_DPI_API int josim_dpi_set_output(int handle, const char* out_csv, const char* events_csv);

// Close a specific session handle or all active handles.
JOSIM_DPI_API int  josim_dpi_close(int handle);
JOSIM_DPI_API void josim_dpi_close_all(void);

#ifdef __cplusplus
}
#endif

#endif // JOSIM_SPECTRE_DPI_H
