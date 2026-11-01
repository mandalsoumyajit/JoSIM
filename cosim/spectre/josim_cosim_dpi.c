// Copyright (c) 2026 JoSIM Developers
// This file is part of JoSIM. Details are in the LICENSE file.
//
// Layer 3A: Cadence Spectre Verilog-A (CDS_VA_DPI) & IEEE 1800 SystemVerilog (DPI-C)
// Co-Simulation Shared Library Implementation (libjosim_spectre_dpi.so).

#include "JoSIM/josim_spectre_dpi.h"
#include "JoSIM/josim_cosim_ipc.h"

#ifndef JOSIM_DPI_SOCKET_ONLY
#include "JoSIM/josim_c_api.h"
#endif

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define JOSIM_DPI_MAX_SESSIONS 64
#define JOSIM_DPI_MAX_PORTS    64
#define JOSIM_DPI_MIN_G        1e-12

typedef struct {
    int              active;
    int              use_socket;
    char             netlist_path[512];
    char             port_nodes_csv[512];
    char             port_modes_csv[256];
    char             endpoint[256];

#ifndef JOSIM_DPI_SOCKET_ONLY
    josim_session_t* session;
#else
    void*            session;
#endif
    int              sock_fd;

    int              n_ports;
    int              port_modes[JOSIM_DPI_MAX_PORTS];
    double           port_z0[JOSIM_DPI_MAX_PORTS];
    double           tstep;
    int64_t          rollback_count;

    double           staged_in[JOSIM_DPI_MAX_PORTS];
    int              has_cache;
    double           cached_t;
    double           cached_in[JOSIM_DPI_MAX_PORTS];
    double           cached_out[JOSIM_DPI_MAX_PORTS];
    double           cached_jac[JOSIM_DPI_MAX_PORTS];
} josim_dpi_slot_t;

static josim_dpi_slot_t g_dpi_slots[JOSIM_DPI_MAX_SESSIONS];
static int              g_dpi_initialized = 0;

static int parse_csv_tokens(const char* csv, char tokens[][128], int max_tokens) {
    if (!csv || !csv[0]) return 0;
    int count = 0;
    const char* p = csv;
    while (*p && count < max_tokens) {
        while (*p && (isspace((unsigned char)*p) || *p == ',' || *p == ';')) ++p;
        if (!*p) break;
        size_t len = 0;
        while (p[len] && !isspace((unsigned char)p[len]) && p[len] != ',' && p[len] != ';') {
            ++len;
        }
        if (len >= 128) len = 127;
        memcpy(tokens[count], p, len);
        tokens[count][len] = '\0';
        ++count;
        p += len;
    }
    return count;
}

static void dpi_slot_free(josim_dpi_slot_t* st) {
    if (!st || !st->active) return;
#if !defined(_WIN32)
    if (st->use_socket) {
        if (st->sock_fd >= 0) {
            uint32_t zero_len = 0;
            if (josim_ipc_send_req(st->sock_fd, JOSIM_IPC_OP_FINALIZE, &zero_len, sizeof(zero_len)) == 0) {
                int32_t  status  = 0;
                uint64_t out_len = 0;
                uint8_t  resp[16];
                josim_ipc_recv_resp(st->sock_fd, &status, resp, sizeof(resp), &out_len);
            }
            close(st->sock_fd);
            st->sock_fd = -1;
        }
    } else
#endif
    {
#ifndef JOSIM_DPI_SOCKET_ONLY
        if (st->session) {
            josim_cosim_finalize(st->session);
            josim_session_destroy(st->session);
            st->session = NULL;
        }
#endif
    }
    memset(st, 0, sizeof(*st));
    st->sock_fd = -1;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((destructor))
static void dpi_library_cleanup(void) {
    josim_dpi_close_all();
}
#endif

static int dpi_init_slot_backend(josim_dpi_slot_t* st) {
    char node_toks[JOSIM_DPI_MAX_PORTS][128];
    char mode_toks[JOSIM_DPI_MAX_PORTS][128];
    int n_nodes = parse_csv_tokens(st->port_nodes_csv, node_toks, JOSIM_DPI_MAX_PORTS);
    int n_modes = parse_csv_tokens(st->port_modes_csv, mode_toks, JOSIM_DPI_MAX_PORTS);
    if (n_nodes <= 0) {
        fprintf(stderr, "[josim_spectre_dpi] ERROR: no port_nodes specified\n");
        return -1;
    }

    for (int i = 0; i < JOSIM_DPI_MAX_PORTS; ++i) {
        st->port_modes[i] = 0;
        st->port_z0[i]    = 50.0;
        if (i < n_modes) {
            st->port_modes[i] = atoi(mode_toks[i]);
            const char* colon = strchr(mode_toks[i], ':');
            if (colon) {
                double z = atof(colon + 1);
                if (z > 0.0) st->port_z0[i] = z;
            }
        } else if (n_modes == 1) {
            st->port_modes[i] = st->port_modes[0];
            st->port_z0[i]    = st->port_z0[0];
        }
    }

    if (st->use_socket) {
#if defined(_WIN32)
        fprintf(stderr, "[josim_spectre_dpi] ERROR: POSIX socket IPC not supported on native Win32 build\n");
        return -1;
#else
        st->sock_fd = josim_ipc_connect(st->endpoint);
        if (st->sock_fd < 0) {
            fprintf(stderr, "[josim_spectre_dpi] ERROR: failed to connect to endpoint '%s'\n", st->endpoint);
            return -1;
        }
        size_t   nl_len  = strlen(st->netlist_path);
        size_t   cap     = 64 + nl_len + (size_t)n_nodes * 256;
        uint8_t* buf     = (uint8_t*)calloc(1, cap);
        uint8_t* p       = buf;
        int32_t  pm32    = 1; /* phase mode */
        int32_t  np32    = (int32_t)n_nodes;
        uint32_t nllen32 = (uint32_t)nl_len;
        memcpy(p, &pm32, 4);
        memcpy(p + 4, &np32, 4);
        memcpy(p + 8, &nllen32, 4);
        p += 12;
        if (nl_len > 0) {
            memcpy(p, st->netlist_path, nl_len);
            p += nl_len;
        }
        for (int i = 0; i < n_nodes; ++i) {
            char p_node[128], n_node[128];
            const char* colon = strchr(node_toks[i], ':');
            if (colon) {
                size_t plen = (size_t)(colon - node_toks[i]);
                if (plen >= sizeof(p_node)) plen = sizeof(p_node) - 1;
                memcpy(p_node, node_toks[i], plen);
                p_node[plen] = '\0';
                strncpy(n_node, colon + 1, sizeof(n_node) - 1);
                n_node[sizeof(n_node) - 1] = '\0';
            } else {
                strncpy(p_node, node_toks[i], sizeof(p_node) - 1);
                p_node[sizeof(p_node) - 1] = '\0';
                strcpy(n_node, "0");
            }
            int32_t  md = (int32_t)st->port_modes[i];
            double   z0 = st->port_z0[i];
            uint16_t pl = (uint16_t)strlen(p_node);
            uint16_t nl = (uint16_t)strlen(n_node);
            memcpy(p, &md, 4);
            memcpy(p + 4, &z0, 8);
            memcpy(p + 12, &pl, 2);
            memcpy(p + 14, &nl, 2);
            p += 16;
            memcpy(p, p_node, pl);
            p += pl;
            memcpy(p, n_node, nl);
            p += nl;
        }
        uint64_t pay_len = (uint64_t)(p - buf);
        int rc = josim_ipc_send_req(st->sock_fd, JOSIM_IPC_OP_INIT_FILE, buf, pay_len);
        free(buf);
        if (rc != 0) {
            fprintf(stderr, "[josim_spectre_dpi] ERROR: failed to send INIT_FILE to '%s'\n", st->endpoint);
            return -1;
        }
        uint8_t  resp[32];
        int32_t  status  = 0;
        uint64_t out_len = 0;
        if (josim_ipc_recv_resp(st->sock_fd, &status, resp, sizeof(resp), &out_len) != 0) {
            fprintf(stderr, "[josim_spectre_dpi] ERROR: remote INIT_FILE failed on '%s'\n", st->endpoint);
            return -1;
        }
        st->n_ports = n_nodes;
        st->tstep   = 1e-13;
        if (out_len >= 8) {
            memcpy(&st->tstep, resp, sizeof(double));
        }
#endif
    } else {
#ifdef JOSIM_DPI_SOCKET_ONLY
        fprintf(stderr, "[josim_spectre_dpi] ERROR: compiled with JOSIM_DPI_SOCKET_ONLY but no socket endpoint specified\n");
        return -1;
#else
        st->session = josim_session_create();
        if (!st->session) return -1;
        const char* argv[2] = {"-a", "1"};
        josim_set_cli_args(st->session, 2, argv);
        if (josim_load_netlist_file(st->session, st->netlist_path) != 0) {
            fprintf(stderr, "[josim_spectre_dpi] ERROR: load_netlist_file('%s'): %s\n",
                    st->netlist_path, josim_session_last_error(st->session));
            return -1;
        }
        for (int i = 0; i < n_nodes; ++i) {
            char p_node[128], n_node[128];
            const char* colon = strchr(node_toks[i], ':');
            if (colon) {
                size_t plen = (size_t)(colon - node_toks[i]);
                if (plen >= sizeof(p_node)) plen = sizeof(p_node) - 1;
                memcpy(p_node, node_toks[i], plen);
                p_node[plen] = '\0';
                strncpy(n_node, colon + 1, sizeof(n_node) - 1);
                n_node[sizeof(n_node) - 1] = '\0';
            } else {
                strncpy(p_node, node_toks[i], sizeof(p_node) - 1);
                p_node[sizeof(p_node) - 1] = '\0';
                strcpy(n_node, "0");
            }
            if (josim_cosim_add_port(st->session, p_node, n_node, st->port_modes[i], st->port_z0[i]) < 0) {
                fprintf(stderr, "[josim_spectre_dpi] ERROR: add_port('%s'): %s\n",
                        node_toks[i], josim_session_last_error(st->session));
                return -1;
            }
        }
        if (josim_cosim_init(st->session) != 0) {
            fprintf(stderr, "[josim_spectre_dpi] ERROR: cosim_init: %s\n",
                    josim_session_last_error(st->session));
            return -1;
        }
        st->n_ports = (int)josim_cosim_num_ports(st->session);
        st->tstep   = josim_cosim_tstep(st->session);
#endif
    }

    st->has_cache      = 0;
    st->cached_t       = -1.0;
    st->rollback_count = 0;
    for (int i = 0; i < JOSIM_DPI_MAX_PORTS; ++i) {
        st->staged_in[i]  = 0.0;
        st->cached_in[i]  = 0.0;
        st->cached_out[i] = 0.0;
        st->cached_jac[i] = JOSIM_DPI_MIN_G;
    }
    return 0;
}

JOSIM_DPI_API int josim_dpi_open(
    const char* netlist_path,
    const char* port_nodes_csv,
    const char* port_modes_csv,
    const char* endpoint) {
    if (!g_dpi_initialized) {
        for (int i = 0; i < JOSIM_DPI_MAX_SESSIONS; ++i) {
            memset(&g_dpi_slots[i], 0, sizeof(josim_dpi_slot_t));
            g_dpi_slots[i].sock_fd = -1;
        }
        g_dpi_initialized = 1;
    }

    const char* nl = netlist_path   ? netlist_path   : "";
    const char* pn = port_nodes_csv ? port_nodes_csv : "";
    const char* pm = port_modes_csv ? port_modes_csv : "0";
    const char* ep = endpoint       ? endpoint       : "";
    if (ep[0] == '\0' || strcmp(ep, "env") == 0) {
        const char* env_ep = getenv("JOSIM_COSIM_ENDPOINT");
        if (env_ep && env_ep[0] != '\0') {
            ep = env_ep;
        }
    }
    int use_sock = (ep[0] != '\0' && strcmp(ep, "inproc") != 0) ? 1 : 0;

    // Idempotent check: Spectre calls @(initial_step) across multiple NR iterations at DC and t=0.
    for (int i = 0; i < JOSIM_DPI_MAX_SESSIONS; ++i) {
        josim_dpi_slot_t* st = &g_dpi_slots[i];
        if (st->active &&
            st->use_socket == use_sock &&
            strcmp(st->netlist_path, nl) == 0 &&
            strcmp(st->port_nodes_csv, pn) == 0 &&
            strcmp(st->port_modes_csv, pm) == 0 &&
            strcmp(st->endpoint, ep) == 0) {
            if (!st->has_cache || st->cached_t <= 1e-18) {
                return i + 1;
            }
            dpi_slot_free(st);
            st->active     = 1;
            st->use_socket = use_sock;
            strncpy(st->netlist_path, nl, sizeof(st->netlist_path) - 1);
            strncpy(st->port_nodes_csv, pn, sizeof(st->port_nodes_csv) - 1);
            strncpy(st->port_modes_csv, pm, sizeof(st->port_modes_csv) - 1);
            strncpy(st->endpoint, ep, sizeof(st->endpoint) - 1);
            if (dpi_init_slot_backend(st) != 0) {
                dpi_slot_free(st);
                return -1;
            }
            return i + 1;
        }
    }

    int slot_idx = -1;
    for (int i = 0; i < JOSIM_DPI_MAX_SESSIONS; ++i) {
        if (!g_dpi_slots[i].active) {
            slot_idx = i;
            break;
        }
    }
    if (slot_idx < 0) {
        fprintf(stderr, "[josim_spectre_dpi] ERROR: maximum concurrent sessions (%d) exceeded\n",
                JOSIM_DPI_MAX_SESSIONS);
        return -1;
    }

    josim_dpi_slot_t* st = &g_dpi_slots[slot_idx];
    st->active     = 1;
    st->use_socket = use_sock;
    strncpy(st->netlist_path, nl, sizeof(st->netlist_path) - 1);
    strncpy(st->port_nodes_csv, pn, sizeof(st->port_nodes_csv) - 1);
    strncpy(st->port_modes_csv, pm, sizeof(st->port_modes_csv) - 1);
    strncpy(st->endpoint, ep, sizeof(st->endpoint) - 1);

    if (dpi_init_slot_backend(st) != 0) {
        dpi_slot_free(st);
        return -1;
    }
    return slot_idx + 1;
}

static josim_dpi_slot_t* get_slot(int handle) {
    if (!g_dpi_initialized || handle < 1 || handle > JOSIM_DPI_MAX_SESSIONS) return NULL;
    josim_dpi_slot_t* st = &g_dpi_slots[handle - 1];
    return st->active ? st : NULL;
}

JOSIM_DPI_API double josim_dpi_tstep(int handle) {
    josim_dpi_slot_t* st = get_slot(handle);
    return st ? st->tstep : 0.0;
}

JOSIM_DPI_API int josim_dpi_num_ports(int handle) {
    josim_dpi_slot_t* st = get_slot(handle);
    return st ? st->n_ports : 0;
}

JOSIM_DPI_API int josim_dpi_rollback_count(int handle) {
    josim_dpi_slot_t* st = get_slot(handle);
    if (!st) return 0;
#ifndef JOSIM_DPI_SOCKET_ONLY
    if (!st->use_socket && st->session) {
        return (int)josim_cosim_rollback_count(st->session);
    }
#endif
    return (int)st->rollback_count;
}

JOSIM_DPI_API int josim_dpi_set_input(int handle, int port_idx, double val) {
    josim_dpi_slot_t* st = get_slot(handle);
    if (!st || port_idx < 0 || port_idx >= st->n_ports) return -1;
    st->staged_in[port_idx] = val;
    return 0;
}

JOSIM_DPI_API int josim_dpi_eval(int handle, double t_s) {
    josim_dpi_slot_t* st = get_slot(handle);
    if (!st || st->n_ports <= 0) return -1;
    if (t_s < 0.0) t_s = 0.0;

    // Same-evaluation cache: if neither t_s nor any staged input changed, reuse cached outputs/Jacobians.
    if (st->has_cache && fabs(t_s - st->cached_t) <= 1e-21) {
        int same_inputs = 1;
        for (int p = 0; p < st->n_ports; ++p) {
            if (fabs(st->staged_in[p] - st->cached_in[p]) > 1e-18) {
                same_inputs = 0;
                break;
            }
        }
        if (same_inputs) {
            return 0;
        }
    }

    int nP = st->n_ports;
    if (st->use_socket) {
#if defined(_WIN32)
        return -1;
#else
        double req_buf[1 + JOSIM_DPI_MAX_PORTS];
        double resp_buf[2 * JOSIM_DPI_MAX_PORTS + 2];
        req_buf[0] = t_s;
        for (int p = 0; p < nP; ++p) {
            req_buf[1 + p] = st->staged_in[p];
        }
        if (josim_ipc_send_req(st->sock_fd, JOSIM_IPC_OP_STEP_TO,
                               req_buf, (uint64_t)((1 + nP) * sizeof(double))) != 0) {
            return -1;
        }
        int32_t  status  = 0;
        uint64_t out_len = 0;
        if (josim_ipc_recv_resp(st->sock_fd, &status, resp_buf,
                                (uint64_t)((2 * nP + 2) * sizeof(double)), &out_len) != 0) {
            return -1;
        }
        for (int p = 0; p < nP; ++p) {
            st->cached_out[p] = resp_buf[p];
            double jac        = resp_buf[nP + p];
            st->cached_jac[p] = (fabs(jac) < JOSIM_DPI_MIN_G) ? JOSIM_DPI_MIN_G : jac;
        }
        memcpy(&st->rollback_count, &resp_buf[2 * nP + 1], sizeof(int64_t));
#endif
    } else {
#ifdef JOSIM_DPI_SOCKET_ONLY
        return -1;
#else
        if (josim_cosim_step_to(
                st->session, t_s,
                st->staged_in, st->cached_out, st->cached_jac) != 0) {
            return -1;
        }
        for (int p = 0; p < nP; ++p) {
            if (fabs(st->cached_jac[p]) < JOSIM_DPI_MIN_G) {
                st->cached_jac[p] = JOSIM_DPI_MIN_G;
            }
        }
#endif
    }

    st->cached_t  = t_s;
    st->has_cache = 1;
    for (int p = 0; p < nP; ++p) {
        st->cached_in[p] = st->staged_in[p];
    }
    return 0;
}

JOSIM_DPI_API double josim_dpi_get_output(int handle, int port_idx) {
    josim_dpi_slot_t* st = get_slot(handle);
    if (!st || port_idx < 0 || port_idx >= st->n_ports) return 0.0;
    return st->cached_out[port_idx];
}

JOSIM_DPI_API double josim_dpi_get_jacobian(int handle, int port_idx) {
    josim_dpi_slot_t* st = get_slot(handle);
    if (!st || port_idx < 0 || port_idx >= st->n_ports) return JOSIM_DPI_MIN_G;
    return st->cached_jac[port_idx];
}

JOSIM_DPI_API double josim_dpi_get_hist(int handle, int port_idx) {
    josim_dpi_slot_t* st = get_slot(handle);
    if (!st || port_idx < 0 || port_idx >= st->n_ports) return 0.0;
    return st->cached_out[port_idx] - st->cached_jac[port_idx] * st->cached_in[port_idx];
}

// Single-port Norton convenience functions (port 0: V -> I, G_eq)
JOSIM_DPI_API double josim_dpi_eval_i(int handle, double t_s, double v_pn) {
    if (josim_dpi_set_input(handle, 0, v_pn) != 0) return 0.0;
    if (josim_dpi_eval(handle, t_s) != 0) return 0.0;
    return josim_dpi_get_output(handle, 0);
}

JOSIM_DPI_API double josim_dpi_eval_ihist(int handle, double t_s, double v_pn) {
    if (josim_dpi_set_input(handle, 0, v_pn) != 0) return 0.0;
    if (josim_dpi_eval(handle, t_s) != 0) return 0.0;
    return josim_dpi_get_hist(handle, 0);
}

JOSIM_DPI_API double josim_dpi_eval_g(int handle, double t_s, double v_pn) {
    if (josim_dpi_set_input(handle, 0, v_pn) != 0) return JOSIM_DPI_MIN_G;
    if (josim_dpi_eval(handle, t_s) != 0) return JOSIM_DPI_MIN_G;
    return josim_dpi_get_jacobian(handle, 0);
}

// Single-port Thevenin convenience functions (port 0: I -> V, R_eq)
JOSIM_DPI_API double josim_dpi_eval_v(int handle, double t_s, double i_pn) {
    if (josim_dpi_set_input(handle, 0, i_pn) != 0) return 0.0;
    if (josim_dpi_eval(handle, t_s) != 0) return 0.0;
    return josim_dpi_get_output(handle, 0);
}

JOSIM_DPI_API double josim_dpi_eval_vhist(int handle, double t_s, double i_pn) {
    if (josim_dpi_set_input(handle, 0, i_pn) != 0) return 0.0;
    if (josim_dpi_eval(handle, t_s) != 0) return 0.0;
    return josim_dpi_get_hist(handle, 0);
}

JOSIM_DPI_API double josim_dpi_eval_r(int handle, double t_s, double i_pn) {
    if (josim_dpi_set_input(handle, 0, i_pn) != 0) return JOSIM_DPI_MIN_G;
    if (josim_dpi_eval(handle, t_s) != 0) return JOSIM_DPI_MIN_G;
    return josim_dpi_get_jacobian(handle, 0);
}

JOSIM_DPI_API int josim_dpi_close(int handle) {
    josim_dpi_slot_t* st = get_slot(handle);
    if (!st) return -1;
    dpi_slot_free(st);
    return 0;
}

JOSIM_DPI_API void josim_dpi_close_all(void) {
    if (!g_dpi_initialized) return;
    for (int i = 0; i < JOSIM_DPI_MAX_SESSIONS; ++i) {
        if (g_dpi_slots[i].active) {
            dpi_slot_free(&g_dpi_slots[i]);
        }
    }
}
