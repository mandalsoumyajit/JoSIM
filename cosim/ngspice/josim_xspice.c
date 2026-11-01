/*
 * Native ngspice XSPICE Code Model for JoSIM Superconducting Co-Simulation (josim_cosim)
 *
 * Supports:
 *   - Mode A (Step 1 In-Process): Direct calls into libjosim_c.so (server_addr = "")
 *   - Mode B (Step 2 Out-of-Process IPC): UNIX/TCP socket connection to josim-cosim-server
 *   - Multi-port differential conductance (%gd: V in -> I out, dI/dV = G_eq) and
 *     differential resistance (%hd: I in -> V out, dV/dI = R_eq) ports
 *   - Automatic Newton-Raphson same-time re-evaluation and LTE step-rejection rollback
 */

#include "ngspice/ngspice.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/mifdefs.h"
#include "ngspice/mifproto.h"
#include "ngspice/mifparse.h"
#include "ngspice/cm.h"

#include "JoSIM/josim_c_api.h"
#include "JoSIM/josim_cosim_ipc.h"

typedef struct JosimXspiceState_s {
    int                        initialized;
    int                        finalized;
    int                        use_socket;
    int                        sock_fd;
    josim_session_t           *sess;
    int                        n_ports;
    double                    *u_in;
    double                    *y_out;
    double                    *dy_du;
    char                      *out_csv;
    int64_t                    last_rb_count;
    struct JosimXspiceState_s *next;
} JosimXspiceState_t;

static JosimXspiceState_t *g_active_states = NULL;

static void write_inproc_csv(josim_session_t *sess, const char *csv_path) {
    if (!sess || !csv_path || !csv_path[0]) { return; }
    int64_t n_tr = josim_num_traces(sess);
    int64_t n_st = josim_num_steps(sess);
    if (n_tr > 0 && n_st > 0) {
        FILE *fp = fopen(csv_path, "w");
        if (fp) {
            const double **ptrs = (const double **) calloc((size_t) n_tr, sizeof(double *));
            for (int64_t c = 0; c < n_tr; ++c) {
                if (c > 0) { fputc(',', fp); }
                const char *nm = josim_trace_name(sess, c);
                fputs(nm ? nm : "", fp);
                if (ptrs) { ptrs[c] = josim_trace_data(sess, c); }
            }
            fputc('\n', fp);
            if (ptrs) {
                for (int64_t r = 0; r < n_st; ++r) {
                    for (int64_t c = 0; c < n_tr; ++c) {
                        if (c > 0) { fputc(',', fp); }
                        fprintf(fp, "%.10e", ptrs[c] ? ptrs[c][r] : 0.0);
                    }
                    fputc('\n', fp);
                }
                free((void *) ptrs);
            }
            fclose(fp);
        }
    }

    char ev_path[1024];
    snprintf(ev_path, sizeof(ev_path), "%s.events.csv", csv_path);
    FILE *efp = fopen(ev_path, "w");
    if (efp) {
        fprintf(efp, "trace,time,t_peak,v_peak,fwhm,dt_pi,area,phase,polarity,winding,trace_id,trial\n");
        int64_t                  n_ev = josim_num_events(sess);
        const josim_sfq_event_t *evs  = josim_events_data(sess);
        for (int64_t i = 0; i < n_ev; ++i) {
            const josim_sfq_event_t *e  = &evs[i];
            const char              *tn = josim_event_trace_name(sess, e->trace_id);
            fprintf(efp,
                    "%s,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%d,%d,%d,%d\n",
                    tn ? tn : "",
                    e->time,
                    e->t_peak,
                    e->v_peak,
                    e->fwhm,
                    e->dt_pi,
                    e->area,
                    e->phase,
                    e->polarity,
                    e->winding,
                    e->trace_id,
                    e->trial);
        }
        fclose(efp);
    }

    char meta_path[1024];
    snprintf(meta_path, sizeof(meta_path), "%s.meta.txt", csv_path);
    FILE *mfp = fopen(meta_path, "w");
    if (mfp) {
        fprintf(mfp, "rollback_count=%lld\n", (long long) josim_cosim_rollback_count(sess));
        fclose(mfp);
    }
}

static void finalize_xspice_state(JosimXspiceState_t *st) {
    if (!st || st->finalized) { return; }
    st->finalized = 1;
    if (st->use_socket) {
        if (st->sock_fd >= 0) {
            const char *csv = (st->out_csv && st->out_csv[0]) ? st->out_csv : "";
            uint64_t    len = (uint64_t) strlen(csv);
            if (josim_ipc_send_req(st->sock_fd, JOSIM_IPC_OP_FINALIZE, csv, len) == 0) {
                int64_t  resp[4] = {0, 0, 0, 0};
                int32_t  status  = 0;
                uint64_t out_len = 0;
                if (josim_ipc_recv_resp(st->sock_fd, &status, resp, sizeof(resp), &out_len) == 0) {
                    st->last_rb_count = resp[3];
                    if (csv[0]) {
                        char meta_path[1024];
                        snprintf(meta_path, sizeof(meta_path), "%s.meta.txt", csv);
                        FILE *mfp = fopen(meta_path, "w");
                        if (mfp) {
                            fprintf(mfp, "rollback_count=%lld\n", (long long) st->last_rb_count);
                            fclose(mfp);
                        }
                    }
                }
            }
            close(st->sock_fd);
            st->sock_fd = -1;
        }
    } else if (st->sess) {
        josim_cosim_finalize(st->sess);
        if (st->out_csv && st->out_csv[0]) {
            write_inproc_csv(st->sess, st->out_csv);
        }
        josim_session_destroy(st->sess);
        st->sess = NULL;
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((destructor))
#endif
static void josim_xspice_atexit_cleanup(void) {
    JosimXspiceState_t *cur = g_active_states;
    while (cur) {
        finalize_xspice_state(cur);
        cur = cur->next;
    }
    g_active_states = NULL;
}

static void cm_josim_cosim_callback(Mif_Private_t *mif_private, Mif_Callback_Reason_t reason) {
    if (reason == MIF_CB_DESTROY) {
        JosimXspiceState_t *st = (JosimXspiceState_t *) mif_private->inst_var[0]->element[0].pvalue;
        if (!st) { return; }
        finalize_xspice_state(st);

        JosimXspiceState_t **pp = &g_active_states;
        while (*pp) {
            if (*pp == st) {
                *pp = st->next;
                break;
            }
            pp = &((*pp)->next);
        }
        free(st->u_in);
        free(st->y_out);
        free(st->dy_du);
        free(st->out_csv);
        free(st);
        mif_private->inst_var[0]->element[0].pvalue = NULL;
    }
}

void cm_josim_cosim(Mif_Private_t *mif_private) {
    if (mif_private->circuit.anal_type == MIF_AC) { return; }

    int n_ports = mif_private->conn[0]->size;
    if (n_ports <= 0) { return; }

    if (mif_private->circuit.init == 1) {
        JosimXspiceState_t *st = (JosimXspiceState_t *) calloc(1, sizeof(JosimXspiceState_t));
        mif_private->inst_var[0]->element[0].pvalue = st;
        *(mif_private->callback)                    = cm_josim_cosim_callback;

        st->n_ports = n_ports;
        st->sock_fd = -1;
        st->u_in    = (double *) calloc((size_t) n_ports, sizeof(double));
        st->y_out   = (double *) calloc((size_t) n_ports, sizeof(double));
        st->dy_du   = (double *) calloc((size_t) n_ports, sizeof(double));

        const char *netlist     = (!mif_private->param[0]->is_null) ? mif_private->param[0]->element[0].svalue : "";
        int         phase_mode  = (!mif_private->param[4]->is_null) ? mif_private->param[4]->element[0].ivalue : 1;
        const char *server_addr = (!mif_private->param[5]->is_null) ? mif_private->param[5]->element[0].svalue : "";
        const char *out_csv     = (!mif_private->param[6]->is_null) ? mif_private->param[6]->element[0].svalue : "";

        if (out_csv && out_csv[0]) { st->out_csv = strdup(out_csv); }

        if (server_addr && server_addr[0]) {
            st->use_socket = 1;
            st->sock_fd    = josim_ipc_connect(server_addr);
            if (st->sock_fd < 0) {
                cm_message_printf("josim_cosim ERROR: Failed to connect to josim-cosim-server at '%s'", server_addr);
                return;
            }
            size_t   nl_len  = strlen(netlist);
            size_t   cap     = 64 + nl_len + (size_t) n_ports * 256;
            uint8_t *buf     = (uint8_t *) calloc(1, cap);
            uint8_t *p       = buf;
            int32_t  pm32    = (int32_t) phase_mode;
            int32_t  np32    = (int32_t) n_ports;
            uint32_t nllen32 = (uint32_t) nl_len;
            memcpy(p, &pm32, 4);
            memcpy(p + 4, &np32, 4);
            memcpy(p + 8, &nllen32, 4);
            p += 12;
            if (nl_len > 0) {
                memcpy(p, netlist, nl_len);
                p += nl_len;
            }
            for (int i = 0; i < n_ports; ++i) {
                const char     *pos   = (!mif_private->param[1]->is_null && i < mif_private->param[1]->size)
                                                ? mif_private->param[1]->element[i].svalue
                                                : "1";
                const char     *neg   = (!mif_private->param[2]->is_null && i < mif_private->param[2]->size)
                                                ? mif_private->param[2]->element[i].svalue
                                                : "0";
                Mif_Port_Type_t ptype = mif_private->conn[0]->port[i]->type;
                int32_t         def_m = (ptype == MIF_DIFF_RESISTANCE || ptype == MIF_RESISTANCE) ? 1 : 0;
                int32_t         md    = (!mif_private->param[3]->is_null && i < mif_private->param[3]->size &&
                                         mif_private->param[3]->element[i].ivalue >= 0)
                                                ? (int32_t) mif_private->param[3]->element[i].ivalue
                                                : def_m;
                double          z0    = 50.0;
                uint16_t        pl    = (uint16_t) strlen(pos);
                uint16_t        nl    = (uint16_t) strlen(neg);
                memcpy(p, &md, 4);
                memcpy(p + 4, &z0, 8);
                memcpy(p + 12, &pl, 2);
                memcpy(p + 14, &nl, 2);
                p += 16;
                memcpy(p, pos, pl);
                p += pl;
                memcpy(p, neg, nl);
                p += nl;
            }
            uint64_t pay_len = (uint64_t) (p - buf);
            int      rc      = josim_ipc_send_req(st->sock_fd, JOSIM_IPC_OP_INIT_FILE, buf, pay_len);
            free(buf);
            if (rc != 0) {
                cm_message_printf("josim_cosim ERROR: Failed to send INIT to server");
                return;
            }
            uint8_t  resp[32];
            int32_t  status  = 0;
            uint64_t out_len = 0;
            if (josim_ipc_recv_resp(st->sock_fd, &status, resp, sizeof(resp), &out_len) != 0) {
                cm_message_printf("josim_cosim ERROR: Server INIT failed");
                return;
            }
            st->initialized = 1;
        } else {
            st->use_socket = 0;
            st->sess       = josim_session_create();
            if (!st->sess) {
                cm_message_printf("josim_cosim ERROR: Failed to create josim_session_t");
                return;
            }
            const char *mode_str = (phase_mode == 1) ? "1" : "0";
            const char *argv[2]  = {"-a", mode_str};
            josim_set_cli_args(st->sess, 2, argv);
            if (josim_load_netlist_file(st->sess, netlist) != 0) {
                cm_message_printf("josim_cosim ERROR: %s", josim_session_last_error(st->sess));
                return;
            }
            for (int i = 0; i < n_ports; ++i) {
                const char     *pos   = (!mif_private->param[1]->is_null && i < mif_private->param[1]->size)
                                                ? mif_private->param[1]->element[i].svalue
                                                : "1";
                const char     *neg   = (!mif_private->param[2]->is_null && i < mif_private->param[2]->size)
                                                ? mif_private->param[2]->element[i].svalue
                                                : "0";
                Mif_Port_Type_t ptype = mif_private->conn[0]->port[i]->type;
                int             def_m = (ptype == MIF_DIFF_RESISTANCE || ptype == MIF_RESISTANCE) ? 1 : 0;
                int             md    = (!mif_private->param[3]->is_null && i < mif_private->param[3]->size &&
                                         mif_private->param[3]->element[i].ivalue >= 0)
                                                ? mif_private->param[3]->element[i].ivalue
                                                : def_m;
                josim_cosim_add_port(st->sess, pos, neg, md, 50.0);
            }
            if (josim_cosim_init(st->sess) != 0) {
                cm_message_printf("josim_cosim ERROR: %s", josim_session_last_error(st->sess));
                return;
            }
            st->initialized = 1;
        }

        st->next        = g_active_states;
        g_active_states = st;
    }

    JosimXspiceState_t *st = (JosimXspiceState_t *) mif_private->inst_var[0]->element[0].pvalue;
    if (!st || !st->initialized) { return; }

    for (int i = 0; i < n_ports; ++i) {
        st->u_in[i] = mif_private->conn[0]->port[i]->input.rvalue;
    }

    double t_eval = (mif_private->circuit.anal_type == MIF_DC) ? 0.0 : mif_private->circuit.time;

    if (st->use_socket) {
        size_t  req_doubles = (size_t) (1 + n_ports);
        double  stack_req[32];
        double *req_buf = (req_doubles <= 32) ? stack_req : (double *) malloc(req_doubles * sizeof(double));
        req_buf[0]      = t_eval;
        memcpy(&req_buf[1], st->u_in, (size_t) n_ports * sizeof(double));
        if (josim_ipc_send_req(st->sock_fd, JOSIM_IPC_OP_STEP_TO, req_buf, req_doubles * sizeof(double)) == 0) {
            size_t   resp_doubles = (size_t) (2 * n_ports + 2);
            double   stack_resp[64];
            double  *resp_buf = (resp_doubles <= 64) ? stack_resp : (double *) malloc(resp_doubles * sizeof(double));
            int32_t  status   = 0;
            uint64_t out_len  = 0;
            if (josim_ipc_recv_resp(st->sock_fd, &status, resp_buf, resp_doubles * sizeof(double), &out_len) == 0) {
                memcpy(st->y_out, resp_buf, (size_t) n_ports * sizeof(double));
                memcpy(st->dy_du, resp_buf + n_ports, (size_t) n_ports * sizeof(double));
                memcpy(&st->last_rb_count, &resp_buf[2 * n_ports + 1], sizeof(int64_t));
            }
            if (resp_buf != stack_resp) { free(resp_buf); }
        }
        if (req_buf != stack_req) { free(req_buf); }
    } else {
        josim_cosim_step_to(st->sess, t_eval, st->u_in, st->y_out, st->dy_du);
    }

    for (int i = 0; i < n_ports; ++i) {
        mif_private->conn[0]->port[i]->output.rvalue = st->y_out[i];
        for (int j = 0; j < n_ports; ++j) {
            mif_private->conn[0]->port[i]->partial[0].port[j] = (i == j) ? st->dy_du[i] : 0.0;
        }
    }
}

static IFparm MIFmPTable[] = {
    IOP("netlist", 0, IF_STRING, "JoSIM netlist file path"),
    IOP("pos_nodes", 1, (IF_STRING | IF_VECTOR), "JoSIM positive port node names"),
    IOP("neg_nodes", 2, (IF_STRING | IF_VECTOR), "JoSIM negative port node names"),
    IOP("port_modes", 3, (IF_INTEGER | IF_VECTOR), "JoSIM port modes (0=gd V->I, 1=hd I->V)"),
    IOP("phase_mode", 4, IF_INTEGER, "JoSIM analysis type (1=phase, 0=voltage)"),
    IOP("server_addr", 5, IF_STRING, "Optional josim-cosim-server address (unix:... or tcp:...)"),
    IOP("out_csv", 6, IF_STRING, "Optional output CSV file path for JoSIM traces and events"),
};

static IFparm MIFpTable[] = {
    OP("locdata", 7, IF_STRING, "local static state"),
};

static Mif_Port_Type_t MIFportEnum0[] = {
    MIF_DIFF_CONDUCTANCE,
    MIF_DIFF_RESISTANCE,
    MIF_CONDUCTANCE,
    MIF_RESISTANCE,
};

static char *MIFportStr0[] = {
    "gd",
    "hd",
    "g",
    "h",
};

static Mif_Conn_Info_t MIFconnTable[] = {
    {
        "port",
        "co-simulation boundary port array",
        MIF_INOUT,
        MIF_DIFF_CONDUCTANCE,
        "gd",
        4,
        MIFportEnum0,
        MIFportStr0,
        MIF_TRUE,
        MIF_TRUE,
        1,
        MIF_FALSE,
        0,
        MIF_FALSE,
    },
};

static union Mif_Parse_Value port_modes_default[] = {{.ivalue = -1}};
static union Mif_Parse_Value phase_mode_default[] = {{.ivalue = 1}};

static Mif_Param_Info_t MIFparamTable[] = {
    {
        "netlist",
        "JoSIM netlist file path",
        MIF_STRING,
        0,
        NULL,
        MIF_FALSE,
        {.bvalue = MIF_FALSE},
        MIF_FALSE,
        {.bvalue = MIF_FALSE},
        MIF_FALSE,
        MIF_FALSE,
        0,
        MIF_FALSE,
        0,
        MIF_FALSE,
        0,
        MIF_TRUE,
    },
    {
        "pos_nodes",
        "JoSIM positive port node names",
        MIF_STRING,
        0,
        NULL,
        MIF_FALSE,
        {.bvalue = MIF_FALSE},
        MIF_FALSE,
        {.bvalue = MIF_FALSE},
        MIF_TRUE,
        MIF_FALSE,
        0,
        MIF_TRUE,
        1,
        MIF_FALSE,
        0,
        MIF_TRUE,
    },
    {
        "neg_nodes",
        "JoSIM negative port node names",
        MIF_STRING,
        0,
        NULL,
        MIF_FALSE,
        {.bvalue = MIF_FALSE},
        MIF_FALSE,
        {.bvalue = MIF_FALSE},
        MIF_TRUE,
        MIF_FALSE,
        0,
        MIF_TRUE,
        1,
        MIF_FALSE,
        0,
        MIF_TRUE,
    },
    {
        "port_modes",
        "JoSIM port modes",
        MIF_INTEGER,
        1,
        port_modes_default,
        MIF_FALSE,
        {.bvalue = MIF_FALSE},
        MIF_FALSE,
        {.bvalue = MIF_FALSE},
        MIF_TRUE,
        MIF_FALSE,
        0,
        MIF_TRUE,
        1,
        MIF_FALSE,
        0,
        MIF_TRUE,
    },
    {
        "phase_mode",
        "JoSIM analysis type",
        MIF_INTEGER,
        1,
        phase_mode_default,
        MIF_FALSE,
        {.bvalue = MIF_FALSE},
        MIF_FALSE,
        {.bvalue = MIF_FALSE},
        MIF_FALSE,
        MIF_FALSE,
        0,
        MIF_FALSE,
        0,
        MIF_FALSE,
        0,
        MIF_TRUE,
    },
    {
        "server_addr",
        "Optional josim-cosim-server address",
        MIF_STRING,
        0,
        NULL,
        MIF_FALSE,
        {.bvalue = MIF_FALSE},
        MIF_FALSE,
        {.bvalue = MIF_FALSE},
        MIF_FALSE,
        MIF_FALSE,
        0,
        MIF_FALSE,
        0,
        MIF_FALSE,
        0,
        MIF_TRUE,
    },
    {
        "out_csv",
        "Optional output CSV file path",
        MIF_STRING,
        0,
        NULL,
        MIF_FALSE,
        {.bvalue = MIF_FALSE},
        MIF_FALSE,
        {.bvalue = MIF_FALSE},
        MIF_FALSE,
        MIF_FALSE,
        0,
        MIF_FALSE,
        0,
        MIF_FALSE,
        0,
        MIF_TRUE,
    },
};

static Mif_Inst_Var_Info_t MIFinst_varTable[] = {
    {
        "locdata",
        "local static state",
        MIF_STRING,
        MIF_FALSE,
    },
};

static int val_terms             = 0;
static int val_numNames          = 0;
static int val_numInstanceParms  = 1;
static int val_numModelParms     = 7;
static int val_sizeofMIFinstance = sizeof(MIFinstance);
static int val_sizeofMIFmodel    = sizeof(MIFmodel);

SPICEdev cm_josim_cosim_info = {
    .DEVpublic =
        {
            .name             = "josim_cosim",
            .description      = "JoSIM Superconducting Co-Simulation Boundary Port",
            .terms            = &val_terms,
            .numNames         = &val_numNames,
            .termNames        = NULL,
            .numInstanceParms = &val_numInstanceParms,
            .instanceParms    = MIFpTable,
            .numModelParms    = &val_numModelParms,
            .modelParms       = MIFmPTable,
            .flags            = 0,

            .cm_func      = cm_josim_cosim,
            .num_conn     = 1,
            .conn         = MIFconnTable,
            .num_param    = 7,
            .param        = MIFparamTable,
            .num_inst_var = 1,
            .inst_var     = MIFinst_varTable,
        },

    .DEVparam      = NULL,
    .DEVmodParam   = MIFmParam,
    .DEVload       = MIFload,
    .DEVsetup      = MIFsetup,
    .DEVunsetup    = MIFunsetup,
    .DEVpzSetup    = NULL,
    .DEVtemperature = NULL,
    .DEVtrunc      = MIFtrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad     = MIFload,
    .DEVaccept     = NULL,
    .DEVdestroy    = NULL,
    .DEVmodDelete  = MIFmDelete,
    .DEVdelete     = MIFdelete,
    .DEVsetic      = NULL,
    .DEVask        = MIFask,
    .DEVmodAsk     = MIFmAsk,
    .DEVpzLoad     = NULL,
    .DEVconvTest   = MIFconvTest,
    .DEVsenSetup   = NULL,
    .DEVsenLoad    = NULL,
    .DEVsenUpdate  = NULL,
    .DEVsenAcLoad  = NULL,
    .DEVsenPrint   = NULL,
    .DEVsenTrunc   = NULL,
    .DEVdisto      = NULL,
    .DEVnoise      = NULL,
    .DEVsoaCheck   = NULL,
    .DEVinstSize   = &val_sizeofMIFinstance,
    .DEVmodSize    = &val_sizeofMIFmodel,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif

#ifdef KLU
    .DEVbindCSC              = MIFbindCSC,
    .DEVbindCSCComplex       = MIFbindCSCComplex,
    .DEVbindCSCComplexToReal = MIFbindCSCComplexToReal,
#endif
};
