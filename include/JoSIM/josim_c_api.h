// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)
#ifndef JOSIM_C_API_H
#define JOSIM_C_API_H

#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef JOSIM_C_EXPORTS
#define JOSIM_API __declspec(dllexport)
#else
#define JOSIM_API __declspec(dllimport)
#endif
#elif __GNUC__ >= 4
#define JOSIM_API __attribute__((visibility("default")))
#else
#define JOSIM_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct josim_session josim_session_t;

typedef struct {
    double  time;
    double  t_peak;
    double  v_peak;
    double  fwhm;
    double  dt_pi;
    double  area;
    double  phase;
    int32_t polarity;
    int32_t winding;
    int32_t trace_id;
    int32_t trial;
} josim_sfq_event_t;

/* Version and session lifecycle */
JOSIM_API const char*      josim_version(void);
JOSIM_API josim_session_t* josim_session_create(void);
JOSIM_API void             josim_session_destroy(josim_session_t* s);
JOSIM_API const char*      josim_session_last_error(const josim_session_t* s);

/* Netlist loading and configuration (in-memory or from file) */
JOSIM_API int josim_set_cli_args(josim_session_t* s, int argc, const char* const* argv);
JOSIM_API int josim_load_netlist_file(josim_session_t* s, const char* path);
JOSIM_API int josim_load_netlist_string(josim_session_t* s, const char* netlist_text, const char* base_dir);
JOSIM_API int josim_set_parameter(josim_session_t* s, const char* name, double value, const char* subckt);
JOSIM_API int josim_clear_parameter_overrides(josim_session_t* s);

/* Execute simulation in memory (re-usable after josim_set_parameter without re-parsing netlist) */
JOSIM_API int josim_run(josim_session_t* s);

/* Zero-copy waveform trace access */
JOSIM_API int64_t       josim_num_traces(const josim_session_t* s);
JOSIM_API int64_t       josim_num_steps(const josim_session_t* s);
JOSIM_API const char*   josim_trace_name(const josim_session_t* s, int64_t trace_idx);
JOSIM_API char          josim_trace_type(const josim_session_t* s, int64_t trace_idx);
JOSIM_API const double* josim_trace_data(const josim_session_t* s, int64_t trace_idx);
JOSIM_API const double* josim_trace_data_by_name(const josim_session_t* s, const char* name, int64_t* out_len);
JOSIM_API const double* josim_traces_matrix_col_major(josim_session_t* s, int64_t* out_cols, int64_t* out_rows);
JOSIM_API const double* josim_mc_trace_matrix(
    const josim_session_t* s, int64_t* out_trials, int64_t* out_rows, int64_t* out_steps);

/* Zero-copy SFQ switching event access */
JOSIM_API int64_t                  josim_num_events(const josim_session_t* s);
JOSIM_API const josim_sfq_event_t* josim_events_data(const josim_session_t* s);
JOSIM_API int64_t                  josim_num_event_traces(const josim_session_t* s);
JOSIM_API const char*              josim_event_trace_name(const josim_session_t* s, int32_t trace_id);

/* Evaluated parameters (including exported IV_*, SHAPIRO_*, SQUID_*, BETA_L_*, JITTER_* metrics) */
JOSIM_API int64_t     josim_num_parameters(const josim_session_t* s);
JOSIM_API const char* josim_parameter_name(const josim_session_t* s, int64_t idx);
JOSIM_API double      josim_parameter_value(const josim_session_t* s, int64_t idx);
JOSIM_API int         josim_get_parameter(const josim_session_t* s, const char* name, double* out_val);

/* Evaluated .meas / .measure results */
JOSIM_API int64_t       josim_num_measurements(const josim_session_t* s);
JOSIM_API const char*   josim_measurement_name(const josim_session_t* s, int64_t idx);
JOSIM_API double        josim_measurement_value(const josim_session_t* s, int64_t idx);
JOSIM_API const double* josim_measurement_trial_values(const josim_session_t* s, int64_t idx, int64_t* out_trials);

/* Co-simulation port modes */
typedef enum {
    JOSIM_PORT_VOLTAGE_IN_CURRENT_OUT = 0, /* SPICE drives V(p,n), JoSIM returns I_into_JoSIM and G_eq = dI/dV */
    JOSIM_PORT_CURRENT_IN_VOLTAGE_OUT = 1, /* SPICE drives I_into_JoSIM, JoSIM returns V(p,n) and R_eq = dV/dI */
    JOSIM_PORT_WAVE_BRANIN            = 2  /* Z0 wave port: input = V_inc, output = V_ref = V(p,n) - V_inc, diag = 1/Z0 */
} josim_port_mode_t;

/* Step-by-step & windowed co-simulation interface (Layer 1) */
JOSIM_API int32_t josim_cosim_add_port(
    josim_session_t* s, const char* pos_node, const char* neg_node, int32_t mode, double z0);
JOSIM_API int     josim_cosim_clear_ports(josim_session_t* s);
JOSIM_API int32_t josim_cosim_num_ports(const josim_session_t* s);
JOSIM_API int     josim_cosim_init(josim_session_t* s);
JOSIM_API double  josim_cosim_tstep(const josim_session_t* s);
JOSIM_API double  josim_cosim_current_time(const josim_session_t* s);
JOSIM_API int64_t josim_cosim_rollback_count(const josim_session_t* s);
JOSIM_API void    josim_cosim_stats(
    const josim_session_t* s, int64_t* out_calls, int64_t* out_rollbacks, int64_t* out_substeps, double* out_ckpt_sec);

/* Advance (or auto-rollback/re-evaluate) to target_time.
   - If target_time == last evaluated time: re-evaluates from last committed checkpoint (Newton-Raphson iteration).
   - If target_time >  last evaluated time: auto-commits last evaluated state and advances to target_time.
   - If target_time <  last evaluated time: auto-rolls back to last committed state and advances to target_time. */
JOSIM_API int josim_cosim_step_to(
    josim_session_t* s, double target_time,
    const double* port_inputs, double* port_outputs, double* port_jacobian_diag);

JOSIM_API int josim_cosim_commit(josim_session_t* s);
JOSIM_API int josim_cosim_rollback(josim_session_t* s);

/* Run an entire time window [t_committed, t_stop] driven by PWL port waveforms (Branin / WR windowing) */
JOSIM_API int josim_cosim_run_window(
    josim_session_t* s, double t_stop,
    int64_t n_in_pts, const double* t_in_pts, const double* port_in_flat,
    int64_t* out_n_pts, const double** out_t_pts, const double** out_port_out_flat);

/* Finalize co-simulation and run post-processing (.meas, .assert, .jitter, traces, events) */
JOSIM_API int josim_cosim_finalize(josim_session_t* s);

#ifdef __cplusplus
}
#endif

#endif // JOSIM_C_API_H
