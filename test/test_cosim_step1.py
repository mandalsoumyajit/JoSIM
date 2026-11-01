#!/usr/bin/env python3
"""Step 1 Benchmark Suite for JoSIM-SPICE Co-Simulation Engine (libjosim_c Layer 1).

Tests the 4 non-trivial co-simulation benchmark circuits defined in COSIMULATION_PLAN.md:
  1. B1_JTL_NORTON_SPLIT     : 4-JJ Josephson Transmission Line split across its middle
                               coupling inductor into two separate JoSIM sessions coupled
                               at every timestep via PORT_VOLTAGE_IN_CURRENT_OUT (Norton)
                               and PORT_CURRENT_IN_VOLTAGE_OUT (Thevenin) ports, verified
                               against a monolithic 4-JJ JTL simulation.
  2. B2_CMOS_TO_SFQ_ROLLBACK : Nonlinear cryo-CMOS FET buffer stage driving a JoSIM
                               DC-to-SFQ / JTL / RSFQ storage loop via Newton-Raphson with
                               exact KLU Norton conductance G_eq, tested with >25 adaptive
                               LTE step rejections and rollbacks (both explicit and implicit
                               non-monotonic time rollback).
  3. B3_NTRON_TO_CMOS_LNA    : SFQ-triggered 3-terminal electrothermal nTron (N1) driving a
                               nonlinear capacitive/active cryo-CMOS LNA load in both Voltage
                               Mode and Phase Mode (-a 1), with rollbacks across hotspot
                               nucleation and thermal recovery.
  4. B4_BRANIN_WAVE_DUPLEX   : Full-duplex windowed Branin wave relaxation (PORT_WAVE_BRANIN
                               + cosim_run_window) across a Z0=50 Ohm, td=20 ps interconnect
                               with bidirectional reflections, verified against a monolithic
                               JoSIM T-line reference simulation.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
    os.environ["JOSIM_C_LIB"] = str(Path(sys.argv[1]).resolve())

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pyjosim import (
    PORT_CURRENT_IN_VOLTAGE_OUT,
    PORT_VOLTAGE_IN_CURRENT_OUT,
    PORT_WAVE_BRANIN,
    Session,
    simulate_netlist,
)


# ==============================================================================
# Benchmark 1: B1_JTL_NORTON_SPLIT
# ==============================================================================
def test_b1_jtl_norton_split() -> None:
    print("[B1] Running B1_JTL_NORTON_SPLIT (Split 4-JJ JTL via Norton/Thevenin ports)...")

    # Monolithic 4-JJ JTL reference circuit:
    # Coupling inductor between B2 and B3 is split into L2A (5pH) + L2B (5pH) at node nport.
    monolithic_netlist = """
* Monolithic 4-JJ JTL Reference
.model jjmod jj(rtype=1, vg=2.8m, cap=0.05p, r0=100, rn=5, icrit=100u)
IIN 0 n1 pwl(0 0 8p 0 9p 240u 11p 240u 12p 0)
IB1 0 n1 pwl(0 0 4p 70u)
B1 n1 0 jjmod
RSH1 n1 0 2.5
L12 n1 n2 10p
IB2 0 n2 pwl(0 0 4p 70u)
B2 n2 0 jjmod
RSH2 n2 0 2.5
L2A n2 nport 5p
L2B nport n3 5p
IB3 0 n3 pwl(0 0 4p 70u)
B3 n3 0 jjmod
RSH3 n3 0 2.5
L34 n3 n4 10p
IB4 0 n4 pwl(0 0 4p 70u)
B4 n4 0 jjmod
RSH4 n4 0 2.5
RLOAD n4 0 5
.tran 0.1p 60p
.print DEVV B1
.print DEVV B2
.print DEVV B3
.print DEVV B4
.print DEVP B1
.print DEVP B2
.print DEVP B3
.print DEVP B4
.print NODEV nport
.end
"""
    ref = simulate_netlist(monolithic_netlist)

    # Left half (Session A): B1, B2, L2A -> Port at (nport, 0) in VOLTAGE_IN_CURRENT_OUT mode
    left_netlist = """
* Left 2-JJ JTL (Session A)
.model jjmod jj(rtype=1, vg=2.8m, cap=0.05p, r0=100, rn=5, icrit=100u)
IIN 0 n1 pwl(0 0 8p 0 9p 240u 11p 240u 12p 0)
IB1 0 n1 pwl(0 0 4p 70u)
B1 n1 0 jjmod
RSH1 n1 0 2.5
L12 n1 n2 10p
IB2 0 n2 pwl(0 0 4p 70u)
B2 n2 0 jjmod
RSH2 n2 0 2.5
L2A n2 nport 5p
.tran 0.1p 60p
.print DEVV B1
.print DEVV B2
.print DEVP B1
.print DEVP B2
.print NODEV nport
.end
"""

    # Right half (Session B): L2B, B3, B4 -> Port at (nport, 0) in CURRENT_IN_VOLTAGE_OUT mode
    right_netlist = """
* Right 2-JJ JTL (Session B)
.model jjmod jj(rtype=1, vg=2.8m, cap=0.05p, r0=100, rn=5, icrit=100u)
L2B nport n3 5p
IB3 0 n3 pwl(0 0 4p 70u)
B3 n3 0 jjmod
RSH3 n3 0 2.5
L34 n3 n4 10p
IB4 0 n4 pwl(0 0 4p 70u)
B4 n4 0 jjmod
RSH4 n4 0 2.5
RLOAD n4 0 5
.tran 0.1p 60p
.print DEVV B3
.print DEVV B4
.print DEVP B3
.print DEVP B4
.print NODEV nport
.end
"""

    sess_a = Session(netlist=left_netlist)
    sess_a.cosim_add_port("nport", "0", mode=PORT_VOLTAGE_IN_CURRENT_OUT)
    sess_a.cosim_init()

    sess_b = Session(netlist=right_netlist)
    sess_b.cosim_add_port("nport", "0", mode=PORT_CURRENT_IN_VOLTAGE_OUT)
    sess_b.cosim_init()

    dt = sess_a.cosim_tstep
    assert math.isclose(dt, 0.1e-12, rel_tol=1e-12)
    n_steps = int(round(60e-12 / dt))

    v_port_guess = 0.0
    max_interface_v_err = 0.0

    for step in range(1, n_steps):
        t_target = step * dt
        # Predictor trial on Session A with v_port_guess
        y_a0, j_a0 = sess_a.cosim_step_to(t_target, [v_port_guess])
        i_a0, g_a = float(y_a0[0]), float(j_a0[0])
        assert g_a > 0.0, f"Expected positive Norton conductance G_A, got {g_a}"

        # Predictor trial on Session B with i_b0 = -i_a0
        i_b0 = -i_a0
        y_b0, j_b0 = sess_b.cosim_step_to(t_target, [i_b0])
        v_b0, r_b = float(y_b0[0]), float(j_b0[0])
        assert r_b > 0.0, f"Expected positive Thevenin resistance R_B, got {r_b}"

        # Exact Norton-Thevenin interface update:
        # V_p* = V_p0 + (V_B0 - V_p0) / (1 + G_A * R_B)
        v_port_star = v_port_guess + (v_b0 - v_port_guess) / (1.0 + g_a * r_b)
        i_b_star = -(i_a0 + g_a * (v_port_star - v_port_guess))

        # Corrector step at the same t_target (triggers automatic same-time re-evaluation rollback)
        y_a_star, _ = sess_a.cosim_step_to(t_target, [v_port_star])
        y_b_star, _ = sess_b.cosim_step_to(t_target, [i_b_star])

        v_err = abs(float(y_b_star[0]) - v_port_star)
        i_err = abs(float(y_a_star[0]) + i_b_star)
        max_interface_v_err = max(max_interface_v_err, v_err)
        assert v_err < 1e-12, f"Step {step}: interface V mismatch {v_err}"
        assert i_err < 1e-14, f"Step {step}: interface I mismatch {i_err}"

        sess_a.cosim_commit()
        sess_b.cosim_commit()
        v_port_guess = v_port_star

    res_a = sess_a.cosim_finalize()
    res_b = sess_b.cosim_finalize()

    # Compare waveforms against monolithic 4-JJ JTL simulation
    v_nport_ref = ref["V(NPORT)"]
    v_nport_a = res_a["V(NPORT)"]
    v_nport_b = res_b["V(NPORT)"]

    nport_rms_err = np.sqrt(np.mean((v_nport_a - v_nport_ref) ** 2)) / np.max(np.abs(v_nport_ref))
    b2_phase_err = np.max(np.abs(res_a["P(B2)"] - ref["P(B2)"]))
    b3_phase_err = np.max(np.abs(res_b["P(B3)"] - ref["P(B3)"]))
    b4_phase_err = np.max(np.abs(res_b["P(B4)"] - ref["P(B4)"]))

    # Compare SFQ event switching times on B1..B4
    ev_b1_ref = ref.events_for("B1")["time"]
    ev_b2_ref = ref.events_for("B2")["time"]
    ev_b3_ref = ref.events_for("B3")["time"]
    ev_b4_ref = ref.events_for("B4")["time"]

    ev_b1_cosim = res_a.events_for("B1")["time"]
    ev_b2_cosim = res_a.events_for("B2")["time"]
    ev_b3_cosim = res_b.events_for("B3")["time"]
    ev_b4_cosim = res_b.events_for("B4")["time"]

    assert len(ev_b1_cosim) == 1 and len(ev_b2_cosim) == 1
    assert len(ev_b3_cosim) == 1 and len(ev_b4_cosim) == 1

    dt_b1 = abs(float(ev_b1_cosim[0] - ev_b1_ref[0]))
    dt_b2 = abs(float(ev_b2_cosim[0] - ev_b2_ref[0]))
    dt_b3 = abs(float(ev_b3_cosim[0] - ev_b3_ref[0]))
    dt_b4 = abs(float(ev_b4_cosim[0] - ev_b4_ref[0]))

    print(
        f"     [B1] Interface V err = {max_interface_v_err:.2e} V, "
        f"nport RMS err = {nport_rms_err:.2e}, "
        f"max phase err (B2,B3,B4) = ({b2_phase_err:.2e}, {b3_phase_err:.2e}, {b4_phase_err:.2e}) rad, "
        f"SFQ timing diff = ({dt_b1*1e15:.2f}, {dt_b2*1e15:.2f}, {dt_b3*1e15:.2f}, {dt_b4*1e15:.2f}) fs"
    )

    assert nport_rms_err < 1e-5
    assert b2_phase_err < 1e-4 and b3_phase_err < 1e-4 and b4_phase_err < 1e-4
    assert max(dt_b1, dt_b2, dt_b3, dt_b4) < 0.05e-12
    assert np.max(np.abs(v_nport_a - v_nport_b)) < 1e-12


# ==============================================================================
# Benchmark 2: B2_CMOS_TO_SFQ_ROLLBACK
# ==============================================================================
def _run_b2_cmos_to_sfq(inject_rollbacks: bool) -> tuple:
    """Run nonlinear Cryo-CMOS driver coupled to JoSIM JTL + RSFQ storage loop."""
    josim_netlist = """
* JoSIM Input JTL + RSFQ Clocked Storage Loop
.model jjmod jj(rtype=1, vg=2.8m, cap=0.05p, r0=100, rn=5, icrit=100u)
RIN nin n1 5
IB1 0 n1 pwl(0 0 5p 70u)
B1 n1 0 jjmod
RSH1 n1 0 2.2
L12 n1 n2 10p
IB2 0 n2 pwl(0 0 5p 70u)
B2 n2 0 jjmod
RSH2 n2 0 2.2
L23 n2 n3 8p
* RSFQ storage interferometer loop: B_SET (n3), L_STOR (35pH), B_RO (n4)
IB_SET 0 n3 pwl(0 0 5p 65u)
B_SET n3 0 jjmod
RSH_SET n3 0 2.2
L_STOR n3 n4 35p
IB_RO 0 n4 pwl(0 0 5p 45u)
B_RO n4 0 jjmod
RSH_RO n4 0 2.5
* Readout clock pulse at t = 42 ps switches B_RO if a fluxon is stored in L_STOR
ICLK 0 n4 pwl(0 0 40p 0 41p 95u 42.5p 95u 43.5p 0)
RLOAD n4 0 5
.tran 0.1p 65p
.print DEVV B1
.print DEVV B2
.print DEVV B_SET
.print DEVV B_RO
.print DEVP B1
.print DEVP B2
.print DEVP B_SET
.print DEVP B_RO
.print DEVFI L_STOR
.print NODEV nin
.end
"""

    sess = Session(netlist=josim_netlist)
    sess.cosim_add_port("nin", "0", mode=PORT_VOLTAGE_IN_CURRENT_OUT)
    sess.cosim_init()

    dt = sess.cosim_tstep
    n_steps = int(round(65e-12 / dt))

    # Cryo-CMOS nonlinear FET buffer parameters driving interface node V_int (= V(nin)):
    # C_int * dV_int/dt + V_int/R_bleed - I_DS(V_G(t), V_int) + I_JoSIM(V_int) = 0
    c_int = 0.15e-12
    g_bleed = 1.0 / 6.0
    v_dd = 4.0e-3
    v_th = 0.8e-3
    i_d0 = 320e-6
    kappa = 1200.0
    lam = 80.0

    def v_gate(t: float) -> float:
        # Cryo-CMOS gate pulse between t=12ps and t=15.5ps
        if t < 12e-12:
            return 0.0
        elif t < 13e-12:
            return 3.2e-3 * (t - 12e-12) / 1e-12
        elif t < 14.5e-12:
            return 3.2e-3
        elif t < 15.5e-12:
            return 3.2e-3 * (15.5e-12 - t) / 1e-12
        return 0.0

    def cmos_ids_and_deriv(vg: float, v_int: float) -> tuple[float, float]:
        u = kappa * (vg - v_int - v_th)
        th = math.tanh(u)
        sech2 = 1.0 - th * th
        early = 1.0 + lam * (v_dd - v_int)
        # Active pull-up only when gate is above threshold
        gate_on = 0.5 * (1.0 + th)
        dgate_dv = -0.5 * kappa * sech2
        i_ds = i_d0 * gate_on * early
        di_ds_dv = i_d0 * (dgate_dv * early - gate_on * lam)
        return i_ds, di_ds_dv

    v_int_committed = 0.0
    max_nr_iters = 0
    explicit_rollbacks = 0

    for step in range(1, n_steps):
        t_target = step * dt

        if inject_rollbacks and (140 <= step <= 260 or 410 <= step <= 470) and (step % 4 == 0):
            # 1) Trial an oversized multi-substep jump (t_target + 3*dt) with a perturbed voltage,
            #    then reject it via explicit cosim_rollback()
            sess.cosim_step_to(min(t_target + 3 * dt, (n_steps - 1) * dt), [v_int_committed + 0.5e-3])
            sess.cosim_rollback()
            explicit_rollbacks += 1

            # 2) Also test implicit non-monotonic time rollback (Spectre $abstime < t_prev style):
            #    Step to t_target + 2*dt, then directly call step_to(t_target) without calling rollback()!
            if step + 2 < n_steps:
                sess.cosim_step_to(t_target + 2 * dt, [v_int_committed - 0.3e-3])

        # Coupled Newton-Raphson loop at t_target using JoSIM's exact KLU Norton conductance G_eq
        vg = v_gate(t_target)
        v_guess = v_int_committed
        c_comp = c_int / dt

        for nr_iter in range(1, 12):
            y_out, j_out = sess.cosim_step_to(t_target, [v_guess])
            i_josim = float(y_out[0])
            g_josim = float(j_out[0])

            i_ds, di_ds_dv = cmos_ids_and_deriv(vg, v_guess)
            # KCL residual at interface node:
            f_res = (
                c_comp * (v_guess - v_int_committed)
                + g_bleed * v_guess
                - i_ds
                + i_josim
            )
            df_dv = c_comp + g_bleed - di_ds_dv + g_josim
            dv = -f_res / df_dv
            v_guess += dv
            if abs(dv) < 1e-13 and abs(f_res) < 1e-13:
                max_nr_iters = max(max_nr_iters, nr_iter)
                break
        else:
            raise RuntimeError(f"NR failed to converge at step {step}")

        # Finalize state at converged v_guess and commit
        sess.cosim_step_to(t_target, [v_guess])
        sess.cosim_commit()
        v_int_committed = v_guess

    res = sess.cosim_finalize()
    return res, sess.cosim_rollback_count, explicit_rollbacks, max_nr_iters


def test_b2_cmos_to_sfq_rollback() -> None:
    print("[B2] Running B2_CMOS_TO_SFQ_ROLLBACK (Nonlinear Cryo-CMOS -> SFQ DFF with LTE rollbacks)...")

    res_clean, rb_clean, _, max_nr_clean = _run_b2_cmos_to_sfq(inject_rollbacks=False)
    res_rb, rb_total, rb_explicit, max_nr_rb = _run_b2_cmos_to_sfq(inject_rollbacks=True)

    # Verify B1, B2, B_SET switched on the input CMOS pulse, and B_RO switched on the readout clock
    for name in ("B1", "B2", "B_SET", "B_RO"):
        ev_c = res_clean.events_for(name)
        ev_r = res_rb.events_for(name)
        assert len(ev_c) == 1, f"Expected 1 SFQ event on {name}, got {len(ev_c)}"
        assert len(ev_r) == 1, f"Expected 1 SFQ event on {name} after rollbacks, got {len(ev_r)}"
        assert abs(float(ev_c["time"][0] - ev_r["time"][0])) < 1e-15

    # Verify zero state drift across >25 multi-step rollbacks (exact bit-for-bit/machine-precision match)
    max_v_diff = np.max(np.abs(res_clean["V(NIN)"] - res_rb["V(NIN)"]))
    max_p_ro_diff = np.max(np.abs(res_clean["P(B_RO)"] - res_rb["P(B_RO)"]))
    max_i_stor_diff = np.max(np.abs(res_clean["I(L_STOR)"] - res_rb["I(L_STOR)"]))

    t_set = float(res_clean.events_for("B_SET")["time"][0])
    t_ro = float(res_clean.events_for("B_RO")["time"][0])

    print(
        f"     [B2] Total rollbacks = {rb_total} (explicit LTE rejections = {rb_explicit}), "
        f"max NR iters = {max_nr_rb}, T_SET = {t_set*1e12:.2f} ps, T_RO = {t_ro*1e12:.2f} ps, "
        f"rollback drift = {max(max_v_diff, max_p_ro_diff, max_i_stor_diff):.2e}"
    )

    assert rb_explicit >= 25
    assert rb_total >= 50
    assert max_nr_clean <= 4 and max_nr_rb <= 4
    assert max_v_diff < 1e-12
    assert max_p_ro_diff < 1e-12
    assert max_i_stor_diff < 1e-12


# ==============================================================================
# Benchmark 3: B3_NTRON_TO_CMOS_LNA
# ==============================================================================
def _run_b3_ntron_lna(phase_mode: bool, inject_rollbacks: bool) -> tuple:
    """Run SFQ-triggered nTron cryotron driving a nonlinear cryo-CMOS LNA input stage."""
    netlist = """
* SFQ-Triggered nTron Cryotron Amplifier Driving Cryo-CMOS LNA Port
.model jjmod jj(rtype=1, vg=2.8m, cap=0.05p, r0=100, rn=5, icrit=100u)
.model nmod ntron(icgate=15u, icch=100u, rnormal=80, tau=12p, eta=0.35, iretrig=0.18, rsheet=40, lsheet=2p, width=0.1u, length=0.4u)

* Input SFQ trigger generator into nTron gate
ITRIG 0 ntrig pwl(0 0 8p 0 10p 180u 14p 180u 16p 0)
IB_T 0 ntrig pwl(0 0 4p 65u)
B_TRIG ntrig 0 jjmod
RSH_T ntrig 0 2.2
L_GATE ntrig ngate 6p
R_GATE ngate n_g_in 8

* 3-terminal nTron: channel (nch, 0), gate (n_g_in, 0)
IBIAS 0 nbias pwl(0 0 5p 85u)
LBIAS nbias nch 80p
N1 nch 0 n_g_in 0 nmod

.tran 0.1p 80p
.print NODEV nch
.print DEVI N1
.print DEVP B_TRIG
.end
"""
    cli_args = ["-a", "1"] if phase_mode else ["-a", "0"]
    sess = Session(netlist=netlist, cli_args=cli_args)
    # Use CURRENT_IN_VOLTAGE_OUT port on nch to test current-mode port + Thevenin R_eq with nTron
    sess.cosim_add_port("nch", "0", mode=PORT_CURRENT_IN_VOLTAGE_OUT)
    sess.cosim_init()

    dt = sess.cosim_tstep
    n_steps = int(round(80e-12 / dt))

    # Nonlinear Cryo-CMOS LNA input stage connected to nch:
    # Input shunt resistance R_in = 60 Ohm + nonlinear MOS gate capacitance C_g(V) = C0*(1 + 0.25*tanh(V/2mV))
    # Plus active LNA voltage gain stage V_lna_out = A_v * V_0 * tanh(V_ch / V_0)
    r_in = 60.0
    c_g0 = 0.04e-12
    v_scale = 2.0e-3
    v_ch_prev = 0.0
    lna_out_trace = np.zeros(n_steps, dtype=np.float64)

    for step in range(1, n_steps):
        t_target = step * dt

        # Inject rollbacks right across nTron hotspot nucleation (~11-18 ps) and thermal recovery (~25-45 ps)
        if inject_rollbacks and (100 <= step <= 200 or 260 <= step <= 420) and (step % 5 == 0):
            sess.cosim_step_to(min(t_target + 2 * dt, (n_steps - 1) * dt), [-20e-6])
            sess.cosim_rollback()

        # Solve coupled nTron (Thevenin V_ch(I_in), R_eq) + nonlinear CMOS LNA input load I_in(V_ch)
        v_guess = v_ch_prev
        for _ in range(8):
            c_eff = c_g0 * (1.0 + 0.25 * math.tanh(v_guess / v_scale))
            dc_dv = c_g0 * 0.25 * (1.0 - math.tanh(v_guess / v_scale) ** 2) / v_scale
            # Current drawn from nch by CMOS LNA input (so current injected into nch is -i_load)
            i_load = v_guess / r_in + c_eff * (v_guess - v_ch_prev) / dt
            di_load_dv = 1.0 / r_in + (c_eff + dc_dv * (v_guess - v_ch_prev)) / dt

            y_out, j_out = sess.cosim_step_to(t_target, [-i_load])
            v_josim = float(y_out[0])
            r_josim = float(j_out[0])

            # Residual: f(v_guess) = v_guess - v_josim(-i_load(v_guess)) = 0
            f_res = v_guess - v_josim
            df_dv = 1.0 + r_josim * di_load_dv
            dv = -f_res / df_dv
            v_guess += dv
            if abs(dv) < 1e-13:
                break

        i_load_final = (
            v_guess / r_in
            + c_g0 * (1.0 + 0.25 * math.tanh(v_guess / v_scale)) * (v_guess - v_ch_prev) / dt
        )
        sess.cosim_step_to(t_target, [-i_load_final])
        sess.cosim_commit()
        v_ch_prev = v_guess
        # Cryo-CMOS LNA output stage (gain = 15 V/V, saturation = 45 mV)
        lna_out_trace[step] = 45e-3 * math.tanh((15.0 * v_guess) / 45e-3)

    res = sess.cosim_finalize()
    return res, lna_out_trace, sess.cosim_rollback_count


def test_b3_ntron_to_cmos_lna() -> None:
    print("[B3] Running B3_NTRON_TO_CMOS_LNA (SFQ -> nTron -> Cryo-CMOS LNA in V & Phase modes)...")

    res_v_clean, lna_v_clean, _ = _run_b3_ntron_lna(phase_mode=False, inject_rollbacks=False)
    res_v_rb, lna_v_rb, rb_count = _run_b3_ntron_lna(phase_mode=False, inject_rollbacks=True)
    res_p_rb, lna_p_rb, _ = _run_b3_ntron_lna(phase_mode=True, inject_rollbacks=True)

    v_ch_peak = float(np.max(res_v_clean["V(NCH)"]))
    lna_peak = float(np.max(lna_v_clean))
    i_n1_min = float(np.min(res_v_clean["I(N1)"]))

    # Verify rollback invariance across nTron electrothermal switching & recovery
    rb_drift_v = float(np.max(np.abs(res_v_clean["V(NCH)"] - res_v_rb["V(NCH)"])))
    rb_drift_i = float(np.max(np.abs(res_v_clean["I(N1)"] - res_v_rb["I(N1)"])))

    # Verify Voltage Mode vs Phase Mode (-a 1) agreement
    vp_rel_err = float(
        np.sqrt(np.mean((res_v_clean["V(NCH)"] - res_p_rb["V(NCH)"]) ** 2))
        / np.max(np.abs(res_v_clean["V(NCH)"]))
    )

    print(
        f"     [B3] V(nch) peak = {v_ch_peak*1e3:.2f} mV, LNA V_out peak = {lna_peak*1e3:.2f} mV, "
        f"I(N1) min = {i_n1_min*1e6:.2f} uA, rollback drift = {max(rb_drift_v, rb_drift_i):.2e}, "
        f"V vs Phase (-a 1) RMS diff = {vp_rel_err:.2e}"
    )

    assert v_ch_peak > 2.0e-3, f"Expected nTron channel voltage > 2 mV, got {v_ch_peak}"
    assert lna_peak > 20.0e-3, f"Expected amplified CMOS LNA output > 20 mV, got {lna_peak}"
    assert rb_count >= 50
    assert rb_drift_v < 1e-12 and rb_drift_i < 1e-12
    assert vp_rel_err < 1e-2


# ==============================================================================
# Benchmark 4: B4_BRANIN_WAVE_DUPLEX
# ==============================================================================
def test_b4_branin_wave_duplex() -> None:
    print("[B4] Running B4_BRANIN_WAVE_DUPLEX (Full-duplex Branin wave relaxation vs monolithic T-line)...")

    # Monolithic reference circuit: Superconducting Tx driver -> TLINE (Z0=50, TD=20p) -> Mismatched JJ Rx
    monolithic_netlist = """
* Monolithic Lossless T-Line Reference (Z0=50 Ohm, TD=20 ps)
.model jjmod jj(rtype=1, vg=2.8m, cap=0.05p, r0=100, rn=5, icrit=100u)
.model nmod ntron(icgate=15u, icch=120u, rnormal=60, tau=10p, eta=0.35, iretrig=0.2)

* Tx side: nTron pulse driver generating a high-amplitude pulse onto the 50-Ohm line
IGATE 0 ng pwl(0 0 6p 0 8p 45u 14p 45u 16p 0)
RG ng 0 50
IBTX 0 nbias pwl(0 0 4p 100u)
LBTX nbias ntx 60p
NTX ntx 0 ng 0 nmod

* 20 ps, 50 Ohm interconnect
TLINE ntx 0 nrx 0 td=20p z0=50

* Rx side: partially mismatched termination (R_MATCH=120 Ohm + JJ receiver B_RX)
* Causes a known positive reflection back to Tx at t + 2*td = 48 ps!
RMATCH nrx nrx_jj 70
IBRX 0 nrx_jj pwl(0 0 4p 68u)
BRX nrx_jj 0 jjmod
RSH_RX nrx_jj 0 2.2

.tran 0.1p 80p
.print NODEV ntx
.print NODEV nrx
.print DEVP BRX
.end
"""
    ref = simulate_netlist(monolithic_netlist)

    tx_netlist = """
* Tx Subsystem (Branin Wave Port at ntx)
.model nmod ntron(icgate=15u, icch=120u, rnormal=60, tau=10p, eta=0.35, iretrig=0.2)
IGATE 0 ng pwl(0 0 6p 0 8p 45u 14p 45u 16p 0)
RG ng 0 50
IBTX 0 nbias pwl(0 0 4p 100u)
LBTX nbias ntx 60p
NTX ntx 0 ng 0 nmod
.tran 0.1p 80p
.print NODEV ntx
.end
"""

    rx_netlist = """
* Rx Subsystem (Branin Wave Port at nrx)
.model jjmod jj(rtype=1, vg=2.8m, cap=0.05p, r0=100, rn=5, icrit=100u)
RMATCH nrx nrx_jj 70
IBRX 0 nrx_jj pwl(0 0 4p 68u)
BRX nrx_jj 0 jjmod
RSH_RX nrx_jj 0 2.2
.tran 0.1p 80p
.print NODEV nrx
.print DEVP BRX
.end
"""

    sess_tx = Session(netlist=tx_netlist)
    sess_tx.cosim_add_port("ntx", "0", mode=PORT_WAVE_BRANIN, z0=50.0)
    sess_tx.cosim_init()

    sess_rx = Session(netlist=rx_netlist)
    sess_rx.cosim_add_port("nrx", "0", mode=PORT_WAVE_BRANIN, z0=50.0)
    sess_rx.cosim_init()

    td = 20e-12
    t_total = 80e-12
    n_windows = int(round(t_total / td))

    # In Window 0 [0, td], no wave has arrived from the far end yet (V_inc = 0)
    t_win_tx = np.array([0.0, td], dtype=np.float64)
    v_inc_tx = np.zeros((2, 1), dtype=np.float64)
    t_win_rx = np.array([0.0, td], dtype=np.float64)
    v_inc_rx = np.zeros((2, 1), dtype=np.float64)

    for w in range(n_windows):
        t_start = w * td
        t_end = (w + 1) * td

        out_t_tx, out_v_ref_tx = sess_tx.cosim_run_window(t_start, t_end, t_win_tx, v_inc_tx)
        out_t_rx, out_v_ref_rx = sess_rx.cosim_run_window(t_start, t_end, t_win_rx, v_inc_rx)

        # Propagate outgoing reflected waves across the lossless delay td to the opposite end for window w+1:
        # V_inc_rx(t + td) = V_ref_tx(t), and V_inc_tx(t + td) = V_ref_rx(t)
        t_win_rx = out_t_tx + td
        v_inc_rx = out_v_ref_tx.copy()

        t_win_tx = out_t_rx + td
        v_inc_tx = out_v_ref_rx.copy()

    res_tx = sess_tx.cosim_finalize()
    res_rx = sess_rx.cosim_finalize()

    v_ntx_ref = ref["V(NTX)"]
    v_nrx_ref = ref["V(NRX)"]
    p_brx_ref = ref["P(BRX)"]

    v_ntx_cosim = res_tx["V(NTX)"]
    v_nrx_cosim = res_rx["V(NRX)"]
    p_brx_cosim = res_rx["P(BRX)"]

    n_pts = min(len(v_ntx_cosim), len(v_ntx_ref))
    ntx_rms_err = float(
        np.sqrt(np.mean((v_ntx_cosim[:n_pts] - v_ntx_ref[:n_pts]) ** 2))
        / np.max(np.abs(v_ntx_ref[:n_pts]))
    )
    nrx_rms_err = float(
        np.sqrt(np.mean((v_nrx_cosim[:n_pts] - v_nrx_ref[:n_pts]) ** 2))
        / np.max(np.abs(v_nrx_ref[:n_pts]))
    )
    brx_phase_err = float(np.max(np.abs(p_brx_cosim[:n_pts] - p_brx_ref[:n_pts])))

    ev_ref = ref.events_for("BRX")["time"]
    ev_cosim = res_rx.events_for("BRX")["time"]
    assert len(ev_ref) == 4 and len(ev_cosim) == 4
    dt_brx = float(np.max(np.abs(ev_cosim - ev_ref)))

    print(
        f"     [B4] V(ntx) RMS err = {ntx_rms_err:.2e}, V(nrx) RMS err = {nrx_rms_err:.2e}, "
        f"P(BRX) max err = {brx_phase_err:.2e} rad, BRX 4-pulse max timing diff = {dt_brx*1e15:.2f} fs"
    )

    assert ntx_rms_err < 1e-4
    assert nrx_rms_err < 1e-4
    assert brx_phase_err < 1e-3
    assert dt_brx < 0.05e-12


def main() -> int:
    print("=" * 78)
    print("JoSIM Co-Simulation Step 1 Benchmark Suite (Layer 1 C API & Norton Engine)")
    print("=" * 78)
    test_b1_jtl_norton_split()
    test_b2_cmos_to_sfq_rollback()
    test_b3_ntron_to_cmos_lna()
    test_b4_branin_wave_duplex()
    print("=" * 78)
    print("ALL 4 STEP 1 CO-SIMULATION BENCHMARKS PASSED!")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
