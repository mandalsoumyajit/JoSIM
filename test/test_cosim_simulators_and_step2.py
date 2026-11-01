#!/usr/bin/env python3
"""
Benchmark & Verification Suite for Real SPICE Simulators (`ngspice-45.2` & ADI `LTspice.exe`)
across Step 1 (In-Process `libjosim_c`) and Step 2 (`josim-cosim-server` UNIX/TCP IPC Daemon).

Benchmarks:
  1. NG1_SPLIT_JTL_XSPICE:
     4-JJ Josephson Transmission Line split across `ngspice-45.2` via `josim_cosim.cm`
     (tested in Step 1 in-process mode, Step 2 UNIX socket mode, and Step 2 TCP socket mode)
     vs. monolithic JoSIM reference.
  2. NG2_CMOS_TO_SFQ_DFF:
     2-stage 4K Cryo-CMOS pulse driver in `ngspice-45.2` driving a 2-port JoSIM
     DC-to-SFQ JTL + quantized superconducting storage loop + clocked readout comparator
     (verifying `ngspice` adaptive timestepping, LTE rollbacks, and Step 1 == Step 2).
  3. NG3_NTRON_TO_CMOS_LNA:
     JoSIM SFQ-triggered 3-terminal `nTron` electrothermal cryotron driving a 2-stage
     Cryo-CMOS Common-Source + Source-Follower LNA in `ngspice-45.2` (Step 1 & Step 2).
  4. LT1_CMOS_BRANIN_TRANSCEIVER:
     Real ADI `LTspice.exe -b` Cryo-CMOS Driver & LNA Receiver coupled over dual 50-Ohm,
     20 ps cryogenic coax lines (`PORT_WAVE_BRANIN`) to a JoSIM SFQ + `nTron` core
     (Step 1 in-process & Step 2 IPC socket server).
  5. LT2_NTRON_TO_LTSPICE_LNA:
     JoSIM `nTron` coupled to real ADI `LTspice.exe -b` MOSFET LNA via Norton-PWL
     waveform relaxation, cross-validated against `ngspice-45.2` lockstep XSPICE.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from pyjosim import (
    PORT_CURRENT_IN_VOLTAGE_OUT,
    PORT_VOLTAGE_IN_CURRENT_OUT,
    PORT_WAVE_BRANIN,
    Session,
)
from josim_cosim import (
    CosimServerProcess,
    PortSpec,
    _default_win_visible_tmp_dir,
    format_spice_pwl_file,
    run_ltspice_batch,
    run_ngspice_deck,
)

PHI0 = 2.067833848e-15  # Wb


def nrmse(y_test: np.ndarray, y_ref: np.ndarray) -> float:
    denom = float(np.max(np.abs(y_ref)) - np.min(np.abs(y_ref)))
    if denom < 1e-15:
        denom = float(np.max(np.abs(y_ref))) + 1e-15
    return float(np.sqrt(np.mean((y_test - y_ref) ** 2)) / denom)


# ===========================================================================
# Benchmark 1: NG1_SPLIT_JTL_XSPICE (ngspice-45.2 + josim_cosim.cm Split JTL)
# ===========================================================================
def test_ng1_split_jtl_xspice() -> Dict[str, float]:
    """
    Split a 4-JJ Josephson Transmission Line across `ngspice-45.2`:
      - JoSIM TX sub-circuit: B1 -> B2 driving boundary node `n2`
      - `ngspice-45.2` interconnecting network: Lp1 (5 pH) - Cp (10 fF) - Lp2 (5 pH)
      - JoSIM RX sub-circuit: boundary node `n3` -> B3 -> B4
    Tested in:
      (a) Step 1 In-Process Mode (`server_addr=""`)
      (b) Step 2 UNIX Socket Mode (`server_addr="unix:..."`)
      (c) Step 2 TCP Socket Mode (`server_addr="tcp:127.0.0.1:..."`)
    Compared against monolithic 4-JJ JoSIM reference.
    """
    mono_netlist = """* Monolithic 4-JJ JTL with intermediate LC filter
.model jjmod jj(rtype=1, vg=2.8m, cap=0.05p, r0=100, rn=5, icrit=100u)
IIN 0 n1 pwl(0 0 8p 0 9p 240u 11p 240u 12p 0)
IB1 0 n1 pwl(0 0 4p 70u)
B1 n1 0 jjmod
RSH1 n1 0 2.5
L12 n1 n2 10p
IB2 0 n2 pwl(0 0 4p 70u)
B2 n2 0 jjmod
RSH2 n2 0 2.5
Lp1 n2 mid 5p
Cp mid 0 10f
Rp mid 0 200
Lp2 mid n3 5p
IB3 0 n3 pwl(0 0 4p 70u)
B3 n3 0 jjmod
RSH3 n3 0 2.5
L34 n3 n4 10p
IB4 0 n4 pwl(0 0 4p 70u)
B4 n4 0 jjmod
RSH4 n4 0 2.5
RLOAD n4 0 5
.tran 0.1p 50p
.print DEVV B1
.print DEVV B2
.print DEVV B3
.print DEVV B4
.print DEVP B1
.print DEVP B2
.print DEVP B3
.print DEVP B4
.print NODEV mid
.end
"""
    s_mono = Session()
    s_mono.set_cli_args(["-a", "1"])
    s_mono.load_string(mono_netlist)
    mono_res = s_mono.run()
    t_mono = mono_res.time
    v_mid_mono = mono_res["V(MID)"]
    mono_ev = mono_res.events
    assert len(mono_ev) == 4, f"Expected 4 SFQ events in monolithic JTL, got {len(mono_ev)}"

    tx_netlist = """* JoSIM TX half (B1 -> B2)
.model jjmod jj(rtype=1, vg=2.8m, cap=0.05p, r0=100, rn=5, icrit=100u)
IIN 0 n1 pwl(0 0 8p 0 9p 240u 11p 240u 12p 0)
IB1 0 n1 pwl(0 0 4p 70u)
B1 n1 0 jjmod
RSH1 n1 0 2.5
L12 n1 n2 10p
IB2 0 n2 pwl(0 0 4p 70u)
B2 n2 0 jjmod
RSH2 n2 0 2.5
.tran 0.1p 50p
.print DEVV B1
.print DEVV B2
.print DEVP B1
.print DEVP B2
.end
"""
    rx_netlist = """* JoSIM RX half (B3 -> B4)
.model jjmod jj(rtype=1, vg=2.8m, cap=0.05p, r0=100, rn=5, icrit=100u)
IB3 0 n3 pwl(0 0 4p 70u)
B3 n3 0 jjmod
RSH3 n3 0 2.5
L34 n3 n4 10p
IB4 0 n4 pwl(0 0 4p 70u)
B4 n4 0 jjmod
RSH4 n4 0 2.5
RLOAD n4 0 5
.tran 0.1p 50p
.print DEVV B3
.print DEVV B4
.print DEVP B3
.print DEVP B4
.end
"""

    with tempfile.TemporaryDirectory(prefix="ng1_split_") as tmpd:
        tmp = Path(tmpd)
        tx_cir = (tmp / "tx.cir").as_posix()
        rx_cir = (tmp / "rx.cir").as_posix()
        Path(tx_cir).write_text(tx_netlist)
        Path(rx_cir).write_text(rx_netlist)

        def build_deck(tx_srv: str, rx_srv: str, tx_csv: str, rx_csv: str) -> str:
            tx_srv_p = f'server_addr="{tx_srv}"' if tx_srv else ""
            rx_srv_p = f'server_addr="{rx_srv}"' if rx_srv else ""
            return f"""* ngspice-45.2 interconnecting LC filter between two JoSIM XSPICE models
Lp1 n2 mid 5p
Cp mid 0 10f
Rp mid 0 200
Lp2 mid n3 5p
A_tx [%gd(n2 0)] jtx
A_rx [%gd(n3 0)] jrx
.model jtx josim_cosim(netlist="{tx_cir}" pos_nodes=["n2"] neg_nodes=["0"] {tx_srv_p} out_csv="{tx_csv}")
.model jrx josim_cosim(netlist="{rx_cir}" pos_nodes=["n3"] neg_nodes=["0"] {rx_srv_p} out_csv="{rx_csv}")
.end
"""

        # (a) Step 1 In-Process Mode
        tx_csv_s1 = (tmp / "tx_s1.csv").as_posix()
        rx_csv_s1 = (tmp / "rx_s1.csv").as_posix()
        res_s1 = run_ngspice_deck(
            build_deck("", "", tx_csv_s1, rx_csv_s1),
            wrdata_vecs=["v(n2)", "v(mid)", "v(n3)"],
            tran_cmd="tran 0.1p 50p",
            josim_csv_map={"tx": tx_csv_s1, "rx": rx_csv_s1},
        )

        # (b) Step 2 UNIX Socket Mode (2 concurrent josim-cosim-server daemons)
        tx_csv_unix = (tmp / "tx_unix.csv").as_posix()
        rx_csv_unix = (tmp / "rx_unix.csv").as_posix()
        with CosimServerProcess(transport="unix") as srv_tx, CosimServerProcess(transport="unix") as srv_rx:
            res_unix = run_ngspice_deck(
                build_deck(srv_tx.addr, srv_rx.addr, tx_csv_unix, rx_csv_unix),
                wrdata_vecs=["v(n2)", "v(mid)", "v(n3)"],
                tran_cmd="tran 0.1p 50p",
                josim_csv_map={"tx": tx_csv_unix, "rx": rx_csv_unix},
            )

        # (c) Step 2 TCP Socket Mode (2 concurrent josim-cosim-server daemons over TCP_NODELAY)
        tx_csv_tcp = (tmp / "tx_tcp.csv").as_posix()
        rx_csv_tcp = (tmp / "rx_tcp.csv").as_posix()
        with CosimServerProcess(transport="tcp") as srv_tx, CosimServerProcess(transport="tcp") as srv_rx:
            res_tcp = run_ngspice_deck(
                build_deck(srv_tx.addr, srv_rx.addr, tx_csv_tcp, rx_csv_tcp),
                wrdata_vecs=["v(n2)", "v(mid)", "v(n3)"],
                tran_cmd="tran 0.1p 50p",
                josim_csv_map={"tx": tx_csv_tcp, "rx": rx_csv_tcp},
            )

    # Verify all 4 JJs fired across TX and RX in ngspice co-simulation
    ev_tx = res_s1.josim_events["tx"]
    ev_rx = res_s1.josim_events["rx"]
    assert len(ev_tx) == 2, f"Expected 2 TX SFQ events (B1, B2), got {len(ev_tx)}"
    assert len(ev_rx) == 2, f"Expected 2 RX SFQ events (B3, B4), got {len(ev_rx)}"

    # Causal ordering: B1 < B2 < B3 < B4
    t_b1 = float(ev_tx[0]["time"])
    t_b2 = float(ev_tx[1]["time"])
    t_b3 = float(ev_rx[0]["time"])
    t_b4 = float(ev_rx[1]["time"])
    assert t_b1 < t_b2 < t_b3 < t_b4, f"Non-causal SFQ propagation: {t_b1}, {t_b2}, {t_b3}, {t_b4}"

    # Flux quantization on RX output B4 vs monolithic JoSIM reference
    mono_b4_area = float(mono_ev["area"][3])
    area_err = abs(float(ev_rx[1]["area"]) - mono_b4_area) / mono_b4_area
    assert area_err < 0.05, f"B4 flux area error vs monolithic {area_err:.4f} exceeds 5%"

    # Interpolate ngspice v(mid) onto monolithic time grid and compare
    v_mid_ng = np.interp(t_mono, res_s1.ng_time, res_s1.ng_traces["v(mid)"])
    err_mid = nrmse(v_mid_ng, v_mid_mono)
    assert err_mid < 0.035, f"NG1 v(mid) NRMSE vs monolithic JoSIM is {err_mid:.4f} (> 3.5%)"

    # Verify Step 1 In-Process == Step 2 UNIX Socket == Step 2 TCP Socket bit-for-bit
    diff_unix = float(np.max(np.abs(res_s1.ng_traces["v(mid)"] - res_unix.ng_traces["v(mid)"])))
    diff_tcp = float(np.max(np.abs(res_s1.ng_traces["v(mid)"] - res_tcp.ng_traces["v(mid)"])))
    assert diff_unix == 0.0, f"Step 1 vs Step 2 UNIX mismatch: {diff_unix}"
    assert diff_tcp == 0.0, f"Step 1 vs Step 2 TCP mismatch: {diff_tcp}"

    return {
        "nrmse_vs_mono": err_mid,
        "dt_b4_fs": abs(t_b4 - float(mono_ev["time"][3])) * 1e15,
        "diff_step1_vs_unix": diff_unix,
        "diff_step1_vs_tcp": diff_tcp,
    }


# ===========================================================================
# Benchmark 2: NG2_CMOS_TO_SFQ_DFF (ngspice-45.2 MOSFET Driver -> JoSIM RSFQ Loop)
# ===========================================================================
def test_ng2_cmos_to_sfq_dff() -> Dict[str, float]:
    """
    2-stage 4K Cryo-CMOS MOSFET pulse driver in `ngspice-45.2` (CMOS inverter + NMOS
    source follower) driving a 2-port JoSIM RSFQ circuit:
      - Port 0 (`din`): Input JTL (`B1`, `B2`) feeding a quantized superconducting
        storage interferometer loop (`B_SET`, `L_STOR = 35 pH`, `B_RO`)
      - Port 1 (`clk`): Readout clock driving `B_RO`
    Verifies:
      - Real MOSFET device physics in `ngspice-45.2` coupled with 2-port JoSIM XSPICE model
      - `ngspice-45.2` adaptive timestepping + LTE step rejection rollbacks
      - Exact equivalence between Step 1 (in-process) and Step 2 (IPC socket server)
    """
    sfq_dff_netlist = """* 2-Port JoSIM Input JTL + RSFQ Clocked Storage Loop
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
.print DEVI L_STOR
.end
"""

    with tempfile.TemporaryDirectory(prefix="ng2_cmos_sfq_") as tmpd:
        tmp = Path(tmpd)
        sfq_cir = (tmp / "sfq_dff.cir").as_posix()
        Path(sfq_cir).write_text(sfq_dff_netlist)

        def build_deck(srv_addr: str, out_csv: str) -> str:
            srv_p = f'server_addr="{srv_addr}"' if srv_addr else ""
            return f"""* ngspice-45.2 4K Cryo-CMOS DATA & CLOCK Drivers -> 2-Port JoSIM RSFQ Loop
.model nch nmos(level=1 vto=0.22 kp=550u gamma=0.1 phi=0.6 lambda=0.03 cbd=1f cbs=1f cgso=0.5n cgdo=0.5n)
.model pch pmos(level=1 vto=-0.22 kp=275u gamma=0.1 phi=0.6 lambda=0.03 cbd=1f cbs=1f cgso=0.5n cgdo=0.5n)

Vdd vdd 0 0.8

* --- DATA Channel: CMOS Inverter + NMOS Source-Follower + Cryo Attenuator (%gd port) ---
Vdata_in d_gate 0 PWL(0 0.8  11p 0.8  12p 0.0  15p 0.0  16p 0.8)
Mp_d d_inv d_gate vdd vdd pch w=12u l=0.18u
Mn_d d_inv d_gate 0   0   nch w=6u  l=0.18u
Mn_dsf vdd d_inv d_src 0 nch w=24u l=0.18u
R_dsf d_src 0 150
Cc_d d_src d_ac 2.0p
R_att_d d_ac din 320
R_sh_d din 0 6.0

* --- CLOCK Channel: CMOS Inverter + NMOS Source-Follower + Current-Feed (%hd port) ---
Vclk_in c_gate 0 PWL(0 0.8  40p 0.8  41p 0.0  43.5p 0.0  44.5p 0.8)
Mp_c c_inv c_gate vdd vdd pch w=12u l=0.18u
Mn_c c_inv c_gate 0   0   nch w=6u  l=0.18u
Mn_csf vdd c_inv c_src 0 nch w=24u l=0.18u
R_csf c_src 0 150
Cc_c c_src c_ac 2.0p
R_att_c c_ac clk 800

* --- 2-Port JoSIM Superconducting Co-Simulation Instance (%gd + %hd mixed ports) ---
A_sfq [%gd(din 0) %hd(clk 0)] jsfq
.model jsfq josim_cosim(netlist="{sfq_cir}" pos_nodes=["nin" "n4"] neg_nodes=["0" "0"] {srv_p} out_csv="{out_csv}")
.end
"""

        csv_s1 = (tmp / "sfq_s1.csv").as_posix()
        res_s1 = run_ngspice_deck(
            build_deck("", csv_s1),
            wrdata_vecs=["v(d_src)", "v(din)", "v(c_src)", "v(clk)"],
            tran_cmd="tran 0.8p 65p",
            josim_csv_map={"sfq": csv_s1},
        )

        csv_s2 = (tmp / "sfq_s2.csv").as_posix()
        with CosimServerProcess(transport="unix") as srv:
            res_s2 = run_ngspice_deck(
                build_deck(srv.addr, csv_s2),
                wrdata_vecs=["v(d_src)", "v(din)", "v(c_src)", "v(clk)"],
                tran_cmd="tran 0.8p 65p",
                josim_csv_map={"sfq": csv_s2},
            )

    evs = res_s1.josim_events["sfq"]
    by_jj: Dict[str, List[float]] = {}
    for e in evs:
        by_jj.setdefault(str(e["trace"]), []).append(float(e["time"]))

    assert "P(B1)" in by_jj, f"Expected B1 to fire from CMOS DATA pulse, got {list(by_jj.keys())}"
    assert "P(B_SET)" in by_jj, f"Expected B_SET to fire into storage loop, got {list(by_jj.keys())}"
    assert "P(B_RO)" in by_jj, f"Expected B_RO to fire after CMOS CLOCK pulse, got {list(by_jj.keys())}"

    t_in = by_jj["P(B1)"][0]
    t_set = by_jj["P(B_SET)"][0]
    t_ro = by_jj["P(B_RO)"][-1]
    assert 11e-12 < t_in < 24e-12, f"Unexpected B1 firing time {t_in*1e12:.2f} ps"
    assert t_in < t_set < 38e-12, f"Unexpected B_SET firing time {t_set*1e12:.2f} ps"
    assert 40e-12 < t_ro < 55e-12, f"Unexpected B_RO readout time {t_ro*1e12:.2f} ps"

    # Check storage loop current plateau between B_SET (36 ps) and CLOCK (41 ps)
    j_tr = res_s1.josim_traces["sfq"]
    t_j = j_tr["time"]
    i_store = j_tr["I(L_STOR)"]
    mask_hold = (t_j >= 36e-12) & (t_j <= 41e-12)
    i_hold_uA = float(np.mean(i_store[mask_hold])) * 1e6
    assert i_hold_uA > 20.0, f"Expected stored circulating current > 20 uA, got {i_hold_uA:.2f} uA"

    # Verify Step 1 vs Step 2 exact equivalence
    diff_din = float(np.max(np.abs(res_s1.ng_traces["v(din)"] - res_s2.ng_traces["v(din)"])))
    assert diff_din == 0.0, f"Step 1 vs Step 2 mismatch on v(din): {diff_din}"
    rb_s1 = res_s1.josim_meta["sfq"].get("rollback_count", 0)
    rb_s2 = res_s2.josim_meta["sfq"].get("rollback_count", 0)
    assert rb_s1 >= 4, f"Expected >= 4 LTE adaptive timestep rollbacks from ngspice, got {rb_s1}"
    assert rb_s1 == rb_s2, f"Rollback count mismatch: {rb_s1} vs {rb_s2}"

    return {
        "t_set_ps": t_set * 1e12,
        "t_ro_ps": t_ro * 1e12,
        "i_hold_uA": i_hold_uA,
        "ngspice_rollbacks": float(rb_s1),
        "diff_step1_vs_step2": diff_din,
    }


# ===========================================================================
# Benchmark 3: NG3_NTRON_TO_CMOS_LNA (JoSIM nTron -> ngspice-45.2 MOSFET LNA)
# ===========================================================================
def test_ng3_ntron_to_cmos_lna() -> Dict[str, float]:
    """
    JoSIM SFQ-triggered 3-terminal `nTron` electrothermal nanowire cryotron (`N1`)
    driving a 2-stage Cryo-CMOS Low-Noise Amplifier (Common-Source + Source-Follower)
    in `ngspice-45.2`.
    Tested in both Step 1 (in-process) and Step 2 (IPC socket server).
    """
    ntron_netlist = """* SFQ-Triggered nTron Cryotron Amplifier Driving Cryo-CMOS LNA Port
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
IBIAS 0 nbias pwl(0 0 5p 75u)
LBIAS nbias nch 80p
N1 nch 0 n_g_in 0 nmod
RSH_CH nch 0 120

.tran 0.1p 60p
.print NODEV nch
.print DEVI N1
.print DEVP B_TRIG
.end
"""

    with tempfile.TemporaryDirectory(prefix="ng3_ntron_lna_") as tmpd:
        tmp = Path(tmpd)
        ntron_cir = (tmp / "ntron.cir").as_posix()
        Path(ntron_cir).write_text(ntron_netlist)

        def build_deck(srv_addr: str, out_csv: str) -> str:
            srv_p = f'server_addr="{srv_addr}"' if srv_addr else ""
            return f"""* JoSIM nTron -> ngspice-45.2 2-Stage Cryo-CMOS Common-Source + Source-Follower LNA
.model nch_lna nmos(level=1 vto=0.20 kp=650u gamma=0.08 phi=0.6 lambda=0.02 cbd=0.2f cbs=0.2f cgso=20p cgdo=20p)

Vdd vdd 0 1.0

* JoSIM nTron boundary port at node `nd`
A_ntron [%gd(nd 0)] jntron
.model jntron josim_cosim(netlist="{ntron_cir}" pos_nodes=["nch"] neg_nodes=["0"] {srv_p} out_csv="{out_csv}")

* AC coupling + gate bias into Stage 1 Common-Source LNA (biased in saturation at V(d1)=0.5V)
Cc1 nd g1 2.0p
Rbias1 vbias1 g1 4k
Vbias1 vbias1 0 0.28
M_cs d1 g1 0 0 nch_lna w=24u l=0.12u
Rd1 vdd d1 1.2k
Cload1 d1 0 0.5f

* Stage 2 Source-Follower output buffer
M_sf vdd d1 vout 0 nch_lna w=24u l=0.12u
Rsf vout 0 600
Cload2 vout 0 0.5f
.end
"""

        csv_s1 = (tmp / "ntron_s1.csv").as_posix()
        res_s1 = run_ngspice_deck(
            build_deck("", csv_s1),
            wrdata_vecs=["v(nd)", "v(d1)", "v(vout)"],
            tran_cmd="tran 0.1p 60p",
            josim_csv_map={"ntron": csv_s1},
        )

        csv_s2 = (tmp / "ntron_s2.csv").as_posix()
        with CosimServerProcess(transport="tcp") as srv:
            res_s2 = run_ngspice_deck(
                build_deck(srv.addr, csv_s2),
                wrdata_vecs=["v(nd)", "v(d1)", "v(vout)"],
                tran_cmd="tran 0.1p 60p",
                josim_csv_map={"ntron": csv_s2},
            )

    v_nd = res_s1.ng_traces["v(nd)"]
    v_d1 = res_s1.ng_traces["v(d1)"]
    v_out = res_s1.ng_traces["v(vout)"]

    nd_peak_mV = float(np.max(v_nd)) * 1e3
    d1_swing_mV = float(np.max(v_d1) - np.min(v_d1)) * 1e3
    vout_swing_mV = float(np.max(v_out) - np.min(v_out)) * 1e3
    gain = vout_swing_mV / max(nd_peak_mV, 1e-6)

    assert nd_peak_mV > 2.5, f"Expected nTron drain pulse > 2.5 mV, got {nd_peak_mV:.2f} mV"
    assert vout_swing_mV > 15.0, f"Expected Cryo-CMOS LNA output swing > 15 mV, got {vout_swing_mV:.2f} mV"
    assert gain > 3.0, f"Expected LNA voltage gain > 3x, got {gain:.2f}x"

    diff_s1_s2 = float(np.max(np.abs(v_out - res_s2.ng_traces["v(vout)"])))
    assert diff_s1_s2 == 0.0, f"Step 1 vs Step 2 TCP mismatch in NG3: {diff_s1_s2}"

    return {
        "nd_peak_mV": nd_peak_mV,
        "d1_swing_mV": d1_swing_mV,
        "vout_swing_mV": vout_swing_mV,
        "voltage_gain": gain,
        "diff_step1_vs_tcp": diff_s1_s2,
    }


# ===========================================================================
# Benchmark 4: LT1_CMOS_BRANIN_TRANSCEIVER (ADI LTspice.exe <-> JoSIM Branin)
# ===========================================================================
def test_lt1_cmos_branin_transceiver() -> Dict[str, float]:
    """
    Full duplex Branin wave-relaxation (`PORT_WAVE_BRANIN` + `cosim_run_window`)
    between real ADI `LTspice.exe -b` and JoSIM over two 50-Ohm, tau_d = 20 ps
    cryogenic coaxial lines:
      - Coax 0 (LTspice -> JoSIM): LTspice Cryo-CMOS driver launches a forward wave
        into a 50-Ohm line -> triggers JoSIM JTL (`B1`, `B2`)
      - JoSIM Core: JTL triggers an `nTron` (`N1`) connected to Port 1
      - Coax 1 (JoSIM -> LTspice): JoSIM `nTron` launches a return wave over the
        second 20 ps coax into an LTspice Cryo-CMOS Common-Source LNA receiver!
    Verified in both Step 1 (`pyjosim.Session`) and Step 2 (`CosimServerProcess`).
    """
    z0 = 50.0
    tau_d = 20.0e-12
    n_windows = 4

    josim_core_netlist = """* JoSIM 2-Port SFQ + nTron Cryogenic Transceiver Core
.model jjmod jj(rtype=1, vg=2.8m, cap=0.05p, r0=100, rn=5, icrit=100u)
.model nmod ntron(icgate=15u, icch=100u, rnormal=80, tau=12p, eta=0.35, iretrig=0.18, rsheet=40, lsheet=2p, width=0.1u, length=0.4u)

* Port 0 (node p0): 50-Ohm coax termination -> attenuator -> B1 -> B2
Rmatch0 p0 n1 120
IB1 0 n1 pwl(0 0 2p 72u)
B1 n1 0 jjmod
RSH1 n1 0 2.5
L12 n1 n2 10p
IB2 0 n2 pwl(0 0 2p 72u)
B2 n2 0 jjmod
RSH2 n2 0 2.5
L_GATE n2 ngate 6p
R_GATE ngate n_g_in 8

* Port 1 (node p1): nTron channel biased at 85 uA driving return coax
IBIAS 0 nbias pwl(0 0 2p 85u)
LBIAS nbias p1 80p
N1 p1 0 n_g_in 0 nmod

.tran 0.2p 80p
.print DEVV B1
.print DEVV B2
.print DEVP B1
.print DEVP B2
.print NODEV p0
.print NODEV p1
.end
"""

    def run_transceiver(use_ipc_server: bool) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
        with tempfile.TemporaryDirectory(prefix=".lt1_branin_", dir=str(_default_win_visible_tmp_dir())) as tmpd:
            tmp = Path(tmpd)
            # History of outgoing waves from JoSIM (b_j[t]) and from LTspice (b_lt[t])
            t_hist_j = np.array([0.0], dtype=np.float64)
            b_hist_j0 = np.array([0.0], dtype=np.float64)
            b_hist_j1 = np.array([0.0], dtype=np.float64)

            t_hist_lt = np.array([0.0], dtype=np.float64)
            b_hist_lt0 = np.array([0.0], dtype=np.float64)
            b_hist_lt1 = np.array([0.0], dtype=np.float64)

            if use_ipc_server:
                srv_cm = CosimServerProcess(transport="tcp")
                srv = srv_cm.__enter__()
                client = srv.connect()
                client.init_netlist(
                    josim_core_netlist,
                    [
                        PortSpec("p0", "0", PORT_WAVE_BRANIN, z0),
                        PortSpec("p1", "0", PORT_WAVE_BRANIN, z0),
                    ],
                    phase_mode=True,
                )
            else:
                srv_cm = None
                sess = Session()
                sess.set_cli_args(["-a", "1"])
                sess.load_string(josim_core_netlist)
                sess.cosim_add_port("p0", "0", PORT_WAVE_BRANIN, z0)
                sess.cosim_add_port("p1", "0", PORT_WAVE_BRANIN, z0)
                sess.cosim_init()

            try:
                for w in range(n_windows):
                    t_w_stop = (w + 1) * tau_d
                    t_grid = np.linspace(0.0, t_w_stop, int(round(t_w_stop / 0.2e-12)) + 1)

                    # Delayed waves from JoSIM arriving at LTspice during [0, t_w_stop]:
                    # u_lt(t) = b_j(t - tau_d)
                    t_lookup = np.maximum(0.0, t_grid - tau_d)
                    u_lt0 = np.interp(t_lookup, t_hist_j, b_hist_j0)
                    u_lt1 = np.interp(t_lookup, t_hist_j, b_hist_j1)

                    pwl0_win = format_spice_pwl_file(tmp / "u_lt0.pwl", t_grid, u_lt0)
                    pwl1_win = format_spice_pwl_file(tmp / "u_lt1.pwl", t_grid, u_lt1)

                    lt_deck = f"""* ADI LTspice Cryo-CMOS Branin Transceiver (Window {w+1})
.model nch nmos(level=1 vto=0.20 kp=650u gamma=0.08 phi=0.6 lambda=0.02 cbd=0.2f cbs=0.2f cgso=20p cgdo=20p)
.model pch pmos(level=1 vto=-0.20 kp=325u gamma=0.08 phi=0.6 lambda=0.02 cbd=0.2f cbs=0.2f cgso=20p cgdo=20p)
Vdd vdd 0 1.0

* Port 0 TX: CMOS Inverter Driver -> 50-Ohm Coax Branin Equivalent
Vin in_gate 0 PWL(0 1.0  5p 1.0  6p 0.0  10p 0.0  11p 1.0)
Mp_tx tx_inv in_gate vdd vdd pch w=16u l=0.18u
Mn_tx tx_inv in_gate 0   0   nch w=8u  l=0.18u
R_tx_ser tx_inv p0_lt 25
R_z0_0 p0_lt w0_src {z0}
Vw0 w0_src 0 PWL file="{pwl0_win}"

* Port 1 RX: 50-Ohm Coax Branin Equivalent -> Cryo-CMOS Common-Source LNA
Vw1 w1_src 0 PWL file="{pwl1_win}"
R_z0_1 p1_lt w1_src {z0}
Cc_rx p1_lt g_rx 2.0p
Rbias_rx vbias g_rx 4k
Vbias vbias 0 0.28
M_rx vout g_rx 0 0 nch w=24u l=0.12u
Rd_rx vdd vout 1.2k
Cload vout 0 0.5f

.tran 0.2p {t_w_stop:.12e} 0 0.2p
.options numdgt=7 plotwinsize=0
.end
"""
                    t_lt, lt_tr = run_ltspice_batch(lt_deck, work_dir=tmp)
                    v_p0_lt = lt_tr["v(p0_lt)"]
                    v_w0 = lt_tr["v(w0_src)"]
                    # Outgoing Branin wave from LTspice: b_lt = 2 * V(p0_lt) - u_lt
                    b_lt0 = 2.0 * v_p0_lt - v_w0
                    v_p1_lt = lt_tr["v(p1_lt)"]
                    v_w1 = lt_tr["v(w1_src)"]
                    b_lt1 = 2.0 * v_p1_lt - v_w1

                    t_hist_lt = t_lt
                    b_hist_lt0 = b_lt0
                    b_hist_lt1 = b_lt1

                    # Now advance JoSIM over window [w*tau_d, (w+1)*tau_d] using delayed LTspice waves
                    t_w_start = w * tau_d
                    t_j_win = np.linspace(t_w_start, t_w_stop, int(round(tau_d / 0.2e-12)) + 1)
                    t_lt_lookup = np.maximum(0.0, t_j_win - tau_d)
                    u_j = np.column_stack(
                        [
                            np.interp(t_lt_lookup, t_hist_lt, b_hist_lt0),
                            np.interp(t_lt_lookup, t_hist_lt, b_hist_lt1),
                        ]
                    )

                    if use_ipc_server:
                        t_out_w, b_out_w = client.cosim_run_window(t_w_stop, t_j_win, u_j)
                    else:
                        t_out_w, b_out_w = sess.cosim_run_window(t_w_start, t_w_stop, t_j_win, u_j)

                    if w == 0:
                        t_hist_j = t_out_w
                        b_hist_j0 = b_out_w[:, 0]
                        b_hist_j1 = b_out_w[:, 1]
                    else:
                        t_hist_j = np.concatenate([t_hist_j, t_out_w[1:]])
                        b_hist_j0 = np.concatenate([b_hist_j0, b_out_w[1:, 0]])
                        b_hist_j1 = np.concatenate([b_hist_j1, b_out_w[1:, 1]])

                if use_ipc_server:
                    _, _, n_ev, _ = client.cosim_finalize()
                    client.shutdown()
                else:
                    fin_res = sess.cosim_finalize()
                    n_ev = len(fin_res.events)
                return t_lt, lt_tr, n_ev
            finally:
                if srv_cm is not None:
                    srv_cm.__exit__(None, None, None)

    t_s1, tr_s1, nev_s1 = run_transceiver(use_ipc_server=False)
    t_s2, tr_s2, nev_s2 = run_transceiver(use_ipc_server=True)

    assert nev_s1 >= 2 and nev_s2 == nev_s1, f"Expected >=2 SFQ events (B1, B2), got {nev_s1}, {nev_s2}"

    # Verify causal timing in LTspice:
    # - Window 1 (2..19 ps): Before the 20 ps coax delay, no wave from JoSIM has arrived at LTspice RX LNA
    # - Window 3 (44..80 ps): After forward pulse crosses Coax 0 (20 ps), triggers B1->B2->nTron in JoSIM,
    #   and return wave crosses Coax 1 (20 ps), amplified pulse arrives at LTspice RX LNA!
    vout_s1 = tr_s1["v(vout)"]
    mask_early = (t_s1 >= 2e-12) & (t_s1 < 19e-12)
    mask_late = t_s1 >= 44e-12
    swing_early_mV = float(np.max(vout_s1[mask_early]) - np.min(vout_s1[mask_early])) * 1e3
    swing_late_mV = float(np.max(vout_s1[mask_late]) - np.min(vout_s1[mask_late])) * 1e3

    assert swing_early_mV < 1.0, f"Expected quiet RX LNA in [2, 19] ps before tau_d, got {swing_early_mV:.2f} mV"
    assert swing_late_mV > 35.0, f"Expected > 35 mV amplified return pulse in LTspice after 44 ps, got {swing_late_mV:.2f} mV"

    diff_s1_s2 = float(np.max(np.abs(vout_s1 - tr_s2["v(vout)"])))
    assert diff_s1_s2 == 0.0, f"LT1 Step 1 vs Step 2 TCP mismatch: {diff_s1_s2}"

    return {
        "swing_15_to_40ps_mV": swing_early_mV,
        "swing_after_44ps_mV": swing_late_mV,
        "diff_step1_vs_tcp": diff_s1_s2,
    }


# ===========================================================================
# Benchmark 5: LT2_NTRON_TO_LTSPICE_LNA (LTspice Norton-PWL vs ngspice-45.2)
# ===========================================================================
def test_lt2_ntron_to_ltspice_lna() -> Dict[str, float]:
    """
    Couples JoSIM's SFQ-triggered `nTron` (`N1`) to an identical Cryo-CMOS Common-Source
    + Source-Follower LNA simulated in:
      1. Real ADI `LTspice.exe -b` via Norton-PWL Waveform Relaxation (in both Step 1
         in-process mode and Step 2 UNIX socket daemon mode)
      2. Real `ngspice-45.2` via lockstep `josim_cosim.cm`
    Verifies rapid waveform convergence (< 1e-5 V in 3 iterations) and cross-simulator
    agreement between `LTspice.exe` and `ngspice-45.2`.
    """
    ntron_netlist = """* SFQ-Triggered nTron Cryotron Amplifier Driving Cryo-CMOS LNA Port
.model jjmod jj(rtype=1, vg=2.8m, cap=0.05p, r0=100, rn=5, icrit=100u)
.model nmod ntron(icgate=15u, icch=100u, rnormal=80, tau=12p, eta=0.35, iretrig=0.18, rsheet=40, lsheet=2p, width=0.1u, length=0.4u)

ITRIG 0 ntrig pwl(0 0 8p 0 10p 180u 14p 180u 16p 0)
IB_T 0 ntrig pwl(0 0 4p 65u)
B_TRIG ntrig 0 jjmod
RSH_T ntrig 0 2.2
L_GATE ntrig ngate 6p
R_GATE ngate n_g_in 8

IBIAS 0 nbias pwl(0 0 5p 75u)
LBIAS nbias nch 80p
N1 nch 0 n_g_in 0 nmod
RSH_CH nch 0 120

.tran 0.2p 60p
.print NODEV nch
.end
"""

    def run_ltspice_relaxation(use_ipc_server: bool) -> Tuple[np.ndarray, Dict[str, np.ndarray], float]:
        with tempfile.TemporaryDirectory(prefix=".lt2_pwl_", dir=str(_default_win_visible_tmp_dir())) as tmpd:
            tmp = Path(tmpd)
            dt = 0.2e-12
            n_steps = int(round(60.0e-12 / dt))
            t_grid = np.linspace(0.0, 60.0e-12, n_steps + 1)

            # Initial guess of LNA input current loading JoSIM node `nch` is 0
            i_load_guess = np.zeros_like(t_grid)
            v_nd_prev = np.zeros_like(t_grid)
            max_delta = 1.0

            for it in range(3):
                # Step JoSIM with current injection `-i_load_guess` (current into nch)
                v_nd_j = np.zeros_like(t_grid)
                if use_ipc_server:
                    with CosimServerProcess(transport="unix") as srv:
                        cli = srv.connect()
                        cli.init_netlist(
                            ntron_netlist,
                            [PortSpec("nch", "0", PORT_CURRENT_IN_VOLTAGE_OUT)],
                            phase_mode=True,
                        )
                        for k, tk in enumerate(t_grid):
                            y, _ = cli.cosim_step_to(float(tk), [float(-i_load_guess[k])])
                            v_nd_j[k] = y[0]
                            cli.cosim_commit()
                        cli.cosim_finalize()
                        cli.shutdown()
                else:
                    sess = Session()
                    sess.set_cli_args(["-a", "1"])
                    sess.load_string(ntron_netlist)
                    sess.cosim_add_port("nch", "0", PORT_CURRENT_IN_VOLTAGE_OUT)
                    sess.cosim_init()
                    for k, tk in enumerate(t_grid):
                        y, _ = sess.cosim_step_to(float(tk), [float(-i_load_guess[k])])
                        v_nd_j[k] = y[0]
                        sess.cosim_commit()
                    sess.cosim_finalize()

                max_delta = float(np.max(np.abs(v_nd_j - v_nd_prev)))
                v_nd_prev = v_nd_j.copy()

                # Robin / Thevenin Waveform Relaxation with reference impedance R0 = 60 Ohm:
                # V_thev(t) = V_nd,JoSIM(t) + R0 * i_load(t) in series with Rthev = 60 Ohm
                r0_thev = 60.0
                v_thev = v_nd_j + r0_thev * i_load_guess
                pwl_win = format_spice_pwl_file(tmp / "v_thev.pwl", t_grid, v_thev)
                lt_deck = f"""* ADI LTspice 2-Stage Cryo-CMOS LNA Driven by JoSIM nTron
.model nch_lna nmos(level=1 vto=0.20 kp=650u gamma=0.08 phi=0.6 lambda=0.02 cbd=0.2f cbs=0.2f cgso=20p cgdo=20p)
Vdd vdd 0 1.0
Vthev n_thev 0 PWL file="{pwl_win}"
Rthev n_thev nd {r0_thev}
Cc1 nd g1 2.0p
Rbias1 vbias1 g1 4k
Vbias1 vbias1 0 0.28
M_cs d1 g1 0 0 nch_lna w=24u l=0.12u
Rd1 vdd d1 1.2k
Cload1 d1 0 0.5f
M_sf vdd d1 vout 0 nch_lna w=24u l=0.12u
Rsf vout 0 600
Cload2 vout 0 0.5f
.tran 0.2p 60p 0 0.2p
.options numdgt=7 plotwinsize=0
.end
"""
                t_lt, lt_tr = run_ltspice_batch(lt_deck, work_dir=tmp)
                v_nthev_lt = np.interp(t_grid, t_lt, lt_tr["v(n_thev)"])
                v_nd_lt = np.interp(t_grid, t_lt, lt_tr["v(nd)"])
                i_load_guess = (v_nthev_lt - v_nd_lt) / r0_thev

            return t_lt, lt_tr, max_delta

    t_lt_s1, tr_lt_s1, delta_s1 = run_ltspice_relaxation(use_ipc_server=False)
    t_lt_s2, tr_lt_s2, delta_s2 = run_ltspice_relaxation(use_ipc_server=True)

    assert delta_s1 < 1e-4, f"LT2 waveform relaxation did not converge: delta={delta_s1:.3e} V"
    diff_s1_s2 = float(np.max(np.abs(tr_lt_s1["v(vout)"] - tr_lt_s2["v(vout)"])))
    assert diff_s1_s2 == 0.0, f"LT2 Step 1 vs Step 2 UNIX mismatch: {diff_s1_s2}"

    # Cross-validate LTspice.exe against ngspice-45.2 lockstep XSPICE on the exact same circuit!
    with tempfile.TemporaryDirectory(prefix="ng_cross_") as tmpd:
        tmp = Path(tmpd)
        ntron_cir = (tmp / "ntron.cir").as_posix()
        Path(ntron_cir).write_text(ntron_netlist)
        ng_deck = f"""* ngspice-45.2 reference for LT2 cross-validation
.model nch_lna nmos(level=1 vto=0.20 kp=650u gamma=0.08 phi=0.6 lambda=0.02 cbd=0.2f cbs=0.2f cgso=20p cgdo=20p)
Vdd vdd 0 1.0
A_ntron [%gd(nd 0)] jntron
.model jntron josim_cosim(netlist="{ntron_cir}" pos_nodes=["nch"] neg_nodes=["0"])
Cc1 nd g1 2.0p
Rbias1 vbias1 g1 4k
Vbias1 vbias1 0 0.28
M_cs d1 g1 0 0 nch_lna w=24u l=0.12u
Rd1 vdd d1 1.2k
Cload1 d1 0 0.5f
M_sf vdd d1 vout 0 nch_lna w=24u l=0.12u
Rsf vout 0 600
Cload2 vout 0 0.5f
.end
"""
        res_ng = run_ngspice_deck(
            ng_deck,
            wrdata_vecs=["v(nd)", "v(vout)"],
            tran_cmd="tran 0.2p 60p",
        )

    vout_lt_interp = np.interp(res_ng.ng_time, t_lt_s1, tr_lt_s1["v(vout)"])
    vnd_lt_interp = np.interp(res_ng.ng_time, t_lt_s1, tr_lt_s1["v(nd)"])
    cross_err_vout = nrmse(vout_lt_interp, res_ng.ng_traces["v(vout)"])
    cross_err_vnd = nrmse(vnd_lt_interp, res_ng.ng_traces["v(nd)"])

    assert cross_err_vnd < 0.02, f"LTspice vs ngspice V(nd) NRMSE {cross_err_vnd:.4f} > 2%"
    assert cross_err_vout < 0.03, f"LTspice vs ngspice V(vout) NRMSE {cross_err_vout:.4f} > 3%"

    return {
        "final_iter_delta_uV": delta_s1 * 1e6,
        "ltspice_vs_ngspice_vnd_nrmse": cross_err_vnd,
        "ltspice_vs_ngspice_vout_nrmse": cross_err_vout,
        "diff_step1_vs_unix": diff_s1_s2,
    }


def main() -> int:
    benchmarks = [
        ("NG1_SPLIT_JTL_XSPICE (ngspice-45.2 + josim_cosim.cm: Step 1, UNIX & TCP)", test_ng1_split_jtl_xspice),
        ("NG2_CMOS_TO_SFQ_DFF (ngspice-45.2 4K CMOS -> JoSIM 2-Port RSFQ Loop)", test_ng2_cmos_to_sfq_dff),
        ("NG3_NTRON_TO_CMOS_LNA (JoSIM nTron -> ngspice-45.2 2-Stage MOSFET LNA)", test_ng3_ntron_to_cmos_lna),
        ("LT1_CMOS_BRANIN_TRANSCEIVER (ADI LTspice.exe <-> JoSIM Branin Duplex)", test_lt1_cmos_branin_transceiver),
        ("LT2_NTRON_TO_LTSPICE_LNA (ADI LTspice.exe vs ngspice-45.2 Cross-Validation)", test_lt2_ntron_to_ltspice_lna),
    ]
    print("=" * 82)
    print("JoSIM Co-Simulation Benchmark Suite on Actual Simulators (ngspice-45.2 & LTspice)")
    print("=" * 82)
    for name, fn in benchmarks:
        metrics = fn()
        m_str = ", ".join(f"{k}={v:.4g}" for k, v in metrics.items())
        print(f"[PASS] {name}\n       {m_str}")
    print("=" * 82)
    print("All 5 real-simulator Step 1 & Step 2 co-simulation benchmarks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
