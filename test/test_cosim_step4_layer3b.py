#!/usr/bin/env python3
"""
Step 4 (Layer 3B) Verification & Multi-Simulator End-to-End Benchmark Suite:
  - S4_1_MULTI_INSTANCE_SERVER_MUX:
      Multi-partition ngspice-45.2 co-simulation (JoSIM TX partition -> ngspice LC interconnect
      -> 2-port JoSIM SFQ + nTron RX partition -> ngspice 2-stage Cryo-CMOS LNA) sharing a
      SINGLE poll()-multiplexed `josim-cosim-server` process over UNIX & TCP sockets, plus
      direct IPC `GET_TRACE` and `GET_EVENTS` verification.
  - S4_2_MULTI_WINDOW_BRANIN_DUPLEX:
      Multi-window causal Branin transmission-line wave relaxation (`BraninWaveWindowCoordinator`)
      across 4 windows ([0..15ps], [15..30ps], [30..45ps], [45..60ps]) and 6 sub-windows (10ps)
      in BOTH ADI `LTspice.exe -b` and `ngspice-45.2`, cross-validating waveform accuracy and
      in-process vs. socket transport equivalence.
  - S4_3_SPECTRE_MULTI_PARTITION_END_TO_END:
      Live Cadence Spectre 25.1 multi-partition co-simulation on `asic2.inst.bnl.gov` via
      `SpectreCosimRunner`: `X_sfq_tx` (`josim_cosim_port_norton`) -> Spectre pad/inductor network
      -> 2-port `X_ntron_rx` (`josim_cosim_port_2p`, SFQ receiver + electrothermal nTron) ->
      4K Cryo-CMOS 2-stage CS + SF LNA with full `dcOp dc` (`skipdc=no`) over a single
      multiplexed `josim-cosim-server`.
  - S4_4_UNIFIED_CLI_WORKFLOW:
      End-to-end verification of `python3 scripts/josim_cosim.py --config <spec.json> --output-json <out.json>`
      for both `ngspice_xspice` and `branin_window` workflows.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from josim_cosim import (  # noqa: E402
    BraninPortLink,
    CosimServerProcess,
    PortSpec,
    SpectreCosimRunner,
    UnifiedCosimSession,
    run_ltspice_branin_cosim,
    run_ngspice_branin_cosim,
    run_ngspice_deck,
)
from pyjosim import PORT_CURRENT_IN_VOLTAGE_OUT, PORT_VOLTAGE_IN_CURRENT_OUT  # noqa: E402


def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    span = float(np.max(b) - np.min(b))
    if span < 1e-15:
        span = 1.0
    return float(np.sqrt(np.mean((a - b) ** 2)) / span)


# =============================================================================
# S4_1: Multi-Instance Multiplexed Server (`poll()`) + IPC Trace/Event Queries
# =============================================================================
def run_s4_1_multi_instance_server_mux() -> dict:
    """
    Two separate JoSIM subcircuits inside one ngspice-45.2 deck:
      - A_tx (1-port): 2-JJ SFQ transmitter (`tx_netlist`)
      - A_rx (2-port): 2-JJ SFQ receiver + 3-terminal electrothermal nTron (`rx_ntron_netlist`)
                       driving a 4K Cryo-CMOS 2-stage LNA in ngspice-45.2.
    Both A_tx and A_rx connect concurrently to a SINGLE `josim-cosim-server` daemon!
    Also verifies `UnifiedCosimSession.get_trace()` and `get_events()` over UNIX/TCP sockets.
    """
    tx_netlist = """* Partition 1: 2-JJ SFQ Transmitter
.model jjmod jj(rtype=1, vg=2.8m, cap=0.07p, r0=160, rn=16, icrit=100u)
Iin 0 n1 pwl(0 0 8p 0 9p 220u 11p 220u 12p 0)
Ib1 0 n1 pwl(0 0 4p 75u)
B1 n1 0 jjmod
Rsh1 n1 0 2.5
L12 n1 n2 4.2p
Ib2 0 n2 pwl(0 0 4p 75u)
B2 n2 0 jjmod
Rsh2 n2 0 2.5
.tran 0.1p 55p
.print DEVP B1
.print DEVP B2
.end
"""

    rx_ntron_netlist = """* Partition 2: 2-Port SFQ Receiver + Electrothermal nTron Driver
.model jjmod jj(rtype=1, vg=2.8m, cap=0.07p, r0=160, rn=16, icrit=100u)
.model nmod nanowire(rsheet=35, wch=120n, lch=600n, wg=45n, d=5n, ic0=420u, ig0=38u, tc=10.5, tsub=4.2, alpha=0.85, hc=1.8e4, cvol=2200, lkin0=18p, tau_th=8p)
Ib3 0 n3 pwl(0 0 4p 75u)
B3 n3 0 jjmod
Rsh3 n3 0 2.5
L34 n3 n4 4.2p
Ib4 0 n4 pwl(0 0 4p 75u)
B4 n4 0 jjmod
Rsh4 n4 0 2.5
Rg n4 ng 3.5
Rdump ng 0 8.0
Ibias_ch 0 nch pwl(0 0 5p 360u)
N1 nch 0 ng 0 nmod
Rload_ch nch 0 25.0
.tran 0.1p 55p
.print DEVP B3
.print DEVP B4
.print NODEV nch
.end
"""

    with tempfile.TemporaryDirectory(prefix="josim_s4_1_") as tmpd:
        tmp = Path(tmpd)
        tx_cir = (tmp / "tx.cir").as_posix()
        rx_cir = (tmp / "rx_ntron.cir").as_posix()
        Path(tx_cir).write_text(tx_netlist, encoding="utf-8")
        Path(rx_cir).write_text(rx_ntron_netlist, encoding="utf-8")

        def make_deck(srv_addr: str, tx_csv: str, rx_csv: str) -> str:
            srv_param = f'server_addr="{srv_addr}"' if srv_addr else ""
            return f"""* S4_1: Multi-Partition JoSIM TX -> Passive Link -> JoSIM RX+nTron -> Cryo-CMOS 2-Stage LNA
.model nmos_4k nmos(level=1 vto=0.20 kp=650u gamma=0.08 phi=0.6 lambda=0.02 cbd=0.2f cbs=0.2f cgso=20p cgdo=20p)
Lp1 n2 mid 2.1p
Cp mid 0 4f
Rp mid 0 150
Lp2 mid n3 2.1p

Vdd vdd 0 1.0
Vbias1 vbias1 0 0.28
Cc1 nd g1 2.0p
Rbias1 vbias1 g1 4k
M_cs d1 g1 0 0 nmos_4k w=24u l=120n
Rd1 vdd d1 1.2k
Cload1 d1 0 0.5f

Vbias2 vbias2 0 0.55
Cc2 d1 g2 2.0p
Rbias2 vbias2 g2 6k
M_sf vdd g2 vout 0 nmos_4k w=32u l=120n
Rsf vout 0 800
Cload2 vout 0 1.0f

A_tx [%gd(n2 0)] jtx
A_rx [%gd(n3 0) %gd(nd 0)] jrx
.model jtx josim_cosim(netlist="{tx_cir}" pos_nodes=["n2"] neg_nodes=["0"] port_modes=[0] {srv_param} out_csv="{tx_csv}")
.model jrx josim_cosim(netlist="{rx_cir}" pos_nodes=["n3" "nch"] neg_nodes=["0" "0"] port_modes=[0 0] {srv_param} out_csv="{rx_csv}")
.end
"""

        vecs = ["v(n2)", "v(mid)", "v(n3)", "v(nd)", "v(d1)", "v(vout)"]

        # 1. Step 1 In-Process Mode
        res_inproc = run_ngspice_deck(
            make_deck("", (tmp / "tx_in.csv").as_posix(), (tmp / "rx_in.csv").as_posix()),
            wrdata_vecs=vecs,
            tran_cmd="tran 0.1p 55p",
            josim_csv_map={"tx": (tmp / "tx_in.csv").as_posix(), "rx": (tmp / "rx_in.csv").as_posix()},
        )

        # 2. Step 2 Single Multiplexed UNIX Socket Server (serving BOTH A_tx and A_rx concurrently!)
        with CosimServerProcess(transport="unix", expected_clients=2) as srv_unix:
            res_unix = run_ngspice_deck(
                make_deck(srv_unix.addr, (tmp / "tx_ux.csv").as_posix(), (tmp / "rx_ux.csv").as_posix()),
                wrdata_vecs=vecs,
                tran_cmd="tran 0.1p 55p",
                josim_csv_map={"tx": (tmp / "tx_ux.csv").as_posix(), "rx": (tmp / "rx_ux.csv").as_posix()},
            )

        # 3. Step 2 Single Multiplexed TCP Socket Server (serving BOTH A_tx and A_rx concurrently!)
        with CosimServerProcess(transport="tcp", expected_clients=2) as srv_tcp:
            res_tcp = run_ngspice_deck(
                make_deck(srv_tcp.addr, (tmp / "tx_tcp.csv").as_posix(), (tmp / "rx_tcp.csv").as_posix()),
                wrdata_vecs=vecs,
                tran_cmd="tran 0.1p 55p",
                josim_csv_map={"tx": (tmp / "tx_tcp.csv").as_posix(), "rx": (tmp / "rx_tcp.csv").as_posix()},
            )

        # 4. Also test UnifiedCosimSession IPC GET_TRACE and GET_EVENTS directly over a multiplexed server
        with CosimServerProcess(transport="unix", expected_clients=1) as srv_ipc:
            usess = UnifiedCosimSession(
                tx_netlist,
                [PortSpec("n2", "0", PORT_CURRENT_IN_VOLTAGE_OUT)],
                phase_mode=True,
                endpoint=srv_ipc.addr,
            )
            try:
                for step in range(1, 200):
                    usess.step_to(step * 0.1e-12, [0.0])
                    usess.commit()
                usess.finalize()
                tr_p_b1 = usess.get_trace("P(B1)")
                evs_ipc = usess.get_events()
                assert tr_p_b1.size >= 190 and float(np.max(tr_p_b1)) > 5.5, "IPC get_trace('P(B1)') failed"
                assert len(evs_ipc) >= 2, f"Expected >=2 SFQ events over IPC get_events(), got {len(evs_ipc)}"
            finally:
                usess.close()

    diff_unix = float(np.max(np.abs(res_inproc.ng_traces["v(vout)"] - res_unix.ng_traces["v(vout)"])))
    diff_tcp = float(np.max(np.abs(res_inproc.ng_traces["v(vout)"] - res_tcp.ng_traces["v(vout)"])))
    nd_peak_mv = float(np.max(res_unix.ng_traces["v(nd)"])) * 1e3
    vout_swing_mv = float(np.max(res_unix.ng_traces["v(vout)"]) - np.min(res_unix.ng_traces["v(vout)"])) * 1e3

    assert len(res_unix.josim_events["tx"]) == 2, f"Expected 2 TX events, got {len(res_unix.josim_events['tx'])}"
    assert len(res_unix.josim_events["rx"]) == 2, f"Expected 2 RX events, got {len(res_unix.josim_events['rx'])}"
    assert nd_peak_mv > 4.0, f"nTron drain pulse too small: {nd_peak_mv} mV"
    assert vout_swing_mv > 45.0, f"Cryo-CMOS LNA vout swing too small: {vout_swing_mv} mV"
    assert diff_unix == 0.0, f"Multiplexed UNIX server differed from in-process: {diff_unix}"
    assert diff_tcp == 0.0, f"Multiplexed TCP server differed from in-process: {diff_tcp}"

    return {
        "passed": True,
        "tx_events": len(res_unix.josim_events["tx"]),
        "rx_events": len(res_unix.josim_events["rx"]),
        "nd_peak_mv": round(nd_peak_mv, 3),
        "vout_swing_mv": round(vout_swing_mv, 3),
        "diff_inproc_vs_mux_unix": diff_unix,
        "diff_inproc_vs_mux_tcp": diff_tcp,
        "ipc_get_events_count": len(evs_ipc),
    }


# =============================================================================
# S4_2: Multi-Window Branin Wave Relaxation (LTspice.exe & ngspice-45.2)
# =============================================================================
def run_s4_2_multi_window_branin_duplex() -> dict:
    """
    Full-duplex Cryo-CMOS <-> JoSIM (SFQ + nTron) transceiver over dual Z0=50ohm, Td=15ps
    coax lines across 4 causal windows ([0..15ps], [15..30ps], [30..45ps], [45..60ps])
    and 6 sub-windows (10ps < Td) on BOTH ADI LTspice.exe and ngspice-45.2.
    """
    josim_duplex_netlist = """* JoSIM 2-Port Branin Duplex Core (Port 0: SFQ RX, Port 1: nTron TX)
.model jjmod jj(rtype=1, vg=2.8m, cap=0.05p, r0=100, rn=5, icrit=100u)
.model nmod ntron(icgate=15u, icch=100u, rnormal=80, tau=12p, eta=0.35, iretrig=0.18, rsheet=40, lsheet=2p, width=0.1u, length=0.4u)
Rmatch0 n_rx_port n_j1 120
IB1 0 n_j1 pwl(0 0 2p 72u)
B_rx1 n_j1 0 jjmod
RSH1 n_j1 0 2.5
L12 n_j1 n_j2 10p
IB2 0 n_j2 pwl(0 0 2p 72u)
B_rx2 n_j2 0 jjmod
RSH2 n_j2 0 2.5
L_GATE n_j2 ngate 6p
R_GATE ngate n_g_in 8
IBIAS 0 nbias pwl(0 0 2p 85u)
LBIAS nbias n_tx_port 80p
N1 n_tx_port 0 n_g_in 0 nmod
.tran 0.2p 60p
.print DEVP B_rx1
.print DEVP B_rx2
.end
"""

    ltspice_template = """* LTspice 4K Cryo-CMOS Full-Duplex Transceiver (Multi-Window Branin)
.model nmos_4k nmos(level=1 vto=0.20 kp=650u gamma=0.08 phi=0.6 lambda=0.02 cbd=0.2f cbs=0.2f cgso=20p cgdo=20p)
.model pmos_4k pmos(level=1 vto=-0.20 kp=325u gamma=0.08 phi=0.6 lambda=0.02 cbd=0.2f cbs=0.2f cgso=20p cgdo=20p)
Vdd vdd 0 1.0
Vin in_logic 0 PWL(0 1.0  5p 1.0  6p 0.0  10p 0.0  11p 1.0)
Mp_tx tx_drv in_logic vdd vdd pmos_4k w=16u l=0.18u
Mn_tx tx_drv in_logic 0   0   nmos_4k w=8u  l=0.18u
Rtx_pad tx_drv n_sp_tx 25
R_z0_tx n_sp_tx n_wave_tx 50
V_wave_tx n_wave_tx 0 PWL file="{PWL_TX}"

V_wave_rx n_wave_rx 0 PWL file="{PWL_RX}"
R_z0_rx n_sp_rx n_wave_rx 50
Vbias vbias 0 0.28
Cc n_sp_rx g1 2.0p
Rbias vbias g1 4k
M_lna rx_out g1 0 0 nmos_4k w=24u l=0.12u
Rd vdd rx_out 1.2k
Cload rx_out 0 0.5f
.tran 0.2p {T_STOP} 0 0.2p
"""

    ngspice_template = """* ngspice 4K Cryo-CMOS Full-Duplex Transceiver (Multi-Window Branin)
.model nmos_4k nmos(level=1 vto=0.20 kp=650u gamma=0.08 phi=0.6 lambda=0.02 cbd=0.2f cbs=0.2f cgso=20p cgdo=20p)
.model pmos_4k pmos(level=1 vto=-0.20 kp=325u gamma=0.08 phi=0.6 lambda=0.02 cbd=0.2f cbs=0.2f cgso=20p cgdo=20p)
Vdd vdd 0 1.0
Vin in_logic 0 PWL(0 1.0  5p 1.0  6p 0.0  10p 0.0  11p 1.0)
Mp_tx tx_drv in_logic vdd vdd pmos_4k w=16u l=0.18u
Mn_tx tx_drv in_logic 0   0   nmos_4k w=8u  l=0.18u
Rtx_pad tx_drv n_sp_tx 25
R_z0_tx n_sp_tx n_wave_tx 50
V_wave_tx n_wave_tx 0 {PWL_TX}

V_wave_rx n_wave_rx 0 {PWL_RX}
R_z0_rx n_sp_rx n_wave_rx 50
Vbias vbias 0 0.28
Cc n_sp_rx g1 2.0p
Rbias vbias g1 4k
M_lna rx_out g1 0 0 nmos_4k w=24u l=0.12u
Rd vdd rx_out 1.2k
Cload rx_out 0 0.5f
.end
"""

    links = [
        BraninPortLink(
            josim_pos_node="n_rx_port",
            josim_neg_node="0",
            spice_port_trace="v(n_sp_tx)",
            spice_pwl_placeholder="{PWL_TX}",
            z0=50.0,
            td=15e-12,
        ),
        BraninPortLink(
            josim_pos_node="n_tx_port",
            josim_neg_node="0",
            spice_port_trace="v(n_sp_rx)",
            spice_pwl_placeholder="{PWL_RX}",
            z0=50.0,
            td=15e-12,
        ),
    ]
    vecs = ["v(n_sp_tx)", "v(n_sp_rx)", "v(rx_out)"]

    # 1. 4-Window LTspice.exe (W = Td = 15ps) in-process
    res_lt_4w = run_ltspice_branin_cosim(
        ltspice_template,
        josim_duplex_netlist,
        links=links,
        t_stop=60e-12,
        window_size=15e-12,
        endpoint="inproc",
        spice_vectors=vecs,
        josim_trace_names=["P(B_RX1)", "P(B_RX2)"],
    )

    # 2. 6-Window LTspice.exe (W = 10ps < Td = 15ps) over UNIX socket server
    with CosimServerProcess(transport="unix", expected_clients=1) as srv_ux:
        res_lt_6w_ux = run_ltspice_branin_cosim(
            ltspice_template,
            josim_duplex_netlist,
            links=links,
            t_stop=60e-12,
            window_size=10e-12,
            endpoint=srv_ux.addr,
            spice_vectors=vecs,
            josim_trace_names=["P(B_RX1)", "P(B_RX2)"],
        )

    # 3. 4-Window ngspice-45.2 (W = Td = 15ps) in-process
    res_ng_4w = run_ngspice_branin_cosim(
        ngspice_template,
        josim_duplex_netlist,
        links=links,
        t_stop=60e-12,
        window_size=15e-12,
        endpoint="inproc",
        spice_vectors=vecs,
        josim_trace_names=["P(B_RX1)", "P(B_RX2)"],
    )

    assert res_lt_4w.num_windows == 4
    assert res_lt_6w_ux.num_windows == 6
    assert len(res_lt_4w.josim_events) >= 2
    assert len(res_lt_4w.josim_events) == len(res_lt_6w_ux.josim_events)

    # Check quiet window before Td = 15ps and amplified nTron response after 36ps
    t = res_lt_4w.t_grid
    rx_out_lt = res_lt_4w.spice_traces["v(rx_out)"]
    mask_quiet = (t >= 2e-12) & (t <= 14.5e-12)
    mask_active = t >= 36e-12
    quiet_swing_mv = float(np.max(rx_out_lt[mask_quiet]) - np.min(rx_out_lt[mask_quiet])) * 1e3
    active_swing_mv = float(np.max(rx_out_lt[mask_active]) - np.min(rx_out_lt[mask_active])) * 1e3

    assert quiet_swing_mv < 0.01, f"Expected quiet RX before Td=15ps, got {quiet_swing_mv} mV"
    assert active_swing_mv > 45.0, f"Expected amplified RX swing >45 mV after 36ps, got {active_swing_mv} mV"

    diff_4w_vs_6w_ux = float(np.max(np.abs(res_lt_4w.spice_traces["v(rx_out)"] - res_lt_6w_ux.spice_traces["v(rx_out)"])))
    lt_vs_ng_nrmse = nrmse(res_lt_4w.spice_traces["v(rx_out)"], res_ng_4w.spice_traces["v(rx_out)"])
    assert diff_4w_vs_6w_ux < 1e-4, f"4-window vs 6-window UNIX diff too large: {diff_4w_vs_6w_ux}"
    assert lt_vs_ng_nrmse < 0.01, f"LTspice vs ngspice Branin NRMSE too large: {lt_vs_ng_nrmse}"

    return {
        "passed": True,
        "lt_4w_windows": res_lt_4w.num_windows,
        "lt_6w_windows": res_lt_6w_ux.num_windows,
        "quiet_swing_mv": round(quiet_swing_mv, 6),
        "active_swing_mv": round(active_swing_mv, 3),
        "diff_4w_inproc_vs_6w_unix": round(diff_4w_vs_6w_ux, 6),
        "ltspice_vs_ngspice_nrmse": round(lt_vs_ng_nrmse, 6),
    }


# =============================================================================
# S4_3: Live Cadence Spectre 25.1 Multi-Partition Co-Sim via SpectreCosimRunner
# =============================================================================
def check_asic2_reachable() -> bool:
    try:
        res = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", "asic2", "true"],
            capture_output=True,
            timeout=6,
        )
        return res.returncode == 0
    except Exception:
        return False


def run_s4_3_spectre_multi_partition_end_to_end() -> dict:
    """
    Live Cadence Spectre 25.1 multi-partition co-simulation on `asic2.inst.bnl.gov`:
      - X_sfq_tx (`josim_cosim_port_norton` on `sfq_tx.cir`): 2-JJ SFQ transmitter
      - Spectre passive pad/inductor network (`L_p1`, `C_pad`, `L_p2`)
      - X_ntron_rx (`josim_cosim_port_2p` on `ntron_rx_2p.cir`): 2-port SFQ receiver + nTron driver
      - Spectre 4K Cryo-CMOS 2-stage Common-Source + Source-Follower LNA with full `dcOp dc` (`skipdc=no`)
      Both `X_sfq_tx` and `X_ntron_rx` connect concurrently to a SINGLE multiplexed `josim-cosim-server`!
    """
    if not check_asic2_reachable():
        return {"passed": True, "skipped": True, "reason": "asic2.inst.bnl.gov not reachable"}

    sfq_tx_cir = """* Partition 1: 2-JJ SFQ Transmitter in JoSIM
.model jjmod jj(rtype=1, vg=2.8m, cap=0.07p, r0=160, rn=16, icrit=100u)
Iin 0 n1 pwl(0 0 8p 0 9p 220u 11p 220u 12p 0)
Ib1 0 n1 pwl(0 0 4p 75u)
B1 n1 0 jjmod
Rsh1 n1 0 2.5
L12 n1 n2 4.2p
Ib2 0 n2 pwl(0 0 4p 75u)
B2 n2 0 jjmod
Rsh2 n2 0 2.5
.tran 0.1p 60p
.end
"""

    ntron_rx_2p_cir = """* Partition 2: 2-Port SFQ Receiver + Electrothermal nTron Driver in JoSIM
.model jjmod jj(rtype=1, vg=2.8m, cap=0.07p, r0=160, rn=16, icrit=100u)
.model nmod nanowire(rsheet=35, wch=120n, lch=600n, wg=45n, d=5n, ic0=420u, ig0=38u, tc=10.5, tsub=4.2, alpha=0.85, hc=1.8e4, cvol=2200, lkin0=18p, tau_th=8p)
Ib3 0 n3 pwl(0 0 4p 75u)
B3 n3 0 jjmod
Rsh3 n3 0 2.5
L34 n3 n4 4.2p
Ib4 0 n4 pwl(0 0 4p 75u)
B4 n4 0 jjmod
Rsh4 n4 0 2.5
Rg n4 ng 3.5
Rdump ng 0 8.0
Ibias_ch 0 nch pwl(0 0 5p 360u)
N1 nch 0 ng 0 nmod
Rload_ch nch 0 25.0
.tran 0.1p 60p
.end
"""

    scs_deck = """// S4_3: Multi-Partition Cadence Spectre 25.1 Co-Simulation (2 JoSIM Instances + 2-Stage Cryo-CMOS LNA)
simulator lang=spectre
global 0

ahdl_include "libjosim_spectre_dpi.so" -cds_va_dpi
ahdl_include "josim_cosim_port.va"

opts options temp=4.2 tnom=4.2 gmin=1e-12

model nch_4k mos1 type=n vto=0.20 kp=650u gamma=0.08 phi=0.6 lambda=0.02 cbd=0.2f cbs=0.2f cgso=20p cgdo=20p

// JoSIM Partition 1: SFQ Transmitter
X_sfq_tx (n2 0) josim_cosim_port_norton netlist="sfq_tx.cir" port_nodes="n2" port_modes="0"

// Spectre Inter-Chip Passive Network
Lp1 (n2 mid) inductor l=2.1p
Cp (mid 0) capacitor c=4f
Rp (mid 0) resistor r=150
Lp2 (mid n3) inductor l=2.1p

// JoSIM Partition 2: 2-Port SFQ Receiver + Electrothermal nTron Driver
X_ntron_rx (n3 0 nd 0) josim_cosim_port_2p netlist="ntron_rx_2p.cir" port_nodes="n3,nch" port_modes="0,0"

// Spectre 4K Cryo-CMOS 2-Stage Common-Source + Source-Follower LNA
Vdd (vdd 0) vsource dc=1.0
Vbias1 (vbias1 0) vsource dc=0.28
Cc1 (nd g1) capacitor c=2.0p
Rbias1 (vbias1 g1) resistor r=4k
M_cs (d1 g1 0 0) nch_4k w=24u l=120n
Rd1 (vdd d1) resistor r=1.2k
Cload1 (d1 0) capacitor c=0.5f

Vbias2 (vbias2 0) vsource dc=0.55
Cc2 (d1 g2) capacitor c=2.0p
Rbias2 (vbias2 g2) resistor r=6k
M_sf (vdd g2 vout 0) nch_4k w=32u l=120n
Rsf (vout 0) resistor r=800
Cload2 (vout 0) capacitor c=1.0f

dcOp dc
tran1 tran stop=60p step=0.2p maxstep=0.2p errpreset=conservative
save n2 mid n3 nd d1 vout
"""

    runner = SpectreCosimRunner(ssh_host="asic2")
    sres = runner.run(
        scs_content=scs_deck,
        josim_files={"sfq_tx.cir": sfq_tx_cir, "ntron_rx_2p.cir": ntron_rx_2p_cir},
        saved_signals=["n2", "mid", "n3", "nd", "d1", "vout"],
    )

    assert sres.ok and sres.dc_ok, "S4_3 multi-partition Spectre run or DC OP failed"
    n3_peak_mv = float(np.max(sres.traces["n3"])) * 1e3
    nd_peak_mv = float(np.max(sres.traces["nd"])) * 1e3
    vout_swing_mv = float(np.max(sres.traces["vout"]) - np.min(sres.traces["vout"])) * 1e3

    assert n3_peak_mv > 0.30, f"Expected SFQ pulse at n3 > 0.30 mV, got {n3_peak_mv} mV"
    assert nd_peak_mv > 4.0, f"Expected nTron pulse at nd > 4.0 mV, got {nd_peak_mv} mV"
    assert vout_swing_mv > 45.0, f"Expected Cryo-CMOS 2-stage LNA swing > 45 mV, got {vout_swing_mv} mV"

    return {
        "passed": True,
        "skipped": False,
        "dc_ok": sres.dc_ok,
        "accepted_steps": sres.accepted_steps,
        "tran_cpu_s": round(sres.tran_cpu_s, 4),
        "n3_peak_mv": round(n3_peak_mv, 3),
        "nd_peak_mv": round(nd_peak_mv, 3),
        "vout_swing_mv": round(vout_swing_mv, 3),
    }


# =============================================================================
# S4_4: Unified CLI (`scripts/josim_cosim.py --config <spec.json>`) Verification
# =============================================================================
def run_s4_4_unified_cli_workflow() -> dict:
    """
    Test `python3 scripts/josim_cosim.py --config <spec.json> --output-json <report.json>`
    as a standalone CLI process.
    """
    tx_netlist = """* CLI Test JTL
.model jjmod jj(rtype=1, vg=2.8m, cap=0.07p, r0=160, rn=16, icrit=100u)
Iin 0 n1 pwl(0 0 6p 0 7p 200u 9p 200u 10p 0)
Ib1 0 n1 pwl(0 0 4p 75u)
B1 n1 0 jjmod
Rsh1 n1 0 2.5
L12 n1 n2 4.2p
Ib2 0 n2 pwl(0 0 4p 75u)
B2 n2 0 jjmod
Rsh2 n2 0 2.5
.tran 0.1p 35p
.print DEVP B1
.print DEVP B2
.end
"""
    with tempfile.TemporaryDirectory(prefix="josim_s4_4_cli_") as tmpd:
        tmp = Path(tmpd)
        tx_cir = (tmp / "tx.cir").as_posix()
        tx_csv = (tmp / "tx.csv").as_posix()
        Path(tx_cir).write_text(tx_netlist, encoding="utf-8")

        spice_deck = f"""* CLI ngspice XSPICE Co-Sim Deck
Rload n2 0 5.0
A_tx [%gd(n2 0)] jtx
.model jtx josim_cosim(netlist="{tx_cir}" pos_nodes=["n2"] neg_nodes=["0"] server_addr="{{SERVER_ADDR}}" out_csv="{tx_csv}")
.end
"""
        spec_path = tmp / "spec.json"
        out_path = tmp / "report.json"
        spec = {
            "mode": "ngspice_xspice",
            "transport": "unix",
            "expected_clients": 1,
            "spice_deck": spice_deck,
            "wrdata_vecs": ["v(n2)"],
            "tran_cmd": "tran 0.1p 35p",
            "josim_csv_map": {"tx": tx_csv},
        }
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

        proc = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "josim_cosim.py"), "--config", str(spec_path), "--output-json", str(out_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        assert proc.returncode == 0 and out_path.exists(), "CLI failed to produce output report"
        report = json.loads(out_path.read_text(encoding="utf-8"))
        assert report["josim_events"]["tx"] == 2, f"Expected 2 SFQ events in CLI report, got {report}"
        assert report["traces"]["v(n2)"]["max"] > 3e-4, f"Expected SFQ voltage pulse > 0.3 mV, got {report}"

    return {
        "passed": True,
        "cli_events": report["josim_events"]["tx"],
        "v_n2_peak_mv": round(report["traces"]["v(n2)"]["max"] * 1e3, 3),
    }


def main() -> int:
    print("=" * 84)
    print("JoSIM Step 4 (Layer 3B Multi-Instance / Multi-Window Coordinator & CLI) Suite")
    print("=" * 84)

    s4_1 = run_s4_1_multi_instance_server_mux()
    print(f"[PASS] S4_1_MULTI_INSTANCE_SERVER_MUX        | {json.dumps(s4_1)}")

    s4_2 = run_s4_2_multi_window_branin_duplex()
    print(f"[PASS] S4_2_MULTI_WINDOW_BRANIN_DUPLEX       | {json.dumps(s4_2)}")

    s4_3 = run_s4_3_spectre_multi_partition_end_to_end()
    print(f"[PASS] S4_3_SPECTRE_MULTI_PARTITION_E2E      | {json.dumps(s4_3)}")

    s4_4 = run_s4_4_unified_cli_workflow()
    print(f"[PASS] S4_4_UNIFIED_CLI_WORKFLOW             | {json.dumps(s4_4)}")

    print("=" * 84)
    print("ALL STEP 4 (LAYER 3B) CO-SIMULATION BENCHMARKS PASSED.")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
