#!/usr/bin/env python3
"""Integration test for Item #9: In-memory Python / C binding (pyjosim) with zero-copy NumPy views."""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pyjosim  # noqa: E402


NETLIST_JTL = """\
* In-memory 2-stage JTL with parameterized bias and SFQ pulse train
.param IBIAS=280u
.model jmitll jj(rtype=1, vg=2.8mV, cap=0.07pF, r0=160, rn=16, icrit=0.1mA)

IB01  0 1 pwl(0 0 5p IBIAS)
LPR01 2 1 0.278p
L01   4 3 2.0p
L02   3 2 2.425p
L03   2 6 2.425p
L04   6 5 2.031p
LP01  0 7 0.086p
LP02  0 8 0.096p
LRB01 7 9 0.086p
LRB02 8 10 0.086p
RB01  9 3 5.23
RB02  10 6 5.23
ROUT  5 0 2.0
B01   3 7 jmitll area=2.16
B02   6 8 jmitll area=2.16
VIN   4 0 sfq(20p 25p 2p 111)

.tran 0.1p 120p 0 0.1p
.print DEVV B01
.print DEVV B02
.print PHASE B01
.print PHASE B02
.jitter J_PROP B02 REF=B01
.meas TRAN t_prop TRIG P(B01) SFQ=1 TARG P(B02) SFQ=1
.meas TRAN n_b02  COUNT P(B02)
.end
"""


def main() -> None:
    lib_path = sys.argv[1] if len(sys.argv) > 1 else None

    # 1. One-shot in-memory simulation from netlist string (zero disk I/O)
    res = pyjosim.simulate_netlist(NETLIST_JTL, lib_path=lib_path)

    # Verify zero-copy NumPy views (owndata is False because memory is owned by C++ session)
    assert res.time.ndim == 1 and len(res.time) > 1000
    assert res.time.flags.owndata is False, "time array must be a zero-copy view"

    v_b01 = res["V(B01)"]
    p_b02 = res["P(B02)"]
    assert v_b01.shape == res.time.shape
    assert v_b01.flags.owndata is False, "trace array must be a zero-copy view"
    assert p_b02.flags.owndata is False, "phase trace must be a zero-copy view"

    mat = res.matrix
    assert mat.shape == (res.num_traces, res.num_steps)
    assert mat.flags.owndata is False, "2D matrix must be a zero-copy view"

    # 2. Verify zero-copy structured SfqEvent array and pulse metrics
    assert res.events.flags.owndata is False, "events array must be a zero-copy view"
    ev_b01 = res.events_for("B01")
    ev_b02 = res.events_for("B02")
    assert len(ev_b01) == 3, f"Expected 3 SFQ pulses on B01, got {len(ev_b01)}"
    assert len(ev_b02) == 3, f"Expected 3 SFQ pulses on B02, got {len(ev_b02)}"
    assert np.all(ev_b01["polarity"] == 1)
    assert np.all(ev_b02["v_peak"] > 0.5e-3)
    assert np.all(np.abs(ev_b02["area"] - 2.0678e-15) < 0.5e-15)

    # 3. Verify .meas results and exported .jitter parameters
    assert "T_PROP" in res.measurements
    assert 1.5e-12 < res.measurements["T_PROP"] < 2.2e-12
    assert res.measurements["N_B02"] == 3.0
    assert "LATENCY_J_PROP" in res.parameters
    assert abs(res.parameters["LATENCY_J_PROP"] - res.measurements["T_PROP"]) < 0.2e-12

    # 4. Persistent Session with in-memory parameter overrides (no netlist re-parsing)
    sess = pyjosim.Session(netlist=NETLIST_JTL, lib_path=lib_path)
    delays = []
    for ibias in (240e-6, 280e-6, 320e-6):
        r = sess.run(params={"IBIAS": ibias})
        assert r.measurements["N_B02"] == 3.0
        delays.append(r.measurements["T_PROP"])

    # Higher bias current speeds up JTL propagation (monotonic delay reduction)
    assert delays[0] > delays[1] > delays[2], f"Expected monotonic delay vs bias, got {delays}"

    print(
        f"pyjosim PASS: {res.num_traces} traces x {res.num_steps} steps, "
        f"{len(res.events)} SFQ events, delays(ps)={[round(d*1e12, 3) for d in delays]}"
    )


if __name__ == "__main__":
    main()
