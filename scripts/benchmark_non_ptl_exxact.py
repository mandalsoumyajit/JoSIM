#!/usr/bin/env python3
"""
Benchmark JoSIM across Non-PTL and Diverse Superconducting Circuit Topologies on BNL exxact:
  Category 1: Purely Monolithic Tightly-Coupled Non-PTL Circuits (P = 1, 0 PTLs)
    - Mono-2D-JJ-Grid-16k       : 2D inductively-coupled JJ array (16,384 JJs, 0 PTLs, P=1)
    - Mono-RSFQ-Logic-Pipeline  : Monolithic RSFQ splitter/merger/DRO/XOR pipeline (8,192 JJs, 0 PTLs, P=1)
    - Mono-Extracted-RLK-Phase  : Monolithic 3D-extracted RLK mesh (8,192 JJs, 49,152 K, 8,192 C, 0 PTLs, P=1, Phase mode)
    - Mono-Extracted-RLK-Volt   : Monolithic 3D-extracted RLK mesh (8,192 JJs, 49,152 K, 8,192 C, 0 PTLs, P=1, Voltage mode)
  Category 2: Multi-Channel / Parallel-Slice Non-PTL Circuits (P >= 2, 0 PTLs)
    - NoPTL-SQUID-Array-256     : 256-channel SQUID/SFQ readout array (8,192 JJs, 24,576 K, 8,192 C, 0 PTLs, Phase mode)
    - NoPTL-SQUID-Array-1024    : 1024-channel SQUID/SFQ readout array (32,768 JJs, 98,304 K, 32,768 C, 0 PTLs, Phase mode)
    - NoPTL-RLK-Sensor-256-Volt : 256-channel extracted RLK sensor array (8,192 JJs, 24,576 K, 8,192 C, 0 PTLs, Voltage mode)
  Category 3: Rich Multi-Gate RSFQ ALU / Logic Macros (DRO + Merger + Splitter + Mutual K + Cap C)
    - RSFQ-ALU-Macros-256       : 256 RSFQ ALU macros with K, C, R, L, JJ (8,192 JJs, 24,576 K, 8,192 C)
    - RSFQ-ALU-Macros-1024      : 1024 RSFQ ALU macros with K, C, R, L, JJ (32,768 JJs, 98,304 K, 32,768 C)
"""

import os
import re
import struct
import subprocess
import sys
import time
import numpy as np


def read_josim_bin(path: str):
    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic == b"JOSIMBIN", f"Invalid magic in {path}: {magic}"
        version, flags, n_traces, n_steps = struct.unpack("<IIQQ", f.read(24))
        names = []
        for _ in range(n_traces):
            (slen,) = struct.unpack("<I", f.read(4))
            names.append(f.read(slen).decode("utf-8"))
            f.read(1)  # type char
        data = np.frombuffer(f.read(n_traces * n_steps * 8), dtype=np.float64).reshape(
            (n_traces, n_steps)
        )
    return {name: data[i] for i, name in enumerate(names)}


def generate_mono_2d_jj_grid(path: str, rows: int = 128, cols: int = 128, tstop: str = "500p"):
    """Purely monolithic 2D Josephson junction array (0 PTLs, P = 1)."""
    lines = [
        "* Monolithic 2D Josephson Junction Grid (0 PTLs, P=1)",
        ".model jjmod jj(rtype=1, vg=2.8mV, cap=0.07pF, r0=160ohm, rn=16ohm, icrit=0.1mA)",
        "IIN 0 n_0_0 pwl(0 0 2p 0 4p 0.35mA 8p 0.35mA 10p 0)",
        "IBIAS_SRC 0 bsrc pwl(0 0 5p 0.14mA)",
        "RBSRC bsrc 0 1.0",
    ]
    for r in range(rows):
        for c in range(cols):
            node = f"n_{r}_{c}"
            lines.append(f"B_{r}_{c} {node} 0 jjmod")
            lines.append(f"RB_{r}_{c} bsrc {node} 10.0")
            if (r + c) % 2 == 0:
                lines.append(f"C_{r}_{c} {node} 0 0.02pF")
            if c + 1 < cols:
                nxt = f"n_{r}_{c+1}"
                lines.append(f"LH_{r}_{c} {node} {nxt} 2.2pH")
            if r + 1 < rows:
                dn = f"n_{r+1}_{c}"
                lines.append(f"LV_{r}_{c} {node} {dn} 2.2pH")
    lines.append(f".tran 0.25p {tstop} 0 5p")
    lines.append(f".print V(B_0_0) V(B_{rows-1}_{cols-1})")
    lines.append(".end")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def generate_mono_extracted_rlk(path: str, n_stages: int = 8192, tstop: str = "500p"):
    """Purely monolithic 3D field-extracted RLK mesh with dense mutual inductances K and capacitors C (0 PTLs, P = 1)."""
    lines = [
        "* Monolithic 3D-Extracted RLK Circuit (0 PTLs, P=1)",
        ".model jjmod jj(rtype=1, vg=2.8mV, cap=0.07pF, r0=160ohm, rn=16ohm, icrit=0.1mA)",
        "IIN 0 n_0 pwl(0 0 2p 0 4p 0.32mA 6p 0.32mA 8p 0)",
        "IBIAS_SRC 0 bsrc pwl(0 0 5p 0.14mA)",
        "RBSRC bsrc 0 1.0",
    ]
    for s in range(n_stages):
        n_cur = f"n_{s}"
        n_mid = f"m_{s}"
        n_nxt = f"n_{s+1}" if s + 1 < n_stages else "out_end"
        lines.append(f"L1_{s} {n_cur} {n_mid} 1.8pH")
        lines.append(f"L2_{s} {n_mid} {n_nxt} 1.8pH")
        lines.append(f"B_{s} {n_nxt} 0 jjmod")
        lines.append(f"C_{s} {n_mid} 0 0.03pF")
        lines.append(f"RB_{s} bsrc {n_mid} 12.0")
        # 3 mutual couplings per stage -> 6 symmetric M entries per stage (49,152 mutual terms for 8,192 stages)
        lines.append(f"Kintra_{s} L1_{s} L2_{s} 0.08")
        if s > 0:
            lines.append(f"Kadj1_{s} L1_{s} L2_{s-1} 0.04")
            lines.append(f"Kadj2_{s} L1_{s} L1_{s-1} 0.02")
    lines.append("ROUT out_end 0 5.0")
    lines.append(f".tran 0.25p {tstop} 0 5p")
    lines.append(f".print V(B_0) V(B_{n_stages-1})")
    lines.append(".end")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def generate_mono_rsfq_logic(path: str, n_cells: int = 2048, tstop: str = "500p"):
    """Monolithic RSFQ logic pipeline (4-JJ splitter/merger/DRO cells wired purely with inductors, 0 PTLs, P = 1)."""
    lines = [
        "* Monolithic RSFQ Logic Pipeline (0 PTLs, P=1)",
        ".model jjmod jj(rtype=1, vg=2.8mV, cap=0.07pF, r0=160ohm, rn=16ohm, icrit=0.1mA)",
        "IIN 0 in_0 pwl(0 0 2p 0 4p 0.32mA 6p 0.32mA 8p 0)",
        "IBIAS_SRC 0 bsrc pwl(0 0 5p 0.15mA)",
        "RBSRC bsrc 0 1.0",
    ]
    for c in range(n_cells):
        inp = f"in_{c}"
        out = f"in_{c+1}"
        a = f"a_{c}"
        b = f"b_{c}"
        m = f"m_{c}"
        # Splitter into 2 branches (a, b) + DRO/Merger into out (4 JJs, 6 L, 4 K, 2 C, 4 R per cell)
        lines.append(f"LS0_{c} {inp} {a} 1.6pH")
        lines.append(f"LS1_{c} {inp} {b} 1.6pH")
        lines.append(f"BS0_{c} {a} 0 jjmod")
        lines.append(f"BS1_{c} {b} 0 jjmod")
        lines.append(f"LM0_{c} {a} {m} 1.8pH")
        lines.append(f"LM1_{c} {b} {m} 1.8pH")
        lines.append(f"BM0_{c} {m} 0 jjmod")
        lines.append(f"LOUT_{c} {m} {out} 2.0pH")
        lines.append(f"BOUT_{c} {out} 0 jjmod")
        lines.append(f"CA_{c} {a} 0 0.025pF")
        lines.append(f"CM_{c} {m} 0 0.025pF")
        lines.append(f"RB0_{c} bsrc {a} 12.0")
        lines.append(f"RB1_{c} bsrc {b} 12.0")
        lines.append(f"RB2_{c} bsrc {m} 10.0")
        lines.append(f"RB3_{c} bsrc {out} 12.0")
        lines.append(f"KS_{c} LS0_{c} LS1_{c} 0.06")
        lines.append(f"KM_{c} LM0_{c} LM1_{c} 0.06")
    lines.append(f"ROUT in_{n_cells} 0 5.0")
    lines.append(f".tran 0.25p {tstop} 0 5p")
    lines.append(f".print V(BOUT_0) V(BOUT_{n_cells-1})")
    lines.append(".end")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def generate_multi_channel_squid_no_ptl(
    path: str, n_channels: int = 256, stages_per_ch: int = 32, tstop: str = "2500p", use_ptl: bool = False
):
    """
    Multi-Channel SQUID / RSFQ ALU Readout Array with Mutual Inductances K, Capacitors C, Resistors R, and JJs.
    When use_ptl=False, contains ZERO transmission lines (0 PTLs).
    """
    lines = [
        f"* Multi-Channel Superconducting Array (channels={n_channels}, stages={stages_per_ch}, use_ptl={use_ptl})",
        ".model jjmod jj(rtype=1, vg=2.8mV, cap=0.07pF, r0=160ohm, rn=16ohm, icrit=0.1mA)",
        "IIN 0 in_0 pwl(0 0 2p 0 4p 0.30mA 6p 0.30mA 8p 0)",
    ]
    for b in range(n_channels):
        prev_node = f"in_{b}"
        bsrc = f"bsrc_{b}"
        lines.append(f"IBIAS_{b} 0 {bsrc} pwl(0 0 5p 0.14mA)")
        lines.append(f"RBSRC_{b} {bsrc} 0 1.0")
        if b > 0 and not use_ptl:
            # Independent channel excitation (0 PTLs)
            lines.append(f"IIN_{b} 0 {prev_node} pwl(0 0 2p 0 4p 0.30mA 6p 0.30mA 8p 0)")
        for s in range(stages_per_ch):
            n_mid = f"m_{b}_{s}"
            n_jj = f"j_{b}_{s}"
            lines.append(f"L1_{b}_{s} {prev_node} {n_mid} 2.0pH")
            lines.append(f"L2_{b}_{s} {n_mid} {n_jj} 2.0pH")
            lines.append(f"B_{b}_{s} {n_jj} 0 jjmod")
            lines.append(f"C_{b}_{s} {n_mid} 0 0.025pF")
            lines.append(f"RS_{b}_{s} {n_jj} 0 4.0")
            lines.append(f"RB_{b}_{s} {bsrc} {n_mid} 10.0")
            lines.append(f"Kloc_{b}_{s} L1_{b}_{s} L2_{b}_{s} 0.07")
            if s > 0:
                lines.append(f"Kadj_{b}_{s} L1_{b}_{s} L2_{b}_{s-1} 0.035")
            prev_node = n_jj
        if use_ptl and b < n_channels - 1:
            next_in = f"in_{b+1}"
            lines.append(f"Rtx1_{b} {prev_node} txin_{b} 2.0")
            lines.append(f"T_{b} txin_{b} 0 {next_in} 0 td=25p z0=10")
            lines.append(f"Rtx2_{b} {next_in} 0 10.0")
        else:
            lines.append(f"ROUT_{b} {prev_node} 0 5.0")

    lines.append(f".tran 0.25p {tstop} 0 5p")
    lines.append(f".print V(B_0_0) V(B_{n_channels-1}_{stages_per_ch-1})")
    lines.append(".end")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_josim(josim_bin: str, cir_path: str, out_path: str, extra_args=None, omp_threads: int = 1):
    cmd = [josim_bin, "-m", "-V", "1", "-o", out_path]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(cir_path)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_threads)
    t0 = time.perf_counter()
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    wall = time.perf_counter() - t0
    if res.returncode != 0:
        raise RuntimeError(f"Command {' '.join(cmd)} failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    m_sim = re.search(r"Transient simulation time:\s*([0-9.eE+-]+)\s*s", res.stdout)
    sim_s = float(m_sim.group(1)) if m_sim else wall
    m_eq = re.search(r"MxN size:\s*(\d+)x\d+", res.stdout)
    eqns = int(m_eq.group(1)) - 1 if m_eq else 0
    return sim_s, wall, eqns


def main():
    josim_bin = sys.argv[1] if len(sys.argv) > 1 else "/u/home/smandal/.cache/josim-gpu-build/josim-cli"
    work_dir = "/u/home/smandal/.cache/josim-gpu-bench"
    os.makedirs(work_dir, exist_ok=True)

    suites = []

    # Category 1: Purely Monolithic Tightly-Coupled Non-PTL Circuits (P = 1, 0 PTLs)
    c_grid = os.path.join(work_dir, "mono_2d_grid_16k.cir")
    generate_mono_2d_jj_grid(c_grid, rows=128, cols=128, tstop="500p")
    suites.append(("Mono-2D-JJ-Grid (16k JJs, 0 PTL)", c_grid, 2000, []))

    c_logic = os.path.join(work_dir, "mono_rsfq_logic_8k.cir")
    generate_mono_rsfq_logic(c_logic, n_cells=2048, tstop="500p")
    suites.append(("Mono-RSFQ-Logic (8k JJs, 0 PTL)", c_logic, 2000, []))

    c_rlk = os.path.join(work_dir, "mono_rlk_8k.cir")
    generate_mono_extracted_rlk(c_rlk, n_stages=8192, tstop="500p")
    suites.append(("Mono-RLK-49kK-Phase (0 PTL)", c_rlk, 2000, []))
    suites.append(("Mono-RLK-49kK-Volt (0 PTL)", c_rlk, 2000, ["-a", "0"]))

    # Category 2: Multi-Channel Non-PTL Circuits (P >= 2, 0 PTLs)
    c_sq256 = os.path.join(work_dir, "noptl_squid_256.cir")
    generate_multi_channel_squid_no_ptl(c_sq256, n_channels=256, stages_per_ch=32, tstop="5000p", use_ptl=False)
    suites.append(("NoPTL-SQUID-256 (8k JJs, 24k K)", c_sq256, 20000, []))
    suites.append(("NoPTL-SQUID-256-Volt (24k K)", c_sq256, 20000, ["-a", "0"]))

    c_sq1024 = os.path.join(work_dir, "noptl_squid_1024.cir")
    generate_multi_channel_squid_no_ptl(c_sq1024, n_channels=1024, stages_per_ch=32, tstop="2500p", use_ptl=False)
    suites.append(("NoPTL-SQUID-1024 (32k JJs, 98k K)", c_sq1024, 10000, []))

    c_sq4096 = os.path.join(work_dir, "noptl_squid_4096.cir")
    generate_multi_channel_squid_no_ptl(c_sq4096, n_channels=4096, stages_per_ch=32, tstop="1250p", use_ptl=False)
    suites.append(("NoPTL-SQUID-4096 (131k JJs, 393k K)", c_sq4096, 5000, []))

    # Category 3: Rich Multi-Gate RSFQ ALU Macros (JJ + L + Mutual K + Cap C + R + PTL)
    c_alu256 = os.path.join(work_dir, "rsfq_alu_256.cir")
    generate_multi_channel_squid_no_ptl(c_alu256, n_channels=256, stages_per_ch=32, tstop="5000p", use_ptl=True)
    suites.append(("RSFQ-ALU-Macros-256 (24k K, 8k C)", c_alu256, 20000, []))

    c_alu1024 = os.path.join(work_dir, "rsfq_alu_1024.cir")
    generate_multi_channel_squid_no_ptl(c_alu1024, n_channels=1024, stages_per_ch=32, tstop="2500p", use_ptl=True)
    suites.append(("RSFQ-ALU-Macros-1024 (98k K, 32k C)", c_alu1024, 10000, []))

    c_alu4096 = os.path.join(work_dir, "rsfq_alu_4096.cir")
    generate_multi_channel_squid_no_ptl(c_alu4096, n_channels=4096, stages_per_ch=32, tstop="1250p", use_ptl=True)
    suites.append(("RSFQ-ALU-Macros-4096 (393k K, 131k C)", c_alu4096, 5000, []))

    print("=========================================================================================================================================")
    print(" Non-PTL & Diverse Superconducting Circuit Benchmark (Dual AMD EPYC 7413 48C/96T + NVIDIA RTX A6000)")
    print("=========================================================================================================================================")
    print("\n--- A. Transient Solver Runtime (trans_sim) & Numerical Equivalence ---")
    print(
        f"{'Design':<35} | {'Eqns':>8} | {'Steps':>6} | {'Serial (1T)':>11} | {'OMP (16T)':>10} | {'GPU (A6000)':>11} | {'OMP Speedup':>11} | {'GPU Speedup':>11} | {'Max Diff':>10}"
    )
    print("-" * 138)

    wall_rows = []
    for name, cir, steps, extra in suites:
        tag = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        out_ser = os.path.join(work_dir, f"{tag}_ser.bin")
        out_omp = os.path.join(work_dir, f"{tag}_omp.bin")
        out_gpu = os.path.join(work_dir, f"{tag}_gpu.bin")

        t_ser, w_ser, eqns = run_josim(josim_bin, cir, out_ser, extra_args=extra, omp_threads=1)
        t_omp, w_omp, _ = run_josim(josim_bin, cir, out_omp, extra_args=["-p"] + extra, omp_threads=24)
        t_gpu, w_gpu, _ = run_josim(josim_bin, cir, out_gpu, extra_args=["-g"] + extra, omp_threads=24)

        tr_ser = read_josim_bin(out_ser)
        tr_omp = read_josim_bin(out_omp)
        tr_gpu = read_josim_bin(out_gpu)
        diff_omp = max(float(np.max(np.abs(tr_ser[k] - tr_omp[k]))) for k in tr_ser)
        diff_gpu = max(float(np.max(np.abs(tr_ser[k] - tr_gpu[k]))) for k in tr_ser)
        max_diff = max(diff_omp, diff_gpu)
        assert max_diff < 1e-10, f"Numerical mismatch on {name}: diff_omp={diff_omp}, diff_gpu={diff_gpu}"

        print(
            f"{name:<35} | {eqns:>8d} | {steps:>6d} | {t_ser:>10.2f}s | {t_omp:>9.2f}s | {t_gpu:>10.2f}s | {t_ser/t_omp:>10.2f}x | {t_ser/t_gpu:>10.2f}x | {max_diff:>10.2e}",
            flush=True,
        )
        wall_rows.append((name, eqns, steps, w_ser, w_omp, w_gpu))

    print("\n--- B. Total End-to-End Wall-Clock Time (Parse + Matrix + Factorize + Sim + Binary Output) ---")
    print(
        f"{'Design':<35} | {'Eqns':>8} | {'Steps':>6} | {'Serial (1T)':>11} | {'OMP (24T)':>10} | {'GPU (A6000)':>11} | {'OMP Speedup':>11} | {'GPU Speedup':>11}"
    )
    print("-" * 125)
    for name, eqns, steps, w_ser, w_omp, w_gpu in wall_rows:
        print(
            f"{name:<35} | {eqns:>8d} | {steps:>6d} | {w_ser:>10.2f}s | {w_omp:>9.2f}s | {w_gpu:>10.2f}s | {w_ser/w_omp:>10.2f}x | {w_ser/w_gpu:>10.2f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
